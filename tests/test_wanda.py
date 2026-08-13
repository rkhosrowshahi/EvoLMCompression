"""Unit tests for Wanda-style activation-weighted pruning (prune.mode == "wanda").

    python -m pytest tests/test_wanda.py -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolmc.config import Config  # noqa: E402
from evolmc.quantize import _wanda_alive, compress_layer  # noqa: E402


# -- _wanda_alive -------------------------------------------------------------

def test_wanda_reorders_against_plain_magnitude():
    """The worked example: a small weight on a heavily-activated input outranks
    a large weight on a barely-activated one -- the opposite of what magnitude
    alone would prune."""
    rows = torch.tensor([[-0.09, -0.05, -0.01, 0.01, 0.05, 0.09]])
    act_norm = torch.tensor([0.2, 0.3, 9.0, 9.0, 0.3, 0.2])

    # frac = (|t_lo| + t_hi) / (2*t_max) = 1/3 -> prune 2 of 6.
    alive = _wanda_alive(rows, act_norm, t_lo=-2 / 3, t_hi=2 / 3, t_max=2.0)
    pruned = rows[~alive]
    assert sorted(pruned.tolist()) == pytest.approx(sorted([-0.05, 0.05]))

    # Plain magnitude would have pruned the two SMALLEST weights instead
    # (-0.01, 0.01) -- exactly the ones Wanda keeps, because they sit on the
    # two highest-activation features.
    threshold = rows.abs().flatten().sort().values[1]
    magnitude_pruned = rows[rows.abs() <= threshold]
    assert sorted(magnitude_pruned.tolist()) == pytest.approx(sorted([-0.01, 0.01]))
    assert sorted(pruned.tolist()) != pytest.approx(sorted(magnitude_pruned.tolist()))


def test_wanda_frac_zero_and_one_are_the_endpoints():
    rows = torch.randn(4, 10)
    act_norm = torch.rand(10) + 0.1

    all_alive = _wanda_alive(rows, act_norm, t_lo=0.0, t_hi=0.0, t_max=2.0)
    assert all_alive.all()

    all_pruned = _wanda_alive(rows, act_norm, t_lo=-2.0, t_hi=2.0, t_max=2.0)
    assert not all_pruned.any()


def test_wanda_prunes_the_same_fraction_regardless_of_t_max():
    """frac normalizes t_max away -- doubling it and doubling t_lo/t_hi to
    match must give the identical mask."""
    torch.manual_seed(0)
    rows = torch.randn(6, 20)
    act_norm = torch.rand(20) + 0.1

    a = _wanda_alive(rows, act_norm, t_lo=-1.0, t_hi=1.0, t_max=2.0)
    b = _wanda_alive(rows, act_norm, t_lo=-2.0, t_hi=2.0, t_max=4.0)
    assert torch.equal(a, b)


def test_wanda_cut_is_per_row():
    """Each row is ranked against its OWN score distribution, not pooled
    across rows -- two rows with very different scales still each lose the
    same fraction."""
    rows = torch.tensor([
        [-0.09, -0.05, -0.01, 0.01, 0.05, 0.09],
        [-0.90, -0.50, -0.10, 0.10, 0.50, 0.90],
    ])
    act_norm = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # magnitude only

    alive = _wanda_alive(rows, act_norm, t_lo=-1 / 3, t_hi=1 / 3, t_max=1.0)
    assert alive.sum(dim=1).tolist() == [4, 4]  # 1/3 of 6 pruned, per row


# -- compress_layer, prune.mode == "wanda" ------------------------------------

def _cfgs(**over):
    cfg = Config()
    cfg.prune.mode = "wanda"
    for k, v in over.get("quant", {}).items():
        setattr(cfg.quant, k, v)
    for k, v in over.get("prune", {}).items():
        setattr(cfg.prune, k, v)
    return cfg


def test_compress_layer_requires_act_norm_for_wanda():
    torch.manual_seed(0)
    w = torch.randn(4, 8)
    cfg = _cfgs()
    with pytest.raises(ValueError, match="wanda"):
        compress_layer(w, w.std(1, keepdim=True), k=4, t_lo=-0.5, t_hi=0.5,
                       quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")


def test_compress_layer_wanda_end_to_end_matches_the_helper():
    """Asymmetric weights (so survivors don't coincidentally sum/bin to
    exactly 0) plus a large K (kc=63, bins far narrower than the gaps between
    survivors) so each survivor keeps its own distinct centroid -- "pruned"
    (forced to the reserved zero codeword) is then unambiguous from "value
    happens to be near zero"."""
    torch.manual_seed(0)
    w = torch.tensor([[-0.09, -0.05, -0.012, 0.008, 0.05, 0.09]])
    scale = w.std(1, keepdim=True)
    act_norm = torch.tensor([0.2, 0.3, 9.0, 9.0, 0.3, 0.2])
    cfg = _cfgs()

    recon, st = compress_layer(w, scale, k=64, t_lo=-2 / 3, t_hi=2 / 3,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune,
                               name="t", act_norm=act_norm)
    pruned_mask = (recon == 0).tolist()
    assert pruned_mask == [[False, True, False, False, True, False]]
    assert st.sparsity == pytest.approx(2 / 6)
    # Survivors reconstruct close to their original value (fine bins), not to 0.
    assert recon[0, 0].item() == pytest.approx(-0.09, abs=0.01)
    assert recon[0, 5].item() == pytest.approx(0.09, abs=0.01)


def test_compress_layer_wanda_error_names_the_fix():
    torch.manual_seed(0)
    w = torch.randn(4, 8)
    cfg = _cfgs()
    with pytest.raises(ValueError, match="calibrate_wanda"):
        compress_layer(w, w.std(1, keepdim=True), k=4, t_lo=-0.5, t_hi=0.5,
                       quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")


# -- config --------------------------------------------------------------------

def test_wanda_is_a_valid_prune_mode():
    cfg = Config.from_dict({"prune": {"mode": "wanda"}})
    assert cfg.prune.mode == "wanda"


def test_unknown_prune_mode_is_rejected_at_compress_time():
    torch.manual_seed(0)
    w = torch.randn(4, 8)
    cfg = Config()
    cfg.prune.mode = "not-a-real-mode"
    with pytest.raises(ValueError, match="unknown prune mode"):
        compress_layer(w, w.std(1, keepdim=True), k=4, t_lo=-0.5, t_hi=0.5,
                       quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")


# -- evolmc.wanda.calibrate ----------------------------------------------------

def test_calibrate_computes_correct_l2_norms():
    """Sum-of-squares across every calibration row, sqrt once at the end --
    checked against a hand-computed expectation, not just internal
    consistency."""
    from evolmc.models import TargetLayer
    from evolmc.wanda import calibrate

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 3, bias=False)

        def forward(self, x, use_cache=False):
            return self.fc(x.float())

    model = TinyModel()
    target = TargetLayer("fc", model.fc, 12, 3, 4, 0, "fc", False)

    class FakeCompressor:
        def __init__(self):
            self.model = model
            self.device = torch.device("cpu")
            self.targets = [target]

        def restore(self):
            pass

    windows = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    result = calibrate(FakeCompressor(), windows, batch_size=1)

    expected = (windows ** 2).sum(dim=0).sqrt()
    assert torch.allclose(result.get("fc"), expected, atol=1e-4)
    assert result.n_tokens == 8  # 2 windows x 4 "features" each


def test_calibrate_restores_before_measuring():
    """A calibration pass must measure the TRUE fp16 model, not whatever a
    previous .apply() left installed."""
    from evolmc.models import TargetLayer
    from evolmc.wanda import calibrate

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x, use_cache=False):
            return self.fc(x.float())

    model = TinyModel()
    target = TargetLayer("fc", model.fc, 4, 2, 2, 0, "fc", False)

    calls = []

    class FakeCompressor:
        def __init__(self):
            self.model = model
            self.device = torch.device("cpu")
            self.targets = [target]

        def restore(self):
            calls.append("restore")

    calibrate(FakeCompressor(), torch.tensor([[1.0, 2.0]]))
    assert calls == ["restore"]
