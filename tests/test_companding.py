"""Unit tests for companding / warp quantization (binning == "companding").

Idea 1 from the non-uniform quantization notes: evolve a monotone warp F and
quantize as round_uniform(F(x)), reconstructed with the same bin-mean
centroid every other binning mode already uses. See evolmc/quantize.py
(_companding_forward, _companding_assign) and evolmc/grouping.py (Genome's
warp gene block).

    python -m pytest tests/test_companding.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolmc.config import Config  # noqa: E402
from evolmc.grouping import Genome  # noqa: E402
from evolmc.models import TargetLayer  # noqa: E402
from evolmc.quantize import compress_layer  # noqa: E402


def _cfgs(**over):
    cfg = Config()
    cfg.quant.binning = "companding"
    for k, v in over.get("quant", {}).items():
        setattr(cfg.quant, k, v)
    for k, v in over.get("prune", {}).items():
        setattr(cfg.prune, k, v)
    return cfg


def _compress(w, k=64, t_lo=0.0, t_hi=0.0, alpha=4.0, gamma=1 / 3, u=None,
             force_zero=False, reassign=False, cfg=None):
    cfg = cfg or _cfgs(prune={"enabled": t_lo < 0.0 or t_hi > 0.0})
    u = np.zeros(cfg.quant.companding_residual_genes) if u is None else u
    return compress_layer(w, w.std(1, keepdim=True), k=k, t_lo=t_lo, t_hi=t_hi,
                          quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t",
                          alpha=alpha, gamma=gamma, u=u,
                          force_zero=force_zero, reassign=reassign)


# -- quantize.compress_layer, binning == "companding" -----------------------

def test_companding_uses_at_most_k_distinct_values_per_row():
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    recon, st = _compress(w, k=16)
    assert recon.shape == w.shape
    for row in recon:
        assert torch.unique(row).numel() <= 16
    assert st.symbol_counts.sum() == w.numel()
    assert st.sparsity == 0.0


def test_companding_error_decreases_with_k():
    torch.manual_seed(0)
    w = torch.randn(16, 1024)
    errs = [_compress(w, k=k)[1].mse for k in (4, 16, 64, 256)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs


def test_companding_requires_warp_params():
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    cfg = _cfgs()
    with pytest.raises(ValueError, match="companding"):
        compress_layer(w, w.std(1, keepdim=True), k=16, t_lo=0.0, t_hi=0.0,
                       quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")


def test_panter_dite_gamma_beats_uniform_backbone_on_gaussian_weights():
    """The theoretical payoff of the backbone: for Gaussian weights, gamma=1/3
    (Panter-Dite) minimizes MSE, so it must beat both the gamma=0 (uniform)
    backbone at the same alpha and plain uniform binning outright."""
    torch.manual_seed(0)
    w = torch.randn(16, 4096)

    mse_g0 = _compress(w, k=64, gamma=0.0)[1].mse
    mse_pd = _compress(w, k=64, gamma=1 / 3)[1].mse
    assert mse_pd < mse_g0

    uniform_cfg = Config()
    uniform_cfg.prune.enabled = False
    _, st_uniform = compress_layer(w, w.std(1, keepdim=True), k=64, t_lo=0.0,
                                   t_hi=0.0, quant_cfg=uniform_cfg.quant,
                                   prune_cfg=uniform_cfg.prune, name="t")
    assert mse_pd < st_uniform.mse


def test_reassign_never_increases_mse():
    """Reassign runs Lloyd from the warp-derived centroids, which only ever
    decreases MSE (same guarantee the existing kmeans binning relies on)."""
    torch.manual_seed(0)
    w = torch.randn(16, 4096)
    mse_plain = _compress(w, k=64, reassign=False)[1].mse
    mse_reassigned = _compress(w, k=64, reassign=True)[1].mse
    assert mse_reassigned <= mse_plain + 1e-9


def test_force_zero_snaps_a_level_to_exactly_zero():
    torch.manual_seed(0)
    w = torch.randn(16, 4096)
    _, st = _compress(w, k=64, force_zero=False)
    recon_off, _ = _compress(w, k=64, force_zero=False)
    recon_on, st_on = _compress(w, k=64, force_zero=True)
    assert (recon_on == 0).any()
    # Symbol accounting is unaffected -- force_zero only moves a codeword's
    # value, not which weights are assigned to it.
    assert st_on.symbol_counts.sum() == w.numel()


def test_zero_residual_genes_leave_the_piecewise_warp_at_identity():
    """u == 0 -> softmax is uniform -> equal segment weights -> F_residual is
    the identity map, so the warp reduces to the bare gamma backbone."""
    torch.manual_seed(0)
    w = torch.randn(16, 2048)
    m = 6
    mse_zero_u = _compress(w, k=32, gamma=1 / 3, u=np.zeros(m))[1].mse
    mse_random_u = _compress(w, k=32, gamma=1 / 3,
                             u=np.array([2.0, -1.0, 0.5, 0.0, 1.5, -0.5]))[1].mse
    assert mse_zero_u != pytest.approx(mse_random_u)


def test_companding_respects_the_pruning_band():
    torch.manual_seed(0)
    w = torch.randn(8, 1024)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs(prune={"enabled": True})
    recon, st = compress_layer(w, scale, k=16, t_lo=-0.5, t_hi=0.5,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune,
                               name="t", alpha=4.0, gamma=1 / 3,
                               u=np.zeros(cfg.quant.companding_residual_genes),
                               force_zero=False, reassign=False)
    inside = (w > -0.5 * scale) & (w < 0.5 * scale)
    assert torch.all(recon[inside] == 0)
    assert st.sparsity == pytest.approx(inside.float().mean().item(), abs=1e-6)
    assert st.k_centroids == 16


def test_companding_handles_k_equals_two_with_pruning():
    """kc collapses to 1 when K=2 and pruning reserves the zero codeword --
    an edge case the uniform/kmeans branches also have to survive."""
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs(prune={"enabled": True})
    recon, st = compress_layer(w, scale, k=2, t_lo=-0.5, t_hi=0.5,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune,
                               name="t", alpha=4.0, gamma=1 / 3,
                               u=np.zeros(cfg.quant.companding_residual_genes),
                               force_zero=False, reassign=True)
    assert recon.shape == w.shape
    assert st.symbol_counts.sum() == w.numel()


# -- Genome: warp gene block -------------------------------------------------

def _layers(n_blocks=2, types=("q_proj", "o_proj")):
    return [
        TargetLayer(f"m.layers.{b}.{t}", torch.nn.Linear(4, 4), 4096 * 4096,
                   4096, 4096, b, t, False)
        for b in range(n_blocks) for t in types
    ]


def test_companding_adds_a_warp_gene_block_sized_to_the_k_groups():
    cfg = _cfgs(prune={"enabled": False})
    layers = _layers()
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert g.n_k == 2  # k_grouping="type" default
    assert g.companding is True
    assert g.n_w == g.n_k
    assert g.warp_dim == 2 + cfg.quant.companding_residual_genes + 2
    assert g.n_var == g.n_k + g.n_w * g.warp_dim


def test_non_companding_genome_is_unaffected():
    """binning != "companding" must add zero warp variables and leave
    LayerSetting's warp fields at their None/False defaults."""
    cfg = Config()
    assert cfg.quant.binning == "uniform"
    layers = _layers()
    cfg.prune.enabled = False
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert g.companding is False
    assert g.n_w == 0
    assert g.n_var == g.n_k  # unchanged from before this feature existed

    s = g.decode(g.encode_uniform(16))[layers[0].name]
    assert s.alpha is None and s.gamma is None and s.u is None
    assert s.force_zero is False and s.reassign is False


def test_zero_genome_decodes_to_the_identity_warp_baseline():
    """A genome of all zeros -- what encode_uniform/seed_population leave the
    warp segment at -- must decode to alpha_min, gamma=0 (uniform backbone)
    and an identity residual, i.e. a sane, inspectable default."""
    cfg = _cfgs(prune={"enabled": False})
    layers = _layers()
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    s = g.decode(np.zeros(g.n_var))[layers[0].name]
    assert s.alpha == pytest.approx(cfg.quant.companding_alpha_min)
    assert s.gamma == pytest.approx(cfg.quant.companding_gamma_min)
    assert np.allclose(s.u, 0.0)
    assert s.force_zero is False and s.reassign is False


def test_encode_uniform_does_not_leak_pruning_fraction_into_warp_genes():
    """Regression: encode_uniform used to write the pruning fraction across
    x[n_k:], which -- before the warp block existed -- happened to end
    exactly at n_var. Appending warp genes after t_hi means that slice must
    be bounded, or a pruned warm-start would silently scramble the warp."""
    cfg = _cfgs(prune={"enabled": True, "t_max": 2.0})
    layers = _layers()
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    x = g.encode_uniform(16, t=1.5)
    warp_segment = x[g.n_k + 2 * g.n_p:]
    assert np.allclose(warp_segment, 0.0), warp_segment
    s = g.decode(x)[layers[0].name]
    assert s.alpha == pytest.approx(cfg.quant.companding_alpha_min)
    assert s.t_hi == pytest.approx(1.5)


def test_seed_population_logspace_does_not_leak_into_warp_genes():
    cfg = _cfgs(prune={"enabled": True, "t_max": 2.0})
    layers = _layers()
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    pop = g.seed_population(10, np.random.default_rng(0), "logspace")
    warp_segment = pop[:, g.n_k + 2 * g.n_p:]
    assert np.allclose(warp_segment, 0.0), warp_segment


def test_warp_genes_are_shared_within_a_k_group_and_vary_across_groups():
    cfg = _cfgs(prune={"enabled": False})
    layers = _layers()
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    rng = np.random.default_rng(0)
    x = rng.random(g.n_var)
    settings = g.decode(x)
    by_type = {t: [s for name, s in settings.items() if name.endswith(t)]
              for t in ("q_proj", "o_proj")}
    for group in by_type.values():
        alphas = {s.alpha for s in group}
        assert len(alphas) == 1, "warp params must be shared across a K group"
    q_alpha = by_type["q_proj"][0].alpha
    o_alpha = by_type["o_proj"][0].alpha
    assert q_alpha != o_alpha, "different K groups should get independent warps"


def test_flag_genes_threshold_at_one_half():
    cfg = _cfgs(prune={"enabled": False})
    layers = [TargetLayer("m.layers.0.q_proj", torch.nn.Linear(4, 4),
                          4096 * 4096, 4096, 4096, 0, "q_proj", False)]
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    x_lo = np.zeros(g.n_var)
    x_hi = np.zeros(g.n_var)
    x_hi[-2:] = 1.0  # force_zero, reassign
    lo = g.decode(x_lo)[layers[0].name]
    hi = g.decode(x_hi)[layers[0].name]
    assert lo.force_zero is False and lo.reassign is False
    assert hi.force_zero is True and hi.reassign is True
