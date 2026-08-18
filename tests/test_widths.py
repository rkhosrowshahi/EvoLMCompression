"""Unit tests for bin-width quantization (binning == "widths").

The plain, backbone-free non-uniform quantizer: kc bin widths are genome
genes directly, in the original weight domain (l_i = (hi-lo) * exp(z_i) /
sum(exp(z)), accumulated into left edges). No density estimate, no warp --
contrast with binning == "companding" in test_companding.py. See
evolmc/quantize.py (_widths_edges) and evolmc/grouping.py (Genome's width
gene block).

    python -m pytest tests/test_widths.py -q
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
from evolmc.quantize import _uniform_edges, _widths_edges, compress_layer  # noqa: E402


def _cfgs(**over):
    cfg = Config()
    cfg.quant.binning = "widths"
    for k, v in over.get("quant", {}).items():
        setattr(cfg.quant, k, v)
    for k, v in over.get("prune", {}).items():
        setattr(cfg.prune, k, v)
    return cfg


def _compress(w, k=64, t_lo=0.0, t_hi=0.0, z=None, cfg=None):
    cfg = cfg or _cfgs(prune={"enabled": t_lo < 0.0 or t_hi > 0.0})
    z = np.zeros(k) if z is None else z
    return compress_layer(w, w.std(1, keepdim=True), k=k, t_lo=t_lo, t_hi=t_hi,
                          quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t", z=z)


# -- quantize._widths_edges ---------------------------------------------------

def test_widths_edges_are_monotone_and_span_the_alive_range():
    torch.manual_seed(0)
    w = torch.randn(4, 2048)
    alive = torch.ones_like(w, dtype=torch.bool)
    rng = np.random.default_rng(0)
    z = torch.tensor(rng.normal(scale=3.0, size=16), dtype=torch.float32)
    edges = _widths_edges(w, alive, kc=16, z=z)
    assert edges.shape == (4, 16)
    assert ((edges[:, 1:] - edges[:, :-1]) > 0).all()
    lo = w.min(dim=1, keepdim=True).values
    hi = w.max(dim=1, keepdim=True).values
    assert torch.allclose(edges[:, :1], lo, atol=1e-4)
    assert (edges[:, -1:] < hi).all()


def test_widths_constant_z_reproduces_uniform_edges():
    """exp(c)/sum(exp(c)) is 1/kc for ANY constant c, not just c=0 -- a
    constant log-width vector must decode to equal-width bins regardless of
    where that constant sits in [widths_log_lo, widths_log_hi]."""
    torch.manual_seed(0)
    w = torch.randn(4, 2048)
    alive = torch.ones_like(w, dtype=torch.bool)
    for c in (-14.0, -3.5, 0.0, 2.0):
        z = torch.full((16,), c)
        got = _widths_edges(w, alive, kc=16, z=z)
        ref = _uniform_edges(w, alive, kc=16)
        assert torch.allclose(got, ref, atol=1e-4), c


# -- quantize.compress_layer, binning == "widths" ----------------------------

def test_widths_uses_at_most_k_distinct_values_per_row():
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    recon, st = _compress(w, k=16)
    assert recon.shape == w.shape
    for row in recon:
        assert torch.unique(row).numel() <= 16
    assert st.symbol_counts.sum() == w.numel()
    assert st.sparsity == 0.0


def test_widths_error_decreases_with_k():
    """At the neutral equal-width baseline (z=0), more bins must strictly
    help -- same sanity check every other binning mode runs. This does NOT
    hold for arbitrary random z at each K (a bad random partition at large K
    can easily lose to a good one at small K, which is exactly why the widths
    are a search target rather than left random -- see DNN compression's own
    finding that random search badly underperforms the evolved one)."""
    torch.manual_seed(0)
    w = torch.randn(16, 1024)
    errs = [_compress(w, k=k, z=np.zeros(k))[1].mse for k in (4, 16, 64, 256)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs


def test_widths_requires_z():
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    cfg = _cfgs()
    with pytest.raises(ValueError, match="widths"):
        compress_layer(w, w.std(1, keepdim=True), k=16, t_lo=0.0, t_hi=0.0,
                       quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")


def test_widths_zero_genes_match_uniform_binning():
    """z == 0 -> equal widths -> the same partition binning="uniform" builds
    by closed form, so reconstruction (hence mse) should match closely."""
    torch.manual_seed(0)
    w = torch.randn(16, 2048)
    _, st_widths = _compress(w, k=32, z=np.zeros(32))

    uni_cfg = Config()
    uni_cfg.prune.enabled = False
    _, st_uniform = compress_layer(w, w.std(1, keepdim=True), k=32, t_lo=0.0,
                                   t_hi=0.0, quant_cfg=uni_cfg.quant,
                                   prune_cfg=uni_cfg.prune, name="t")
    assert st_widths.mse == pytest.approx(st_uniform.mse, rel=1e-3)


def test_widths_nonuniform_genes_change_the_reconstruction():
    torch.manual_seed(0)
    w = torch.randn(16, 2048)
    mse_uniform = _compress(w, k=32, z=np.zeros(32))[1].mse
    z_skewed = np.concatenate([np.full(30, -8.0), [4.0, 4.0]])  # 2 wide bins, 30 collapsed
    mse_skewed = _compress(w, k=32, z=z_skewed)[1].mse
    assert mse_uniform != pytest.approx(mse_skewed)


def test_widths_respects_the_pruning_band():
    torch.manual_seed(0)
    w = torch.randn(8, 1024)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs(prune={"enabled": True})
    recon, st = compress_layer(w, scale, k=16, t_lo=-0.5, t_hi=0.5,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune,
                               name="t", z=np.zeros(16))
    inside = (w > -0.5 * scale) & (w < 0.5 * scale)
    assert torch.all(recon[inside] == 0)
    assert st.sparsity == pytest.approx(inside.float().mean().item(), abs=1e-6)
    assert st.k_centroids == 16


def test_widths_handles_k_equals_two_with_pruning():
    """kc collapses to 1 when K=2 and pruning reserves the zero codeword --
    the single leftover width gene should normalize to exactly 1 bin."""
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs(prune={"enabled": True})
    recon, st = compress_layer(w, scale, k=2, t_lo=-0.5, t_hi=0.5,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune,
                               name="t", z=np.zeros(2))
    assert recon.shape == w.shape
    assert st.symbol_counts.sum() == w.numel()


# -- Genome: width gene block -------------------------------------------------

def _layers(n_blocks=2, types=("q_proj", "o_proj")):
    return [
        TargetLayer(f"m.layers.{b}.{t}", torch.nn.Linear(4, 4), 4096 * 4096,
                   4096, 4096, b, t, False)
        for b in range(n_blocks) for t in types
    ]


def test_widths_adds_a_width_gene_block_sized_to_k_max_group():
    cfg = _cfgs(prune={"enabled": False})
    layers = _layers()
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert g.n_k == 2  # k_grouping="type" default
    assert g.widths is True
    assert g.n_ws == g.n_k
    assert g.width_dim == int(g.k_max_group.max())
    assert g.n_var == g.n_k + g.n_ws * g.width_dim


def test_non_widths_genome_is_unaffected():
    cfg = Config()
    assert cfg.quant.binning == "uniform"
    layers = _layers()
    cfg.prune.enabled = False
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert g.widths is False
    assert g.n_ws == 0
    assert g.n_var == g.n_k

    s = g.decode(g.encode_uniform(16))[layers[0].name]
    assert s.widths_z is None


def test_zero_genome_decodes_to_the_width_log_lo_baseline():
    """A genome of all zeros -- what encode_uniform/seed_population leave the
    width segment at -- decodes every gene to width_log_lo, a constant
    vector, which (see test_widths_constant_z_reproduces_uniform_edges)
    still means equal widths."""
    cfg = _cfgs(prune={"enabled": False})
    layers = _layers()
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    s = g.decode(np.zeros(g.n_var))[layers[0].name]
    assert np.allclose(s.widths_z, cfg.quant.widths_log_lo)


def test_width_genes_are_shared_within_a_k_group_and_vary_across_groups():
    cfg = _cfgs(prune={"enabled": False})
    layers = _layers()
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    rng = np.random.default_rng(0)
    x = rng.random(g.n_var)
    settings = g.decode(x)
    by_type = {t: [s for name, s in settings.items() if name.endswith(t)]
              for t in ("q_proj", "o_proj")}
    for group in by_type.values():
        firsts = {tuple(s.widths_z.tolist()) for s in group}
        assert len(firsts) == 1, "width genes must be shared across a K group"
    q_z = by_type["q_proj"][0].widths_z
    o_z = by_type["o_proj"][0].widths_z
    assert not np.allclose(q_z, o_z), "different K groups should get independent widths"


def test_only_the_first_kc_width_genes_are_read():
    """A group whose decoded K is small must still ignore the unused tail of
    its width_dim-length gene block -- changing genes past kc must not
    change the reconstruction."""
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    rng = np.random.default_rng(0)
    kc = 8
    z_short = rng.normal(scale=2.0, size=kc)
    z_long_same_prefix = np.concatenate([z_short, rng.normal(scale=5.0, size=40)])
    mse_short = _compress(w, k=kc, z=z_short)[1].mse
    mse_long = _compress(w, k=kc, z=z_long_same_prefix)[1].mse
    assert mse_short == pytest.approx(mse_long)
