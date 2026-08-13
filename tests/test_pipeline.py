"""Unit tests for the parts a reviewer would check by hand.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolmc.codec import huffman_bits, price_layer, shannon_bits  # noqa: E402
from evolmc.config import Config  # noqa: E402
from evolmc.quantize import compress_layer  # noqa: E402


# -- entropy coding --------------------------------------------------------

def test_huffman_uniform_matches_fixed_width():
    counts = [100] * 8  # 800 symbols, 8 equiprobable -> exactly 3 bits each
    bits, maxlen = huffman_bits(counts)
    assert bits == pytest.approx(800 * 3)
    assert maxlen == 3
    # With a flat histogram there is nothing for the entropy coder to win.
    assert bits == pytest.approx(shannon_bits(counts))


def test_huffman_never_below_entropy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        counts = rng.integers(0, 1000, size=rng.integers(2, 64)).tolist()
        if sum(counts) == 0:
            continue
        hb, _ = huffman_bits(counts)
        sb = shannon_bits(counts)
        assert hb >= sb - 1e-6
        # Huffman's classic bound: never more than 1 bit per symbol above H.
        assert hb <= sb + sum(counts) + 1e-6


def test_huffman_rewards_skew():
    flat, _ = huffman_bits([250] * 4)
    skewed, _ = huffman_bits([997, 1, 1, 1])
    assert skewed < flat


# -- quantization ----------------------------------------------------------

def _cfgs(**over):
    cfg = Config()
    for k, v in over.get("quant", {}).items():
        setattr(cfg.quant, k, v)
    for k, v in over.get("prune", {}).items():
        setattr(cfg.prune, k, v)
    return cfg


def test_reconstruction_uses_only_k_distinct_values_per_row():
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    cfg = _cfgs(quant={"granularity": "per_channel"}, prune={"enabled": False})
    recon, stats = compress_layer(w, w.std(1, keepdim=True), k=16, t_lo=0.0,
                                  t_hi=0.0, quant_cfg=cfg.quant,
                                  prune_cfg=cfg.prune, name="t")
    assert recon.shape == w.shape
    for row in recon:
        assert torch.unique(row).numel() <= 16
    assert stats.symbol_counts.sum() == w.numel()
    assert stats.sparsity == 0.0


def test_error_decreases_with_k():
    torch.manual_seed(0)
    w = torch.randn(16, 1024)
    cfg = _cfgs(prune={"enabled": False})
    errs = []
    for k in (4, 16, 64, 256):
        _, st = compress_layer(w, w.std(1, keepdim=True), k=k, t_lo=0.0, t_hi=0.0,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
        errs.append(st.mse)
    assert all(a > b for a, b in zip(errs, errs[1:])), errs


def _binning_sweep(w, k):
    out = {}
    for binning in ("uniform", "quantile", "kmeans"):
        cfg = _cfgs(quant={"binning": binning}, prune={"enabled": False})
        _, st = compress_layer(w, w.std(1, keepdim=True), k=k, t_lo=0.0, t_hi=0.0,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
        out[binning] = st
    return out


@pytest.mark.parametrize("k", [16, 64, 256])
def test_binning_ordering(k):
    """The core ablation for the paper.

    k-means is seeded from the uniform edges and Lloyd only decreases MSE, so
    it must dominate uniform at every K -- a regression here means the init or
    the empty-bin handling broke. Quantile binning is consistently *worst*:
    equal-population bins crowd centroids into the dense center of the weight
    distribution, where the residual is already small.
    """
    torch.manual_seed(0)
    w = torch.randn(16, 4096)
    out = _binning_sweep(w, k)
    assert out["kmeans"].mse <= out["uniform"].mse
    assert out["uniform"].mse < out["quantile"].mse


def test_uniform_binning_wastes_codewords_at_large_k():
    """Uniform edges over a Gaussian leave tail bins empty.

    Those unused codewords are still paid for at the fixed index width, which
    is precisely the slack the entropy coder recovers -- and the reason to
    report deployable and archival CR separately.
    """
    torch.manual_seed(0)
    w = torch.randn(16, 4096)
    out = _binning_sweep(w, 256)
    assert out["uniform"].k_used_mean < 240
    assert out["quantile"].k_used_mean == pytest.approx(256.0)


def test_pruning_zeroes_the_band_and_reserves_a_codeword():
    torch.manual_seed(0)
    w = torch.randn(8, 1024)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()
    recon, st = compress_layer(w, scale, k=16, t_lo=-0.5, t_hi=0.5,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    inside = (w > -0.5 * scale) & (w < 0.5 * scale)
    assert torch.all(recon[inside] == 0)
    assert st.sparsity == pytest.approx(inside.float().mean().item(), abs=1e-6)
    # K-1 centroids for survivors plus the reserved zero.
    assert st.k_centroids == 16
    assert st.symbol_counts[0] == inside.sum().item()


def test_dense_format_gives_pruning_no_credit_at_all():
    """The defect the sparse formats exist to fix, pinned so it cannot creep back.

    Under `dense` every weight POSITION carries a full index, so a pruned
    weight costs exactly what a live one does. Measured on the finished pruned
    runs, candidates at the same avg_bits spanned 0.00 to 0.95 sparsity with
    cr_deploy identical to 6 decimals. That is correct for a dense LUT
    kernel and wrong as a statement about parameter count.
    """
    torch.manual_seed(0)
    w = torch.randn(32, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()

    sizes, sparsities = [], []
    for t in (0.0, 0.5, 1.0, 1.5):
        _, st = compress_layer(w, scale, k=64, t_lo=-t, t_hi=t,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
        sizes.append(price_layer(st, fmt="dense").total_deployable)
        sparsities.append(st.sparsity)

    assert sparsities[0] == 0.0 and sparsities[-1] > 0.85
    assert len({round(s, 6) for s in sizes}) == 1, sizes


@pytest.mark.parametrize("fmt", ["bitmap", "csr"])
def test_sparse_formats_turn_pruning_into_real_size_reduction(fmt):
    """What the corrected accounting has to deliver: bits fall with sparsity."""
    torch.manual_seed(0)
    w = torch.randn(32, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()

    sizes, alive = [], []
    for t in (0.0, 0.5, 1.0, 1.5):
        _, st = compress_layer(w, scale, k=64, t_lo=-t, t_hi=t,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
        c = price_layer(st, fmt=fmt)
        sizes.append(c.total_deployable)
        alive.append(c.n_alive)

    # Survivors fall, and so does the bill -- strictly, at every step.
    assert all(a > b for a, b in zip(alive, alive[1:])), alive
    assert all(a > b for a, b in zip(sizes, sizes[1:])), sizes
    # n_alive is the real count, not a rounding of the ratio.
    assert alive[0] == w.numel()
    assert 0 < alive[-1] < 0.15 * w.numel()


def test_bitmap_cost_matches_the_closed_form():
    """n mask bits + ceil(log2(K-1)) per survivor + (K-1) codebook entries."""
    torch.manual_seed(0)
    w = torch.randn(16, 2048)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()
    _, st = compress_layer(w, scale, k=16, t_lo=-1.0, t_hi=1.0,
                           quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    c = price_layer(st, codebook_bits=16, fmt="bitmap")

    n_alive = int(round(w.numel() * (1 - st.sparsity)))
    expect = w.numel() + 4 * n_alive + st.n_groups * 15 * 16
    assert c.mask_bits == pytest.approx(w.numel())
    assert c.total_deployable == pytest.approx(expect)
    # The reserved zero codeword is redundant once a mask records positions.
    assert c.codebook_bits_sparse < c.codebook_bits


def test_bitmap_loses_to_dense_when_nothing_is_pruned():
    """The mask is a flat 1 bit/weight, so sparse is not free.

    At zero sparsity bitmap pays the mask for nothing. The break-even is
    roughly sparsity > 1/ceil(log2 K), and a paper that quotes the sparse
    number without saying this is overstating the method.
    """
    torch.manual_seed(0)
    w = torch.randn(16, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()

    _, st0 = compress_layer(w, scale, k=256, t_lo=0.0, t_hi=0.0,
                            quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    assert st0.sparsity == 0.0
    assert (price_layer(st0, fmt="bitmap").total_deployable
            > price_layer(st0, fmt="dense").total_deployable)

    # Prune past the break-even and the ordering flips.
    _, st1 = compress_layer(w, scale, k=256, t_lo=-1.0, t_hi=1.0,
                            quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    assert st1.sparsity > 0.6
    assert (price_layer(st1, fmt="bitmap").total_deployable
            < price_layer(st1, fmt="dense").total_deployable)


def test_csr_filler_count_matches_a_simulated_sparsity_pattern():
    """Regression: the filler count was estimated by spreading the MEAN gap.

    Gaps between survivors are geometric, so most are far shorter than the mean
    and need no filler at all, while a thin tail needs several each. The linear
    estimate `(mean_gap - 1) / 2^span` overcharged badly -- at 90% sparsity with
    an 8-bit span the true count is zero and it claimed 7,031 per two million
    weights -- which made wide gap fields look better than they are and pushed
    `auto` toward spans it should not have chosen.
    """
    from evolmc.codec import _expected_fillers

    rng = np.random.default_rng(0)
    n = 400_000
    for sparsity in (0.90, 0.95, 0.98):
        mask = rng.random(n) < (1 - sparsity)
        pos = np.flatnonzero(mask)
        gaps = np.diff(np.concatenate([[-1], pos]))
        n_alive = len(pos)
        for span in (4, 6, 8):
            true = float(np.floor(gaps / 2 ** span).sum())
            got = _expected_fillers(n, n_alive, span)
            # Within 10% of simulation, or within a handful of entries when the
            # true count is near zero.
            assert abs(got - true) <= max(0.10 * true, 20.0), (
                sparsity, span, true, got)


def test_wider_csr_span_is_not_free():
    """Every survivor pays the gap field, so widening it is a real trade.

    Without fillers modelled correctly a wider span looks strictly better, which
    it is not: at moderate sparsity the extra bits per survivor cost more than
    the fillers they avoid.
    """
    torch.manual_seed(0)
    w = torch.randn(16, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()
    _, st = compress_layer(w, scale, k=256, t_lo=-1.0, t_hi=1.0,
                           quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    assert 0.6 < st.sparsity < 0.75

    costs = {s: price_layer(st, fmt="csr", csr_span_bits=s).total_deployable
             for s in (2, 4, 8, 12, 16)}
    best = min(costs, key=costs.get)
    assert best not in (2, 16), costs      # neither extreme wins here
    assert costs[16] > costs[best], costs  # widening past the optimum costs


def test_auto_format_never_loses_to_a_fixed_one():
    """`auto` is the lower envelope, so it must beat every fixed choice.

    Which format wins moves with sparsity -- bitmap's flat mask beats CSR below
    roughly 85% and loses badly above it -- and the best CSR gap width moves
    too, so any fixed choice is wrong somewhere. The only cost is a tag per
    layer, which is a few bits against megabytes of indices.
    """
    from evolmc.codec import CSR_SPANS

    torch.manual_seed(0)
    w = torch.randn(32, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()

    for k in (16, 256):
        for t in (0.0, 0.5, 1.0, 1.65, 2.0):
            _, st = compress_layer(w, scale, k=k, t_lo=-t, t_hi=t,
                                   quant_cfg=cfg.quant, prune_cfg=cfg.prune,
                                   name="t")
            auto = price_layer(st, fmt="auto").total_deployable
            fixed = [price_layer(st, fmt="bitmap").total_deployable]
            fixed += [price_layer(st, fmt="csr", csr_span_bits=s).total_deployable
                      for s in CSR_SPANS]
            # Within the tag, which is the only thing auto pays extra.
            assert auto <= min(fixed) + 64, (k, t, auto, min(fixed))


def test_which_format_wins_flips_with_sparsity():
    """The reason a fixed format is the wrong call.

    Below the crossover the bitmask is cheap insurance; above it, one flat bit
    per ORIGINAL position dwarfs the handful of survivors and CSR's relative
    indices win. `auto` has to actually switch, not just pick one and stay.
    """
    torch.manual_seed(0)
    w = torch.randn(32, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()

    picks = {}
    for t in (0.5, 1.0, 1.65, 2.2):
        _, st = compress_layer(w, scale, k=256, t_lo=-t, t_hi=t,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
        picks[round(st.sparsity, 2)] = price_layer(st, fmt="auto").fmt

    assert len(set(picks.values())) > 1, picks
    lo = min(picks), max(picks)
    assert picks[lo[0]].startswith("bitmap"), picks
    assert picks[lo[1]].startswith("csr"), picks


def test_bitmap_has_a_one_bit_floor_and_csr_does_not():
    """Why the choice is qualitative, not just a few percent.

    The mask is one bit per ORIGINAL weight whatever the sparsity, so bitmap
    cannot go below 1.0 avg_bits however much is pruned. CSR stores nothing for a
    pruned weight and has no such floor. That is the whole reason the bitmap
    configs pin plot.xlim_min at 1.0.
    """
    from evolmc.codec import ModelCost

    torch.manual_seed(0)
    w = torch.randn(32, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()
    _, st = compress_layer(w, scale, k=16, t_lo=-2.5, t_hi=2.5,
                           quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    assert st.sparsity > 0.97

    avg_bits = lambda f, **kw: ModelCost(
        layers=[price_layer(st, fmt=f, **kw)], n_untouched_weights=0).avg_bits
    assert avg_bits("bitmap") > 1.0
    assert avg_bits("csr", csr_span_bits=8) < 1.0


def test_param_reduction_is_reported_separately_from_size():
    """Parameter count and storage are different claims and must not be mixed.

    They do not even move together: quantization compounds with pruning, so
    the size ratio can exceed the parameter ratio, while the untouched fp16
    embeddings drag the whole-model figure the other way. Reporting one as if
    it were the other is wrong in both directions.
    """
    from evolmc.codec import ModelCost

    torch.manual_seed(0)
    w = torch.randn(32, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()
    _, st = compress_layer(w, scale, k=16, t_lo=-1.65, t_hi=1.65,
                           quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    mc = ModelCost(layers=[price_layer(st, fmt="bitmap")],
                   n_untouched_weights=w.numel() // 2)

    assert st.sparsity > 0.88
    # Target weights are ~90% gone, but a third of the checkpoint is untouched,
    # so the whole-model parameter reduction is far below the layer sparsity.
    assert 0.55 < mc.param_reduction < 0.68
    # The two are genuinely different numbers, not one restated.
    assert abs(mc.cr_deploy - 1.0 / (1.0 - mc.param_reduction)) > 0.02
    assert mc.n_alive_total < mc.n_total_weights


def test_the_bitmask_dominates_storage_at_high_sparsity():
    """Why pruning saturates: past a point you are storing addresses, not weights.

    At 90% sparsity with 4-bit indices the mask is one flat bit per ORIGINAL
    position while the surviving indices are 4 bits per SURVIVOR, so the mask
    ends up the majority of the layer's deployable cost. Pruning harder shrinks
    the smaller term and leaves the larger one untouched, which is the ceiling
    on what unstructured sparsity can buy in this format.
    """
    torch.manual_seed(0)
    w = torch.randn(32, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()
    _, st = compress_layer(w, scale, k=16, t_lo=-1.65, t_hi=1.65,
                           quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    c = price_layer(st, fmt="bitmap")

    assert c.mask_bits / c.total_deployable > 0.6
    # CSR pays a gap field per survivor instead of a bit per position, so it
    # is the cheaper format exactly in this regime.
    assert (price_layer(st, fmt="csr").total_deployable
            < c.total_deployable)


def test_pruning_moves_the_archival_objective_and_not_the_deployable_one():
    """The premise of the *_3obj_prune configs, checked at objective level.

    The reserved zero codeword keeps the index width at ceil(log2 K) and
    k_centroids at K no matter how much is pruned, so t_lo/t_hi are invisible
    to avg_bits and fully visible to cr_archive. Two genes that move one
    size objective and not the other are what make the front genuinely 3-D --
    if this ever fails, those configs are searching a 2-D problem.
    """
    from evolmc.codec import ModelCost
    from evolmc.objectives import ObjectiveSet

    torch.manual_seed(0)
    w = torch.randn(64, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()
    objset = ObjectiveSet(("ppl_proxy", "avg_bits", "cr_archive"))

    deployable, archival, sparsity = [], [], []
    for t in (0.0, 0.25, 0.5, 1.0, 1.5):
        _, st = compress_layer(w, scale, k=64, t_lo=-t, t_hi=t,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
        cost = ModelCost(layers=[price_layer(st)], n_untouched_weights=10_000)
        _, avg_bits, cr = objset.values(1.0, cost.summary())
        deployable.append(avg_bits)
        archival.append(cr)
        sparsity.append(st.sparsity)

    # Pruning really is happening ...
    assert sparsity[0] == 0.0 and sparsity[-1] > 0.8
    # ... the deployable objective does not move at all ...
    assert len({round(b, 9) for b in deployable}) == 1, deployable
    # ... and every step of it buys archival compression.
    assert all(a < b for a, b in zip(archival, archival[1:])), archival
    assert archival[-1] > archival[0] * 1.5

    # k_centroids stays K, which is why the index width never widens.
    _, st = compress_layer(w, scale, k=64, t_lo=-1.5, t_hi=1.5,
                           quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    assert st.k_centroids == 64


def test_asymmetric_pruning_bands_are_reachable():
    """The band is two genes, so it need not be symmetric about zero."""
    torch.manual_seed(0)
    w = torch.randn(16, 2048)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()
    _, sym = compress_layer(w, scale, k=16, t_lo=-0.6, t_hi=0.6,
                            quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    _, asym = compress_layer(w, scale, k=16, t_lo=-0.6, t_hi=1.4,
                             quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
    # A wider positive side prunes strictly more.
    assert asym.sparsity > sym.sparsity
    # Still one reserved zero codeword, still the same index width.
    assert asym.k_centroids == sym.k_centroids == 16


def test_pruning_lowers_archival_cost_at_fixed_index_width():
    torch.manual_seed(0)
    w = torch.randn(8, 4096)
    scale = w.std(1, keepdim=True)
    cfg = _cfgs()
    costs = []
    for t in (0.0, 0.5, 1.0):
        _, st = compress_layer(w, scale, k=16, t_lo=-t, t_hi=t,
                               quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="t")
        costs.append(price_layer(st))
    # Fixed-width indices are unaffected ...
    assert len({c.index_bits_fixed for c in costs}) == 1
    # ... but the entropy coder turns sparsity into real savings.
    assert costs[0].index_bits_huffman > costs[1].index_bits_huffman
    assert costs[1].index_bits_huffman > costs[2].index_bits_huffman


# -- accounting ------------------------------------------------------------

def test_avg_bits_matches_hand_calculation():
    """4096x4096 per-channel, K=16: 4 index bits + 4096 codebooks x 16 x 16 b."""
    n, out_f, k = 4096 * 4096, 4096, 16
    counts = torch.full((k,), n / k, dtype=torch.float64)

    class S:
        name, n_weights, n_groups = "t", n, out_f
        k_nominal = k_centroids = k
        symbol_counts, k_used_mean, sparsity, mse = counts, k, 0.0, 0.0

    c = price_layer(S(), codebook_bits=16)
    assert c.index_bits_fixed == 4 * n
    assert c.codebook_bits == out_f * k * 16
    # 4 index bits + 4096*16*16 / 4096^2 = 4 + 0.0625 bits per weight.
    assert c.avg_bits_deployable == pytest.approx(4.0625)
    assert c.avg_bits_deployable - 4.0 == pytest.approx(0.0625)  # per-channel overhead


def test_per_group_codebooks_are_expensive():
    """Justifies per_channel as the default for codebook quantization."""
    n, k, gs = 4096 * 4096, 16, 128
    counts = torch.full((k,), n / k, dtype=torch.float64)

    def cost(n_groups):
        class S:
            name, n_weights = "t", n
            k_nominal = k_centroids = k
            symbol_counts, k_used_mean, sparsity, mse = counts, k, 0.0, 0.0
        S.n_groups = n_groups
        return price_layer(S(), 16).avg_bits_deployable

    per_channel = cost(4096)
    per_group = cost(n // gs)
    # Per channel: +0.0625 avg_bits of codebook. Per group of 128: +2.0 avg_bits, i.e.
    # the codebook costs half again as much as the indices it serves.
    assert per_channel == pytest.approx(4.0625)
    assert per_group == pytest.approx(6.0)


def test_model_cost_counts_untouched_weights():
    from evolmc.codec import ModelCost

    n = 4096 * 4096
    counts = torch.full((16,), n / 16, dtype=torch.float64)

    class S:
        name, n_weights, n_groups = "t", n, 4096
        k_nominal = k_centroids = 16
        symbol_counts, k_used_mean, sparsity, mse = counts, 16, 0.0, 0.0

    mc = ModelCost(layers=[price_layer(S())], n_untouched_weights=n)
    # Half the checkpoint stays fp16, so the honest whole-model avg_bits is the
    # AVERAGE of the compressed half (4.0625 bpw) and the untouched half (still
    # fp16, 16 bpw) -- not just the compressed matrices' own 4.0625, which is
    # what a target-only average would hide.
    assert mc.avg_bits == pytest.approx((4.0625 + 16) / 2)
    assert mc.cr_deploy == pytest.approx(32 / (4.0625 + 16))
    assert mc.cr_deploy < 1.6
    # Flat histogram: entropy coding cannot help, so archival must not look
    # better than deployable.
    assert mc.cr_archive <= mc.cr_deploy + 1e-6


# -- genome ----------------------------------------------------------------

def test_genome_decode_is_deterministic_and_covers_all_k():
    from evolmc.grouping import Genome
    from evolmc.models import TargetLayer

    layers = [
        TargetLayer(f"m.layers.{b}.{t}", torch.nn.Linear(4, 4), 4096 * 4096, 4096,
                    4096, b, t, False)
        for b in range(2)
        for t in ("q_proj", "o_proj")
    ]
    cfg = Config()
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert g.n_k == 2  # k_grouping="type"
    assert g.n_p == 1  # prune_grouping="global"
    assert g.n_var == 2 + 2

    for k in g.k_choices:
        s = g.decode(g.encode_uniform(k))
        assert {v.k for v in s.values()} == {k}

    # Every K choice is reachable from a uniform sweep of the unit interval.
    seen = {g.decode(np.full(g.n_var, u))[layers[0].name].k
            for u in np.linspace(0, 1, 200)}
    assert seen == set(g.k_choices)


def test_grouping_variable_counts():
    from evolmc.grouping import Genome
    from evolmc.models import TargetLayer

    layers = [
        TargetLayer(f"m.layers.{b}.{t}", torch.nn.Linear(4, 4), 4096 * 4096, 4096,
                    4096, b, t, False)
        for b in range(32)
        for t in ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
    ]
    cfg = Config()
    expected = {"global": 1, "type": 7, "block": 32, "block_type": 224}
    for scheme, n in expected.items():
        cfg.variables.k_grouping = scheme
        cfg.prune.enabled = False
        assert Genome(layers, cfg.quant, cfg.prune, cfg.variables).n_k == n

    # The full per-layer encoding is 672 variables -- the number that motivates
    # switching to U-NSGA-III.
    cfg.prune.enabled = True
    cfg.variables.k_grouping = cfg.variables.prune_grouping = "block_type"
    assert Genome(layers, cfg.quant, cfg.prune, cfg.variables).n_var == 224 * 3


# -- plot limits -----------------------------------------------------------

def test_perplexity_floor_of_zero_is_rejected_on_log_axis():
    """Perplexity is exp(cross-entropy) >= 1, and 0 is -inf on a log axis."""
    from evolmc.plotting import _check_ylim

    with pytest.raises(ValueError, match="bounded below by 1.0"):
        _check_ylim((0.0, 100.0), "log")
    with pytest.raises(ValueError, match="bounded below by 1.0"):
        _check_ylim((-5.0, 100.0), "log")

    # 1.0 is the true theoretical floor and must be allowed.
    assert _check_ylim((1.0, 100.0), "log") == (1.0, 100.0)
    # Linear axes can legitimately start at 0.
    assert _check_ylim((0.0, 100.0), "linear") == (0.0, 100.0)
    with pytest.raises(ValueError, match="increasing"):
        _check_ylim((100.0, 1.0), "log")


def test_ylim_min_ratio_lowers_the_box():
    from evolmc.plotting import derive_limits

    class FakeCost:
        def __init__(self, b): self.avg_bits = b

    class FakeGenome:
        k_choices = (4, 16)
        def encode_uniform(self, k): return k

    class FakeComp:
        genome = FakeGenome()
        def cost_only(self, k): return FakeCost(2.0 if k == 4 else 4.0)

    cfg = Config()
    cfg.plot.ylim_max_ratio = 10.0
    cfg.plot.ylim_pad = 0.0        # isolate the ratio from the padding
    _, tight = derive_limits(FakeComp(), cfg, baselines=[30.0, 50.0], fp16_ppl=25.0)
    cfg.plot.ylim_min_ratio = 0.5
    _, loose = derive_limits(FakeComp(), cfg, baselines=[30.0, 50.0], fp16_ppl=25.0)

    assert tight[0] == pytest.approx(25.0 * 0.9)
    assert loose[0] == pytest.approx(25.0 * 0.5)
    assert loose[0] < tight[0] and loose[1] == tight[1]


# -- layer role naming -----------------------------------------------------

def test_proj_type_disambiguates_by_parent_module():
    """GPT-2 has two distinct layers both named `c_proj`."""
    from evolmc.models import _proj_type

    assert _proj_type("transformer.h.0.attn.c_proj") == "attn.c_proj"
    assert _proj_type("transformer.h.0.mlp.c_proj") == "mlp.c_proj"
    assert _proj_type("transformer.h.0.attn.c_proj") != \
           _proj_type("transformer.h.0.mlp.c_proj")

    # Llama-style names were already unique; qualifying is just clearer.
    assert _proj_type("model.layers.3.self_attn.q_proj") == "self_attn.q_proj"
    assert _proj_type("model.layers.3.mlp.down_proj") == "mlp.down_proj"

    # A layer sitting directly under the block index keeps its bare name.
    assert _proj_type("transformer.h.0.c_fc") == "c_fc"
    assert _proj_type("dense") == "dense"

    # Biases keep the parent qualifier so attn/mlp c_proj.bias stay distinct.
    assert _proj_type("transformer.h.0.attn.c_proj.bias") == "attn.c_proj.bias"
    assert _proj_type("transformer.h.0.mlp.c_proj.bias") == "mlp.c_proj.bias"
    assert _proj_type("transformer.h.0.ln_1.bias") == "ln_1.bias"


def test_include_1d_picks_up_norms_and_biases():
    """`all` means every parameter, including LayerNorms and biases."""
    import torch.nn as nn

    from evolmc.models import (
        MasterWeights, count_untouched_weights, discover_targets,
    )

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8, bias=True)
            self.ln = nn.LayerNorm(8)

    m = Tiny()
    matrices = discover_targets(m, [])
    assert {t.name for t in matrices} == {"fc"}
    assert count_untouched_weights(m, matrices) == 8 + 8 + 8  # bias + ln w/b

    full = discover_targets(m, [], include_1d=True)
    names = {t.name for t in full}
    assert names == {"fc", "fc.bias", "ln", "ln.bias"}
    assert count_untouched_weights(m, full) == 0
    assert all(t.is_vector for t in full if t.name != "fc")

    master = MasterWeights(full, "cpu")
    layer = next(t for t in full if t.name == "ln")
    orig = master.original(layer).clone()
    master.write(layer, orig * 0.5)
    assert torch.allclose(layer.rows_view(layer.param()), orig * 0.5)
    master.restore(layer)
    assert torch.allclose(layer.rows_view(layer.param()), orig)


def test_vector_layers_do_not_cap_a_shared_k():
    """A 768-wide LayerNorm must not drag global K from 8192 down to 768."""
    from evolmc.grouping import Genome
    from evolmc.models import TargetLayer

    big = TargetLayer("fc", torch.nn.Linear(4, 4), 768 * 768, 768, 768,
                      0, "fc", False)
    vec = TargetLayer("ln", torch.nn.LayerNorm(768), 768, 1, 768,
                      0, "ln", False, param_name="weight", is_vector=True)
    cfg = Config()
    cfg.prune.enabled = False
    cfg.quant.granularity = "per_tensor"
    cfg.quant.k_encoding = "integer"
    cfg.quant.k_min, cfg.quant.k_max = 2, 8192
    cfg.variables.k_grouping = "global"
    g = Genome([big, vec], cfg.quant, cfg.prune, cfg.variables)
    assert int(g.k_max_group[0]) == 8192
    settings = g.decode(np.ones(g.n_var))
    assert settings["fc"].k == 8192
    assert settings["ln"].k == 768


def test_gpt2_type_grouping_separates_attn_and_mlp_projections():
    from evolmc.grouping import Genome
    from evolmc.models import TargetLayer, _proj_type

    names = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    layers = [
        TargetLayer(f"transformer.h.{b}.{n}", torch.nn.Linear(4, 4),
                    4096 * 4096, 4096, 4096, b,
                    _proj_type(f"transformer.h.{b}.{n}"), True)
        for b in range(12) for n in names
    ]
    cfg = Config()
    cfg.prune.enabled = False
    cfg.variables.k_grouping = "type"
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert g.n_k == 4, g.k_groups
    assert set(g.k_groups) == {"attn.c_attn", "attn.c_proj",
                               "mlp.c_fc", "mlp.c_proj"}


# -- integer K encoding ----------------------------------------------------

def _int_genome(k_min=2, k_max=8192, grouping="global"):
    from evolmc.grouping import Genome
    from evolmc.models import TargetLayer

    layers = [TargetLayer(f"m.layers.{b}.q_proj", torch.nn.Linear(4, 4),
                          4096 * 4096, 4096, 4096, b, "q_proj", False)
              for b in range(3)]
    cfg = Config()
    cfg.prune.enabled = False
    cfg.quant.k_encoding = "integer"
    cfg.quant.k_min, cfg.quant.k_max = k_min, k_max
    # per_tensor so the per-layer ceiling (16.7M values here) is not binding;
    # the ceiling itself is tested separately below.
    cfg.quant.granularity = "per_tensor"
    cfg.variables.k_grouping = grouping
    return Genome(layers, cfg.quant, cfg.prune, cfg.variables)


def test_integer_encoding_reaches_non_powers_of_two():
    g = _int_genome()
    seen = {g.decode(np.full(g.n_var, u))[g.layers[0].name].k
            for u in np.linspace(0, 1, 400)}
    assert not all(k & (k - 1) == 0 for k in seen), "only powers of two reached"
    assert min(seen) == 2 and max(seen) == 8192


def test_integer_encoding_is_log_spaced():
    """A linear map would spend half the gene range above K/2, where quality
    has plateaued, and leave almost no resolution at small K.

    Equal gene steps give equal *index-bit* steps. The match is approximate,
    not exact: rounding to an integer necessarily perturbs the spacing, and
    most so at small K, where between 2 and 4 only K=3 exists.
    """
    g = _int_genome(2, 8192)
    mid = g.decode(np.full(g.n_var, 0.5))[g.layers[0].name].k
    assert mid == pytest.approx(128, rel=0.05)      # 2^((1+13)/2) = 2^7

    ks = [g.decode(np.full(g.n_var, u))[g.layers[0].name].k
          for u in np.linspace(0, 1, 14)]
    steps = np.diff([math.log2(k) for k in ks])
    assert steps.min() > 0.75 and steps.max() < 1.1   # ~1 bit per step
    assert steps.std() < 0.07

    # A linear map over the same range would put the midpoint at ~4097 and
    # spend the whole upper half of the gene above 4096.
    linear_mid = (2 + 8192) / 2
    assert mid < linear_mid / 30


def test_integer_encode_decode_round_trips():
    g = _int_genome(2, 8192)
    for k in (2, 3, 7, 100, 255, 1000, 4095, 8192):
        x = g.encode_uniform(k)
        assert g.decode(x)[g.layers[0].name].k == k


def test_integer_references_stay_on_powers_of_two():
    """Baselines and warm starts must remain the interpretable fixed-bit
    configurations even when the search itself is unconstrained."""
    g = _int_genome(2, 8192)
    assert g.k_choices == (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                           4096, 8192)
    for k in g.k_choices:
        assert g.decode(g.encode_uniform(k))[g.layers[0].name].k == k


def test_integer_bounds_are_validated():
    with pytest.raises(ValueError, match="k_min"):
        _int_genome(1, 256)
    with pytest.raises(ValueError, match="k_min"):
        _int_genome(256, 256)
    g = _int_genome(2, 256)
    with pytest.raises(ValueError, match="outside"):
        g.encode_uniform(500)


def test_choices_encoding_is_unchanged():
    from evolmc.grouping import Genome
    from evolmc.models import TargetLayer

    layers = [TargetLayer("m.layers.0.q_proj", torch.nn.Linear(4, 4),
                          4096 * 4096, 4096, 4096, 0, "q_proj", False)]
    cfg = Config()
    cfg.prune.enabled = False
    cfg.quant.granularity = "per_tensor"
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert g.k_encoding == "choices"
    seen = {g.decode(np.full(g.n_var, u))[layers[0].name].k
            for u in np.linspace(0, 1, 200)}
    assert seen == set(cfg.quant.k_choices)


# -- per-layer K ceilings --------------------------------------------------

def _layer(name, out_f, in_f, block=0, ptype="q_proj"):
    from evolmc.models import TargetLayer
    return TargetLayer(name, torch.nn.Linear(4, 4), out_f * in_f, out_f, in_f,
                       block, ptype, False)


def test_max_k_for_layer_follows_granularity():
    """A codebook cannot hold more entries than the group has values."""
    from evolmc.grouping import max_k_for_layer

    lyr = _layer("m.layers.0.c_fc", 3072, 768)
    cfg = Config()
    for gran, expected in (("per_tensor", 3072 * 768),
                           ("per_channel", 768),
                           ("per_group", 128)):
        cfg.quant.granularity = gran
        assert max_k_for_layer(lyr, cfg.quant) == expected


def test_per_channel_patterns_override_per_tensor_for_wte():
    """A per_tensor run can still give wte / lm_head a codebook per token."""
    from evolmc.compressor import _n_groups
    from evolmc.grouping import layer_granularity, max_k_for_layer
    from evolmc.quantize import compress_layer

    wte = _layer("transformer.wte", 50257, 768, block=-1, ptype="transformer.wte")
    proj = _layer("transformer.h.0.mlp.c_fc", 3072, 768)
    cfg = Config()
    cfg.quant.granularity = "per_tensor"
    cfg.quant.per_channel_patterns = ("wte",)
    cfg.prune.enabled = False

    assert layer_granularity(wte.name, cfg.quant) == "per_channel"
    assert layer_granularity(proj.name, cfg.quant) == "per_tensor"
    assert max_k_for_layer(wte, cfg.quant) == 768
    assert max_k_for_layer(proj, cfg.quant) == 3072 * 768
    assert _n_groups(wte, cfg.quant) == 50257
    assert _n_groups(proj, cfg.quant) == 1

    rows = torch.randn(8, 16)
    recon, stats = compress_layer(
        rows, rows.std(dim=1, keepdim=True).clamp_min(1e-8),
        k=4, t_lo=0.0, t_hi=0.0,
        quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="transformer.wte")
    assert stats.n_groups == 8
    recon_p, stats_p = compress_layer(
        rows, rows.std(dim=1, keepdim=True).clamp_min(1e-8),
        k=4, t_lo=0.0, t_hi=0.0,
        quant_cfg=cfg.quant, prune_cfg=cfg.prune, name="transformer.h.0.mlp.c_fc")
    assert stats_p.n_groups == 1
    assert recon.shape == rows.shape
    assert recon_p.shape == rows.shape


def test_group_ceiling_is_the_minimum_over_member_layers():
    """One K must be valid for every layer the group covers, so a block's
    narrowest layer sets the limit for the whole block."""
    from evolmc.grouping import Genome

    layers = [_layer("transformer.h.0.mlp.c_fc", 3072, 768, 0, "mlp.c_fc"),
              _layer("transformer.h.0.mlp.c_proj", 768, 3072, 0, "mlp.c_proj")]
    cfg = Config()
    cfg.prune.enabled = False
    cfg.quant.granularity = "per_channel"
    cfg.quant.k_encoding = "integer"
    cfg.quant.k_min, cfg.quant.k_max = 2, 8192

    cfg.variables.k_grouping = "block"          # both layers share one K
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert g.n_k == 1
    assert int(g.k_max_group[0]) == 768         # the narrower layer wins
    assert g.capped is True

    cfg.variables.k_grouping = "block_type"     # independent K each
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert sorted(int(v) for v in g.k_max_group) == [768, 3072]


def test_genes_never_decode_above_a_group_ceiling():
    from evolmc.grouping import Genome

    layers = [_layer("transformer.h.0.mlp.c_fc", 3072, 768, 0, "mlp.c_fc"),
              _layer("transformer.h.0.mlp.c_proj", 768, 3072, 0, "mlp.c_proj")]
    cfg = Config()
    cfg.prune.enabled = False
    cfg.quant.granularity = "per_channel"
    cfg.quant.k_encoding = "integer"
    cfg.quant.k_min, cfg.quant.k_max = 2, 8192
    cfg.variables.k_grouping = "block_type"
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)

    caps = {l.name: l.in_features for l in layers}
    for u in np.linspace(0, 1, 50):
        for name, s in g.decode(np.full(g.n_var, u)).items():
            assert 2 <= s.k <= caps[name], (name, s.k)
    # The full gene range is used: each group reaches its own ceiling.
    top = g.decode(np.ones(g.n_var))
    assert {n: s.k for n, s in top.items()} == caps


def test_encode_uniform_clamps_groups_that_cannot_reach_k():
    from evolmc.grouping import Genome

    layers = [_layer("transformer.h.0.mlp.c_fc", 3072, 768, 0, "mlp.c_fc"),
              _layer("transformer.h.0.mlp.c_proj", 768, 3072, 0, "mlp.c_proj")]
    cfg = Config()
    cfg.prune.enabled = False
    cfg.quant.granularity = "per_channel"
    cfg.quant.k_encoding = "integer"
    cfg.quant.k_min, cfg.quant.k_max = 2, 8192
    cfg.variables.k_grouping = "block_type"
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)

    got = {n: s.k for n, s in g.decode(g.encode_uniform(2048)).items()}
    assert got["transformer.h.0.mlp.c_proj"] == 2048   # 2048 <= 3072, fine
    assert got["transformer.h.0.mlp.c_fc"] == 768      # clamped to its ceiling


def test_per_tensor_ceiling_is_not_binding_on_gpt2_sized_layers():
    """The three shipped GPT-2 configs use per_tensor, where the smallest
    layer still holds 589,824 values -- far above k_max=8192."""
    from evolmc.grouping import Genome

    layers = [_layer("transformer.h.0.attn.c_proj", 768, 768, 0, "attn.c_proj")]
    cfg = Config()
    cfg.prune.enabled = False
    cfg.quant.granularity = "per_tensor"
    cfg.quant.k_encoding = "integer"
    cfg.quant.k_min, cfg.quant.k_max = 2, 8192
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)
    assert g.capped is False
    assert g.decode(np.ones(g.n_var))[layers[0].name].k == 8192


# -- population initialization ---------------------------------------------

def test_logspace_init_is_uniform_per_individual_and_even_in_index_bits():
    """Every individual carries one K applied to all groups; the population
    spreads those K evenly in log space, i.e. evenly in index bits."""
    g = _int_genome(2, 8192, grouping="block")
    rng = np.random.default_rng(0)
    pop = g.seed_population(14, rng, "logspace")

    assert pop.shape == (14, g.n_var)
    ks = []
    for row in pop:
        per_group = {s.k for s in g.decode(row).values()}
        assert len(per_group) == 1, "an individual must be uniform across groups"
        ks.append(per_group.pop())

    assert ks[0] == 2 and ks[-1] == 8192
    assert ks == sorted(ks)
    steps = np.diff(np.log2(ks))
    assert steps.std() < 0.05          # even in bits, not in K
    assert max(ks) / min(ks) > 1000    # spans the range


def test_logspace_beats_linear_spread_at_the_low_end():
    """A linear spread would put most individuals above K/2, where quality has
    plateaued, and almost none in the small-K region that matters."""
    g = _int_genome(2, 8192, grouping="global")
    rng = np.random.default_rng(0)
    ks = [next(iter({s.k for s in g.decode(r).values()}))
          for r in g.seed_population(40, rng, "logspace")]
    below_256 = sum(k <= 256 for k in ks)
    linear = np.linspace(2, 8192, 40)
    assert below_256 >= 20                     # half the population
    assert sum(k <= 256 for k in linear) <= 2  # a linear spread gives ~1


def test_logspace_respects_per_group_ceilings():
    from evolmc.grouping import Genome

    layers = [_layer("transformer.h.0.mlp.c_fc", 3072, 768, 0, "mlp.c_fc"),
              _layer("transformer.h.0.mlp.c_proj", 768, 3072, 0, "mlp.c_proj")]
    cfg = Config()
    cfg.prune.enabled = False
    cfg.quant.granularity = "per_channel"
    cfg.quant.k_encoding = "integer"
    cfg.quant.k_min, cfg.quant.k_max = 2, 8192
    cfg.variables.k_grouping = "block_type"
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)

    pop = g.seed_population(20, np.random.default_rng(0), "logspace")
    caps = {l.name: l.in_features for l in layers}
    for row in pop:
        for name, st in g.decode(row).items():
            assert st.k <= caps[name]
    # Individuals below every ceiling stay genuinely uniform.
    low = g.decode(pop[0])
    assert len({s.k for s in low.values()}) == 1


def test_init_modes_are_selectable():
    g = _int_genome(2, 8192, grouping="block")
    rng = np.random.default_rng(0)

    rand = g.seed_population(20, rng, "random")
    assert rand.shape == (20, g.n_var)
    # Random individuals are essentially never uniform across 3 groups.
    assert sum(len({s.k for s in g.decode(r).values()}) == 1 for r in rand) < 3

    ladder = g.seed_population(20, rng, "ladder")
    assert ladder.shape == (20, g.n_var)

    with pytest.raises(ValueError, match="unknown init mode"):
        g.seed_population(4, rng, "nonsense")


def test_logspace_sweeps_the_pruning_band_too():
    from evolmc.grouping import Genome

    layers = [_layer("m.layers.0.q_proj", 4096, 4096, 0, "q_proj")]
    cfg = Config()
    cfg.prune.enabled = True
    cfg.prune.t_max = 2.0
    cfg.quant.granularity = "per_tensor"
    g = Genome(layers, cfg.quant, cfg.prune, cfg.variables)

    pop = g.seed_population(10, np.random.default_rng(0), "logspace")
    bands = [g.decode(r)[layers[0].name] for r in pop]
    assert bands[0].t_hi == pytest.approx(0.0)
    assert bands[-1].t_hi == pytest.approx(2.0)
    assert all(b.t_lo == -b.t_hi for b in bands)


# -- NSGA-II operator wiring -----------------------------------------------

def _val(x):
    """pymoo wraps operator parameters in Real/Choice objects."""
    return float(x.value) if hasattr(x, "value") else (x if x is None else float(x))


def _algo(**over):
    from evolmc.search import build_algorithm
    g = _int_genome(2, 8192, grouping="block")
    cfg = Config()
    for k, v in over.items():
        setattr(cfg.search, k, v)
    return build_algorithm(cfg, g, np.zeros((cfg.search.pop_size, g.n_var))), g


def test_mutation_prob_is_per_individual_not_per_variable():
    """The regression this guards against.

    pymoo's `prob` fires the operator on an individual; `prob_var` decides each
    gene. Putting the familiar 1/n_var into `prob` leaves only 1/n_var of the
    population mutated at all -- 2% at 48 variables, 0.1% at 672.
    """
    algo, g = _algo()
    assert _val(algo.mating.mutation.prob) == pytest.approx(1.0)
    # prob_var left to pymoo, which already uses min(0.5, 1/n_var).
    assert algo.mating.mutation.prob_var is None


def test_operator_parameters_reach_the_operators():
    algo, _ = _algo(crossover_prob=0.75, crossover_prob_var=0.3,
                    crossover_eta=22.0, mutation_prob=0.85,
                    mutation_prob_var=0.05, mutation_eta=11.0)
    cx, mu = algo.mating.crossover, algo.mating.mutation
    assert _val(cx.prob) == pytest.approx(0.75)
    assert _val(cx.prob_var) == pytest.approx(0.3)
    assert _val(cx.eta) == pytest.approx(22.0)
    assert _val(mu.prob) == pytest.approx(0.85)
    assert _val(mu.prob_var) == pytest.approx(0.05)
    assert _val(mu.eta) == pytest.approx(11.0)


def test_eliminate_duplicates_and_offspring_count_are_configurable():
    algo, _ = _algo(eliminate_duplicates=False, n_offsprings=7)
    assert algo.n_offsprings == 7
    algo, _ = _algo(pop_size=20)
    assert algo.n_offsprings == 20      # generational by default


@pytest.mark.parametrize("name", ["nsga2", "unsga3", "moead"])
def test_every_algorithm_builds_with_the_shared_operator_settings(name):
    algo, _ = _algo(algorithm=name, pop_size=12, crossover_eta=9.0,
                    mutation_eta=17.0)
    assert _val(algo.mating.crossover.eta) == pytest.approx(9.0)
    assert _val(algo.mating.mutation.eta) == pytest.approx(17.0)


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match="unknown algorithm"):
        _algo(algorithm="cmaes")


# -- history persistence ---------------------------------------------------

def test_history_survives_ragged_generations(tmp_path):
    """Regression: np.stack demands uniform shapes and blows up at the very
    end, after the whole search has been paid for. Population size is not
    guaranteed constant -- duplicate elimination, MOEA/D's own sizing, or a
    partial final generation can all vary it."""
    from evolmc.search import load_history, save_history

    history = [
        {"gen": 1, "X": np.zeros((10, 3)), "F": np.ones((10, 2))},
        {"gen": 2, "X": np.zeros((7, 3)),  "F": np.full((7, 2), 2.0)},
        {"gen": 3, "X": np.zeros((12, 3)), "F": np.full((12, 2), 3.0)},
    ]
    logged = []

    class Run:
        def log(self, msg="", echo=True): logged.append(msg)

    path = str(tmp_path / "history.npz")
    save_history(path, history, Run())
    assert any("varied" in m for m in logged)

    back = load_history(path)
    assert [g for g, _, _ in back] == [1, 2, 3]
    assert [len(F) for _, _, F in back] == [10, 7, 12]
    assert back[1][2][0, 0] == pytest.approx(2.0)


def test_uniform_generations_round_trip_without_a_note(tmp_path):
    from evolmc.search import load_history, save_history

    history = [{"gen": g, "X": np.full((5, 2), g), "F": np.full((5, 2), g)}
               for g in (1, 2, 3)]
    logged = []

    class Run:
        def log(self, msg="", echo=True): logged.append(msg)

    path = str(tmp_path / "h.npz")
    save_history(path, history, Run())
    assert not any("varied" in m for m in logged)
    back = load_history(path)
    assert [len(F) for _, _, F in back] == [5, 5, 5]
    assert back[2][1][0, 0] == pytest.approx(3.0)


def test_legacy_stacked_history_still_loads(tmp_path):
    """Runs made before ragged support stored 3-D stacked arrays."""
    from evolmc.search import load_history

    path = str(tmp_path / "old.npz")
    np.savez_compressed(path, gens=np.array([1, 2]),
                        X=np.zeros((2, 5, 3)), F=np.ones((2, 5, 2)))
    back = load_history(path)
    assert [g for g, _, _ in back] == [1, 2]
    assert all(F.shape == (5, 2) for _, _, F in back)
