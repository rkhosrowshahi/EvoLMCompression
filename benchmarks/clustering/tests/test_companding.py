"""Properties the warp must have, independent of whether it wins anything.

Monotonicity is the load-bearing one: if F is not non-decreasing then uniform
binning in the warped domain no longer corresponds to an interval partition of
the input, bins interleave, and both the distortion and the validity numbers
stop meaning what the report says they mean.
"""

import numpy as np
import pytest

from cluster_bench.companding import (_residual_curve, companding_assign,
                                      companding_edges, companding_forward,
                                      companding_quantize_1d,
                                      companding_quantize_md)
from cluster_bench.metrics import sse

RNG = np.random.default_rng(0)
SAMPLES = {
    "gaussian": RNG.normal(size=5000),
    "laplace": RNG.laplace(size=5000),
    "lognormal": RNG.lognormal(0, 1, 5000),
    "bimodal": np.concatenate([RNG.normal(-5, 0.4, 2500),
                               RNG.normal(6, 1.5, 2500)]),
}


@pytest.mark.parametrize("name", list(SAMPLES))
@pytest.mark.parametrize("gamma", [0.0, 1 / 3, 1.0, 1.5])
@pytest.mark.parametrize("residual", ["linear", "ispline"])
def test_forward_is_monotone_and_bounded(name, gamma, residual):
    x = SAMPLES[name]
    u = RNG.normal(size=6) * 2.0
    f = companding_forward(x, 4.0, gamma, u, residual_type=residual)
    assert f.min() >= 0.0 and f.max() <= 1.0
    order = np.argsort(x)
    diffs = np.diff(f[order])
    assert diffs.min() >= -1e-9, "warp is not monotone in x"


def test_linear_residual_at_zero_is_the_identity():
    """u = 0 must decode to the exact identity for the piecewise-linear basis.

    The i-spline basis deliberately does NOT -- a clamped B-spline reproduces a
    straight line only at the Greville abscissae -- so it is checked as "close",
    not "equal", and the gap must shrink with more control points.
    """
    t = np.linspace(0, 1, 257)
    assert np.abs(_residual_curve(np.zeros(6), 256, "linear", 3) - t).max() < 1e-12
    gap6 = np.abs(_residual_curve(np.zeros(6), 256, "ispline", 3) - t).max()
    gap32 = np.abs(_residual_curve(np.zeros(32), 256, "ispline", 3) - t).max()
    assert gap32 < gap6 < 0.2


def test_gamma_zero_is_uniform_binning_over_the_clip():
    """gamma=0 flattens the density term, so F is the linear ramp on [lo, hi]."""
    x = SAMPLES["laplace"]
    f = companding_forward(x, 3.0, 0.0, np.zeros(6))
    mu, sd = x.mean(), x.std()
    lo, hi = mu - 3 * sd, mu + 3 * sd
    expect = (np.clip(x, lo, hi) - lo) / (hi - lo)
    assert np.abs(f - expect).max() < 5e-3


def test_gamma_one_equalizes_bin_populations():
    """gamma=1 makes F the empirical CDF, so the bins should be near-equal."""
    x = SAMPLES["lognormal"]
    labels, cent = companding_quantize_1d(x, 16, 12.0, 1.0, np.zeros(6))
    counts = np.bincount(labels)
    assert counts.min() / counts.max() > 0.6


def test_empty_bins_are_dropped_not_counted():
    """K_eff must be occupancy, not the K the genome asked for."""
    x = np.concatenate([np.zeros(500), np.ones(500)])
    labels, cent = companding_quantize_1d(x, 64, 4.0, 0.0, np.zeros(6))
    assert cent.shape[0] == len(np.unique(labels)) <= 2


def test_lloyd_iters_only_helps():
    """The reassign ablation is seeded from the warp, so it cannot lose SSE."""
    x = SAMPLES["bimodal"]
    u = RNG.normal(size=6)
    plain = sse(x, *companding_quantize_1d(x, 24, 4.0, 0.6, u))
    polished = sse(x, *companding_quantize_1d(x, 24, 4.0, 0.6, u, lloyd_iters=20))
    assert polished <= plain + 1e-9


def test_product_quantizer_cells_are_the_intersection_of_scalar_bins():
    """The multi-D partition must equal the intersection of the d scalar ones.

    That identity is what makes this a product quantizer -- and it is exactly
    why it cannot represent a rotated cluster, which is the limitation the
    multi-D suite exists to quantify. Checked both ways: same code tuple implies
    same label, and same label implies same code tuple.
    """
    ks, alphas, gammas = [4, 5, 3], [4.0] * 3, [1 / 3] * 3
    us = [np.zeros(6)] * 3
    x = RNG.normal(size=(2000, 3))
    labels, cent = companding_quantize_md(x, ks, alphas, gammas, us)
    assert cent.shape[0] == len(np.unique(labels))

    codes = np.column_stack([
        companding_assign(companding_forward(x[:, j], alphas[j], gammas[j], us[j]),
                          ks[j]) for j in range(3)])
    _, code_id = np.unique(codes, axis=0, return_inverse=True)
    code_id = code_id.ravel()
    # A bijection between cell labels and code tuples: neither may split or
    # merge the other.
    assert len(np.unique(np.column_stack([labels, code_id]), axis=0)) == \
        len(np.unique(labels)) == len(np.unique(code_id))

    # And the centroid of a cell is the mean of the points assigned to it.
    for c in np.unique(labels)[:20]:
        assert np.allclose(cent[c], x[labels == c].mean(axis=0))


def test_degenerate_input_does_not_crash():
    labels, cent = companding_quantize_1d(np.full(100, 3.0), 8, 4.0, 0.5,
                                          np.zeros(6))
    assert cent.shape[0] == 1
    assert np.isfinite(cent).all()


@pytest.mark.parametrize("name", ["gaussian", "laplace", "bimodal"])
@pytest.mark.parametrize("k", [3, 8, 16])
def test_recovered_edges_sit_at_the_realised_transitions(name, k):
    """`companding_edges` must invert the SAME warp the assignment used.

    The first version inverted F sampled on a uniform grid over the data range.
    `companding_forward` estimates its density histogram from whatever array it
    is given, so that fitted the warp to the grid -- a uniform density -- and
    returned boundaries for a quantizer that never ran: an edge at 1.61 where
    the true one was at 0.54. Only a plot would have shown it, and only if
    someone looked closely.
    """
    x = SAMPLES[name]
    u = np.linspace(-1, 1, 6)
    labels, _ = companding_quantize_1d(x, k, 4.0, 0.6, u)
    edges = companding_edges(x, k, 4.0, 0.6, u)

    order = np.argsort(x)
    xs, ls = x[order], labels[order]
    assert np.all(np.diff(ls) >= 0), "1-D bins must be intervals"
    gaps = [(xs[i], xs[i + 1]) for i in np.flatnonzero(np.diff(ls) != 0)]
    assert len(edges) == len(gaps)
    for e in edges:
        assert any(a - 1e-9 <= e <= b + 1e-9 for a, b in gaps), e
