"""The baseline has to be right or nothing else means anything.

The exact 1-D DP is checked against brute force over all interval partitions on
small inputs -- the only check that actually proves optimality rather than
plausibility. Everything downstream ("companding costs +x% over k-means") is a
statement about this number.
"""

import itertools

import numpy as np
import pytest

from cluster_bench.kmeans import (ExactKMeans1D, lloyd_1d, lloyd_multistart,
                                  sklearn_available, sklearn_kmeans)
from cluster_bench.metrics import sse


def brute_force_1d(x, k):
    """Optimal 1-D k-means by enumerating every way to cut the sorted array."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = len(xs)
    best = np.inf
    for cuts in itertools.combinations(range(1, n), k - 1):
        b = (0,) + cuts + (n,)
        tot = 0.0
        for i in range(k):
            seg = xs[b[i]:b[i + 1]]
            if seg.size:
                tot += float(((seg - seg.mean()) ** 2).sum())
        best = min(best, tot)
    return best


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_dp_matches_brute_force(k):
    x = np.round(np.random.default_rng(7).normal(size=25), 3)
    labels, cent = ExactKMeans1D(x, 5).fit(k)
    assert cent.shape[0] == k
    assert sse(x, labels, cent) == pytest.approx(brute_force_1d(x, k), rel=1e-9)


def test_dp_handles_ties():
    """Values are collapsed to (value, weight) pairs; the answer must not move."""
    x = np.repeat([0.0, 0.0, 1.0, 1.0, 1.0, 5.0, 9.0], 6)
    ek = ExactKMeans1D(x, 4)
    for k in (2, 3, 4):
        labels, cent = ek.fit(k)
        assert sse(x, labels, cent) == pytest.approx(brute_force_1d(x, k), rel=1e-9)


def test_dp_never_worse_than_lloyd():
    x = np.random.default_rng(3).normal(size=800)
    ek = ExactKMeans1D(x, 16)
    for k in (4, 8, 16):
        dp = sse(x, *ek.fit(k))
        ll = sse(x, *lloyd_multistart(x, k, n_init=12))
        assert dp <= ll * (1 + 1e-9), f"exact DP lost to Lloyd at K={k}"


def test_lloyd_fills_empty_clusters():
    """K distinct points requested from data with fewer distinct values.

    The failure this guards is silent: an emptied cluster whose mean is NaN
    swallows every point on the next assignment sweep, and the run reports a
    partition of one.
    """
    x = np.repeat([[0.0, 0.0], [1.0, 1.0]], 20, axis=0)
    labels, cent = lloyd_multistart(x, 6, n_init=4)
    assert np.isfinite(cent).all()
    assert cent.shape[0] <= 6
    assert sse(x, labels, cent) == pytest.approx(0.0, abs=1e-12)


def test_lloyd_1d_monotone():
    x = np.random.default_rng(5).normal(size=2000)
    init = np.quantile(x, (np.arange(8) + 0.5) / 8)
    labels, cent = lloyd_1d(x, init)
    start = float(((x - init[np.abs(x[:, None] - init[None, :]).argmin(1)]) ** 2).sum())
    assert sse(x, labels, cent) <= start + 1e-9


@pytest.mark.skipif(not sklearn_available(), reason="scikit-learn not installed")
def test_sklearn_agrees_with_lloyd():
    """Two independent implementations should land within a few percent."""
    x = np.random.default_rng(11).normal(size=(1500, 3))
    a = sse(x, *sklearn_kmeans(x, 8, n_init=10))
    b = sse(x, *lloyd_multistart(x, 8, n_init=10))
    assert abs(a - b) / min(a, b) < 0.05
