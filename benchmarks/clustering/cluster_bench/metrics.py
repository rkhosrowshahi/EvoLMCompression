"""Distortion and cluster-validity measures, computed the same way for every method.

The whole benchmark turns on this file being method-blind: companding, Lloyd,
the exact DP and sklearn all hand back (labels, centroids) and are scored here
by identical code. Anything measured differently between arms is not a result.

Everything is a MINIMIZATION quantity by the time it leaves `evaluate`, so
`neg_silhouette` rather than `silhouette` (which is reported alongside, raw).
Degenerate partitions -- fewer than two occupied clusters -- return the worst
finite value the axis admits rather than NaN, so NSGA-II's non-dominated sort
can still rank them instead of choking.

No scikit-learn import here on purpose: the validity indices are twenty lines
of numpy each, and making the core scoring path depend on an optional package
would mean the benchmark's own numbers change with what happens to be
installed. sklearn appears only as one of the k-means baselines, in kmeans.py.
"""

from __future__ import annotations

import numpy as np

WORST_DB = 1e6      # Davies-Bouldin is unbounded above; this stands in for "useless"
WORST_SIL = 1.0     # neg_silhouette in [-1, 1]; +1 is the worst possible


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def sse(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Within-cluster sum of squares -- the objective k-means minimizes."""
    x = _as2d(x)
    return float(((x - centroids[labels]) ** 2).sum())


def mse(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """SSE per sample. Scale-free across datasets of different size, unlike SSE."""
    x = _as2d(x)
    return sse(x, labels, centroids) / x.shape[0]


def davies_bouldin(x: np.ndarray, labels: np.ndarray,
                   centroids: np.ndarray) -> float:
    """Mean over clusters of the worst-case (spread_i + spread_j) / dist(c_i, c_j).

    Lower is better. Spread is the mean distance to the own centroid, which is
    the standard Davies-Bouldin (1979) choice with p=q=1 -- deliberately not the
    root-mean-square variant, so DB is not a monotone transform of the SSE axis
    and the two objectives can genuinely conflict.

    Coincident centroids would divide by zero. They can happen here: a product
    quantizer cell holding one point sits exactly on that point, and two such
    cells can be arbitrarily close. Clamping the denominator turns that into a
    very large ratio -- correct, since two indistinguishable clusters ARE a bad
    partition -- instead of an inf that would poison the mean.
    """
    x = _as2d(x)
    k = centroids.shape[0]
    if k < 2:
        return WORST_DB
    spread = np.zeros(k)
    d = np.linalg.norm(x - centroids[labels], axis=1)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    sums = np.bincount(labels, weights=d, minlength=k)
    np.divide(sums, counts, out=spread, where=counts > 0)

    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    ratio = (spread[:, None] + spread[None, :]) / np.maximum(dist, 1e-12)
    np.fill_diagonal(ratio, -np.inf)
    return float(min(np.max(ratio, axis=1).mean(), WORST_DB))


def silhouette(x: np.ndarray, labels: np.ndarray, max_n: int = 2000,
               seed: int = 0) -> float:
    """Mean silhouette width in [-1, 1]. Higher is better.

    Exact silhouette is O(n^2) in memory and time, and this runs inside a
    fitness function -- so it is computed on a fixed random subsample of at most
    `max_n` points. Fixed, not resampled per call: a subsample that moved
    between evaluations would make the objective stochastic and NSGA-II would
    spend its budget chasing sampling noise. The same subsample indices are used
    for every method on a given dataset, so the arms stay comparable.

    Singleton clusters get a silhouette of 0 by the usual convention (a(i) is
    undefined with no same-cluster neighbours).
    """
    x = _as2d(x)
    n = x.shape[0]
    if n > max_n:
        idx = np.random.default_rng(seed).choice(n, max_n, replace=False)
        idx.sort()
        x, labels = x[idx], labels[idx]
        _, labels = np.unique(labels, return_inverse=True)
    k = int(labels.max()) + 1 if labels.size else 0
    if k < 2 or x.shape[0] < 3:
        return -1.0

    # GEMM form of the pairwise distances: the broadcast version allocates an
    # [n, n, d] temporary, which at max_n=2000 in 32-D is 1 GB.
    sq = (x ** 2).sum(1)
    dist = np.sqrt(np.maximum(sq[:, None] - 2.0 * (x @ x.T) + sq[None, :], 0.0))
    onehot = np.zeros((x.shape[0], k))
    onehot[np.arange(x.shape[0]), labels] = 1.0
    counts = onehot.sum(0)
    tot = dist @ onehot                          # [n, k] summed distance per cluster
    own = counts[labels]
    a = np.where(own > 1, tot[np.arange(len(labels)), labels] /
                 np.maximum(own - 1, 1), 0.0)
    mean_other = np.divide(tot, np.maximum(counts, 1)[None, :],
                           out=np.full_like(tot, np.inf), where=counts[None, :] > 0)
    mean_other[np.arange(len(labels)), labels] = np.inf
    b = mean_other.min(axis=1)
    s = np.where(own > 1, (b - a) / np.maximum(a, b), 0.0)
    return float(np.mean(s[np.isfinite(s)]))


def entropy_bits(labels: np.ndarray) -> float:
    """Shannon entropy of the label distribution, in bits per sample.

    Not an objective by default, but reported everywhere: it is the rate a real
    entropy coder would pay for this partition, and it is the axis the parent
    project optimizes. A companding front that matches k-means on distortion
    while spending fewer bits is a different -- and stronger -- claim than
    matching it at equal K.
    """
    counts = np.bincount(labels)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def adjusted_rand(labels: np.ndarray, y_true: np.ndarray) -> float:
    """Adjusted Rand index against the generating labels. 1 = exact recovery.

    The only measure here that cannot be gamed by partition shape, and the
    reason it earns its place: EVERY internal index in this file rewards
    "one far outlier versus everything else". On a standard normal sample that
    split scores Davies-Bouldin 0.100 against a balanced split's 0.597, AND
    silhouette 0.854 against 0.560 -- both indices prefer it, and neither is
    measuring a clustering. ARI scores it at essentially zero, because it asks
    a different question: do these labels agree with the truth?

    Reported, never optimized. The search is not shown `y_true` -- a method that
    tuned itself on the answer would not be a clustering method.

    Chance-corrected in the usual way (Hubert & Arabie 1985): the expected index
    under a random partition with the same marginals is subtracted off, so the
    score cannot be inflated by simply choosing more clusters.
    """
    labels = np.asarray(labels).ravel()
    y_true = np.asarray(y_true).ravel()
    n = labels.size
    if n < 2 or y_true.size != n:
        return float("nan")
    _, a_idx = np.unique(labels, return_inverse=True)
    _, b_idx = np.unique(y_true, return_inverse=True)
    na, nb = a_idx.max() + 1, b_idx.max() + 1
    cont = np.bincount(a_idx * nb + b_idx, minlength=na * nb).reshape(na, nb)

    comb2 = lambda v: (v * (v - 1) / 2.0).sum()
    sum_ij = comb2(cont.astype(np.float64))
    a = comb2(cont.sum(axis=1).astype(np.float64))
    b = comb2(cont.sum(axis=0).astype(np.float64))
    total = n * (n - 1) / 2.0
    expected = a * b / total
    maximum = 0.5 * (a + b)
    if maximum == expected:
        return 0.0
    return float((sum_ij - expected) / (maximum - expected))


def calinski_harabasz(x: np.ndarray, labels: np.ndarray,
                      centroids: np.ndarray) -> float:
    """Variance-ratio criterion. Higher is better; reported, never optimized."""
    x = _as2d(x)
    n, k = x.shape[0], centroids.shape[0]
    if k < 2 or n <= k:
        return 0.0
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    grand = x.mean(axis=0)
    between = float((counts * ((centroids - grand) ** 2).sum(axis=1)).sum())
    within = sse(x, labels, centroids)
    if within <= 0.0:
        return 0.0
    return float(between / within * (n - k) / (k - 1))


def evaluate(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
             silhouette_max_n: int = 2000, seed: int = 0,
             with_silhouette: bool = True, y_true=None) -> dict:
    """Every measure for one partition. Keys here are the objective registry.

    `with_silhouette=False` returns NaN in the two silhouette fields and skips
    the pairwise-distance matrix that produces them. Silhouette is O(n^2) and
    everything else here is O(n) or O(K^2), so when it is not being optimized
    it is the entire cost of a fitness evaluation; the search turns it off and
    the reporting path -- a few dozen rows -- turns it back on.
    """
    x = _as2d(x)
    k_eff = int(centroids.shape[0])
    counts = np.bincount(labels, minlength=max(k_eff, 1))
    smallest = int(counts[counts > 0].min()) if k_eff else 0
    if k_eff < 2:
        # One cluster is a legal quantizer and a useless clustering. Give the
        # distortion axis its true value -- it is honest and finite -- and the
        # validity axes their worst, so the point is dominated rather than
        # invisible.
        return {
            "sse": sse(x, labels, centroids),
            "mse": mse(x, labels, centroids),
            "davies_bouldin": WORST_DB,
            "neg_silhouette": WORST_SIL,
            "silhouette": -1.0,
            "calinski_harabasz": 0.0,
            "k_eff": k_eff,
            "min_cluster_size": smallest,
            "min_cluster_frac": smallest / max(x.shape[0], 1),
            "adjusted_rand": (adjusted_rand(labels, y_true)
                              if y_true is not None else float("nan")),
            "entropy_bits": entropy_bits(labels),
            "index_bits": float(np.log2(max(k_eff, 1))),
        }
    sil = silhouette(x, labels, silhouette_max_n, seed) if with_silhouette \
        else float("nan")
    return {
        "sse": sse(x, labels, centroids),
        "mse": mse(x, labels, centroids),
        "davies_bouldin": davies_bouldin(x, labels, centroids),
        "neg_silhouette": -sil,
        "silhouette": sil,
        "calinski_harabasz": calinski_harabasz(x, labels, centroids),
        "k_eff": k_eff,
        # Always reported, because Davies-Bouldin can be GAMED by a tiny
        # cluster: a singleton has zero spread, DB is a ratio of spreads, so
        # "one outlier vs everything else" scores better than any real
        # partition. Observed on the gaussian set -- a 1-vs-3999 split scored
        # DB 0.206 against k-means' 0.594 at the same K. Any DB win should be
        # read next to this column. The silhouette does not have the same hole:
        # a singleton scores 0 by convention and drags the mean DOWN.
        "min_cluster_size": smallest,
        "min_cluster_frac": smallest / max(x.shape[0], 1),
        "adjusted_rand": (adjusted_rand(labels, y_true)
                          if y_true is not None else float("nan")),
        "entropy_bits": entropy_bits(labels),
        "index_bits": float(np.log2(k_eff)),
    }


#: Measures NSGA-II may minimize. Anything else in `evaluate` is reported only.
MINIMIZED = ("mse", "sse", "davies_bouldin", "neg_silhouette", "k_eff",
             "entropy_bits", "index_bits")
