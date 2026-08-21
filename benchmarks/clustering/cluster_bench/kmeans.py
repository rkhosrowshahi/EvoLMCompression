"""The k-means baselines. Three of them, because "k-means" is not one number.

Lloyd is a LOCAL optimiser and what it returns depends heavily on where it
started. An earlier comparison in this project's history reported companding
"beating k-means" by 86%; that was a weak reference, not a result. So the
baseline is deliberately plural:

  dp      exact 1-D k-means by dynamic programming (Ckmeans.1d.dp, Wang & Song
          2011). Globally optimal SSE for the given K. Only exists in 1-D --
          k-means is NP-hard in general -- which is the main reason the 1-D
          suite is the primary result: there is a true optimum to measure
          against, not just whatever Lloyd happened to converge to.
  lloyd   multi-start Lloyd, k-means++ and friends, best of `n_init`. This is
          what a practitioner actually gets.
  sklearn sklearn.cluster.KMeans, same K, as an independent implementation
          check. Optional: if scikit-learn is not installed the arm is skipped
          and the run says so, rather than failing.

All three return (labels, centroids) with labels compacted to 0..K_eff-1, so
metrics.evaluate scores them by exactly the same code that scores companding.
"""

from __future__ import annotations

import numpy as np

from .companding import compact_labels


# --------------------------------------------------------------------------
# exact 1-D k-means, by dynamic programming
# --------------------------------------------------------------------------

def _dp_tables(vals: np.ndarray, w: np.ndarray, k_max: int):
    """Run the DP once, keep the argmin tables for every m <= k_max.

    dp[m][j] = min cost of partitioning the first j+1 sorted values into m
    contiguous runs. Contiguity is the whole trick: in 1-D an optimal k-means
    partition is always a set of intervals, so the search collapses from
    Stirling-many assignments to a shortest-path problem.

    The cost matrix is concave-Monge, so the argmin is monotone in j and the
    layer can be filled by divide and conquer in O(n log n) instead of O(n^2).
    Each recursion node evaluates its whole candidate range as one vectorized
    numpy expression -- with a scalar inner loop this would be the slowest
    thing in the benchmark by a wide margin.

    Returns the argmin table; backtracking for any m <= k_max is then free,
    which is what makes a whole K sweep cost one DP run.
    """
    n = len(vals)
    p0 = np.concatenate([[0.0], np.cumsum(w)])
    p1 = np.concatenate([[0.0], np.cumsum(w * vals)])
    p2 = np.concatenate([[0.0], np.cumsum(w * vals * vals)])

    def cost(i, j):
        """Weighted within-run sum of squares over sorted values i..j inclusive."""
        s0 = np.maximum(p0[j + 1] - p0[i], 1e-300)
        s1 = p1[j + 1] - p1[i]
        s2 = p2[j + 1] - p2[i]
        return np.maximum(s2 - s1 * s1 / s0, 0.0)

    prev = cost(0, np.arange(n))
    args = np.zeros((k_max + 1, n), dtype=np.int64)
    for m in range(2, k_max + 1):
        cur = np.full(n, np.inf)
        arg = np.zeros(n, dtype=np.int64)

        stack = [(m - 1, n - 1, m - 1, n - 1)]
        while stack:
            jlo, jhi, ilo, ihi = stack.pop()
            if jlo > jhi:
                continue
            jm = (jlo + jhi) // 2
            hi = min(ihi, jm)
            if ilo > hi:
                cur[jm], arg[jm] = np.inf, ilo
                stack.append((jlo, jm - 1, ilo, ihi))
                stack.append((jm + 1, jhi, ilo, ihi))
                continue
            cand = np.arange(ilo, hi + 1)
            vals_c = prev[cand - 1] + cost(cand, jm)
            b = int(np.argmin(vals_c))
            best_i = int(cand[b])
            cur[jm], arg[jm] = vals_c[b], best_i
            stack.append((jlo, jm - 1, ilo, best_i))
            stack.append((jm + 1, jhi, best_i, ihi))

        args[m] = arg
        prev = cur
    return args


def _dp_backtrack(vals, w, args, k: int):
    """Read one K's interval boundaries out of the shared argmin table."""
    n = len(vals)
    bounds = []
    j = n - 1
    for m in range(k, 1, -1):
        i = int(args[m][j])
        bounds.append(i)
        j = i - 1
    bounds.reverse()
    return np.array(bounds, dtype=np.int64)      # left index of runs 2..k


class ExactKMeans1D:
    """Globally optimal 1-D k-means for every K up to `k_max`, from one DP run.

    Duplicate values are collapsed to (value, weight) pairs first -- exact, and
    on data with heavy ties (quantized sensor readings, integer features) it
    shrinks the DP by orders of magnitude. If the number of DISTINCT values
    still exceeds `max_n` the DP runs on a random subsample and the instance is
    flagged `exact=False`; the boundaries it found are then polished by Lloyd
    against the FULL data. Without that polish a subsampled DP can score worse
    than plain multi-start Lloyd, which reads as a weak baseline when it is
    really just a sampling artefact. Either way the returned partition is
    scored on the full data, never on the subsample.
    """

    def __init__(self, x: np.ndarray, k_max: int, max_n: int = 4000,
                 seed: int = 0):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.x = x
        vals, counts = np.unique(x, return_counts=True)
        self.exact = True
        if len(vals) > max_n:
            self.exact = False
            sub = np.random.default_rng(seed).choice(x, max_n, replace=False)
            vals, counts = np.unique(sub, return_counts=True)
        self.vals = vals
        self.w = counts.astype(np.float64)
        self.k_max = int(min(k_max, len(vals)))
        self._args = _dp_tables(self.vals, self.w, self.k_max)

    def fit(self, k: int):
        k = int(min(k, self.k_max))
        if k <= 1:
            return compact_labels(self.x, np.zeros(len(self.x), dtype=np.int64))
        left = _dp_backtrack(self.vals, self.w, self._args, k)
        edges = self.vals[left]                  # left endpoints of runs 2..k
        # side="right", so a value sitting exactly ON an edge joins the run
        # that edge OPENS -- the convention the DP used when it chose the
        # split. "left" would push it back into the previous run and quietly
        # cost a few percent of SSE, which looks like a weak baseline rather
        # than like a bug.
        idx = np.searchsorted(edges, self.x, side="right")
        labels, cent = compact_labels(self.x, idx)
        if not self.exact:
            labels, cent = lloyd_1d(self.x, cent[:, 0])
        return labels, cent


# --------------------------------------------------------------------------
# Lloyd, multi-start
# --------------------------------------------------------------------------

def lloyd_1d(x: np.ndarray, centroids: np.ndarray, iters: int = 300):
    """Lloyd on the line: assignment is a searchsorted against the midpoints.

    O(n log K) per sweep instead of the O(nK) distance matrix the general
    routine builds, which is what makes it cheap enough to run inside a fitness
    evaluation (the `lloyd_iters` companding ablation) and as the polish step
    above.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    cent = np.sort(np.asarray(centroids, dtype=np.float64).ravel())
    # The sweep is bincount-only and the labels are compacted once, at the end.
    # Compacting every iteration means an np.unique -- a full O(n log n) sort of
    # the label array -- inside the hot loop, which on the 1-D suite costs more
    # than everything else in the baseline put together.
    for _ in range(iters):
        if len(cent) < 2:
            break
        idx = np.searchsorted((cent[:-1] + cent[1:]) / 2.0, x)
        cnt = np.bincount(idx, minlength=len(cent)).astype(np.float64)
        sums = np.bincount(idx, weights=x, minlength=len(cent))
        occupied = cnt > 0
        new = np.sort(sums[occupied] / cnt[occupied])
        if len(new) == len(cent) and np.allclose(new, cent, rtol=0, atol=1e-12):
            break
        cent = new
    if len(cent) < 2:
        return compact_labels(x, np.zeros(len(x), dtype=np.int64))
    return compact_labels(x, np.searchsorted((cent[:-1] + cent[1:]) / 2.0, x))


def _kmeanspp(x: np.ndarray, k: int, rng) -> np.ndarray:
    """D^2 seeding (Arthur & Vassilvitskii)."""
    n = x.shape[0]
    cent = [x[rng.integers(n)]]
    d2 = ((x - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        tot = d2.sum()
        if tot <= 0:
            cent.append(x[rng.integers(n)])
        else:
            cent.append(x[rng.choice(n, p=np.maximum(d2, 0) / tot)])
        d2 = np.minimum(d2, ((x - cent[-1]) ** 2).sum(1))
    return np.array(cent)


def _init_centroids(x: np.ndarray, k: int, how: str, rng) -> np.ndarray:
    n, d = x.shape
    if how == "kmeans++":
        return _kmeanspp(x, k, rng)
    if how == "random":
        return x[rng.choice(n, size=min(k, n), replace=False)]
    if how == "quantile":
        q = (np.arange(k) + 0.5) / k
        return np.column_stack([np.quantile(x[:, j], q) for j in range(d)])
    if how == "uniform":
        return np.column_stack([
            np.linspace(x[:, j].min(), x[:, j].max(), k + 2)[1:-1]
            for j in range(d)])
    raise ValueError(f"unknown init: {how}")


def lloyd(x: np.ndarray, k: int, init: str = "kmeans++", iters: int = 300,
          tol: float = 1e-10, rng=None):
    """One Lloyd run. Empty clusters are reseeded to the current worst-fit point.

    Dropping empty clusters instead would silently change K and make the
    matched-K comparison a lie, so they are refilled -- the standard fix, and
    the one sklearn uses.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    rng = np.random.default_rng(0) if rng is None else rng
    cent = _init_centroids(x, k, init, rng)
    k = cent.shape[0]
    if x.shape[1] == 1:
        # Same fixed point, ~27x faster at K=256 on 20k points: on the line the
        # assignment is a searchsorted against the midpoints instead of an
        # [n, K] distance matrix per sweep. The 1-D suite fits this at ten K
        # values for ten inits per dataset, so it is not a micro-optimization.
        return lloyd_1d(x[:, 0], cent[:, 0], iters)
    prev = np.inf
    labels = np.zeros(x.shape[0], dtype=np.int64)
    xsq = (x ** 2).sum(1)[:, None]
    for _ in range(iters):
        # ||x - c||^2 expanded into a GEMM. The broadcast form allocates an
        # [n, k, d] temporary every sweep, which on the larger benchmark sets
        # is the single slowest thing in the whole run -- and the baseline is
        # fitted at every K, for several inits, on every dataset.
        d2 = xsq - 2.0 * (x @ cent.T) + (cent ** 2).sum(1)[None, :]
        labels = d2.argmin(1)
        counts = np.bincount(labels, minlength=k)
        for j in np.flatnonzero(counts == 0):
            far = int(d2[np.arange(len(labels)), labels].argmax())
            cent[j] = x[far]
            labels[far] = j
            counts = np.bincount(labels, minlength=k)
        # Reseeding can itself empty a cluster (the donated point may have been
        # its only member), so the mean is taken only where there is something
        # to average and the rest keep the centroid they had. Dividing by the
        # raw count here produces NaN centroids that then swallow every point.
        cnt = counts.astype(np.float64)
        new = cent.copy()
        nonempty = cnt > 0
        for dd in range(x.shape[1]):
            sums = np.bincount(labels, weights=x[:, dd], minlength=k)
            new[nonempty, dd] = sums[nonempty] / cnt[nonempty]
        cent = new
        cur = float(((x - cent[labels]) ** 2).sum())
        if abs(prev - cur) <= tol * max(cur, 1e-30):
            break
        prev = cur
    return compact_labels(x, labels)


def lloyd_multistart(x: np.ndarray, k: int, n_init: int = 10,
                     inits=("kmeans++", "random", "quantile", "uniform"),
                     iters: int = 300, seed: int = 0):
    """Best-of-`n_init` Lloyd, cycling through the initialisation schemes.

    Cycling rather than sampling one scheme: k-means++ is the right default but
    on 1-D data quantile and uniform seeds land in different basins, and taking
    the best across schemes is what keeps the baseline strong enough that a
    companding win means something.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    best = None
    best_sse = np.inf
    # "quantile" and "uniform" ignore the rng, so repeating them across restarts
    # re-runs an identical Lloyd. Skipping the repeats spends the whole budget
    # on seeds that can actually land somewhere new.
    deterministic_done = set()
    for i in range(n_init):
        how = inits[i % len(inits)]
        if how in ("quantile", "uniform"):
            if how in deterministic_done:
                continue
            deterministic_done.add(how)
        rng = np.random.default_rng(seed + i)
        labels, cent = lloyd(x, k, how, iters, rng=rng)
        s = float(((x - cent[labels]) ** 2).sum())
        if s < best_sse:
            best, best_sse = (labels, cent), s
    return best


# --------------------------------------------------------------------------
# scikit-learn
# --------------------------------------------------------------------------

def sklearn_available() -> bool:
    try:
        import sklearn.cluster  # noqa: F401
        return True
    except ImportError:
        return False


def sklearn_kmeans(x: np.ndarray, k: int, n_init: int = 10, seed: int = 0):
    """sklearn.cluster.KMeans, for cross-implementation agreement.

    Raises ImportError if scikit-learn is missing; callers check
    `sklearn_available()` first and skip the arm.
    """
    from sklearn.cluster import KMeans
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)
    return compact_labels(x, km.labels_)
