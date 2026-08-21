"""Non-uniform quantization by companding -- the scalar quantizer under test.

A companding quantizer replaces "where do I put the K decision boundaries?"
with "what monotone warp F: R -> [0,1] should I bin uniformly in?". Given F the
assignment is a closed form -- floor(K * F(x)) -- so a candidate costs one pass
over the data, which is what makes thousands of NSGA-II fitness evaluations
affordable. This is a numpy port of the torch code in `evolmc/quantize.py`; the
benchmark deliberately does not import the parent package, so the two can drift
only by intent.

F is a composition, F = F_residual o F_gamma:

  F_gamma     the Bennett / Panter-Dite density-matched backbone. The optimal
              level density for MSE is lambda(x) ~ p(x)^(1/3); here the
              exponent is a free gene, estimated from a `grid`-bin histogram of
              the data clipped to mean +- alpha*std. gamma=0 gives a uniform
              CDF over the clip range (plain uniform binning), gamma=1/3 is the
              classic MSE-optimal quantizer, gamma=1 equalizes bin populations.

  F_residual  a monotone correction on [0,1] built from M positive segment
              weights (softmax of the genes `u`). It is what lets the search
              leave the analytic backbone when the high-rate asymptotics that
              justify Panter-Dite do not hold -- which, at the K a clustering
              benchmark cares about, is most of the time.

ONE DELIBERATE DEVIATION from evolmc: the clip window is centred on the sample
mean rather than on zero. Network weights are near zero-mean so the two agree
there, but clustering benchmarks are not, and a zero-centred window would throw
the data away before the warp ever saw it.

Reconstruction never inverts F. As in evolmc, a codeword is the mean of the
original (unwarped) values that landed in its bin, so F is only ever evaluated
forward, at the data points.
"""

from __future__ import annotations

import functools

import numpy as np


def softmax(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    e = np.exp(u - u.max())
    return e / e.sum()


@functools.lru_cache(maxsize=None)
def _bspline_collocation(n_ctrl: int, degree: int, grid: int) -> np.ndarray:
    """[n_ctrl, grid+1] collocation matrix for a clamped uniform B-spline.

    Row i, column g holds B_i(g/grid). A curve S(u) = sum_i c_i B_i(u) built
    from non-decreasing control points is itself non-decreasing -- the
    variation-diminishing property, the same thing that makes Ramsay's
    I-splines monotone, reached by evaluating the basis rather than by
    integrating M-splines by hand. degree=1 reproduces the piecewise-linear
    hats the "linear" residual builds in closed form.

    Cached: the knots depend only on (n_ctrl, degree, grid), never on the
    genome, so this recursion runs once per shape for the life of the process.
    """
    if n_ctrl <= degree:
        raise ValueError(
            f"ispline degree={degree} needs at least {degree + 1} control "
            f"points, i.e. residual_genes >= {degree} (got {n_ctrl - 1})")
    order = degree + 1
    n_interior = n_ctrl - order
    knots = np.concatenate([
        np.zeros(order),
        np.linspace(0.0, 1.0, n_interior + 2)[1:-1],
        np.ones(order),
    ])
    # Nudge the right edge so every span's membership test stays half-open; the
    # true value at u=1 is patched in below from the clamped-spline identity
    # S(1) = last control point, which needs no numerical care.
    x = np.linspace(0.0, 1.0, grid + 1)
    x_open = np.minimum(x, 1.0 - 1e-12)

    n_spans = len(knots) - 1
    basis = np.zeros((n_spans, grid + 1))
    for i in range(n_spans):
        lo, hi = knots[i], knots[i + 1]
        if hi > lo:
            basis[i] = (x_open >= lo) & (x_open < hi)
    for p in range(1, order):
        nb = np.zeros((n_spans - p, grid + 1))
        for i in range(n_spans - p):
            left = np.zeros(grid + 1)
            den = knots[i + p] - knots[i]
            if den > 0:
                left = (x_open - knots[i]) / den * basis[i]
            right = np.zeros(grid + 1)
            den = knots[i + p + 1] - knots[i + 1]
            if den > 0:
                right = (knots[i + p + 1] - x_open) / den * basis[i + 1]
            nb[i] = left + right
        basis = nb
    basis[:, -1] = 0.0
    basis[-1, -1] = 1.0
    return basis


def _residual_curve(u: np.ndarray, grid: int, residual_type: str,
                    degree: int) -> np.ndarray:
    """The monotone correction sampled on `grid`+1 equally spaced points of [0,1].

    u == 0 decodes to equal increments either way, but only "linear" turns that
    into the exact identity: a clamped B-spline reproduces a straight line only
    when its control points sit at the (boundary-crowded) Greville abscissae,
    not at uniform increments. So "ispline" at u == 0 is a smooth monotone curve
    close to but not equal to the identity, and the gap shrinks as M grows.
    """
    m = len(u)
    slopes = m * softmax(u)                                       # positive, mean 1
    breakpoints = np.concatenate([[0.0], np.cumsum(slopes / m)])  # [M+1], 0 -> 1
    breakpoints[-1] = 1.0

    t = np.linspace(0.0, 1.0, grid + 1)
    if residual_type == "linear":
        pos = np.clip(t * m, 0.0, m - 1e-9)
        seg = np.floor(pos).astype(np.int64)
        frac = pos - seg
        curve = breakpoints[seg] + frac * slopes[seg] / m
    elif residual_type == "ispline":
        curve = breakpoints @ _bspline_collocation(m + 1, degree, grid)
    else:
        raise ValueError(f"unknown residual_type: {residual_type}")
    curve = np.clip(curve, 0.0, 1.0)
    curve[0], curve[-1] = 0.0, 1.0
    return np.maximum.accumulate(curve)   # float-precision insurance only


def companding_forward(x: np.ndarray, alpha: float, gamma: float,
                       u: np.ndarray, grid: int = 256,
                       residual_type: str = "linear",
                       degree: int = 3) -> np.ndarray:
    """Evaluate F(x) in [0,1] for every sample. Same shape as `x`."""
    x = np.asarray(x, dtype=np.float64).ravel()
    mu, sd = x.mean(), x.std()
    if not np.isfinite(sd) or sd <= 0.0:
        return np.zeros_like(x)
    lo, hi = mu - alpha * sd, mu + alpha * sd
    xc = np.clip(x, lo, hi)

    span = max(hi - lo, 1e-12)
    bin_w = span / grid
    bucket = np.clip(((xc - lo) / bin_w).astype(np.int64), 0, grid - 1)
    hist = np.bincount(bucket, minlength=grid).astype(np.float64)
    dens = np.maximum(hist / max(hist.sum(), 1.0), 1e-8)
    lam = dens ** gamma
    cdf = np.cumsum(lam)
    cdf /= max(cdf[-1], 1e-12)
    f_base = np.concatenate([[0.0], cdf])          # [grid+1]

    # Compose by table: read the residual curve at F_gamma's own grid values,
    # then read the composed curve at the data. Two interpolations, whatever
    # the sample size.
    resid = _residual_curve(np.asarray(u, dtype=np.float64), grid,
                            residual_type, degree)
    t = np.linspace(0.0, 1.0, grid + 1)
    f_grid = np.interp(f_base, t, resid)
    f_grid[0], f_grid[-1] = 0.0, 1.0
    f_grid = np.maximum.accumulate(np.clip(f_grid, 0.0, 1.0))

    return np.clip(np.interp(xc, np.linspace(lo, hi, grid + 1), f_grid), 0.0, 1.0)


def companding_assign(f: np.ndarray, k: int) -> np.ndarray:
    """Uniform binning in the warped domain -- floor(K * F(x))."""
    return np.clip((f * k).astype(np.int64), 0, k - 1)


def compact_labels(x: np.ndarray, idx: np.ndarray):
    """Drop empty bins, relabel 0..K_eff-1, return (labels, centroids).

    An empty bin is not a cluster. Counting it would let the search inflate K
    for free and would leave Davies-Bouldin measuring a centroid that no point
    ever chose.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    _, inv = np.unique(np.asarray(idx).ravel(), return_inverse=True)
    inv = inv.ravel().astype(np.int64)
    n_eff = int(inv.max()) + 1 if inv.size else 0
    counts = np.bincount(inv, minlength=n_eff).astype(np.float64)
    cent = np.empty((n_eff, x.shape[1]), dtype=np.float64)
    for d in range(x.shape[1]):
        cent[:, d] = np.bincount(inv, weights=x[:, d], minlength=n_eff) / counts
    return inv, cent


def companding_quantize_1d(x: np.ndarray, k: int, alpha: float, gamma: float,
                           u: np.ndarray, grid: int = 256,
                           residual_type: str = "linear", degree: int = 3,
                           lloyd_iters: int = 0):
    """Warp, bin, take bin means. Returns (labels, centroids [K_eff, 1]).

    `lloyd_iters` > 0 runs that many Lloyd sweeps seeded from the warp's own
    assignment. Zero -- the default -- keeps the quantizer a pure compander,
    which is the thing being benchmarked; the knob exists so the "warp as a
    k-means initialiser" ablation is one config line away.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    f = companding_forward(x, alpha, gamma, u, grid, residual_type, degree)
    idx = companding_assign(f, k)
    labels, cent = compact_labels(x, idx)
    for _ in range(lloyd_iters):
        if len(cent) < 2:
            break
        c = np.sort(cent[:, 0])
        idx = np.searchsorted((c[:-1] + c[1:]) / 2.0, x)
        new_labels, new_cent = compact_labels(x, idx)
        if np.array_equal(new_labels, labels):
            break
        labels, cent = new_labels, new_cent
    return labels, cent


def companding_quantize_md(x: np.ndarray, ks, alphas, gammas, us,
                           grid: int = 256, residual_type: str = "linear",
                           degree: int = 3):
    """Per-dimension companding -- a product quantizer.

    Each column gets its own warp and its own K_d, so the cells form an
    axis-aligned non-uniform grid of prod(K_d) boxes. Only occupied boxes are
    clusters, so K_eff <= min(n, prod K_d), and K_eff is what gets reported and
    what the k-means baseline is matched against.

    This is the honest multi-dimensional reading of a scalar compander, and it
    is expected to lose to k-means on SSE: k-means places free centroids, this
    places a separable grid. What the benchmark asks is how much it loses, and
    whether the cluster-validity axis narrows the gap.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    codes = np.empty((n, d), dtype=np.int64)
    for j in range(d):
        f = companding_forward(x[:, j], float(alphas[j]), float(gammas[j]),
                               np.asarray(us[j]), grid, residual_type, degree)
        codes[:, j] = companding_assign(f, int(ks[j]))
    # np.unique over rows rather than a mixed-radix integer code: prod(K_d)
    # overflows int64 for even modest d, and the row form costs one lexsort.
    _, inv = np.unique(codes, axis=0, return_inverse=True)
    return compact_labels(x, inv.ravel())
