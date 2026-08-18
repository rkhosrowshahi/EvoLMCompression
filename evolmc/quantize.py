"""Codebook construction and weight replacement.

Pipeline for one weight matrix, given (K, t_lo, t_hi):

  1. prune   -- zero every weight inside the band [t_lo, t_hi]
  2. bin     -- partition the *surviving* weights into K-1 bins
  3. center  -- each codeword is the mean of the weights in its bin
  4. replace -- hard-substitute every weight by its codeword

Pruned weights are folded into the codebook as a reserved zero codeword rather
than carried in a separate bitmask. That costs no extra index bits and makes
pruning show up where it should: as a sharply skewed symbol distribution that
the entropy coder in `codec.py` then exploits. It is also why the nominal K
buys K-1 centroids whenever the pruning band is non-degenerate.

Everything is batched over codebook groups so a whole layer is one pass of GPU
work -- this is what makes thousands of fitness evaluations affordable.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch

from .grouping import layer_granularity

NEG_SYMBOL = -1  # internal marker for "pruned", remapped to symbol 0 at the end


@dataclass
class LayerQuantStats:
    """Everything `codec.py` needs to price this layer."""

    name: str
    n_weights: int
    n_groups: int  # number of codebooks stored
    k_nominal: int  # codebook size the genome asked for
    k_centroids: int  # entries actually stored per codebook
    symbol_counts: torch.Tensor  # [k_nominal] histogram over the whole layer
    k_used_mean: float  # mean non-empty symbols per group (uniform-binning waste)
    sparsity: float
    mse: float  # relative reconstruction error, useful as a layer-wise proxy


def _reshape_groups(rows: torch.Tensor, granularity: str, group_size: int) -> torch.Tensor:
    """[out, in] -> [n_codebooks, weights_per_codebook]."""
    out_f, in_f = rows.shape
    if granularity == "per_tensor":
        return rows.reshape(1, -1)
    if granularity == "per_channel":
        return rows
    if granularity == "per_group":
        if in_f % group_size != 0:
            raise ValueError(
                f"in_features={in_f} is not divisible by group_size={group_size}"
            )
        return rows.reshape(out_f * (in_f // group_size), group_size)
    raise ValueError(f"unknown granularity: {granularity}")


def _uniform_range(w: torch.Tensor, alive: torch.Tensor, kc: int):
    """(lo, step) for `kc` equal-width bins spanning the surviving weights."""
    big = torch.finfo(w.dtype).max
    lo = torch.where(alive, w, torch.full_like(w, big)).min(dim=1, keepdim=True).values
    hi = torch.where(alive, w, torch.full_like(w, -big)).max(dim=1, keepdim=True).values
    # Groups with no survivors collapse to a degenerate range; harmless because
    # every one of their entries is masked out below.
    step = (hi - lo).clamp_min(1e-12) / kc
    return lo, step


def _uniform_edges(w: torch.Tensor, alive: torch.Tensor, kc: int) -> torch.Tensor:
    lo, step = _uniform_range(w, alive, kc)
    offsets = torch.arange(kc, device=w.device, dtype=w.dtype).view(1, -1)
    return lo + offsets * step


def _uniform_assign(w: torch.Tensor, lo: torch.Tensor, step: torch.Tensor,
                    kc: int) -> torch.Tensor:
    """Bin index by arithmetic rather than by search.

    Equal-width bins make the index a closed form, which avoids
    `torch.searchsorted` entirely. That matters far more than it looks: on MPS
    searchsorted over a 2304x768 layer costs ~420 ms against ~0.9 ms here, and
    the assignment step runs once per layer per fitness evaluation.
    """
    idx = ((w - lo) / step).floor_()
    return idx.clamp_(0, kc - 1).long()


def _quantile_edges(w: torch.Tensor, alive: torch.Tensor, kc: int) -> torch.Tensor:
    """Left edges of `kc` equal-population bins over the surviving weights."""
    big = torch.finfo(w.dtype).max
    masked = torch.where(alive, w, torch.full_like(w, big))
    ordered, _ = torch.sort(masked, dim=1)
    n_alive = alive.sum(dim=1, keepdim=True).clamp_min(1)
    frac = torch.arange(kc, device=w.device, dtype=w.dtype).view(1, -1) / kc
    ranks = (frac * n_alive.to(w.dtype)).long().clamp_(max=w.shape[1] - 1)
    edges = torch.gather(ordered, 1, ranks)
    # Enforce strict monotonicity so searchsorted stays well defined when many
    # survivors share a value (common once pruning bites).
    return torch.cummax(edges, dim=1).values


def _widths_edges(w: torch.Tensor, alive: torch.Tensor, kc: int,
                  z: torch.Tensor) -> torch.Tensor:
    """kc left edges from kc genome-controlled log-widths.

    The plain, backbone-free non-uniform quantizer: the genes ARE the bin
    widths, directly in the original weight domain -- no density estimate,
    no warp. `z` already lives in [widths_log_lo, widths_log_hi] (grouping.py
    puts it there), so no runtime clamp is needed the way a hand-rolled
    clip(z, -700, 700) would be -- the genome cannot leave that box.

    l_i = (hi-lo) * exp(z_i) / sum(exp(z)), accumulated into kc left edges,
    spanning this group's own surviving-weight range. Only the first `kc` of
    `z` are read: the gene block is sized to the widest K any group in this
    run could reach (Genome.width_dim), so a group whose own K is smaller
    just leaves the rest of z unused for this call.

    Reuses _uniform_range purely to recover this group's own (lo, hi) in one
    min/max pass; `step` itself is discarded once span = step * kc rebuilds
    hi - lo exactly.
    """
    lo, step = _uniform_range(w, alive, kc)
    span = step * kc
    g = torch.exp(z[:kc]).clamp_min(1e-30)
    frac = g / g.sum()
    cum_before = torch.cat([torch.zeros(1, device=z.device, dtype=frac.dtype),
                            torch.cumsum(frac, dim=0)[:-1]])  # [kc] left-edge fractions
    edges = lo + cum_before.view(1, -1) * span
    # Enforce strict monotonicity so searchsorted stays well defined -- exp()
    # is already strictly positive, this is float-precision insurance only,
    # the same guard _quantile_edges applies to its own edges.
    return torch.cummax(edges, dim=1).values


def _assign(w: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(edges.contiguous(), w.contiguous(), right=True) - 1
    return idx.clamp_(0, edges.shape[1] - 1)


def _centroids(w: torch.Tensor, idx: torch.Tensor, alive: torch.Tensor, kc: int):
    """Bin means over surviving weights. Returns (centroids, counts)."""
    g = w.shape[0]
    wf = torch.where(alive, w, torch.zeros_like(w)).to(torch.float32)
    ones = alive.to(torch.float32)
    sums = torch.zeros(g, kc, device=w.device, dtype=torch.float32).scatter_add_(1, idx, wf)
    cnts = torch.zeros(g, kc, device=w.device, dtype=torch.float32).scatter_add_(1, idx, ones)
    cent = sums / cnts.clamp_min(1.0)
    return cent.to(w.dtype), cnts


def _lloyd(w, alive, kc, idx, init, iters):
    """1-D k-means, batched over groups.

    Seeded from the *uniform* binning rather than the quantile one. Lloyd only
    ever decreases MSE, so starting from the better of the two closed-form
    binnings guarantees `kmeans` dominates `uniform` at equal K -- seeding from
    quantile edges instead leaves it stuck in a local optimum that is worse
    than plain uniform binning for K >= 64.
    """
    cent, cnts = _centroids(w, idx, alive, kc)
    # Empty bins have no defined mean; park them on their initial midpoint so
    # they do not drag the decision boundaries around.
    cent = torch.where(cnts > 0, cent, init)

    if kc > 1:
        for _ in range(iters):
            cent, _ = torch.sort(cent, dim=1)
            mids = 0.5 * (cent[:, 1:] + cent[:, :-1])
            new_idx = torch.searchsorted(mids.contiguous(), w.contiguous())
            if torch.equal(new_idx, idx):
                break
            idx = new_idx
            new_cent, cnts = _centroids(w, idx, alive, kc)
            cent = torch.where(cnts > 0, new_cent, cent)

    # Make counts consistent with the assignment actually returned.
    final_cent, cnts = _centroids(w, idx, alive, kc)
    cent = torch.where(cnts > 0, final_cent, cent)
    return idx, cent, cnts


def _masked_std(w: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Per-group std over the surviving weights only."""
    alive_f = alive.to(torch.float32)
    n_alive = alive_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (w * alive_f).sum(dim=1, keepdim=True) / n_alive
    var = ((w - mean).pow(2) * alive_f).sum(dim=1, keepdim=True) / n_alive
    return var.clamp_min(1e-12).sqrt()


@functools.lru_cache(maxsize=None)
def _bspline_collocation(n_ctrl: int, degree: int, grid: int) -> torch.Tensor:
    """[n_ctrl, grid+1] collocation matrix for a clamped uniform B-spline.

    Row i, column g holds B_i(g/grid) for the degree-`degree` basis function
    of a curve with `n_ctrl` control points. A curve S(u) = sum_i c_i * B_i(u)
    built from non-decreasing control points c is itself non-decreasing --
    the variation-diminishing property that is also what makes Ramsay's
    I-splines monotone, reached here by integrating M-splines instead of by
    hand-rolling that recursion. `degree=1` reproduces the piecewise-linear
    hat functions the "linear" residual already builds by closed form;
    `degree=3` is the cubic I-spline case.

    Cached: knots depend only on (n_ctrl, degree, grid), never on the genome,
    so this recursion runs once per shape config for the life of the process.
    """
    if n_ctrl <= degree:
        raise ValueError(
            f"companding_ispline_degree={degree} needs at least {degree + 1} "
            f"control points, i.e. companding_residual_genes >= {degree} "
            f"(got companding_residual_genes={n_ctrl - 1})")
    order = degree + 1
    n_interior = n_ctrl - order
    knots = np.concatenate([
        np.zeros(order),
        np.linspace(0.0, 1.0, n_interior + 2)[1:-1],
        np.ones(order),
    ])
    # Nudge the right edge so every span's membership test can stay half-open;
    # the true value at u=1 is patched in afterwards from the clamped-spline
    # identity S(1) = last control point, which needs no numerical care.
    x = np.linspace(0.0, 1.0, grid + 1)
    x_open = np.minimum(x, 1.0 - 1e-12)

    n_spans = len(knots) - 1
    basis = np.zeros((n_spans, grid + 1))
    for i in range(n_spans):
        lo, hi = knots[i], knots[i + 1]
        if hi > lo:
            basis[i] = (x_open >= lo) & (x_open < hi)
    for p in range(1, order):
        new_basis = np.zeros((n_spans - p, grid + 1))
        for i in range(n_spans - p):
            left = np.zeros(grid + 1)
            denom = knots[i + p] - knots[i]
            if denom > 0:
                left = (x_open - knots[i]) / denom * basis[i]
            right = np.zeros(grid + 1)
            denom = knots[i + p + 1] - knots[i + 1]
            if denom > 0:
                right = (knots[i + p + 1] - x_open) / denom * basis[i + 1]
            new_basis[i] = left + right
        basis = new_basis
    basis[:, -1] = 0.0
    basis[-1, -1] = 1.0
    return torch.from_numpy(basis).to(torch.float32)


def _companding_forward(w: torch.Tensor, alive: torch.Tensor, alpha: float,
                        gamma: float, u: torch.Tensor, grid: int = 256,
                        residual_type: str = "linear", degree: int = 3) -> torch.Tensor:
    """Companding warp F = F_residual o F_gamma, evaluated at every weight.

    F_gamma is the Bennett/Panter-Dite density-matched backbone: level density
    lambda(x) ~ p(x)^gamma, estimated per group from a `grid`-bin histogram of
    its own surviving weights, clipped to +-alpha standard deviations. gamma=0
    reduces the backbone to a uniform CDF (the clip range becomes the only
    difference from plain uniform binning); gamma=1/3 is the classic MSE-
    optimal quantizer; gamma=1 equalizes bin probabilities.

    F_residual is a monotone correction on [0,1], built from `len(u)` positive
    segment weights (softmax of `u`) that become the successive increments of
    M+1 non-decreasing control points (0 = ... = 1). u == 0 decodes to equal
    increments in both cases, but only "linear" turns that into the exact
    identity map -- a clamped B-spline reproduces a straight line only when
    its control points sit at the (non-uniform, boundary-crowded) Greville
    abscissae, not at uniform arithmetic increments, so "ispline" at u == 0
    is a smooth, monotone, F(0)=0/F(1)=1 curve that is close to but not
    exactly the identity (the gap shrinks as M grows, e.g. ~0.08 of the unit
    range at M=6 vs ~0.03 at M=32, measured at companding_ispline_degree=3).

    `residual_type` picks the basis those control points are read through:
    "linear" connects them with straight segments (closed form, the original
    construction). "ispline" reads them through a degree-`degree` monotone
    B-spline (see `_bspline_collocation`) evaluated on a `grid`-point lookup
    table and then linearly interpolated -- the same table-plus-interpolate
    treatment already used for F_gamma's histogram and for the final gather
    back to actual weight values below, so it costs one more cheap gather,
    not a slower assignment path.

    Returns F(w) in [0,1], same shape as `w`. Reconstruction never inverts F:
    like every other binning mode here, the codeword is the bin mean of the
    original (unwarped) values (`_centroids`), so F only needs to be evaluated
    forward, at the actual data points.
    """
    std = _masked_std(w, alive)
    lo = -alpha * std
    hi = alpha * std
    xc = torch.min(torch.max(w, lo), hi)

    span = (hi - lo).clamp_min(1e-12)
    bin_w = span / grid
    bucket = ((xc - lo) / bin_w).floor().clamp_(0, grid - 1).long()
    hist = torch.zeros(w.shape[0], grid, device=w.device, dtype=torch.float32)
    ones = alive.to(torch.float32)
    hist.scatter_add_(1, bucket, ones)
    dens = (hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)).clamp_min(1e-8)
    lam = dens.pow(gamma)
    cdf = torch.cumsum(lam, dim=1)
    cdf = cdf / cdf[:, -1:].clamp_min(1e-12)
    f_base = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=1)  # [G, grid+1]

    m = u.shape[0]
    slopes = m * torch.softmax(u, dim=0)  # [M], positive, mean 1
    seg = slopes / m
    breakpoints = torch.cat([torch.zeros(1, device=u.device, dtype=seg.dtype),
                             torch.cumsum(seg, dim=0)])  # [M+1], 0 -> 1

    if residual_type == "linear":
        pos = (f_base * m).clamp(0, m - 1e-6)
        seg_idx = pos.floor().long().clamp(0, m - 1)
        frac = pos - seg_idx.to(pos.dtype)
        f_grid = breakpoints[seg_idx] + frac * slopes[seg_idx] / m
    elif residual_type == "ispline":
        basis = _bspline_collocation(m + 1, degree, grid).to(
            device=breakpoints.device, dtype=breakpoints.dtype)  # [M+1, grid+1]
        lookup = breakpoints @ basis  # [grid+1], the residual curve on its own grid
        pos = (f_base * grid).clamp(0, grid)
        idx0 = pos.floor().long().clamp(0, grid - 1)
        idx1 = idx0 + 1
        frac = (pos - idx0.to(pos.dtype)).clamp(0, 1)
        f_grid = lookup[idx0] + frac * (lookup[idx1] - lookup[idx0])
    else:
        raise ValueError(f"unknown companding residual_type: {residual_type}")
    f_grid = f_grid.clamp(0.0, 1.0)
    f_grid[:, -1] = 1.0

    pos_w = ((xc - lo) / bin_w).clamp(0, grid)
    idx0 = pos_w.floor().long().clamp(0, grid - 1)
    idx1 = idx0 + 1
    frac_w = (pos_w - idx0.to(pos_w.dtype)).clamp(0, 1)
    v0 = torch.gather(f_grid, 1, idx0)
    v1 = torch.gather(f_grid, 1, idx1)
    return (v0 + frac_w * (v1 - v0)).clamp(0.0, 1.0)


def _companding_assign(f: torch.Tensor, kc: int) -> torch.Tensor:
    """Uniform binning in the warped domain -- round_uniform(F(x))."""
    idx = (f * kc).floor()
    return idx.clamp_(0, kc - 1).long()


def _wanda_alive(rows: torch.Tensor, act_norm: torch.Tensor, t_lo: float,
                 t_hi: float, t_max: float) -> torch.Tensor:
    """Score-based prune mask: keep the highest-scoring weights per row.

    score = |w_ij| * ||X_j|| (Wanda). (t_lo, t_hi) are not a magnitude band
    here -- there is no natural "sign" to a non-negative score -- they are
    reinterpreted as a sparsity FRACTION: each ranges over [0, t_max] in
    magnitude, same as sigma/raw, so (|t_lo| + t_hi) / (2*t_max) always lands
    in [0, 1] regardless of t_max, and needs no separate genome or config
    field. frac=0 prunes nothing; frac=1 prunes the whole row.

    The cut is PER ROW (per output channel), matching how sigma/raw already
    scale per row -- the fraction pruned is the same on every row, but which
    specific weights survive depends on that row's own score distribution.
    """
    frac = (abs(t_lo) + t_hi) / (2.0 * max(t_max, 1e-12))
    frac = min(max(frac, 0.0), 1.0)
    score = rows.abs() * act_norm.to(rows.dtype).reshape(1, -1)
    if frac <= 0.0:
        return torch.ones_like(rows, dtype=torch.bool)
    if frac >= 1.0:
        return torch.zeros_like(rows, dtype=torch.bool)
    ordered, _ = torch.sort(score, dim=1)
    n = score.shape[1]
    rank = min(int(frac * n), n - 1)
    thresh = ordered[:, rank : rank + 1]
    return score >= thresh


def compress_layer(
    rows: torch.Tensor,
    row_scale: torch.Tensor,
    k: int,
    t_lo: float,
    t_hi: float,
    quant_cfg,
    prune_cfg,
    name: str = "",
    alpha: float | None = None,
    gamma: float | None = None,
    u=None,
    z=None,
    force_zero: bool = False,
    reassign: bool = False,
    act_norm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LayerQuantStats]:
    """Compress one [out, in] weight matrix. Returns (reconstruction, stats)."""
    if k < 2:
        raise ValueError("K must be at least 2")
    rows = rows.to(torch.float32)

    # -- 1. pruning band, in units of the per-output-row weight scale --------
    pruning_on = prune_cfg.enabled and (t_lo < 0.0 or t_hi > 0.0)
    if pruning_on:
        if prune_cfg.mode == "sigma":
            lo_abs, hi_abs = t_lo * row_scale, t_hi * row_scale
            alive_rows = (rows <= lo_abs) | (rows >= hi_abs)
        elif prune_cfg.mode == "raw":
            lo_abs = torch.full_like(row_scale, t_lo)
            hi_abs = torch.full_like(row_scale, t_hi)
            alive_rows = (rows <= lo_abs) | (rows >= hi_abs)
        elif prune_cfg.mode == "wanda":
            if act_norm is None:
                raise ValueError(
                    "prune.mode == 'wanda' requires act_norm -- call "
                    "Compressor.calibrate_wanda(windows) before the search")
            alive_rows = _wanda_alive(rows, act_norm, t_lo, t_hi, prune_cfg.t_max)
        else:
            raise ValueError(f"unknown prune mode: {prune_cfg.mode}")
    else:
        alive_rows = torch.ones_like(rows, dtype=torch.bool)

    # -- 2. group, then bin the survivors -----------------------------------
    gran = layer_granularity(name, quant_cfg)
    w = _reshape_groups(rows, gran, quant_cfg.group_size)
    alive = _reshape_groups(alive_rows, gran, quant_cfg.group_size)

    # Reserve one symbol for zero when pruning is active, so the index width
    # stays exactly ceil(log2(K)) no matter how much gets pruned.
    kc = k - 1 if pruning_on else k
    if kc < 1:
        raise ValueError("K=2 is the minimum when pruning is enabled")

    if quant_cfg.binning == "uniform":
        lo, step = _uniform_range(w, alive, kc)
        idx = _uniform_assign(w, lo, step, kc)
        cent, cnts = _centroids(w, idx, alive, kc)
    elif quant_cfg.binning == "quantile":
        edges = _quantile_edges(w, alive, kc)
        idx = _assign(w, edges)
        cent, cnts = _centroids(w, idx, alive, kc)
    elif quant_cfg.binning == "kmeans":
        lo, step = _uniform_range(w, alive, kc)
        idx = _uniform_assign(w, lo, step, kc)
        offsets = torch.arange(kc, device=w.device, dtype=w.dtype).view(1, -1)
        init = lo + (offsets + 0.5) * step  # bin midpoints
        idx, cent, cnts = _lloyd(w, alive, kc, idx, init, quant_cfg.kmeans_iters)
    elif quant_cfg.binning == "companding":
        if alpha is None or gamma is None or u is None:
            raise ValueError("companding binning requires alpha, gamma and u")
        u_t = torch.as_tensor(u, dtype=torch.float32, device=w.device)
        f = _companding_forward(
            w, alive, float(alpha), float(gamma), u_t,
            grid=getattr(quant_cfg, "companding_grid", 256),
            residual_type=getattr(quant_cfg, "companding_residual_type", "linear"),
            degree=getattr(quant_cfg, "companding_ispline_degree", 3))
        idx = _companding_assign(f, kc)
        cent, cnts = _centroids(w, idx, alive, kc)
        if reassign:
            iters = getattr(quant_cfg, "companding_reassign_iters", 3)
            idx, cent, cnts = _lloyd(w, alive, kc, idx, cent, iters)
        if force_zero and kc > 0:
            cent = cent.clone()
            flat = cent.abs().argmin(dim=1)
            cent[torch.arange(cent.shape[0], device=cent.device), flat] = 0.0
    elif quant_cfg.binning == "widths":
        if z is None:
            raise ValueError("widths binning requires z")
        z_t = torch.as_tensor(z, dtype=torch.float32, device=w.device)
        edges = _widths_edges(w, alive, kc, z_t)
        idx = _assign(w, edges)
        cent, cnts = _centroids(w, idx, alive, kc)
    else:
        raise ValueError(f"unknown binning: {quant_cfg.binning}")

    # -- 3. reconstruct ------------------------------------------------------
    recon = torch.gather(cent, 1, idx)
    if pruning_on:
        recon = torch.where(alive, recon, torch.zeros_like(recon))

    # -- 4. symbol histogram for the entropy coder ---------------------------
    # Symbol 0 is the reserved zero codeword; survivors occupy 1..kc.
    #
    # Accumulate on the CPU in float64: per-group counts are small, but their
    # sum over a 45M-weight layer exceeds float32's exact-integer range (2^24),
    # and MPS has no float64 at all.
    counts = torch.zeros(k, dtype=torch.float64)
    alive_cnts = cnts.detach().cpu().to(torch.float64).sum(dim=0)
    n_alive = int(alive.sum().item())
    if pruning_on:
        counts[0] = float(w.numel() - n_alive)
        counts[1 : 1 + kc] = alive_cnts
        k_used = (cnts > 0).sum(dim=1).float().mean().item() + 1.0
    else:
        counts[:kc] = alive_cnts
        k_used = (cnts > 0).sum(dim=1).float().mean().item()

    denom = rows.pow(2).sum().clamp_min(1e-12)
    stats = LayerQuantStats(
        name=name,
        n_weights=rows.numel(),
        n_groups=w.shape[0],
        k_nominal=k,
        k_centroids=kc + (1 if pruning_on else 0),
        symbol_counts=counts,
        k_used_mean=float(k_used),
        sparsity=1.0 - n_alive / max(alive.numel(), 1),
        mse=float(((recon - w).pow(2).sum() / denom).item()),
    )

    out = recon.reshape(rows.shape)
    return out, stats


class LayerPrecompute:
    """Pruning-independent per-layer state, computed once and reused forever.

    When pruning is disabled the (layer, K) result is fully deterministic, so
    reconstructions are cached outright and a fitness evaluation degenerates to
    a handful of memcpys plus the forward passes.
    """

    def __init__(self, enabled: bool = True, max_entries: int = 512):
        self.enabled = enabled
        self.max_entries = max_entries
        self._cache: dict[tuple[str, int], tuple[torch.Tensor, LayerQuantStats]] = {}
        self.hits = 0
        self.misses = 0

    def key(self, name: str, k: int, t_lo: float, t_hi: float, extra=None):
        if t_lo < 0.0 or t_hi > 0.0:
            return None  # pruning-dependent results are not cached
        return (name, k) if extra is None else (name, k, extra)

    def get(self, key):
        if key is None or not self.enabled:
            return None
        hit = self._cache.get(key)
        if hit is None:
            self.misses += 1
        else:
            self.hits += 1
        return hit

    def put(self, key, value):
        if key is None or not self.enabled or len(self._cache) >= self.max_entries:
            return
        self._cache[key] = value

    def clear(self):
        self._cache.clear()
