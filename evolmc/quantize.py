"""Codebook construction and weight replacement.

Pipeline for one weight matrix, given (K, t_lo, t_hi):

  1. prune   -- zero every weight inside the band [t_lo, t_hi]
  2. bin     -- partition the *surviving* weights into K-1 bins
  3. centre  -- each codeword is the mean of the weights in its bin
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

from dataclasses import dataclass

import torch

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


def compress_layer(
    rows: torch.Tensor,
    row_scale: torch.Tensor,
    k: int,
    t_lo: float,
    t_hi: float,
    quant_cfg,
    prune_cfg,
    name: str = "",
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
        else:
            lo_abs = torch.full_like(row_scale, t_lo)
            hi_abs = torch.full_like(row_scale, t_hi)
        alive_rows = (rows <= lo_abs) | (rows >= hi_abs)
    else:
        alive_rows = torch.ones_like(rows, dtype=torch.bool)

    # -- 2. group, then bin the survivors -----------------------------------
    w = _reshape_groups(rows, quant_cfg.granularity, quant_cfg.group_size)
    alive = _reshape_groups(alive_rows, quant_cfg.granularity, quant_cfg.group_size)

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

    def key(self, name: str, k: int, t_lo: float, t_hi: float):
        if t_lo < 0.0 or t_hi > 0.0:
            return None  # pruning-dependent results are not cached
        return (name, k)

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
