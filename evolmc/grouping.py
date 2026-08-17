"""Decision-variable layout.

Per-layer variables blow up fast: Llama-2-7B has 224 projection matrices, so a
fully per-layer encoding with pruning bands is 224 + 448 = 672 variables --
well outside the range where NSGA-II's non-dominated sorting still gives useful
selection pressure. This module lets you dial that down by tying layers
together, which is also the ablation axis the paper needs
(global vs. type vs. block vs. per-layer).

The genome is always a real vector in [0, 1]^D. `Genome.decode` maps it to
concrete (K, t_lo, t_hi) per layer. Keeping the encoding continuous and
normalized means plain NSGA-II, U-NSGA-III and MOEA/D all work unchanged, and
a surrogate can be dropped in later without touching the operators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import TargetLayer


def _group_key(layer: TargetLayer, scheme: str) -> str:
    if scheme == "global":
        return "all"
    if scheme == "type":
        return layer.proj_type
    if scheme == "block":
        return f"b{layer.block}"
    if scheme == "block_type":
        return f"b{layer.block}.{layer.proj_type}"
    raise ValueError(f"unknown grouping scheme: {scheme}")


def build_groups(layers: list[TargetLayer], scheme: str) -> tuple[list[str], dict[str, int]]:
    """Return (ordered group names, layer-name -> group index)."""
    order: list[str] = []
    seen: dict[str, int] = {}
    assign: dict[str, int] = {}
    for layer in layers:
        key = _group_key(layer, scheme)
        if key not in seen:
            seen[key] = len(order)
            order.append(key)
        assign[layer.name] = seen[key]
    return order, assign


def layer_granularity(name: str, quant_cfg) -> str:
    """Effective codebook grouping for one layer.

    `quant.per_channel_patterns` overrides the global `quant.granularity` for
    matching names, so a per_tensor run can still give `wte` / `lm_head` a
    codebook per token row.
    """
    for p in getattr(quant_cfg, "per_channel_patterns", ()) or ():
        if p and p in name:
            return "per_channel"
    return quant_cfg.granularity


def max_k_for_layer(layer: TargetLayer, quant_cfg) -> int:
    """Largest codebook size that layer can actually fill.

    A codebook cannot have more entries than there are values in the group it
    serves, so the ceiling is the number of weights per codebook group -- and
    that is a property of the layer and the granularity, not a global setting.
    Asking for more just stores empty centroids that are still paid for.
    """
    gran = layer_granularity(layer.name, quant_cfg)
    if gran == "per_tensor":
        return int(layer.n_weights)
    if gran == "per_channel":
        return int(layer.in_features)
    return int(quant_cfg.group_size)


@dataclass
class LayerSetting:
    k: int
    t_lo: float  # lower edge of the pruning band (<= 0)
    t_hi: float  # upper edge of the pruning band (>= 0)
    # Companding / warp quantization (binning == "companding" only). None for
    # every other binning mode, which never reads these fields.
    alpha: float | None = None       # clip threshold, in units of group std
    gamma: float | None = None       # backbone density exponent
    u: np.ndarray | None = None      # raw residual-warp genes, length M
    force_zero: bool = False         # snap the nearest centroid to exactly 0
    reassign: bool = False           # Lloyd-reassign the warp bins after binning


class Genome:
    """Maps a normalized real vector to per-layer compression settings."""

    def __init__(self, layers, quant_cfg, prune_cfg, var_cfg):
        self.layers = layers
        self.quant = quant_cfg
        self.prune = prune_cfg
        self.k_encoding = getattr(quant_cfg, "k_encoding", "choices")
        if self.k_encoding == "integer":
            self.k_min, self.k_max = int(quant_cfg.k_min), int(quant_cfg.k_max)
            if self.k_min < 2 or self.k_max <= self.k_min:
                raise ValueError(f"need 2 <= k_min < k_max, got "
                                 f"{self.k_min}, {self.k_max}")
            # Reference points and warm-start seeds stay on the power-of-two
            # ladder inside the range: they are the interpretable fixed-bit
            # configurations the front has to beat.
            ladder = [1 << b for b in range(1, self.k_max.bit_length() + 1)
                      if self.k_min <= (1 << b) <= self.k_max]
            self.k_choices = tuple(sorted(set(ladder or [self.k_min, self.k_max])))
        else:
            self.k_choices = tuple(sorted(set(int(k) for k in quant_cfg.k_choices)))
            self.k_min, self.k_max = self.k_choices[0], self.k_choices[-1]

        self.k_groups, self.k_assign = build_groups(layers, var_cfg.k_grouping)
        self.n_k = len(self.k_groups)

        # A group's ceiling is the MINIMUM over its 2-D member layers: one K
        # has to be valid for every matrix the group covers. With `block`
        # grouping a block's narrowest layer sets the limit for all four.
        # 1-D norms/biases are clamped per layer in decode instead -- a 768-wide
        # LayerNorm must not drag a shared K down from 8192 to 768.
        caps = np.full(self.n_k, self.k_max, dtype=np.int64)
        for layer in layers:
            if layer.is_vector:
                continue
            g = self.k_assign[layer.name]
            caps[g] = min(caps[g], max_k_for_layer(layer, quant_cfg))
        self.k_max_group = np.maximum(caps, self.k_min)
        self.capped = bool((self.k_max_group < self.k_max).any())

        if prune_cfg.enabled:
            self.p_groups, self.p_assign = build_groups(layers, var_cfg.prune_grouping)
            self.n_p = len(self.p_groups)
        else:
            self.p_groups, self.p_assign, self.n_p = [], {}, 0

        # Companding warp shares its groups with K: a warp shape and a
        # codebook size jointly define one quantizer, so there is no
        # separate `warp_grouping` dial. Per group: alpha, gamma, M residual
        # slopes, and -- only when companding_flag_genes is on -- the two
        # boolean genes force_zero and reassign. See the note in config.py:
        # the search drives both to False, so they are off by default.
        self.companding = (getattr(quant_cfg, "binning", "uniform") == "companding")
        if self.companding:
            self.warp_m = int(getattr(quant_cfg, "companding_residual_genes", 6))
            self.warp_flags = bool(getattr(quant_cfg, "companding_flag_genes", False))
            self.warp_dim = 2 + self.warp_m + (2 if self.warp_flags else 0)
            self.n_w = self.n_k
            self.alpha_min = float(getattr(quant_cfg, "companding_alpha_min", 2.0))
            self.alpha_max = float(getattr(quant_cfg, "companding_alpha_max", 6.0))
            self.gamma_min = float(getattr(quant_cfg, "companding_gamma_min", 0.0))
            self.gamma_max = float(getattr(quant_cfg, "companding_gamma_max", 1.0))
        else:
            self.warp_m = self.warp_dim = self.n_w = 0

        # Layout: [K vars ...][t_lo vars ...][t_hi vars ...][warp vars ...]
        self.n_var = self.n_k + 2 * self.n_p + self.n_w * self.warp_dim

    # -- encoding helpers -------------------------------------------------

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.n_var), np.ones(self.n_var)

    def decode(self, x: np.ndarray) -> dict[str, LayerSetting]:
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        k_per_group = self._decode_k(x[: self.n_k])

        if self.n_p:
            lo = x[self.n_k : self.n_k + self.n_p] * self.prune.t_max
            hi = x[self.n_k + self.n_p : self.n_k + 2 * self.n_p] * self.prune.t_max
        else:
            lo = hi = np.zeros(0)

        warp = None
        if self.companding:
            off = self.n_k + 2 * self.n_p
            wx = x[off : off + self.n_w * self.warp_dim].reshape(self.n_w, self.warp_dim)
            alpha = self.alpha_min + wx[:, 0] * (self.alpha_max - self.alpha_min)
            gamma = self.gamma_min + wx[:, 1] * (self.gamma_max - self.gamma_min)
            u = wx[:, 2 : 2 + self.warp_m]
            if self.warp_flags:
                force_zero = wx[:, 2 + self.warp_m] >= 0.5
                reassign = wx[:, 2 + self.warp_m + 1] >= 0.5
            else:
                force_zero = reassign = np.zeros(self.n_w, dtype=bool)
            warp = (alpha, gamma, u, force_zero, reassign)

        out: dict[str, LayerSetting] = {}
        for layer in self.layers:
            g = self.k_assign[layer.name]
            k = k_per_group[g]
            cap = max_k_for_layer(layer, self.quant)
            if k > cap:
                k = max(int(cap), self.k_min)
            if self.n_p:
                pg = self.p_assign[layer.name]
                t_lo, t_hi = -float(lo[pg]), float(hi[pg])
            else:
                t_lo = t_hi = 0.0
            if warp is None:
                out[layer.name] = LayerSetting(k=k, t_lo=t_lo, t_hi=t_hi)
            else:
                alpha, gamma, u, force_zero, reassign = warp
                out[layer.name] = LayerSetting(
                    k=k, t_lo=t_lo, t_hi=t_hi,
                    alpha=float(alpha[g]), gamma=float(gamma[g]),
                    u=u[g].copy(), force_zero=bool(force_zero[g]),
                    reassign=bool(reassign[g]))
        return out

    def group_choices(self, g: int) -> tuple[int, ...]:
        """The k_choices ladder truncated to what group `g` can fill."""
        ks = tuple(k for k in self.k_choices if k <= self.k_max_group[g])
        return ks or (self.k_min,)

    def _decode_k(self, u: np.ndarray) -> list[int]:
        """Map [0,1] genes to codebook sizes, one range per group.

        Each gene spans its own group's [k_min, k_max_group] rather than a
        shared global range, so no part of the gene maps to a K the group
        cannot fill. The search space is therefore prod_g (k_max_g - k_min + 1),
        not k_max ** n_groups.
        """
        if self.k_encoding == "integer":
            # Log-spaced: the cost axis is index width = log2(K), so a linear
            # map would spend half the gene range above K/2, where quality has
            # already plateaued, and leave almost no resolution at small K.
            lo = math.log2(self.k_min)
            hi = np.log2(self.k_max_group[: len(u)].astype(float))
            k = np.rint(np.exp2(lo + u * (hi - lo))).astype(int)
            return np.clip(k, self.k_min, self.k_max_group[: len(u)]).tolist()
        out = []
        for g, ug in enumerate(u):
            ks = self.group_choices(g)
            out.append(ks[min(int(ug * len(ks)), len(ks) - 1)])
        return out

    def _encode_k(self, k: int, g: int = 0) -> float:
        """Gene value that decodes to `k` in group `g` (clamped to its ceiling)."""
        cap = int(self.k_max_group[g])
        k = int(min(max(k, self.k_min), cap))
        if self.k_encoding == "integer":
            lo, hi = math.log2(self.k_min), math.log2(cap)
            if hi <= lo:
                return 0.0
            return float(np.clip((math.log2(k) - lo) / (hi - lo), 0.0, 1.0))
        ks = self.group_choices(g)
        if k not in ks:
            k = max(c for c in ks if c <= k)
        # Land in the middle of the bin so mutation doesn't flip K immediately.
        return (ks.index(k) + 0.5) / len(ks)

    def encode_uniform(self, k: int, t: float = 0.0) -> np.ndarray:
        """A genome with the same K everywhere -- used for warm starts and for
        the fixed-bit baselines the paper compares against."""
        if self.k_encoding == "integer" and not self.k_min <= k <= self.k_max:
            raise ValueError(f"K={k} outside [{self.k_min}, {self.k_max}]")
        x = np.zeros(self.n_var)
        # Per group, because ceilings differ: a group that cannot reach `k`
        # sits at its own maximum instead.
        x[: self.n_k] = [self._encode_k(int(k), g) for g in range(self.n_k)]
        if self.n_p:
            frac = 0.0 if self.prune.t_max <= 0 else t / self.prune.t_max
            x[self.n_k : self.n_k + 2 * self.n_p] = np.clip(frac, 0.0, 1.0)
        return x

    def seed_population(self, pop_size, rng, mode: str = "logspace") -> np.ndarray:
        """Build generation 0.

        `logspace` gives every individual a single K applied uniformly to all
        groups, with those K spread evenly in LOG space across the range --
        even, because the cost axis is index width log2(K), so a linear spread
        would pile most of the population into the high-K end where quality has
        already plateaued.

        Generation 0 then sits exactly on the fixed-bit baseline curve, spanning
        it end to end. That makes the first frame interpretable on its own and
        means any later movement is unambiguously the search's doing rather than
        a lucky draw. Diversity in the *relative* K between groups is created by
        SBX crossover and PM mutation from there; it does not need to be seeded.

        Groups whose ceiling is below a target K sit at their own maximum, so a
        `logspace` individual is uniform wherever uniformity is reachable.
        """
        if mode == "random":
            return rng.random((pop_size, self.n_var))

        if mode == "ladder":
            seeds = [self.encode_uniform(k) for k in self.k_choices]
            if self.n_p:
                mid = self.k_choices[len(self.k_choices) // 2]
                seeds += [self.encode_uniform(mid, t) for t in
                          np.linspace(0.0, self.prune.t_max, 4)[1:]]
            seeds = seeds[:pop_size]
            n_rand = max(0, pop_size - len(seeds))
            if n_rand:
                return np.vstack([np.array(seeds).reshape(len(seeds), self.n_var),
                                  rng.random((n_rand, self.n_var))])
            return np.array(seeds)[:pop_size]

        if mode != "logspace":
            raise ValueError(f"unknown init mode: {mode}")

        # Even in log2(K), i.e. even in index bits.
        ks = np.rint(np.exp2(np.linspace(math.log2(self.k_min),
                                         math.log2(self.k_max),
                                         pop_size))).astype(int)
        ks = np.clip(ks, self.k_min, self.k_max)
        pop = np.array([self.encode_uniform(int(k)) for k in ks])
        if self.n_p:
            # Sweep the pruning band across the population too, so generation 0
            # spans both objectives' extremes rather than only the K axis.
            ts = np.linspace(0.0, self.prune.t_max, pop_size)
            for i, t in enumerate(ts):
                frac = 0.0 if self.prune.t_max <= 0 else t / self.prune.t_max
                pop[i, self.n_k : self.n_k + 2 * self.n_p] = np.clip(frac, 0.0, 1.0)
        return pop

    # -- reporting --------------------------------------------------------

    def describe(self) -> str:
        lines = [
            f"decision variables : {self.n_var}",
            f"  K groups         : {self.n_k} ({', '.join(self.k_groups[:8])}"
            + (" ..." if self.n_k > 8 else "") + ")",
        ]
        if self.n_p:
            lines.append(f"  pruning groups   : {self.n_p} x 2 vars")
        else:
            lines.append("  pruning          : disabled")
        if self.companding:
            lines.append(f"  companding warp  : {self.n_w} groups x "
                         f"{self.warp_dim} vars (alpha, gamma, "
                         f"{self.warp_m} residual"
                         + (", 2 flags)" if self.warp_flags else ", no flag genes)"))
        if self.k_encoding == "integer":
            lines.append(f"  K encoding       : integer, log-spaced in "
                         f"[{self.k_min}, {self.k_max}] "
                         f"({math.log2(self.k_min):.0f}-"
                         f"{math.log2(self.k_max):.0f} index bits)")
            lines.append(f"  K references     : {self.k_choices}")
        else:
            lines.append(f"  K choices        : {self.k_choices} "
                         f"({math.log2(min(self.k_choices)):.0f}-"
                         f"{math.log2(max(self.k_choices)):.0f} index bits)")
        return "\n".join(lines)
