"""Decision-variable layout for the companding search.

The genome is a real vector in [0, 1]^D, exactly as in `evolmc/grouping.py`, and
for the same reason: keeping it continuous and normalized means NSGA-II,
U-NSGA-III and MOEA/D all run unchanged, and the SBX/PM operator parameters
mean the same thing on every gene.

1-D:   [k, alpha, gamma, u_1..u_M]                      D = 3 + M
multi: per dimension the same block, or -- with `share_warp` -- one shared
       warp and a per-dimension K:
         share_warp=False   [k_j, alpha_j, gamma_j, u_j1..u_jM] * d
         share_warp=True    [k_1..k_d, alpha, gamma, u_1..u_M]

`share_warp` matters more than it looks. Per-dimension warps on a 32-D problem
is 32*(3+M) variables, which is past the point where non-dominated sorting
still gives NSGA-II useful selection pressure -- the same wall the parent
project hits with per-layer encodings, and the same fix.

K is mapped log-uniformly. Quantizer quality moves with log K (a bit of rate),
not with K, so a uniform mapping would spend most of the genome's resolution
distinguishing K=200 from K=210.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenomeSpec:
    d: int = 1                       # data dimensionality
    k_min: int = 2
    k_max: int = 256
    alpha_min: float = 0.5
    alpha_max: float = 12.0
    gamma_min: float = 0.0
    gamma_max: float = 1.5
    residual_genes: int = 6          # M
    u_lo: float = -3.0
    u_hi: float = 3.0
    share_warp: bool = False
    residual_type: str = "linear"    # "linear" | "ispline"
    ispline_degree: int = 3
    grid: int = 256
    lloyd_iters: int = 0             # >0 turns the compander into a k-means seed

    def __post_init__(self):
        if self.k_min < 2:
            raise ValueError("k_min must be >= 2")
        if self.k_max < self.k_min:
            raise ValueError("k_max must be >= k_min")
        if self.residual_type == "ispline" and self.residual_genes < self.ispline_degree:
            raise ValueError(
                f"residual_type=ispline with degree={self.ispline_degree} needs "
                f"residual_genes >= {self.ispline_degree}")


@dataclass
class Setting:
    """One decoded candidate: what the quantizer is actually asked to do."""
    ks: np.ndarray                   # [d] int
    alphas: np.ndarray               # [d] float
    gammas: np.ndarray               # [d] float
    us: list = field(default_factory=list)   # d arrays of length M


class Genome:
    def __init__(self, spec: GenomeSpec):
        self.spec = spec
        s = spec
        self.per_dim = 3 + s.residual_genes
        if s.d == 1 or not s.share_warp:
            self.n_var = s.d * self.per_dim
        else:
            self.n_var = s.d + 2 + s.residual_genes

    def bounds(self):
        return np.zeros(self.n_var), np.ones(self.n_var)

    # -- scalar gene maps ---------------------------------------------------

    def _k(self, g: float) -> int:
        s = self.spec
        lo, hi = np.log(s.k_min), np.log(s.k_max)
        return int(np.clip(round(float(np.exp(lo + g * (hi - lo)))),
                           s.k_min, s.k_max))

    def _alpha(self, g: float) -> float:
        s = self.spec
        return float(s.alpha_min + g * (s.alpha_max - s.alpha_min))

    def _gamma(self, g: float) -> float:
        s = self.spec
        return float(s.gamma_min + g * (s.gamma_max - s.gamma_min))

    def _u(self, g: np.ndarray) -> np.ndarray:
        s = self.spec
        return s.u_lo + np.asarray(g, dtype=np.float64) * (s.u_hi - s.u_lo)

    # -- decode -------------------------------------------------------------

    def decode(self, x: np.ndarray) -> Setting:
        s = self.spec
        x = np.clip(np.asarray(x, dtype=np.float64).ravel(), 0.0, 1.0)
        m = s.residual_genes
        if s.d == 1 or not s.share_warp:
            ks, alphas, gammas, us = [], [], [], []
            for j in range(s.d):
                blk = x[j * self.per_dim:(j + 1) * self.per_dim]
                ks.append(self._k(blk[0]))
                alphas.append(self._alpha(blk[1]))
                gammas.append(self._gamma(blk[2]))
                us.append(self._u(blk[3:3 + m]))
        else:
            ks = [self._k(g) for g in x[:s.d]]
            alpha = self._alpha(x[s.d])
            gamma = self._gamma(x[s.d + 1])
            u = self._u(x[s.d + 2:s.d + 2 + m])
            alphas = [alpha] * s.d
            gammas = [gamma] * s.d
            us = [u] * s.d
        return Setting(np.array(ks, dtype=np.int64),
                       np.array(alphas), np.array(gammas), us)

    def describe(self, x: np.ndarray) -> dict:
        """JSON-safe view of a decoded genome, for the results file."""
        st = self.decode(x)
        return {
            "k": [int(v) for v in st.ks],
            "alpha": [round(float(v), 4) for v in st.alphas],
            "gamma": [round(float(v), 4) for v in st.gammas],
            "u": [[round(float(v), 4) for v in u] for u in st.us],
        }
