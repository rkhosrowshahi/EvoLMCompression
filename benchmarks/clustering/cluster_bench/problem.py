"""The pymoo problem: decode a genome, quantize, score the partition.

Objectives are named by config and resolved against `metrics.MINIMIZED`, so a
run can be re-pointed at (mse, entropy_bits) -- the parent project's rate axis
-- without touching this file. The default pair is

  f1 = mse             distortion, the thing k-means minimizes
  f2 = davies_bouldin  cluster validity, which k-means does not optimize at all

and that asymmetry is the point. k-means is the SSE optimum, so on f1 alone the
comparison is decided before it starts (in 1-D, provably). The interesting
question is whether a search that can trade a little distortion for a better
separated partition finds points k-means cannot reach at any K.

Everything else `metrics.evaluate` computes is carried along per candidate and
written to the results file, so the front can be re-plotted on axes that were
not optimized.
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .companding import companding_quantize_1d, companding_quantize_md
from .genome import Genome
from .metrics import MINIMIZED, evaluate


class ClusteringProblem(ElementwiseProblem):
    def __init__(self, dataset, genome: Genome, objectives=("mse", "davies_bouldin"),
                 silhouette_max_n: int = 2000, seed: int = 0,
                 min_k_eff: int | None = None, max_k_eff: int | None = None,
                 min_cluster_size: int = 1):
        self.dataset = dataset
        self.genome = genome
        self.objectives = tuple(objectives)
        for name in self.objectives:
            if name not in MINIMIZED:
                raise ValueError(
                    f"objective {name!r} is not minimizable; pick from {MINIMIZED}")
        self.silhouette_max_n = silhouette_max_n
        self.seed = seed
        # Silhouette is the only O(n^2) measure in the set. Computing it on
        # every candidate when nothing is optimizing it would make it the whole
        # cost of the search; the archive rows that end up in a table are
        # re-scored with it afterwards, so nothing is lost.
        self.search_silhouette = "neg_silhouette" in self.objectives
        # Bounds on the number of occupied clusters, as pymoo constraints rather
        # than penalties: a degenerate partition is feasible for the quantizer
        # and useless as a clustering, and stating that as g(x) <= 0 keeps it
        # out of the front without inventing a magic objective value that would
        # distort the hypervolume.
        #
        # The FLOOR rules out the collapsed 1-cluster answer. The CEILING is
        # what stops the opposite degeneracy, and it is not optional in
        # multi-D: a product quantizer over d axes has prod(K_d) cells, so in
        # 32 dimensions even two levels per axis can put every point in its own
        # cell. That scores MSE = 0 and -- because a singleton cluster has zero
        # spread, and Davies-Bouldin is a ratio of spreads -- DB = 0 as well.
        # A perfect score on both axes, dominating the entire front, meaning
        # nothing. Observed, not hypothesised: an early dim32 run reported
        # exactly [0.0, 0.0] at generation 1.
        self.min_k_eff = min_k_eff
        self.max_k_eff = max_k_eff
        # Off by default. See config.SearchCfg.min_cluster_size for why the
        # DB-gaming solutions are left in the front rather than constrained out.
        self.min_cluster_size = int(min_cluster_size)
        self.history: list[dict] = []
        self.n_evals = 0
        self._t0 = time.perf_counter()

        # NOT `n_constr`: pymoo's Problem exposes that as a read-only property.
        self._n_g = ((min_k_eff is not None) + (max_k_eff is not None)
                     + (self.min_cluster_size > 1))
        xl, xu = genome.bounds()
        super().__init__(n_var=genome.n_var, n_obj=len(self.objectives),
                         n_ieq_constr=self._n_g, xl=xl, xu=xu)

    def partition(self, x: np.ndarray):
        """Genome -> (labels, centroids). Shared with the reporting code."""
        s = self.genome.spec
        st = self.genome.decode(x)
        if self.dataset.kind == "1d" or self.dataset.d == 1:
            return companding_quantize_1d(
                self.dataset.x[:, 0], int(st.ks[0]), float(st.alphas[0]),
                float(st.gammas[0]), st.us[0], grid=s.grid,
                residual_type=s.residual_type, degree=s.ispline_degree,
                lloyd_iters=s.lloyd_iters)
        return companding_quantize_md(
            self.dataset.x, st.ks, st.alphas, st.gammas, st.us, grid=s.grid,
            residual_type=s.residual_type, degree=s.ispline_degree)

    def score(self, x: np.ndarray, with_silhouette: bool = True,
              with_truth: bool = True) -> dict:
        """Score one genome. `with_truth` adds the ARI against the generating
        labels -- reporting only, and never reachable from `_evaluate`, so the
        search cannot see the answer it is supposed to be finding."""
        labels, cent = self.partition(x)
        return evaluate(self.dataset.x, labels, cent, self.silhouette_max_n,
                        self.seed, with_silhouette,
                        self.dataset.y_true if with_truth else None)

    def _evaluate(self, x, out, *args, **kwargs):
        t0 = time.perf_counter()
        m = self.score(x, self.search_silhouette, with_truth=False)
        self.n_evals += 1
        out["F"] = [float(m[name]) for name in self.objectives]
        if self._n_g:
            g = []
            if self.min_k_eff is not None:
                g.append(float(self.min_k_eff - m["k_eff"]))
            if self.max_k_eff is not None:
                g.append(float(m["k_eff"] - self.max_k_eff))
            if self.min_cluster_size > 1:
                g.append(float(self.min_cluster_size - m["min_cluster_size"]))
            out["G"] = g
        self.history.append({
            "eval": self.n_evals,
            "t": round(time.perf_counter() - self._t0, 3),
            "eval_seconds": round(time.perf_counter() - t0, 4),
            # The genome itself, so an archive row picked out for a table can
            # be re-scored exactly rather than approximately reconstructed.
            "_x": np.asarray(x, dtype=np.float64).copy(),
            **{k: (int(v) if k == "k_eff" else float(v)) for k, v in m.items()},
        })
