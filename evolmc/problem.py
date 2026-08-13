"""The pymoo problem definition.

Which objectives are optimized is set by `search.objectives` and resolved
through `objectives.ObjectiveSet`; see that module for the registry and the
sign convention. The default list is

  f1 = proxy perplexity on held-in calibration windows   (minimized)
  f2 = bits per weight over the whole checkpoint         (minimized)

which is the original two-objective problem.

avg_bits is preferred over -CR for the size axis because it is linear in the thing
the search actually controls and because every baseline in the literature is
quoted in bits. CR is derived and logged alongside, and can be made an
objective in its own right -- but only usefully when it is drawn from a
*different* bit total than the avg_bits objective, since `cr_deploy` is exactly
`16 / avg_bits` and adds nothing next to it.

Optional constraint: g1 = avg_bits - max_avg_bits <= 0, always applied to the
`size_objective` measure regardless of what is being optimized.
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .evaluate import proxy_fitness
from .objectives import ObjectiveSet, canonical


class CompressionProblem(ElementwiseProblem):
    def __init__(self, compressor, windows, cfg, run=None, latency=None):
        self.compressor = compressor
        self.windows = windows
        self.cfg = cfg
        self.run = run
        # latency.LatencyProxy: coefficients fitted once on the target GPU and
        # frozen. Required only when `latency_proxy` is an objective or a
        # reported metric; None otherwise, and then `latency_proxy` is simply
        # not in the summary the objective set reads from.
        self.latency = latency
        self.history: list[dict] = []
        self.n_baseline_evals = 0
        self._t0 = time.perf_counter()
        self.objectives = ObjectiveSet(getattr(cfg.search, "objectives", None))
        # Reported per generation but never optimized; must not collide with an
        # objective, or the log would print the same column twice. Canonicalized
        # first, so a config naming a retired spelling still collides correctly
        # with the objective it duplicates rather than slipping through as an
        # extra column of the same numbers.
        self.report_metrics = tuple(
            m for m in map(canonical, getattr(cfg.search, "report_metrics", ()))
            if m not in self.objectives.names)

        xl, xu = compressor.genome.bounds()
        super().__init__(
            n_var=compressor.genome.n_var,
            n_obj=self.objectives.n_obj,
            n_ieq_constr=1 if cfg.search.max_avg_bits is not None else 0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        t0 = time.perf_counter()
        cand = self.compressor.apply(x)
        ppl = proxy_fitness(self.compressor.model, self.windows,
                            device=self.compressor.device)

        summary = cand.cost.summary()
        if self.latency is not None:
            # Predicted, not timed. Pure arithmetic over the per-layer bit
            # accounting against frozen coefficients, so it costs nothing per
            # candidate -- the only reason a latency axis is affordable across
            # 10,000 evaluations. See evolmc/latency.py for the model and for
            # which constants are measured and which are stated.
            summary["latency_proxy"] = self.latency.predict(cand.cost)
        values = self.objectives.values(ppl, summary)
        out["F"] = self.objectives.to_min(values)

        # Carried alongside F, not optimized. pymoo stores any extra key on the
        # individual and keeps it through selection, so the per-generation log
        # can report the front's sparsity without a second pass over genomes.
        for name in self.report_metrics:
            out[name] = float(summary[name])

        if self.cfg.search.max_avg_bits is not None:
            # The budget is a statement about deployed size, so it is checked
            # against size_objective whether or not that measure is being
            # optimized. Reading it off the objective vector instead would
            # silently constrain the wrong quantity when the front is drawn on
            # archival axes.
            size = summary[self.cfg.search.size_objective]
            out["G"] = [size - self.cfg.search.max_avg_bits]

        record = {
            "eval": self.compressor.n_evals,
            "t": round(time.perf_counter() - self._t0, 2),
            "eval_seconds": round(time.perf_counter() - t0, 3),
            "apply_seconds": round(cand.apply_seconds, 3),
            "ppl_proxy": ppl,
            "x": [round(float(v), 5) for v in np.asarray(x)],
            **{k: round(v, 5) for k, v in summary.items()},
        }
        self.history.append(record)
        if self.run is not None:
            self.run.jsonl("evals", record)

    # The model is left holding the last candidate's weights; restore before
    # doing anything else with it.
    def restore(self):
        self.compressor.restore()
