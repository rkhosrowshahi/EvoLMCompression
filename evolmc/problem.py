"""The pymoo problem definition.

Objectives (both minimised):
  f1 = proxy perplexity on held-in calibration windows
  f2 = bits per weight

bpw is used rather than -CR because it is linear in the thing the search
actually controls and because every baseline in the literature is quoted in
bits. CR is derived and logged alongside.

Optional constraint: g1 = bpw - max_bpw <= 0.
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .evaluate import proxy_fitness


class CompressionProblem(ElementwiseProblem):
    def __init__(self, compressor, windows, cfg, run=None):
        self.compressor = compressor
        self.windows = windows
        self.cfg = cfg
        self.run = run
        self.history: list[dict] = []
        self.n_baseline_evals = 0
        self._t0 = time.perf_counter()

        xl, xu = compressor.genome.bounds()
        super().__init__(
            n_var=compressor.genome.n_var,
            n_obj=2,
            n_ieq_constr=1 if cfg.search.max_bpw is not None else 0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        t0 = time.perf_counter()
        cand = self.compressor.apply(x)
        ppl = proxy_fitness(self.compressor.model, self.windows,
                            device=self.compressor.device)

        cost = cand.cost
        size = (cost.bpw_target if self.cfg.search.size_objective == "bpw_target"
                else cost.bpw_model)

        out["F"] = [ppl, size]
        if self.cfg.search.max_bpw is not None:
            out["G"] = [size - self.cfg.search.max_bpw]

        record = {
            "eval": self.compressor.n_evals,
            "t": round(time.perf_counter() - self._t0, 2),
            "eval_seconds": round(time.perf_counter() - t0, 3),
            "apply_seconds": round(cand.apply_seconds, 3),
            "ppl_proxy": ppl,
            "x": [round(float(v), 5) for v in np.asarray(x)],
            **{k: round(v, 5) for k, v in cost.summary().items()},
        }
        self.history.append(record)
        if self.run is not None:
            self.run.jsonl("evals", record)

    # The model is left holding the last candidate's weights; restore before
    # doing anything else with it.
    def restore(self):
        self.compressor.restore()
