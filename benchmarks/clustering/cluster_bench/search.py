"""NSGA-II over the companding genome.

Thin by design: pymoo does the work, this file fixes the operator settings,
records one snapshot of the population per generation, and hands back the final
front. The per-generation snapshots exist so convergence can be re-scored later
against a reference point that is only known once the baselines have run --
computing hypervolume during the search would pin it to a nadir chosen before
the comparison arm was even fitted.
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class Snapshots(Callback):
    """One (X, F) snapshot per generation, kept in memory."""

    def __init__(self, verbose=True, every=10):
        super().__init__()
        self.gens: list[dict] = []
        self.verbose = verbose
        self.every = every
        self._t0 = time.perf_counter()

    def notify(self, algorithm):
        f = np.asarray(algorithm.pop.get("F"), dtype=np.float64)
        self.gens.append({
            "gen": int(algorithm.n_gen),
            "n_eval": int(algorithm.evaluator.n_eval),
            "t": round(time.perf_counter() - self._t0, 2),
            "F": f.copy(),
            "X": np.asarray(algorithm.pop.get("X"), dtype=np.float64).copy(),
        })
        if self.verbose and (algorithm.n_gen == 1 or algorithm.n_gen % self.every == 0):
            best = f.min(axis=0)
            print(f"    gen {algorithm.n_gen:>4}  evals {algorithm.evaluator.n_eval:>6}  "
                  f"best-per-objective {np.round(best, 6)}  "
                  f"({time.perf_counter() - self._t0:.0f}s)")


def run_nsga2(problem, pop_size: int = 100, n_gen: int = 100, seed: int = 0,
              eta_cx: float = 15.0, eta_mut: float = 20.0,
              p_cx: float = 0.9, verbose: bool = True, log_every: int = 10):
    """Returns a dict with the final front, the archive and the snapshots.

    LHS sampling rather than uniform random: with only ~100 individuals over a
    genome whose first gene sets K on a log scale, a bad initial spread costs
    more generations than the sampler costs to run.

    `eliminate_duplicates` is left on. The K gene decodes through a rounding
    step, so genuinely different vectors routinely produce identical partitions
    and identical F; without it the front fills up with copies and the crowding
    distance stops meaning anything.
    """
    algo = NSGA2(
        pop_size=pop_size,
        sampling=LHS(),
        crossover=SBX(prob=p_cx, eta=eta_cx),
        mutation=PM(eta=eta_mut),
        eliminate_duplicates=True,
    )
    cb = Snapshots(verbose=verbose, every=log_every)
    t0 = time.perf_counter()
    res = minimize(problem, algo, get_termination("n_gen", n_gen), seed=seed,
                   callback=cb, save_history=False, verbose=False)
    seconds = time.perf_counter() - t0

    x = np.atleast_2d(res.X) if res.X is not None else np.empty((0, problem.n_var))
    f = np.atleast_2d(res.F) if res.F is not None else np.empty((0, problem.n_obj))
    order = np.lexsort(tuple(f[:, i] for i in range(f.shape[1] - 1, -1, -1))) \
        if len(f) else np.array([], dtype=int)
    return {
        "X": x[order],
        "F": f[order],
        "seconds": seconds,
        "n_eval": problem.n_evals,
        "snapshots": cb.gens,
    }
