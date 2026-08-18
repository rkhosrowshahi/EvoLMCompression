"""Search drivers, with per-generation logging, plotting and checkpointing.

The loop uses pymoo's explicit `setup()` / `has_next()` / `next()` API rather
than `minimize()`. That is what makes it possible to write a figure, append a
generation record and drop a resumable checkpoint after *every* generation --
and to resume a long 7B run that died at generation 31 of 40.

NSGA-II is the default. U-NSGA-III is the one to switch to when you move to a
per-layer encoding: its reference-direction selection keeps working at variable
counts where non-dominated sorting alone stops discriminating. MOEA/D is
included for the algorithm-comparison table.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import time

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.ref_dirs import get_reference_directions

from .evaluate import proxy_fitness
from .objectives import ObjectiveSet, check_redundancy, spearman_matrix
from .plotting import (
    ParetoPlotter, derive_bounds, hv_indicator_nd, refit_bounds,
)
from .video import make_run_video


def build_algorithm(cfg, genome, sampling):
    """Construct the configured algorithm with its operators.

    Note which probability is which. `prob` fires the operator on an individual
    or mating pair; `prob_var` decides each gene once it has fired. The usual
    "1/n_var" rule is about prob_var, and pymoo already defaults it to
    min(0.5, 1/n_var) -- so mutation_prob_var is left null unless overridden.
    Putting 1/n_var into `prob` instead is a silent near-disabling of mutation:
    at 48 variables only 2% of individuals would be touched at all.
    """
    s = cfg.search
    name = s.algorithm

    crossover = SBX(prob=s.crossover_prob, prob_var=s.crossover_prob_var,
                    eta=s.crossover_eta)
    mut_kw = {} if s.mutation_prob_var is None else {"prob_var": s.mutation_prob_var}
    mutation = PM(prob=s.mutation_prob, eta=s.mutation_eta, **mut_kw)

    common = dict(pop_size=s.pop_size, sampling=sampling, crossover=crossover,
                  mutation=mutation)
    if s.n_offsprings is not None:
        common["n_offsprings"] = s.n_offsprings

    if name == "nsga2":
        return NSGA2(eliminate_duplicates=s.eliminate_duplicates, **common)

    n_obj = len(getattr(s, "objectives", None) or ("ppl_proxy", "avg_bits"))
    n_part = s.ref_dir_partitions or das_dennis_partitions(n_obj, s.pop_size)
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_part)
    if name == "unsga3":
        return UNSGA3(ref_dirs=ref_dirs, **common)
    if name == "moead":
        # MOEA/D in pymoo does not handle inequality constraints; enforce the
        # budget through the K bounds instead when using it. It also has no
        # pop_size of its own -- the reference directions set it.
        common.pop("pop_size", None)
        common.pop("n_offsprings", None)
        return MOEAD(ref_dirs=ref_dirs, n_neighbors=s.moead_neighbors,
                     prob_neighbor_mating=s.moead_prob_neighbor_mating, **common)
    raise ValueError(f"unknown algorithm: {name}")


def das_dennis_partitions(n_obj: int, pop_size: int) -> int:
    """Smallest p whose Das-Dennis simplex has at least `pop_size` directions.

    The old default was `pop_size - 1`, which is right only at two objectives,
    where the count IS p + 1. In three it is C(p+2, 2), so pop_size-1 = 99
    would ask for 5151 reference directions to steer a population of 100 --
    U-NSGA-III then has a niche per individual and its selection pressure
    collapses. At M=3, pop 100, this returns 13 (105 directions).
    """
    if n_obj <= 2:
        return max(pop_size - 1, 1)
    p = 1
    while math.comb(p + n_obj - 1, n_obj - 1) < pop_size:
        p += 1
    return p


# -- reference points -------------------------------------------------------

def baseline_sweep(problem, run):
    """fp16 and every uniform-K configuration, on the same proxy windows.

    These are the fixed-bit points every PTQ paper reports, so the front is
    interpretable from generation 1 onwards rather than only at the end.
    """
    comp = problem.compressor
    run.log("evaluating reference points ...")

    comp.restore()
    fp16 = proxy_fitness(comp.model, problem.windows, device=comp.device)
    run.log(f"  fp16                     ppl {fp16:10.3f}")

    rows = [{"tag": "fp16", "K": 0, "ppl_proxy": fp16, "avg_bits": 16.0,
             "avg_bits_archival": 16.0, "cr_deploy": 1.0}]
    points, measured = [], []
    for k in comp.genome.k_choices:
        cand = comp.apply(comp.genome.encode_uniform(k))
        ppl = proxy_fitness(comp.model, problem.windows, device=comp.device)
        s = cand.cost.summary()
        avg_bits = (s["avg_bits_archival"]
               if problem.cfg.search.size_objective == "avg_bits_archival"
               else s["avg_bits"])
        # The reference sweep is where the frozen objective box comes from, so
        # every objective must appear in it -- including a predicted one.
        # Without this, derive_bounds raises "absent from the reference sweep"
        # the moment latency_ms is an objective.
        if getattr(problem, "latency", None) is not None:
            s = {**s, "latency_proxy": problem.latency.predict(cand.cost)}
        rows.append({"tag": f"uniform-K{k}", "K": k, "ppl_proxy": ppl, **s})
        # Real, measured summaries -- these are what the objective bounds are
        # derived from. cost_only's flat-histogram estimate would put the true
        # archival values outside any box derived from it.
        measured.append({"ppl_proxy": ppl, **s})
        points.append((avg_bits, ppl, k))
        run.log(f"  uniform K={k:<4}            ppl {ppl:10.3f}   avg_bits {avg_bits:5.2f}"
                f"   CR {s['cr_deploy']:5.2f}x")
    comp.restore()

    path = run.file("data", "baselines.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    return fp16, points, measured


def _log_objective_correlations(run, objset, measured, cfg=None):
    """Say up front whether any two objectives can actually disagree.

    An objective that is a monotone transform of another costs a full search
    budget to reproduce a front you already have, and the symptom -- a front
    that looks exactly like the 2-objective one -- is easy to misread as the
    search failing. Measured on the reference sweep, which is a curve rather
    than a sample of the whole space, so |rho| here is an *upper* bound on how
    redundant the pair is: points off the uniform-K curve can only disagree
    more. |rho| = 1.000 therefore means genuinely redundant.
    """
    if len(objset) < 2 or len(measured) < 3:
        return
    F = np.array([[m[n] for n in objset.names] for m in measured], dtype=float)
    rho = spearman_matrix(objset.to_min(F))

    run.log("\nobjective correlation on the reference sweep (Spearman, "
            "minimization space)")
    width = max(len(n) for n in objset.names) + 2
    run.log("  " + "".ljust(width) + "".join(n[:10].rjust(11)
                                             for n in objset.names))
    for i, name in enumerate(objset.names):
        cells = "".join(f"{rho[i, j]:>11.3f}" for j in range(len(objset)))
        run.log(f"  {name.ljust(width)}{cells}")

    for i in range(len(objset)):
        for j in range(i + 1, len(objset)):
            if abs(rho[i, j]) > 0.9999:
                run.log(f"  NOTE {objset.names[i]} and {objset.names[j]} are "
                        f"perfectly rank-correlated ON THE REFERENCE SWEEP. "
                        f"That sweep is a one-parameter family (uniform K, no "
                        f"pruning), so any pair monotone in K ties here whether "
                        f"or not it is redundant off the curve. Judge "
                        f"redundancy from the warnings below and from the front "
                        f"itself, not from this number.")
    for msg in check_redundancy(objset, cfg):
        run.log(f"  WARNING {msg}")


# -- main loop --------------------------------------------------------------

def run_search(problem, cfg, run, resume_from: str | None = None):
    comp = problem.compressor
    rng = np.random.default_rng(cfg.search.seed)
    genome = comp.genome

    objset = problem.objectives
    run.log("\n" + objset.describe())

    fp16, baselines, measured = (baseline_sweep(problem, run)
                                 if cfg.search.baseline_sweep
                                 else (None, [], []))
    problem.n_baseline_evals = comp.n_evals
    _log_objective_correlations(run, objset, measured, problem.cfg)

    bounds = derive_bounds(comp, cfg, objset, measured, fp16)
    xlim = _axis(bounds[1])
    ylim = _axis(bounds[0])
    run.log(f"\nfrozen plot box: {objset[1].name} {xlim[0]:.2f}-{xlim[1]:.2f}  "
            f"{objset[0].name} {ylim[0]:.2f}-{ylim[1]:.2f} ({cfg.plot.yscale})")
    _save_box(run, cfg, objset, bounds, fp16, baselines)
    plotter = (ParetoPlotter(run, cfg, xlim, ylim, fp16, baselines,
                             objset=objset, bounds=bounds)
               if cfg.plot.enabled else None)
    hv = hv_indicator_nd(bounds, [s.log for s in objset], objset.names)
    _log_hv_reference(run, hv, fp16, baselines, cfg, objset)

    if resume_from:
        with open(resume_from, "rb") as f:
            algorithm = pickle.load(f)
        algorithm.problem = problem
        run.log(f"resumed from {resume_from} at generation {algorithm.n_gen}")
    else:
        mode = getattr(cfg.search, "init", "logspace")
        if not cfg.search.warm_start:
            mode = "random"       # deprecated alias, kept working
        sampling = genome.seed_population(cfg.search.pop_size, rng, mode)
        run.log(f"initialization: {mode} "
                f"({cfg.search.pop_size} individuals)")
        pv = (cfg.search.mutation_prob_var
              if cfg.search.mutation_prob_var is not None
              else min(0.5, 1.0 / max(genome.n_var, 1)))
        run.log(f"operators: SBX(prob={cfg.search.crossover_prob}, "
                f"prob_var={cfg.search.crossover_prob_var}, "
                f"eta={cfg.search.crossover_eta})  "
                f"PM(prob={cfg.search.mutation_prob}, "
                f"prob_var={pv:.4g}, eta={cfg.search.mutation_eta})")
        # The number that actually matters, spelled out. `prob_var` is a
        # per-GENE rate, so at a fixed value the disruption grows with n_var --
        # and n_var is what the grouping axis varies. Reporting expected genes
        # touched per offspring makes an unintended 50x difference between cells
        # visible in the log instead of only in the fronts.
        run.log(f"           expected genes changed per offspring: "
                f"{cfg.search.mutation_prob * pv * genome.n_var:.2f} of "
                f"{genome.n_var}"
                + ("   (1/n_var regime)" if cfg.search.mutation_prob_var is None
                   else "   <- pinned; see the note in the config"))
        algorithm = build_algorithm(cfg, genome, sampling)
        algorithm.setup(problem, termination=("n_gen", cfg.search.n_gen),
                        seed=cfg.search.seed, verbose=False)

    # One column per objective, so a 3-objective run reports its third axis
    # instead of silently optimizing something the log never mentions.
    # One column per objective, then per reported metric. Each cell holds the
    # front's IDEAL and NADIR for that quantity -- the best and the worst value
    # it takes over the non-dominated set, which is the standard multi-objective
    # pair. Note this is NOT the hypervolume box printed above: that box is
    # fixed for the whole run and derived from the reference sweep, while these
    # move every generation and describe the front itself.
    metrics = tuple(problem.report_metrics)
    cols = list(objset.names) + list(metrics)
    W = 17
    SW = 12  # survival-count column width
    run.log(f"\n{'':>25}{'ideal / nadir over the front':<{W * len(cols)}}")
    run.log(f"{'gen':>5}{'evals':>8}{'front_size':>12}"
            + "".join(c[:W - 2].rjust(W) for c in cols)
            + f"{'HV':>9}{'sec':>8}{'surv_pop':>{SW}}{'surv_front':>{SW}}")
    run.log("-" * (25 + W * len(cols) + 17 + 2 * SW))

    history, records = [], []
    while algorithm.has_next():
        t0 = time.perf_counter()
        algorithm.next()
        # pymoo numbers the initial population as generation 1 and increments
        # before we see it; count evaluated generations ourselves so the first
        # frame is gen_0001.
        gen = len(records) + 1

        X_pop = algorithm.pop.get("X")
        X_front = algorithm.opt.get("X")
        # `algorithm.off` is this generation's offspring (advance() sets it
        # unconditionally, gen 1 included, where "offspring" is just the
        # sampled initial population -- see Algorithm.advance). Population is
        # a numpy object-array of Individual instances, and environmental
        # selection only ever SELECTS existing ones into pop/opt (`pop[survivors]`
        # in RankAndCrowding._do) -- it never copies or recomputes them -- so
        # Python object identity is an exact, dependency-free membership test.
        off = algorithm.off if algorithm.off is not None else algorithm.pop[:0]
        n_off = len(off)
        n_surv_pop = _survivor_count(off, algorithm.pop)
        n_surv_front = _survivor_count(off, algorithm.opt)
        # Everything below here is REAL space: the sign flip that makes a
        # maximized objective minimizable belongs to pymoo, not to the log,
        # the figures or the stored front.
        F_pop = objset.to_real(algorithm.pop.get("F"))
        F_front = objset.to_real(algorithm.opt.get("F"))
        hv_now = hv(F_front)
        dt = time.perf_counter() - t0

        # Ideal = best over the front, nadir = worst over the front, each read
        # in the objective's own direction.
        best = {s.name: (float(np.min(F_front[:, j])) if s.sense == 1
                         else float(np.max(F_front[:, j])))
                for j, s in enumerate(objset)}
        worst = {s.name: (float(np.max(F_front[:, j])) if s.sense == 1
                          else float(np.min(F_front[:, j])))
                 for j, s in enumerate(objset)}
        # Reported metrics are not optimized, so they have no direction and
        # "ideal" is meaningless for them: report the range the front spans.
        mvals = {}
        for name in metrics:
            v = algorithm.opt.get(name)
            if v is None:
                continue
            v = np.ravel(np.asarray(v, dtype=float))
            v = v[np.isfinite(v)]
            if len(v):
                mvals[name] = (float(v.min()), float(v.max()))
        rec = {
            "gen": gen,
            "n_eval": comp.n_evals,
            "n_front": int(len(F_front)),
            "objectives": list(objset.names),
            "ideal": best,
            "nadir": worst,
            "metrics": {k: {"min": lo, "max": hi} for k, (lo, hi) in mvals.items()},
            # Kept under the old names so existing readers do not break.
            "best": best,
            "worst": worst,
            "hypervolume": hv_now,
            "seconds": round(dt, 2),
            "front": [[float(v) for v in row] for row in F_front],
            "n_offspring": n_off,
            "n_offspring_survived_pop": n_surv_pop,
            "n_offspring_survived_front": n_surv_front,
        }
        # Kept so the existing readers of generations.jsonl keep working.
        rec["best_ppl"] = best.get(objset[0].name)
        rec["min_avg_bits"] = best.get(objset[1].name)
        rec["max_avg_bits"] = worst.get(objset[1].name)
        records.append(rec)
        run.jsonl("generations", rec)
        cells = [_pair(best[s.name], worst[s.name]) for s in objset]
        cells += [_pair(*mvals[m]) if m in mvals else "-" for m in metrics]
        run.log(f"{gen:>5}{comp.n_evals:>8}{len(F_front):>12}"
                + "".join(c.rjust(W) for c in cells)
                + f"{hv_now:>9.4f}{dt:>8.1f}"
                + f"{f'{n_surv_pop}/{n_off}':>{SW}}"
                + f"{f'{n_surv_front}/{n_off}':>{SW}}", echo=True)

        if cfg.log.save_history:
            # X and F stay in the algorithm's own space so a checkpoint and its
            # history describe the same numbers.
            history.append({"gen": gen, "X": np.asarray(X_pop, dtype=float),
                            "F": np.asarray(algorithm.pop.get("F"), dtype=float)})

        if plotter and (gen % cfg.plot.every == 0):
            plotter.frame(gen, F_pop, F_front, comp.n_evals, hv_now)

        if cfg.log.checkpoint_every and gen % cfg.log.checkpoint_every == 0:
            _checkpoint(algorithm, run, gen)

    res = algorithm.result()
    problem.restore()
    _checkpoint(algorithm, run, len(records), latest_only=True)

    if plotter and cfg.plot.refit_at_end:
        seen = [np.array(r["front"], dtype=float) for r in records]
        seen += [objset.to_real(h["F"]) for h in history]
        new_bounds = refit_bounds(bounds, objset, cfg,
                                  np.vstack(seen) if seen else np.empty((0, len(objset))))
        if new_bounds is not None:
            moved = [f"{s.name} [{bounds[j][0]:,.2f}, {bounds[j][1]:,.2f}] -> "
                     f"[{new_bounds[j][0]:,.2f}, {new_bounds[j][1]:,.2f}]"
                     for j, s in enumerate(objset)
                     if new_bounds[j] != bounds[j]]
            run.log(f"refitting the box and re-rendering {len(records)} frames")
            for m in moved:
                run.log(f"  {m}")
            bounds = new_bounds
            xlim, ylim = _axis(bounds[1]), _axis(bounds[0])
            _save_box(run, cfg, objset, bounds, fp16, baselines)
            plotter = ParetoPlotter(run, cfg, xlim, ylim, fp16, baselines,
                                    objset=objset, bounds=bounds)
            hv = hv_indicator_nd(bounds, [s.log for s in objset], objset.names)
            run.log(f"  hypervolume reference moved: ideal {objset[0].name} is "
                    f"now {hv.ideal[0]:,.2f}; every HV below is on the new box")
            for i, rec in enumerate(records):
                front = np.array(rec["front"], dtype=float)
                pop = objset.to_real(history[i]["F"]) if i < len(history) else front
                rec["hypervolume"] = hv(front)
                if rec["gen"] % cfg.plot.every == 0:
                    plotter.frame(rec["gen"], pop, front, rec["n_eval"],
                                  rec["hypervolume"])

    if plotter:
        F_front = objset.to_real(np.atleast_2d(res.F))
        plotter.frame(len(records), F_front, F_front, comp.n_evals,
                      hv(F_front), stem=run.file("figures", "pareto_final"))
        plotter.convergence([r["gen"] for r in records],
                            [r["hypervolume"] for r in records])
        if cfg.plot.video:
            run.log("encoding Pareto evolution video ...")
            make_run_video(run, cfg)

    if cfg.log.save_history and history:
        save_history(run.file("data", "history.npz"), history, run)
    return res, records


def _survivor_count(offspring, target) -> int:
    """How many `offspring` Individuals are also members of `target`, by
    Python object identity.

    Environmental selection (e.g. RankAndCrowding._do's `pop[survivors]`)
    only ever SELECTS existing Individual objects into the next population
    or front -- it never copies or recomputes them -- so `id()` is an exact
    membership test, with no dependency on X happening to be unique or on
    any pymoo-internal bookkeeping.
    """
    target_ids = {id(ind) for ind in target}
    return sum(1 for ind in offspring if id(ind) in target_ids)


def _num(v: float) -> str:
    """Four significant figures in as few characters as possible.

    Values in one table span perplexities from 27 to 1e8 and ratios down to
    0.001, so any fixed decimal format either loses the small ones or spends
    the whole column width on the large ones.
    """
    if not np.isfinite(v):
        return "inf" if v > 0 else "-inf"
    a = abs(v)
    if a and (a >= 1e4 or a < 1e-2):
        return f"{v:.1e}".replace("e+0", "e").replace("e-0", "e-")
    if a >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}"


def _pair(lo: float, hi: float) -> str:
    """One cell holding a quantity's ideal and nadir."""
    return f"{_num(lo)}/{_num(hi)}"


def _axis(bound):
    """(ideal, nadir) -> an increasing (lo, hi) matplotlib can take."""
    return (min(bound), max(bound))


def _save_box(run, cfg, objset, bounds, fp16, baselines):
    """Persist the frozen box.

    A later replot must reuse the identical box or its frames will not be
    comparable with the ones written during the run. `objectives` and `bounds`
    are what a 3-objective reader needs; `xlim`/`ylim` are kept alongside so
    tools written against the 2-objective layout keep working.
    """
    xlim, ylim = _axis(bounds[1]), _axis(bounds[0])
    with open(run.file("data", "plot_box.json"), "w") as f:
        json.dump({
            "objectives": list(objset.names),
            "senses": [int(s.sense) for s in objset],
            "bounds": [list(b) for b in bounds],   # (ideal, nadir) per objective
            "xlim": list(xlim), "ylim": list(ylim),
            "yscale": cfg.plot.yscale, "fp16_ppl": fp16,
            "baselines": [list(b) for b in baselines],
        }, f, indent=2)


def _log_hv_reference(run, hv, fp16, baselines, cfg, objset):
    """Print the reference points hypervolume is measured against.

    HV is meaningless without stating its reference: the same front scores
    differently under a different box. Everything here is read off the
    indicator object itself, so the printout cannot drift from the maths.
    """
    run.log("\nhypervolume reference")
    for j, spec in enumerate(objset):
        ideal, nadir = hv.ideal[j], hv.ref_point[j]
        axis = "  (log10)" if spec.log else ""
        arrow = "min" if spec.sense == 1 else "MAX"
        run.log(f"  objective {j + 1}  {spec.name:<22} {arrow}  "
                f"{ideal:>12,.3f} .. {nadir:>12,.3f}{axis}")
    corner = ", ".join(f"{n} {v:,.3f}"
                       for n, v in zip(objset.names, hv.ref_point))
    ideal = ", ".join(f"{n} {v:,.3f}" for n, v in zip(objset.names, hv.ideal))
    ones = ", ".join(["1.0"] * len(objset))
    zeros = ", ".join(["0.0"] * len(objset))
    run.log(f"  reference point (worst corner)  {corner}  ->  ({ones}) normalized")
    run.log(f"  ideal point     (best corner)   {ideal}  ->  ({zeros}) normalized")

    run.log(f"  y lower: {_bound_origin(cfg.plot.ylim_min, cfg.plot.ylim_min_ratio, fp16, 'lowest reference')}")
    run.log(f"  y upper: {_bound_origin(cfg.plot.ylim_max, cfg.plot.ylim_max_ratio, fp16, 'highest reference')}")
    if baselines:
        lo = min(baselines, key=lambda b: b[0])
        hi = max(baselines, key=lambda b: b[0])
        run.log(f"                avg_bits span from uniform K={lo[2]} "
                f"({lo[0]:.3f} avg_bits) to K={hi[2]} ({hi[0]:.3f} avg_bits), +5% pad")
        y0, y1 = _axis(hv.bounds[0])
        off = [b for b in baselines if not (y0 <= b[1] <= y1)]
        if off:
            run.log(f"  note: {len(off)} baseline point(s) lie outside the box "
                    f"(K={', '.join(str(b[2]) for b in off)}) and are clipped "
                    f"to the reference corner when scored")
    run.log("  HV is normalized to [0,1]; comparable only across runs sharing "
            "this box")


def save_history(path, history, run=None):
    """Persist every generation's population, allowing ragged generations.

    Population size is not guaranteed constant: duplicate elimination, an
    algorithm with its own sizing (MOEA/D takes it from the reference
    directions), or a partial final generation can all vary it. Stacking into
    one 3-D array assumes otherwise and raises "all input arrays must have the
    same shape" -- at the very end, after the whole search has been paid for.

    So rows are concatenated and a per-generation count is stored alongside,
    which costs nothing and cannot fail.
    """
    counts = np.array([len(h["F"]) for h in history], dtype=np.int64)
    if run is not None and len(set(counts.tolist())) > 1:
        run.log(f"  note: population size varied across generations "
                f"({counts.min()}-{counts.max()}); history stored ragged")
    np.savez_compressed(
        path,
        gens=np.array([h["gen"] for h in history]),
        counts=counts,
        X=np.concatenate([np.atleast_2d(h["X"]) for h in history]),
        F=np.concatenate([np.atleast_2d(h["F"]) for h in history]),
    )


def load_history(path):
    """Read a history file back as a list of (gen, X, F) per generation.

    Handles both layouts: the current concatenated one, and the older stacked
    3-D arrays from runs made before ragged generations were supported.
    """
    with np.load(path) as z:
        gens, X, F = z["gens"], z["X"], z["F"]
        counts = z["counts"] if "counts" in z.files else None
    if counts is None:                      # legacy [gen, pop, k] stacking
        return [(int(g), X[i], F[i]) for i, g in enumerate(gens)]
    out, start = [], 0
    for g, n in zip(gens, counts):
        out.append((int(g), X[start:start + n], F[start:start + n]))
        start += n
    return out


def _bound_origin(absolute, ratio, fp16, uncapped_desc):
    """Say where a y bound came from.

    Each bound has three possible origins and the message has to match the one
    that actually applied -- naming a ratio that was overridden by an absolute
    bound, or that is None because the axis is uncapped, is worse than saying
    nothing.
    """
    if absolute is not None:
        return f"pinned at {absolute:,.4g}"
    if ratio is None:
        return f"uncapped -- opens to the {uncapped_desc}"
    if fp16 and np.isfinite(fp16):
        return f"{ratio:g} x fp16 ppl {fp16:,.2f} = {ratio * fp16:,.2f}"
    return f"{ratio:g} x the {uncapped_desc}"


def _refit_box(records, history, ylim, cfg):
    """Reopen the y box so nothing evaluated sits outside it.

    The floor is always reopened when a candidate beat it -- otherwise those
    points get clipped onto the spine instead of excluded. The ceiling is only
    reopened when `ylim_max_ratio` is null, because a finite headroom is an
    explicit instruction to cap the axis and count the rest as off-scale.

    Returns None when the box already contains everything.
    """
    seen = [np.array(r["front"], dtype=float)[:, 0] for r in records]
    seen += [h["F"][:, 0] for h in history]
    if not seen:
        return None
    vals = np.concatenate(seen)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if not len(vals):
        return None

    lo, hi = ylim
    # Perplexity cannot go below 1.0, so never open the axis past it. An
    # explicitly configured floor is left exactly where it was asked for.
    if cfg.plot.ylim_min is None and float(vals.min()) < lo:
        lo = max(float(vals.min()) * cfg.plot.ylim_min_ratio, 1.0)
    if (cfg.plot.ylim_max is None
            and cfg.plot.ylim_max_ratio is None
            and float(vals.max()) > hi):
        hi = float(vals.max()) * 1.05
    return None if (lo, hi) == tuple(ylim) else (lo, hi)


def _checkpoint(algorithm, run, gen, latest_only=False):
    problem = algorithm.problem
    algorithm.problem = None  # the problem holds a live model; never pickle it
    try:
        blob = pickle.dumps(algorithm)
    except Exception as e:  # pragma: no cover
        run.log(f"  (checkpoint skipped: {e})", echo=False)
        return
    finally:
        algorithm.problem = problem
    if not latest_only:
        with open(run.checkpoint(gen), "wb") as f:
            f.write(blob)
    with open(run.file("checkpoints", "latest.pkl"), "wb") as f:
        f.write(blob)


# -- outputs ----------------------------------------------------------------

def save_front(res, problem, cfg, run):
    """Write the final front as both JSON (full genomes) and CSV (a table).

    Objective values are written in REAL space under their own names, so a
    maximized objective appears as the ratio it is rather than as a negative
    number nobody can interpret three months later.
    """
    objset = problem.objectives
    n_obj = objset.n_obj
    X = np.atleast_2d(res.X) if res.X is not None else np.zeros((0, problem.n_var))
    F = np.atleast_2d(res.F) if res.F is not None else np.zeros((0, n_obj))
    F = objset.to_real(F) if len(F) else F
    # Sorted along objective 1, the size axis, which is what makes the CSV
    # readable as a trade-off table.
    order = np.argsort(F[:, 1]) if len(F) else []

    front, rows = [], []
    for rank, i in enumerate(order):
        cost = problem.compressor.cost_only(X[i])
        settings = problem.compressor.genome.decode(X[i])
        values = {n: float(F[i, j]) for j, n in enumerate(objset.names)}
        front.append({
            "rank": rank,
            "x": X[i].tolist(),
            "objectives": values,
            # Kept for readers written against the 2-objective layout.
            "ppl_proxy": values.get(objset[0].name, float(F[i, 0])),
            "avg_bits_objective": float(F[i, 1]),
            "settings": {n: {"k": s.k, "t_lo": round(s.t_lo, 4),
                             "t_hi": round(s.t_hi, 4)}
                         for n, s in settings.items()},
            "estimated": cost.summary(),
        })
        # The cost columns come from `cost_only`, which prices a genome without
        # touching the weights: it assumes a flat symbol histogram and, more
        # importantly, NO PRUNING -- its stub reports sparsity 0.0 whatever the
        # genome says. So they are estimates and are prefixed to say so. The
        # real numbers come from run_eval.py into results.csv, which re-applies
        # each genome and measures. Anything about sparsity must be read there.
        rows.append({"rank": rank,
                     **{f"f{j + 1}_{n}": round(float(F[i, j]), 5)
                        for j, n in enumerate(objset.names)},
                     "ppl_proxy": round(float(F[i, 0]), 4),
                     "avg_bits_objective": round(float(F[i, 1]), 4),
                     **{f"est_{k}": round(v, 5) for k, v in cost.summary().items()}})

    with open(run.file("data", "front.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "front": front}, f, indent=2)
    if rows:
        with open(run.file("data", "front.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return run.file("data", "front.json")
