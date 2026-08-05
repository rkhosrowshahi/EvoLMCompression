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
from .plotting import ParetoPlotter, derive_limits, hv_indicator
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

    n_part = s.ref_dir_partitions or (s.pop_size - 1)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_part)
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

    rows = [{"tag": "fp16", "K": 0, "ppl_proxy": fp16, "bpw_target": 16.0,
             "bpw_model": 16.0, "cr_deployable": 1.0}]
    points = []
    for k in comp.genome.k_choices:
        cand = comp.apply(comp.genome.encode_uniform(k))
        ppl = proxy_fitness(comp.model, problem.windows, device=comp.device)
        s = cand.cost.summary()
        bpw = (s["bpw_target"] if problem.cfg.search.size_objective == "bpw_target"
               else s["bpw_model"])
        rows.append({"tag": f"uniform-K{k}", "K": k, "ppl_proxy": ppl, **s})
        points.append((bpw, ppl, k))
        run.log(f"  uniform K={k:<4}            ppl {ppl:10.3f}   bpw {bpw:5.2f}"
                f"   CR {s['cr_deployable']:5.2f}x")
    comp.restore()

    path = run.file("data", "baselines.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    return fp16, points


# -- main loop --------------------------------------------------------------

def run_search(problem, cfg, run, resume_from: str | None = None):
    comp = problem.compressor
    rng = np.random.default_rng(cfg.search.seed)
    genome = comp.genome

    fp16, baselines = (baseline_sweep(problem, run)
                       if cfg.search.baseline_sweep else (None, []))
    problem.n_baseline_evals = comp.n_evals

    xlim, ylim = derive_limits(comp, cfg, [b[1] for b in baselines], fp16)
    run.log(f"\nfrozen plot box: bpw {xlim[0]:.2f}-{xlim[1]:.2f}  "
            f"ppl {ylim[0]:.2f}-{ylim[1]:.2f} ({cfg.plot.yscale})")
    # Persist it: a later replot must reuse the identical box, or its frames
    # will not be comparable with the ones written during the run.
    with open(run.file("data", "plot_box.json"), "w") as f:
        json.dump({"xlim": list(xlim), "ylim": list(ylim),
                   "yscale": cfg.plot.yscale, "fp16_ppl": fp16,
                   "baselines": [list(b) for b in baselines]}, f, indent=2)
    plotter = (ParetoPlotter(run, cfg, xlim, ylim, fp16, baselines)
               if cfg.plot.enabled else None)
    hv = hv_indicator(xlim, ylim, cfg.plot.yscale)
    _log_hv_reference(run, hv, fp16, baselines, cfg)

    if resume_from:
        with open(resume_from, "rb") as f:
            algorithm = pickle.load(f)
        algorithm.problem = problem
        run.log(f"resumed from {resume_from} at generation {algorithm.n_gen}")
    else:
        mode = getattr(cfg.search, "init", "linspace")
        if not cfg.search.warm_start:
            mode = "random"       # deprecated alias, kept working
        sampling = genome.seed_population(cfg.search.pop_size, rng, mode)
        run.log(f"initialisation: {mode} "
                f"({cfg.search.pop_size} individuals)")
        pv = (cfg.search.mutation_prob_var
              if cfg.search.mutation_prob_var is not None
              else min(0.5, 1.0 / max(genome.n_var, 1)))
        run.log(f"operators: SBX(prob={cfg.search.crossover_prob}, "
                f"prob_var={cfg.search.crossover_prob_var}, "
                f"eta={cfg.search.crossover_eta})  "
                f"PM(prob={cfg.search.mutation_prob}, "
                f"prob_var={pv:.4g}, eta={cfg.search.mutation_eta})")
        algorithm = build_algorithm(cfg, genome, sampling)
        algorithm.setup(problem, termination=("n_gen", cfg.search.n_gen),
                        seed=cfg.search.seed, verbose=False)

    run.log(f"\n{'gen':>5}{'evals':>8}{'|front|':>9}{'best ppl':>12}"
            f"{'min bpw':>10}{'HV':>9}{'sec':>8}")
    run.log("-" * 61)

    history, records = [], []
    while algorithm.has_next():
        t0 = time.perf_counter()
        algorithm.next()
        # pymoo numbers the initial population as generation 1 and increments
        # before we see it; count evaluated generations ourselves so the first
        # frame is gen_0001.
        gen = len(records) + 1

        F_pop = algorithm.pop.get("F")
        X_pop = algorithm.pop.get("X")
        F_front = algorithm.opt.get("F")
        X_front = algorithm.opt.get("X")
        hv_now = hv(F_front)
        dt = time.perf_counter() - t0

        rec = {
            "gen": gen,
            "n_eval": comp.n_evals,
            "n_front": int(len(F_front)),
            "best_ppl": float(np.min(F_front[:, 0])),
            "min_bpw": float(np.min(F_front[:, 1])),
            "max_bpw": float(np.max(F_front[:, 1])),
            "hypervolume": hv_now,
            "seconds": round(dt, 2),
            "front": [[float(a), float(b)] for a, b in F_front],
        }
        records.append(rec)
        run.jsonl("generations", rec)
        run.log(f"{gen:>5}{comp.n_evals:>8}{len(F_front):>9}"
                f"{rec['best_ppl']:>12.3f}{rec['min_bpw']:>10.2f}"
                f"{hv_now:>9.4f}{dt:>8.1f}", echo=True)

        if cfg.log.save_history:
            history.append({"gen": gen, "X": np.asarray(X_pop, dtype=float),
                            "F": np.asarray(F_pop, dtype=float)})

        if plotter and (gen % cfg.plot.every == 0):
            plotter.frame(gen, F_pop, F_front, comp.n_evals, hv_now)

        if cfg.log.checkpoint_every and gen % cfg.log.checkpoint_every == 0:
            _checkpoint(algorithm, run, gen)

    res = algorithm.result()
    problem.restore()
    _checkpoint(algorithm, run, len(records), latest_only=True)

    if plotter and cfg.plot.refit_at_end and not cfg.plot.ylim:
        new_ylim = _refit_floor(records, history, ylim, cfg)
        if new_ylim is not None:
            run.log(f"refitting y floor {ylim[0]:.2f} -> {new_ylim[0]:.2f} "
                    f"and re-rendering {len(records)} frames")
            ylim = new_ylim
            with open(run.file("data", "plot_box.json"), "w") as f:
                json.dump({"xlim": list(xlim), "ylim": list(ylim),
                           "yscale": cfg.plot.yscale, "fp16_ppl": fp16,
                           "baselines": [list(b) for b in baselines]}, f, indent=2)
            plotter = ParetoPlotter(run, cfg, xlim, ylim, fp16, baselines)
            hv = hv_indicator(xlim, ylim, cfg.plot.yscale)
            run.log(f"  hypervolume reference moved: ideal ppl is now "
                    f"{hv.ideal[0]:,.2f}; every HV below is on the new box")
            for i, rec in enumerate(records):
                front = np.array(rec["front"], dtype=float)
                pop = history[i]["F"] if i < len(history) else front
                rec["hypervolume"] = hv(front)
                if rec["gen"] % cfg.plot.every == 0:
                    plotter.frame(rec["gen"], pop, front, rec["n_eval"],
                                  rec["hypervolume"])

    if plotter:
        F_front = np.atleast_2d(res.F)
        plotter.frame(len(records), F_front, F_front, comp.n_evals,
                      hv(F_front), stem=run.file("figures", "pareto_final"))
        plotter.convergence([r["gen"] for r in records],
                            [r["hypervolume"] for r in records])
        if cfg.plot.video:
            run.log("encoding Pareto evolution video ...")
            make_run_video(run, cfg)

    if cfg.log.save_history and history:
        np.savez_compressed(
            run.file("data", "history.npz"),
            gens=np.array([h["gen"] for h in history]),
            X=np.stack([h["X"] for h in history]),
            F=np.stack([h["F"] for h in history]),
        )
    return res, records


def _log_hv_reference(run, hv, fp16, baselines, cfg):
    """Print the reference points hypervolume is measured against.

    HV is meaningless without stating its reference: the same front scores
    differently under a different box. Everything here is read off the
    indicator object itself, so the printout cannot drift from the maths.
    """
    ref_ppl, ref_bpw = hv.ref_point
    id_ppl, id_bpw = hv.ideal
    axis = "log10" if hv.yscale == "log" else "linear"

    run.log("\nhypervolume reference")
    run.log(f"  objective 1  proxy perplexity   {id_ppl:>12,.2f} .. "
            f"{ref_ppl:>12,.2f}   ({axis})")
    run.log(f"  objective 2  bits per weight    {id_bpw:>12.3f} .. "
            f"{ref_bpw:>12.3f}")
    run.log(f"  reference point (worst corner)  ppl {ref_ppl:,.2f}, "
            f"bpw {ref_bpw:.3f}  ->  (1.0, 1.0) normalised")
    run.log(f"  ideal point     (best corner)   ppl {id_ppl:,.2f}, "
            f"bpw {id_bpw:.3f}  ->  (0.0, 0.0) normalised")

    if fp16 and np.isfinite(fp16):
        run.log(f"  derived from: fp16 ppl {fp16:,.2f} "
                f"x {cfg.plot.ylim_floor_ratio:g} (floor) "
                f"and x {cfg.plot.ylim_headroom:g} (ceiling)")
    if baselines:
        lo = min(baselines, key=lambda b: b[0])
        hi = max(baselines, key=lambda b: b[0])
        run.log(f"                bpw span from uniform K={lo[2]} "
                f"({lo[0]:.3f} bpw) to K={hi[2]} ({hi[0]:.3f} bpw), +5% pad")
        off = [b for b in baselines if not (id_ppl <= b[1] <= ref_ppl)]
        if off:
            run.log(f"  note: {len(off)} baseline point(s) lie outside the box "
                    f"(K={', '.join(str(b[2]) for b in off)}) and are clipped "
                    f"to the reference corner when scored")
    run.log("  HV is normalised to [0,1]; comparable only across runs sharing "
            "this box")


def _refit_floor(records, history, ylim, cfg):
    """A lower y floor if anything was evaluated below the current one.

    Returns None when the box already contains every observed point, which is
    the normal case on real data -- no PTQ candidate beats the fp16 model by
    enough to fall through a 10% margin.
    """
    seen = [np.array(r["front"], dtype=float)[:, 0] for r in records]
    seen += [h["F"][:, 0] for h in history]
    if not seen:
        return None
    vals = np.concatenate(seen)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if not len(vals) or float(vals.min()) >= ylim[0]:
        return None
    # Perplexity cannot go below 1.0, so never open the axis past it.
    return (max(float(vals.min()) * cfg.plot.ylim_floor_ratio, 1.0), ylim[1])


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
    """Write the final front as both JSON (full genomes) and CSV (a table)."""
    X = np.atleast_2d(res.X) if res.X is not None else np.zeros((0, problem.n_var))
    F = np.atleast_2d(res.F) if res.F is not None else np.zeros((0, 2))
    order = np.argsort(F[:, 1]) if len(F) else []

    front, rows = [], []
    for rank, i in enumerate(order):
        cost = problem.compressor.cost_only(X[i])
        settings = problem.compressor.genome.decode(X[i])
        front.append({
            "rank": rank,
            "x": X[i].tolist(),
            "ppl_proxy": float(F[i, 0]),
            "bpw_objective": float(F[i, 1]),
            "settings": {n: {"k": s.k, "t_lo": round(s.t_lo, 4),
                             "t_hi": round(s.t_hi, 4)}
                         for n, s in settings.items()},
            "estimated": cost.summary(),
        })
        rows.append({"rank": rank, "ppl_proxy": round(float(F[i, 0]), 4),
                     "bpw_objective": round(float(F[i, 1]), 4),
                     **{k: round(v, 5) for k, v in cost.summary().items()}})

    with open(run.file("data", "front.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "front": front}, f, indent=2)
    if rows:
        with open(run.file("data", "front.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return run.file("data", "front.json")
