#!/usr/bin/env python3
"""Run one compression search.

    python scripts/run_search.py configs/uq_pruning/gpt2_124m/gpt2_124m-type_quant-global_prune_sigma-bitmap-2obj.yaml
    python scripts/run_search.py configs/uq_pruning/pythia_410m/pythia_410m.yaml --name ablation-kmeans
    python scripts/run_search.py configs/uq_pruning/llama2_7b/llama2_7b.yaml --resume gpt2-k-layer

Everything for the run lands in one timestamped directory under logs/.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolmc import Compressor, CompressionProblem, Config  # noqa: E402
from evolmc.data import build_splits  # noqa: E402
from evolmc import latency as latency_mod  # noqa: E402
from evolmc.rundir import RunDir, find_run  # noqa: E402
from evolmc.search import run_search, save_front  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--name", default=None, help="run directory name")
    ap.add_argument("--root", default=None, help="log root (default: logs/)")
    ap.add_argument("--n-gen", type=int, default=None)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--resume", default=None,
                    help="run directory to resume from (uses checkpoints/latest.pkl)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--emit-run-dir", default=None, metavar="FILE",
                    help="write the resolved run directory to FILE. The "
                         "directory is not always what log.run_name says: a "
                         "rerun takes the next free -2/-3 suffix rather than "
                         "appending to an existing run, so a caller that "
                         "guesses the name can act on the wrong one.")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.n_gen:
        cfg.search.n_gen = args.n_gen
    if args.pop:
        cfg.search.pop_size = args.pop
    if args.no_plots:
        cfg.plot.enabled = False

    resume_ckpt = None
    if args.resume:
        prev = find_run(args.resume, args.root or cfg.log.root)
        resume_ckpt = os.path.join(prev, "checkpoints", "latest.pkl")
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"no checkpoint in {prev}")
        run = RunDir(cfg, name=os.path.basename(prev), root=args.root,
                     reuse=True)   # resuming continues the same files
    else:
        run = RunDir(cfg, name=args.name, root=args.root)

    if args.emit_run_dir:
        with open(args.emit_run_dir, "w") as f:
            f.write(run.path)
    run.log(f"run directory: {run.path}\n")
    run.log("loading model ...")
    comp = Compressor(cfg)
    run.log(comp.summary())

    run.log("\npreparing corpora ...")
    splits = build_splits(cfg.data, comp.tokenizer)
    run.log(f"proxy windows {tuple(splits['proxy'].shape)} | "
            f"eval windows {tuple(splits['eval'].shape)}")

    budget = cfg.search.pop_size * cfg.search.n_gen
    run.log(f"\nsearch: {cfg.search.algorithm} pop={cfg.search.pop_size} "
            f"gen={cfg.search.n_gen} -> ~{budget} evaluations")

    # prune.mode == "wanda" scores weights against calibration-data activation
    # norms rather than magnitude alone. Those norms depend only on the frozen
    # fp16 weights and the calibration data, never on a candidate genome, so
    # they are measured once here rather than recomputed per evaluation.
    if cfg.prune.enabled and cfg.prune.mode == "wanda":
        run.log("\ncalibrating wanda activation norms ...")
        wanda_norms = comp.calibrate_wanda(splits["proxy"])
        run.log(f"  {len(wanda_norms.norms)} layers, "
                f"{wanda_norms.n_tokens} calibration tokens")

    # `latency_proxy` is PREDICTED from the per-layer bit accounting against
    # coefficients fitted once on this GPU and frozen to a file. The fit costs a
    # few seconds; every candidate afterwards costs only arithmetic.
    latency = None
    if "latency_proxy" in tuple(cfg.search.objectives) + tuple(
            cfg.search.report_metrics):
        latency = latency_mod.load_or_calibrate(comp, cfg, log=run.log)
        run.log(latency.describe())
        # Numeric redundancy check. A config-shaped heuristic gets this wrong:
        # under bitmap the mask floors bytes at 1 bit/position, so the memory
        # roof can bind everywhere even with pruning on, leaving latency an
        # affine copy of the size objective.
        lo, hi = min(comp.genome.k_choices), max(comp.genome.k_choices)
        for k in (lo, hi):
            c = comp.apply(comp.genome.encode_uniform(k))
            b, t, verdict = latency.roof_diagnostic(c.cost)
            run.log(f"  roof at K={k}: {b}/{t} layers compute-bound -- {verdict}")
        comp.restore()

    problem = CompressionProblem(comp, splits["proxy"], cfg, run=run,
                                 latency=latency)
    try:
        res, records = run_search(problem, cfg, run, resume_from=resume_ckpt)
        save_front(res, problem, cfg, run)
    finally:
        run.save_meta(
            model=cfg.model.name,
            n_target_layers=len(comp.targets),
            n_target_weights=comp.master.n_target_weights,
            n_untouched_weights=comp.n_untouched,
            n_var=comp.genome.n_var,
            n_evaluations=comp.n_evals,
        )

    run.log(f"\ndone. artifacts in {run.path}")
    run.log(f"  figures/pareto/       {cfg.search.n_gen} frames "
            f"({', '.join(cfg.plot.formats)})")
    run.log(f"  figures/pareto_final  final front")
    run.log(f"  figures/convergence   hypervolume vs generation")
    run.log(f"  data/front.csv        the front as a table")
    run.log(f"\nnext: python scripts/run_eval.py {args.config} {run.path}")
    run.close()


if __name__ == "__main__":
    main()
