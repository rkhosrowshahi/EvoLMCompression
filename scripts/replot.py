#!/usr/bin/env python3
"""Re-render a finished run's figures at a different size, font or style.

Reads only the run's own artefacts -- no model load, no re-evaluation -- so
regenerating every figure for a paper takes seconds instead of re-running the
search. The frozen axis box is read back from `data/plot_box.json`, so replotted
frames stay directly comparable with the originals.

    # camera-ready IEEE single-column figures (CEC, TEVC)
    python scripts/replot.py latest --venue ieee --width column

    # GECCO / ACM, full page width, real LaTeX text
    python scripts/replot.py latest --venue acm --width page --usetex

    # just the final front and convergence, no per-generation frames
    python scripts/replot.py latest --venue ieee --no-frames
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from evolmc.config import Config  # noqa: E402
from evolmc.plotting import ParetoPlotter, hv_indicator, latex_snippet  # noqa: E402
from evolmc.rundir import RunDir, find_run  # noqa: E402
from evolmc.video import make_video  # noqa: E402


def load_run(run_path):
    with open(os.path.join(run_path, "config.yaml")) as f:
        import yaml

        cfg = Config.from_dict(yaml.safe_load(f))

    box_path = os.path.join(run_path, "data", "plot_box.json")
    if not os.path.exists(box_path):
        raise SystemExit(
            f"{box_path} not found -- this run predates replotting.\n"
            "Re-run the search, or pass --xlim/--ylim explicitly."
        )
    with open(box_path) as f:
        box = json.load(f)

    gens = []
    gpath = os.path.join(run_path, "logs", "generations.jsonl")
    with open(gpath) as f:
        for line in f:
            gens.append(json.loads(line))
    return cfg, box, gens


def load_populations(run_path, n_gen):
    """Per-generation objective values, if history was saved."""
    from evolmc.search import load_history

    path = os.path.join(run_path, "data", "history.npz")
    if not os.path.exists(path):
        return None
    return [F for _, _, F in load_history(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--root", default="logs")
    ap.add_argument("--venue", default=None,
                    choices=["none", "ieee", "acm", "neurips", "icml", "lncs"])
    ap.add_argument("--width", default=None, help="'column', 'page', or inches")
    ap.add_argument("--aspect", type=float, default=None)
    ap.add_argument("--font-pt", type=float, default=None)
    ap.add_argument("--usetex", action="store_true")
    ap.add_argument("--style", default=None, choices=["paper", "dark"])
    ap.add_argument("--formats", default=None, help="e.g. pdf or png,pdf")
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--no-frames", action="store_true",
                    help="only the final front and convergence figures")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fit-box", action="store_true",
                    help="recompute the y floor from the perplexities actually "
                         "achieved, so front points that beat fp16 are not "
                         "clipped against the axis")
    ap.add_argument("--ylim", default=None, metavar="LO,HI",
                    help="explicit y limits, e.g. 1.0,2e6 (PPL >= 1 always)")
    ap.add_argument("--xlim", default=None, metavar="LO,HI")
    ap.add_argument("--out", default=None,
                    help="write into this subdirectory of the run (default: figures)")
    args = ap.parse_args()

    run_path = find_run(args.run, args.root)
    cfg, box, gens = load_run(run_path)

    for key, val in [("venue", args.venue), ("aspect", args.aspect),
                     ("font_pt", args.font_pt), ("style", args.style),
                     ("dpi", args.dpi)]:
        if val is not None:
            setattr(cfg.plot, key, val)
    if args.width is not None:
        try:
            cfg.plot.width = float(args.width)
        except ValueError:
            cfg.plot.width = args.width
    if args.usetex:
        cfg.plot.usetex = True
    if args.formats:
        cfg.plot.formats = tuple(f.strip() for f in args.formats.split(","))

    # A lightweight stand-in for RunDir: same paths, no new directory, no log.
    class _Run:
        path = run_path
        def file(self, *p): return os.path.join(run_path, args.out or "figures",
                                                *p[1:]) if p[0] == "figures" \
                                   else os.path.join(run_path, *p)
        def frame(self, gen): return self.file("figures", "pareto",
                                               f"gen_{gen:04d}")
        def log(self, msg="", echo=True): print(msg)

    run = _Run()
    os.makedirs(run.file("figures", "pareto"), exist_ok=True)

    xlim, ylim = tuple(box["xlim"]), tuple(box["ylim"])
    pops = load_populations(run_path, len(gens))

    if args.fit_box:
        # The run freezes its box from the reference points alone, before any
        # candidate exists. A search that beats fp16 then lands under the floor
        # and gets clipped against the spine. Replotting has every objective
        # value, so the floor can be refit -- still frozen across all frames
        # here, just fitted to what actually happened.
        seen = [np.array(g["front"], dtype=float)[:, 0] for g in gens]
        if pops is not None:
            seen += [F[:, 0] for F in pops]
        vals = np.concatenate(seen)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals):
            floor = max(float(vals.min()) * 0.9, 1.0)
            if floor < ylim[0]:
                print(f"fit     y floor {ylim[0]:.2f} -> {floor:.2f} "
                      f"(best observed {vals.min():.2f})")
                ylim = (floor, ylim[1])

    for name, raw in (("ylim", args.ylim), ("xlim", args.xlim)):
        if raw:
            lo, hi = (float(v) for v in raw.split(","))
            if name == "ylim":
                ylim = (lo, hi)
            else:
                xlim = (lo, hi)

    from evolmc.plotting import _check_ylim
    ylim = _check_ylim(ylim, cfg.plot.yscale)

    baselines = [tuple(b) for b in box["baselines"]]
    plotter = ParetoPlotter(run, cfg, xlim, ylim, box.get("fp16_ppl"), baselines)
    hv = hv_indicator(xlim, ylim, cfg.plot.yscale)

    print(f"run     {run_path}")
    print(f"figure  {plotter.figsize[0]:.2f} x {plotter.figsize[1]:.2f} in"
          f"  ({cfg.plot.venue}, {', '.join(cfg.plot.formats)})")
    print(f"box     bpw {xlim[0]:.2f}-{xlim[1]:.2f}  "
          f"ppl {ylim[0]:.2f}-{ylim[1]:.2f}")
    if not args.no_frames:
        for i, g in enumerate(gens):
            front = np.array(g["front"], dtype=float)
            pop = pops[i] if pops is not None and i < len(pops) else front
            plotter.frame(g["gen"], pop, front, g["n_eval"], g["hypervolume"])
        print(f"frames  {len(gens)} written")

    final = np.array(gens[-1]["front"], dtype=float)
    plotter.frame(gens[-1]["gen"], final, final, gens[-1]["n_eval"], hv(final),
                  stem=run.file("figures", "pareto_final"))
    plotter.convergence([g["gen"] for g in gens],
                        [g["hypervolume"] for g in gens])
    print("final   pareto_final, convergence")

    if not args.no_frames and not args.no_video and cfg.plot.video:
        if "png" in cfg.plot.formats:
            make_video(run.file("figures", "pareto"),
                       run.file("figures", "pareto_evolution"),
                       fps=cfg.plot.video_fps, formats=cfg.plot.video)
        else:
            print("video   skipped (needs png in --formats)")

    if cfg.plot.venue != "none":
        rel = os.path.relpath(run.file("figures", "pareto_final.pdf"), run_path)
        print("\nLaTeX (figure is already the exact printed width; do not rescale):\n")
        print(latex_snippet(cfg, rel, "Pareto front of perplexity against "
                            "bits per weight.", "fig:pareto"))


if __name__ == "__main__":
    main()
