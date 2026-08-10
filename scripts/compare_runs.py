#!/usr/bin/env python3
"""Overlay the Pareto fronts of several runs and tabulate the difference.

Built for the granularity ablation (global vs block-wise vs layer-wise K), but
works for any set of runs over the same model and objective.

    python scripts/compare_runs.py gpt2-k-global gpt2-k-block gpt2-k-layer
    python scripts/compare_runs.py gpt2-k-block gpt2-k-layer --venue ieee --name my-ablation

Hypervolume is **recomputed for every run on one common box**, not read from
each run's log. Stored HV is normalized by that run's own axis box, and the
end-of-run refit can move a box, so the stored numbers are not comparable
across runs. Same for the matched-bpw table: the point of the comparison is
"at equal size, whose perplexity is lower", which requires interpolating each
front at shared bpw levels rather than comparing whichever points each run
happened to land on.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evolmc.config import Config  # noqa: E402
from evolmc.objectives import from_box  # noqa: E402
from evolmc.plotting import (  # noqa: E402
    THEMES, apply_style, hv_indicator_nd, latex_snippet, resolve_figsize,
)
from evolmc.rundir import find_run  # noqa: E402

# Categorical slots 1-3. These three clear the all-pairs CVD and normal-vision
# gates; a fourth would put yellow beside orange and fail. Past three runs, the
# marker shapes below carry the identity and the table is the source of truth.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load(run_path):
    with open(os.path.join(run_path, "config.yaml")) as f:
        import yaml
        cfg = Config.from_dict(yaml.safe_load(f))
    with open(os.path.join(run_path, "data", "plot_box.json")) as f:
        box = json.load(f)
    gens = [json.loads(l) for l in
            open(os.path.join(run_path, "logs", "generations.jsonl"))]

    objset, bounds = from_box(box, cfg.search.size_objective)
    # Columns are the run's objectives in order, in real space; column 0 is
    # the quality axis and column 1 the size axis for every layout so far.
    front = np.array(gens[-1]["front"], dtype=float)
    front = front[np.argsort(front[:, 1])]

    meta = {}
    mpath = os.path.join(run_path, "meta.json")
    if os.path.exists(mpath):
        meta = json.load(open(mpath))
    return dict(path=run_path, name=os.path.basename(run_path), cfg=cfg,
                box=box, gens=gens, front=front, meta=meta,
                objset=objset, bounds=bounds)


def ppl_at(front, bpw):
    """Interpolate a front's perplexity at a given bits-per-weight.

    Linear in log-perplexity, which is the space the objective actually lives
    in. Returns None outside the front's range rather than extrapolating -- a
    front that never reached 2 bpw has no answer there, and inventing one would
    flatter it.
    """
    x, y = front[:, 1], np.log10(np.clip(front[:, 0], 1e-12, None))
    if bpw < x.min() or bpw > x.max():
        return None
    return float(10 ** np.interp(bpw, x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--root", default="logs")
    ap.add_argument("--name", default=None, help="output subdirectory name")
    ap.add_argument("--labels", default=None, help="comma-separated legend labels")
    ap.add_argument("--venue", default=None)
    ap.add_argument("--width", default=None)
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--bpw", default="2,3,4,6,8",
                    help="bpw levels for the matched-size table")
    args = ap.parse_args()

    runs = [load(find_run(r, args.root)) for r in args.runs]
    labels = (args.labels.split(",") if args.labels
              else [r["name"] for r in runs])

    cfg = runs[0]["cfg"]
    if args.venue:
        cfg.plot.venue = args.venue
    if args.width:
        try:
            cfg.plot.width = float(args.width)
        except ValueError:
            cfg.plot.width = args.width
    cfg.plot.formats = tuple(f.strip() for f in args.formats.split(","))
    apply_style(cfg, log=lambda m: print(m.strip()))

    # Hypervolume is only comparable between runs measuring the same things.
    # A 2-objective and a 3-objective run produce numbers in different units
    # of volume, and averaging or tabling them side by side is meaningless --
    # so this is an error rather than a note.
    objsets = {r["objset"].names for r in runs}
    if len(objsets) > 1:
        listing = "\n".join(f"    {r['name']}: {list(r['objset'].names)}"
                            for r in runs)
        raise SystemExit(
            "cannot compare runs with different objectives -- hypervolume is "
            "measured over different spaces and the numbers are not "
            f"commensurable:\n{listing}\n"
            "Compare 2-objective runs with each other and 3-objective runs "
            "with each other, or plot the shared axes without HV."
        )
    objset = runs[0]["objset"]

    # One box for everybody, so hypervolume and the figure are comparable.
    # Union per objective, taking each end in its own direction: `ideal` is the
    # best corner, which for a maximized objective is the LARGER number.
    shared = []
    for j, spec in enumerate(objset):
        ideals = [r["bounds"][j][0] for r in runs]
        nadirs = [r["bounds"][j][1] for r in runs]
        shared.append((min(ideals), max(nadirs)) if spec.sense == 1
                      else (max(ideals), min(nadirs)))
    xlim = (min(shared[1]), max(shared[1]))
    ylim = (min(shared[0]), max(shared[0]))
    boxes = {tuple(np.round(np.ravel(r["bounds"]), 4)) for r in runs}
    if len(boxes) > 1:
        print("note   runs had different axis boxes; using their union and "
              "recomputing every hypervolume on it")
    hv = hv_indicator_nd(shared, [s.log for s in objset], objset.names)

    fp16 = next((r["box"].get("fp16_ppl") for r in runs
                 if r["box"].get("fp16_ppl")), None)
    baselines = runs[0]["box"].get("baselines") or []

    out_dir = os.path.join(args.root, "_compare",
                           args.name or "-vs-".join(labels)[:80])
    os.makedirs(out_dir, exist_ok=True)

    # ---- table -----------------------------------------------------------
    levels = [float(v) for v in args.bpw.split(",")]
    rows = []
    for r, label in zip(runs, labels):
        f = r["front"]
        rows.append({
            "run": label,
            "grouping": r["cfg"].variables.k_grouping,
            "n_var": r["meta"].get("n_var", ""),
            "evals": r["meta"].get("n_evaluations", r["gens"][-1]["n_eval"]),
            "gens": len(r["gens"]),
            "front": len(f),
            "hypervolume": round(hv(f), 5),
            "min_bpw": round(float(f[:, 1].min()), 3),
            "best_ppl": round(float(f[:, 0].min()), 4),
            # Best value of any objective past the second, in its own
            # direction. Absent for 2-objective runs.
            **{f"best_{s.name}": round(
                float(f[:, j].min() if s.sense == 1 else f[:, j].max()), 4)
               for j, s in enumerate(objset) if j >= 2},
            **{f"ppl@{b:g}bpw": (round(v, 4) if (v := ppl_at(f, b)) else "")
               for b in levels},
        })

    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cols = list(rows[0].keys())
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) + 2 for c in cols]
    print()
    print("".join(c.rjust(w) for c, w in zip(cols, widths)))
    print("-" * sum(widths))
    for r in rows:
        print("".join(str(r[c]).rjust(w) for c, w in zip(cols, widths)))

    best = {}
    for b in levels:
        vals = [(r[f"ppl@{b:g}bpw"], r["run"]) for r in rows
                if r[f"ppl@{b:g}bpw"] != ""]
        if vals:
            best[b] = min(vals)
    if best:
        print("\nlowest perplexity at matched size:")
        for b, (v, who) in best.items():
            print(f"  {b:g} bpw -> {who}  (ppl {v:.4f})")

    # ---- front overlay ---------------------------------------------------
    t = THEMES[cfg.plot.style]
    pt = float(plt.rcParams["font.size"])
    figsize = resolve_figsize(cfg)
    exact = cfg.plot.venue != "none"

    fig, ax = plt.subplots(figsize=figsize, facecolor=t["surface"])
    ax.set_facecolor(t["surface"])

    if baselines:
        b = np.array([[x[0], x[1]] for x in sorted(baselines)], dtype=float)
        ax.plot(b[:, 0], b[:, 1], ls=(0, (6, 3)), lw=1.2, color=t["muted"],
                marker="x", ms=4, zorder=2, label=cfg.plot.baseline_label)
    if fp16:
        ax.axhline(fp16, color=t["muted"], lw=1.0, ls=(0, (5, 4)), zorder=2)
        ax.annotate(f"fp16 ({fp16:.2f})", xy=(xlim[0], fp16), xytext=(4, 3),
                    textcoords="offset points", fontsize=pt - 1,
                    color=t["muted"])

    for i, (r, label) in enumerate(zip(runs, labels)):
        f = r["front"]
        ax.plot(f[:, 1], f[:, 0], lw=1.8, color=SERIES[i % len(SERIES)],
                marker=MARKERS[i % len(MARKERS)], ms=5,
                mfc=SERIES[i % len(SERIES)], mec=t["surface"], mew=1.0,
                zorder=4 + i, label=label)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yscale(cfg.plot.yscale)
    ax.set_xlabel(objset[1].axis_label, fontsize=pt, color=t["ink_2"])
    ax.set_ylabel(objset[0].axis_label
                  + (" (log)" if cfg.plot.yscale == "log" else ""),
                  fontsize=pt, color=t["ink_2"])
    _style(ax, t, pt, cfg.plot.legend_alpha)
    _save(fig, os.path.join(out_dir, "fronts"), cfg, t, exact)
    if len(objset) > 2:
        print(f"note   objective(s) {[s.name for s in objset][2:]} are "
              "optimized but not drawn; the overlay projects onto the first two")

    # ---- convergence overlay ---------------------------------------------
    side = figsize[0]
    fig, ax = plt.subplots(figsize=(side, side), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_box_aspect(1)
    for i, (r, label) in enumerate(zip(runs, labels)):
        g = [x["gen"] for x in r["gens"]]
        # Recomputed on the shared box, not the stored per-run value.
        v = [hv(np.array(x["front"], dtype=float)) for x in r["gens"]]
        ax.plot(g, v, lw=1.8, color=SERIES[i % len(SERIES)],
                marker=MARKERS[i % len(MARKERS)], ms=4,
                mec=t["surface"], mew=0.8, label=label)
    ax.set_xlabel("generation", fontsize=pt, color=t["ink_2"])
    ax.set_ylabel("hypervolume (shared box)", fontsize=pt, color=t["ink_2"])
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    _style(ax, t, pt, cfg.plot.legend_alpha)
    _save(fig, os.path.join(out_dir, "convergence"), cfg, t, exact)

    print(f"\nwrote {out_dir}/")
    print("  comparison.csv   fronts.png|pdf   convergence.png|pdf")
    if cfg.plot.venue != "none":
        print("\n" + latex_snippet(cfg, os.path.join(out_dir, "fronts.pdf"),
                                   "Pareto fronts under three K groupings.",
                                   "fig:granularity"))


def _style(ax, t, pt, legend_alpha=0.3):
    ax.grid(True, which="major", color=t["grid"], lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=t["grid"], lw=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(t["axis"])
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(which="major", colors=t["muted"], labelsize=pt - 1,
                   length=3, width=0.7)
    ax.tick_params(which="minor", colors=t["muted"], labelsize=pt - 2,
                   length=2, width=0.5)
    leg = ax.legend(loc="best", fontsize=pt - 1, frameon=True,
                    facecolor=t["surface"], edgecolor=t["grid"],
                    framealpha=legend_alpha, borderpad=0.5, handlelength=1.8,
                    labelspacing=0.35)
    for txt in leg.get_texts():
        txt.set_color(t["ink_2"])


def _save(fig, stem, cfg, theme, exact):
    fig.tight_layout()
    for fmt in cfg.plot.formats:
        fig.savefig(f"{stem}.{fmt}", format=fmt, dpi=cfg.plot.dpi,
                    facecolor=theme["surface"],
                    bbox_inches=None if exact else "tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
