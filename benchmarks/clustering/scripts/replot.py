"""Redraw a finished run's figures from its CSVs, without re-running anything.

    python scripts/replot.py                        # every run under results/
    python scripts/replot.py results/suite_1d-...   # one run

A search costs minutes; a change to an axis label costs nothing, and the two
should never be coupled. Everything the figures need is already on disk --
front.csv, baselines.csv, archive.csv, convergence.csv -- so restyling, fixing
a label, or correcting a plotting bug is a second-long operation on results that
do not move.

Redraws `objective_space.png` and `convergence.png`. `warp.png` and
`partitions.png` come from `plot_partitions.py`, which rebuilds partitions from
the stored genomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_bench import report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ("mse", "sse", "davies_bouldin", "neg_silhouette", "silhouette",
           "k_eff", "entropy_bits", "index_bits", "min_cluster_size",
           "adjusted_rand", "calinski_harabasz", "hv", "gen", "n_eval", "t")


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        row = dict(r)
        for k in NUMERIC:
            if row.get(k) not in (None, ""):
                try:
                    row[k] = float(row[k])
                except ValueError:
                    pass
        out.append(row)
    return out


def replot_run(run: Path, arms=None, suffix="") -> int:
    cfg_path = run / "config.json"
    objectives = tuple(json.loads(cfg_path.read_text())["objectives"]) \
        if cfg_path.exists() else ("mse", "davies_bouldin")

    n = 0
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        front, base = rows(d / "front.csv"), rows(d / "baselines.csv")
        if front and base:
            report.plot_objective_space(
                d / f"objective_space{suffix}.png", d.name, front, base,
                objectives, rows(d / "archive.csv"), arms=arms)
            n += 1
        curve = rows(d / "convergence.csv")
        if curve and not suffix:
            report.plot_convergence(d / "convergence.png", d.name, curve)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="restrict the K-means arms drawn, e.g. --arms sklearn")
    ap.add_argument("--suffix", default="",
                    help="appended to the output filename, e.g. _v2")
    args = ap.parse_args()
    runs = args.runs or sorted(p for p in (ROOT / "results").glob("suite_*")
                               if p.is_dir())
    if not runs:
        sys.exit("no runs found under results/")
    for run in runs:
        print(f"{run.name}: redrew {replot_run(run, args.arms, args.suffix)} datasets")


if __name__ == "__main__":
    main()
