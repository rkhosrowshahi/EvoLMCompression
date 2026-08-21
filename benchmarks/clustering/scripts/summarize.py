"""Tabulate one or more finished runs.

    python scripts/summarize.py results/suite_1d-20260820-151016
    python scripts/summarize.py results/*/ --markdown > RESULTS.md
    python scripts/summarize.py results/suite_1d-* --matched

Prints the same two tables and legend the run itself printed, rebuilt from what
it wrote to disk -- so re-rendering never disagrees with the original output.
With --matched it also prints the per-dataset head-to-head at identical cluster
counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_bench.report import (_fmt, _render,  # noqa: E402
                                  format_per_arm_tables,
                                  format_suite_tables)

MATCHED_HEAD = ["clusters", "best k-means arm", "MSE k-means", "MSE companding",
                "extra error", "DB k-means", "DB companding", "bits k-means",
                "bits companding"]

MATCHED_LEGEND = """\
  clusters      occupied clusters -- both methods produced exactly this many.
  extra error   how much more squared error companding pays here.  In 1-D the
                k-means column is the PROVEN optimum, so this is an absolute cost.
  DB            Davies-Bouldin, lower is better.  Neither k-means arm targets it.
  bits          entropy of the cluster labels: what an entropy coder would pay
                per point.  Fewer bits at equal error is a real compression win."""


def matched_rows(summary):
    return [[m["k_eff"], m["baseline_method"].replace("kmeans_", ""),
             _fmt(m["kmeans_mse"], "{:.4g}"),
             _fmt(m["companding_mse"], "{:.4g}"),
             _fmt(m["excess_pct"], "{:+.1f}%"),
             _fmt(m["kmeans_db"], "{:.3f}"),
             _fmt(m["companding_db"], "{:.3f}"),
             _fmt(m["kmeans_entropy_bits"], "{:.2f}"),
             _fmt(m["companding_entropy_bits"], "{:.2f}")]
            for m in summary["matched_k"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--matched", action="store_true",
                    help="also print the head-to-head at identical cluster counts")
    args = ap.parse_args()

    for run in args.runs:
        path = run / "suite.json"
        if not path.exists():
            print(f"# {run}: no suite.json (unfinished, or wrong directory)")
            continue
        summaries = json.loads(path.read_text())
        print(f"\n## {run.name}\n" if args.markdown else f"\n=== {run.name} ===\n")
        print(format_per_arm_tables(summaries, args.markdown))
        print()
        print(format_suite_tables(summaries, args.markdown))

        if args.matched:
            for s in summaries:
                if not s["matched_k"]:
                    continue
                title = (f"\n#### {s['dataset']}: same cluster count, head to head\n"
                         if args.markdown else
                         f"\n--- {s['dataset']}: same cluster count, head to head ---\n")
                print(title)
                print(_render(MATCHED_HEAD, matched_rows(s), args.markdown))
                print()
                print(MATCHED_LEGEND)


if __name__ == "__main__":
    main()
