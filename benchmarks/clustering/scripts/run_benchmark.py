"""Run one benchmark config.

    python scripts/run_benchmark.py configs/quick.yaml
    python scripts/run_benchmark.py configs/suite_1d.yaml --out results/my_run
    python scripts/run_benchmark.py configs/quick.yaml --datasets gmm3 laplace

Each run writes a timestamped directory holding the config it ran, one
subdirectory per dataset (front, baselines, matched-K table, convergence,
figures) and a suite-level table across all of them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_bench.config import load_config          # noqa: E402
from cluster_bench.runner import run_suite            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: results/<name>-<timestamp>)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="override the config's dataset list")
    ap.add_argument("--n-gen", type=int, default=None)
    ap.add_argument("--pop-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.datasets:
        cfg.datasets = tuple(args.datasets)
    if args.n_gen is not None:
        cfg.search.n_gen = args.n_gen
    if args.pop_size is not None:
        cfg.search.pop_size = args.pop_size
    if args.seed is not None:
        cfg.seed = cfg.search.seed = args.seed
    if args.no_figures:
        cfg.figures = False

    root = Path(__file__).resolve().parents[1]
    out = args.out or root / cfg.out_dir / f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(out)
    print(f"config {args.config}\noutput {out}")
    run_suite(cfg, out, verbose=not args.quiet)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
