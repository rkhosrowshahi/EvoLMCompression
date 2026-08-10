#!/usr/bin/env python3
"""Add the cost columns that run_eval.py's old allowlist dropped.

`run_eval.py` used to filter its output through a hardcoded FIELDS list, so any
column added to `ModelCost.summary()` afterwards never reached results.csv.
Five went missing, and they are the ones that make pruning legible:

    param_reduction   fraction of live parameters removed
    n_alive_total     surviving weights, whole checkpoint
    cr_dense          what the same genome would cost under dense storage
    size_mb_dense     the same, in MB
    bpw_model_archival

This rewrites existing results.csv files with the full set. Perplexity is NOT
recomputed -- it is read from the file and carried over unchanged, so no corpus
and no `datasets` install is needed. Only the cost columns are re-derived, and
they depend on the genome alone.

Every recomputed value is checked against the one already in the file, so a
mismatch in an existing column surfaces instead of being quietly overwritten.

    python scripts/backfill_results.py
    python scripts/backfill_results.py --runs gpt2-k-global-prune-bitmap-2obj-np100-ng100
    python scripts/backfill_results.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from evolmc import Compressor, Config  # noqa: E402
from evolmc.objectives import canonical, canonicalize_row  # noqa: E402
from evolmc.rundir import find_run  # noqa: E402

LEAD = ["tag", "ppl_eval", "ppl_calib"]


def genome_for(tag, comp, front):
    """The genome a results.csv row was produced from, or None for fp16."""
    if tag == "fp16":
        return None
    if tag.startswith("uniform-K"):
        return comp.genome.encode_uniform(int(tag.split("K")[1]))
    if tag.startswith("front-"):
        i = int(tag.split("-")[1])
        return np.asarray(front[i]["x"], dtype=float) if i < len(front) else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run names; default is every run with a results.csv")
    ap.add_argument("--root", default="logs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = args.runs or sorted(
        os.path.basename(os.path.dirname(os.path.dirname(p)))
        for p in glob.glob(os.path.join(args.root, "*", "data", "results.csv")))
    if not runs:
        raise SystemExit(f"no results.csv found under {args.root}/")

    comp = None
    for name in runs:
        path = find_run(name, args.root)
        rpath = os.path.join(path, "data", "results.csv")
        reader = csv.DictReader(open(rpath))
        # Captured before the rows are mutated below: the fp16 row is filled in
        # place, so reading its keys afterwards would under-report what was new.
        # Retired column spellings are folded to the current ones on read, so
        # the drift guard below compares a stored cr_archive against a freshly
        # computed cr_archive as the same quantity instead of silently adding a
        # second column and validating neither.
        orig_cols = [canonical(c) for c in (reader.fieldnames or [])]
        rows = [canonicalize_row(r) for r in reader]
        cfg = Config.from_dict(yaml.safe_load(open(os.path.join(path, "config.yaml"))))
        if not torch.cuda.is_available():
            cfg.model.device = cfg.model.master_device = "cpu"
            cfg.model.dtype = "float32"

        # One model for the whole sweep; only the genome wiring is per-run.
        if comp is None:
            print(f"loading {cfg.model.name} ...")
            comp = Compressor(cfg)
        comp.cfg = cfg
        comp.genome = type(comp.genome)(comp.targets, cfg.quant, cfg.prune,
                                        cfg.variables)
        comp.cache.enabled = not cfg.prune.enabled

        fpath = os.path.join(path, "data", "front.json")
        front = json.load(open(fpath))["front"] if os.path.exists(fpath) else []

        out, drift, done = [], [], 0
        for r in rows:
            x = genome_for(r["tag"], comp, front)
            if x is None:                     # fp16 row: fill what is exact
                n_all = comp.master.n_target_weights + comp.n_untouched
                r.setdefault("bpw_model_archival", 16.0)
                r.setdefault("cr_dense", 1.0)
                r.setdefault("param_reduction", 0.0)
                r.setdefault("n_alive_total", float(n_all))
                r.setdefault("cr_archive", 1.0)
                out.append(r)
                continue
            s = comp.apply(x).cost.summary()
            for k, v in s.items():
                v = round(v, 5)
                old = r.get(k, "")
                if old not in ("", None):
                    # Guard: an existing column must reproduce, or the genome
                    # no longer corresponds to the row and nothing here is safe.
                    if abs(float(old) - v) > max(2e-4, 2e-4 * abs(v)):
                        drift.append((r["tag"], k, float(old), v))
                r[k] = v
            out.append(r)
            done += 1

        if drift:
            print(f"{name}: MISMATCH on {len(drift)} recomputed value(s), "
                  "skipping (the genome no longer matches the row)")
            for t, k, a, b in drift[:5]:
                print(f"    {t} {k}: file {a} vs recomputed {b}")
            continue

        cols = list(LEAD)
        for r in out:
            for k in r:
                if k not in cols:
                    cols.append(k)
        added = [c for c in cols if c not in orig_cols]
        if args.dry_run:
            print(f"{name}: would add {added} to {len(out)} rows")
            continue
        with open(rpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(out)
        print(f"{name}: {done} genomes re-priced, added {added}")

    if comp is not None:
        comp.restore()


if __name__ == "__main__":
    main()
