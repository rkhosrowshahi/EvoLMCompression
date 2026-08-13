#!/usr/bin/env python3
"""Re-price finished fronts under every deployable storage format.

The runs were priced with `deployable_format: dense`, which charges one index
per weight POSITION. A pruned weight therefore costs exactly what a live one
does, and sparsity changes the deployable size by zero -- verified on the
finished pruned runs, where candidates spanning 0.00 to 0.95 sparsity at the
same avg_bits had identical cr_deploy to six decimals.

This script re-derives the real per-layer statistics for every front member and
prices each one three ways, so the size benefit of pruning becomes visible
without re-running any search:

    dense   one index per position                     (what the runs used)
    bitmap  1 bit/position + one index per survivor
    csr     index + gap field per survivor, Deep Compression style

Perplexity is not recomputed; it is read from each run's results.csv, so this
needs no forward passes.

    python scripts/reprice_fronts.py
    python scripts/reprice_fronts.py --runs gpt2-k-layer-3obj-prune-np100-ng100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evolmc import Compressor, Config  # noqa: E402
from evolmc.codec import ModelCost, price_layer  # noqa: E402
from evolmc.quantize import compress_layer  # noqa: E402
from evolmc.rundir import find_run  # noqa: E402

FORMATS = ("dense", "bitmap", "csr")

DEFAULT_RUNS = [
    "gpt2-k-global-3obj-prune-np100-ng100",
    "gpt2-k-block-3obj-prune-np100-ng100",
    "gpt2-k-layer-3obj-prune-np100-ng100",
    "gpt2-k-global-3obj-np100-ng100",
    "gpt2-k-block-3obj-np100-ng100",
    "gpt2-k-layer-3obj-np100-ng100",
]


def layer_stats(comp, cfg, x):
    """Real per-layer quantizer statistics for one genome.

    Mirrors Compressor.apply but keeps the stats and never writes the
    reconstruction back, because nothing here needs a forward pass.
    """
    settings = comp.genome.decode(np.asarray(x, dtype=float))
    out = []
    for layer in comp.targets:
        s = settings[layer.name]
        _, st = compress_layer(
            comp.master.original(layer), comp.master.row_scale(layer),
            k=s.k, t_lo=s.t_lo, t_hi=s.t_hi,
            quant_cfg=cfg.quant, prune_cfg=cfg.prune, name=layer.name)
        out.append(st)
    return out, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--root", default="logs")
    ap.add_argument("--config", default="configs/uq_pruning/gpt2_124m/gpt2_124m-only_proj-layer_quant-global_prune_sigma-bitmap-3obj.yaml",
                    help="only the model/quant settings are used")
    ap.add_argument("--limit", type=int, default=None,
                    help="price only the first N front members (for a quick look)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if not torch.cuda.is_available():
        cfg.model.device = cfg.model.master_device = "cpu"
        cfg.model.dtype = "float32"
    print(f"loading {cfg.model.name} on {cfg.model.device} ...")
    comp = Compressor(cfg)
    n_untouched = comp.n_untouched

    for run_name in args.runs:
        try:
            run_path = find_run(run_name, args.root)
        except FileNotFoundError:
            print(f"skip {run_name}: not found")
            continue

        with open(os.path.join(run_path, "data", "front.json")) as fh:
            members = json.load(fh)["front"]
        if args.limit:
            members = members[: args.limit]

        # Held-out perplexity per front member, by position, from run_eval.py.
        ppl = {}
        rpath = os.path.join(run_path, "data", "results.csv")
        if os.path.exists(rpath):
            with open(rpath) as fh:
                for r in csv.DictReader(fh):
                    if r["tag"].startswith("front-"):
                        ppl[int(r["tag"].split("-")[1])] = float(r["ppl_eval"])

        # This run's own prune setting decides whether the genome even has a
        # threshold band; read it from the stored config rather than assuming.
        with open(os.path.join(run_path, "config.yaml")) as fh:
            import yaml
            run_cfg = Config.from_dict(yaml.safe_load(fh))
        cfg.prune = run_cfg.prune
        cfg.variables = run_cfg.variables
        cfg.quant.k_encoding = run_cfg.quant.k_encoding
        cfg.quant.k_min, cfg.quant.k_max = run_cfg.quant.k_min, run_cfg.quant.k_max
        comp.genome = type(comp.genome)(comp.targets, cfg.quant, cfg.prune,
                                        cfg.variables)

        rows, t0 = [], time.perf_counter()
        for i, m in enumerate(members):
            stats, settings = layer_stats(comp, cfg, m["x"])
            row = {"rank": i, "ppl_eval": ppl.get(i, "")}
            for fmt in FORMATS:
                mc = ModelCost(
                    layers=[price_layer(st, cfg.quant.codebook_bits, fmt=fmt,
                                        csr_span_bits=cfg.quant.csr_span_bits)
                            for st in stats],
                    n_untouched_weights=n_untouched)
                s = mc.summary()
                row[f"{fmt}_avg_bits"] = round(s["avg_bits"], 5)
                row[f"{fmt}_cr"] = round(s["cr_deploy"], 5)
                row[f"{fmt}_size_mb"] = round(s["size_mb_deployable"], 4)
                if fmt == "dense":
                    row["sparsity"] = round(s["sparsity"], 5)
                    row["param_reduction"] = round(s["param_reduction"], 5)
                    row["n_alive_total"] = int(s["n_alive_total"])
                    row["cr_archive"] = round(s["cr_archive"], 5)
                    row["mean_k_used"] = round(s["mean_k_used"], 3)
                    row["mean_k"] = round(
                        float(np.mean([st.k_nominal for st in stats])), 2)
                    # Where the bits actually go, under bitmap.
                    bm = [price_layer(st, cfg.quant.codebook_bits, fmt="bitmap")
                          for st in stats]
                    tot = sum(c.total_deployable for c in bm)
                    row["bitmap_mask_share"] = round(
                        sum(c.mask_bits for c in bm) / max(tot, 1), 4)
                    row["bitmap_idx_share"] = round(
                        sum(c.index_bits_sparse for c in bm) / max(tot, 1), 4)
                    row["bitmap_cb_share"] = round(
                        sum(c.codebook_bits_sparse for c in bm) / max(tot, 1), 4)
            rows.append(row)
            if (i + 1) % 20 == 0:
                el = time.perf_counter() - t0
                print(f"  {run_name}: {i+1}/{len(members)}  "
                      f"({el:.0f}s, {el/(i+1):.2f}s each)")

        out = os.path.join(run_path, "data", "reprice.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        sp = np.array([r["sparsity"] for r in rows])
        print(f"{run_name}\n  wrote {out}  ({len(rows)} members, "
              f"sparsity {sp.min():.3f}-{sp.max():.3f})")

    comp.restore()


if __name__ == "__main__":
    main()
