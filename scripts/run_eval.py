#!/usr/bin/env python3
"""Full evaluation of a Pareto front on BOTH corpora, plus the reference sweep.

Re-prices every front member with true symbol histograms (the front file stores
a flat-histogram estimate) and measures perplexity twice: on the calibration
corpus the search optimised, and on the held-out corpus it never saw.

That second number is the one a paper reports. A front drawn on the corpus the
search optimised is partly a picture of overfitting -- thousands of evaluations
against a handful of proxy windows will find configurations that suit those
windows. The gap between the two curves is the honest statement of how much of
the front survives, and the Spearman correlation says whether the cheap proxy
ranked candidates the way the full metric does at all.

    python scripts/run_eval.py configs/gpt2_k_layer.yaml logs/gpt2-k-layer
    python scripts/run_eval.py configs/gpt2_k_layer.yaml logs/gpt2-k-layer --skip-baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from evolmc import Compressor, Config, perplexity  # noqa: E402
from evolmc.data import build_splits  # noqa: E402
from evolmc.evaluate import rank_correlation  # noqa: E402
from evolmc.plotting import (  # noqa: E402
    apply_style, latex_snippet, plot_calib_vs_eval, plot_front_on_corpus,
    plot_proxy_correlation,
)
from evolmc.rundir import find_run  # noqa: E402

FIELDS = [
    "tag", "ppl_eval", "ppl_calib", "bpw_target", "bpw_target_archival",
    "bpw_model", "cr_deployable", "cr_archival", "sparsity", "mean_k_used",
    "size_mb_original", "size_mb_deployable", "size_mb_archival",
]


def evaluate_genome(comp, x, w_eval, w_calib, tag):
    """Score one genome on both corpora from a single quantization."""
    cand = comp.apply(x)
    row = {
        "tag": tag,
        "ppl_eval": round(perplexity(comp.model, w_eval, device=comp.device), 4),
        "ppl_calib": round(perplexity(comp.model, w_calib, device=comp.device), 4),
    }
    row.update({k: round(v, 5) for k, v in cand.cost.summary().items()})
    return {k: row.get(k, "") for k in FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("run", nargs="?", default=None,
                    help="run directory; results go to its data/results.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    run_path = find_run(args.run, cfg.log.root) if args.run else None
    front_path = os.path.join(run_path, "data", "front.json") if run_path else None
    comp = Compressor(cfg)
    print(comp.summary())

    splits = build_splits(cfg.data, comp.tokenizer)
    w_eval, w_calib = splits["eval"], splits["proxy"]
    calib_name, eval_name = cfg.data.calib_dataset, cfg.data.eval_dataset
    print(f"\nheld-out : {w_eval.shape[0]} x {w_eval.shape[1]} tokens of {eval_name}")
    print(f"calib    : {w_calib.shape[0]} x {w_calib.shape[1]} tokens of "
          f"{calib_name} (what the search optimised)\n")

    rows = []
    fp16_eval = perplexity(comp.model, w_eval, device=comp.device)
    fp16_calib = perplexity(comp.model, w_calib, device=comp.device)
    rows.append({**{k: "" for k in FIELDS}, "tag": "fp16",
                 "ppl_eval": round(fp16_eval, 4), "ppl_calib": round(fp16_calib, 4),
                 "bpw_target": 16.0, "bpw_model": 16.0, "cr_deployable": 1.0})
    print(f"{'':<22}{eval_name:>12}{calib_name:>12}")
    print(f"{'fp16':<22}{fp16_eval:>12.3f}{fp16_calib:>12.3f}")

    baselines = []
    if not args.skip_baselines:
        for k in comp.genome.k_choices:
            r = evaluate_genome(comp, comp.genome.encode_uniform(k), w_eval,
                                w_calib, f"uniform-K{k}")
            rows.append(r)
            baselines.append((r["bpw_target"], r["ppl_eval"]))
            print(f"{r['tag']:<22}{r['ppl_eval']:>12.3f}{r['ppl_calib']:>12.3f}"
                  f"   bpw {r['bpw_target']:5.2f}  CR {r['cr_deployable']:5.2f}x")

    front = []
    if front_path and os.path.exists(front_path):
        with open(front_path) as f:
            members = json.load(f)["front"]
        for i, member in enumerate(members):
            r = evaluate_genome(comp, np.array(member["x"]), w_eval, w_calib,
                                f"front-{i:02d}")
            rows.append(r)
            front.append(r)
            print(f"{r['tag']:<22}{r['ppl_eval']:>12.3f}{r['ppl_calib']:>12.3f}"
                  f"   bpw {r['bpw_target']:5.2f}  CR {r['cr_deployable']:5.2f}x"
                  f"  (archival {r['cr_archival']:.2f}x)")
    comp.restore()

    out = args.out or os.path.join(run_path or ".", "data", "results.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    if not front:
        return

    # -- does the proxy rank candidates the way the full metric does? --------
    calib = [r["ppl_calib"] for r in front]
    evald = [r["ppl_eval"] for r in front]
    bpw = [r["bpw_target"] for r in front]
    rho = rank_correlation(calib, evald)

    gap = [e / c for e, c in zip(evald, calib) if c > 0]
    print(f"\nproxy vs held-out over {len(front)} front members")
    print(f"  Spearman rho          {rho:+.4f}", end="  ")
    print("(>0.9 means the proxy is a valid surrogate)"
          if rho >= 0.9 else
          "-- BELOW 0.9: the search partly optimised noise; raise data.n_proxy_seq")
    if gap:
        print(f"  held-out / calib ppl  {min(gap):.3f} .. {max(gap):.3f}  "
              f"(median {sorted(gap)[len(gap)//2]:.3f})")
    print(f"  fp16 on each corpus   {eval_name} {fp16_eval:.3f}   "
          f"{calib_name} {fp16_calib:.3f}")

    if args.no_plots or not run_path:
        return

    apply_style(cfg, log=lambda m: print(m.strip()))
    fig_dir = os.path.join(run_path, "figures")
    written = []
    written += plot_front_on_corpus(
        os.path.join(fig_dir, "front_eval"), cfg,
        front=list(zip(bpw, evald)), baseline=baselines or None,
        fp16=fp16_eval, corpus=f"{eval_name} (held-out)")
    written += plot_calib_vs_eval(
        os.path.join(fig_dir, "front_calib_vs_eval"), cfg,
        bpw=bpw, ppl_calib=calib, ppl_eval=evald,
        fp16_calib=fp16_calib, fp16_eval=fp16_eval,
        calib_name=calib_name, eval_name=eval_name)
    written += plot_proxy_correlation(
        os.path.join(fig_dir, "proxy_vs_eval"), cfg,
        ppl_calib=calib, ppl_eval=evald, rho=rho,
        calib_name=calib_name, eval_name=eval_name)

    print("\nfigures")
    for p in written:
        print(f"  {os.path.relpath(p)}")
    if cfg.plot.venue != "none":
        print("\n" + latex_snippet(
            cfg, os.path.relpath(os.path.join(fig_dir, "front_eval.pdf"), run_path),
            f"Pareto front evaluated on held-out {eval_name}.", "fig:front-eval"))


if __name__ == "__main__":
    main()
