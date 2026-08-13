#!/usr/bin/env python3
"""Full evaluation of a Pareto front on BOTH corpora, plus the reference sweep.

Re-prices every front member with true symbol histograms (the front file stores
a flat-histogram estimate) and measures perplexity twice: on the calibration
corpus the search optimized, and on the held-out corpus it never saw.

That second number is the one a paper reports. A front drawn on the corpus the
search optimized is partly a picture of overfitting -- thousands of evaluations
against a handful of proxy windows will find configurations that suit those
windows. The gap between the two curves is the honest statement of how much of
the front survives, and the Spearman correlation says whether the cheap proxy
ranked candidates the way the full metric does at all.

    python scripts/run_eval.py configs/uq/gpt2_124m/gpt2_124m-only_proj-layer_quant-2obj.yaml logs/gpt2_124m-only_proj-layer_quant-2obj-np100-ng100
    python scripts/run_eval.py configs/uq/gpt2_124m/gpt2_124m-only_proj-layer_quant-2obj.yaml logs/gpt2_124m-only_proj-layer_quant-2obj-np100-ng100 --skip-baselines
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
from evolmc import benchmark as bench  # noqa: E402
from evolmc.data import build_splits  # noqa: E402
from evolmc.evaluate import rank_correlation  # noqa: E402
from evolmc.plotting import (  # noqa: E402
    apply_style, latex_snippet, plot_calib_vs_eval, plot_front_on_corpus,
    plot_proxy_correlation,
)
from evolmc.rundir import find_run  # noqa: E402

# Columns that are not part of the cost accounting. Everything else is taken
# from ModelCost.summary() as it comes, rather than from an allowlist: a fixed
# list silently drops any field added to the accounting later, which is exactly
# what happened to param_reduction, n_alive_total, cr_dense, size_mb_dense and
# avg_bits_archival -- the columns that make pruning visible.
LEAD = ["tag", "ppl_eval", "ppl_calib"]


def evaluate_genome(comp, x, w_eval, w_calib, tag, cfg=None, ref=None,
                    do_measure=False):
    """Score one genome on both corpora from a single quantization.

    When `ref` is present, also attaches the runtime columns. Two kinds, and the
    names keep them apart:

      measured_*   timed on the live model. CONSTANT across a front, because
                   compression here is simulated and every candidate executes
                   the same dense fp16 graph. Carried anyway, because the
                   constancy is the evidence for that claim.
      *_projected  weight term from the bit accounting, runtime terms from the
                   fp16 reference. These are what vary and what a table should
                   report. See evolmc/benchmark.py for what each assumes.
    """
    cand = comp.apply(x)
    row = {
        "tag": tag,
        "ppl_eval": round(perplexity(comp.model, w_eval, device=comp.device), 4),
        "ppl_calib": round(perplexity(comp.model, w_calib, device=comp.device), 4),
        **{k: round(v, 5) for k, v in cand.cost.summary().items()},
    }
    if ref is not None:
        s = cand.cost.summary()
        row["peak_mb_projected"] = bench.project_peak_mb(
            s["size_mb_deployable"], ref)
        row["latency_ms_projected"] = bench.project_latency_ms(
            s["size_mb_deployable"], ref)
        if do_measure:
            m = bench.measure(comp.model, comp.tokenizer, cfg.benchmark,
                              device=comp.device)
            if m is not None:
                row.update(m.row(prefix="measured_"))
    return row


def write_results(path, rows):
    """Write every column any row carries, leads first, cost fields in order."""
    seen = list(LEAD)
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seen, restval="")
        w.writeheader()
        w.writerows(rows)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("run", nargs="?", default=None,
                    help="run directory; results go to its data/results.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-benchmark", action="store_true",
                    help="skip the latency / peak-memory measurement even when "
                         "benchmark.enabled is true in the config")
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
          f"{calib_name} (what the search optimized)\n")

    # -- fp16 runtime reference ---------------------------------------------
    # Measured BEFORE any genome is applied, so the model is genuinely
    # untouched. Everything the projections need comes from this one point: the
    # weight term they replace, and the KV/activation/workspace term they carry
    # over unchanged.
    ref = None
    if cfg.benchmark.enabled and not args.no_benchmark:
        b = cfg.benchmark
        print(f"benchmark protocol : {b.gen_tokens} single-token decode steps, "
              f"batch {b.batch_size}, past_key_values carried, CUDA-synced per "
              f"step (SqueezeLLM llama.py benchmark(); "
              f"latency is the MEDIAN PER-TOKEN time)")
        ref = bench.measure(comp.model, comp.tokenizer, b, device=comp.device)
        if ref is None:
            print("  SKIPPED: peak-memory figures come from the CUDA allocator "
                  "and the model is not on CUDA.\n")
        else:
            print(f"  fp16 latency     {ref.latency_ms:9.4f} ms/token "
                  f"(p10 {ref.latency_ms_p10:.4f}, p90 {ref.latency_ms_p90:.4f})"
                  f"  over {ref.n_tokens} steps = {ref.total_ms:.1f} ms total")
            print(f"  fp16 peak bytes  {ref.peak_alloc_bytes:>12,.0f} B sampled "
                  f"({ref.peak_alloc_bytes_true:,.0f} B true)")
            print(f"  fp16 peak alloc  {ref.peak_alloc_mb:9.2f} MB sampled "
                  f"(true {ref.peak_alloc_mb_true:.2f}, "
                  f"reserved {ref.peak_reserved_mb:.2f} MB)")
            print(f"  decomposed       {ref.weight_mb:9.2f} MB weights + "
                  f"{ref.runtime_mb:.2f} MB KV/activations/workspace")
            print("  NOTE compression here is SIMULATED -- every candidate runs "
                  "the same dense fp16\n       model -- so the measured_* "
                  "columns below are constant by construction. The\n"
                  "       per-candidate numbers to report are *_projected. "
                  "See evolmc/benchmark.py.\n")

    rows = []
    fp16_eval = perplexity(comp.model, w_eval, device=comp.device)
    fp16_calib = perplexity(comp.model, w_calib, device=comp.device)
    # The uncompressed reference. Its cost fields are known exactly rather than
    # measured, so they are stated rather than left blank: nothing is quantized,
    # nothing is pruned, and every ratio is 1.
    rows.append({"tag": "fp16",
                 "ppl_eval": round(fp16_eval, 4), "ppl_calib": round(fp16_calib, 4),
                 "avg_bits": 16.0, "avg_bits_archival": 16.0,
                 "cr_deploy": 1.0, "cr_archive": 1.0, "cr_dense": 1.0,
                 "sparsity": 0.0, "param_reduction": 0.0,
                 "n_alive_total": float(comp.master.n_target_weights
                                        + comp.n_untouched)})
    if ref is not None:
        # The reference row carries the measurement itself, and its projections
        # are trivially equal to it -- fp16 IS the deployable format here. Stated
        # rather than left blank so the CSV has a row where measured and
        # projected provably agree, which is the sanity check on the projection.
        rows[-1].update(ref.row(prefix="measured_"))
        rows[-1]["peak_mb_projected"] = round(ref.peak_alloc_mb_true, 3)
        rows[-1]["latency_ms_projected"] = round(ref.latency_ms, 3)
    print(f"{'':<22}{eval_name:>12}{calib_name:>12}")
    print(f"{'fp16':<22}{fp16_eval:>12.3f}{fp16_calib:>12.3f}")

    baselines = []
    if not args.skip_baselines:
        for i, k in enumerate(comp.genome.k_choices):
            r = evaluate_genome(comp, comp.genome.encode_uniform(k), w_eval,
                                w_calib, f"uniform-K{k}", cfg=cfg, ref=ref,
                                do_measure=ref is not None
                                and i % max(cfg.benchmark.every, 1) == 0)
            rows.append(r)
            baselines.append((r["avg_bits"], r["ppl_eval"]))
            print(f"{r['tag']:<22}{r['ppl_eval']:>12.3f}{r['ppl_calib']:>12.3f}"
                  f"   avg_bits {r['avg_bits']:5.2f}  CR {r['cr_deploy']:5.2f}x")

    front = []
    if front_path and os.path.exists(front_path):
        with open(front_path) as f:
            members = json.load(f)["front"]
        for i, member in enumerate(members):
            r = evaluate_genome(comp, np.array(member["x"]), w_eval, w_calib,
                                f"front-{i:02d}", cfg=cfg, ref=ref,
                                do_measure=ref is not None
                                and i % max(cfg.benchmark.every, 1) == 0)
            rows.append(r)
            front.append(r)
            line = (f"{r['tag']:<22}{r['ppl_eval']:>12.3f}{r['ppl_calib']:>12.3f}"
                    f"   avg_bits {r['avg_bits']:5.2f}  CR {r['cr_deploy']:5.2f}x"
                    f"  (archive {r['cr_archive']:.2f}x)")
            if ref is not None:
                line += (f"  ->{r['peak_mb_projected']:8.0f} MB"
                         f" {r['latency_ms_projected']:8.3f} ms/tok proj")
            print(line)
    comp.restore()

    out = args.out or os.path.join(run_path or ".", "data", "results.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cols = write_results(out, rows)
    print(f"\nwrote {out}  ({len(cols)} columns, {len(rows)} rows)")

    if ref is not None and front:
        print(f"\nruntime over {len(front)} front members")
        print("  MEASURED (expect no spread -- simulated compression, one graph)")
        for key in ("measured_latency_ms", "measured_peak_alloc_mb"):
            text = bench.summarize_spread(
                front, key, ref if key.endswith("latency_ms") else None)
            print("    " + text.replace("\n    ", "\n      "))
        print("  PROJECTED (this is what a results table should carry)")
        for key in ("latency_ms_projected", "peak_mb_projected"):
            print("    " + bench.summarize_spread(front, key, None))
        best = min(front, key=lambda r: r["peak_mb_projected"])
        print(f"  smallest projected footprint: {best['tag']} at "
              f"{best['peak_mb_projected']:.0f} MB "
              f"({ref.peak_alloc_mb / max(best['peak_mb_projected'], 1e-9):.2f}x "
              f"under fp16), ppl_eval {best['ppl_eval']:.3f}")
        print("  Projections assume a kernel that is perfectly weight-bandwidth "
              "bound and add no\n  dequant cost. They bound the available "
              "speedup; they are not measurements.")

    if not front:
        return

    # -- does the proxy rank candidates the way the full metric does? --------
    calib = [r["ppl_calib"] for r in front]
    evald = [r["ppl_eval"] for r in front]
    avg_bits = [r["avg_bits"] for r in front]
    rho = rank_correlation(calib, evald)

    gap = [e / c for e, c in zip(evald, calib) if c > 0]
    print(f"\nproxy vs held-out over {len(front)} front members")
    print(f"  Spearman rho          {rho:+.4f}", end="  ")
    print("(>0.9 means the proxy is a valid surrogate)"
          if rho >= 0.9 else
          "-- BELOW 0.9: the search partly optimized noise; raise data.n_proxy_seq")
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
        front=list(zip(avg_bits, evald)), baseline=baselines or None,
        fp16=fp16_eval, corpus=f"{eval_name} (held-out)")
    written += plot_calib_vs_eval(
        os.path.join(fig_dir, "front_calib_vs_eval"), cfg,
        avg_bits=avg_bits, ppl_calib=calib, ppl_eval=evald,
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
