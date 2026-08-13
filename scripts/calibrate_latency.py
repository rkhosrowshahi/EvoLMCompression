#!/usr/bin/env python3
"""Fit the latency-proxy coefficients once and freeze them to a file.

    python3 scripts/calibrate_latency.py configs/gpt2_3obj_block_core_uq.yaml
    python3 scripts/calibrate_latency.py <config> --out logs/latency_coeffs.json
    python3 scripts/calibrate_latency.py <config> --show     # print, write nothing

Run this ONCE per (GPU, model, target set) before a sweep. Every run then loads
the same frozen file, which is what makes the latency axis comparable across the
twelve cells: refitting per run would compare numbers taken on a differently
loaded GPU. `run_search.py` fits automatically if the file is missing, so this
script is for doing it deliberately, on an idle GPU, and inspecting the result.

WHAT IS MEASURED HERE, on the real device:
    beta   achieved fp16 streaming bandwidth, and
    tau    per-kernel launch overhead,
           both from ONE least-squares fit of T = a*B + d over real batch-1
           GEMV shapes, so they are mutually consistent
    phi    achieved fp16 throughput, from a compute-bound square matmul

WHAT IS STATED, from the `latency:` block of the config:
    the efficiency factors for the LUT and sparse kernel classes, the dequant
    op count, and the kernels-per-layer counts. Those kernels do not exist in
    this project, so they cannot be benchmarked. They are written into the file
    under `provenance` and must be quoted with any latency number.

The target set is part of the fit: layer geometry is baked in, so a file fitted
for `core` will not be reused for `full`. `load_or_calibrate` checks that and
refits rather than silently skipping layers it has no geometry for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolmc import Compressor, Config  # noqa: E402
from evolmc import latency as latency_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=None,
                    help="output path; default is latency.coeffs_path")
    ap.add_argument("--show", action="store_true",
                    help="print the fit and exit without writing")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    out = args.out or cfg.latency.coeffs_path

    comp = Compressor(cfg)
    print(comp.summary())
    print()

    proxy = latency_mod.calibrate(comp, cfg)
    print(proxy.describe())
    print()
    print("provenance")
    for k, v in proxy.provenance.items():
        print(f"  {k:<32} {v}")

    # A sanity sweep: what the proxy says at uniform K, end to end. If the fp16
    # row is not close to a real measurement the constants are wrong, and it is
    # much cheaper to notice that here than after a 10,000-evaluation search.
    fp16 = proxy.predict_fp16()
    print(f"\nproxy at uniform K (ms/token).  fp16 baseline {fp16:.4f} ms/token")
    print(f"  {'K':>6}  {'avg_bits':>14}  {'latency':>9}  {'speedup':>8}")
    for k in comp.genome.k_choices:
        cand = comp.apply(comp.genome.encode_uniform(k))
        t = proxy.predict(cand.cost)
        cs = cand.cost.summary()
        print(f"  {k:>6}  {cs['avg_bits']:>14.3f}  {t:>9.4f}  "
              f"{fp16 / max(t, 1e-9):>7.2f}x")
    comp.restore()

    # Where the time actually goes. At GPT-2's size this is dominated by kernel
    # launches (48 small layers), which is worth seeing before trusting f3.
    cand = comp.apply(comp.genome.encode_uniform(min(comp.genome.k_choices)))
    kls = proxy.classes.get(cfg.quant.deployable_format, proxy.classes["dense"])
    launch = len(cand.cost.layers) * kls.n_kernels * kls.launch_ms
    total = proxy.predict(cand.cost)
    print(f"\nbreakdown at the cheapest K ({total:.4f} ms/token)")
    print(f"  roofline  {total - launch - proxy.fixed_ms:8.4f} ms  "
          f"{100 * (total - launch - proxy.fixed_ms) / total:5.1f}%")
    print(f"  launches  {launch:8.4f} ms  {100 * launch / total:5.1f}%  "
          f"({len(cand.cost.layers)} layers x {kls.n_kernels} x "
          f"{kls.launch_ms * 1000:.1f}us)")
    print(f"  fixed     {proxy.fixed_ms:8.4f} ms  "
          f"{100 * proxy.fixed_ms / total:5.1f}%")
    comp.restore()

    if args.show:
        print("\n--show: nothing written")
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    proxy.save(out)
    size = os.path.getsize(out)
    print(f"\nwrote {out}  ({size:,} bytes, {len(proxy.geometry)} layers)")
    print("every run pointing at this file now shares one latency axis")


if __name__ == "__main__":
    main()
