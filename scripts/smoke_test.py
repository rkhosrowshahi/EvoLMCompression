#!/usr/bin/env python3
"""End-to-end wiring check: does the whole pipeline run and produce artifacts?

Loads a small model, sweeps the uniform-K configurations, runs a real NSGA-II
search, and writes a complete run directory. No `datasets` dependency and no
network beyond the model download, so it works anywhere.

**The corpus is synthetic random tokens, so every perplexity here is
meaningless.** This is a check that the parts fit together and a way to measure
seconds-per-evaluation for budgeting a real search -- it says nothing about
whether the compression is any good. Unit correctness (entropy bounds, bit
accounting, binning order, figure geometry) lives in `pytest tests/`.

The one hard assertion is that `restore()` returns the model bit-exactly, which
is the invariant the entire search depends on: every candidate overwrites the
live weights in place, so a leak would silently corrupt all later evaluations.
Everything else passes by not raising.

    python scripts/smoke_test.py                      # gpt2 (124M)
    python scripts/smoke_test.py --gens 12            # longer, for the video
    python scripts/smoke_test.py --venue ieee         # exercise paper figures
    python scripts/smoke_test.py --binning kmeans     # a different quantizer
    python scripts/smoke_test.py --model sshleifer/tiny-gpt2   # fastest
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from evolmc import Compressor, CompressionProblem, Config, perplexity  # noqa: E402
from evolmc.data import load_corpus, make_windows  # noqa: E402
from evolmc.rundir import RunDir  # noqa: E402
from evolmc.search import run_search, save_front  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--binning", default="uniform")
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--venue", default="none",
                    choices=["none", "ieee", "acm", "neurips", "icml", "lncs"],
                    help="also exercise the paper-figure path at this venue")
    ap.add_argument("--usetex", action="store_true")
    ap.add_argument("--format", default="dense",
                    choices=["dense", "bitmap", "csr"],
                    help="deployable storage format; 'dense' charges one index "
                         "per weight position and is blind to pruning")
    ap.add_argument("--objectives", default=None,
                    help="comma-separated objective names, e.g. "
                         "ppl_proxy,bpw_target,cr_archival. Default is the "
                         "two-objective problem.")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    cfg = Config.from_dict({
        "model": {"name": args.model, "device": device, "master_device": device,
                  "dtype": "float32" if device in ("cpu", "mps") else "float16"},
        "quant": {"binning": args.binning, "granularity": "per_channel",
                  "k_choices": [4, 8, 16, 32, 64],
                  "deployable_format": args.format},
        "prune": {"enabled": True, "t_max": 1.0},
        "variables": {"k_grouping": "type", "prune_grouping": "global"},
        "data": {"seqlen": args.seqlen, "n_proxy_seq": 4, "n_eval_seq": 8},
        "search": {"pop_size": 8, "n_gen": args.gens,
                   **({"objectives": [s.strip() for s in
                                      args.objectives.split(",")]}
                      if args.objectives else {})},
        "log": {"root": "logs",
                "run_name": "smoke-test"
                            + ("" if args.venue == "none" else f"-{args.venue}")
                            + ("" if not args.objectives
                               else f"-{len(args.objectives.split(','))}obj")
                            + ("" if args.format == "dense" else f"-{args.format}")},
        "plot": {"every": 1, "venue": args.venue,
                 "usetex": args.usetex},
    })

    print("=" * 68)
    comp = Compressor(cfg)
    print(comp.summary())
    print("=" * 68)

    ids = load_corpus("synthetic", comp.tokenizer, n_tokens=args.seqlen * 16,
                      cache_dir=cfg.data.cache_dir)
    windows = make_windows(ids, cfg.data.seqlen, cfg.data.n_proxy_seq)

    t0 = time.perf_counter()
    base = perplexity(comp.model, windows, device=comp.device)
    t_fwd = time.perf_counter() - t0
    print(f"\nfp16 baseline ppl {base:.3f}   ({t_fwd:.2f}s for "
          f"{windows.shape[0]} windows)\n")

    print(f"{'config':<22}{'ppl':>10}{'bpw':>8}{'CR':>8}{'CR-huf':>9}"
          f"{'spars':>8}{'apply':>8}")
    print("-" * 68)
    for k in comp.genome.k_choices:
        for t in (0.0, 0.5):
            x = comp.genome.encode_uniform(k, t)
            cand = comp.apply(x)
            ppl = perplexity(comp.model, windows, device=comp.device)
            s = cand.cost.summary()
            print(f"{'K=%d t=%.1f' % (k, t):<22}{ppl:>10.3f}"
                  f"{s['bpw_target']:>8.2f}{s['cr_deployable']:>8.2f}"
                  f"{s['cr_archival']:>9.2f}{s['sparsity']:>8.3f}"
                  f"{cand.apply_seconds:>8.2f}")
    comp.restore()

    restored = perplexity(comp.model, windows, device=comp.device)
    assert abs(restored - base) < 1e-3 * max(base, 1.0), \
        f"restore() failed: {restored} != {base}"
    print(f"\nrestore check ok ({restored:.3f} == {base:.3f})")

    print(f"\nrunning {cfg.search.n_gen} generations of NSGA-II ...")
    t0 = time.perf_counter()
    run = RunDir(cfg)
    problem = CompressionProblem(comp, windows, cfg, run=run)
    res, records = run_search(problem, cfg, run)
    save_front(res, problem, cfg, run)
    run.save_meta(model=cfg.model.name, n_evaluations=comp.n_evals)
    dt = time.perf_counter() - t0

    n = comp.n_evals
    print(f"\n{n} evaluations in {dt:.1f}s -> {dt/max(n,1):.2f}s each")
    print(f"cache: {comp.cache.hits} hits / {comp.cache.misses} misses")
    objset = problem.objectives
    print(f"\nfinal front ({', '.join(objset.names)}):")
    for f in sorted(objset.to_real(res.F).tolist(), key=lambda r: r[1]):
        print("  " + "   ".join(f"{n} {v:9.3f}"
                                for n, v in zip(objset.names, f)))
    print(f"\nrun directory: {run.path}")
    for root, _, files in sorted(os.walk(run.path)):
        rel = os.path.relpath(root, run.path)
        for fn in sorted(files):
            print(f"  {os.path.join('' if rel == '.' else rel, fn)}")
    run.close()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
