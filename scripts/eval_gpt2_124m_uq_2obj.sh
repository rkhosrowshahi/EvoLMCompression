#!/usr/bin/env bash
# Re-evaluate the already-finished GPT-2 124M UQ 2-obj fronts (global/block/
# layer quant, no prune / global prune / matched-granularity prune) on the
# true held-out corpus. Search already ran for these; do NOT re-run it here.
#
# Each run directory carries its own resolved config.yaml, saved by
# run_search.py at the time it ran, so we read that instead of the source
# config under configs/ (which may since have been renamed or removed).
#
# Writes logs/<run>/data/results.csv (ppl_eval on WikiText-2, ppl_calib on
# C4, full cost accounting) plus the front_eval / front_calib_vs_eval /
# proxy_vs_eval figures, per run_eval.py.

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=(
  logs/gpt2_124m-block_quant-2obj-np100-ng100
  logs/gpt2_124m-block_quant-block_prune_sigma-bitmap-2obj-np100-ng100
  logs/gpt2_124m-block_quant-global_prune_sigma-bitmap-2obj-np100-ng100
  logs/gpt2_124m-global_quant-2obj-np100-ng100
  logs/gpt2_124m-global_quant-global_prune_sigma-bitmap-2obj-np100-ng100
  logs/gpt2_124m-layer_quant-2obj-np100-ng100
  logs/gpt2_124m-layer_quant-global_prune_sigma-bitmap-2obj-np100-ng100
  logs/gpt2_124m-layer_quant-layer_prune_sigma-bitmap-2obj-np100-ng100
)

for run in "${RUNS[@]}"; do
  echo "== $run =="
  python3 scripts/run_eval.py "$run/config.yaml" "$run" "$@"
done
