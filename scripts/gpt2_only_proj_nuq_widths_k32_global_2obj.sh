#!/usr/bin/env bash
# GPT-2 only_proj 2-obj runs: NUQ widths, K fixed at 32, global K-grouping
# only -- quant-only, then matching quant+prune bitmap.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/nuq_widths_k32/gpt2_124m/gpt2_124m-only_proj-global_quant-2obj.yaml
  configs/nuq_widths_pruning_k32/gpt2_124m/gpt2_124m-only_proj-global_quant-global_prune_sigma-bitmap-2obj.yaml
)

for cfg in "${CONFIGS[@]}"; do
  scripts/search_then_eval.sh "$cfg" "$@"
done
