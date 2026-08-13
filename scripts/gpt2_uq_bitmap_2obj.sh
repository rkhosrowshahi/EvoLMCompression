#!/usr/bin/env bash
# GPT-2 2-obj runs: UQ (global/block/layer), matching UQ+prune bitmap,
# then unmatched prune grouping.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/uq/gpt2_124m/gpt2_124m-global_quant-2obj.yaml
  configs/uq/gpt2_124m/gpt2_124m-block_quant-2obj.yaml
  configs/uq/gpt2_124m/gpt2_124m-layer_quant-2obj.yaml
  configs/uq_pruning/gpt2_124m/gpt2_124m-global_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/uq_pruning/gpt2_124m/gpt2_124m-block_quant-block_prune_sigma-bitmap-2obj.yaml
  configs/uq_pruning/gpt2_124m/gpt2_124m-layer_quant-layer_prune_sigma-bitmap-2obj.yaml
  configs/uq_pruning/gpt2_124m/gpt2_124m-block_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/uq_pruning/gpt2_124m/gpt2_124m-layer_quant-global_prune_sigma-bitmap-2obj.yaml
)

for cfg in "${CONFIGS[@]}"; do
  python3 scripts/run_search.py "$cfg" "$@"
done
