#!/usr/bin/env bash
# GPT-2 only_proj 2-obj runs: NUQ companding (global/block/layer), matching
# NUQ+prune bitmap, then unmatched prune grouping.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/nuq_companding/gpt2_124m/gpt2_124m-only_proj-global_quant-2obj.yaml
  configs/nuq_companding/gpt2_124m/gpt2_124m-only_proj-block_quant-2obj.yaml
  configs/nuq_companding/gpt2_124m/gpt2_124m-only_proj-layer_quant-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-only_proj-global_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-only_proj-block_quant-block_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-only_proj-layer_quant-layer_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-only_proj-block_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-only_proj-layer_quant-global_prune_sigma-bitmap-2obj.yaml
)

for cfg in "${CONFIGS[@]}"; do
  python3 scripts/run_search.py "$cfg" "$@"
done
