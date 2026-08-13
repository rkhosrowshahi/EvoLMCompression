#!/usr/bin/env bash
# GPT-2 proj_w_lm_head 2-obj runs: NUQ companding (global/block/layer), matching
# NUQ+prune bitmap, then unmatched prune grouping.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/nuq_companding/gpt2_124m/gpt2_124m-proj_w_lm_head-global_quant-2obj.yaml
  configs/nuq_companding/gpt2_124m/gpt2_124m-proj_w_lm_head-block_quant-2obj.yaml
  configs/nuq_companding/gpt2_124m/gpt2_124m-proj_w_lm_head-layer_quant-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-proj_w_lm_head-global_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-proj_w_lm_head-block_quant-block_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-proj_w_lm_head-layer_quant-layer_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-proj_w_lm_head-block_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning/gpt2_124m/gpt2_124m-proj_w_lm_head-layer_quant-global_prune_sigma-bitmap-2obj.yaml
)

for cfg in "${CONFIGS[@]}"; do
  python3 scripts/run_search.py "$cfg" "$@"
done
