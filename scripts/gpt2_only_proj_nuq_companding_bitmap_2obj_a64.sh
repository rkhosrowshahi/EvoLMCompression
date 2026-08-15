#!/usr/bin/env bash
# Same 8 only_proj NUQ companding runs as gpt2_only_proj_nuq_companding_bitmap_2obj.sh,
# but with the clip gene widened to alpha in [2, 64] instead of [2, 6].
#
# Motivation: sweeping alpha against measured PPL (presentation/sweep_alpha_ppl.py) found the
# optimum at alpha = 12-48 for every K -- never inside [2, 6]. Pinned at the old ceiling, PPL
# flattens at ~29.5 no matter how large K grows, which matches where the published companding
# fronts stop. Widening the range leaves n_var unchanged, so this is the same search space
# dimensionality and the same 10,000-evaluation budget; only the reachable alpha values differ.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/nuq_companding_a64/gpt2_124m/gpt2_124m-only_proj-global_quant-2obj.yaml
  configs/nuq_companding_a64/gpt2_124m/gpt2_124m-only_proj-block_quant-2obj.yaml
  configs/nuq_companding_a64/gpt2_124m/gpt2_124m-only_proj-layer_quant-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-global_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-block_quant-block_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-layer_quant-layer_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-block_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-layer_quant-global_prune_sigma-bitmap-2obj.yaml
)

for cfg in "${CONFIGS[@]}"; do
  scripts/search_then_eval.sh "$cfg" "$@"
done
