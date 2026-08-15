#!/usr/bin/env bash
# All 16 GPT-2 NUQ companding runs with the widened clip gene, alpha in [2, 64]:
# the 8 only_proj configs first, then the 8 proj_w_lm_head ones.
#
# Why the wider range: sweeping alpha against measured PPL (presentation/sweep_alpha_ppl.py)
# put the optimum at alpha = 12-48 for every K, never inside the old [2, 6]. Pinned at 6, PPL
# flattens near 29.5 however large K grows -- which is where the published companding fronts
# stop. Widening leaves n_var unchanged, so this is the same search space and the same
# 10,000-evaluation budget; only the reachable alpha values differ.
#
#   scripts/gpt2_nuq_companding_a64_only_proj_then_lm_head.sh            # all 16
#   scripts/gpt2_nuq_companding_a64_only_proj_then_lm_head.sh --dry-run  # passed through

set -euo pipefail
cd "$(dirname "$0")/.."

ONLY_PROJ=(
  configs/nuq_companding_a64/gpt2_124m/gpt2_124m-only_proj-global_quant-2obj.yaml
  configs/nuq_companding_a64/gpt2_124m/gpt2_124m-only_proj-block_quant-2obj.yaml
  configs/nuq_companding_a64/gpt2_124m/gpt2_124m-only_proj-layer_quant-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-global_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-block_quant-block_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-layer_quant-layer_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-block_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-only_proj-layer_quant-global_prune_sigma-bitmap-2obj.yaml
)

PROJ_W_LM_HEAD=(
  configs/nuq_companding_a64/gpt2_124m/gpt2_124m-proj_w_lm_head-global_quant-2obj.yaml
  configs/nuq_companding_a64/gpt2_124m/gpt2_124m-proj_w_lm_head-block_quant-2obj.yaml
  configs/nuq_companding_a64/gpt2_124m/gpt2_124m-proj_w_lm_head-layer_quant-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-proj_w_lm_head-global_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-proj_w_lm_head-block_quant-block_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-proj_w_lm_head-layer_quant-layer_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-proj_w_lm_head-block_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a64/gpt2_124m/gpt2_124m-proj_w_lm_head-layer_quant-global_prune_sigma-bitmap-2obj.yaml
)

CONFIGS=("${ONLY_PROJ[@]}" "${PROJ_W_LM_HEAD[@]}")

# Fail before burning GPU hours if any path is wrong.
for cfg in "${CONFIGS[@]}"; do
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 1; }
done

total=${#CONFIGS[@]}
echo "== ${total} runs: ${#ONLY_PROJ[@]} only_proj, then ${#PROJ_W_LM_HEAD[@]} proj_w_lm_head =="
i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i + 1))
  echo
  echo "== [${i}/${total}] $(basename "$cfg") =="
  scripts/search_then_eval.sh "$cfg" "$@"
done
echo
echo "== all ${total} runs finished =="
