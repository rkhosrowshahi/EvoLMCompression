#!/usr/bin/env bash
# All 10 GPT-2 NUQ companding runs at alpha in [12, 48] (the sweep_alpha_ppl.py optimum,
# see gpt2_nuq_companding_a64.sh), with K free over [2, 1024] integers and SBX/mutation
# both at prob=0.9 (mutation_prob_var=0.9 too): the 5 only_proj configs first, then the
# 5 proj_w_lm_head ones. Global and block K grouping only -- no layer grouping.
#
#   scripts/gpt2_nuq_companding_a12_48.sh            # all 10
#   scripts/gpt2_nuq_companding_a12_48.sh --dry-run  # passed through

set -euo pipefail
cd "$(dirname "$0")/.."

ONLY_PROJ=(
  configs/nuq_companding_a12_48/gpt2_124m/gpt2_124m-only_proj-global_quant-2obj.yaml
  configs/nuq_companding_a12_48/gpt2_124m/gpt2_124m-only_proj-block_quant-2obj.yaml
  configs/nuq_companding_pruning_a12_48/gpt2_124m/gpt2_124m-only_proj-global_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a12_48/gpt2_124m/gpt2_124m-only_proj-block_quant-block_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a12_48/gpt2_124m/gpt2_124m-only_proj-block_quant-global_prune_sigma-bitmap-2obj.yaml
)

PROJ_W_LM_HEAD=(
  configs/nuq_companding_a12_48/gpt2_124m/gpt2_124m-proj_w_lm_head-global_quant-2obj.yaml
  configs/nuq_companding_a12_48/gpt2_124m/gpt2_124m-proj_w_lm_head-block_quant-2obj.yaml
  configs/nuq_companding_pruning_a12_48/gpt2_124m/gpt2_124m-proj_w_lm_head-global_quant-global_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a12_48/gpt2_124m/gpt2_124m-proj_w_lm_head-block_quant-block_prune_sigma-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a12_48/gpt2_124m/gpt2_124m-proj_w_lm_head-block_quant-global_prune_sigma-bitmap-2obj.yaml
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
