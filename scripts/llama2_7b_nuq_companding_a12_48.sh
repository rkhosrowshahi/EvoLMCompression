#!/usr/bin/env bash
# Llama-2-7B NUQ companding at alpha in [12, 48] (the sweep_alpha_ppl.py optimum, see
# gpt2_nuq_companding_a64.sh), global K grouping, K free over [2, 1024] integers, 200
# generations of NSGA-II at pop 100. Two runs: quantization only, then the same search
# with global sigma pruning on a bitmap deployable format.
#
#   scripts/llama2_7b_nuq_companding_a12_48.sh            # both
#   scripts/llama2_7b_nuq_companding_a12_48.sh --dry-run  # passed through

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/nuq_companding_a12_48/llama2_7b/llama2_7b-only_proj-global_quant-2obj.yaml
  configs/nuq_companding_pruning_a12_48/llama2_7b/llama2_7b-only_proj-global_quant-global_prune_sigma-bitmap-2obj.yaml
)

# Fail before burning GPU hours if any path is wrong.
for cfg in "${CONFIGS[@]}"; do
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 1; }
done

total=${#CONFIGS[@]}
echo "== ${total} Llama-2-7B runs: no-prune, then global sigma prune =="
i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i + 1))
  echo
  echo "== [${i}/${total}] $(basename "$cfg") =="
  scripts/search_then_eval.sh "$cfg" "$@"
done
echo
echo "== all ${total} runs finished =="
