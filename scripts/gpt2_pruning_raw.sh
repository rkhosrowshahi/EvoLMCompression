#!/usr/bin/env bash
# All 4 raw-threshold pruning runs: UQ and NUQ (companding, alpha in [12,48]),
# Global and Block K-granularity, Global-granularity pruning throughout.
#
# `prune.mode: raw` uses an absolute weight-value threshold instead of sigma's
# per-row-rescaled one -- see the configs' own comments (t_max=1.0, chosen from
# the only_proj target set's weight-magnitude percentiles) for the rationale.
# Same NSGA-II operator settings as the alpha in [12,48] companding configs
# throughout (crossover 0.9/1.0, mutation 0.9/0.9).
#
#   scripts/gpt2_pruning_raw.sh            # all 4
#   scripts/gpt2_pruning_raw.sh --dry-run  # passed through

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/uq_pruning_raw/gpt2_124m/gpt2_124m-global_quant-global_prune_raw-bitmap-2obj.yaml
  configs/uq_pruning_raw/gpt2_124m/gpt2_124m-block_quant-global_prune_raw-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a12_48_raw/gpt2_124m/gpt2_124m-only_proj-global_quant-global_prune_raw-bitmap-2obj.yaml
  configs/nuq_companding_pruning_a12_48_raw/gpt2_124m/gpt2_124m-only_proj-block_quant-global_prune_raw-bitmap-2obj.yaml
)

# Fail before burning GPU hours if any path is wrong.
for cfg in "${CONFIGS[@]}"; do
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 1; }
done

total=${#CONFIGS[@]}
echo "== ${total} raw-pruning runs =="
i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i + 1))
  echo
  echo "== [${i}/${total}] $(basename "$cfg") =="
  scripts/search_then_eval.sh "$cfg" "$@"
done
echo
echo "== all ${total} runs finished =="
