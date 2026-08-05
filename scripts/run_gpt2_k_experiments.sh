#!/usr/bin/env bash
# GPT-2 codebook-size sweep: global vs block-wise vs layer-wise K.
#
# For each config in turn: run the search, then immediately evaluate that
# run's Pareto front on the held-out corpus, then move to the next config.
# Doing the eval straight away means a crash later still leaves you with
# complete results for everything finished so far.
#
#   bash scripts/run_gpt2_k_experiments.sh
#   bash scripts/run_gpt2_k_experiments.sh --n-gen 5 --pop 20   # quick trial
#
# Extra arguments are forwarded to run_search.py.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/gpt2_k_global.yaml
  configs/gpt2_k_block.yaml
  configs/gpt2_k_layer.yaml
)

python3 - <<'PY'
try:
    import datasets  # noqa: F401
except ImportError:
    raise SystemExit(
        "\nThese configs read real corpora (C4 for calibration, WikiText-2 for\n"
        "evaluation), which needs the `datasets` package:\n\n"
        "    pip install datasets\n\n"
        "The first run downloads and tokenizes them into .cache/evolmc/; later\n"
        "runs reuse that cache.\n"
    )
PY

RUN_DIRS=()
STAMP=$(date +%Y%m%d-%H%M%S)
DIR_FILE=$(mktemp)
trap 'rm -f "$DIR_FILE"' EXIT

echo "=============================================================="
echo " GPT-2 K sweep: 3 experiments, K in [2, 8192], no pruning"
echo " started $(date)"
echo "=============================================================="

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "--------------------------------------------------------------"
  echo ">>> SEARCH  $cfg"
  echo "--------------------------------------------------------------"
  # --emit-run-dir reports where the run actually landed. Reading it back
  # beats guessing from log.run_name: a rerun takes the next free -2/-3
  # suffix, and evaluating the previous run's front would pass silently.
  python3 scripts/run_search.py "$cfg" --emit-run-dir "$DIR_FILE" "$@"
  run_dir=$(cat "$DIR_FILE")
  RUN_DIRS+=("$run_dir")

  echo
  echo "--------------------------------------------------------------"
  echo ">>> EVAL    $run_dir"
  echo "--------------------------------------------------------------"
  python3 scripts/run_eval.py "$cfg" "$run_dir"
done

echo
echo "=============================================================="
echo " all three finished $(date)"
echo "=============================================================="
for d in "${RUN_DIRS[@]}"; do echo "   $d"; done
echo
echo " compare them with:"
echo "   python3 scripts/compare_runs.py \\"
for i in "${!RUN_DIRS[@]}"; do
  sep=$([ "$i" -lt $(( ${#RUN_DIRS[@]} - 1 )) ] && echo " \\" || echo " \\")
  echo "     $(basename "${RUN_DIRS[$i]}")$sep"
done
echo "     --labels 'global (1 var),block-wise (12 var),layer-wise (48 var)' \\"
echo "     --name granularity-$STAMP --bpw 2,3,4,6,8"
