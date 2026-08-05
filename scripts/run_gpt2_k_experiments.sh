#!/usr/bin/env bash
# GPT-2 codebook-size sweep: global vs block-wise vs layer-wise K.
#
# Runs the three configs one after another, then evaluates each resulting
# Pareto front on the full WikiText-2 split. The three configs differ only in
# variables.k_grouping, so the fronts are directly comparable.
#
#   bash scripts/run_gpt2_k_experiments.sh
#   bash scripts/run_gpt2_k_experiments.sh --n-gen 10       # quick trial
#
# Any extra arguments are forwarded to run_search.py.

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

echo "=============================================================="
echo " GPT-2 K sweep: 3 experiments, K in [2, 8192], no pruning"
echo "=============================================================="
echo

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "--------------------------------------------------------------"
  echo ">>> SEARCH  $cfg"
  echo "--------------------------------------------------------------"
  python3 scripts/run_search.py "$cfg" "$@"
done

for cfg in "${CONFIGS[@]}"; do
  run_name=$(python3 -c "
import sys, yaml
print(yaml.safe_load(open('$cfg'))['log']['run_name'])")
  echo
  echo "--------------------------------------------------------------"
  echo ">>> FULL EVAL  $run_name"
  echo "--------------------------------------------------------------"
  python3 scripts/eval_front.py "$cfg" "logs/$run_name"
done

echo
echo "=============================================================="
echo " done. compare the three runs with:"
echo "   python3 scripts/compare_runs.py gpt2-k-global gpt2-k-block gpt2-k-layer"
echo "=============================================================="
