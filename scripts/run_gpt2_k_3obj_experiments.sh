#!/usr/bin/env bash
# GPT-2 codebook-size sweep, THREE objectives: global vs block-wise vs layer-wise K.
#
#   f1  ppl_proxy    proxy perplexity                            (min)
#   f2  bpw_target   deployable bits per weight, target matrices (min)
#   f3  cr_archival  archival compression ratio, full checkpoint  (MAX)
#
# f2 and f3 are drawn from DIFFERENT bit totals -- fixed-width indices versus
# entropy-coded ones -- which is what makes the third axis carry information.
# Pairing two measures from the same total (e.g. bpw_model with cr_deployable,
# where cr_deployable is exactly 16/bpw_model) would reproduce the 2-objective
# front exactly. The search warns about that at startup if you try it.
#
# For each config in turn: run the search, then immediately evaluate that run's
# Pareto front on the held-out corpus, then move to the next config. Doing the
# eval straight away means a crash later still leaves you with complete results
# for everything finished so far.
#
#   bash scripts/run_gpt2_k_3obj_experiments.sh
#   bash scripts/run_gpt2_k_3obj_experiments.sh --n-gen 5 --pop 20   # quick trial
#
# Extra arguments are forwarded to run_search.py.
#
# Each run pairs with its 2-objective counterpart (gpt2_k_<grouping>.yaml),
# which differs only in log.run_name and search.objectives. Compare a pair by
# eye rather than with compare_runs.py: hypervolume over two objectives and
# over three are volumes in different spaces, and the script refuses to put
# them in one table for exactly that reason.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/gpt2_k_global_3obj.yaml
  configs/gpt2_k_block_3obj.yaml
  configs/gpt2_k_layer_3obj.yaml
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
echo " GPT-2 K sweep, 3 objectives: ppl / bpw_target / cr_archival"
echo " 3 experiments, K in [2, 8192], no pruning"
echo " started $(date)"
echo "=============================================================="
echo
echo " Expect the third objective to do NOTHING at global (one K is a"
echo " one-parameter family, so archival cost is a deterministic monotone"
echo " function of it), something at block, and most at layer. That"
echo " progression is the result; global is the control."

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
echo " compare the three 3-objective runs with each other:"
echo "   python3 scripts/compare_runs.py \\"
for i in "${!RUN_DIRS[@]}"; do
  echo "     $(basename "${RUN_DIRS[$i]}") \\"
done
echo "     --labels 'global (1 var),block-wise (12 var),layer-wise (48 var)' \\"
echo "     --name granularity-3obj-$STAMP --bpw 2,3,4,6,8"
echo
echo " the 2-objective vs 3-objective question is answered per grouping, by"
echo " checking whether the 3-objective front spreads in cr_archival at"
echo " matched bpw_target -- not by comparing hypervolumes across the two."
