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

# Six runs: three groupings x pruning off/on. Each pruned config differs from
# its unpruned pair in exactly ONE effective setting, prune.enabled, so the
# pair isolates what sparsity buys.
#
#   bash scripts/run_gpt2_k_3obj_experiments.sh --only noprune
#   bash scripts/run_gpt2_k_3obj_experiments.sh --only prune
#
# Anything else is forwarded to run_search.py.
NOPRUNE=(
  configs/gpt2_k_global_3obj.yaml
  configs/gpt2_k_block_3obj.yaml
  configs/gpt2_k_layer_3obj.yaml
)
PRUNE=(
  configs/gpt2_k_global_3obj_prune.yaml
  configs/gpt2_k_block_3obj_prune.yaml
  configs/gpt2_k_layer_3obj_prune.yaml
)

ONLY=all
if [ "${1:-}" = "--only" ]; then
  ONLY="${2:-all}"
  shift 2
fi
case "$ONLY" in
  all)     CONFIGS=("${NOPRUNE[@]}" "${PRUNE[@]}") ;;
  noprune) CONFIGS=("${NOPRUNE[@]}") ;;
  prune)   CONFIGS=("${PRUNE[@]}") ;;
  *) echo "--only takes: all, noprune, prune (got '$ONLY')" >&2; exit 2 ;;
esac

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
echo " ${#CONFIGS[@]} experiments (--only $ONLY), K in [2, 8192]"
echo " started $(date)"
echo "=============================================================="
echo
echo " WITHOUT pruning, f2 and f3 separate only through how K is spread"
echo " between groups. So expect the third objective to do NOTHING at"
echo " global (one K is a one-parameter family), something at block, and"
echo " most at layer. Global is the control for that progression."
echo
echo " WITH pruning, t_lo/t_hi are invisible to f2 by construction -- the"
echo " reserved zero codeword keeps the index width at ceil(log2 K) no"
echo " matter how much is pruned -- and fully visible to f3. That is a"
echo " source of separation orthogonal to K, so even the global config"
echo " should open a third dimension. If it does not, the separation"
echo " depends on distributing K rather than on sparsity."

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
