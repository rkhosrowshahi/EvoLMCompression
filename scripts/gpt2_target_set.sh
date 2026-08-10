#!/usr/bin/env bash
# GPT-2 target-set comparison: 3 K groupings x 3 target sets x 2 methods.
#
# THE QUESTION. How much of the model should the search be allowed to compress?
#
#   no_head     transformer blocks only; LM head, embeddings, norms, biases fp16
#   with_head   + the tied LM head / token-embedding matrix
#   all         + the positional table; only norms and biases left fp16
#
# crossed with global / block-wise / layer-wise K, and with the method:
#
#   uq          uniform quantization only       (dense format, pruning off)
#   uq_pruning  uniform quantization + pruning  (bitmap format)
#
# 18 runs on one budget. `uq` is the ablation for `uq_pruning`: identical in
# every other respect, so the pair compared at matched bpw_model isolates what
# sparsity actually buys. Within one method, two lines differ per cell:
# variables.k_grouping and the model target-set block.
#
#   objectives:  f1 ppl_proxy   proxy perplexity          (min)
#                f2 bpw_model   effective bits per weight (min)  WHOLE checkpoint
#
# f2 is bpw_model, not bpw_target, and that is what makes the comparison mean
# anything: bpw_target divides by whatever is in the target set, so all three
# cells would span 1-16 bits and the difference would vanish into the
# denominator. What the three cells actually differ by:
#
#                      no_head      with_head           all
#   target layers          48             49             50
#   target weights     84,934,656    123,532,032    124,318,464
#   untouched (fp16)   39,505,152        907,776        121,344
#   f2 floor            5.762 bpw      1.109 bpw      1.015 bpw
#   CR ceiling             2.78x         14.42x         15.77x
#
# Note the step from no_head to with_head is enormous and with_head to all is
# small: almost all of no_head's untouched mass is the tied 38.6M-weight
# token-embedding matrix. So the interesting question is not whether `all` beats
# `with_head` on size -- it barely can -- but whether either pays for the reach
# in PERPLEXITY. The LM head produces the
# logits directly, so quantization error there is not attenuated downstream.
#
# LATENCY AND MEMORY are measured AFTER each search, on the front, by the EVAL
# step below. Latency follows SqueezeLLM's protocol (per-token decode loop
# carrying past_key_values, median per-token time); memory is reported in bytes
# as well as MB. Neither is a search objective: compression here is simulated,
# so every candidate runs the same dense fp16 graph and a stopwatch cannot tell
# them apart. run_eval prints the measured spread against the fp16 noise band so
# that is visible rather than assumed.
#
#   bash scripts/gpt2_target_set.sh                      # all 18
#   bash scripts/gpt2_target_set.sh --only uq_pruning    # one method      (9)
#   bash scripts/gpt2_target_set.sh --only with_head     # one target set  (6)
#   bash scripts/gpt2_target_set.sh --only layer         # one grouping    (6)
#   bash scripts/gpt2_target_set.sh --n-gen 3 --pop 8    # smoke trial
#
# --only must come FIRST; it is parsed here and shifted off. Everything else is
# forwarded to run_search.py, so --n-gen overrides the YAML without editing it.
#
# BUDGET. 100 x 500 = 50,000 evaluations per run, EIGHTEEN runs. n_proxy_seq is 128
# against 8 in the earlier sweeps, so the proxy forward pass costs 16x what it
# did and is essentially the whole budget. Time one run before committing the
# server to all eighteen, and use --only to split across jobs. For just the
# pruning arm -- the nine runs originally scoped -- use --only uq_pruning.

set -euo pipefail
cd "$(dirname "$0")/.."

UQ=(
  configs/gpt2_scope_global_no_head_uq.yaml
  configs/gpt2_scope_global_with_head_uq.yaml
  configs/gpt2_scope_global_all_uq.yaml
  configs/gpt2_scope_block_no_head_uq.yaml
  configs/gpt2_scope_block_with_head_uq.yaml
  configs/gpt2_scope_block_all_uq.yaml
  configs/gpt2_scope_layer_no_head_uq.yaml
  configs/gpt2_scope_layer_with_head_uq.yaml
  configs/gpt2_scope_layer_all_uq.yaml
)
UQ_PRUNING=(
  configs/gpt2_scope_global_no_head_uq_pruning.yaml
  configs/gpt2_scope_global_with_head_uq_pruning.yaml
  configs/gpt2_scope_global_all_uq_pruning.yaml
  configs/gpt2_scope_block_no_head_uq_pruning.yaml
  configs/gpt2_scope_block_with_head_uq_pruning.yaml
  configs/gpt2_scope_block_all_uq_pruning.yaml
  configs/gpt2_scope_layer_no_head_uq_pruning.yaml
  configs/gpt2_scope_layer_with_head_uq_pruning.yaml
  configs/gpt2_scope_layer_all_uq_pruning.yaml
)
NO_HEAD=(
  configs/gpt2_scope_global_no_head_uq.yaml
  configs/gpt2_scope_global_no_head_uq_pruning.yaml
  configs/gpt2_scope_block_no_head_uq.yaml
  configs/gpt2_scope_block_no_head_uq_pruning.yaml
  configs/gpt2_scope_layer_no_head_uq.yaml
  configs/gpt2_scope_layer_no_head_uq_pruning.yaml
)
WITH_HEAD=(
  configs/gpt2_scope_global_with_head_uq.yaml
  configs/gpt2_scope_global_with_head_uq_pruning.yaml
  configs/gpt2_scope_block_with_head_uq.yaml
  configs/gpt2_scope_block_with_head_uq_pruning.yaml
  configs/gpt2_scope_layer_with_head_uq.yaml
  configs/gpt2_scope_layer_with_head_uq_pruning.yaml
)
SET_ALL=(
  configs/gpt2_scope_global_all_uq.yaml
  configs/gpt2_scope_global_all_uq_pruning.yaml
  configs/gpt2_scope_block_all_uq.yaml
  configs/gpt2_scope_block_all_uq_pruning.yaml
  configs/gpt2_scope_layer_all_uq.yaml
  configs/gpt2_scope_layer_all_uq_pruning.yaml
)
GLOBAL=(
  configs/gpt2_scope_global_no_head_uq.yaml
  configs/gpt2_scope_global_no_head_uq_pruning.yaml
  configs/gpt2_scope_global_with_head_uq.yaml
  configs/gpt2_scope_global_with_head_uq_pruning.yaml
  configs/gpt2_scope_global_all_uq.yaml
  configs/gpt2_scope_global_all_uq_pruning.yaml
)
BLOCK=(
  configs/gpt2_scope_block_no_head_uq.yaml
  configs/gpt2_scope_block_no_head_uq_pruning.yaml
  configs/gpt2_scope_block_with_head_uq.yaml
  configs/gpt2_scope_block_with_head_uq_pruning.yaml
  configs/gpt2_scope_block_all_uq.yaml
  configs/gpt2_scope_block_all_uq_pruning.yaml
)
LAYER=(
  configs/gpt2_scope_layer_no_head_uq.yaml
  configs/gpt2_scope_layer_no_head_uq_pruning.yaml
  configs/gpt2_scope_layer_with_head_uq.yaml
  configs/gpt2_scope_layer_with_head_uq_pruning.yaml
  configs/gpt2_scope_layer_all_uq.yaml
  configs/gpt2_scope_layer_all_uq_pruning.yaml
)
EVERY=(
  configs/gpt2_scope_global_no_head_uq.yaml
  configs/gpt2_scope_global_with_head_uq.yaml
  configs/gpt2_scope_global_all_uq.yaml
  configs/gpt2_scope_block_no_head_uq.yaml
  configs/gpt2_scope_block_with_head_uq.yaml
  configs/gpt2_scope_block_all_uq.yaml
  configs/gpt2_scope_layer_no_head_uq.yaml
  configs/gpt2_scope_layer_with_head_uq.yaml
  configs/gpt2_scope_layer_all_uq.yaml
  configs/gpt2_scope_global_no_head_uq_pruning.yaml
  configs/gpt2_scope_global_with_head_uq_pruning.yaml
  configs/gpt2_scope_global_all_uq_pruning.yaml
  configs/gpt2_scope_block_no_head_uq_pruning.yaml
  configs/gpt2_scope_block_with_head_uq_pruning.yaml
  configs/gpt2_scope_block_all_uq_pruning.yaml
  configs/gpt2_scope_layer_no_head_uq_pruning.yaml
  configs/gpt2_scope_layer_with_head_uq_pruning.yaml
  configs/gpt2_scope_layer_all_uq_pruning.yaml
)

ONLY=all
if [ "${1:-}" = "--only" ]; then
  ONLY="${2:-all}"
  shift 2
fi
case "$ONLY" in
  all)        CONFIGS=("${EVERY[@]}") ;;
  uq)         CONFIGS=("${UQ[@]}") ;;
  uq_pruning) CONFIGS=("${UQ_PRUNING[@]}") ;;
  no_head)    CONFIGS=("${NO_HEAD[@]}") ;;
  with_head)  CONFIGS=("${WITH_HEAD[@]}") ;;
  # The TARGET SET named `all` would be ambiguous with `all` meaning every run,
  # so it is selected as `everything`. Naming the two apart beats a footnote.
  everything) CONFIGS=("${SET_ALL[@]}") ;;
  global)     CONFIGS=("${GLOBAL[@]}") ;;
  block)      CONFIGS=("${BLOCK[@]}") ;;
  layer)      CONFIGS=("${LAYER[@]}") ;;
  *) echo "--only takes one of:" >&2
     echo "    all          every run (18)" >&2
     echo "    uq | uq_pruning          one method     (9 each)" >&2
     echo "    no_head | with_head | everything   one target set (6 each)" >&2
     echo "    global | block | layer   one grouping   (6 each)" >&2
     echo "  got '$ONLY'" >&2
     exit 2 ;;
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
        "runs reuse that cache. Do that download once BEFORE launching if the\n"
        "compute nodes have no outbound network.\n"
    )
PY

RUN_DIRS=()
STAMP=$(date +%Y%m%d-%H%M%S)
DIR_FILE=$(mktemp)
trap 'rm -f "$DIR_FILE"' EXIT

echo "=============================================================="
echo " GPT-2 target-set comparison: ppl_proxy / bpw_model"
echo " ${#CONFIGS[@]} experiments (--only $ONLY), K in [2, 8192]"
echo " started $(date)"
echo "=============================================================="
echo
echo " no_head cannot beat 2.78x however hard it searches -- that ceiling is"
echo " the fp16 token embedding, not the method. with_head lifts it to 14.42x"
echo " and all to 15.77x. The size answer is therefore already known; what the"
echo " runs decide is the PERPLEXITY price of reaching it."
echo
echo " Watch for: does with_head CROSS no_head at matched bpw_model, or only"
echo " extend past a ceiling no_head cannot reach? The second is a much weaker"
echo " claim. Same question for each uq_pruning run against its uq pair."

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "--------------------------------------------------------------"
  echo ">>> SEARCH  $cfg"
  echo "--------------------------------------------------------------"
  # --emit-run-dir reports where the run actually landed. Reading it back beats
  # guessing from log.run_name: a rerun takes the next free -2/-3 suffix, and
  # evaluating the previous run's front would pass silently.
  python3 scripts/run_search.py "$cfg" --emit-run-dir "$DIR_FILE" "$@"
  run_dir=$(cat "$DIR_FILE")
  RUN_DIRS+=("$run_dir")

  echo
  echo "--------------------------------------------------------------"
  echo ">>> EVAL    $run_dir   (PPL both corpora + latency + memory)"
  echo "--------------------------------------------------------------"
  python3 scripts/run_eval.py "$cfg" "$run_dir"
done

echo
echo "=============================================================="
echo " ${#CONFIGS[@]} runs finished $(date)"
echo "=============================================================="
for d in "${RUN_DIRS[@]}"; do echo "   $d"; done
echo
echo " compare along ONE axis at a time, holding the other fixed:"
echo "   python3 scripts/compare_runs.py \\"
for d in "${RUN_DIRS[@]}"; do
  echo "     $(basename "$d") \\"
done
echo "     --name gpt2-scope-$STAMP --bpw 6,7,8,10,12"
echo
echo " --bpw points are on bpw_model, which FLOORS AT 5.762 on no_head. Asking"
echo " for 2,3,4 returns nothing there: no genome can reach them while 31.7% of"
echo " the checkpoint is frozen at fp16. Compare no_head against the others only"
echo " on the range it can actually reach, and report the rest as reach that"
echo " no_head does not have."
