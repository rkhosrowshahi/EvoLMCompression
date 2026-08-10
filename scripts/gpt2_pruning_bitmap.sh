#!/usr/bin/env bash
# GPT-2, weight sharing + PRUNING under sparse-aware costing.
#
# Six runs: three K groupings x {2, 3} objectives. All with prune.enabled and
# all priced under quant.deployable_format: bitmap, so pruning finally reduces
# the model instead of only skewing the symbol histogram.
#
#   PPL + BPW        the control
#   PPL + BPW + ACR  adds the archival axis
#
#   bash scripts/run_gpt2_prune_bitmap.sh
#   bash scripts/run_gpt2_prune_bitmap.sh --only 2obj
#   bash scripts/run_gpt2_prune_bitmap.sh --only 3obj
#   bash scripts/run_gpt2_prune_bitmap.sh --n-gen 5 --pop 20     # quick trial
#
# Anything after --only is forwarded to run_search.py.
#
# WHY THIS SUPERSEDES THE EARLIER PRUNED RUNS. Those were priced under `dense`,
# which charges one index per weight POSITION, so a pruned weight cost exactly
# what a live one did. Candidates spanning 0.000 to 0.953 sparsity at the same
# bpw had cr_deployable identical to six decimals. The search was shown a knob
# that costs quality and returns nothing, and correctly refused to turn it:
# median front sparsity came out at 0.13 to 0.29. Under bitmap the cost is
#
#   n  +  ceil(log2(K-1)) * n_alive  +  G * (K-1) * b_cb
#
# so pruning only pays above sparsity > 1/ceil(log2 K). K and the threshold are
# now COUPLED, which they were not before. Expect sparsity and K to come out
# positively correlated; the dense runs showed rho = -0.59 at global, and that
# reversal is the headline to check.

set -euo pipefail
cd "$(dirname "$0")/.."

TWO=(
  configs/gpt2_k_global_prune_bitmap_2obj.yaml
  configs/gpt2_k_block_prune_bitmap_2obj.yaml
  configs/gpt2_k_layer_prune_bitmap_2obj.yaml
)
THREE=(
  configs/gpt2_k_global_prune_bitmap_3obj.yaml
  configs/gpt2_k_block_prune_bitmap_3obj.yaml
  configs/gpt2_k_layer_prune_bitmap_3obj.yaml
)

ONLY=all
if [ "${1:-}" = "--only" ]; then
  ONLY="${2:-all}"
  shift 2
fi
case "$ONLY" in
  all)  CONFIGS=("${TWO[@]}" "${THREE[@]}") ;;
  2obj) CONFIGS=("${TWO[@]}") ;;
  3obj) CONFIGS=("${THREE[@]}") ;;
  *) echo "--only takes: all, 2obj, 3obj (got '$ONLY')" >&2; exit 2 ;;
esac

python3 - <<'PY'
try:
    import datasets  # noqa: F401
except ImportError:
    raise SystemExit(
        "\nThese configs read real corpora (C4 for calibration, WikiText-2 for\n"
        "evaluation), which needs the `datasets` package:\n\n"
        "    pip install datasets\n"
    )
PY

# Fail before spending GPU hours if the cost model is not what we think it is.
python3 - <<'PY'
import sys
from evolmc.config import Config
bad = []
for g in ("global", "block", "layer"):
    for n in (2, 3):
        p = f"configs/gpt2_k_{g}_prune_bitmap_{n}obj.yaml"
        c = Config.from_yaml(p)
        if c.quant.deployable_format != "bitmap":
            bad.append(f"{p}: deployable_format is {c.quant.deployable_format!r}")
        if not c.prune.enabled:
            bad.append(f"{p}: prune.enabled is False")
        if len(c.search.objectives) != n:
            bad.append(f"{p}: {len(c.search.objectives)} objectives, expected {n}")
if bad:
    raise SystemExit("preflight failed:\n  " + "\n  ".join(bad))
print("preflight ok: 6 configs, pruning on, bitmap costing")
PY

RUN_DIRS=()
STAMP=$(date +%Y%m%d-%H%M%S)
DIR_FILE=$(mktemp)
trap 'rm -f "$DIR_FILE"' EXIT

echo "=============================================================="
echo " GPT-2 pruning + weight sharing, BITMAP costing"
echo " ${#CONFIGS[@]} runs (--only $ONLY), started $(date)"
echo "=============================================================="

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "--------------------------------------------------------------"
  echo ">>> SEARCH  $cfg"
  echo "--------------------------------------------------------------"
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
echo " finished $(date)"
echo "=============================================================="
for d in "${RUN_DIRS[@]}"; do echo "   $d"; done
echo
echo " next:"
echo "   # price every front under all three formats, for the comparison"
echo "   python3 scripts/reprice_fronts.py --runs \\"
for d in "${RUN_DIRS[@]}"; do echo "     $(basename "$d") \\"; done
echo
echo "   # compare like objective counts only -- hypervolumes over 2 and 3"
echo "   # objectives are volumes in different spaces and compare_runs refuses"
echo "   # to table them together"
echo "   python3 scripts/compare_runs.py \\"
for d in "${RUN_DIRS[@]}"; do
  case "$(basename "$d")" in *-2obj-*) echo "     $(basename "$d") \\";; esac
done
echo "     --labels 'global,block,layer' --name prune-bitmap-2obj-$STAMP"
