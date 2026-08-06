#!/usr/bin/env bash
# GPT-2, weight sharing + PRUNING under CSR costing.
#
# Companion to run_gpt2_prune_bitmap.sh. Same six runs, same pruning, same
# objectives -- the ONLY difference is how a survivor's position is recorded:
#
#   bitmap  1 bit per ORIGINAL position, alive or dead
#   csr     a gap field per SURVIVOR, counting from the previous one
#
# So the two scripts together answer one question: which representation
# reaches a smaller model at equal quality?
#
#   bash scripts/run_gpt2_prune_csr.sh
#   bash scripts/run_gpt2_prune_csr.sh --only 2obj
#   bash scripts/run_gpt2_prune_csr.sh --n-gen 5 --pop 20     # quick trial
#
# WHAT TO EXPECT, so the result is interpretable either way.
#
# The formats cross over around 87% sparsity, measured on one GPT-2 layer at
# K=256:
#
#   sparsity   0.20   0.55   0.87   0.96   0.99
#   bitmap     7.42   4.63   2.07   1.37   1.10
#   csr        8.09   5.06   1.81   0.67   0.21
#
# Below the crossover bitmap wins, because CSR pays a gap field on every
# survivor and at low sparsity there are many. Above it CSR wins and the margin
# compounds, because bitmap's mask is one bit per ORIGINAL weight and becomes a
# hard floor at 1.0 bpw that no amount of pruning can cross. CSR charges
# nothing for a pruned weight and has no floor.
#
# So CSR should win IF the search pushes past ~87% sparsity AND quality holds
# up there. Both halves are open. The dense runs never explored that regime,
# because pruning was worth nothing to them.
#
# Two things that are NOT decision variables, and should not be read as such:
#   * the gap width, selected per layer in closed form from sparsity and index
#     width, recorded in a 6-bit tag
#   * the format itself, fixed per run here precisely so the comparison is
#     between formats rather than within one
#
# The comparison to run afterwards is NOT hypervolume -- the two runs optimise
# different cost functions, so their HVs are not commensurable. Compare best
# whole-model CR at a matched held-out perplexity budget, which is what
# scripts/reprice_fronts.py tabulates.

set -euo pipefail
cd "$(dirname "$0")/.."

TWO=(
  configs/gpt2_k_global_prune_csr_2obj.yaml
  configs/gpt2_k_block_prune_csr_2obj.yaml
  configs/gpt2_k_layer_prune_csr_2obj.yaml
)
THREE=(
  configs/gpt2_k_global_prune_csr_3obj.yaml
  configs/gpt2_k_block_prune_csr_3obj.yaml
  configs/gpt2_k_layer_prune_csr_3obj.yaml
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
from evolmc.config import Config
bad = []
for g in ("global", "block", "layer"):
    for n in (2, 3):
        p = f"configs/gpt2_k_{g}_prune_csr_{n}obj.yaml"
        c = Config.from_yaml(p)
        if c.quant.deployable_format != "csr":
            bad.append(f"{p}: deployable_format is {c.quant.deployable_format!r}")
        if c.quant.csr_span_bits is not None:
            bad.append(f"{p}: csr_span_bits is {c.quant.csr_span_bits}, expected null "
                       "(per-layer selection)")
        if not c.prune.enabled:
            bad.append(f"{p}: prune.enabled is False")
        if len(c.search.objectives) != n:
            bad.append(f"{p}: {len(c.search.objectives)} objectives, expected {n}")
        # CSR has no floor; a box pinned at 1.0 would clip the whole point.
        if c.plot.xlim_min != 0.0:
            bad.append(f"{p}: plot.xlim_min is {c.plot.xlim_min}, expected 0.0")
if bad:
    raise SystemExit("preflight failed:\n  " + "\n  ".join(bad))
print("preflight ok: 6 configs, pruning on, CSR costing, per-layer span")
PY

RUN_DIRS=()
STAMP=$(date +%Y%m%d-%H%M%S)
DIR_FILE=$(mktemp)
trap 'rm -f "$DIR_FILE"' EXIT

echo "=============================================================="
echo " GPT-2 pruning + weight sharing, CSR costing"
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
echo " to answer 'which format reaches a smaller model', price both sets under"
echo " every format and compare at matched perplexity -- not by hypervolume,"
echo " which is not commensurable across different cost functions:"
echo
echo "   python3 scripts/reprice_fronts.py --runs \\"
for d in "${RUN_DIRS[@]}"; do echo "     $(basename "$d") \\"; done
echo "     gpt2-k-global-prune-bitmap-2obj-np100-ng100 \\"
echo "     gpt2-k-block-prune-bitmap-2obj-np100-ng100 \\"
echo "     gpt2-k-layer-prune-bitmap-2obj-np100-ng100"
echo
echo " the diagnostic that explains whichever way it goes: what sparsity did"
echo " each front actually reach? CSR only wins above ~87%."
