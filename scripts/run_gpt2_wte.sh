#!/usr/bin/env bash
# GPT-2 with the tied token embedding IN the compressed target set.
#
# Six runs: three K groupings x {weight sharing, weight sharing + pruning}.
# All NSGA-II, pop 100, 1000 generations.
#
#   bash scripts/run_gpt2_wte.sh
#   bash scripts/run_gpt2_wte.sh --only share
#   bash scripts/run_gpt2_wte.sh --only prune
#   bash scripts/run_gpt2_wte.sh --n-gen 5 --pop 20        # quick trial
#
# Anything after --only is forwarded to run_search.py.
#
# -- WHAT THESE RUNS TEST -----------------------------------------------------
#
# Every previous GPT-2 run left 39.5M weights (31.7% of the checkpoint) at fp16
# because model.exclude_patterns excluded the LM head. That put a HARD CEILING
# of 3.15x on cr_deployable which no genome could reach past. uniform K=2
# already scored 2.78x, so 88% of everything the index axis could ever deliver
# had already been spent and the remaining 0.37x was unreachable.
#
# These six configs drop "lm_head" from the exclusion list. GPT-2 ties its head
# to its embedding, so lm_head is an nn.Linear holding transformer.wte.weight
# itself, and removing the string brings the whole 38.6M-weight embedding into
# the target set through the head:
#
#     target layers      48  ->  49
#     target weights   84.9M -> 123.5M
#     untouched        39.5M ->   0.91M   (wpe + biases + norms)
#     CR ceiling       3.15x ->  137x
#
# Removing "wte" or "embed" instead does NOTHING. discover_targets filters on
# module type before it looks at the patterns, and transformer.wte is an
# nn.Embedding. The preflight below checks the target set is really 49 rather
# than trusting that.
#
# -- WHAT TO READ IN THE RESULTS ----------------------------------------------
#
# bpw_target will look almost unchanged against the ng100 runs. At per_tensor
# the codebook is ~0.06 bpw, so bpw_target is essentially ceil(log2 K) either
# way; only the denominator moved. The whole effect lands on cr_deployable,
# which is why every config carries it in report_metrics. Expected:
#
#     bpw   CR before   CR after
#       1     2.78x      14.4x
#       2     2.48x       7.61x
#       4     2.05x       3.91x
#       8     1.52x       1.99x
#
# The second thing to look for is the lm_head K itself. Under `block` and
# `layer` grouping it gets its own variable (block=-1 puts it in group "b-1"),
# so the front can assign the head a different codebook size from the
# projections. Head quantization is more sensitive than projection
# quantization, so expect a HIGHER K there. That asymmetry is a result in its
# own right: a uniform-K baseline cannot express it. The `global` config cannot
# either, which is what makes it the control.
#
# -- COST ---------------------------------------------------------------------
#
# 100,000 evaluations per run against the 10,000 of the ng100 sweep, on 45% more
# weights. The three prune runs are much the slowest: pruning disables the
# reconstruction cache outright, so every candidate recompresses all 123.5M
# target weights. Share runs go first so the cheap, most legible results land
# before the expensive ones. checkpoint_every is 10, so any run resumes with
#
#     python3 scripts/run_search.py <config> --resume <run_dir>/checkpoints/latest.pkl

set -euo pipefail
cd "$(dirname "$0")/.."

SHARE=(
  configs/gpt2_k_global_wte.yaml
  configs/gpt2_k_block_wte.yaml
  configs/gpt2_k_layer_wte.yaml
)
PRUNE=(
  configs/gpt2_k_global_wte_prune_bitmap_2obj.yaml
  configs/gpt2_k_block_wte_prune_bitmap_2obj.yaml
  configs/gpt2_k_layer_wte_prune_bitmap_2obj.yaml
)

ONLY=all
if [ "${1:-}" = "--only" ]; then
  ONLY="${2:-all}"
  shift 2
fi
case "$ONLY" in
  all)   CONFIGS=("${SHARE[@]}" "${PRUNE[@]}") ;;
  share) CONFIGS=("${SHARE[@]}") ;;
  prune) CONFIGS=("${PRUNE[@]}") ;;
  *) echo "--only takes: all, share, prune (got '$ONLY')" >&2; exit 2 ;;
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

# Fail before spending GPU DAYS if the target set is not what we think it is.
# This loads GPT-2 once, which is the only way to check the thing that actually
# matters: that lm_head really is discovered. A config whose exclude_patterns
# look right but whose target count is still 48 would produce six runs
# indistinguishable from the ng100 sweep, and nothing else would notice.
python3 - <<'PY'
import torch
from transformers import AutoModelForCausalLM
from evolmc.config import Config
from evolmc.models import count_untouched_weights, discover_targets
from evolmc.grouping import Genome

EXPECT_VARS = {
    "gpt2_k_global_wte.yaml": 1,
    "gpt2_k_block_wte.yaml": 13,
    "gpt2_k_layer_wte.yaml": 49,
    "gpt2_k_global_wte_prune_bitmap_2obj.yaml": 3,
    "gpt2_k_block_wte_prune_bitmap_2obj.yaml": 15,
    "gpt2_k_layer_wte_prune_bitmap_2obj.yaml": 51,
}

model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
if model.lm_head.weight is not model.transformer.wte.weight:
    raise SystemExit(
        "preflight failed: gpt2's lm_head is NOT tied to wte on this "
        "transformers version. These configs assume the tie -- without it, "
        "dropping 'lm_head' compresses the head only and leaves the input "
        "embedding at fp16."
    )

bad = []
for name, n_var in EXPECT_VARS.items():
    path = f"configs/{name}"
    cfg = Config.from_yaml(path)

    if "lm_head" in cfg.model.exclude_patterns:
        bad.append(f"{name}: exclude_patterns still contains 'lm_head'")
    if cfg.search.algorithm != "nsga2":
        bad.append(f"{name}: algorithm is {cfg.search.algorithm!r}, expected nsga2")
    if cfg.search.n_gen != 1000:
        bad.append(f"{name}: n_gen is {cfg.search.n_gen}, expected 1000")
    if "cr_deployable" not in cfg.search.report_metrics:
        bad.append(f"{name}: cr_deployable missing from report_metrics")
    # cr_deployable must stay a REPORTED metric. With untouched down to 14.5
    # Mbits it is a near-monotone transform of bpw_target, so as an objective it
    # would buy a third axis that cannot disagree with the second.
    if "cr_deployable" in cfg.search.objectives:
        bad.append(f"{name}: cr_deployable is an OBJECTIVE; it is redundant "
                   f"against bpw_target once wte is a target")
    if cfg.prune.enabled and cfg.quant.deployable_format != "bitmap":
        bad.append(f"{name}: pruning on but deployable_format is "
                   f"{cfg.quant.deployable_format!r}; pruning would be free")

    targets = discover_targets(model, cfg.model.exclude_patterns)
    if len(targets) != 49:
        bad.append(f"{name}: {len(targets)} targets, expected 49")
    if "lm_head" not in {t.name for t in targets}:
        bad.append(f"{name}: lm_head is not in the target set")

    genome = Genome(targets, cfg.quant, cfg.prune, cfg.variables)
    if genome.n_var != n_var:
        bad.append(f"{name}: n_var is {genome.n_var}, expected {n_var}")

if bad:
    raise SystemExit("preflight failed:\n  " + "\n  ".join(bad))

cfg = Config.from_yaml("configs/gpt2_k_block_wte.yaml")
targets = discover_targets(model, cfg.model.exclude_patterns)
tw = sum(t.n_weights for t in targets)
un = count_untouched_weights(model, targets)
print(f"preflight ok: 6 configs, nsga2, n_gen 1000, lm_head tied to wte")
print(f"  targets    {len(targets)}  ({tw:,} weights, "
      f"{100 * tw / (tw + un):.2f}% of checkpoint)")
print(f"  untouched  {un:,} weights ({100 * un / (tw + un):.2f}%)  "
      f"-> CR ceiling {(tw + un) / un:.1f}x")
PY

RUN_DIRS=()
STAMP=$(date +%Y%m%d-%H%M%S)
DIR_FILE=$(mktemp)
trap 'rm -f "$DIR_FILE"' EXIT

echo "=============================================================="
echo " GPT-2 + wte: 49 targets, CR ceiling 137x (was 3.15x)"
echo " ${#CONFIGS[@]} runs (--only $ONLY), pop 100 x 1000 gen, started $(date)"
echo "=============================================================="

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
echo
echo "   # Compare the three groupings within this sweep."
echo "   python3 scripts/compare_runs.py \\"
for d in "${RUN_DIRS[@]}"; do echo "     $(basename "$d") \\"; done
echo "     --labels 'global,block,layer' --name wte-$STAMP --bpw 2,3,4,6,8"
echo
echo "   # Against the ng100 runs, compare on cr_deployable, NOT bpw_target:"
echo "   # the target denominator moved from 84.9M to 123.5M weights, so the"
echo "   # two sweeps' bpw figures are not the same quantity."
