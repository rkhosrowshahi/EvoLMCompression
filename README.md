# EvoLMCompression

> **All rights reserved — not open source.** Public for inspection only.
> See [LICENSE](LICENSE).

Post-training compression for language models. Weights are replaced by codebook
centroids and optionally pruned into a zero band, then entropy-coded. **NSGA-II**
searches the per-layer codebook size `K` and pruning thresholds, returning a
Pareto front of perplexity against bits per weight.

## Install

```bash
pip install -r requirements.txt
```

## Run

Check the pipeline end to end — no `datasets`, no GPU, ~1 minute:

```bash
python scripts/smoke_test.py --model gpt2
```

Search, then evaluate the front on held-out data:

```bash
python scripts/run_search.py configs/uq/gpt2_124m/gpt2_124m-only_proj-layer_quant-2obj.yaml
```

```bash
python scripts/run_eval.py configs/uq/gpt2_124m/gpt2_124m-only_proj-layer_quant-2obj.yaml logs/gpt2_124m-only_proj-layer_quant-2obj-np100-ng100
```

Each run writes one timestamped directory under `logs/` containing its config,
logs, checkpoints, Pareto figures and a video of the front converging.

## Scripts

| | |
|---|---|
| `run_search.py` | run one search from a config |
| `run_eval.py` | score the front on both corpora, write the paper figures |
| `compare_runs.py` | overlay several runs' fronts and tabulate them |
| `replot.py` | re-render a finished run at a different size or venue |
| `make_video.py` | encode the per-generation frames to mp4/gif |
| `smoke_test.py` | end-to-end wiring check on a synthetic corpus |

## Experiments

`configs/uq*/gpt2_124m/gpt2_k_*.yaml` are a granularity ablation — global, block-wise and
layer-wise `K` — differing in exactly one line so the fronts are comparable.

```bash
bash scripts/gpt2_granularity.sh
```

## Notes

Two compression ratios are always reported: **deployable** (fixed-width indices,
what a LUT kernel reads) and **archival** (Huffman-coded, storage only). Bits per
weight is given both for the compressed matrices and for the whole checkpoint —
embeddings stay fp16 and are counted.

Design rationale, accounting details and the reasoning behind each default are in
[docs/DESIGN.md](docs/DESIGN.md).

```bash
python -m pytest tests/ -q
```
