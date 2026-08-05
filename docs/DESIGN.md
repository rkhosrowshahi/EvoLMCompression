# EvoLMCompression

> **Licence: all rights reserved — not open source.** This repository is public
> for inspection and to establish authorship. Redistribution, derivative works,
> and use of the code or its results in any published work require prior
> written permission. See [LICENSE](LICENSE).

Multi-objective post-training compression for language models: codebook
quantization + threshold-band pruning + entropy coding, with NSGA-II choosing
the per-layer codebook size `K` and the pruning band `(t_lo, t_hi)`.

The search minimises two objectives — **proxy perplexity** and **bits per
weight** — and returns a Pareto front of quality/size trade-offs.

## Quick start

```bash
pip install -r requirements.txt
```

Verify the pipeline without needing `datasets` or a GPU:

```bash
python scripts/smoke_test.py --model gpt2
```

Run a real search, then evaluate the resulting front against the fixed-bit
baselines:

```bash
python scripts/run_search.py configs/dev_gpt2.yaml
```

```bash
python scripts/run_eval.py configs/dev_gpt2.yaml latest
```

### Which corpus the figures show

The **search runs entirely on the calibration corpus** (`data.calib_dataset`,
C4). Every per-generation frame, the baseline curve, the fp16 line and the
convergence plot are proxy perplexity on `data.n_proxy_seq` windows of it —
which is why the y axis reads *proxy* perplexity.

`run_eval.py` then scores every front member on **both** corpora from a
single quantization and writes three more figures:

| Figure | What it shows |
|---|---|
| `figures/front_eval` | the front on held-out `wikitext2` — **the figure a paper leads with** |
| `figures/front_calib_vs_eval` | both curves on one axis; the vertical gap is the generalisation gap |
| `figures/proxy_vs_eval` | per-member scatter with Spearman ρ — is the cheap proxy a valid surrogate? |

`data/results.csv` gains `ppl_eval` and `ppl_calib` columns side by side.

A front drawn on the calibration corpus is partly a picture of overfitting:
thousands of evaluations against a handful of windows will find configurations
that suit *those windows*. Reporting held-out numbers, and showing the gap
rather than hiding it, is what makes the result defensible. The terminal warns
when ρ drops below 0.9 — that means the search partly optimised noise and
`data.n_proxy_seq` should go up.

```bash
python -m pytest tests/ -q
```

## Run directories

Every run writes one self-describing directory under `logs/`, and `logs/latest`
symlinks to the most recent one. Nothing is ever overwritten between runs.

```
logs/20260804-142530__gpt2__nsga2-p24g20__k-type__p-global__uniform/
  config.yaml                fully resolved config — the run is reproducible from this alone
  meta.json                  GPU, peak VRAM, torch version, timings, argv
  logs/
    run.log                  everything printed, with elapsed timestamps
    evals.jsonl              one line per fitness evaluation
    generations.jsonl        one line per generation
  data/
    baselines.csv            fp16 + every uniform-K reference point
    front.csv                the final front as a table
    front.json               the front with full per-layer genomes
    history.npz              every X and F, per generation
    results.csv              full re-evaluation (written by run_eval.py)
  checkpoints/
    gen_0005.pkl … latest.pkl
  figures/
    pareto/gen_0001.png|pdf  one frame per generation
    pareto_final.png|pdf
    convergence.png|pdf
    pareto_evolution.mp4|gif  the frames encoded into a video
```

Name it yourself with `--name ablation-kmeans`, or let it auto-generate from
timestamp + model + algorithm + grouping + binning.

### What gets logged

`logs/evals.jsonl` — one record per candidate, appended live, so you can watch
a run with `tail -f`:

```json
{"eval": 16, "t": 3.48, "eval_seconds": 0.537, "apply_seconds": 0.366,
 "ppl_proxy": 8554355.8, "x": [0.1, 0.1, 0.1, 0.0, 0.0],
 "bpw_target": 2.0625, "bpw_target_archival": 1.70962, "bpw_model": 6.48715,
 "cr_deployable": 2.46641, "cr_archival": 2.56152, "sparsity": 0.0, ...}
```

The full genome `x` is stored on every line, so any evaluation from any run can
be replayed exactly.

`logs/generations.jsonl` — one record per generation: front size, best
perplexity, bpw range, hypervolume, wall-clock, and the front itself.
`data/history.npz` holds the raw `X` and `F` arrays for every generation if you
want to re-plot or re-analyse offline.

### Figures

The Pareto plot is the main artefact, written after every generation as **both
PNG and PDF** (`pdf.fonttype 42`, so text stays selectable in LaTeX).

**Axis limits are computed once, before generation 1, and frozen for the whole
run.** That is the point: every frame is drawn in the same box, so flipping
through `figures/pareto/` shows the front moving rather than the axes rescaling
under it.

- **x limits** come from `quant.k_choices` in closed form — the reachable bpw
  interval is known before a single evaluation runs, so the box never depends
  on what the search happened to sample.
- **y limits**: the ceiling is **uncapped by default** (`plot.ylim_max_ratio:
  null`) — it opens to the highest reference point, and `refit_at_end` raises
  it further to cover every candidate, so nothing is ever drawn off-scale. Set
  a number to cap at that multiple of the fp16 perplexity instead; the excess
  is then excluded and counted. The floor is `plot.ylim_min` when set (1.0 is
  the theoretical minimum), otherwise `ylim_min_ratio ×` the lowest reference.
  Both ends are opened by `plot.ylim_pad`, measured in decades on a log axis,
  so extreme points do not sit on the spines.
- Points outside the box are **clipped and counted** in a corner annotation,
  never silently dropped.

Set `plot.xlim` / `plot.ylim` explicitly for the final paper figures. Every
frame also carries the fp16 line and the uniform-K baseline curve, so "is the
front below the fixed-bit line?" is answerable from any single frame.

#### On the perplexity floor

Perplexity is `exp(cross-entropy)`, and cross-entropy is non-negative, so
**PPL >= 1** — a hard mathematical bound, not a convention. PPL = 1 is a model
that assigns probability 1 to every correct token. A floor of **0 is
unreachable**, and on a log axis it sits at negative infinity and cannot be
drawn at all; `plot.ylim: [0, ...]` with `yscale: log` raises an explanatory
error rather than producing a broken figure.

To open up room at the bottom, in increasing order of bluntness:

| Want | Setting |
|---|---|
| A little more space under fp16 | `plot.ylim_min_ratio: 0.7` (default 0.9) |
| The true theoretical floor | `plot.ylim: [1.0, <top>]` |
| An axis that really starts at 0 | `plot.yscale: linear` + `plot.ylim: [0, <top>]` |

The default floor is `0.9 x fp16` because fp16 is the *practical* bound — no
PTQ method beats the uncompressed model except by noise — so the sliver below
it is there to show a candidate that ties or marginally wins. Setting the floor
to 1.0 is legal but usually wastes most of the axis height: with fp16
perplexity around 25, a box from 1 to 300 is mostly a region nothing can
occupy.

**The box is frozen before any candidate exists**, so a search that *does* beat
fp16 lands under the floor — and matplotlib clips those points onto the spine
rather than excluding them the way it does for the scatter cloud. `refit_at_end`
(on by default) handles this: when anything was evaluated below the floor, the
floor is reopened once at the end and **every frame is re-rendered in the
corrected box**, so the run still has exactly one box and the frames stay
mutually comparable. It never opens below 1.0, and it is skipped when
`plot.ylim` is set explicitly.

```
refitting y floor 149255.82 -> 121458.81 and re-rendering 6 frames
```

When replotting, `--fit-box` does the same from the stored history, and
`--ylim 1.0,2e6` / `--xlim` override the box outright.

### Paper figures

Figure text mismatches a paper for **two independent reasons**, and the second
is usually the larger effect:

1. **Typeface.** matplotlib defaults to DejaVu Sans; papers are set in a serif.
   Note that *math* has its own font — setting `font.family` alone leaves `$K$`
   in DejaVu next to Times body text, so `mathtext.fontset` must be set too.
2. **Scaling.** A figure saved 7 in wide and included at `\columnwidth`
   (3.5 in) is scaled by 0.5, so its 8 pt labels arrive on the page at **4 pt**.
   The fix is not a bigger font — it is to save the figure at exactly the width
   it will occupy, so LaTeX scales it by 1.0.

Setting `plot.venue` handles both. It selects the style file's typeface and
sizes the canvas to the exact printed width, and it disables the tight bbox
(which would otherwise crop the canvas to an unpredictable width and
reintroduce the scale factor).

| Venue | Column | Page | Font | Used by |
|---|---|---|---|---|
| `ieee` | 3.50 in | 7.16 in | Times, 8 pt | CEC, TEVC, IEEE conferences |
| `acm` | 3.33 in | 7.00 in | Libertine, 8 pt | **GECCO**, ACM conferences |
| `icml` | 3.25 in | 6.75 in | Times, 8 pt | ICML |
| `neurips` | 5.50 in | 5.50 in | Times, 9 pt | NeurIPS (single column) |
| `lncs` | 4.80 in | 4.80 in | Times, 8 pt | Springer LNCS |

Re-render a finished run without reloading the model or re-evaluating anything
— seconds, not another search:

```bash
python scripts/replot.py latest --venue ieee --width column
```

The frozen axis box is read back from `data/plot_box.json`, so replotted frames
stay directly comparable with the ones written during the run. `--venue acm
--width page` emits `figure*` in the LaTeX snippet it prints; `--usetex`
renders text with real LaTeX (needs `latex` + `dvipng`), which on an IEEE run
produces TeX Gyre Termes + NewTXMI — precisely what `IEEEtran` + `newtx` set
your body text in.

Standalone figures in venue mode also switch to a **minimal** layout: no
generation title, no evaluation counter, no population cloud. The generation
number belongs in the caption, and the sampled cloud is noise once the search
has converged. Per-generation video frames keep all of it.

Verified invariant, pinned in `tests/test_paper.py`: the saved PDF's MediaBox
equals the target width to within 0.002 in, so the LaTeX scale factor is
exactly 1.0000.

### Video

The frames are encoded into a video automatically when a run finishes, and
`scripts/make_video.py` re-encodes any run at a different frame rate — or
produces one for a run made before the encoder existed:

```bash
python scripts/make_video.py latest --fps 8
```

- **mp4** via ffmpeg when it is on `PATH` (H.264, CRF 18, `+faststart`). Frames
  are fed through a concat list rather than `-i gen_%04d.png`, so a gap in the
  numbering from `plot.every > 1` does not truncate the video at the first
  missing index.
- **gif** via Pillow, which ships with matplotlib — no external dependency, so
  this always works. One palette is derived from a downsampled stack of *every*
  frame, not from the first one: a frame-1 palette misses colours introduced
  later and can collapse near-identical frames entirely.

Both hold the final frame for `hold_last` frame-durations so the converged
front is readable before the loop restarts. Missing encoders are reported and
skipped, never raised — losing a video must not fail a search that already
produced its real artefacts.

### Resuming

Checkpoints are written every `log.checkpoint_every` generations. A 7B run that
dies at generation 31 of 40 resumes with:

```bash
python scripts/run_search.py configs/llama2_7b.yaml --resume logs/latest
```

## Experiments

### GPT-2 codebook-size sweep (`configs/gpt2_k_*.yaml`)

Three configs that differ in **exactly one line** (`variables.k_grouping`), so
any difference in the Pareto fronts is attributable to granularity alone.
Any integer K in [2, 8192], pruning off.

| Config | Grouping | Variables | Search space |
|---|---|---|---|
| `gpt2_k_global.yaml` | `global` | 1 | 8191 |
| `gpt2_k_block.yaml` | `block` | 12 | 8191^12 |
| `gpt2_k_layer.yaml` | `block_type` | 48 | 8191^48 |

```bash
bash scripts/run_gpt2_k_experiments.sh
```

Runs all three sequentially, then evaluates each front on full WikiText-2.

Budget: `pop_size: 100` x `n_gen: 50` = **5000 evaluations each**. Measured cost
is 0.62 s to apply a candidate plus 1.62 s of forward passes (GPT-2, seqlen
1024, 8 proxy windows) — so roughly **0.5 h per experiment on an H100**, 1.6 h
on an RTX 3060, and 3 h on Apple silicon. Pop 100 rather than a smaller value
because the layer-wise config has 48 variables and NSGA-II needs the population
to resolve a front in that dimension; lower it in all three together if you
want a faster trial, since the comparison is only controlled when the budget
matches.

### NSGA-II operator controls

Exposed in every config under `search:`. pymoo gives each operator **two**
probabilities and confusing them is easy:

| | meaning |
|---|---|
| `crossover_prob` | chance SBX fires on a mating pair at all |
| `crossover_prob_var` | chance each gene is exchanged once it fires |
| `crossover_eta` | SBX spread; higher = children hug their parents |
| `mutation_prob` | chance PM fires on an individual at all |
| `mutation_prob_var` | chance each gene is perturbed; `null` -> `1/n_var` |
| `mutation_eta` | PM spread; higher = smaller steps |
| `eliminate_duplicates`, `n_offsprings` | NSGA-II survival / offspring count |
| `ref_dir_partitions`, `moead_*` | U-NSGA-III and MOEA/D only |

The familiar "1/n_var" rule refers to **`prob_var`**, and pymoo already defaults
it to `min(0.5, 1/n_var)`. Putting `1/n_var` into `mutation_prob` instead
silently mutates only that fraction of the *population* — 2% at 48 variables,
0.1% for a per-layer Llama-2 encoding. This codebase had exactly that bug until
the operators were exposed; `tests/test_pipeline.py` now pins it.
Needs `pip install datasets` for the real corpora. Compare afterwards:

```bash
python scripts/compare_runs.py gpt2-k-global gpt2-k-block gpt2-k-layer --bpw 2,3,4,6,8
```

Two choices baked into these configs that differ from the defaults:

**`granularity: per_tensor`** — one codebook per layer. A K of 10^4 is only
meaningful this way. GPT-2 rows hold 768 weights, so a per-channel codebook can
never fill more than 768 entries, and storing 8192 fp16 centroids for a
768-weight row costs **170 bpw of codebook alone**; per tensor the same
codebook costs 0.06 bpw. This is also the original Deep Compression setup.

**`k_encoding: integer`** — the search picks any K in [2, 8192], not just
powers of two, mapped **log-spaced** because the cost axis is index width
`log2 K`. A linear map would spend half the gene range above K=4096, where
quality has plateaued, and leave almost no resolution at small K.

Know what this buys at `per_tensor`. Index width is `ceil(log2 K)`, so every K
inside a band costs identical indices and differs only in codebook size — and
that band spans just **0.0004 bpw** here (measured on `h.6.mlp.c_fc`,
K ∈ [65, 128], where mse improves 74%). So the integer freedom is real but
nearly free to ignore at this granularity. The regime where it genuinely pays
is `per_channel`, where the same band spans **1.31 bpw**. `k_choices` no longer
bounds the search; it is now only the ladder of reference points, warm-start
seeds, and plot-axis anchors.

Expect `global` to converge at generation 1: its whole space is 13 points,
already enumerated by the baseline sweep. It is there as a like-for-like
reference, not to discover anything. And note GPT-2's whole-model CR is capped
at **2.8x** regardless of K, because the embedding table is 31% of the
checkpoint and stays fp16 — see the exclusion table below.

`compare_runs.py` **recomputes every hypervolume on one shared axis box**
rather than reading each run's logged value. Stored HV is normalised by that
run's own box, and the end-of-run refit can move a box, so stored values are
not comparable across runs — in testing, the ranking flipped once they were
put on a common box.

## Method

For each weight matrix, given `(K, t_lo, t_hi)`:

1. **Prune** — zero every weight inside the band `[t_lo, t_hi]`.
2. **Bin** — partition the surviving weights into `K-1` bins.
3. **Centre** — each codeword is the mean of the weights in its bin.
4. **Replace** — hard-substitute every weight by its codeword.
5. **Price** — count index bits, codebook bits and Huffman table bits.

Pruned weights are folded into the codebook as a **reserved zero codeword**
rather than carried in a separate bitmask. This costs no extra index bits and
makes pruning show up where it belongs: as a sharply skewed symbol
distribution that the entropy coder then exploits. It is why nominal `K` buys
`K-1` centroids whenever the band is non-degenerate.

## Layout

| File | Role |
|---|---|
| [config.py](evolmc/config.py) | All run options; one YAML per experiment |
| [models.py](evolmc/models.py) | Layer discovery, master weights, in-place restore |
| [grouping.py](evolmc/grouping.py) | Genome encoding and the variable-count dial |
| [quantize.py](evolmc/quantize.py) | Binning, codebooks, weight replacement |
| [codec.py](evolmc/codec.py) | Huffman, entropy, bpw and CR accounting |
| [compressor.py](evolmc/compressor.py) | Applies a genome to the live model |
| [problem.py](evolmc/problem.py) | The pymoo problem: `f1 = ppl`, `f2 = bpw` |
| [search.py](evolmc/search.py) | NSGA-II / U-NSGA-III / MOEA/D drivers |
| [rundir.py](evolmc/rundir.py) | Run directory layout and logging |
| [plotting.py](evolmc/plotting.py) | Pareto frames with frozen axes, convergence |
| [video.py](evolmc/video.py) | Frames to mp4 (ffmpeg) / gif (Pillow) |

## Four things this codebase is opinionated about

**1. Report two compression ratios and never mix them.**
`cr_deployable` uses fixed-width indices — this is what a LUT dequant kernel
reads, and the only number that supports a memory or latency claim.
`cr_archival` uses Huffman-coded indices — a smaller checkpoint that must be
decoded before use, valid only for storage/transmission claims. Both come out
of every evaluation. Similarly `bpw_target` (compressed matrices only, what the
GPTQ/AWQ tables quote) is reported next to `bpw_model` (whole checkpoint).

The gap is not cosmetic. On GPT-2 at 4 index bits, `bpw_target` is 4.25 but
whole-model CR is only **2.0×**, because the embedding table is a third of the
checkpoint and stays fp16. Any CR headline that omits this is inflated.

**2. Codebook quantization wants per-channel granularity, not group-wise.**
Group-wise (`g=128`) is correct for *affine* quantization, where each group
stores two fp16 scalars. A per-group *codebook* stores `K` fp16 entries: for a
single Llama-2 `down_proj` that is 4096·11008/128 codebooks, about **+2.0 bits
per weight** — the codebook costs half again as much as the indices it serves.
Per-channel costs +0.0625 bpw for the same `K=16`. `tests/test_pipeline.py`
pins both numbers; run the group-wise setting once as an ablation and leave it
off.

**3. Dimensionality is a dial, and it is an ablation axis.**
A fully per-layer encoding of Llama-2-7B is 224 `K` variables + 448 pruning
variables = **672**, well outside where NSGA-II's non-dominated sorting still
gives useful selection pressure. `variables.k_grouping` selects
`global` (1) / `type` (7) / `block` (32) / `block_type` (224). Start at `type`;
when you move to `block_type`, switch `search.algorithm` to `unsga3`.

**4. Pruning thresholds are scale-normalised.**
`prune.mode: sigma` expresses the band in units of each row's weight standard
deviation. Raw thresholds differ by orders of magnitude across layers, which
makes the search space badly conditioned and wastes most of the EA's budget.

## Performance notes

The model is loaded **once** and never reloaded; each candidate overwrites the
live weights in place from an immutable master copy (`MasterWeights`). Two
things dominate the per-evaluation cost, in this order:

- **Forward passes.** Controlled by `data.n_proxy_seq`. Validate the proxy once
  with `evaluate.rank_correlation` over ~30 sampled genomes and report the
  Spearman rho; below about 0.9, raise `n_proxy_seq`. This single number is
  what justifies searching against a cheap signal.
- **Applying the genome.** Uniform binning uses closed-form bin indices rather
  than `torch.searchsorted`. This is worth far more than it looks: on MPS,
  searchsorted over one 2304×768 layer costs ~420 ms against ~0.9 ms for the
  arithmetic form. Applying a whole GPT-2 candidate went from 20 s to 0.4 s.
  `quantile` and `kmeans` binning still need a real search and are
  correspondingly slower.

When pruning is disabled, `(layer, K)` results are deterministic and cached
outright, so an evaluation degenerates to a few memcpys plus the forwards.

## Memory

fp16 weights are 2 bytes/param; the master copy doubles that. Inference only,
`use_cache=False`, batch 1:

| Model | Live + master | RTX 3060 (12 GB) | H100 (80 GB) |
|---|---|---|---|
| 160M–1.4B | 0.6–5.6 GB | yes | yes |
| 2.7B | 11 GB | tight, use `master_device: cpu` | yes |
| 7B | 27 GB | no — `master_device: cpu` + offload | yes |
| 13B | 52 GB | no | yes |
| 70B | 280 GB | no | no — needs layer-wise offload |

Set `model.master_device: cpu` to halve peak VRAM at the cost of one host↔device
copy per layer per candidate.

## Status

Working and tested end to end: quantization, accounting, evaluation, and the
NSGA-II loop. `tests/` pins the entropy-coding bounds, the binning ordering,
and the bit accounting against hand calculations.

Not yet implemented:

- **lm-evaluation-harness integration** for zero-shot tasks (PIQA, ARC, HellaSwag,
  WinoGrande, BoolQ, LAMBADA) and MMLU. Currently perplexity only.
- **A LUT dequant kernel.** Without one, the compression is storage-side; there
  is no wall-clock or VRAM win to report yet. Decide early which claim you are
  making, because it determines whether Huffman belongs in the headline number.
- **Fisher/Hessian-weighted binning**, the sensitivity weighting SqueezeLLM
  uses. The hook is `LayerQuantStats.mse`.
- **Layer-wise offload** for models that exceed one device.

## Licence

[Restricted Academic License](LICENSE) — all rights reserved. You may read the
code and run it to verify its results. Copying, redistribution, derivative
works, commercial use, and use in published or publicly disseminated work all
require prior written permission.

Note that a licence governs this *source code*, not the underlying methods:
copyright protects expression, not ideas, and it cannot prevent independent
development or publication of similar techniques by others. Priority is
established by publication date, not by a licence file.
