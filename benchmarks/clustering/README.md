# Companding + NSGA-II vs. k-means on clustering benchmarks

A self-contained benchmark, **out of scope of EvoLMCompression** and importing
none of it. The question is narrow and stated on purpose:

> Companding quantization is used in this project as a *codebook* method.
> Judged as a *clustering* method, against k-means on standard benchmark
> problems, how does it do?

The answer is not expected to be "it wins". k-means minimizes SSE by
construction, and in 1-D its global optimum is computable exactly, so on the
distortion axis alone the result is decided before the run starts. What the
benchmark is actually for is measuring the *size* of that gap, finding the
regimes where it closes, and testing whether a multi-objective search buys
anything on a second axis that k-means does not optimize at all.

## Install and run

```bash
pip install -r requirements.txt
```

```bash
python scripts/run_benchmark.py configs/quick.yaml
```

`quick.yaml` is a wiring check — a few minutes, two 1-D problems and two
multi-D ones at toy budgets. The real experiments:

| config | what it asks |
|---|---|
| `suite_1d.yaml` | primary result: 1-D problems, distortion vs. Davies-Bouldin, exact k-means as the reference |
| `suite_1d_silhouette.yaml` | does a Davies-Bouldin win survive switching the validity index? |
| `suite_1d_rate.yaml` | the parent project's axes: distortion vs. label entropy (bits), where companding controls something k-means does not |
| `suite_md.yaml` | secondary result / limitations: per-dimension companding as a product quantizer on the standard multi-D benchmarks |

```bash
python scripts/run_benchmark.py configs/suite_1d.yaml
python scripts/summarize.py results/suite_1d-*/
```

```bash
python -m pytest
```

## What is being compared

**Companding.** A monotone warp `F: R -> [0,1]`, then uniform binning of `F(x)`.
`F = F_residual o F_gamma`, where `F_gamma` is the Bennett/Panter-Dite
density-matched backbone (level density `~ p(x)^gamma`, clipped to `+- alpha`
standard deviations) and `F_residual` is a monotone spline correction driven by
`M` genes. Codewords are bin means of the original values, so `F` is only ever
evaluated forward. This is a numpy port of `evolmc/quantize.py`, with one
deliberate change: the clip window is centred on the sample mean rather than on
zero, because clustering data is not zero-mean.

NSGA-II searches `(K, alpha, gamma, u_1..u_M)` — 9 variables in 1-D. In
multi-D each dimension gets its own `K_d` and, unless `share_warp` is set, its
own warp; cells are the intersection of the per-axis bins, i.e. a product
quantizer, and only **occupied** cells count as clusters.

**k-means.** Three arms, because "k-means" is not one number:

- **exact DP** — globally optimal 1-D k-means (Ckmeans.1d.dp) by concave-Monge
  dynamic programming with divide-and-conquer. Verified against brute-force
  enumeration in the tests. 1-D only, and only while the number of *distinct*
  values stays under `baselines.dp_max_n` — the 1-D generators are sized
  (`datasets.N_1D = 4000`) so that it does. Past that it subsamples, polishes
  with Lloyd, and says so in the run's notes; the reference is then strong but
  no longer provably optimal.
- **Lloyd, multi-start** — best of `n_init`, cycling k-means++, random,
  quantile and uniform seeding. What a practitioner actually gets.
- **scikit-learn** — `KMeans`, as an independent implementation check. Skipped
  with a printed note if scikit-learn is absent.

The matched-K table scores companding against the **best** of the arms at each
K, never against whichever one stalled.

## What is being measured

Objectives (both minimized, set per config):

| | |
|---|---|
| `mse` / `sse` | distortion — what k-means minimizes |
| `davies_bouldin` | cluster validity, default second axis |
| `neg_silhouette` | cluster validity, `O(n^2)`, subsampled |
| `entropy_bits` | label entropy — the rate a real entropy coder would pay |
| `k_eff` | occupied clusters |

Reported but never optimized: `silhouette`, `calinski_harabasz`, `index_bits`,
`min_cluster_size` and — where the dataset has generating labels — the
**adjusted Rand index**. ARI is the only measure here that is not a function of
partition shape, and the tables print it beside every internal index. Neither
method is shown the true labels; it is scored afterwards.
Every method is scored by the same function on the same partition
representation, so nothing is measured two ways.

Three separate verdicts get written, because they can disagree:

- **attainment** — how many points of each front the other cannot reach, by
  plain Pareto dominance. No normalization, no reference point, nothing to
  argue about. The most robust of the three.
- **hypervolume** — each front normalized against the ideal and nadir of their
  *union*, so it is a share of the same box.
- **matched K** — excess distortion at identical occupied-cluster counts,
  against the best baseline. In 1-D that baseline is the true optimum, so this
  is an absolute number.

## Reading a results directory

```
results/<name>-<timestamp>/
  config.json            what actually ran
  suite.csv / suite.json one row per dataset
  <dataset>/
    front.csv            the Pareto front, all metrics, decoded genomes
    baselines.csv        every k-means arm at every K
    matched_k.csv        head-to-head at identical K_eff
    convergence.csv      per-generation hypervolume on the shared box
    objective_space.png  the headline figure
    convergence.png
    warp.png             1-D only: density, learned warp, decision boundaries
```

`warp.png` is the one worth looking at first. A companding front that ties
k-means on the numbers but reproduces its decision boundaries is a different
finding from one that gets there by a different partition.

## Caveats, stated up front

- **Companding cannot beat exact k-means on SSE at equal K in 1-D.** The DP is
  the optimum. Any such result in the output is a bug — or a sign the DP
  subsampled, which the notes will say. The tests assert the DP is never worse
  than Lloyd for exactly this reason.
- **Every internal validity index prefers "one outlier versus everything
  else".** Measured, not theorised: on the gaussian set a 1-vs-3999 split scores
  Davies-Bouldin 0.206 where balanced k-means scores 0.594 at the same K — a
  singleton has zero spread and DB is a ratio of spreads. The silhouette does
  **not** rescue it (0.85 against 0.56 on the same pair). So a DB or silhouette
  win means nothing on its own, and the tables print `min=` — the smallest
  cluster behind that score — next to every one of them, plus ARI, which rates
  that split at chance. `search.min_cluster_size` can exclude the whole family;
  it defaults to 1 (off) because the behaviour is a finding about the indices
  and burying it would be worse than reporting it.
- **A cluster-count ceiling is mandatory in multi-D**, and `search.max_k_eff`
  defaults to `baselines.match_k_cap` for that reason. A product quantizer over
  32 axes can give every point its own cell, which scores MSE = 0 *and*
  Davies-Bouldin = 0 — a perfect score on both objectives, dominating the whole
  front, and not a clustering. The metrics are right to report it; the search
  is what has to exclude it.
- **`K_eff` is not `K`.** Empty bins are dropped, so a genome asking for
  `K=256` may deliver 180 clusters — and a multi-D product quantizer with
  `k_max=32` on two axes can deliver several hundred. Everything is reported
  and matched on `K_eff`.
- **The silhouette is subsampled** on a *fixed* index set (`silhouette_max_n`),
  shared by every method on a dataset. A resampled subsample would make the
  objective stochastic and the search would chase sampling noise.
- **Multi-D is a limitations study.** A separable quantizer cannot represent a
  rotated cluster. `birch_grid` is the control: it is axis-aligned and
  product-shaped, so it is companding's best case, and a loss there is decisive.
- **Real datasets need scikit-learn** (`iris`, `wine`, `breast_cancer`,
  `digits`). The synthetic generators — including the Fränti-style S-sets,
  A-sets, Unbalance and DIM-sets — are always available and run offline.
- **`gpt2_weights`** reads the parent project's `.cache/gpt2_all_targets.npz`
  if it is there. It is not a clustering benchmark; it is the real target, kept
  as a sanity check that the synthetic suite resembles the actual problem.

## Layout

| | |
|---|---|
| `cluster_bench/companding.py` | the warp, the assignment, the product quantizer |
| `cluster_bench/kmeans.py` | exact 1-D DP, Lloyd multi-start, sklearn wrapper |
| `cluster_bench/metrics.py` | distortion and validity, method-blind |
| `cluster_bench/datasets.py` | both suites, all generated offline |
| `cluster_bench/genome.py` | decision-variable layout |
| `cluster_bench/problem.py` | the pymoo problem |
| `cluster_bench/search.py` | NSGA-II driver |
| `cluster_bench/baselines.py` | the k-means sweep, matched to the front's K |
| `cluster_bench/report.py` | attainment, hypervolume, matched-K, figures |
| `cluster_bench/runner.py` | one dataset end to end, and the suite loop |
