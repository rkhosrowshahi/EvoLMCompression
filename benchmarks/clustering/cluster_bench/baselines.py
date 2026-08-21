"""Run the k-means arms and score them with the same code path as companding.

Two K sweeps get unioned:

  configured   a fixed geometric ladder, so every dataset has a baseline curve
               regardless of where the search ended up;
  matched      every K_eff that actually appears on the companding front, so
               the head-to-head "at the same number of clusters" table has no
               interpolation in it.

The matched half is why baselines run AFTER the search rather than alongside
it. Comparing a front against a baseline sampled at different K and then
interpolating is the standard way these comparisons go wrong.
"""

from __future__ import annotations

import time

import numpy as np

from .kmeans import (ExactKMeans1D, lloyd_multistart, sklearn_available,
                     sklearn_kmeans)
from .metrics import evaluate


def k_ladder(k_min: int, k_max: int, n: int = 8) -> list[int]:
    """Geometric ladder of distinct integers -- log spacing, as in the genome."""
    raw = np.geomspace(max(k_min, 2), max(k_max, k_min + 1), n)
    return sorted({int(round(v)) for v in raw})


def run_baselines(dataset, k_values, arms=("dp", "lloyd", "sklearn"),
                  lloyd_n_init: int = 10, sklearn_n_init: int = 10,
                  dp_max_n: int = 4000, silhouette_max_n: int = 2000,
                  seed: int = 0, verbose: bool = True, sklearn_k=None):
    """Fit every requested arm at every K. Returns (rows, notes).

    `sklearn_k` restricts the scikit-learn arm to a subset of `k_values`. It
    exists because that arm is the slowest by a wide margin -- ten restarts of
    a general-purpose k-means at K=256 -- and it is a cross-implementation
    check, not the reference: the DP is the reference in 1-D and multi-start
    Lloyd is the practitioner's baseline, and both are cheap enough to fit at
    every matched K. Passing the ladder here keeps the check meaningful while
    the matched-K comparison stays complete.

    `rows` are one dict per (arm, K) with exactly the keys `metrics.evaluate`
    produces, so they drop straight into the same table and the same plot as
    the companding front. `notes` records anything that was skipped -- a
    missing scikit-learn, or a DP that had to subsample -- because a silently
    absent baseline is worse than no baseline.
    """
    x = dataset.x
    x1d = x[:, 0] if x.shape[1] == 1 else None
    k_values = sorted({int(k) for k in k_values if k >= 2})
    sk_k = set(k_values) if sklearn_k is None else {int(k) for k in sklearn_k}
    rows, notes = [], []

    exact = None
    if "dp" in arms:
        if x1d is None:
            notes.append("dp: skipped -- exact k-means DP is 1-D only "
                         "(k-means is NP-hard in general)")
        else:
            t0 = time.perf_counter()
            exact = ExactKMeans1D(x1d, max(k_values), max_n=dp_max_n, seed=seed)
            notes.append(
                f"dp: tables built in {time.perf_counter() - t0:.1f}s, "
                f"{'globally optimal' if exact.exact else f'subsampled to {dp_max_n} distinct values then Lloyd-polished on the full data'}")

    have_sklearn = sklearn_available()
    if "sklearn" in arms and not have_sklearn:
        notes.append("sklearn: skipped -- scikit-learn is not installed "
                     "(pip install scikit-learn)")

    for k in k_values:
        if k > dataset.n:
            continue
        fits = {}
        if exact is not None:
            fits["dp"] = exact.fit(k)
        if "lloyd" in arms:
            fits["lloyd"] = lloyd_multistart(x, k, n_init=lloyd_n_init, seed=seed)
        if "sklearn" in arms and have_sklearn and k in sk_k:
            fits["sklearn"] = sklearn_kmeans(x, k, n_init=sklearn_n_init, seed=seed)

        for arm, (labels, cent) in fits.items():
            m = evaluate(x, labels, cent, silhouette_max_n, seed,
                         y_true=dataset.y_true)
            rows.append({"method": f"kmeans_{arm}", "k_requested": int(k), **m})
        if verbose:
            best = min((r for r in rows if r["k_requested"] == k),
                       key=lambda r: r["sse"], default=None)
            if best is not None:
                print(f"    K={k:>4}  best sse={best['sse']:.6g}  "
                      f"db={best['davies_bouldin']:.4f}  "
                      f"sil={best['silhouette']:+.4f}  [{best['method']}]")
    return rows, notes


def best_per_k(rows) -> dict:
    """Lowest-SSE baseline at each K -- the yardstick for the excess-MSE table.

    Taking the best across arms, not a single arm, is deliberate: Lloyd's
    result depends on its seeding, and quoting a companding win against
    whichever arm happened to stall is exactly the artefact this benchmark
    exists to avoid.
    """
    out: dict[int, dict] = {}
    for r in rows:
        k = int(r["k_eff"])
        if k not in out or r["sse"] < out[k]["sse"]:
            out[k] = r
    return out
