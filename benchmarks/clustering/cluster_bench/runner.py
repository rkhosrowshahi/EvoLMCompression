"""One dataset end to end, and the loop over a whole suite.

Order matters and is not arbitrary: search first, then baselines at the K the
search actually reached, then scoring. Fitting the baselines first would force
the matched-K table to interpolate, and interpolating a k-means curve is how
these comparisons quietly become unfalsifiable.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from . import datasets as ds
from . import report
from .baselines import best_per_k, k_ladder, run_baselines
from .config import Config, as_dict
from .genome import Genome, GenomeSpec
from .kmeans import ExactKMeans1D, lloyd_multistart
from .problem import ClusteringProblem
from .search import run_nsga2


def _write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_dataset(name: str, cfg: Config, out_dir: Path, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    dataset = ds.load(name, seed=cfg.seed, standardize=cfg.standardize)
    if verbose:
        print(f"\n=== {dataset.name}  n={dataset.n} d={dataset.d} "
              f"kind={dataset.kind} ===")
        if dataset.notes:
            print(f"    {dataset.notes}")

    g = cfg.genome
    spec = GenomeSpec(
        d=dataset.d, k_min=g.k_min, k_max=min(g.k_max, dataset.n),
        alpha_min=g.alpha_min, alpha_max=g.alpha_max,
        gamma_min=g.gamma_min, gamma_max=g.gamma_max,
        residual_genes=g.residual_genes, u_lo=g.u_lo, u_hi=g.u_hi,
        share_warp=g.share_warp, residual_type=g.residual_type,
        ispline_degree=g.ispline_degree, grid=g.grid, lloyd_iters=g.lloyd_iters)
    genome = Genome(spec)
    # An unset ceiling defaults to the largest K any baseline is fitted at, so
    # the search cannot escape into a region with nothing to compare against --
    # and, in multi-D, cannot reach the all-singletons partition that scores
    # perfectly on every index while being no clustering at all.
    max_k_eff = cfg.search.max_k_eff
    # A FIXED ceiling is not enough -- it has to scale with n. iris has 150
    # points, so a cap of 512 let the search reach ~149 clusters, i.e. almost
    # one point each: MSE 0, DB 0, a one-point front, and a hypervolume of
    # 1.210 (the theoretical maximum) for BOTH methods with nothing separating
    # them. n // 10 keeps ten points per cluster on average, which is the
    # loosest rule under which "cluster" still means something.
    ceiling = max(2, dataset.n // 10)
    max_k_eff = min(max_k_eff or cfg.baselines.match_k_cap, ceiling)
    problem = ClusteringProblem(dataset, genome, cfg.objectives,
                                cfg.silhouette_max_n, cfg.seed,
                                cfg.search.min_k_eff, max_k_eff,
                                cfg.search.min_cluster_size)
    if verbose:
        print(f"    genome: {genome.n_var} variables, objectives "
              f"{list(cfg.objectives)}, clusters constrained to "
              f"[{cfg.search.min_k_eff}, {max_k_eff}]")

    s = cfg.search
    res = run_nsga2(problem, s.pop_size, s.n_gen, s.seed, s.eta_cx, s.eta_mut,
                    s.p_cx, verbose, s.log_every)
    if verbose:
        print(f"    search: {res['n_eval']} evals in {res['seconds']:.1f}s, "
              f"front size {len(res['F'])}")

    # Re-score the front through the full metric set (the search only stored
    # the objectives it was optimizing).
    front_rows = []
    for i, x in enumerate(res["X"]):
        m = problem.score(x)
        front_rows.append({"method": "companding", "idx": i, **m,
                           **{f"gene_{k}": json.dumps(v)
                              for k, v in genome.describe(x).items()}})

    b = cfg.baselines
    # The ladder respects the same ceiling: fitting k-means at a K the search
    # was forbidden to reach produces baseline rows nothing can be matched to.
    ladder = set(k_ladder(b.k_min, min(b.k_max, max_k_eff), b.k_steps))
    ks = set(ladder)
    if b.match_front_k:
        ks |= {int(r["k_eff"]) for r in front_rows
               if 2 <= r["k_eff"] <= min(b.match_k_cap, max_k_eff)}
    if verbose:
        print(f"    baselines at {len(ks)} K values "
              f"[{min(ks)}..{max(ks)}], arms {list(b.arms)} "
              + (f" (sklearn on the {len(ladder)}-rung ladder only)"
                 if b.sklearn_ladder_only else ""))
    base_rows, notes = run_baselines(
        dataset, sorted(ks), b.arms, b.lloyd_n_init, b.sklearn_n_init,
        b.dp_max_n, cfg.silhouette_max_n, cfg.seed, verbose=False,
        sklearn_k=ladder if b.sklearn_ladder_only else None)
    for note in notes:
        if verbose:
            print(f"    note: {note}")

    archive = front_rows + _archive_best(problem)
    comp = report.compare(front_rows, base_rows, cfg.objectives)
    matched = report.matched_k_table(archive, best_per_k(base_rows), "mse")
    # The same comparison repeated against each arm ON ITS OWN. Pooling the
    # arms answers "can companding beat the best k-means anyone would get",
    # which is one fair question; it cannot answer "how does companding compare
    # to Lloyd", which is the question a reader with Lloyd in their pipeline
    # actually has. Both are now reported, and they disagree.
    per_arm = {}
    for arm in sorted({str(r["method"]).replace("kmeans_", "") for r in base_rows}):
        rows_a = [r for r in base_rows if str(r["method"]).endswith(arm)]
        per_arm[arm] = {
            **report.compare(front_rows, rows_a, cfg.objectives),
            "matched_k": report.matched_k_table(archive, best_per_k(rows_a), "mse"),
            "n_runs": len(rows_a),
        }
    curve = report.convergence(res["snapshots"], cfg.objectives,
                               comp["ideal"], comp["nadir"])

    dd = out_dir / dataset.name
    dd.mkdir(parents=True, exist_ok=True)
    _write_csv(dd / "front.csv", front_rows)
    _write_csv(dd / "baselines.csv", base_rows)
    _write_csv(dd / "matched_k.csv", matched)
    _write_csv(dd / "convergence.csv", curve)
    # The whole evaluated population, objectives only. Kept because a figure
    # cannot be redrawn from a front alone -- the front is what SURVIVED, and
    # showing only survivors makes a search look tidier than it was.
    _write_csv(dd / "archive.csv", _archive_rows(problem, cfg.objectives))

    if cfg.figures:
        try:
            report.plot_objective_space(dd / "objective_space.png", dataset.name,
                                        front_rows, base_rows, cfg.objectives,
                                        _archive_rows(problem, cfg.objectives))
            report.plot_convergence(dd / "convergence.png", dataset.name, curve)
            if dataset.d == 1 and len(res["X"]):
                # Draw the median-K front member rather than an extreme one:
                # the endpoints of a distortion/validity front are the
                # degenerate-ish corners (K at its ceiling, or K barely above
                # two) and neither shows what the warp is doing in between.
                keff = np.array([r["k_eff"] for r in front_rows])
                pick = int(np.argsort(keff)[len(keff) // 2])
                kb = int(front_rows[pick]["k_eff"])
                ref = _kmeans_centroids_1d(dataset, kb, b.dp_max_n, cfg.seed)
                st = genome.decode(res["X"][pick])
                _, cent = problem.partition(res["X"][pick])
                report.plot_warp_1d(
                    dd / "warp.png", dataset,
                    (st.ks, st.alphas, st.gammas, st.us), cent, ref,
                    spec.grid, spec.residual_type, spec.ispline_degree)
        except Exception as exc:                      # pragma: no cover
            print(f"    figures failed: {type(exc).__name__}: {exc}")

    summary = {
        "dataset": dataset.name,
        "n": dataset.n, "d": dataset.d, "kind": dataset.kind,
        "k_true": dataset.k_true, "notes": dataset.notes,
        "n_var": genome.n_var,
        "search_seconds": round(res["seconds"], 2),
        "n_eval": res["n_eval"],
        "baseline_notes": notes,
        "n_kmeans_runs": len(base_rows),
        "n_companding_evals": res["n_eval"],
        **comp,
        # Each "best" carries the cluster count it was achieved at. Without it a
        # score is unreadable: k-means reaching DB 0.44 at K=2 and companding
        # reaching 0.21 at K=3 is a very different statement from the same two
        # numbers at K=200, and the K is the first thing anyone asks.
        **_best_with_k(front_rows, base_rows),
        "matched_k": matched,
        "per_arm": per_arm,
        "total_seconds": round(time.perf_counter() - t0, 2),
    }
    (dd / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if verbose:
        _print_verdict(summary, matched)
    return summary


def _extreme(rows, key, best=min):
    """(value, K, smallest cluster) of the best row on `key`.

    The smallest cluster rides along because a validity score is not
    interpretable without it: Davies-Bouldin's best solution is frequently a
    single outlier against everything else, which the score alone hides.
    """
    usable = [r for r in rows if np.isfinite(r.get(key, np.nan))]
    if not usable:
        return None, None, None, None
    pick = best(usable, key=lambda r: r[key])
    small = pick.get("min_cluster_size")
    # Which arm produced it. The k-means column is a best-of-three ENVELOPE
    # recomputed per measure, not one algorithm -- in 1-D the MSE entry is
    # always the exact DP, but the validity entries are sometimes Lloyd or
    # scikit-learn. A reader cannot check the claim without knowing which.
    return (float(pick[key]), int(pick["k_eff"]),
            int(small) if small is not None else None,
            str(pick.get("method", "")).replace("kmeans_", "") or None)


def _best_with_k(front_rows, base_rows) -> dict:
    """Best score per method per measure, each paired with the K that produced it."""
    out = {}
    for label, rows in (("companding", front_rows), ("kmeans", base_rows)):
        for name, key, how in (("mse", "mse", min),
                               ("db", "davies_bouldin", min),
                               ("silhouette", "silhouette", max),
                               # Reported last but read first: the only measure
                               # here that a badly shaped partition cannot fake.
                               ("adjusted_rand", "adjusted_rand", max)):
            v, k, small, arm = _extreme(rows, key, how)
            out[f"best_{name}_{label}"] = v
            out[f"best_{name}_k_{label}"] = k
            out[f"best_{name}_minsize_{label}"] = small
            if label == "kmeans":
                out[f"best_{name}_arm_kmeans"] = arm
    out["kmeans_arms"] = _arm_breakdown(base_rows)
    return out


def _archive_rows(problem, objectives) -> list[dict]:
    """Every evaluated candidate, reduced to what a plot or a table needs.

    Only feasible candidates are kept -- those satisfying the same cluster-count
    constraints the search enforced. The infeasible ones are dominated by
    construction on both axes (an all-singletons partition scores MSE 0 and
    Davies-Bouldin 0) and drawing them would put a dense cloud in the ideal
    corner that no legal solution can reach.
    """
    keep = ("mse", "sse", "davies_bouldin", "neg_silhouette", "k_eff",
            "entropy_bits", "index_bits", "min_cluster_size")
    lo = problem.min_k_eff or 2
    hi = problem.max_k_eff or float("inf")
    out = []
    for r in problem.history:
        if not (lo <= r["k_eff"] <= hi):
            continue
        if r.get("min_cluster_size", 1) < problem.min_cluster_size:
            continue
        out.append({"eval": r["eval"],
                    **{k: r[k] for k in keep if k in r}})
    return out


def _arm_breakdown(base_rows) -> dict:
    """Per-arm bests, so the envelope can always be taken apart again.

    Note the arms are NOT equally sampled: `sklearn` is fitted only on the
    configured K ladder while `dp` and `lloyd` also run at every K the
    companding front reached, so its "best over K" is drawn from a coarser
    grid. It is an independent-implementation check, not a third competitor,
    and `n_runs` here is what makes that visible rather than implied.
    """
    out: dict[str, dict] = {}
    for r in base_rows:
        arm = str(r.get("method", "")).replace("kmeans_", "")
        cur = out.setdefault(arm, {"n_runs": 0, "k_min": None, "k_max": None})
        cur["n_runs"] += 1
        k = int(r["k_eff"])
        cur["k_min"] = k if cur["k_min"] is None else min(cur["k_min"], k)
        cur["k_max"] = k if cur["k_max"] is None else max(cur["k_max"], k)
        for name, key, how in (("mse", "mse", min),
                               ("db", "davies_bouldin", min),
                               ("silhouette", "silhouette", max),
                               ("adjusted_rand", "adjusted_rand", max)):
            v = r.get(key)
            if v is None or not np.isfinite(v):
                continue
            prev = cur.get(f"best_{name}")
            if prev is None or how(float(v), prev) == float(v):
                cur[f"best_{name}"] = float(v)
                cur[f"best_{name}_k"] = k
    return out


def _archive_best(problem, metric: str = "mse") -> list[dict]:
    """Best archive candidate at each K_eff, re-scored with the full metric set.

    The matched-K table asks what companding CAN do at a given number of
    clusters, and the final front does not answer that: it keeps only points
    that are Pareto-optimal on the objective PAIR, so at a K whose front member
    is a validity-favouring compromise the excess-distortion column would
    report the cost of that trade rather than the cost of the method. The whole
    evaluation archive does answer it -- and since the archive skipped
    silhouette during the search, the handful of rows that survive this filter
    get re-scored properly here.
    """
    best: dict[int, dict] = {}
    for r in problem.history:
        k = int(r["k_eff"])
        if k < 2:
            continue
        if k not in best or r[metric] < best[k][metric]:
            best[k] = r
    return [{"method": "companding_archive", **problem.score(r["_x"])}
            for r in best.values()]


def _kmeans_centroids_1d(dataset, k, dp_max_n, seed):
    """Reference centroids for the warp figure -- exact if the DP can run."""
    x = dataset.x[:, 0]
    try:
        return ExactKMeans1D(x, k, max_n=dp_max_n, seed=seed).fit(k)[1][:, 0]
    except Exception:
        return lloyd_multistart(x, k, n_init=5, seed=seed)[1][:, 0]


def _print_verdict(summary, matched):
    """Three sentences per dataset, each saying what it means in words."""
    if summary.get("degenerate_box"):
        print("    WARNING     both fronts collapsed to the same single point; "
              "the coverage numbers below are the maximum by construction and "
              "compare nothing. Lower search.max_k_eff for this dataset.")
    if matched:
        ex = [m["excess_pct"] for m in matched if np.isfinite(m["excess_pct"])]
        if ex:
            print(f"    tightness   at the same cluster count, companding's error is "
                  f"{float(np.median(ex)):+.1f}% vs k-means (median over "
                  f"{len(ex)} shared K; best case {min(ex):+.1f}%, "
                  f"worst {max(ex):+.1f}%)")
        wins = sum(1 for m in matched if m["companding_db"] < m["kmeans_db"])
        print(f"    separation  companding has the better Davies-Bouldin at "
              f"{wins} of {len(matched)} shared cluster counts")
    print(f"    reach       {summary['companding_only_points']} companding "
          f"solutions are unreachable by k-means at any K; "
          f"{summary['kmeans_only_points']} the other way round")
    print(f"    coverage    companding front covers {summary['hv_companding']:.3f} "
          f"of the shared objective box, k-means {summary['hv_kmeans']:.3f}")


def run_suite(cfg: Config, out_dir: Path, verbose: bool = True) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(as_dict(cfg), indent=2))
    names = ds.resolve(cfg.datasets)
    summaries = []
    for name in names:
        try:
            summaries.append(run_dataset(name, cfg, out_dir, verbose))
        except FileNotFoundError as exc:
            # A dataset that needs something absent (scikit-learn, the parent
            # project's weight cache) is skipped loudly and the suite goes on.
            print(f"\n=== {name}: SKIPPED -- {exc}")
        except Exception as exc:                      # pragma: no cover
            print(f"\n=== {name}: FAILED -- {type(exc).__name__}: {exc}")
    _write_suite_table(out_dir, summaries)
    return summaries


def _write_suite_table(out_dir: Path, summaries):
    rows = report.suite_rows(summaries)
    _write_csv(out_dir / "suite.csv", rows)
    (out_dir / "suite.json").write_text(
        json.dumps(summaries, indent=2, default=float))
    if rows:
        print()
        print(report.format_per_arm_tables(summaries))
        print()
        print(report.format_suite_tables(summaries))
