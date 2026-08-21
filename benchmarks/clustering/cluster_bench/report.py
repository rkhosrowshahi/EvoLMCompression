"""Scoring the comparison, and drawing it.

Three questions, answered separately because they can disagree:

  attainment   does either method reach points the other cannot? Counted by
               plain Pareto dominance on the objective pair, with no
               normalization and no reference point to argue about. This is the
               claim that survives the most scrutiny.
  hypervolume  how much of the objective space does each method's front cover?
               Both fronts are normalized against the ideal and nadir of their
               UNION, so the number is a like-for-like share of the same box.
               A front measured against its own bounds would be meaningless.
  matched K    at exactly the same number of occupied clusters, how much worse
               is companding on the distortion axis? In 1-D the reference is
               the DP optimum, so this is an absolute statement, not a
               comparison between two heuristics.

Degenerate candidates -- fewer than two occupied clusters, which `metrics`
scores at WORST_DB -- are dropped before any of this. They are legal
quantizers and non-clusterings, and leaving them in would let a method inflate
its hypervolume with a corner point that means nothing.
"""

from __future__ import annotations

import numpy as np

from .metrics import WORST_DB, WORST_SIL

_DEGENERATE = {"davies_bouldin": WORST_DB, "neg_silhouette": WORST_SIL}


def objective_matrix(rows, objectives) -> np.ndarray:
    return np.array([[float(r[o]) for o in objectives] for r in rows],
                    dtype=np.float64) if rows else np.empty((0, len(objectives)))


def drop_degenerate(rows, objectives):
    """Remove candidates that are not clusterings at all."""
    keep = []
    for r in rows:
        if int(r.get("k_eff", 2)) < 2:
            continue
        if any(o in _DEGENERATE and float(r[o]) >= _DEGENERATE[o] for o in objectives):
            continue
        if not all(np.isfinite(float(r[o])) for o in objectives):
            continue
        keep.append(r)
    return keep


def nondominated(f: np.ndarray) -> np.ndarray:
    """Boolean mask of the Pareto-optimal rows of a minimization matrix."""
    n = len(f)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def dominance_counts(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    """(a-only, b-only, tied) -- how many points of each front the other cannot reach.

    "a-only" counts points of `a` that no point of `b` weakly dominates. It is
    the direct answer to "can this method do something the other cannot", and
    it does not depend on any scaling choice.
    """
    if len(a) == 0 or len(b) == 0:
        return len(a), len(b), 0
    a_only = b_only = tied = 0
    for p in a:
        dom = np.all(b <= p, axis=1) & np.any(b < p, axis=1)
        eq = np.all(b == p, axis=1)
        if dom.any():
            pass
        elif eq.any():
            tied += 1
        else:
            a_only += 1
    for p in b:
        dom = np.all(a <= p, axis=1) & np.any(a < p, axis=1)
        if not dom.any() and not np.all(a == p, axis=1).any():
            b_only += 1
    return a_only, b_only, tied


def hypervolume(f: np.ndarray, ideal: np.ndarray, nadir: np.ndarray,
                ref: float = 1.1) -> float:
    """Normalized hypervolume against a shared box. 0 when the front is empty."""
    if len(f) == 0:
        return 0.0
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    z = (f - ideal) / span
    z = z[np.all(z <= ref, axis=1)]
    if len(z) == 0:
        return 0.0
    try:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=np.full(f.shape[1], ref))(z))
    except ImportError:                        # pragma: no cover
        return float("nan")


def compare(front_rows, baseline_rows, objectives) -> dict:
    """Everything the results file says about one dataset."""
    fr = drop_degenerate(front_rows, objectives)
    br = drop_degenerate(baseline_rows, objectives)
    fa = objective_matrix(fr, objectives)
    ba = objective_matrix(br, objectives)
    fa = fa[nondominated(fa)] if len(fa) else fa
    ba_nd = ba[nondominated(ba)] if len(ba) else ba

    if len(fa) or len(ba_nd):
        allf = np.vstack([m for m in (fa, ba_nd) if len(m)])
        ideal, nadir = allf.min(axis=0), allf.max(axis=0)
    else:
        ideal = nadir = np.zeros(len(objectives))

    # When both fronts collapse to the same single point the box has zero
    # width and every hypervolume comes out at the maximum, ref^n_obj, for
    # everyone -- which reads as a perfect tie rather than as "there was
    # nothing to compare". Flag it; the runner prints a warning.
    degenerate = bool(len(fa) and len(ba_nd)) and         bool(np.all(nadir - ideal <= 1e-12))

    a_only, b_only, tied = dominance_counts(fa, ba_nd)
    return {
        "objectives": list(objectives),
        "degenerate_box": degenerate,
        "n_front_companding": int(len(fa)),
        "n_front_kmeans": int(len(ba_nd)),
        "hv_companding": hypervolume(fa, ideal, nadir),
        "hv_kmeans": hypervolume(ba_nd, ideal, nadir),
        "companding_only_points": int(a_only),
        "kmeans_only_points": int(b_only),
        "tied_points": int(tied),
        "ideal": [float(v) for v in ideal],
        "nadir": [float(v) for v in nadir],
    }


def matched_k_table(front_rows, best_baseline, metric: str = "mse") -> list[dict]:
    """Companding vs. the strongest baseline at identical K_eff.

    Only K values that BOTH arms actually produced appear. The companding entry
    at each K is its best on `metric` across everything passed in -- callers
    hand over the whole evaluation archive, not just the final front, so this
    reports what companding CAN do at that K rather than what the front's
    representative there happened to trade away.
    """
    by_k: dict[int, dict] = {}
    for r in front_rows:
        k = int(r["k_eff"])
        if k < 2:
            continue
        if k not in by_k or r[metric] < by_k[k][metric]:
            by_k[k] = r

    out = []
    for k in sorted(set(by_k) & set(best_baseline)):
        c, b = by_k[k], best_baseline[k]
        ref = float(b[metric])
        out.append({
            "k_eff": k,
            "baseline_method": b["method"],
            f"kmeans_{metric}": ref,
            f"companding_{metric}": float(c[metric]),
            "excess_pct": 100.0 * (float(c[metric]) / ref - 1.0) if ref > 0 else float("nan"),
            "kmeans_db": float(b["davies_bouldin"]),
            "companding_db": float(c["davies_bouldin"]),
            "kmeans_silhouette": float(b["silhouette"]),
            "companding_silhouette": float(c["silhouette"]),
            "kmeans_entropy_bits": float(b["entropy_bits"]),
            "companding_entropy_bits": float(c["entropy_bits"]),
        })
    return out


def convergence(snapshots, objectives, ideal, nadir) -> list[dict]:
    """Per-generation hypervolume against the SAME box the comparison used.

    Scored after the fact, which is the only way the curve and the head-to-head
    number can be read on the same scale.
    """
    ideal, nadir = np.asarray(ideal), np.asarray(nadir)
    out = []
    for s in snapshots:
        f = np.asarray(s["F"], dtype=np.float64)
        finite = f[np.all(np.isfinite(f), axis=1)]
        for i, o in enumerate(objectives):
            if o in _DEGENERATE and len(finite):
                finite = finite[finite[:, i] < _DEGENERATE[o]]
        nd = finite[nondominated(finite)] if len(finite) else finite
        out.append({"gen": s["gen"], "n_eval": s["n_eval"], "t": s["t"],
                    "hv": hypervolume(nd, ideal, nadir)})
    return out


# --------------------------------------------------------------------------
# the suite table
# --------------------------------------------------------------------------
#
# Split in two on purpose. The first table answers "how good are the
# clusterings each method produced"; the second answers "what can each method
# reach that the other cannot". Those are different questions with different
# units, and cramming them into one row of abbreviations is what made the
# earlier version unreadable. Every heading names its quantity and its
# direction, and the legend spells out the rest -- a table nobody can read
# without the source next to them is not a result.


def _fmt(v, spec="{:.4g}"):
    if v is None or v == "" or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return spec.format(v) if isinstance(v, (int, float)) else str(v)


def _with_k(value, k, spec="{:.4g}", smallest=None) -> str:
    """Score, the cluster count behind it, and optionally its smallest cluster.

    Neither annotation is decoration. A Davies-Bouldin of 0.21 at three clusters
    and the same figure at two hundred are opposite claims. And a DB reached
    with a one-point cluster is not a clustering result at all -- "min=1" is the
    tell, printed right next to the number it undermines.
    """
    v = _fmt(value, spec)
    if k is None or v == "-":
        return v
    tail = "" if smallest is None else f" min={smallest}"
    return f"{v} @K={k}{tail}"


def _render(headers, rows, markdown=False):
    cells = [[str(c) for c in r] for r in rows]
    w = [max(len(h), max((len(r[i]) for r in cells), default=0))
         for i, h in enumerate(headers)]
    if markdown:
        line = lambda cs: "| " + " | ".join(c.ljust(x) for c, x in zip(cs, w)) + " |"
        sep = "|-" + "-|-".join("-" * x for x in w) + "-|"
    else:
        line = lambda cs: "  ".join(c.ljust(x) for c, x in zip(cs, w))
        sep = "  ".join("-" * x for x in w)
    return "\n".join([line(headers), sep] + [line(c) for c in cells])


def suite_rows(summaries) -> list[dict]:
    """One machine-readable row per dataset. This is what suite.csv holds."""
    out = []
    for s in summaries:
        ex = [m["excess_pct"] for m in s["matched_k"]
              if np.isfinite(m["excess_pct"])]
        out.append({
            "dataset": s["dataset"], "n": s["n"], "d": s["d"], "kind": s["kind"],
            "k_true": s.get("k_true"),
            "best_mse_kmeans": s["best_mse_kmeans"],
            "best_mse_k_kmeans": s.get("best_mse_k_kmeans"),
            "best_mse_companding": s["best_mse_companding"],
            "best_mse_k_companding": s.get("best_mse_k_companding"),
            "median_excess_mse_pct": float(np.median(ex)) if ex else None,
            "worst_excess_mse_pct": max(ex) if ex else None,
            "matched_k_count": len(ex),
            "best_db_kmeans": s["best_db_kmeans"],
            "best_db_k_kmeans": s.get("best_db_k_kmeans"),
            "best_db_minsize_kmeans": s.get("best_db_minsize_kmeans"),
            "best_db_companding": s["best_db_companding"],
            "best_db_k_companding": s.get("best_db_k_companding"),
            "best_db_minsize_companding": s.get("best_db_minsize_companding"),
            "best_silhouette_kmeans": s["best_silhouette_kmeans"],
            "best_silhouette_k_kmeans": s.get("best_silhouette_k_kmeans"),
            "best_silhouette_companding": s["best_silhouette_companding"],
            "best_silhouette_k_companding": s.get("best_silhouette_k_companding"),
            "best_silhouette_minsize_kmeans": s.get("best_silhouette_minsize_kmeans"),
            "best_silhouette_minsize_companding":
                s.get("best_silhouette_minsize_companding"),
            "best_ari_kmeans": s.get("best_adjusted_rand_kmeans"),
            "best_ari_k_kmeans": s.get("best_adjusted_rand_k_kmeans"),
            "best_ari_companding": s.get("best_adjusted_rand_companding"),
            "best_ari_k_companding": s.get("best_adjusted_rand_k_companding"),
            "kmeans_runs": s.get("n_kmeans_runs"),
            "companding_evals": s.get("n_eval"),
            "front_points_companding": s["n_front_companding"],
            "front_points_kmeans": s["n_front_kmeans"],
            "only_companding": s["companding_only_points"],
            "only_kmeans": s["kmeans_only_points"],
            "hv_companding": s["hv_companding"],
            "hv_kmeans": s["hv_kmeans"],
            "search_seconds": s["search_seconds"],
        })
    return out


QUALITY_LEGEND = """\
  true K  the generating process's own cluster count, where it has one.  Neither
        method is told it; it is here to judge what they chose.
  @K    the number of occupied clusters that score was achieved at.  Every
        method was swept over many K; this is where its best landed.
  MSE   mean squared distance from a point to its cluster centre.  LOWER is better.
        This is the quantity k-means minimizes, so k-means should win this column.
  cost@K  extra MSE companding pays at the SAME number of clusters, median over
        every K both methods produced.  +50% means "half again as much error".
  DB    Davies-Bouldin separation index -- cluster spread over cluster distance.
        LOWER is better.  Neither k-means arm optimizes this.
  min=  points in the SMALLEST cluster of that DB-best solution.  Read it before
        believing the DB.  A singleton has zero spread and DB is a ratio of
        spreads, so "one outlier vs everything else" beats any honest partition;
        min=1 means the score is an artifact of the index, not a clustering.
  silh  silhouette width, is a point nearer its own cluster than the next one.
        HIGHER is better, range -1 to +1.  Also unoptimized by k-means -- and
        also inflated by an outlier-vs-rest split, so read its min= too.
  ARI   adjusted Rand index against the GENERATING labels, where the dataset has
        them.  1 = exact recovery, 0 = no better than chance.  The only column
        here that cannot be gamed by partition shape: the outlier-vs-rest split
        that wins on both DB and silhouette scores ARI 0.  Neither method is
        shown the true labels; this is scored afterwards."""

REACH_LEGEND = """\
  runs / evals  how many clusterings each method actually produced: one k-means
                run per cluster count per arm, one companding evaluation per
                candidate genome.
  on its front  how many of those survive as non-dominated on the two searched
                axes.  k-means usually collapses to a handful -- raising K cuts
                error and worsens separation together, so most of its runs are
                beaten outright by some other K.
  only X        clusterings on X's front that the OTHER method never reaches at
                any K.  Plain Pareto dominance -- no scaling, nothing to tune,
                and the claim that survives the most scrutiny.
  coverage      hypervolume of each front, both normalized against the ideal and
                nadir of their union so they share one box.  HIGHER is better."""


def format_suite_tables(summaries, markdown=False) -> str:
    """The human-facing summary: two tables and the legend that decodes them."""
    rows = suite_rows(summaries)
    if not rows:
        return "(no datasets completed)"

    quality_head = ["dataset", "n", "d", "true K", "MSE k-means",
                    "MSE companding", "cost@K", "shared K", "DB k-means",
                    "DB companding", "silh k-means", "silh companding",
                    "ARI k-means", "ARI companding"]
    quality = [[r["dataset"], r["n"], r["d"], _fmt(r["k_true"], "{:d}"),
                _with_k(r["best_mse_kmeans"], r["best_mse_k_kmeans"], "{:.4g}"),
                _with_k(r["best_mse_companding"], r["best_mse_k_companding"], "{:.4g}"),
                _fmt(r["median_excess_mse_pct"], "{:+.1f}%"),
                r["matched_k_count"],
                _with_k(r["best_db_kmeans"], r["best_db_k_kmeans"], "{:.3f}",
                        r["best_db_minsize_kmeans"]),
                _with_k(r["best_db_companding"], r["best_db_k_companding"],
                        "{:.3f}", r["best_db_minsize_companding"]),
                _with_k(r["best_silhouette_kmeans"],
                        r["best_silhouette_k_kmeans"], "{:+.3f}",
                        r["best_silhouette_minsize_kmeans"]),
                _with_k(r["best_silhouette_companding"],
                        r["best_silhouette_k_companding"], "{:+.3f}",
                        r["best_silhouette_minsize_companding"]),
                _with_k(r["best_ari_kmeans"], r["best_ari_k_kmeans"], "{:.3f}"),
                _with_k(r["best_ari_companding"], r["best_ari_k_companding"],
                        "{:.3f}")]
               for r in rows]

    reach_head = ["dataset", "k-means runs", "on its front",
                  "companding evals", "on its front", "only companding",
                  "only k-means", "coverage companding", "coverage k-means"]
    reach = [[r["dataset"], _fmt(r["kmeans_runs"], "{:d}"),
              r["front_points_kmeans"], _fmt(r["companding_evals"], "{:d}"),
              r["front_points_companding"], r["only_companding"],
              r["only_kmeans"], _fmt(r["hv_companding"], "{:.3f}"),
              _fmt(r["hv_kmeans"], "{:.3f}")] for r in rows]

    h = "### " if markdown else ""
    return "\n".join([
        f"{h}Cluster quality", "",
        _render(quality_head, quality, markdown), "",
        QUALITY_LEGEND, "",
        f"{h}What each method can reach", "",
        _render(reach_head, reach, markdown), "",
        REACH_LEGEND,
    ])


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

_LABEL = {
    "mse": "MSE (distortion)",
    "sse": "SSE (distortion)",
    "davies_bouldin": "Davies-Bouldin (lower better)",
    "neg_silhouette": "-silhouette (lower better)",
    "k_eff": "clusters used",
    "entropy_bits": "label entropy (bits/sample)",
    "index_bits": "index width (bits)",
}


def plot_objective_space(path, dataset_name, front_rows, baseline_rows,
                         objectives):
    """The headline figure: both fronts on the two axes that were optimized."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = drop_degenerate(front_rows, objectives)
    br = drop_degenerate(baseline_rows, objectives)
    fa, ba = objective_matrix(fr, objectives), objective_matrix(br, objectives)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    styles = {"kmeans_dp": ("o", "#1b7837", "k-means (exact DP)"),
              "kmeans_lloyd": ("s", "#2166ac", "k-means (Lloyd, multi-start)"),
              "kmeans_sklearn": ("^", "#7fbf7b", "k-means (scikit-learn)")}
    for method, (marker, colour, label) in styles.items():
        idx = [i for i, r in enumerate(br) if r["method"] == method]
        if idx:
            ax.plot(ba[idx, 0], ba[idx, 1], marker, ms=5, color=colour,
                    label=label, alpha=0.85, ls="none")
    if len(fa):
        nd = nondominated(fa)
        ax.plot(fa[~nd, 0], fa[~nd, 1], ".", ms=3, color="#d6a0c0", alpha=0.5,
                ls="none", label="companding (dominated)")
        pts = fa[nd]
        pts = pts[np.argsort(pts[:, 0])]
        ax.plot(pts[:, 0], pts[:, 1], "-o", ms=4, lw=1.4, color="#b2182b",
                label="companding front (NSGA-II)")

    if objectives[0] in ("mse", "sse"):
        ax.set_xscale("log")
    ax.set_xlabel(_LABEL.get(objectives[0], objectives[0]))
    ax.set_ylabel(_LABEL.get(objectives[1], objectives[1]))
    ax.set_title(f"{dataset_name}: companding vs. k-means")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(path, dataset_name, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=140)
    ax.plot([c["n_eval"] for c in curve], [c["hv"] for c in curve],
            lw=1.6, color="#b2182b")
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("hypervolume (shared box)")
    ax.set_title(f"{dataset_name}: NSGA-II convergence")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_warp_1d(path, dataset, problem, x_best, kmeans_centroids=None):
    """What the search actually learned: the warp, and where it puts its edges.

    Three panels, because the interesting claim is geometric rather than
    numeric -- the data density, the monotone map the search chose, and the
    resulting decision boundaries next to k-means'. A companding front that
    ties k-means on the metrics but reproduces its boundaries is a different
    story from one that gets there another way.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .companding import companding_forward

    x = dataset.x[:, 0]
    st = problem.genome.decode(x_best)
    spec = problem.genome.spec
    grid = np.linspace(x.min(), x.max(), 1000)
    f = companding_forward(grid, float(st.alphas[0]), float(st.gammas[0]),
                           st.us[0], spec.grid, spec.residual_type,
                           spec.ispline_degree)
    labels, cent = problem.partition(x_best)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.6), dpi=140, sharex=True)
    axes[0].hist(x, bins=200, color="#4d4d4d", alpha=0.8)
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{dataset.name}: density, learned warp, decision boundaries")

    axes[1].plot(grid, f, lw=1.6, color="#b2182b")
    axes[1].set_ylabel("F(x)")
    # F reaches 0 and 1 only at the clip window, which sits outside the data
    # range whenever the search picks a generous alpha -- so the curve looks
    # like it fails to span [0,1] unless the window is drawn. Marking it also
    # makes the one gene with a geometric meaning visible in the figure.
    mu, sd = x.mean(), x.std()
    for edge in (mu - st.alphas[0] * sd, mu + st.alphas[0] * sd):
        if grid[0] <= edge <= grid[-1]:
            axes[1].axvline(edge, color="#666666", lw=0.8, ls=":")
    axes[1].text(0.02, 0.86, f"K={int(st.ks[0])}  alpha={st.alphas[0]:.2f} "
                             f"(clip at {mu - st.alphas[0] * sd:.2f}, "
                             f"{mu + st.alphas[0] * sd:.2f})  "
                             f"gamma={st.gammas[0]:.3f}  K_eff={len(cent)}",
                 transform=axes[1].transAxes, fontsize=7.5)

    order = np.argsort(cent[:, 0])
    c = cent[order, 0]
    axes[2].vlines(c, 0, 1, color="#b2182b", lw=1.0, label="companding centroids")
    if kmeans_centroids is not None and len(kmeans_centroids):
        axes[2].vlines(np.asarray(kmeans_centroids), 0, 0.6, color="#1b7837",
                       lw=1.0, ls="--", label="k-means centroids")
    axes[2].set_yticks([])
    axes[2].set_ylim(0, 1.35)     # headroom so the legend cannot sit on the rules
    axes[2].set_xlabel("x")
    axes[2].legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
