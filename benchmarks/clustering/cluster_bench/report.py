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

from pathlib import Path

import numpy as np

from .metrics import WORST_DB, WORST_SIL

_DEGENERATE = {"davies_bouldin": WORST_DB, "neg_silhouette": WORST_SIL}


#: Display names for figure titles. Spelled out rather than derived, because a
#: mechanical title-caser gets these wrong in ways a reader notices: GMM and DIM
#: are acronyms, Laplace and Student are surnames, "lognormal" is hyphenated in
#: the literature, and the S-/A-/DIM-set families carry their K in the name. The
#: raw identifier still names the results directory and every CSV, so nothing
#: here breaks the path from a figure back to its numbers.
DISPLAY_NAMES = {
    "gaussian": "Gaussian",
    "laplace": "Laplace",
    "student_t3": "Student-t (df 3)",
    "lognormal": "Log-normal",
    "uniform": "Uniform",
    "gmm3": "GMM3",
    "gmm5_unbalanced": "GMM5 Unbalanced",
    "bimodal_asym": "Bimodal Asymmetric",
    "gpt2_c_attn": "GPT-2 Attention Weights",
    "s_set_k15": "S-Set (K=15)",
    "a_set_k20": "A-Set (K=20)",
    "unbalance": "Unbalance",
    "birch_grid10x10": "Birch Grid (10x10)",
    "dim32": "DIM-32",
    "iris": "Iris",
    "wine": "Wine",
    "breast_cancer": "Breast Cancer",
    "digits": "Digits",
}

#: Words that stay lowercase inside a title unless they lead it.
_MINOR = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on",
          "or", "the", "to", "vs", "with"}


def save_figure(fig, path, formats=("png", "pdf")):
    """Write one figure in every requested format, from a single `.png` path.

    PNG for looking at, PDF for the report: LaTeX embeds vector text and lines
    at the printer's resolution, so axis labels stay sharp instead of being
    resampled from a 140-dpi bitmap.

    The dominated cloud is drawn with `rasterized=True`, which matters most
    here: inside the PDF those ~12,000 points stay a raster while the axes,
    text and front remain vector. Without it the file would carry twelve
    thousand individual vector circles per figure.
    """
    path = Path(path)
    for ext in formats:
        fig.savefig(path.with_suffix("." + ext), bbox_inches="tight")


def pretty_name(name: str) -> str:
    """Dataset name as a figure title: title case, with the known names spelled out.

    Falls back to title-casing the identifier for anything not in
    `DISPLAY_NAMES` -- underscores become spaces, each word takes an initial
    capital, and minor words keep lowercase unless they lead. A token already
    carrying capitals (an acronym someone typed deliberately) is left alone
    rather than being flattened to Title Case.
    """
    if not name:
        return name
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    words = name.replace("_", " ").split()
    out = []
    for i, w in enumerate(words):
        if w != w.lower():
            out.append(w)                       # respect a deliberate acronym
        elif i and w in _MINOR:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


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
    """Boolean mask of the Pareto-optimal rows of a minimization matrix.

    Two objectives get a sweep: sort by the first, walk left to right, and keep
    a point only if it beats the best second objective seen so far. That is
    O(n log n) against the pairwise form's O(n^2), which matters once this is
    applied to a 12,000-point evaluation archive rather than a 100-point front.

    Exact duplicates are collapsed first and their fate shared. Without that the
    strict comparison in the sweep would mark the second copy of an identical
    pair as dominated, when by definition neither dominates the other -- a point
    is dominated only if some other is at least as good everywhere and strictly
    better somewhere.
    """
    n = len(f)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if f.shape[1] != 2:
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            dominated = np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)
            if dominated.any():
                keep[i] = False
        return keep

    uniq, inverse = np.unique(f, axis=0, return_inverse=True)
    order = np.lexsort((uniq[:, 1], uniq[:, 0]))
    keep_u = np.zeros(len(uniq), dtype=bool)
    best = np.inf
    for i in order:
        if uniq[i, 1] < best:
            keep_u[i] = True
            best = uniq[i, 1]
    return keep_u[inverse.ravel()]


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
    """Fraction of the reference box dominated by the front. In [0, 1].

    The objectives are first mapped to the unit box using the ideal and nadir
    of the two fronts' UNION, then the dominated volume is measured against a
    reference point at `ref` on every axis.

    That raw volume is then DIVIDED BY ref**m. Without it the measure runs to
    ref**m, not 1: a front touching the ideal corner scored 1.21 on two
    objectives, which is indefensible in a column labelled "coverage" and was
    reported that way until it was queried. Worse, the ceiling moves with the
    number of objectives -- 1.21 for two, 1.331 for three -- so an unnormalized
    figure is not even comparable between two runs of this same benchmark that
    optimize different objective counts. Dividing makes it a fraction of the
    same box in every configuration.
    """
    if len(f) == 0:
        return 0.0
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    z = (f - ideal) / span
    z = z[np.all(z <= ref, axis=1)]
    if len(z) == 0:
        return 0.0
    try:
        from pymoo.indicators.hv import HV
        raw = float(HV(ref_point=np.full(f.shape[1], ref))(z))
    except ImportError:                        # pragma: no cover
        return float("nan")
    return raw / ref ** f.shape[1]


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
    degenerate = (bool(len(fa) and len(ba_nd))
                  and bool(np.all(nadir - ideal <= 1e-12)))

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
  n / d         samples and dimensions of the dataset.
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
  coverage      fraction of the reference box each front dominates, in [0,1].
                Both fronts are normalized against the ideal and nadir of their
                union, so it is a share of the SAME box.  HIGHER is better."""


ARM_LABEL = {"dp": "K-means (DP)", "lloyd": "K-means (Lloyd x10)",
             "sklearn": "K-means (scikit-learn)", "companding": "Companding (NSGA-II)"}
ARM_ORDER = ("dp", "lloyd", "sklearn", "companding")


def method_rows(summary) -> list[dict]:
    """One row per METHOD, not one row per dataset.

    The three k-means arms are different algorithms with different guarantees --
    the DP is a proven optimum, Lloyd is a restarted local search, sklearn is an
    independent implementation -- and collapsing them into a single best-of-three
    column answers only "what is the best k-means anyone would get". It cannot
    answer "how does this compare to Lloyd", which is the question a reader with
    Lloyd in their pipeline actually has. So each arm gets its own row.
    """
    arms = summary.get("kmeans_arms") or {}
    per_arm = summary.get("per_arm") or {}
    out = []
    for arm in ARM_ORDER:
        if arm == "companding":
            out.append({
                "method": ARM_LABEL[arm], "arm": arm,
                "runs": summary.get("n_companding_evals"),
                "mse": summary.get("best_mse_companding"),
                "mse_k": summary.get("best_mse_k_companding"),
                "db": summary.get("best_db_companding"),
                "db_k": summary.get("best_db_k_companding"),
                "db_min": summary.get("best_db_minsize_companding"),
                "silhouette": summary.get("best_silhouette_companding"),
                "silhouette_k": summary.get("best_silhouette_k_companding"),
                "ari": summary.get("best_adjusted_rand_companding"),
                "ari_k": summary.get("best_adjusted_rand_k_companding"),
                "cost": None, "shared_k": None, "front": summary.get("n_front_companding"),
            })
            continue
        a = arms.get(arm)
        if not a:
            continue
        pa = per_arm.get(arm, {})
        ex = [m["excess_pct"] for m in pa.get("matched_k", [])
              if np.isfinite(m["excess_pct"])]
        out.append({
            "method": ARM_LABEL[arm], "arm": arm, "runs": a.get("n_runs"),
            "mse": a.get("best_mse"), "mse_k": a.get("best_mse_k"),
            "db": a.get("best_db"), "db_k": a.get("best_db_k"), "db_min": None,
            "silhouette": a.get("best_silhouette"),
            "silhouette_k": a.get("best_silhouette_k"),
            "ari": a.get("best_adjusted_rand"),
            "ari_k": a.get("best_adjusted_rand_k"),
            # Excess is the COMPANDING cost measured against THIS arm alone.
            "cost": float(np.median(ex)) if ex else None,
            "shared_k": len(ex),
            "front": pa.get("n_front_kmeans"),
        })
    return out


PER_ARM_LEGEND = """  Each k-means arm is listed separately; they are different algorithms.
    DP            globally optimal 1-D k-means by dynamic programming.
                  A proven optimum, and 1-D only.
    Lloyd x10     best of ten restarts (k-means++/random/quantile/uniform).
    scikit-learn  independent implementation, same K grid as the others.
  runs      clusterings that method produced (one per K for k-means arms).
  cost@K    excess MSE COMPANDING pays against THAT ARM at equal cluster count,
            median over the K they share. Blank on the companding row itself.
  @K        cluster count the score was achieved at; min= its smallest cluster.
  ARI       adjusted Rand index vs. the generating labels, where they exist.
            The only column a badly shaped partition cannot fake."""


def format_per_arm_tables(summaries, markdown=False) -> str:
    """The main table: every dataset, every method, one row each."""
    head = ["dataset", "method", "runs", "best MSE", "cost@K", "shared K",
            "best DB", "best silh", "best ARI"]
    rows = []
    for s in summaries:
        first = True
        for m in method_rows(s):
            rows.append([
                f"{s['dataset']} (n={s['n']}, d={s['d']})" if first else "",
                m["method"], _fmt(m["runs"], "{:d}"),
                _with_k(m["mse"], m["mse_k"], "{:.4g}"),
                _fmt(m["cost"], "{:+.1f}%"), _fmt(m["shared_k"], "{:d}"),
                _with_k(m["db"], m["db_k"], "{:.3f}", m["db_min"]),
                _with_k(m["silhouette"], m["silhouette_k"], "{:+.3f}"),
                _with_k(m["ari"], m["ari_k"], "{:.3f}"),
            ])
            first = False
    h = "### " if markdown else ""
    parts = [f"{h}Every method, separately", "",
             _render(head, rows, markdown), "", PER_ARM_LEGEND]
    return "\n".join(parts)


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

    # n and d repeat here rather than being left to the first table: each table
    # gets read on its own, and "only companding 81" means something different
    # at n=150, d=4 than at n=10000, d=2.
    reach_head = ["dataset", "n", "d", "k-means runs", "on its front",
                  "companding evals", "on its front", "only companding",
                  "only k-means", "coverage companding", "coverage k-means"]
    reach = [[r["dataset"], r["n"], r["d"], _fmt(r["kmeans_runs"], "{:d}"),
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

#: Axis labels. Every searchable objective in `metrics.MINIMIZED` is minimized,
#: so each carries a down arrow rather than a parenthetical: the arrow is read
#: at a glance next to the axis, whereas "(lower better)" is prose competing
#: with the tick labels for the same space. Anything shown with an up arrow is
#: a quantity where more is better -- there are none on the objective axes by
#: construction, but the convergence plot's hypervolume is one.
_LABEL = {
    "mse": "MSE ↓",
    "sse": "SSE ↓",
    "davies_bouldin": "Davies-Bouldin ↓",
    "neg_silhouette": "−silhouette ↓",
    "k_eff": "clusters used ↓",
    "entropy_bits": "label entropy, bits/sample ↓",
    "index_bits": "index width, bits ↓",
}


def plot_objective_space(path, dataset_name, front_rows, baseline_rows,
                         objectives, archive_rows=None, arms=None):
    """The headline figure: both fronts on the two axes that were optimized.

    `archive_rows` is every candidate the search EVALUATED, drawn as a faint
    cloud behind the front. It has to be passed in explicitly: pymoo's `res.X`
    is already reduced to the non-dominated set, so a "dominated" series derived
    from it is empty by construction -- which is exactly what this figure showed
    for several runs, legend entry and all, until someone noticed the points
    were missing. The cloud is the honest picture of what the search explored
    and how much of it the front summarises.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = drop_degenerate(front_rows, objectives)
    br = drop_degenerate(baseline_rows, objectives)
    fa, ba = objective_matrix(fr, objectives), objective_matrix(br, objectives)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)

    # Draw order is explicit, not incidental. The dominated cloud is thousands
    # of points and the k-means arms are a few dozen, so whichever is drawn
    # last wins the pixels -- with the cloud on top the baselines disappeared
    # under it. zorder rather than call order, so a later edit that reorders
    # these blocks cannot silently bury the baselines again.
    Z_CLOUD, Z_KMEANS, Z_FRONT = 1, 3, 4

    if archive_rows:
        ar = objective_matrix(drop_degenerate(archive_rows, objectives), objectives)
        if len(ar):
            # Only the DOMINATED ones. The archive also holds the survivors, and
            # drawing those under a "dominated" label would be wrong -- the
            # front is plotted separately, in its own colour.
            dom = ar[~nondominated(ar)]
            if len(dom):
                ax.plot(dom[:, 0], dom[:, 1], ".", ms=2, color="#A6DBA0",
                        alpha=0.35, ls="none", rasterized=True,
                        zorder=Z_CLOUD,
                        label="Companding dominated (NSGA-II)")

    # Blue / orange / red for the three k-means arms; companding takes green,
    # its dominated cloud a pale tint of the same green so the two read as one
    # method at a glance.
    #
    # Red and green together are the pair most affected by deuteranopia, so the
    # figure is built not to depend on hue: every arm has its own marker
    # (circle, square, triangle) and companding is the only series drawn as a
    # connected line. Colour is the fast path, shape is the reliable one.
    #
    # Semi-transparent because the arms overlap heavily wherever they agree,
    # and that overlap is itself informative: solid markers hid whichever was
    # drawn last.
    styles = {"kmeans_dp": ("o", "#0072B2", "K-means (DP)"),
              "kmeans_lloyd": ("s", "#E69F00", "K-means (Lloyd, multi-start)"),
              "kmeans_sklearn": ("^", "#B2182B", "K-means (scikit-learn)")}
    if arms:
        styles = {f"kmeans_{a}": styles[f"kmeans_{a}"] for a in arms
                  if f"kmeans_{a}" in styles}
        if len(styles) == 1:
            # Nothing to distinguish it from, so drop the qualifier.
            k, (m, c, _) = next(iter(styles.items()))
            styles = {k: (m, c, "K-means")}
    for method, (marker, colour, label) in styles.items():
        idx = [i for i, r in enumerate(br) if r["method"] == method]
        if idx:
            ax.plot(ba[idx, 0], ba[idx, 1], marker, ms=5.5, color=colour,
                    label=label, alpha=0.6, ls="none", zorder=Z_KMEANS)
    if len(fa):
        nd = nondominated(fa)
        pts = fa[nd]
        pts = pts[np.argsort(pts[:, 0])]
        ax.plot(pts[:, 0], pts[:, 1], "-o", ms=4, lw=1.6, color="#1B7837",
                zorder=Z_FRONT, label="Companding front (NSGA-II)")

    if objectives[0] in ("mse", "sse"):
        ax.set_xscale("log")
    ax.set_xlabel(_LABEL.get(objectives[0], objectives[0]))
    ax.set_ylabel(_LABEL.get(objectives[1], objectives[1]))
    ax.set_title(pretty_name(dataset_name))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_convergence(path, dataset_name, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=140)
    ax.plot([c["n_eval"] for c in curve], [c["hv"] for c in curve],
            lw=1.6, color="#1B7837")
    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_ylim(0, 1)
    ax.set_title(pretty_name(dataset_name))
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_warp_1d(path, dataset, genes, centroids, kmeans_centroids=None,
                 grid_n=256, residual_type="linear", degree=3):
    """What the search learned: the density with both methods' centroids, and
    the monotone warp that produced them.

    Two panels, not three. The old third panel was a bare rug of centroid
    positions, which is the same information as marking them on the density --
    and far less useful there, because the whole question is WHERE the levels
    sit relative to the mass.

    Takes decoded genes rather than a problem object, so a finished run can be
    redrawn from `front.csv` without reconstructing the search.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .companding import companding_forward

    k, alpha, gamma, u = genes
    alpha, gamma = float(alpha[0]), float(gamma[0])
    x = np.asarray(dataset.x)[:, 0]
    # Evaluate F at the DATA, not on a uniform grid. `companding_forward`
    # estimates its density histogram from whatever array it is handed, so
    # passing a grid fits the warp to a UNIFORM density and draws a curve the
    # quantizer never used -- which is why this panel showed a near-straight
    # line for Laplace at gamma=0.294. It is the same trap `companding_edges`
    # documents, and it survived here until the boundary markers were added
    # and visibly failed to sit on the curve.
    order = np.argsort(x)
    grid = x[order]
    f = companding_forward(x, alpha, gamma, np.asarray(u[0]), grid_n,
                           residual_type, degree)[order]
    cent = np.sort(np.asarray(centroids)[:, 0])

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.0), dpi=140, sharex=True)

    counts, _, _ = axes[0].hist(x, bins=200, color="#B0B0B0")
    top = counts.max() if len(counts) else 1.0
    # Centroids ON the density: a level sitting where there is no mass, or a
    # gap where there is plenty, is the entire story of a quantizer and it is
    # invisible when the marks live in a separate strip.
    axes[0].vlines(cent, 0, top * 1.02, color="#1B7837", lw=0.9,
                   label="Companding (NSGA-II)")
    if kmeans_centroids is not None and len(kmeans_centroids):
        axes[0].vlines(np.asarray(kmeans_centroids), 0, top * 0.62,
                       color="#B2182B", lw=0.9, ls="--", label="K-means")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, top * 1.25)
    axes[0].legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.95)
    axes[0].set_title(pretty_name(dataset.name))

    axes[1].plot(grid, f, lw=1.6, color="#1B7837")
    axes[1].set_ylabel("$F(x)$")

    # The bin boundaries, marked where they are DECIDED. Assignment is
    # floor(K F(x)), so bin j opens where the curve crosses j/K: the levels are
    # equally spaced up the F axis and unequally spaced along x, and the warp
    # is exactly the function that converts one into the other. Without these
    # the panel shows a curve with no visible connection to the partition
    # above it.
    from .companding import companding_edges
    edges = companding_edges(x, int(k[0]), alpha, gamma, np.asarray(u[0]),
                             residual_type, degree)
    if len(edges):
        targets = (np.arange(1, int(k[0])) / int(k[0]))[:len(edges)]
        step = max(1, len(edges) // 40)          # keep a large K readable
        e, t = edges[::step], targets[::step]
        axes[1].hlines(t, grid[0], e, color="#999999", lw=0.5, ls=":")
        axes[1].vlines(e, 0, t, color="#999999", lw=0.5, ls=":")
        axes[1].plot(e, t, "o", ms=3.5, color="#1B7837",
                     label="bin boundaries")
        axes[1].legend(fontsize=7, loc="lower right", framealpha=0.95)
        # And on the density above, so the two panels line up visually.
        axes[0].vlines(e, 0, top * 1.02, color="#999999", lw=0.5, ls=":",
                       zorder=0)
    # F reaches 0 and 1 only at the clip window, which sits outside the data
    # range whenever the search picks a generous alpha -- so the curve looks
    # like it fails to span [0,1] unless the window is drawn.
    mu, sd = x.mean(), x.std()
    for edge in (mu - alpha * sd, mu + alpha * sd):
        if grid[0] <= edge <= grid[-1]:
            axes[1].axvline(edge, color="#666666", lw=0.8, ls=":")
    axes[1].text(0.02, 0.88,
                 # Raw: in a plain f-string "\a" is the BEL character, so
                 # "$\alpha$" reaches mathtext as "$<BEL>lpha$" and fails to
                 # parse. Same trap for "\gamma" via no escape at all.
                 rf"$K$={int(k[0])}   $\alpha$={alpha:.2f}   "
                 rf"$\gamma$={gamma:.3f}   $K_{{\mathrm{{eff}}}}$={len(cent)}",
                 transform=axes[1].transAxes, fontsize=8)
    axes[1].set_xlabel("$x$")
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)
