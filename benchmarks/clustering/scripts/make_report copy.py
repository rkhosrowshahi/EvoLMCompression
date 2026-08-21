r"""Build the LaTeX tables from finished runs.

    python scripts/make_report.py
    python scripts/make_report.py --out report_v2/tables
    python scripts/make_report.py --1d results/suite_1d-... --md results/suite_md-...

Every number in every table is generated here, from the runs' own `suite.json`,
so a table cannot end up quoting a figure the data no longer supports.

Tables and nothing else: no figures are copied, no prose is written, no
document is assembled and no PDF is built. Each file is a self-contained
booktabs float carrying its own \caption and \label, so whatever hand-written
.tex \inputs it decides where it goes and what is said about it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_bench.report import pretty_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: What each 1-D generator is FOR. The suite is not a grab bag: every entry
#: varies tail weight or modality, the two things the Panter-Dite backbone is
#: sensitive to, because gamma is exactly the gene that trades them off.
PROBES = {
    "gaussian": "reference case; Panter--Dite is derived for smooth unimodal densities",
    "laplace": "sharp peak, exponential tails --- the shape of a trained weight matrix",
    "student_t3": "heavy tails; the clip $\\alpha$ should dominate $\\gamma$ here",
    "lognormal": "strong right skew with a floor at zero; an asymmetric warp should pay",
    "uniform": "degenerate case; uniform binning is already optimal, so $\\gamma\\to0$",
    "gmm3": "three separated modes; a genuine clustering problem on the line",
    "gmm5_unbalanced": "rare modes carry the structure but not the probability mass",
    "bimodal_asym": "narrow Gaussian beside a long-tailed lump; the halves want different densities",
    "s_set_k15": "overlapping Gaussians on a plane (Fr\\\"anti S-set family)",
    "a_set_k20": "many well-separated spherical clusters (A-set family)",
    "unbalance": "20:1 population imbalance between dense and sparse clusters",
    "birch_grid10x10": "axis-aligned grid of clusters --- the product quantizer's best case",
    "dim32": "nine separated clusters in 32 dimensions",
    "iris": "real data, 150$\\times$4",
    "wine": "real data, 178$\\times$13",
}


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def sci(v, digits=3) -> str:
    """LaTeX scientific notation. MSE spans five orders across the suite."""
    if v is None or not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    if -3 <= exp <= 3:
        return f"{v:.{digits}g}"
    return f"${v / 10 ** exp:.2f}\\times10^{{{exp}}}$"


def num(v, spec="{:.3f}") -> str:
    if v is None or not np.isfinite(v):
        return "--"
    return spec.format(v)


def at_k(v, k, spec="{:.3f}", smallest=None) -> str:
    """Score with its cluster count, and optionally its smallest cluster.

    The annotations are the whole point of the validity table: a Davies--Bouldin
    of 0.21 reached with a one-point cluster is an artifact of the index, and
    the bare score cannot say so.
    """
    body = num(v, spec)
    if body == "--" or k is None:
        return body
    # Score and its cluster count, inline: "0.490 (K=10)". The smallest cluster
    # behind each score used to ride along here as ", min=329", because on the
    # validity axes a score reached with a one-point cluster is an artifact of
    # the index rather than a clustering. It made the cell two lines deep and
    # the table hard to scan, so it now lives in the per-dataset tables, which
    # carry a "Smallest cluster" column, and in the note beneath this one.
    return f"{body} ($K={k}$)"


def pending(path: Path, caption: str, label: str, what: str):
    """Stand-in for a table whose run has not finished, stated as such."""
    path.write_text(
        "\n".join([
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{" + caption + "}", r"\label{" + label + "}",
            r"\footnotesize\emph{" + what + "}",
            r"\end{table}", "",
        ]),
        encoding="utf-8")


def _translate_spec(spec: str, flexible: bool) -> str:
    r"""Turn this module's column spec into a LaTeX one.

    Tokens: `l` left, `c` centred, `n` centred at natural width, `p{...}` a
    fixed-width paragraph. With `flexible` (tabularx) `c` becomes the centred
    X column `C` and `p{...}` becomes `X`, so those columns share the leftover
    width; `n` stays a plain `c` and takes only what it needs.

    A `p{...}` group is stepped over rather than scanned, because its argument
    contains letters -- a naive `replace("n", "c")` turned `\linewidth` into
    `\licewidth` and cost a build.
    """
    out, i = [], 0
    while i < len(spec):
        ch = spec[i]
        if ch == "p" and i + 1 < len(spec) and spec[i + 1] == "{":
            depth = 0
            start = i
            while i < len(spec):
                if spec[i] == "{":
                    depth += 1
                elif spec[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            # Centred like every other non-first column; "X" alone
            # would be left aligned.
            out.append("C" if flexible else spec[start:i + 1])
        elif ch == "n":
            out.append("c")
        elif ch == "c":
            out.append("C" if flexible else "c")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def table(path: Path, caption: str, label: str, spec: str, header, rows,
          small: bool = True, fit: bool = True, groups=None):
    r"""Emit one booktabs table, set to the full text width.

    `tabularx` rather than `\resizebox`: both make a table span the page, but
    resizebox does it by SCALING, so a four-row table blown up to the text width
    comes out in oversized type while a wide one shrinks below the body size.
    Nothing on the page then shares a font size with anything else. tabularx
    instead distributes the leftover width across the flexible columns and
    leaves the type alone, so every table matches the body text and matches the
    other tables.

    `fit=False` keeps a fixed-width `tabular` for the rare table that should not
    stretch.
    """
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{" + caption + "}", r"\label{" + label + "}"]
    if small:
        lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # "n" is this module's marker for a narrow, natural-width column. It is
    # not a LaTeX column type, so it has to be translated on BOTH paths --
    # tabularx turns it into "c", and so must plain tabular.
    env = "tabularx" if fit else "tabular"
    cols = _translate_spec(spec, flexible=fit)
    open_tag = (r"\begin{tabularx}{\linewidth}{" + cols + "}" if fit
                else r"\begin{tabular}{" + cols + "}")
    lines += [open_tag, r"\toprule"]
    if groups:
        # Two header rows: the measure spans its methods, the methods name
        # themselves underneath. Repeating "Davies--Bouldin, K-means" and
        # "Davies--Bouldin, companding" side by side spends most of two column
        # widths restating the measure, and leaves the reader to spot that the
        # differing half is at the end of each label.
        top, rule, col = [], [], 1
        for title, span in groups:
            if not title:
                top.append("" if span == 1 else r"\multicolumn{%d}{c}{}" % span)
            elif span == 1:
                top.append(title)
            else:
                top.append(r"\multicolumn{%d}{c}{%s}" % (span, title))
                rule.append(r"\cmidrule(lr){%d-%d}" % (col, col + span - 1))
            col += span
        lines.append(" & ".join(top) + r" \\")
        if rule:
            lines.append(" ".join(rule))
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{" + env + "}"]
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load(run: Path):
    return json.loads((run / "suite.json").read_text())


def median_cost(s):
    ex = [m["excess_pct"] for m in s["matched_k"] if np.isfinite(m["excess_pct"])]
    return (float(np.median(ex)) if ex else None,
            min(ex) if ex else None, max(ex) if ex else None, len(ex))


def datasets_table(out, summaries, label, caption):
    rows = []
    for s in summaries:
        med, lo, hi, cnt = median_cost(s)
        rows.append([esc(pretty_name(s["dataset"])), f"{s['n']:,}".replace(",", "\\,"),
                     str(s["d"]),
                     str(s["k_true"]) if s["k_true"] else "--",
                     PROBES.get(s["dataset"], "")])
    # Flexible, so the Details column absorbs whatever the narrow n/d/K columns
    # leave and the table fills the text block. This used fit=False back when
    # tables were set with \resizebox, which would have SCALED the paragraph
    # text rather than rewrapping it; tabularx has no such problem.
    table(out, caption, label, "lnnnp{0.42\\linewidth}",
          ["Dataset", r"$n$", r"$d$", r"$K^\ast$", "Details"], rows)


def distortion_table(out, summaries, label, caption):
    rows = []
    for s in summaries:
        rows.append([
            esc(pretty_name(s["dataset"])),
            sci(s["best_mse_kmeans"]), sci(s["best_mse_companding"]),
        ])
    table(out, caption, label, "lcc",
          ["Dataset", "K-means", "Companding (NSGA-II)"], rows,
          groups=[("", 1), ("Best MSE", 2)])


#: Column order for the per-arm tables, and the label each arm gets.
ALL_ARMS = [("dp", "K-means (DP)"), ("lloyd", "K-means (Lloyd)"),
            ("sklearn", "K-means (sklearn)")]
#: Set from --arms. When a single arm is selected it is labelled plainly
#: "K-means": with nothing to distinguish it from, naming the implementation
#: adds a qualifier the reader has to carry through every table for no reason.
ARMS = list(ALL_ARMS)
#: One-element list so the gallery builder can see the choice without a global
#: import dance; set alongside ARMS.
SINGLE_ARM = [False]


def set_arms(keys):
    global ARMS
    chosen = [(k, lbl) for k, lbl in ALL_ARMS if k in keys]
    if not chosen:
        raise SystemExit(f"no known arms in {keys}; pick from "
                         f"{[k for k, _ in ALL_ARMS]}")
    ARMS = [(chosen[0][0], "K-means")] if len(chosen) == 1 else chosen
    SINGLE_ARM[0] = len(chosen) == 1
    return ARMS


def per_arm_table(out, summaries, key, label, caption, spec="{:.3f}",
                  scientific=False, want_k=True):
    """One column per method, with the cluster count each score was reached at.

    The earlier version of these tables carried a single "K-means" column that
    was the BEST of the three arms, recomputed per measure -- so on one row it
    meant the dynamic program and on the next it meant Lloyd, without saying so.
    That answers "what is the best k-means anyone would get" and nothing else.
    The three arms are different algorithms with different guarantees, and a
    reader comparing against Lloyd needs the Lloyd column.
    """
    fmt = (lambda v: sci(v)) if scientific else (lambda v: num(v, spec))
    # Spelled out, and kept on one line with \mbox -- but only when the table
    # is narrow enough to hold it. With all three K-means arms this header is
    # repeated across nine columns of about 1.6cm each, and an unbreakable
    # "Davies--Bouldin" then runs past the rule. Two methods leave room; more
    # than two do not, and a wrapped header beats an overfull box.
    wide = len(ARMS) + 1 <= 3
    # An arrow on the header carries the direction, so the table needs no
    # footnote saying which way is better. Down for the measures that are
    # minimized, up for those that are maximized.
    metric = {"mse": r"MSE $\downarrow$",
              "db": (r"\mbox{Davies--Bouldin $\downarrow$}" if wide
                     else r"Davies--Bouldin $\downarrow$"),
              "silhouette": r"Silhouette $\uparrow$",
              "adjusted_rand": r"ARI $\uparrow$"}.get(key, "Score")
    rows = []
    for s in summaries:
        arms = s.get("kmeans_arms") or {}
        cells = []
        for arm, _ in ARMS:
            a = arms.get(arm) or {}
            v, k = a.get(f"best_{key}"), a.get(f"best_{key}_k")
            cells += [fmt(v), ("--" if k is None else str(k))] if want_k else [fmt(v)]
        v = s.get(f"best_{key}_companding")
        k = s.get(f"best_{key}_k_companding")
        cells += [fmt(v), ("--" if k is None else str(k))] if want_k else [fmt(v)]
        rows.append([esc(pretty_name(s["dataset"]))] + cells)

    labels = [lbl for _, lbl in ARMS] + ["Companding (NSGA-II)"]
    if want_k:
        # Score and cluster count get their own columns under a spanning header
        # for the method. Packed as "0.0834 (K=8)" the K reads as part of the
        # number and the column cannot be scanned down.
        table(out, caption, label, "l" + "cn" * len(labels),
              ["Dataset"] + [metric if i % 2 == 0 else "$K$"
                             for i in range(2 * len(labels))],
              rows,
              groups=[("", 1)] + [(lbl, 2) for lbl in labels])
    else:
        table(out, caption, label, "l" + "c" * len(labels),
              ["Dataset"] + labels, rows)


def validity_table(out, summaries, label, caption):
    rows = []
    for s in summaries:
        rows.append([
            esc(pretty_name(s["dataset"])),
            at_k(s["best_db_kmeans"], s.get("best_db_k_kmeans"), "{:.3f}"),
            at_k(s["best_db_companding"], s.get("best_db_k_companding"), "{:.3f}"),
            at_k(s["best_silhouette_kmeans"],
                       s.get("best_silhouette_k_kmeans"), "{:+.3f}"),
            at_k(s["best_silhouette_companding"],
                       s.get("best_silhouette_k_companding"), "{:+.3f}"),
        ])
    table(out, caption, label, "lcccc",
          ["Dataset", "K-means", "Companding (NSGA-II)", "K-means", "Companding (NSGA-II)"],
          rows,
          groups=[("", 1), (r"\mbox{Davies--Bouldin}", 2), ("Silhouette", 2)])


def ari_table(out, groups, label, caption):
    rows = []
    for title, summaries in groups:
        rows.append([r"\multicolumn{4}{l}{\emph{" + title + r"}}"])
        for s in summaries:
            a_km, a_cp = (s.get("best_adjusted_rand_kmeans"),
                          s.get("best_adjusted_rand_companding"))
            if a_km is None or not np.isfinite(a_km):
                continue
            # The better score is bolded, which says who won without spending a
            # column on the word -- and the excess-MSE figure lives in its own
            # table, where it is reported against every arm rather than once.
            bold = lambda v, k, on: (r"\textbf{" + at_k(v, k) + "}") if on else at_k(v, k)
            rows.append([
                r"\quad " + esc(pretty_name(s["dataset"])), str(s["k_true"]),
                bold(a_km, s.get("best_adjusted_rand_k_kmeans"),
                     a_km > a_cp + 1e-9),
                bold(a_cp, s.get("best_adjusted_rand_k_companding"),
                     a_cp > a_km + 1e-9),
            ])
    table(out, caption, label, "lncc",
          ["Dataset", r"$K^\ast$", "K-means", "Companding (NSGA-II)"], rows,
          groups=[("", 1), ("", 1), ("ARI", 2)])


def reach_table(out, groups, label, caption):
    rows = []
    for title, summaries in groups:
        rows.append([r"\multicolumn{7}{l}{\emph{" + title + r"}}"])
        for s in summaries:
            rows.append([
                r"\quad " + esc(pretty_name(s["dataset"])), str(s["n"]), str(s["d"]),
                str(s["n_front_kmeans"]), str(s["n_front_companding"]),
                f"{s['hv_kmeans']:.3f}", f"{s['hv_companding']:.3f}",
            ])
    # Two things, named plainly: how many solutions each method's Pareto front
    # holds, and how much of the objective box it covers. The earlier version
    # also carried a pair of "unreachable by the other" counts, which needed a
    # paragraph of legend to explain and told the reader nothing the coverage
    # figure does not.
    #
    # n and d ride along because a front size means something different at
    # n=150, d=4 than at n=10000, d=2, and this table is read on its own.
    # Seven columns is the ceiling: at ten the header word "Companding (NSGA-II)" was
    # wider than its column and the headings overprinted each other.
    table(out, caption, label, "lnncccc",
          ["Dataset", r"$n$", r"$d$", "K-means", "Companding (NSGA-II)",
           "K-means", "Companding (NSGA-II)"], rows,
          groups=[("", 1), ("", 1), ("", 1), ("Pareto front size", 2),
                  ("Coverage", 2)])


def matched_table(out, summary, label, caption, rows_max=10):
    """Head to head at identical cluster counts.

    With a single arm selected, the rows come from THAT arm's own matched-K
    table rather than the cross-arm envelope, and the "Arm" column is dropped:
    otherwise a v2 report showing only one K-means still names "dp" in a column
    whose arm appears nowhere else in the document.
    """
    single = SINGLE_ARM[0]
    if single:
        arm = ARMS[0][0]
        m = ((summary.get("per_arm") or {}).get(arm) or {}).get(
            "matched_k", summary["matched_k"])
    else:
        m = summary["matched_k"]
    step = max(1, len(m) // rows_max)
    rows = []
    for e in m[::step][:rows_max]:
        row = [str(e["k_eff"])]
        if not single:
            row.append(esc(e["baseline_method"].replace("kmeans_", "")))
        rows.append(row + [
            sci(e["kmeans_mse"]), sci(e["companding_mse"]),
            num(e["kmeans_db"]), num(e["companding_db"]),
            num(e["kmeans_entropy_bits"], "{:.2f}"),
            num(e["companding_entropy_bits"], "{:.2f}"),
        ])
    # No "Excess" column: it is the ratio of the two MSE columns beside it, so
    # it restates what the reader can already see and costs a column's width.
    head = [r"$K$"] + ([] if single else ["Arm"]) + [
        "K-means", "Companding", "K-means", "Companding",
        "K-means", "Companding"]
    groups = [("", 1)] + ([] if single else [("", 1)]) + [
        (r"MSE $\downarrow$", 2),
        (r"\mbox{Davies--Bouldin $\downarrow$}", 2), ("Bits", 2)]
    table(out, caption, label, "n" + ("" if single else "c") + "cccccc",
          head, rows, groups=groups)


def partition_table(out, run: Path, name: str, label: str, suffix: str = ""):
    """Every method's scores at the one cluster count the figure draws.

    Read straight from the `partitions.json` that `plot_partitions.py` writes
    beside each figure, so the table and the picture above it are guaranteed to
    describe the same partitions -- the alternative is retyping numbers into
    prose and hoping a regenerated figure still matches.
    """
    src = run / name / f"partitions{suffix}.json"
    if not src.exists():
        return None
    d = json.loads(src.read_text())
    has_ari = any(m.get("adjusted_rand") is not None for m in d["methods"].values())
    rows = []
    for method, m in d["methods"].items():
        row = [esc(method[:1].upper() + method[1:]), sci(m["mse"]),
               num(m["davies_bouldin"]), num(m["silhouette"], "{:+.3f}"),
               num(m["entropy_bits"], "{:.2f}")]
        if has_ari:
            row.append(num(m.get("adjusted_rand")))
        rows.append(row)
    # \mbox keeps the name on one line: in a narrow flexible column LaTeX
    # breaks it after the dash, and "Davies--" over "Bouldin" reads as two
    # different things.
    head = ["Method", "MSE", r"\mbox{Davies--Bouldin}", "Silhouette", "Bits"]
    if has_ari:
        head.append("ARI")
    table(out, f"{esc(pretty_name(name))} at $K={d['k']}.$",
          label, "l" + "c" * (len(head) - 1), head, rows)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--1d", dest="one_d", type=Path, default=None)
    ap.add_argument("--md", dest="multi_d", type=Path, default=None)
    # The directory the .tex files land in, named directly rather than inferred
    # from a report directory: this script no longer knows about a document.
    ap.add_argument("--out", type=Path, default=ROOT / "tables",
                    help="directory to write the .tex tables into")
    ap.add_argument("--suffix", "--fig-suffix", dest="suffix", default="",
                    help="partitions json suffix to read, e.g. _v2")
    ap.add_argument("--arms", nargs="+", default=[k for k, _ in ALL_ARMS],
                    help="which K-means arms to report (default: all three)")
    args = ap.parse_args()

    set_arms(args.arms)
    res = ROOT / "results"
    latest = lambda pat: max(res.glob(pat), key=lambda p: p.name, default=None)
    one_d = args.one_d or latest("suite_1d-*")
    multi_d = args.multi_d or latest("suite_md-*")
    if one_d is None or not (one_d / "suite.json").exists():
        sys.exit("need a finished suite_1d-* run under results/")

    # The multi-D suite is allowed to be absent or still running. The
    # single-dimensional tables are useful on their own, and refusing to write
    # them because a second run has not landed yet just means nobody sees any
    # numbers until everything is finished.
    s1 = load(one_d)
    if multi_d is not None and (multi_d / "suite.json").exists():
        sm = load(multi_d)
    else:
        sm = []
        print("NOTE: no finished suite_md-* run; the multi-dimensional tables "
              "will say so instead of showing results.")
    tab = Path(args.out)
    tab.mkdir(parents=True, exist_ok=True)
    print(f"1-D  {one_d.name}  ({len(s1)} datasets)")
    print(f"m-D  {multi_d.name if multi_d else '--'}  ({len(sm)} datasets)")

    datasets_table(tab / "datasets_1d.tex", s1, "tab:datasets-1d",
                   "The single-dimensional suite. $K^\ast$ is the generating "
                   "process's own cluster count where it has one; neither method "
                   "is shown it.")
    if not sm:
        for stem, lab, cap in (
                ("datasets_md", "tab:datasets-md", "The multi-dimensional suite"),
                ("distortion_md", "tab:distortion-md", "Distortion, multi-dimensional suite"),
                ("validity_md", "tab:validity-md", "Internal validity indices, multi-dimensional suite")):
            pending(tab / f"{stem}.tex", cap + ".", lab,
                    "The multi-dimensional run had not finished when these "
                    r"tables were built. Re-run scripts/make\_report.py once "
                    "it has.")
    else:
        datasets_table(tab / "datasets_md.tex", sm, "tab:datasets-md",
                   "The multi-dimensional suite. All generated offline and "
                   "seeded, except iris and wine.")

    per_arm_table(
        tab / "distortion_1d.tex", s1, "mse", "tab:distortion-1d",
        "Best MSE reached by each method, single-dimensional suite.",
        scientific=True)
    per_arm_table(
        tab / "distortion_md.tex", sm, "mse", "tab:distortion-md",
        "Best MSE reached by each method, multi-dimensional suite.",
        scientific=True)

    per_arm_table(
        tab / "validity_1d.tex", s1, "db", "tab:validity-1d",
        "Best Davies--Bouldin reached by each method, single-dimensional suite.")
    per_arm_table(
        tab / "validity_md.tex", sm, "db", "tab:validity-md",
        "Best Davies--Bouldin reached by each method, multi-dimensional suite.")
    per_arm_table(
        tab / "silhouette_1d.tex", s1, "silhouette", "tab:silhouette-1d",
        "Best silhouette reached by each method, single-dimensional suite. "
        "Higher is better, unlike Davies--Bouldin; the same caution applies, "
        "since an outlier-versus-rest split scores well on this index too.",
        spec="{:+.3f}")

    ari_table(tab / "ari.tex", [("Single-dimensional", s1),
                                ("Multi-dimensional", sm)], "tab:ari",
              "Adjusted Rand index against the generating labels --- the only "
              "measure here that a badly shaped partition cannot fake.")

    reach_table(tab / "reach.tex", [("Single-dimensional", s1),
                                    ("Multi-dimensional", sm)], "tab:reach",
                "Number of solutions on each method's Pareto front, and the "
                "share of the objective box it covers.")

    by_name = {s["dataset"]: s for s in s1}
    matched_table(tab / "matched_gmm3.tex", by_name["gmm3"], "tab:matched-gmm3",
                  "Head to head at identical cluster counts on GMM3 "
                  "(every tenth shared $K$).")

    # One table per dataset, at the cluster count plot_partitions.py drew, read
    # from the partitions.json it wrote beside the figure. A dataset without
    # one is skipped rather than given an empty table.
    n_part = 0
    for summaries, run in ((s1, one_d), (sm, multi_d)):
        for s in summaries or []:
            name = s["dataset"]
            if partition_table(tab / f"part_{name}.tex", run, name,
                               f"tab:part-{name}", args.suffix) is not None:
                n_part += 1

    print(f"tables -> {tab}  ({len(list(tab.glob('*.tex')))} files, "
          f"{n_part} per-dataset)")


if __name__ == "__main__":
    main()
