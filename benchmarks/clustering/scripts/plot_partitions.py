"""Draw the partitions themselves, coloured by cluster, for 1-D and 2-D sets.

    python scripts/plot_partitions.py                       # latest runs
    python scripts/plot_partitions.py --run results/suite_md-...
    python scripts/plot_partitions.py --datasets gmm3 birch_grid10x10

Every method is drawn at the SAME number of clusters, so the panels differ only
in where the boundaries went. That is the whole point: the score tables say a
method is 40% worse, and these figures say what that looks like. The target K is
the dataset's true cluster count where it has one, otherwise a mid-range value.

The two mechanisms are geometrically different and the figures make that plain:
k-means owns a Voronoi tessellation of freely placed centroids, while companding
owns an axis-aligned grid whose spacing is set by the warp. In 2-D the k-means
panels shade the Voronoi cells; the companding panels draw the per-axis bin
edges, which ARE its boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_bench import datasets as ds  # noqa: E402
from cluster_bench.companding import (companding_edges,  # noqa: E402
                                      companding_quantize_1d,
                                      companding_quantize_md)
from cluster_bench.kmeans import (ExactKMeans1D, lloyd_multistart,  # noqa: E402
                                  sklearn_available, sklearn_kmeans)
from cluster_bench.metrics import evaluate  # noqa: E402
from cluster_bench.report import (pretty_name,  # noqa: E402
                                  plot_warp_1d, save_figure)

ROOT = Path(__file__).resolve().parents[1]
# Qualitative, deliberately not a sequential map: cluster ids are labels, not
# magnitudes, and a viridis-style ramp would imply an ordering that only exists
# in 1-D and is meaningless in 2-D.
CMAP = plt.get_cmap("tab20")


def colours(labels):
    return CMAP(np.asarray(labels) % 20)


def _quantize(dataset, k, alpha, gamma, u):
    if dataset.d == 1:
        return companding_quantize_1d(dataset.x[:, 0], int(k[0]), float(alpha[0]),
                                      float(gamma[0]), u[0])
    return companding_quantize_md(dataset.x, k, alpha, gamma, u)


def companding_at_k(run: Path, dataset, target_k: int, span: int = 4):
    """The best companding partition with EXACTLY `target_k` occupied clusters.

    Reading the front alone is not enough. The front is Pareto-optimal on
    (MSE, DB), and it frequently contains no member at a given cluster count --
    on \\texttt{gmm3} the nearest to K=3 was K=148. Drawing that under a heading
    of "K=3" would compare a 148-cluster partition against three-cluster
    k-means, which is not a comparison at all.

    So each front member's WARP is reused and its K gene alone is swept until
    the realised K_eff hits the target. Empty bins are dropped, so K_eff <= K
    and the sweep has to run upwards; `span` bounds how far. Among everything
    that lands on the target exactly, the lowest-MSE partition wins -- the same
    rule the matched-K table uses, so the figure and that table agree.

    Returns None if no warp in the front can be made to produce `target_k`,
    which is itself worth knowing and is reported rather than papered over.
    """
    path = run / dataset.name / "front.csv"
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if int(r["k_eff"]) >= 2]
    if not rows:
        return None

    d = dataset.d
    # In 1-D, K_eff <= K, so sweeping K upward from the target must pass
    # through it. In d dimensions the cells number prod_j K_j, so the same
    # sweep runs over a shared per-axis K starting just below target^(1/d).
    if d == 1:
        candidates = range(target_k, target_k * span + 1)
    else:
        root = max(2, int(round(target_k ** (1.0 / d))))
        candidates = range(2, root + span + 1)

    best = None
    for r in rows:
        alpha = json.loads(r["gene_alpha"])
        gamma = json.loads(r["gene_gamma"])
        u = [np.asarray(v) for v in json.loads(r["gene_u"])]
        for k_try in candidates:
            k = [k_try] * d
            labels, cent = _quantize(dataset, k, alpha, gamma, u)
            if cent.shape[0] != target_k:
                continue
            m = float(((dataset.x - cent[labels]) ** 2).sum())
            if best is None or m < best[0]:
                best = (m, {"labels": labels, "centroids": cent,
                            "genes": (k, alpha, gamma, u)})
    return best[1] if best else None


def kmeans_fits(dataset, target_k, dp_max_n=4000, seed=0, arms=None):
    """Fit the requested K-means arms. With one arm it is labelled "K-means".

    A qualifier only earns its place when there is something to distinguish
    from; a lone "K-means (scikit-learn)" makes the reader carry an
    implementation detail through every panel for nothing.
    """
    arms = arms or ("dp", "lloyd", "sklearn")
    out = {}
    if "dp" in arms and dataset.d == 1:
        try:
            out["K-means (DP)"] = ExactKMeans1D(
                dataset.x[:, 0], target_k, max_n=dp_max_n, seed=seed).fit(target_k)
        except Exception:
            pass
    if "lloyd" in arms:
        out["K-means (Lloyd x10)"] = lloyd_multistart(dataset.x, target_k,
                                                      n_init=10, seed=seed)
    if "sklearn" in arms and sklearn_available():
        out["K-means (scikit-learn)"] = sklearn_kmeans(dataset.x, target_k,
                                                       n_init=10, seed=seed)
    if len(out) == 1:
        out = {"K-means": next(iter(out.values()))}
    return {k: {"labels": v[0], "centroids": v[1]} for k, v in out.items()}


def annotate(ax, dataset, sol, inside=False):
    """Scores for one panel: distortion, cluster count, and ARI where it exists.

    The smallest-cluster population is deliberately not here. It still matters
    -- a validity score reached with a one-point cluster is an artifact of the
    index rather than a partition -- but this figure shows the partition
    directly, so an isolated outlier is visible in the picture itself. The
    number is kept in `partitions.json` and in the report's per-dataset tables,
    where the validity scores it qualifies actually appear.
    """
    m = evaluate(dataset.x, sol["labels"], sol["centroids"], 1500, 0,
                 y_true=dataset.y_true)
    bits = f"MSE {m['mse']:.3g}   K={m['k_eff']}"
    if dataset.y_true is not None:
        bits += f"   ARI {m['adjusted_rand']:.3f}"
    # Directly above the axes, centred. On the 2-D panels this lands just under
    # the method name, which carries extra title padding to make room; on the
    # 1-D panels the method name is the row label on the left, so there is
    # nothing above the axes to collide with. `inside` is kept for a caller
    # that has no room above the plot.
    if inside:
        ax.text(0.02, 0.98, bits, transform=ax.transAxes, fontsize=7,
                ha="left", va="top",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    else:
        ax.text(0.5, 1.015, bits, transform=ax.transAxes, fontsize=7.5,
                ha="center", va="bottom")


def hist_by_cluster(ax, x, labels, bins=140):
    """Histogram with each bar coloured by the cluster that owns it.

    The conventional way to show a partition of one variable: the shape of the
    density stays legible and the clusters are read off as contiguous bands of
    colour. A jittered scatter shows the same assignment but hides the density,
    which on these datasets is the thing being partitioned.

    A bar is coloured by the MAJORITY label among its points, so a bin straddling
    a boundary takes the side it mostly belongs to rather than whichever point
    happened to be first.
    """
    edges = np.histogram_bin_edges(x, bins=bins)
    which = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    counts = np.zeros(len(edges) - 1)
    owner = np.zeros(len(edges) - 1, dtype=int)
    for b in range(len(edges) - 1):
        sel = labels[which == b]
        counts[b] = sel.size
        if sel.size:
            owner[b] = np.bincount(sel).argmax()
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color=colours(owner), linewidth=0)
    return counts.max() if counts.size else 1.0


def plot_1d(path, dataset, sols, target_k):
    """One row per method: the density, coloured by cluster, with boundaries."""
    x = dataset.x[:, 0]
    n_rows = len(sols)
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 1.8 * n_rows + 1.2),
                             dpi=150, sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (name, sol) in zip(axes, sols.items()):
        top = hist_by_cluster(ax, x, sol["labels"])
        cent = np.sort(sol["centroids"][:, 0])
        # Each method's OWN boundaries. For k-means the cells are Voronoi by
        # construction, so on the line the boundary is the midpoint between
        # adjacent centroids. Companding is not a nearest-prototype method --
        # it assigns by floor(K F(x)) and only then takes bin means -- so its
        # boundaries are the bin edges, which do NOT generally sit at those
        # midpoints. Drawing midpoints for both was wrong by up to 9.7% of the
        # data range on gmm5_unbalanced: rules where companding does not cut.
        if "companding" in name.lower():
            k, alpha, gamma, u = sol["genes"]
            cuts = companding_edges(x, int(k[0]), float(alpha[0]),
                                    float(gamma[0]), u[0])
        else:
            cuts = (cent[:-1] + cent[1:]) / 2.0 if len(cent) > 1 else []
        if len(cuts):
            ax.vlines(cuts, 0, top * 1.02, color="k", lw=0.7, ls="--",
                      alpha=0.7)
        ax.plot(cent, np.full_like(cent, top * 1.10), "v", ms=4.5,
                color="#1B7837", clip_on=False)
        ax.set_ylim(0, top * 1.16)
        # Wrap the qualifier onto a second line: the row labels sit in the left
        # margin, and "Companding (NSGA-II)" on one line would push the axes in.
        ax.set_ylabel(name.replace("K-means ", "K-means\n")
                          .replace("Companding ", "Companding\n"), fontsize=8,
                      rotation=0, ha="right", va="center", labelpad=8)
        ax.tick_params(labelsize=8)
        annotate(ax, dataset, sol)

    axes[-1].set_xlabel("$x$")
    # Just the dataset name. K is already stated on every panel, and a title
    # that repeats it competes with the thing it is labelling.
    fig.suptitle(pretty_name(dataset.name), fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    save_figure(fig, path)
    plt.close(fig)


def companding_regions(x, labels, mesh, edges, shape):
    """Which cluster owns each mesh point under the product quantizer.

    Not a nearest-centroid assignment. Companding decides membership by which
    axis-aligned CELL a point falls in, so the mesh is binned with the same
    per-axis edges the data was, and each cell inherits the cluster label of the
    data points inside it. Colouring it by nearest centroid instead would draw a
    Voronoi diagram the method never computed -- which is why these panels went
    unshaded at first, and why they looked as though companding had no decision
    regions at all.

    Cells no data point occupies get -1: the grid tiles the whole plane, but an
    empty box is not a cluster and has no centroid to name it.
    """
    cell = lambda pts: (np.searchsorted(edges[0], pts[:, 0], side="right")
                        * (len(edges[1]) + 1)
                        + np.searchsorted(edges[1], pts[:, 1], side="right"))
    data_cell = cell(np.asarray(x))
    order = np.argsort(data_cell)
    uniq, first = np.unique(data_cell[order], return_index=True)
    label_of = np.asarray(labels)[order][first]

    idx = np.searchsorted(uniq, cell(mesh))
    idx_safe = np.clip(idx, 0, len(uniq) - 1)
    owner = np.where(uniq[idx_safe] == cell(mesh), label_of[idx_safe], -1)
    return owner.reshape(shape)


def plot_2d(path, dataset, sols, target_k):
    """One panel per method: points coloured by cluster, plus the boundaries.

    Both methods get shaded decision regions, computed the way each method
    actually decides: nearest centroid for k-means (a Voronoi tessellation),
    occupied grid cell for companding (a product quantizer). Same colormap,
    same alpha, so the panels are comparable at a glance and the difference the
    reader sees is the difference in the partitions.
    """
    x = dataset.x
    n_cols = len(sols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.4 * n_cols, 4.6), dpi=150,
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    pad = 0.04 * (x.max(0) - x.min(0))
    xlim = (x[:, 0].min() - pad[0], x[:, 0].max() + pad[0])
    ylim = (x[:, 1].min() - pad[1], x[:, 1].max() + pad[1])
    gx, gy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    mesh = np.column_stack([gx.ravel(), gy.ravel()])

    for ax, (name, sol) in zip(axes, sols.items()):
        cent = sol["centroids"]
        # Case-insensitive: the display label is "Companding (NSGA-II)",
        # and a lowercase test silently sent it down the k-means branch,
        # drawing Voronoi cells for a method that has none.
        if "companding" in name.lower():
            k, alpha, gamma, u = sol["genes"]
            edges = [companding_edges(x[:, j], int(k[j]), float(alpha[j]),
                                      float(gamma[j]), u[j]) for j in (0, 1)]
            owner = companding_regions(x, sol["labels"], mesh, edges,
                                       gx.shape)
            # Shaded the same way and at the same alpha as the k-means panels.
            # Cells no data point occupies are left blank (NaN), because they
            # are not clusters -- a product quantizer's grid subdivides the
            # whole plane, but only the occupied boxes have a centroid.
            ax.pcolormesh(gx, gy, np.where(owner < 0, np.nan, owner % 20),
                          cmap=CMAP, alpha=0.18, shading="auto",
                          vmin=0, vmax=19, rasterized=True)
            for j in (0, 1):
                for e in edges[j]:
                    (ax.axvline if j == 0 else ax.axhline)(
                        e, color="k", lw=0.5, alpha=0.5)
        else:
            d2 = ((mesh[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            owner = d2.argmin(1).reshape(gx.shape)
            ax.pcolormesh(gx, gy, owner % 20, cmap=CMAP, alpha=0.18,
                          shading="auto", vmin=0, vmax=19,
                          rasterized=True)
            ax.contour(gx, gy, owner, levels=np.arange(len(cent)) + 0.5,
                       colors="k", linewidths=0.4, alpha=0.6)
        ax.scatter(x[:, 0], x[:, 1], s=4, c=colours(sol["labels"]), alpha=0.75,
                   linewidths=0)
        ax.plot(cent[:, 0], cent[:, 1], "x", ms=5, color="k", mew=1.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(name, fontsize=9, pad=15)
        annotate(ax, dataset, sol)

    fig.suptitle(pretty_name(dataset.name), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, path)
    plt.close(fig)


def plot_grid(path, entries):
    """The canonical benchmark figure: datasets down, methods across.

    One row per dataset, one column per method, scored in the corner of each
    panel -- the layout every clustering comparison uses (scikit-learn's cluster
    comparison, the Fraenti benchmark pages) because it is read by scanning a
    row and asking which method broke the structure.

    Rows are drawn in whatever form suits the dataset: a colour-coded density
    for one variable, a scatter for two. Panels are NOT shared across rows, since
    the datasets live on different scales.
    """
    n_rows = len(entries)
    n_cols = max(len(e["sols"]) for e in entries)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.8 * n_rows),
                             dpi=150, squeeze=False)

    for r, e in enumerate(entries):
        dataset, sols, k = e["dataset"], e["sols"], e["k"]
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(sols):
                ax.axis("off")
                continue
            name, sol = list(sols.items())[c]
            if dataset.d == 1:
                top = hist_by_cluster(ax, dataset.x[:, 0], sol["labels"], bins=90)
                cent = np.sort(sol["centroids"][:, 0])
                if len(cent) > 1:
                    ax.vlines((cent[:-1] + cent[1:]) / 2.0, 0, top, color="k",
                              lw=0.5, ls="--", alpha=0.6)
                ax.set_ylim(0, top * 1.05)
                ax.set_yticks([])
            else:
                ax.scatter(dataset.x[:, 0], dataset.x[:, 1], s=3,
                           c=colours(sol["labels"]), alpha=0.75, linewidths=0)
                ax.plot(sol["centroids"][:, 0], sol["centroids"][:, 1], "x",
                        ms=4, color="k", mew=1.0)
                ax.set_xticks([])
                ax.set_yticks([])
            m = evaluate(dataset.x, sol["labels"], sol["centroids"], 1500, 0,
                         y_true=dataset.y_true)
            txt = (f"ARI {m['adjusted_rand']:.3f}" if dataset.y_true is not None
                   else f"MSE {m['mse']:.3g}")
            ax.text(0.03, 0.93, txt, transform=ax.transAxes, fontsize=7.5,
                    va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{dataset.name}\nK={k}", fontsize=8)

    fig.suptitle("Partitions at matched cluster counts", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, path)
    plt.close(fig)


def target_for(dataset, override):
    if override:
        return override
    if dataset.k_true:
        return dataset.k_true
    # No ground truth: a mid-range K, small enough that individual bins are
    # still visible on the page. The point of these figures is the geometry of
    # the boundaries, and at K=256 they are a solid smear.
    return 8


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", default=None,
                    help="results directory; repeatable. Default: latest of each suite")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--k", type=int, default=None, help="override the target K")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <run>/<dataset>/partitions.png")
    ap.add_argument("--arms", nargs="+", default=["dp", "lloyd", "sklearn"],
                    help="which K-means arms to draw (default: all three)")
    ap.add_argument("--suffix", default="",
                    help="appended to the output filename, e.g. _v2")
    ap.add_argument("--grid", type=Path, default=None, metavar="PATH",
                    help="also write the datasets-x-methods comparison grid here")
    args = ap.parse_args()

    res = ROOT / "results"
    runs = args.run or [p for p in (max(res.glob("suite_1d-*"), key=lambda q: q.name,
                                        default=None),
                                    max(res.glob("suite_md-*"), key=lambda q: q.name,
                                        default=None)) if p]
    if not runs:
        sys.exit("no finished runs under results/")

    grid_entries = []
    for run in runs:
        names = args.datasets or [p.name for p in sorted(run.iterdir()) if p.is_dir()]
        for name in names:
            if not (run / name).is_dir():
                continue
            key = {"birch_grid10x10": "birch_grid", "a_set_k20": "a_set",
                   "s_set_k15": "s_set"}.get(name, name)
            try:
                dataset = ds.load(key, seed=0)
            except Exception as exc:
                print(f"  {name}: skipped ({type(exc).__name__})")
                continue
            if dataset.d > 2:
                print(f"  {name}: skipped (d={dataset.d}; only 1-D and 2-D are drawable)")
                continue

            k = target_for(dataset, args.k)
            comp = companding_at_k(run, dataset, k)
            if comp is None:
                print(f"  {name}: skipped -- no warp on the companding front "
                      f"can be made to produce exactly K={k}")
                continue
            sols = {"Companding (NSGA-II)": comp,
                    **kmeans_fits(dataset, k, arms=args.arms)}
            out = args.out or run / name / f"partitions{args.suffix}.png"
            (plot_1d if dataset.d == 1 else plot_2d)(out, dataset, sols, k)
            # The scores behind the figure, written beside it. The report reads
            # these rather than having the numbers retyped into its prose, so a
            # regenerated figure and the paragraph describing it cannot drift.
            Path(out).with_suffix(".json").write_text(json.dumps({
                "dataset": dataset.name, "k": k, "k_true": dataset.k_true,
                "n": dataset.n, "d": dataset.d,
                "methods": {
                    m: {kk: (None if isinstance(vv, float) and not np.isfinite(vv)
                             else vv)
                        for kk, vv in evaluate(dataset.x, sol["labels"],
                                               sol["centroids"], 1500, 0,
                                               y_true=dataset.y_true).items()}
                    for m, sol in sols.items()},
            }, indent=2, default=float))
            grid_entries.append({"dataset": dataset, "sols": sols, "k": k})
            if dataset.d == 1:
                # The warp figure is redrawn here too: it needs the decoded
                # genes, which this script already has, and replot.py works
                # from CSVs alone so it cannot produce one.
                km = next((v["centroids"][:, 0] for kk, v in sols.items()
                           if "K-means" in kk), None)
                plot_warp_1d(Path(out).with_name(f"warp{args.suffix}.png"),
                             dataset, comp["genes"], comp["centroids"], km)
            print(f"  {name}: K={k}, {len(sols)} methods -> {out}")

    if args.grid and grid_entries:
        plot_grid(args.grid, grid_entries)
        print(f"  grid: {len(grid_entries)} datasets -> {args.grid}")


if __name__ == "__main__":
    main()
