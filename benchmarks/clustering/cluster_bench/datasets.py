"""The benchmark problems, in two suites.

1-D suite (primary). Scalar quantization is what companding actually is, and
1-D is the only case where k-means has a tractable global optimum, so this is
where a claim can be made against a true reference rather than against
whatever Lloyd converged to. The generators are chosen to vary the one thing
the Panter-Dite backbone is sensitive to -- tail weight and modality -- since
gamma is exactly the gene that trades those off.

Multi-D suite (secondary). The standard clustering-benchmark shapes, generated
rather than downloaded so the whole thing runs offline and reproducibly: the
Fraenti-style S-sets (overlapping Gaussians), A-sets (many spherical clusters),
Unbalance (wildly different cluster sizes), DIM-sets (well-separated in high
dimension), and a Birch-1-style grid. Real datasets (iris, wine, digits) come
through scikit-learn when it is installed and are skipped with a note when it
is not -- no silent substitution.

Every generator is seeded and every dataset carries `y_true` when the
generating process defines one, so external validity indices remain available
even though the search never sees the labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[3] / ".cache" / "gpt2_all_targets.npz"

#: Sample size for the 1-D generators, and the reason the 1-D suite exists.
#: The exact k-means DP is O(K n log n) with a Python-level divide-and-conquer
#: recursion, so it is affordable up to a few thousand DISTINCT values -- past
#: that `ExactKMeans1D` falls back to a subsample and the reference stops being
#: a proven optimum. Sizing the suite to stay under `baselines.dp_max_n`
#: (default 4000) is what makes "companding costs +x% over k-means" an absolute
#: statement rather than a comparison between two heuristics. Raise it and the
#: run still works; it just loses that guarantee, and the run says so in its
#: notes.
N_1D = 4000


@dataclass
class Dataset:
    name: str
    x: np.ndarray                       # [n, d], float64
    kind: str                           # "1d" or "md"
    y_true: np.ndarray | None = None
    k_true: int | None = None
    notes: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @property
    def d(self) -> int:
        return self.x.shape[1]

    def standardized(self) -> "Dataset":
        """Z-score every column.

        k-means measures distance in the raw units and companding measures its
        clip window in per-column sigma, so on unstandardized data the two arms
        would not even be looking at the same geometry. Standardizing is the
        usual convention for these benchmarks and it makes the comparison mean
        something.
        """
        sd = self.x.std(axis=0)
        sd[sd <= 0] = 1.0
        z = (self.x - self.x.mean(axis=0)) / sd
        return Dataset(self.name, z, self.kind, self.y_true, self.k_true,
                       (self.notes + " standardized").strip(), dict(self.meta))


def _1d(name, values, notes="", k_true=None, y=None) -> Dataset:
    v = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return Dataset(name, v, "1d", y, k_true, notes)


# --------------------------------------------------------------------------
# 1-D generators
# --------------------------------------------------------------------------

def gaussian_1d(seed=0, n=N_1D):
    r = np.random.default_rng(seed)
    return _1d("gaussian", r.normal(0, 1, n),
               "the reference case: Panter-Dite is derived for smooth unimodal densities")


def laplace_1d(seed=0, n=N_1D):
    r = np.random.default_rng(seed)
    return _1d("laplace", r.laplace(0, 1, n),
               "sharp peak, exponential tails -- the shape of a trained weight matrix")


def student_t_1d(seed=0, n=N_1D):
    r = np.random.default_rng(seed)
    return _1d("student_t3", r.standard_t(3, n),
               "heavy tails: alpha (the clip) should matter far more than gamma here")


def lognormal_1d(seed=0, n=N_1D):
    r = np.random.default_rng(seed)
    return _1d("lognormal", r.lognormal(0, 0.9, n),
               "strong right skew, hard floor at zero -- an asymmetric warp should pay off")


def uniform_1d(seed=0, n=N_1D):
    r = np.random.default_rng(seed)
    return _1d("uniform", r.uniform(-1, 1, n),
               "the degenerate case: uniform binning is already optimal, so gamma -> 0")


def gmm3_1d(seed=0, n=N_1D):
    """Three well-separated modes -- a genuine clustering problem on the line."""
    r = np.random.default_rng(seed)
    comp = r.choice(3, n, p=[0.4, 0.35, 0.25])
    mu = np.array([-4.0, 0.0, 5.0])
    sd = np.array([0.6, 0.8, 0.5])
    y = comp.astype(np.int64)
    return _1d("gmm3", r.normal(mu[comp], sd[comp]),
               "K=3 is the right answer; does the front find it?", k_true=3, y=y)


def gmm5_unbalanced_1d(seed=0, n=N_1D):
    """Five modes, populations spanning two orders of magnitude.

    The case where equiprobable binning (gamma=1) and MSE-optimal binning
    (gamma=1/3) disagree most: the rare modes carry almost no probability mass
    but they are exactly the structure a clustering method is supposed to find.
    """
    r = np.random.default_rng(seed)
    p = np.array([0.6, 0.25, 0.1, 0.04, 0.01])
    comp = r.choice(5, n, p=p)
    mu = np.array([-6.0, -1.0, 2.0, 6.0, 12.0])
    sd = np.array([1.0, 0.7, 0.4, 0.3, 0.2])
    return _1d("gmm5_unbalanced", r.normal(mu[comp], sd[comp]),
               "rare modes carry the structure but not the probability mass",
               k_true=5, y=comp.astype(np.int64))


def bimodal_asym_1d(seed=0, n=N_1D):
    r = np.random.default_rng(seed)
    comp = r.choice(2, n, p=[0.85, 0.15])
    x = np.where(comp == 0, r.normal(0, 0.5, n), r.gamma(2.0, 2.0, n) + 3.0)
    return _1d("bimodal_asym", x,
               "narrow Gaussian next to a long-tailed lump: the two halves want "
               "different bin densities", k_true=2, y=comp.astype(np.int64))


def gpt2_weights_1d(seed=0, n=200000,
                    tensor="transformer.h.0.attn.c_attn"):
    """One real weight tensor from the parent project's cache, if it is there.

    Not a clustering benchmark -- it is the actual target the method was built
    for, included so the synthetic suite can be checked against the real thing.
    Skipped cleanly when the cache is absent.
    """
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} not found -- run the parent project's cache step, or drop "
            f"'gpt2_weights' from the config's dataset list")
    w = np.load(CACHE)[tensor].astype(np.float64).ravel()
    if len(w) > n:
        w = np.random.default_rng(seed).choice(w, n, replace=False)
    return _1d("gpt2_c_attn", w, f"real weights from {tensor}")


# --------------------------------------------------------------------------
# multi-D generators
# --------------------------------------------------------------------------

def _blobs(rng, centers, sd, sizes):
    xs, ys = [], []
    for i, (c, s, m) in enumerate(zip(centers, sd, sizes)):
        xs.append(rng.normal(c, s, size=(m, len(c))))
        ys.append(np.full(m, i))
    return np.vstack(xs), np.concatenate(ys)


def s_set(seed=0, n_clusters=15, n=5000, overlap=1.0):
    """Fraenti's S-sets: Gaussian clusters on a plane, overlap dialled by `overlap`."""
    r = np.random.default_rng(seed)
    centers = r.uniform(0, 100, size=(n_clusters, 2))
    sd = np.full((n_clusters, 2), 4.0 * overlap)
    sizes = np.full(n_clusters, n // n_clusters)
    x, y = _blobs(r, centers, sd, sizes)
    return Dataset(f"s_set_k{n_clusters}", x, "md", y, n_clusters,
                   "overlapping 2-D Gaussians (S-set family)")


def a_set(seed=0, n_clusters=20, n=6000):
    """A-sets: many well-separated spherical clusters on a lattice."""
    r = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_clusters)))
    grid = np.array([(i, j) for i in range(side) for j in range(side)],
                    dtype=float)[:n_clusters] * 20.0
    grid += r.uniform(-2, 2, grid.shape)
    x, y = _blobs(r, grid, np.full((n_clusters, 2), 2.0),
                  np.full(n_clusters, n // n_clusters))
    return Dataset(f"a_set_k{n_clusters}", x, "md", y, n_clusters,
                   "well-separated spherical clusters (A-set family)")


def unbalance(seed=0):
    """Unbalance: 3 dense clusters and 5 sparse ones, 20:1 in population.

    The adversarial case for any density-matched quantizer: the sparse clusters
    are the interesting ones and they carry 6% of the mass, so a warp that
    follows p(x) will spend almost all its levels on the dense side.
    """
    r = np.random.default_rng(seed)
    centers = np.array([[0., 0.], [10., 0.], [5., 9.],
                        [40., 0.], [45., 6.], [50., 0.],
                        [55., 8.], [60., 2.]])
    sizes = np.array([2000, 2000, 2000, 100, 100, 100, 100, 100])
    x, y = _blobs(r, centers, np.full((8, 2), 2.0), sizes)
    return Dataset("unbalance", x, "md", y, 8,
                   "20:1 population imbalance between dense and sparse clusters")


def dim_set(seed=0, d=32, n_clusters=9, n=2700):
    """DIM-sets: well-separated Gaussians in high dimension.

    Here the product quantizer should suffer worst: prod(K_d) cells over 32 axes
    means even K_d=2 everywhere is 4 billion boxes for 2700 points, so K_eff is
    driven by occupancy, not by the genome.
    """
    r = np.random.default_rng(seed)
    centers = r.normal(0, 8, size=(n_clusters, d))
    x, y = _blobs(r, centers, np.full((n_clusters, d), 1.0),
                  np.full(n_clusters, n // n_clusters))
    return Dataset(f"dim{d}", x, "md", y, n_clusters,
                   f"{n_clusters} separated clusters in {d}-D")


def birch_grid(seed=0, side=10, per=100):
    """Birch-1 style: a regular grid of identical clusters.

    The one multi-D shape a separable quantizer should handle WELL -- the
    structure is axis-aligned and product-shaped, which is exactly the family a
    per-dimension compander can represent. If companding never wins here it
    cannot win anywhere in multi-D.
    """
    r = np.random.default_rng(seed)
    centers = np.array([(i * 10.0, j * 10.0) for i in range(side)
                        for j in range(side)])
    x, y = _blobs(r, centers, np.full((side * side, 2), 1.2),
                  np.full(side * side, per))
    return Dataset(f"birch_grid{side}x{side}", x, "md", y, side * side,
                   "axis-aligned grid of clusters -- the product quantizer's best case")


def _sklearn_real(loader_name: str, pretty: str):
    def load(seed=0):
        try:
            from sklearn import datasets as skd
        except ImportError as exc:
            raise FileNotFoundError(
                f"{pretty} needs scikit-learn (pip install scikit-learn); "
                f"drop it from the config's dataset list to run without") from exc
        bunch = getattr(skd, loader_name)()
        x = np.asarray(bunch.data, dtype=np.float64)
        y = np.asarray(bunch.target, dtype=np.int64)
        return Dataset(pretty, x, "md", y, int(y.max()) + 1,
                       f"real data ({pretty}), {x.shape[0]}x{x.shape[1]}")
    return load


REGISTRY = {
    # 1-D
    "gaussian": gaussian_1d,
    "laplace": laplace_1d,
    "student_t3": student_t_1d,
    "lognormal": lognormal_1d,
    "uniform": uniform_1d,
    "gmm3": gmm3_1d,
    "gmm5_unbalanced": gmm5_unbalanced_1d,
    "bimodal_asym": bimodal_asym_1d,
    "gpt2_weights": gpt2_weights_1d,
    # multi-D
    "s_set": s_set,
    "a_set": a_set,
    "unbalance": unbalance,
    "dim32": dim_set,
    "birch_grid": birch_grid,
    "iris": _sklearn_real("load_iris", "iris"),
    "wine": _sklearn_real("load_wine", "wine"),
    "breast_cancer": _sklearn_real("load_breast_cancer", "breast_cancer"),
    "digits": _sklearn_real("load_digits", "digits"),
}

SUITE_1D = ("gaussian", "laplace", "student_t3", "lognormal", "uniform",
            "gmm3", "gmm5_unbalanced", "bimodal_asym")
SUITE_MD = ("s_set", "a_set", "unbalance", "birch_grid", "dim32", "iris", "wine")


def load(name: str, seed: int = 0, standardize: bool | None = None,
         **kwargs) -> Dataset:
    """Build one dataset by registry name.

    `standardize=None` means "decide by suite": multi-D data is z-scored,
    1-D data is left alone (a monotone rescaling of a single axis changes SSE
    by a constant factor and changes nothing about which partition wins).
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    ds = REGISTRY[name](seed=seed, **kwargs)
    if standardize is None:
        standardize = ds.kind == "md"
    return ds.standardized() if standardize else ds


def resolve(names) -> list[str]:
    """Expand the shorthands 'suite_1d', 'suite_md' and 'all'."""
    out: list[str] = []
    for n in names:
        if n == "suite_1d":
            out.extend(SUITE_1D)
        elif n == "suite_md":
            out.extend(SUITE_MD)
        elif n == "all":
            out.extend(SUITE_1D + SUITE_MD)
        else:
            out.append(n)
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq
