"""Metrics against hand-computable cases, and the comparison logic against
cases where the right answer is obvious by construction.

The report code is where a benchmark most easily flatters itself, so the
dominance and hypervolume paths are tested with fronts whose relationship is
known in advance rather than with whatever a search happens to produce.
"""

import numpy as np
import pytest

from cluster_bench import report
from cluster_bench.metrics import (WORST_DB, adjusted_rand, calinski_harabasz,
                                   davies_bouldin, entropy_bits, evaluate, mse,
                                   silhouette, sse)


def two_blobs(sep=10.0, n=100, seed=0):
    r = np.random.default_rng(seed)
    x = np.vstack([r.normal(0, 0.3, (n, 2)), r.normal(sep, 0.3, (n, 2))])
    labels = np.repeat([0, 1], n)
    cent = np.vstack([x[labels == 0].mean(0), x[labels == 1].mean(0)])
    return x, labels, cent


def test_sse_and_mse_are_consistent():
    x, labels, cent = two_blobs()
    assert mse(x, labels, cent) == pytest.approx(sse(x, labels, cent) / len(x))


def test_db_rewards_separation():
    """Pull the blobs apart and Davies-Bouldin must fall."""
    close = davies_bouldin(*two_blobs(sep=1.5))
    far = davies_bouldin(*two_blobs(sep=30.0))
    assert far < close


def test_db_of_a_single_cluster_is_the_worst_value():
    x = np.random.default_rng(0).normal(size=(50, 2))
    assert davies_bouldin(x, np.zeros(50, dtype=int), x.mean(0)[None, :]) == WORST_DB


def test_db_survives_coincident_centroids():
    """Two clusters on the same point must give a large finite number, not inf."""
    x = np.zeros((10, 2))
    labels = np.repeat([0, 1], 5)
    cent = np.zeros((2, 2))
    v = davies_bouldin(x, labels, cent)
    assert np.isfinite(v) and v <= WORST_DB


def test_silhouette_endpoints():
    x, labels, _ = two_blobs(sep=50.0)
    assert silhouette(x, labels) > 0.9
    r = np.random.default_rng(1)
    x2 = r.normal(size=(300, 2))
    assert silhouette(x2, r.integers(0, 3, 300)) < 0.2


def test_entropy_bits_matches_uniform_case():
    labels = np.repeat(np.arange(8), 50)
    assert entropy_bits(labels) == pytest.approx(3.0)
    assert entropy_bits(np.zeros(100, dtype=int)) == pytest.approx(0.0)


def test_calinski_rewards_separation():
    assert calinski_harabasz(*two_blobs(sep=30.0)) > \
           calinski_harabasz(*two_blobs(sep=1.5))


def test_evaluate_skips_silhouette_on_request():
    x, labels, cent = two_blobs()
    m = evaluate(x, labels, cent, with_silhouette=False)
    assert np.isnan(m["silhouette"]) and np.isnan(m["neg_silhouette"])
    assert np.isfinite(m["mse"]) and np.isfinite(m["davies_bouldin"])


# -- report ----------------------------------------------------------------

def _rows(points, method="companding"):
    return [{"method": method, "k_eff": 4, "mse": a, "davies_bouldin": b,
             "silhouette": 0.5, "entropy_bits": 2.0, "sse": a}
            for a, b in points]


def test_nondominated_picks_the_right_points():
    f = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [3.0, 3.0]])
    assert list(report.nondominated(f)) == [True, True, True, False]


def test_dominance_counts_when_one_front_dominates():
    """b sits strictly inside a's shadow, so every b point is unreachable-free."""
    a = np.array([[1.0, 1.0], [2.0, 0.5]])
    b = np.array([[3.0, 3.0]])
    a_only, b_only, tied = report.dominance_counts(a, b)
    assert a_only == 2 and b_only == 0


def test_degenerate_rows_are_dropped():
    rows = _rows([(1.0, 2.0)]) + [{"method": "companding", "k_eff": 1,
                                   "mse": 0.0, "davies_bouldin": WORST_DB,
                                   "sse": 0.0}]
    kept = report.drop_degenerate(rows, ("mse", "davies_bouldin"))
    assert len(kept) == 1 and kept[0]["k_eff"] == 4


def test_hypervolume_ordering():
    ideal, nadir = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    good = np.array([[0.1, 0.1]])
    bad = np.array([[0.9, 0.9]])
    assert report.hypervolume(good, ideal, nadir) > \
           report.hypervolume(bad, ideal, nadir)
    assert report.hypervolume(np.empty((0, 2)), ideal, nadir) == 0.0


def test_matched_k_table_takes_the_best_candidate_per_k():
    front = [{"k_eff": 8, "mse": 0.5, "davies_bouldin": 1.0, "silhouette": 0.2,
              "entropy_bits": 3.0},
             {"k_eff": 8, "mse": 0.2, "davies_bouldin": 1.4, "silhouette": 0.1,
              "entropy_bits": 3.0}]
    base = {8: {"method": "kmeans_dp", "k_eff": 8, "mse": 0.1,
                "davies_bouldin": 1.2, "silhouette": 0.3, "entropy_bits": 3.0}}
    out = report.matched_k_table(front, base, "mse")
    assert len(out) == 1
    assert out[0]["companding_mse"] == 0.2
    assert out[0]["excess_pct"] == pytest.approx(100.0)


def test_every_internal_index_prefers_outlier_vs_rest():
    """The artifact that made an early 'companding wins on separation' reading wrong.

    On the gaussian set a 1-vs-3999 split scored Davies-Bouldin 0.206 against
    k-means' 0.594 at the same K, and it is not a better clustering: a singleton
    has zero spread and DB is a ratio of spreads.

    The first fix attempted here was "report the silhouette too, it does not
    share the hole". It does. Below, the outlier split beats a balanced split on
    BOTH indices -- silhouette rates it 0.85 against 0.56, because the 400
    points in the big cluster really are far closer to each other than to the
    outlier, and the singleton's own score of 0 is one term in 401.

    So no internal index rescues this, and the honest answers are the two the
    benchmark now ships: report `min_cluster_size` beside every validity score,
    and score the adjusted Rand index against the generating labels, which is
    not a function of partition shape and rates this split at chance.
    """
    r = np.random.default_rng(0)
    x = np.vstack([r.normal(0, 1, (400, 1)), [[-8.0]]])
    outlier = np.zeros(401, dtype=int)
    outlier[-1] = 1
    cent_out = np.array([[x[:-1].mean()], [-8.0]])

    half = (x[:, 0] > np.median(x[:, 0])).astype(int)
    cent_half = np.array([[x[half == 0].mean()], [x[half == 1].mean()]])

    assert davies_bouldin(x, outlier, cent_out) < davies_bouldin(x, half, cent_half)
    assert silhouette(x, outlier) > silhouette(x, half)

    m = evaluate(x, outlier, cent_out, with_silhouette=False)
    assert m["min_cluster_size"] == 1, "the tell must be reported"
    assert m["min_cluster_frac"] == pytest.approx(1 / 401)


def test_adjusted_rand_is_not_fooled_by_partition_shape():
    y = np.repeat([0, 1, 2], 100)
    outlier = np.zeros(300, dtype=int)
    outlier[-1] = 1
    assert adjusted_rand(y, y) == pytest.approx(1.0)
    assert abs(adjusted_rand(outlier, y)) < 0.01
    # Relabelling must not matter -- ARI compares partitions, not label names.
    assert adjusted_rand(np.where(y == 0, 2, np.where(y == 1, 0, 1)), y) == \
        pytest.approx(1.0)
    rand = np.random.default_rng(0).integers(0, 3, 300)
    assert abs(adjusted_rand(rand, y)) < 0.05


def test_evaluate_reports_ari_only_when_given_truth():
    x, labels, cent = two_blobs(n=50)
    assert np.isnan(evaluate(x, labels, cent, with_silhouette=False)["adjusted_rand"])
    m = evaluate(x, labels, cent, with_silhouette=False, y_true=labels)
    assert m["adjusted_rand"] == pytest.approx(1.0)


def test_min_cluster_size_is_reported_for_balanced_partitions():
    x, labels, cent = two_blobs(n=50)
    m = evaluate(x, labels, cent, with_silhouette=False)
    assert m["min_cluster_size"] == 50
    assert m["min_cluster_frac"] == pytest.approx(0.5)
