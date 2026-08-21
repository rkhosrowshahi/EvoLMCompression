"""Genome decoding, and one whole run at toy budgets.

The end-to-end test is deliberately tiny -- it checks that the wiring holds and
that every artefact gets written, not that the method works. Conclusions come
from configs/, not from pytest.
"""

import json

import numpy as np
import pytest

from cluster_bench import datasets as ds
from cluster_bench.config import Config, GenomeCfg, SearchCfg, load_config
from cluster_bench.genome import Genome, GenomeSpec
from cluster_bench.problem import ClusteringProblem
from cluster_bench.runner import run_dataset


def test_genome_dimensions():
    assert Genome(GenomeSpec(d=1, residual_genes=6)).n_var == 9
    assert Genome(GenomeSpec(d=4, residual_genes=6)).n_var == 36
    assert Genome(GenomeSpec(d=4, residual_genes=6, share_warp=True)).n_var == 12


def test_k_gene_is_log_uniform_and_hits_both_ends():
    g = Genome(GenomeSpec(d=1, k_min=2, k_max=256))
    assert g.decode(np.zeros(g.n_var)).ks[0] == 2
    assert g.decode(np.ones(g.n_var)).ks[0] == 256
    mid = g.decode(np.full(g.n_var, 0.5)).ks[0]
    # Log-uniform puts the midpoint at the geometric mean, not the arithmetic
    # one; a uniform map would land near 129 and waste most of the gene's
    # resolution on K values that are indistinguishable in practice.
    assert 20 <= mid <= 26


def test_shared_warp_gives_every_dimension_the_same_warp():
    g = Genome(GenomeSpec(d=3, share_warp=True))
    st = g.decode(np.random.default_rng(0).uniform(size=g.n_var))
    assert len(set(st.alphas)) == 1 and len(set(st.gammas)) == 1
    assert all(np.array_equal(st.us[0], u) for u in st.us)


def test_decode_is_clipped_to_the_box():
    g = Genome(GenomeSpec(d=1, k_min=2, k_max=32))
    st = g.decode(np.full(g.n_var, 5.0))
    assert st.ks[0] == 32


def test_ispline_needs_enough_control_points():
    with pytest.raises(ValueError):
        GenomeSpec(residual_type="ispline", residual_genes=2, ispline_degree=3)


@pytest.mark.parametrize("name", ["gmm3", "laplace", "unbalance", "birch_grid"])
def test_datasets_load(name):
    d = ds.load(name, seed=0)
    assert d.n > 0 and d.d >= 1
    assert np.isfinite(d.x).all()
    if d.kind == "md":
        # multi-D is z-scored by default, or the two arms would not be looking
        # at the same geometry
        assert np.allclose(d.x.mean(0), 0, atol=1e-8)


def test_resolve_expands_suites():
    names = ds.resolve(["suite_1d", "gmm3"])
    assert "gaussian" in names and names.count("gmm3") == 1


def test_problem_objectives_are_validated():
    d = ds.load("gmm3", seed=0)
    with pytest.raises(ValueError):
        ClusteringProblem(d, Genome(GenomeSpec(d=1)), ("mse", "silhouette"))


def test_min_k_eff_constraint_is_reported():
    d = ds.load("gmm3", seed=0)
    p = ClusteringProblem(d, Genome(GenomeSpec(d=1)), min_k_eff=4)
    out = {}
    p._evaluate(np.zeros(p.n_var), out)          # K decodes to 2 at the box corner
    assert out["G"][0] > 0, "a 2-cluster partition must violate min_k_eff=4"


def test_end_to_end(tmp_path):
    cfg = Config(
        name="t", datasets=("gmm3",), figures=False, silhouette_max_n=300,
        search=SearchCfg(pop_size=8, n_gen=3, log_every=100),
        genome=GenomeCfg(k_max=16, residual_genes=4))
    cfg.baselines.k_max = 16
    cfg.baselines.k_steps = 3
    cfg.baselines.lloyd_n_init = 2
    cfg.baselines.dp_max_n = 500

    summary = run_dataset("gmm3", cfg, tmp_path, verbose=False)
    dd = tmp_path / "gmm3"
    for f in ("front.csv", "baselines.csv", "matched_k.csv", "convergence.csv",
              "summary.json"):
        assert (dd / f).exists(), f
    assert json.loads((dd / "summary.json").read_text())["dataset"] == "gmm3"
    assert summary["n_eval"] == 24
    assert summary["hv_companding"] >= 0.0
    assert summary["n_front_kmeans"] > 0


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nnot_a_key: 3\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(p)


def test_1d_suite_stays_inside_the_exact_dp_budget():
    """The 1-D suite's value is that its reference is provably optimal.

    If the generators ever outgrow the default `dp_max_n`, ExactKMeans1D
    silently falls back to a subsample and every "excess over k-means" number
    in the primary result quietly becomes a comparison between two heuristics.
    """
    from cluster_bench.config import BaselineCfg
    from cluster_bench.datasets import N_1D
    from cluster_bench.kmeans import ExactKMeans1D

    assert N_1D <= BaselineCfg().dp_max_n
    for name in ("gaussian", "gmm5_unbalanced", "lognormal"):
        x = ds.load(name, seed=0).x[:, 0]
        assert ExactKMeans1D(x, 8, max_n=BaselineCfg().dp_max_n).exact, name


def test_singleton_partition_scores_perfectly_and_must_be_constrained():
    """The failure mode that made a dim32 run report [0.0, 0.0] at generation 1.

    A product quantizer over many axes can give every point its own cell. MSE is
    then 0, and Davies-Bouldin is 0 too, because a singleton cluster has zero
    spread and DB is a ratio of spreads. That partition dominates every real
    solution on both objectives while being no clustering at all -- so the
    metrics are ALLOWED to score it perfectly, and the search must be the thing
    that rules it out.
    """
    from cluster_bench.metrics import evaluate

    x = np.random.default_rng(0).normal(size=(40, 3))
    labels = np.arange(40)
    m = evaluate(x, labels, x.copy(), with_silhouette=False)
    assert m["mse"] == pytest.approx(0.0)
    assert m["davies_bouldin"] == pytest.approx(0.0)

    d = ds.load("birch_grid", seed=0)
    p = ClusteringProblem(d, Genome(GenomeSpec(d=d.d, share_warp=True)),
                          min_k_eff=2, max_k_eff=64)
    out = {}
    p._evaluate(np.ones(p.n_var), out)          # every K_d at its ceiling
    assert len(out["G"]) == 2
    assert out["G"][1] > 0, "an over-split partition must violate max_k_eff"


def test_runner_defaults_the_ceiling_to_the_baseline_cap():
    """No config should be able to leave the ceiling unset by accident."""
    cfg = Config(datasets=("birch_grid",), figures=False,
                 search=SearchCfg(pop_size=4, n_gen=2, log_every=99))
    assert cfg.search.max_k_eff is None
    assert cfg.baselines.match_k_cap == 512


def test_small_datasets_get_a_ceiling_that_leaves_room_for_clusters():
    """iris has 150 points; a fixed cap of 512 let the search isolate every one.

    That produced MSE 0, Davies-Bouldin 0, a one-point front for each method,
    and a hypervolume of 1.210 -- the theoretical maximum -- for BOTH, which
    reads as a perfect tie when in fact nothing was compared.
    """
    from cluster_bench.config import BaselineCfg

    for n, expected_max in ((150, 15), (178, 17), (10000, 512)):
        ceiling = max(2, n // 10)
        assert min(BaselineCfg().match_k_cap, ceiling) == expected_max


def test_degenerate_comparison_box_is_flagged():
    """Identical single-point fronts must be reported, not scored as a tie."""
    from cluster_bench import report

    row = {"method": "x", "k_eff": 4, "mse": 0.0, "davies_bouldin": 0.0,
           "silhouette": 1.0, "entropy_bits": 2.0, "sse": 0.0}
    out = report.compare([dict(row)], [dict(row)], ("mse", "davies_bouldin"))
    assert out["degenerate_box"] is True
    assert out["hv_companding"] == out["hv_kmeans"]

    # The two points must TRADE OFF, or the second is simply dominated and
    # dropped, leaving the same collapsed box.
    spread = [{**row, "mse": 0.0, "davies_bouldin": 2.0},
              {**row, "mse": 1.0, "davies_bouldin": 0.0}]
    assert report.compare(spread, [dict(row)],
                          ("mse", "davies_bouldin"))["degenerate_box"] is False
