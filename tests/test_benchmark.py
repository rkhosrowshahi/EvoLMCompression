"""Evaluation-phase runtime measurement and the projections built on it.

The memory half of `benchmark.measure` needs a CUDA allocator and cannot run in
CI, so it is covered here by exercising the two halves it is made of: the
generation loop, on CPU with a tiny model, and the pure arithmetic of the
projections, on constructed measurements.

The property these tests are really defending is the one in benchmark.py's
header: compression in this project is simulated, so a MEASURED runtime column
cannot separate candidates, and only the `*_projected` columns should ever be
read as per-candidate results.
"""

from __future__ import annotations

import math

import pytest
import torch

from evolmc import benchmark as bench
from evolmc.config import BenchmarkConfig, Config


def _ref(latency_ms=100.0, weight=200.0, runtime=50.0, p10=99.0, p90=101.0,
         n_tokens=128):
    """A RuntimeMeasurement with the SqueezeLLM-shaped fields.

    latency_ms is MS PER TOKEN, the median over `n_tokens` single-token decode
    steps, matching what benchmark.measure returns.
    """
    return bench.RuntimeMeasurement(
        latency_ms=latency_ms, latency_ms_p10=p10, latency_ms_p90=p90,
        total_ms=latency_ms * n_tokens,
        peak_alloc_mb=weight + runtime,
        peak_alloc_mb_true=weight + runtime,
        peak_reserved_mb=weight + runtime + 10, weight_mb=weight,
        runtime_mb=runtime,
        peak_alloc_bytes=(weight + runtime) * bench.MB,
        peak_alloc_bytes_true=(weight + runtime) * bench.MB,
        device="cuda:0", n_tokens=n_tokens)


# -- projections ---------------------------------------------------------------


def test_peak_projection_adds_runtime_terms_to_the_weight_term():
    """M_peak = M_deployable_weights + (KV + activations + workspace)."""
    ref = _ref(weight=200.0, runtime=50.0)
    assert bench.project_peak_mb(50.0, ref) == pytest.approx(100.0)
    # fp16 round-trip: feeding the reference's own weight term back must
    # reproduce the measured peak, which is the check the fp16 row exists for.
    assert bench.project_peak_mb(ref.weight_mb, ref) == pytest.approx(
        ref.peak_alloc_mb)


def test_peak_projection_never_falls_below_the_runtime_floor():
    """Even a free model still pays for KV cache, activations and workspace.

    A projection that could reach zero would imply compression buys memory the
    weights never occupied, which is the error the decomposition prevents.
    """
    ref = _ref(weight=200.0, runtime=50.0)
    assert bench.project_peak_mb(0.0, ref) == pytest.approx(ref.runtime_mb)


def test_latency_projection_is_linear_in_deployable_bytes():
    """The roofline assumption: halve the bytes read, halve the time."""
    ref = _ref(latency_ms=100.0, weight=200.0)
    assert bench.project_latency_ms(200.0, ref) == pytest.approx(100.0)
    assert bench.project_latency_ms(100.0, ref) == pytest.approx(50.0)
    assert bench.project_latency_ms(400.0, ref) == pytest.approx(200.0)


def test_latency_projection_is_a_monotone_transform_of_size():
    """Which is exactly why latency is not one of the search objectives.

    If this ever stops holding, the docstring in benchmark.py and the "latency
    would add nothing next to the memory axis" argument in the config headers
    both need revisiting.
    """
    ref = _ref()
    sizes = [10.0, 25.0, 60.0, 61.0, 300.0]
    proj = [bench.project_latency_ms(s, ref) for s in sizes]
    assert proj == sorted(proj)


def test_latency_projection_survives_a_degenerate_reference():
    assert math.isnan(bench.project_latency_ms(10.0, _ref(weight=0.0)))


# -- the spread report ---------------------------------------------------------


def test_spread_within_noise_band_is_reported_as_no_separation():
    ref = _ref(p10=99.0, p90=101.0)          # 2.0 ms noise band
    rows = [{"measured_latency_ms": v} for v in (100.0, 100.4, 99.8, 100.9)]
    out = bench.summarize_spread(rows, "measured_latency_ms", ref)
    assert "did not distinguish" not in out  # key is not the bare latency name
    # The noise comparison keys off the exact column name the caller passes.
    rows = [{"latency_ms": v} for v in (100.0, 100.4, 99.8, 100.9)]
    out = bench.summarize_spread(rows, "latency_ms", ref)
    assert "within the fp16 noise band" in out
    assert "did not distinguish" in out


def test_spread_beyond_noise_band_is_flagged_for_investigation():
    ref = _ref(p10=99.9, p90=100.1)          # 0.2 ms noise band
    rows = [{"latency_ms": v} for v in (100.0, 140.0)]
    out = bench.summarize_spread(rows, "latency_ms", ref)
    assert "EXCEEDS" in out
    assert "thermal drift" in out


def test_spread_needs_two_points():
    assert "nothing to compare" in bench.summarize_spread(
        [{"latency_ms": 1.0}], "latency_ms", _ref())
    assert "nothing to compare" in bench.summarize_spread([], "latency_ms", _ref())


def test_spread_ignores_rows_missing_the_column():
    """Rows skipped by benchmark.every carry no measurement and must not count."""
    rows = [{"latency_ms": 10.0}, {}, {"latency_ms": 12.0}, {"other": 5.0}]
    out = bench.summarize_spread(rows, "latency_ms", None)
    assert "across 2 candidates" in out


# -- config --------------------------------------------------------------------


def test_benchmark_block_round_trips_through_yaml_and_back():
    cfg = Config.from_dict({"benchmark": {"gen_tokens": 64, "every": 4}})
    assert cfg.benchmark.gen_tokens == 64
    assert cfg.benchmark.every == 4
    assert cfg.benchmark.batch_size == BenchmarkConfig().batch_size
    assert Config.from_dict(cfg.to_dict()).benchmark == cfg.benchmark


def test_retired_benchmark_protocol_fields_are_gone():
    """warmup/iters/prompt_tokens belonged to the generate()-based protocol.

    SqueezeLLM's loop has no warm-up and no repeated generations: it decodes
    gen_tokens single tokens and takes the median. A config still naming the old
    fields is a config written against the old protocol, so it must fail loudly
    rather than silently measure something else.
    """
    # dequant_ns_per_lookup belonged to the superseded single-ratio latency
    # model; the roofline proxy takes dequant_ops_per_weight in `latency:`.
    for dead in ("warmup", "iters", "prompt_tokens", "dequant_ns_per_lookup"):
        with pytest.raises(KeyError):
            Config.from_dict({"benchmark": {dead: 3}})


def test_unknown_benchmark_option_is_rejected():
    with pytest.raises(KeyError):
        Config.from_dict({"benchmark": {"n_iters": 3}})


def test_shipped_configs_carry_a_benchmark_block():
    import glob
    paths = sorted(glob.glob("configs/gpt2_scope_*.yaml"))
    assert len(paths) == 18, "3 groupings x 3 target sets x 2 methods"
    for p in paths:
        cfg = Config.from_yaml(p)
        assert cfg.benchmark.enabled, p
        assert cfg.benchmark.gen_tokens > 0, p


# -- the measurement itself ----------------------------------------------------


def test_measure_returns_none_off_cuda():
    """The projections need the memory half, so a partial result is refused."""
    model = torch.nn.Linear(4, 4)
    assert bench.measure(model, None, BenchmarkConfig(),
                         device=torch.device("cpu")) is None


def test_weight_mb_counts_tied_parameters_once():
    """GPT-2 ties lm_head to wte, and double-counting would inflate the term the
    projection replaces."""
    lin = torch.nn.Linear(64, 32, bias=False)
    tied = torch.nn.Linear(64, 32, bias=False)
    tied.weight = lin.weight
    model = torch.nn.Sequential(lin, tied)
    expected = 64 * 32 * lin.weight.element_size() / bench.MB
    assert bench.weight_mb(model) == pytest.approx(expected)


@pytest.mark.slow
def test_generation_loop_runs_on_cpu():
    """Exercises the generate() call, the KV-cache toggle and the timing loop.

    The half of `measure` most likely to break on a transformers upgrade, and
    the half that does not need CUDA.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    model.config.use_cache = False

    cfg = BenchmarkConfig(gen_tokens=6)
    times, peak = bench.decode_steps(model, tok, cfg, torch.device("cpu"))

    assert len(times) == cfg.gen_tokens
    assert all(t > 0 for t in times)
    # Step order is preserved, NOT sorted: the caller takes a median, and the
    # ordering is what shows the first steps paying for context setup.
    assert times != sorted(times) or len(set(times)) == 1
    assert peak == 0.0        # no CUDA allocator on CPU
    # The toggle must leave the model exactly as it found it: the search's
    # perplexity path depends on use_cache being off.
    assert model.config.use_cache is False


# -- the cr_deploy / cr_archive rename ----------------------------------------
#
# `cr_deployable` -> `cr_deploy` and `cr_archival` -> `cr_archive`. 27 finished
# runs store the old spellings in their config.yaml, plot_box.json, front.csv and
# results.csv, so the alias is not politeness -- it is what keeps replot,
# compare_runs, reprice_fronts and backfill_results working on them.


def test_registry_carries_the_new_names_only():
    from evolmc.objectives import REGISTRY

    assert "cr_deploy" in REGISTRY
    assert "cr_archive" in REGISTRY
    assert "cr_deployable" not in REGISTRY
    assert "cr_archival" not in REGISTRY


def test_summary_emits_the_new_keys():
    from evolmc.codec import LayerCost, ModelCost

    mc = ModelCost(layers=[LayerCost(
        name="l", n_weights=100, n_groups=1, k_nominal=4, k_centroids=4,
        sparsity=0.0, k_used_mean=4.0, mse=0.0,
        index_bits_fixed=200, index_bits_huffman=150, index_bits_entropy=140,
        codebook_bits=64, table_bits=16)],
        n_untouched_weights=0)
    s = mc.summary()
    assert "cr_deploy" in s and "cr_archive" in s
    assert "cr_deployable" not in s and "cr_archival" not in s
    # Huffman must never cost more than fixed-width indices on the same layer,
    # which is the whole reason the two ratios are separate columns.
    assert s["cr_archive"] >= s["cr_deploy"] - 1e-9


def test_old_objective_names_still_resolve_with_a_warning():
    """A finished run's plot_box.json names cr_archival; it must still replot."""
    from evolmc.objectives import ObjectiveSet

    with pytest.warns(DeprecationWarning, match="cr_archival -> cr_archive"):
        o = ObjectiveSet(("ppl_proxy", "bpw_target", "cr_archival"))
    assert o.names == ("ppl_proxy", "bpw_target", "cr_archive")

    with pytest.warns(DeprecationWarning, match="cr_deployable -> cr_deploy"):
        d = ObjectiveSet(("ppl_proxy", "cr_deployable"))
    assert d.names == ("ppl_proxy", "cr_deploy")


def test_legacy_and_current_name_together_is_still_a_duplicate():
    """Aliasing must not open a back door to listing one objective twice."""
    from evolmc.objectives import ObjectiveSet

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="repeats"):
            ObjectiveSet(("ppl_proxy", "cr_archive", "cr_archival"))


def test_from_box_reads_a_legacy_run():
    """objectives.from_box is the replot path for every finished run."""
    from evolmc.objectives import from_box

    box = {"objectives": ["ppl_proxy", "bpw_target", "cr_archival"],
           "bounds": [[1.0, 1e5], [1.0, 16.0], [2.78, 1.0]]}
    with pytest.warns(DeprecationWarning):
        objset, bounds = from_box(box)
    assert objset.names == ("ppl_proxy", "bpw_target", "cr_archive")
    assert bounds[2] == (2.78, 1.0)


def test_values_resolves_a_legacy_summary():
    """A summary read back from an old results.csv still scores."""
    from evolmc.objectives import ObjectiveSet

    o = ObjectiveSet(("ppl_proxy", "bpw_target", "cr_archive"))
    legacy = {"bpw_target": 4.0, "cr_archival": 2.5}
    assert o.values(30.0, legacy) == [30.0, 4.0, 2.5]


def test_canonicalize_row_prefers_the_current_name():
    from evolmc.objectives import canonicalize_row

    both = canonicalize_row({"cr_archive": 3.0, "cr_archival": 99.0})
    assert both == {"cr_archive": 3.0}
    assert canonicalize_row({"cr_deployable": 2.0}) == {"cr_deploy": 2.0}
    assert canonicalize_row({"bpw_model": 6.0}) == {"bpw_model": 6.0}


def test_redundancy_still_catches_the_renamed_pair():
    """cr_deploy is exactly 16/bpw_model, so pairing them adds nothing."""
    from evolmc.objectives import ObjectiveSet, check_redundancy

    assert check_redundancy(ObjectiveSet(
        ("ppl_proxy", "bpw_model", "cr_deploy")))
    assert check_redundancy(ObjectiveSet(
        ("ppl_proxy", "bpw_model", "cr_archive"))) == []


def test_shipped_configs_use_the_new_names():
    import glob

    from evolmc.config import Config

    for p in sorted(glob.glob("configs/*.yaml")):
        raw = open(p, encoding="utf-8").read()
        assert "cr_archival" not in raw, p
        assert "cr_deployable" not in raw, p
    for p in sorted(glob.glob("configs/gpt2_scope_*.yaml")):
        cfg = Config.from_yaml(p)
        # TWO objectives. Latency and memory are eval-phase measurements, not
        # search objectives -- compression here is simulated, so a stopwatch
        # cannot distinguish candidates.
        assert cfg.search.objectives == ("ppl_proxy", "bpw_model"), p
        assert cfg.search.size_objective == "bpw_model", p
        for m in ("cr_deploy", "cr_archive", "bytes_deployable"):
            assert m in cfg.search.report_metrics, p
            assert m not in cfg.search.objectives, p


# -- the latency_proxy objective ----------------------------------------------
#
# T(C) = sum_l [ max(B_l/beta_k, F_l/phi_k) + n_k*tau_k ] + T_fixed
#
# Coefficients are fitted on a GPU, so the fitting functions are not exercised
# here. The model arithmetic is, on a hand-built proxy, because that is where a
# silent error would change every candidate's score.


def _proxy(beta=1e6, phi=1e9, tau=0.01, n_kernels=1, fixed=0.5,
           sparse_beta=0.5e6):
    from evolmc.latency import KernelClass, LatencyProxy, LayerGeometry

    def cls(name, b, n):
        return KernelClass(name=name, bytes_per_ms=b, flops_per_ms=phi,
                           launch_ms=tau, n_kernels=n)
    return LatencyProxy(
        classes={"dense": cls("dense", beta, n_kernels),
                 "bitmap": cls("bitmap", sparse_beta, n_kernels + 1),
                 "fp16": cls("fp16", beta, n_kernels)},
        geometry={"l0": LayerGeometry("l0", 100, 200, 20000)},
        fixed_ms=fixed, act_bytes=2, dequant_ops_per_weight=1.0)


class _FakeLayer:
    def __init__(self, name="l0", fmt="dense", total_deployable=80000.0,
                 n_alive=20000):
        self.name, self.fmt = name, fmt
        self.total_deployable = total_deployable      # bits
        self.n_alive = n_alive


class _FakeCost:
    def __init__(self, *layers):
        self.layers = list(layers)


def test_roofline_takes_the_slower_of_the_two_roofs():
    """A kernel is bounded by memory OR compute, whichever is worse."""
    from evolmc.latency import KernelClass

    k = KernelClass("k", bytes_per_ms=1e6, flops_per_ms=1e9, launch_ms=0.0,
                    n_kernels=0)
    assert k.time_ms(1e6, 1e6) == pytest.approx(1.0)      # memory bound
    assert k.time_ms(1e3, 1e9) == pytest.approx(1.0)      # compute bound
    # At the crossover the two agree, and neither term is double counted.
    assert k.time_ms(1e6, 1e9) == pytest.approx(1.0)


def test_compute_roof_stops_the_speedup_once_bytes_get_small():
    """The reason bytes alone are a bad latency proxy.

    Halving the bit width halves the memory term but leaves the FLOP count
    alone, so below the crossover further compression buys nothing. A model
    without the max would keep reporting a speedup the hardware cannot deliver.
    """
    # phi chosen so the compute roof bites once the weight stream gets small.
    p = _proxy(beta=1e6, phi=1e7, tau=0.0, fixed=0.0)
    compute_floor = 20000 * 3 / 1e7          # n_weights * (2 MAC + 1 dequant)

    wide = p.predict(_FakeCost(_FakeLayer(total_deployable=8e8)))    # 100 MB
    mid = p.predict(_FakeCost(_FakeLayer(total_deployable=8e7)))     # 10 MB
    # 1 kB of weights; with 601 B of activation traffic the memory roof is
    # 1.6e-3 ms, well under the 6e-3 ms compute roof.
    narrow = p.predict(_FakeCost(_FakeLayer(total_deployable=8e3)))  # 1 kB

    assert wide > mid                                  # still memory bound
    assert wide == pytest.approx(1e8 / 1e6, rel=1e-3)  # bytes / beta
    # Below the crossover the compute roof takes over and further compression
    # buys NOTHING. A bytes-only proxy would keep reporting a speedup here.
    assert narrow == pytest.approx(compute_floor)
    assert mid > compute_floor


def test_launch_overhead_scales_with_kernels_per_layer():
    p = _proxy(tau=0.01, n_kernels=1, fixed=0.0)
    one = p.predict(_FakeCost(_FakeLayer(fmt="dense")))
    # bitmap carries n_kernels+1 launches in the fixture.
    two = p.predict(_FakeCost(_FakeLayer(fmt="bitmap")))
    assert two - one > 0.009      # at least one extra launch


def test_fixed_term_is_added_once_and_is_candidate_independent():
    p = _proxy(fixed=0.5)
    a = p.predict(_FakeCost(_FakeLayer(total_deployable=80000.0)))
    b = p.predict(_FakeCost(_FakeLayer(total_deployable=8000.0)))
    assert a > b
    # Both pay exactly the same fixed cost, so it cancels in the difference and
    # neither can fall below it.
    assert min(a, b) > p.fixed_ms


def test_unknown_layer_is_skipped_not_guessed():
    """A proxy fitted for another target set must not silently mis-score."""
    p = _proxy(fixed=0.5)
    only_fixed = p.predict(_FakeCost(_FakeLayer(name="not-in-geometry")))
    assert only_fixed == pytest.approx(p.fixed_ms)


def test_sparse_format_can_cost_more_than_dense_at_equal_bytes():
    """The SqueezeLLM inversion: sparsity buys bytes and pays for access.

    Same byte count, but the sparse class streams at half the bandwidth and
    launches an extra kernel, so it is slower. A byte-only proxy would call
    these two identical.
    """
    p = _proxy(beta=1e6, sparse_beta=0.5e6, tau=0.01, n_kernels=1, fixed=0.0)
    dense = p.predict(_FakeCost(_FakeLayer(fmt="dense", total_deployable=80000.0)))
    sparse = p.predict(_FakeCost(_FakeLayer(fmt="bitmap", total_deployable=80000.0)))
    assert sparse > dense


def test_proxy_round_trips_through_json(tmp_path):
    """Coefficients are frozen to a file and reused across a whole sweep."""
    from evolmc.latency import LatencyProxy

    p = _proxy()
    path = tmp_path / "coeffs.json"
    p.save(str(path))
    back = LatencyProxy.load(str(path))
    cost = _FakeCost(_FakeLayer())
    assert back.predict(cost) == p.predict(cost)
    assert back.classes["dense"] == p.classes["dense"]
    assert back.geometry == p.geometry


def test_prediction_is_deterministic():
    """Same chromosome, same score, forever. No CUDA, no forward pass."""
    p = _proxy()
    cost = _FakeCost(_FakeLayer())
    assert len({p.predict(cost) for _ in range(20)}) == 1


def test_fp16_baseline_is_the_slowest_point():
    """predict_fp16 is the speedup denominator, so nothing may exceed it."""
    p = _proxy(fixed=0.1)
    fp16 = p.predict_fp16()
    # 20000 weights at fp16 is 40000 bytes; any quantized candidate reads less.
    quant = p.predict(_FakeCost(_FakeLayer(total_deployable=80000.0)))
    assert fp16 > quant


def test_roof_diagnostic_calls_the_degenerate_case():
    """0 compute-bound layers means latency is affine in the bit total."""
    p = _proxy(beta=1e6, phi=1e12, tau=0.0, fixed=0.0)   # compute never binds
    bound, total, verdict = p.roof_diagnostic(_FakeCost(_FakeLayer()))
    assert bound == 0 and total == 1
    assert "adds nothing" in verdict

    q = _proxy(beta=1e12, phi=1e3, tau=0.0, fixed=0.0)   # compute always binds
    bound, total, verdict = q.roof_diagnostic(_FakeCost(_FakeLayer()))
    assert bound == total == 1
    assert "decoupled" in verdict


def test_csr_span_suffixes_map_to_the_csr_class_not_dense():
    """price_layer records the CHOSEN format, and CSR carries its gap width.

    Under `csr_span_bits: null` or `deployable_format: auto`, LayerCost.fmt is
    "csr2"/"csr6"/"csr8" per layer. Keying the coefficient table on that raw
    string matched nothing and fell through to `dense` -- charging LUT bandwidth
    and one kernel instead of sparse bandwidth and two, i.e. wrong in the
    optimistic direction, on every CSR layer.
    """
    from evolmc.latency import class_key

    assert class_key("dense") == "dense"
    assert class_key("bitmap") == "bitmap"
    assert class_key("csr") == "csr"
    for span in (2, 3, 4, 6, 8, 12):
        assert class_key(f"csr{span}") == "csr"
    # `auto` never reaches predict as a literal: price_layer resolves it to the
    # family it picked. If one ever does, it is a sparse path, not a dense one.
    assert class_key("auto") == "bitmap"


def test_unknown_format_raises_instead_of_falling_back():
    """A silent fallback to `dense` is the bug; loud failure is the fix."""
    from evolmc.latency import class_key

    with pytest.raises(KeyError, match="no latency kernel class"):
        class_key("wavelet")


def test_bitmap_and_csr_are_separate_kernel_classes():
    from evolmc.config import Config
    from evolmc.latency import KernelClass, LatencyProxy, LayerGeometry

    cfg = Config.from_dict({"latency": {"sparse_bandwidth_eff": 0.5,
                                        "csr_bandwidth_eff": 0.3,
                                        "kernels_csr": 3}})
    assert cfg.latency.csr_bandwidth_eff == 0.3
    assert cfg.latency.kernels_csr == 3
    # Defaults leave them equal, so a run that has not measured the difference
    # does not silently assert one.
    d = Config.from_dict({}).latency
    assert d.bitmap_bandwidth_eff is None and d.csr_bandwidth_eff is None

    p = LatencyProxy(
        classes={n: KernelClass(n, b, 1e12, 0.0, k) for n, b, k in
                 (("dense", 1e6, 1), ("bitmap", 5e5, 2), ("csr", 3e5, 3),
                  ("fp16", 1e6, 1))},
        geometry={"l0": LayerGeometry("l0", 10, 10, 1000)}, fixed_ms=0.0)
    cost = _FakeCost(_FakeLayer(total_deployable=8e6))
    dense = p.predict(cost)
    cost.layers[0].fmt = "bitmap"
    bitmap = p.predict(cost)
    cost.layers[0].fmt = "csr6"
    csr = p.predict(cost)
    assert dense < bitmap < csr


# -- the three target sets ----------------------------------------------------
#
# The comparison the gpt2_scope_* configs exist to make. These are cheap
# structural checks; the weight counts come from a real GPT-2 load.


@pytest.mark.slow
def test_the_three_target_sets_are_actually_distinct():
    """`(b) include LM head` and `(c) no exclusion` were IDENTICAL before
    include_embeddings existed.

    discover_targets filters on module TYPE before it looks at exclude_patterns,
    so both nn.Embedding tables were invisible whatever the list said: wte only
    entered through the tied lm_head Linear, and wpe could not enter at all.
    An empty exclude list therefore reproduced the head cell exactly. This is
    the regression guard on the flag that fixed it.
    """
    from evolmc.config import ModelConfig
    from evolmc.models import (
        count_untouched_weights, discover_targets, load_model,
    )

    model, _ = load_model(ModelConfig(device="cpu", dtype="float32"))
    scopes = {
        "core": (["lm_head", "embed", "wte", "wpe"], False),
        "head": (["embed", "wte", "wpe"], False),
        "full": ([], True),
    }
    seen = {}
    for name, (excl, emb) in scopes.items():
        t = discover_targets(model, excl, include_embeddings=emb)
        seen[name] = (len(t), sum(x.n_weights for x in t),
                      count_untouched_weights(model, t))

    assert seen["core"] == (48, 84_934_656, 39_505_152)
    assert seen["head"] == (49, 123_532_032, 907_776)
    assert seen["full"] == (50, 124_318_464, 121_344)
    assert len({v for v in seen.values()}) == 3, "the three scopes must differ"

    # Every scope must account for the whole checkpoint exactly once.
    for name, (_, target, untouched) in seen.items():
        assert target + untouched == 124_439_808, name


@pytest.mark.slow
def test_tied_weights_are_claimed_once_when_embeddings_are_included():
    """lm_head.weight IS transformer.wte.weight. Returning both would give
    MasterWeights two entries for one tensor: the second write would overwrite
    the first and the bit accounting would double-count."""
    from evolmc.config import ModelConfig
    from evolmc.models import discover_targets, load_model

    model, _ = load_model(ModelConfig(device="cpu", dtype="float32"))
    assert model.lm_head.weight is model.transformer.wte.weight

    t = discover_targets(model, [], include_embeddings=True)
    ids = [id(x.module.weight) for x in t]
    assert len(ids) == len(set(ids)), "a weight was claimed twice"
    names = {x.name for x in t}
    assert "transformer.wte" in names and "transformer.wpe" in names
    assert "lm_head" not in names      # deduped against wte, which came first


def test_scope_configs_form_the_full_grid():
    """3 groupings x 3 target sets x 2 methods, no duplicates, nothing missing."""
    import glob

    from evolmc.config import Config

    paths = sorted(glob.glob("configs/gpt2_scope_*.yaml"))
    assert len(paths) == 18
    grid = set()
    for p in paths:
        cfg = Config.from_yaml(p)
        method = "uq_pruning" if cfg.prune.enabled else "uq"
        assert p.endswith(f"_{method}.yaml"), (
            f"{p} name and prune.enabled disagree")
        grid.add((cfg.variables.k_grouping, tuple(cfg.model.exclude_patterns),
                  cfg.model.include_embeddings, method))
        # The format has to follow the method or pruning is invisible to f2.
        assert cfg.quant.deployable_format == (
            "bitmap" if cfg.prune.enabled else "dense"), p
        # Everything else is a constant of the experiment.
        assert cfg.quant.binning == "uniform", p
        assert cfg.search.n_gen * cfg.search.pop_size == 50_000, p
        assert cfg.data.seqlen == 1024, p
        assert cfg.data.n_proxy_seq == 8, p
        assert cfg.data.n_eval_seq == 128, p
        assert cfg.search.objectives == ("ppl_proxy", "bpw_model"), p
        # Latency and memory are measured after the search, not optimized.
        assert cfg.benchmark.enabled, p
    assert len(grid) == 18, "duplicate cell in the grid"
    assert len({g for g, _, _, _ in grid}) == 3          # groupings
    assert len({(e, m) for _, e, m, _ in grid}) == 3     # target sets
    assert len({m for _, _, _, m in grid}) == 2          # methods


def test_scope_file_names_state_target_set_and_method():
    """`no_head` / `with_head` / `all` and `uq` / `uq_pruning` are readable in
    the file name, so a run directory is identifiable without opening the YAML."""
    import glob
    import re

    names = {p.replace("\\", "/").split("/")[-1]
             for p in glob.glob("configs/gpt2_scope_*.yaml")}
    pat = re.compile(
        r"^gpt2_scope_(global|block|layer)_(no_head|with_head|all)"
        r"_(uq|uq_pruning)\.yaml$")
    unmatched = sorted(n for n in names if not pat.match(n))
    assert not unmatched, unmatched
    assert len(names) == 18


def test_auto_is_accepted_where_null_means_derive_it():
    """`auto` reads better than `null` for fields that mean "work it out"."""
    from evolmc.config import Config

    cfg = Config.from_dict({
        "search": {"mutation_prob_var": "auto", "n_offsprings": "AUTO",
                   "ref_dir_partitions": " Auto "},
        "quant": {"csr_span_bits": "auto"}})
    assert cfg.search.mutation_prob_var is None
    assert cfg.search.n_offsprings is None
    assert cfg.search.ref_dir_partitions is None
    assert cfg.quant.csr_span_bits is None


def test_auto_is_not_a_blanket_rule():
    """quant.deployable_format takes the literal string "auto" as a REAL value.

    Normalizing it to None would turn per-layer format selection into a crash
    deep in price_layer, which is why the alias is restricted to an explicit
    field list.
    """
    from evolmc.config import Config

    assert Config.from_dict(
        {"quant": {"deployable_format": "auto"}}).quant.deployable_format == "auto"


def test_scope_configs_leave_mutation_rate_to_n_var():
    """A pinned per-gene rate makes disruption scale with dimensionality, and
    dimensionality is the axis this experiment varies. See the config note."""
    import glob

    from evolmc.config import Config

    for p in sorted(glob.glob("configs/gpt2_scope_*.yaml")):
        cfg = Config.from_yaml(p)
        assert cfg.search.mutation_prob_var is None, p
        assert cfg.search.crossover_prob_var == 0.5, p   # canonical SBX
        assert cfg.search.mutation_prob == 1.0, p
        assert cfg.search.crossover_prob == 0.9, p
        raw = open(p, encoding="utf-8").read()
        assert "mutation_prob_var: auto" in raw, p
