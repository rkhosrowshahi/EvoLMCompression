"""End-to-end decode latency and peak GPU memory, under a fixed protocol.

Used by the EVALUATION phase, never by the search. Two reasons it does not
belong in the fitness loop: it costs whole generations per candidate, and -- see
the warning below -- it cannot distinguish candidates in this codebase anyway.

    M_peak = M_weights + M_metadata + M_KV + M_activations + M_workspace

`measure` follows SqueezeLLM's `benchmark()` in llama.py so the numbers are
quotable beside their table: a manual per-token decode loop carrying
past_key_values, the MEDIAN PER-TOKEN time as the headline latency, and peak
memory sampled as `torch.cuda.memory_allocated()` after each step. The
allocator's true high-water mark and `max_memory_reserved` are reported
alongside, since their sampled figure misses transient spikes by construction.
All of it is a runtime figure, not a checkpoint size; the two differ by the last
four terms above.

LATENCY IS PER TOKEN throughout this module -- `latency_ms` means ms/token, what
SqueezeLLM prints as "Median:". Multiply by n_tokens for a whole run.

-- THE WARNING ---------------------------------------------------------------

COMPRESSION IN THIS PROJECT IS SIMULATED. `compress_layer` returns a
RECONSTRUCTED fp16 tensor and `MasterWeights.write` copies it in place into the
live module's weight (models.py). Every candidate therefore executes the
identical dense fp16 model: same shapes, same dtype, same kernels, same
allocator behavior. So the MEASURED columns this module produces are constant
across a Pareto front, to within run-to-run noise.

That constancy is not a bug to be worked around and it is not something the
protocol can fix. It is what "simulated" means. Reporting a measured latency
column that happens to wiggle by 2% and calling the wiggle a speedup would be
the failure mode this module exists to prevent, so `summarize_spread` measures
the spread explicitly and says whether it exceeds the noise floor.

What IS honest to report per candidate is a PROJECTION: take the terms that do
not depend on the compression from the measurement, take the weight term from
the bit accounting, and combine them. `project_peak_mb` and
`project_latency_ms` do that, and both are named `*_projected` everywhere they
surface so no reader mistakes one for a measurement.

Getting real measured numbers needs a real quantized inference path: a packed
LUT kernel at the deployable format, plus a sparse kernel for the pruned runs.
Until that exists, the projections are the claim and the measurements are the
fp16 reference point they are anchored to.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch

MB = 1024 * 1024


@dataclass(frozen=True)
class RuntimeMeasurement:
    """One measurement under one protocol. All times in ms, all sizes in MB.

    The protocol follows SqueezeLLM's `benchmark()` in llama.py, so the numbers
    are quotable beside their table:

      * a MANUAL per-token decode loop carrying `past_key_values`, not
        `model.generate()`. generate() adds logits processors, stopping-criteria
        checks and cache bookkeeping to every step, which is real cost in a
        product and pure noise in a kernel comparison.
      * the reported latency is the MEDIAN OF PER-TOKEN times, not the total.
        A median over a few hundred single-token steps is what makes their
        no-warm-up loop sound: the first few tokens carry CUDA context setup and
        autotuning, and the median discards them.
      * `attention_mask` grows with the step index, so the KV cache really does
        lengthen across the run.

    `latency_ms` here therefore means MILLISECONDS PER TOKEN, which is what
    SqueezeLLM prints as "Median:". Multiply by n_tokens for the whole run.
    """

    latency_ms: float           # MEDIAN per-token decode time
    latency_ms_p10: float
    latency_ms_p90: float
    total_ms: float             # sum over every timed step
    # SqueezeLLM samples torch.cuda.memory_allocated() after each token and
    # keeps the running max, which is what their "max memory(MiB)" reports.
    # It misses transient spikes inside a step by construction.
    peak_alloc_mb: float
    # The allocator's own high-water mark over the same region: never lower
    # than peak_alloc_mb, and the honest number for "will this fit". Reported
    # alongside rather than instead, so the comparable figure stays comparable.
    peak_alloc_mb_true: float
    peak_reserved_mb: float     # max_memory_reserved -- what the allocator held
    weight_mb: float            # parameters resident, tied weights counted once
    runtime_mb: float           # peak_alloc_true - weight_mb: KV + activations
    # The same peaks in BYTES, so a results table can quote memory without a
    # MiB-vs-MB conversion step.
    peak_alloc_bytes: float
    peak_alloc_bytes_true: float
    device: str
    n_tokens: int

    def row(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()
                if k not in ("device",)}


def weight_mb(model) -> float:
    """Bytes of resident parameters, in MB. Tied weights counted once.

    This is the term the compression is supposed to move, and the only term of
    M_peak that the bit accounting knows anything about.
    """
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * p.element_size()
    return total / MB


@contextmanager
def _kv_cache_enabled(model):
    """load_model sets use_cache=False for the perplexity path.

    A decode benchmark without a KV cache measures a quadratic re-prefill, not
    decoding, and its memory profile is wrong in the opposite direction (no
    cache to hold). Toggled here and restored, so the search's forward passes
    are unaffected.
    """
    prev = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    try:
        yield
    finally:
        model.config.use_cache = prev


def decode_steps(model, tokenizer, cfg, device, input_ids=None):
    """SqueezeLLM's decode loop: one token at a time, carrying past_key_values.

    Returns `(per_token_ms, sampled_peak_mb)`. Times are in step order, NOT
    sorted, because the caller reports a median and the ordering is what shows
    the first few steps paying for CUDA context setup.

    Faithful to `benchmark()` in SqueezeLLM's llama.py:

      * `model(input_ids[:, i:i+1], past_key_values=..., attention_mask=...)`
        with the mask reshaped to (1, i+1) each step, so the KV cache grows.
      * `torch.cuda.synchronize()` after every step before the clock is read.
      * NO warm-up. The median over a few hundred steps absorbs the outliers,
        which is why their loop can get away without one.
      * peak memory sampled as `torch.cuda.memory_allocated()` after each step,
        running max. That is their reported "max memory(MiB)".

    `input_ids` defaults to deterministic random ids. Decode does the same work
    whatever the token values are, and using random ids keeps the measurement
    free of any corpus dependency. Pass real calibration ids to match their
    setup exactly.

    Split out from `measure` so the loop -- the part most likely to break on a
    transformers upgrade, and the part with no CUDA dependency -- is testable on
    CPU. Only the memory sampling needs a GPU.
    """
    n = cfg.gen_tokens
    if input_ids is None:
        gen = torch.Generator(device="cpu").manual_seed(0)
        input_ids = torch.randint(0, int(model.config.vocab_size),
                                  (cfg.batch_size, n),
                                  generator=gen, dtype=torch.long)
    input_ids = input_ids[:, :n].to(device)
    attention_mask = torch.ones((cfg.batch_size, input_ids.shape[1]), device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times, peak = [], 0.0
    past = None
    with _kv_cache_enabled(model), torch.inference_mode():
        sync()
        for i in range(input_ids.shape[1]):
            t0 = time.perf_counter()
            out = model(
                input_ids[:, i:i + 1],
                past_key_values=past,
                attention_mask=attention_mask[:, : (i + 1)].reshape(
                    (cfg.batch_size, -1)),
            )
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
            if device.type == "cuda":
                peak = max(peak, torch.cuda.memory_allocated(device) / MB)
            past = out.past_key_values
            del out
        sync()
    return times, peak


def measure(model, tokenizer, cfg, device=None,
            input_ids=None) -> RuntimeMeasurement | None:
    """Decode `cfg.gen_tokens` tokens one at a time and record time and memory.

    Returns None when the model is not on CUDA: the memory figures come from the
    CUDA allocator and have no CPU or MPS equivalent, and a wall-clock time
    without them would be half a measurement -- worse, `project_peak_mb`,
    `project_latency_ms` and `latency_model` all need the memory half, so a
    partial result would silently produce meaningless numbers.
    """
    device = device or next(model.parameters()).device
    if device.type != "cuda":
        return None

    torch.cuda.reset_peak_memory_stats(device)
    times, sampled_peak = decode_steps(model, tokenizer, cfg, device, input_ids)
    peak_true = torch.cuda.max_memory_allocated(device) / MB
    peak_reserved = torch.cuda.max_memory_reserved(device) / MB

    ordered = sorted(times)
    w = weight_mb(model)
    med = statistics.median(ordered)
    return RuntimeMeasurement(
        latency_ms=round(med, 4),
        latency_ms_p10=round(ordered[int(0.1 * (len(ordered) - 1))], 4),
        latency_ms_p90=round(ordered[int(0.9 * (len(ordered) - 1))], 4),
        total_ms=round(sum(times), 3),
        peak_alloc_mb=round(sampled_peak, 3),
        peak_alloc_mb_true=round(peak_true, 3),
        peak_reserved_mb=round(peak_reserved, 3),
        weight_mb=round(w, 3),
        runtime_mb=round(peak_true - w, 3),
        peak_alloc_bytes=round(sampled_peak * MB),
        peak_alloc_bytes_true=round(peak_true * MB),
        device=str(device),
        n_tokens=len(times),
    )


# -- projections ---------------------------------------------------------------
#
# Both take the compression-independent part from a real fp16 measurement and
# the weight part from the bit accounting. Neither is a measurement, and the
# column names say so.


def project_peak_mb(size_mb_deployable: float, ref: RuntimeMeasurement) -> float:
    """Peak GPU memory this candidate WOULD reach under a real dequant kernel.

        M_peak = M_deployable_weights + (KV + activations + workspace)

    The first term is `size_mb_deployable` from ModelCost, which already charges
    packed indices, the LUTs, the bitmap or CSR metadata for pruned runs, and
    every tensor left at fp16 (embeddings, LM head where excluded, norms,
    biases). The second is `ref.runtime_mb`, measured on fp16 and carried over
    unchanged because none of those terms depend on how the weights are stored.

    Two things this deliberately does NOT model, both of which push the real
    number UP, so read this as a lower bound:

      * a kernel that dequantizes into an fp16 scratch buffer holds the packed
        weights and the unpacked tile at once;
      * the caching allocator's fragmentation, which is why `peak_reserved_mb`
        exceeds `peak_alloc_mb` in the measurement.

    It is also why a sparse configuration can cost MORE than its dense
    counterpart at equal bit width, exactly as SqueezeLLM reports (2.9 GB dense
    versus 3.1 GB at 0.45% sparsity): the CSR row pointers, column indices and
    full-precision values are in `size_mb_deployable` already, so that inversion
    reproduces here as long as the config's `deployable_format` is not `dense`.
    """
    return round(size_mb_deployable + ref.runtime_mb, 3)


def project_latency_ms(size_mb_deployable: float, ref: RuntimeMeasurement) -> float:
    """Per-token decode latency under a PERFECT memory-bound kernel.

    An optimistic bound, in ms/token to match `ref.latency_ms`.

        t_proj = t_fp16 * (deployable weight bytes / fp16 weight bytes)

    The assumption is that batch-1 decode is entirely weight-bandwidth bound, so
    halving the bytes read per token halves the time. That is the standard
    first-order model and it is the BEST CASE. Every real effect moves the
    number the wrong way:

      * dequantization is work that fp16 does not do at all;
      * a LUT gather is a dependent random access per index;
      * sparse formats decode irregularly and occupy the GPU poorly;
      * packing to non-byte widths costs shifts and masks;
      * at small batch the kernel may be launch-bound, where bytes stop
        mattering entirely.

    So a configuration that looks Pareto-optimal on this projection may be
    SLOWER in reality. Quote it as an upper bound on the speedup available, and
    do not present it as a measured latency -- there is no kernel behind it.

    Note the projection is affine in deployable bytes and therefore a monotone
    transform of `avg_bits`. That is precisely why latency is not one of the
    search objectives: as an optimization axis it would add nothing that the
    memory axis does not already carry.
    """
    if ref.weight_mb <= 0:
        return float("nan")
    return round(ref.latency_ms * (size_mb_deployable / ref.weight_mb), 3)


# -- the search objective lives elsewhere --------------------------------------
#
# `latency_proxy`, the third search objective, is NOT built here. It is a
# per-layer roofline model over the bit accounting, calibrated once and frozen:
# see evolmc/latency.py. This module is the EVALUATION-phase measurement that
# validates it, which is the two-level split the design calls for -- a cheap
# deterministic proxy during evolution, real timing on the final front only.
#
# `project_latency_ms` below is the cruder single-ratio estimate, kept because
# run_eval reports it beside the measurement as a sanity bound.


# -- did the measurement distinguish anything? ---------------------------------


def summarize_spread(rows, key: str, ref: RuntimeMeasurement | None) -> str:
    """Compare the spread of a measured column against its own noise floor.

    The point of this project's simulated compression is that it should NOT
    distinguish candidates on a measured runtime column. This states whether it
    did, rather than leaving a reader to eyeball a CSV and find a 2% wiggle
    persuasive.

    The noise floor is the fp16 reference's own p10..p90 band, which is measured
    under exactly the protocol that produced the column.
    """
    vals = [r[key] for r in rows if r.get(key) is not None]
    if len(vals) < 2:
        return f"{key}: fewer than 2 measurements, nothing to compare"
    lo, hi = min(vals), max(vals)
    span = hi - lo
    rel = span / max(abs(statistics.median(vals)), 1e-9)
    line = (f"{key}: {lo:.3f} .. {hi:.3f} across {len(vals)} candidates "
            f"(span {span:.3f}, {100 * rel:.2f}%)")
    if ref is None or key != "latency_ms":
        return line
    noise = ref.latency_ms_p90 - ref.latency_ms_p10
    if span <= noise:
        return (line + f"\n    within the fp16 noise band ({noise:.3f} ms p10-p90)."
                       " The measurement did not distinguish these candidates,"
                       " which is expected: compression here is simulated and"
                       " every candidate runs the same dense fp16 model.")
    return (line + f"\n    EXCEEDS the fp16 noise band ({noise:.3f} ms p10-p90)."
                   " Investigate before reporting -- with simulated compression"
                   " the model is identical across candidates, so a real spread"
                   " means thermal drift, a busy GPU, or contention, not a"
                   " property of the compression.")
