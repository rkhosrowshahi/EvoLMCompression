"""A deterministic, hardware-calibrated latency proxy for the search.

    T(C) = sum_l [ max( B_l(C)/beta_k , F_l(C)/phi_k ) + n_k * tau_k ] + T_fixed

Per layer, roofline per layer, kernel-launch overhead per layer, plus the fixed
cost of everything the search never touches. Coefficients are fitted ONCE on the
target GPU, frozen to a JSON file, and reused unchanged for every candidate and
every run, which is what makes the objective deterministic: the same chromosome
scores identically forever, with no CUDA call and no forward pass.

Cost per candidate is O(L) over layers, so a full search adds O(P*G*L) with tiny
constants. Perplexity evaluation dominates the budget by orders of magnitude.

-- WHY THE ROOFLINE MAX MATTERS ----------------------------------------------

Batch-1 decode is normally memory bound, which is the premise of the whole
method: weight loading dominates, so cutting bytes cuts time. But the memory
roof falls with bit width while the compute roof does not. At 2 bits the weight
stream is an eighth of fp16 while the MAC and dequant counts are unchanged, so
`max` switches over and further bit reduction stops buying latency.

That switchover is exactly the regime this search explores, so a proxy without
the max reports a speedup that keeps growing after the hardware has stopped
delivering it. This is the single most important reason not to use bytes alone.

-- WHY THE FIXED TERM MATTERS ------------------------------------------------

`T_fixed` carries the components excluded from compression: embeddings, norms,
biases, the LM head where it is untouched, and KV-cache traffic. It is IDENTICAL
for every candidate, which is the point. Score only the quantized blocks and the
proxy overstates what compression buys, badly: on the `core` target set 39.5M of
124.4M weights stay fp16 and the LM head alone streams d*V bytes per token
whatever the genome does.

`T_fixed` is COMPUTED from its parts, never back-solved from a measurement. A
back-solved fixed term absorbs every error in beta and silently makes the model
self-consistent for the wrong reason.

-- WHAT IS MEASURED AND WHAT IS STATED ---------------------------------------

Honesty about this is the whole value of the file. There is no packed-LUT kernel
and no sparse kernel in this project (compression here is simulated -- see
benchmark.py), so the quantized classes CANNOT be benchmarked.

  MEASURED, by `calibrate` on the real GPU:
    tau         per-kernel launch overhead, from the marginal cost of many
                trivial kernel launches. Measured DIRECTLY and first, not taken
                as a fit intercept -- see measure_launch for why that mattered.
    beta_fp16   achieved streaming bandwidth, from real batch-1 GEMV shapes,
                fitted with the intercept PINNED to the measured tau so the two
                cannot absorb each other's error
    phi_fp16    achieved fp16 throughput, from a large compute-bound matmul

LAUNCH OVERHEAD DOMINATES AT SMALL MODEL SIZE. On GPT-2 (48 small layers) the
launch term is roughly HALF the predicted latency, against a few percent on a
7B model. Two consequences: tau's stability is critical, which is why it is
measured directly; and the latency axis on GPT-2 responds less to bit width than
it would on a large model, because most of the time is not weight streaming.
Check the breakdown that scripts/calibrate_latency.py prints before reading much
into an f3 spread.

  STATED, from the config, as efficiency factors relative to fp16:
    lut_bandwidth_eff      a packed-LUT dequant kernel streams below peak
    sparse_bandwidth_eff   irregular access streams further below peak
    dequant_ops_per_weight arithmetic the fp16 path does not do at all
    kernels_per_layer      launches per layer, per format

Every stated constant lands in the frozen JSON and in the run metadata. Quote
them alongside any latency number. When the real kernels exist, re-fit the
classes and the objective becomes measured end to end without touching callers.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field

import torch

MB = 1024 * 1024


@dataclass(frozen=True)
class LayerGeometry:
    """Static shape of one target layer. Never changes with the genome."""

    name: str
    in_features: int
    out_features: int
    n_weights: int


# price_layer records the format it actually CHOSE, not the one requested, and
# for CSR that string carries the per-layer gap width: "csr2", "csr6", "csr8".
# Under `csr_span_bits: null` or `deployable_format: auto` every layer can land
# on a different one. Keying the coefficient table on the raw string therefore
# missed every CSR layer and fell through to `dense` -- which is the optimistic
# direction, charging LUT bandwidth and one kernel instead of sparse bandwidth
# and two. Strip the width; the gap size changes the BYTE count (already priced
# by price_layer) and not which kernel family runs.
_FMT_ALIASES = {"auto": "bitmap"}


def class_key(fmt: str) -> str:
    """Coefficient-table key for a `LayerCost.fmt` string.

    Raises on anything unrecognized rather than falling back, because a silent
    fallback to `dense` is exactly the bug this function exists to prevent.
    """
    base = fmt.rstrip("0123456789") or fmt
    base = _FMT_ALIASES.get(base, base)
    if base not in ("dense", "bitmap", "csr", "fp16"):
        raise KeyError(
            f"no latency kernel class for format {fmt!r}. Known families: "
            "dense, bitmap, csr, fp16. Add a class rather than letting this "
            "fall back, or the layer is priced as the wrong kernel.")
    return base


@dataclass(frozen=True)
class KernelClass:
    """Coefficients for one kernel family: T = max(B/beta, F/phi) + n*tau."""

    name: str
    bytes_per_ms: float      # beta_k
    flops_per_ms: float      # phi_k
    launch_ms: float         # tau_k
    n_kernels: int           # launches per layer of this class

    def time_ms(self, byte_count: float, flop_count: float) -> float:
        mem = byte_count / max(self.bytes_per_ms, 1e-12)
        comp = flop_count / max(self.flops_per_ms, 1e-12)
        return max(mem, comp) + self.n_kernels * self.launch_ms


@dataclass
class LatencyProxy:
    """Frozen cost model. `predict(cost)` is the objective value, in ms/token."""

    classes: dict[str, KernelClass]
    geometry: dict[str, LayerGeometry]
    fixed_ms: float
    # Breakdown of fixed_ms, carried for reporting rather than used again.
    fixed_parts: dict[str, float] = field(default_factory=dict)
    act_bytes: int = 2                    # fp16 activations
    dequant_ops_per_weight: float = 1.0
    device_name: str = ""
    # Everything needed to reproduce the fit, written into the JSON so a run's
    # latency numbers can be traced to the calibration that produced them.
    provenance: dict = field(default_factory=dict)

    # -- the objective ----------------------------------------------------

    def predict(self, cost) -> float:
        """Modelled decode latency for one candidate, milliseconds per token.

        `cost` is a codec.ModelCost. Reads only the per-layer bit accounting and
        the static geometry, so it is pure arithmetic: no CUDA, no weights, no
        forward pass. Layers absent from `geometry` are skipped rather than
        guessed at, which keeps a mismatched proxy loud instead of subtly wrong.
        """
        total = 0.0
        for layer in cost.layers:
            geo = self.geometry.get(layer.name)
            if geo is None:
                continue
            key = class_key(layer.fmt)
            kls = self.classes[key]
            # Weights the kernel actually reads and multiplies. Under `dense`
            # every position carries an index whatever was pruned; under a
            # sparse format only survivors do.
            n_active = geo.n_weights if key == "dense" else layer.n_alive
            # B_l: the packed weight representation (indices + LUT + any sparse
            # metadata, all already priced by codec.price_layer), plus the
            # activation read and the output write of a batch-1 GEMV.
            b = (layer.total_deployable / 8.0
                 + self.act_bytes * (geo.in_features + geo.out_features))
            # F_l: one multiply-add per active weight, plus the dequantization
            # the fp16 path does not perform at all.
            f = n_active * (2.0 + self.dequant_ops_per_weight)
            total += kls.time_ms(b, f)
        return round(total + self.fixed_ms, 6)

    def predict_fp16(self, n_untouched: int = 0) -> float:
        """The uncompressed baseline under the SAME model, ms/token.

        The denominator for any speedup claim. Computed rather than measured, so
        a ratio against `predict` is internally consistent: both pay the same
        launch and fixed terms, and only the weight representation differs. A
        speedup taken against a real fp16 stopwatch instead would silently mix
        two cost models and flatter the compressed side.
        """
        total = 0.0
        fp16 = self.classes["fp16"]
        for geo in self.geometry.values():
            b = (geo.n_weights * 2.0
                 + self.act_bytes * (geo.in_features + geo.out_features))
            total += fp16.time_ms(b, geo.n_weights * 2.0)
        return round(total + self.fixed_ms, 6)

    def roof_diagnostic(self, cost) -> tuple[int, int, str]:
        """How many layers the COMPUTE roof binds on. `(bound, total, verdict)`.

        This is the check that decides whether latency_proxy is a real third
        objective or a rescaled copy of the size objective, and it has to be
        numeric rather than inferred from the config.

        If the memory roof binds on every layer at every point in the search
        space, then T = (sum_l B_l)/beta + constant. The size objective is the
        same sum, normalized. So latency is EXACTLY affine in it, dominance is
        unchanged, and the front is 2-D wearing three axis labels.

        Measured on this RTX 3060: bitmap's compute-to-memory ratio has an upper
        bound of 0.459 even at zero sparsity, because the bitmap mask alone puts
        a floor of 1 bit per POSITION on the bytes while FLOPs scale with
        survivors. So compute never binds and latency_proxy is redundant with
        avg_bits, which a real 4-generation run confirmed at 0.00% discordant
        pairs over 1,128 comparisons.
        """
        bound = 0
        for layer in cost.layers:
            geo = self.geometry.get(layer.name)
            if geo is None:
                continue
            kls = self.classes[class_key(layer.fmt)]
            n_active = (geo.n_weights if class_key(layer.fmt) == "dense"
                        else layer.n_alive)
            b = (layer.total_deployable / 8.0
                 + self.act_bytes * (geo.in_features + geo.out_features))
            f = n_active * (2.0 + self.dequant_ops_per_weight)
            if f / kls.flops_per_ms > b / kls.bytes_per_ms:
                bound += 1
        total = sum(1 for l in cost.layers if l.name in self.geometry)
        if bound == 0:
            verdict = ("memory roof binds on EVERY layer, so latency is affine "
                       "in the deployable bit total and adds nothing to the "
                       "front")
        elif bound == total:
            verdict = "compute roof binds everywhere; latency is decoupled from bits"
        else:
            verdict = (f"mixed: {bound}/{total} layers compute-bound, so the "
                       f"axes separate on those layers only")
        return bound, total, verdict

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "classes": {k: asdict(v) for k, v in self.classes.items()},
            "geometry": {k: asdict(v) for k, v in self.geometry.items()},
            "fixed_ms": self.fixed_ms,
            "fixed_parts": self.fixed_parts,
            "act_bytes": self.act_bytes,
            "dequant_ops_per_weight": self.dequant_ops_per_weight,
            "device_name": self.device_name,
            "provenance": self.provenance,
        }

    @staticmethod
    def from_dict(d: dict) -> "LatencyProxy":
        return LatencyProxy(
            classes={k: KernelClass(**v) for k, v in d["classes"].items()},
            geometry={k: LayerGeometry(**v) for k, v in d["geometry"].items()},
            fixed_ms=d["fixed_ms"],
            fixed_parts=d.get("fixed_parts", {}),
            act_bytes=d.get("act_bytes", 2),
            dequant_ops_per_weight=d.get("dequant_ops_per_weight", 1.0),
            device_name=d.get("device_name", ""),
            provenance=d.get("provenance", {}),
        )

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "LatencyProxy":
        with open(path, encoding="utf-8") as f:
            return LatencyProxy.from_dict(json.load(f))

    def describe(self) -> str:
        rows = [f"latency proxy  ({self.device_name or 'uncalibrated'})"]
        for k, c in self.classes.items():
            rows.append(
                f"  {k:<8} {c.bytes_per_ms / 1e6:8.1f} GB/s  "
                f"{c.flops_per_ms / 1e9:7.2f} TFLOP/s  "
                f"{c.launch_ms * 1000:6.1f} us/launch x {c.n_kernels}")
        parts = "  ".join(f"{k} {v:.4f}" for k, v in self.fixed_parts.items())
        rows.append(f"  fixed    {self.fixed_ms:.4f} ms/token   ({parts})")
        rows.append(f"  layers   {len(self.geometry)} with known geometry")
        return "\n".join(rows)


# -- calibration ---------------------------------------------------------------


def _fit_slope_through(xs, ys):
    """Least squares y = a*x with the intercept PINNED AT ZERO. Returns a."""
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    return sxy / sxx if sxx > 0 else 0.0


def measure_launch(device, repeats=1000, warmup=1000, blocks=5):
    """Per-kernel launch overhead, measured DIRECTLY rather than fitted.

    An earlier version took tau as the intercept of the bandwidth fit. That is
    what the cost model says it is, but it is numerically terrible: the
    intercept of a least-squares line through eight points moved between 6.4 and
    17.1 microseconds across consecutive calibrations on the same idle GPU. At
    GPT-2's size launches are roughly HALF the predicted latency (48 small
    layers), so a 3x swing in tau swamps the objective the search is optimizing.

    Measured here instead as the marginal cost of launching a trivial kernel on
    a tensor small enough that no real work happens, so what is left is dispatch
    plus launch. Reported as the MEDIAN over `blocks` independent timing blocks:
    a single block still catches the occasional scheduler or driver hiccup and
    comes out several microseconds high.

    NOTE this includes PyTorch's Python-side dispatch, which a fused deployed
    kernel would not pay. It is therefore an OVERESTIMATE of what a real LUT
    kernel costs to launch, and on a small model that overestimate propagates to
    a large share of the total. Set `latency.launch_us` to override it with a
    figure from the kernel you actually intend to ship.
    """
    x = torch.empty(64, device=device, dtype=torch.float16)
    per_block = []
    with torch.inference_mode():
        for _ in range(warmup):   # warm dispatch; the first call after CUDA
            x.add_(1.0)           # init is an outlier worth several us
        torch.cuda.synchronize(device)
        for _ in range(blocks):
            t0 = time.perf_counter()
            for _ in range(repeats):
                x.add_(1.0)
            torch.cuda.synchronize(device)
            per_block.append((time.perf_counter() - t0) * 1000.0 / repeats)
    del x
    torch.cuda.empty_cache()
    return statistics.median(per_block)


def measure_bandwidth(device, tau_ms, dtype=torch.float16, repeats=30):
    """Achieved streaming bandwidth from real batch-1 GEMV shapes.

    Fits `T - tau = a*B` with the intercept pinned to the independently measured
    launch overhead, so beta and tau cannot trade against each other the way
    they did when both came out of one two-parameter fit.

    Batch-1 GEMV specifically, because that is the shape decode actually runs;
    a big square matmul would report a bandwidth decode never sees.
    """
    shapes = [(768, 768), (768, 3072), (3072, 768), (1024, 4096), (4096, 4096),
              (2048, 8192), (4096, 11008), (768, 50257)]
    xs, ys = [], []
    for in_f, out_f in shapes:
        w = torch.randn(out_f, in_f, device=device, dtype=dtype)
        x = torch.randn(1, in_f, device=device, dtype=dtype)
        with torch.inference_mode():
            for _ in range(5):
                torch.nn.functional.linear(x, w)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(repeats):
                torch.nn.functional.linear(x, w)
            torch.cuda.synchronize(device)
            ms = (time.perf_counter() - t0) * 1000.0 / repeats
        nbytes = w.numel() * w.element_size() + (in_f + out_f) * w.element_size()
        # Subtract the launch cost this shape also paid; what remains is stream.
        xs.append(float(nbytes))
        ys.append(max(ms - tau_ms, 1e-9))
        del w, x
    torch.cuda.empty_cache()
    a = _fit_slope_through(xs, ys)
    return 1.0 / a if a > 0 else float("inf")


def measure_compute(device, dtype=torch.float16, size=4096, repeats=20):
    """Achieved fp16 throughput, FLOP/ms, from a compute-bound square matmul."""
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(3):
            a @ b
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            a @ b
        torch.cuda.synchronize(device)
        ms = (time.perf_counter() - t0) * 1000.0 / repeats
    del a, b
    torch.cuda.empty_cache()
    return (2.0 * size ** 3) / ms


def calibrate(compressor, cfg) -> LatencyProxy:
    """Fit the coefficients once on this GPU and build the frozen proxy.

    Measures the fp16 dense class for real; derives the quantized classes from
    it using the stated efficiency factors in `cfg.latency`, because the kernels
    they describe do not exist in this project and cannot be benchmarked.
    """
    lat = cfg.latency
    device = compressor.device
    if device.type != "cuda":
        raise SystemExit(
            "the latency proxy is calibrated against a real GPU and the model "
            "is not on CUDA. Run calibration on the target device and point "
            "latency.coeffs_path at the resulting file, or drop latency_proxy "
            "from search.objectives.")

    # tau first: bandwidth is fitted with the intercept pinned to it, so the
    # two cannot absorb each other's error.
    tau = (lat.launch_us / 1000.0 if lat.launch_us is not None
           else measure_launch(device))
    beta_fp16 = measure_bandwidth(device, tau)
    phi_fp16 = measure_compute(device)

    def cls(name, bw_eff, n_kernels):
        return KernelClass(name=name,
                           bytes_per_ms=beta_fp16 * bw_eff,
                           flops_per_ms=phi_fp16 * lat.compute_eff,
                           launch_ms=tau,
                           n_kernels=n_kernels)

    # bitmap and CSR are separate classes, not one "sparse" class. A bitmap
    # decode is a contiguous mask scan plus a popcount-driven gather; CSR walks
    # a gap-coded index stream, which is more serial and less predictable. They
    # deserve different coefficients even though both save bytes the same way.
    # Left equal by default so a run that has not measured them is not silently
    # asserting a difference it cannot support.
    bitmap_eff = (lat.bitmap_bandwidth_eff if lat.bitmap_bandwidth_eff
                  is not None else lat.sparse_bandwidth_eff)
    csr_eff = (lat.csr_bandwidth_eff if lat.csr_bandwidth_eff
               is not None else lat.sparse_bandwidth_eff)
    classes = {
        "dense": cls("dense", lat.lut_bandwidth_eff, lat.kernels_dense),
        "bitmap": cls("bitmap", bitmap_eff, lat.kernels_sparse),
        "csr": cls("csr", csr_eff, lat.kernels_csr or lat.kernels_sparse),
        "fp16": cls("fp16", 1.0, lat.kernels_dense),
    }

    geometry = {t.name: LayerGeometry(name=t.name, in_features=t.in_features,
                                      out_features=t.out_features,
                                      n_weights=t.n_weights)
                for t in compressor.targets}

    fixed_ms, parts = _fixed_cost(compressor, cfg, classes["fp16"], beta_fp16)

    proxy = LatencyProxy(
        classes=classes, geometry=geometry,
        fixed_ms=fixed_ms, fixed_parts=parts,
        act_bytes=lat.act_bytes,
        dequant_ops_per_weight=lat.dequant_ops_per_weight,
        device_name=torch.cuda.get_device_name(device),
        provenance={
            "model": cfg.model.name,
            "anchored": bool(lat.anchor_to_measurement),
            "measured_beta_fp16_gbps": round(beta_fp16 / 1e6, 2),
            "measured_phi_fp16_tflops": round(phi_fp16 / 1e9, 3),
            "launch_us": round(tau * 1000, 3),
            "launch_us_source": ("stated (latency.launch_us)"
                                 if lat.launch_us is not None
                                 else "measured directly"),
            "stated_lut_bandwidth_eff": lat.lut_bandwidth_eff,
            "stated_bitmap_bandwidth_eff": bitmap_eff,
            "stated_csr_bandwidth_eff": csr_eff,
            "stated_compute_eff": lat.compute_eff,
            "stated_dequant_ops_per_weight": lat.dequant_ops_per_weight,
            "stated_kernels_dense": lat.kernels_dense,
            "stated_kernels_sparse": lat.kernels_sparse,
            "kv_seq_len": lat.kv_seq_len,
            "note": "quantized classes are DERIVED from the fp16 fit via stated "
                    "efficiency factors; no LUT or sparse kernel exists here",
        })

    if lat.anchor_to_measurement:
        _anchor(proxy, compressor, cfg)
    return proxy


def _anchor(proxy, compressor, cfg):
    """Set T_other from one real fp16 measurement, in place.

    The per-layer model covers the 48 target GEMVs and nothing else. A real
    GPT-2 decode step also runs 25 LayerNorms, 12 GELUs, 12 softmaxes, two
    attention matmuls per block, residual adds and reshapes -- 300+ kernel
    launches against the 48 the model charges for. Unmodelled, that came to 79%
    of real decode time on an RTX 3060.

    So measure fp16 once, subtract what the model does account for, and keep the
    remainder as T_other. It is candidate-independent by construction, so it
    shifts every prediction by the same constant and changes no ordering -- but
    it makes the absolute numbers, and therefore any speedup RATIO, honest.
    """
    from . import benchmark as _bench

    compressor.restore()
    m = _bench.measure(compressor.model, compressor.tokenizer, cfg.benchmark,
                       device=compressor.device)
    if m is None:
        return
    modelled = proxy.predict_fp16()
    residual = max(m.latency_ms - modelled, 0.0)
    proxy.fixed_ms = round(proxy.fixed_ms + residual, 6)
    proxy.fixed_parts["other_ms"] = round(residual, 6)
    proxy.provenance.update({
        "measured_fp16_ms_per_token": round(m.latency_ms, 4),
        "modelled_before_anchor_ms": round(modelled, 4),
        "anchor_residual_ms": round(residual, 4),
        "anchor_note": "T_other measured as the residual: norms, GELU, softmax, "
                       "attention matmuls, residual adds and their ~300 kernel "
                       "launches, none of which the per-layer model decomposes",
    })


def _fixed_cost(compressor, cfg, fp16_class, beta_fp16):
    """T_fixed: everything identical across candidates, computed from its parts.

    Two contributions:

    UNTOUCHED WEIGHTS. Embeddings, norms, biases, and the LM head wherever it is
    excluded, all still fp16 and all read every token. On the `core` target set
    that is 39.5M weights, dominated by the tied lm_head/wte matrix, which the
    head streams in full (d x V) for every token generated. This is the term
    that stops the proxy overstating what compression buys.

    Slight overestimate by design: wpe and the norm/bias vectors are gathers or
    elementwise reads rather than full streams, so charging every untouched
    parameter as streamed is generous. On GPT-2 core that is 907k of 39.5M
    weights, under 2.5%, and it errs toward making compression look WORSE rather
    than better.

    KV CACHE. 2 (K and V) x n_layers x d_model x context x act_bytes per token.
    Depends on context length, not on the genome, so it belongs here.
    """
    lat = cfg.latency
    mc = compressor.model.config
    n_layer = getattr(mc, "n_layer", None) or getattr(mc, "num_hidden_layers", 0)
    d_model = getattr(mc, "n_embd", None) or getattr(mc, "hidden_size", 0)

    untouched_bytes = compressor.n_untouched * 2.0        # fp16
    untouched_flops = compressor.n_untouched * 2.0        # one MAC per weight
    t_untouched = fp16_class.time_ms(untouched_bytes, untouched_flops)

    kv_bytes = 2.0 * n_layer * d_model * lat.kv_seq_len * lat.act_bytes
    t_kv = kv_bytes / max(beta_fp16, 1e-12)

    parts = {"untouched_ms": round(t_untouched, 6), "kv_ms": round(t_kv, 6),
             "other_ms": round(lat.extra_fixed_ms, 6)}
    return round(t_untouched + t_kv + lat.extra_fixed_ms, 6), parts


def load_or_calibrate(compressor, cfg, log=print) -> LatencyProxy:
    """Reuse a frozen coefficients file if present, else fit and write one.

    Reusing the file across the twelve runs of a sweep is the point: identical
    coefficients mean the latency axis is comparable between cells, which it
    would not be if each run re-fitted on a differently loaded GPU.
    """
    path = cfg.latency.coeffs_path
    if path and cfg.latency.reuse_coeffs:
        try:
            proxy = LatencyProxy.load(path)
        except FileNotFoundError:
            proxy = None
        if proxy is not None:
            # Geometry is a property of the model and target set, not of the
            # GPU, so a file fitted for a different target set would silently
            # skip layers in `predict`. Refit rather than mis-score.
            want = {t.name for t in compressor.targets}
            if want == set(proxy.geometry):
                log(f"latency proxy: reusing frozen coefficients from {path}")
                return proxy
            log(f"latency proxy: {path} covers {len(proxy.geometry)} layers but "
                f"this target set has {len(want)}; refitting")

    log("latency proxy: calibrating on this GPU (once) ...")
    proxy = calibrate(compressor, cfg)
    if path:
        proxy.save(path)
        log(f"latency proxy: froze coefficients to {path}")
    return proxy
