"""Configuration dataclasses for the whole pipeline.

Everything the search touches is declared here so a run is reproducible from a
single YAML file. Load with `Config.from_yaml(path)`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

import warnings

import yaml

# Old option names still accepted, so a run's stored config.yaml keeps
# replotting after a rename.
_RENAMED = {
    "ylim_headroom": "ylim_max_ratio",
    "ylim_ceiling": "ylim_max",
    "ylim_ceiling_ratio": "ylim_max_ratio",
    "ylim_floor": "ylim_min",
    "ylim_floor_ratio": "ylim_min_ratio",
    "ylim_lower": "ylim_min",
    "ylim_lower_ratio": "ylim_min_ratio",
    "ylim_upper": "ylim_max",
    "ylim_upper_ratio": "ylim_max_ratio",
}

# `xlim: [lo, hi]` / `ylim: [lo, hi]` used to carry both bounds in one pair.
# They now split across two scalars, so a stored config still loads.
_SPLIT = {"xlim": ("xlim_min", "xlim_max"), "ylim": ("ylim_min", "ylim_max")}

Granularity = Literal["per_tensor", "per_channel", "per_group"]
Binning = Literal["uniform", "quantile", "kmeans"]
Grouping = Literal["global", "type", "block", "block_type"]


@dataclass
class ModelConfig:
    name: str = "gpt2"
    dtype: str = "float16"
    device: str = "cuda"
    # Where to hold the untouched master copy of the weights. "cpu" costs a
    # host->device copy per candidate but halves VRAM; use it on the RTX 3060.
    master_device: str = "cuda"
    trust_remote_code: bool = False
    # Layers matching any of these substrings are never quantized. They stay
    # fp16 and are still *counted* in the compression accounting.
    exclude_patterns: tuple[str, ...] = ("lm_head", "embed", "wte", "wpe")


@dataclass
class QuantConfig:
    granularity: Granularity = "per_channel"
    group_size: int = 128  # only used when granularity == "per_group"
    binning: Binning = "uniform"
    kmeans_iters: int = 12
    # How the genome maps to a codebook size.
    #   "choices" -- pick from the discrete k_choices ladder below.
    #   "integer" -- any integer in [k_min, k_max], log-spaced.
    #
    # Which is right depends on granularity, because it decides whether the
    # codebook term is visible. Index width is ceil(log2 K), so every K inside
    # a band costs the same indices and differs only in codebook size. At
    # per_tensor that band spans ~0.0004 bpw on GPT-2 -- the top of the band
    # (a power of two) wins outright and the extra integers buy nothing. At
    # per_channel the same band spans ~1.3 bpw, so intermediate K are genuine
    # operating points and restricting to powers of two discards most of them.
    k_encoding: Literal["choices", "integer"] = "choices"
    k_min: int = 2
    k_max: int = 256
    # Candidate codebook sizes the search may choose from, per variable group.
    # Also supplies the reference points and warm-start seeds in either
    # encoding, so the baselines stay at interpretable powers of two.
    k_choices: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)
    # Bit width used to store each codebook entry.
    codebook_bits: int = 16
    # Reconstructions cached per (layer, K). Only active when pruning is off,
    # where the result is deterministic. Each entry holds a full weight tensor,
    # so the cap bounds memory: n_entries x layer_size x 4 bytes.
    cache_entries: int = 256


@dataclass
class PruneConfig:
    enabled: bool = True
    # "sigma": threshold = t * per-group scale (std). Well-conditioned across
    # layers, which matters a lot for the EA. "raw": threshold is used directly.
    mode: Literal["sigma", "raw"] = "sigma"
    t_max: float = 2.0  # upper bound on each of (t_lo, t_hi)
    # Pruned weights are folded into the codebook as a reserved zero codeword,
    # so pruning costs no extra mask bits and shows up as entropy reduction.
    reserve_zero_codeword: bool = True


@dataclass
class VariableConfig:
    """Controls the dimensionality of the search space.

    global      -> 1 K variable  (+2 pruning vars)
    type        -> 1 per projection type (q/k/v/o/gate/up/down)
    block       -> 1 per transformer block
    block_type  -> 1 per (block, type) == fully per-layer
    """

    k_grouping: Grouping = "type"
    prune_grouping: Grouping = "global"


@dataclass
class DataConfig:
    calib_dataset: str = "c4"
    eval_dataset: str = "wikitext2"
    seqlen: int = 2048
    # Sequences used for the *proxy* fitness inside the EA loop.
    n_proxy_seq: int = 16
    # Sequences used for the full evaluation of the final Pareto front.
    n_eval_seq: int = 128
    seed: int = 0
    cache_dir: str = ".cache/evolmc"


@dataclass
class SearchConfig:
    algorithm: Literal["nsga2", "unsga3", "moead"] = "nsga2"
    pop_size: int = 40
    n_gen: int = 30
    seed: int = 0
    # Optional hard budget; candidates above this bpw are infeasible.
    max_bpw: float | None = None
    # Objective 2: "bpw_target" matches what GPTQ/AWQ papers quote;
    # "bpw_model" is the honest whole-checkpoint number.
    size_objective: Literal["bpw_target", "bpw_model"] = "bpw_target"
    # How generation 0 is built.
    #   "linspace" -- pop_size uniform-K individuals, K spread evenly in log
    #                 space across the whole range. Generation 0 then lies
    #                 exactly on the fixed-bit baseline curve, so any later
    #                 improvement is visibly the search's doing.
    #   "ladder"   -- one individual per k_choices entry, random fill after.
    #   "random"   -- uniform random genes.
    init: Literal["linspace", "ladder", "random"] = "linspace"
    # Deprecated alias: warm_start=False forces init="random".
    warm_start: bool = True
    # Evaluate the uniform-K configurations once before the search. These are
    # the fixed-bit reference points the front has to beat, and they anchor the
    # frozen plot axes.
    baseline_sweep: bool = True

    # -- operator controls -------------------------------------------------
    # pymoo gives crossover and mutation TWO probabilities each, and confusing
    # them is easy: `prob` is the chance the operator fires on an individual or
    # mating pair at all, `prob_var` is the chance each gene is touched once it
    # does. The familiar "1/n_var" convention refers to prob_var, and pymoo
    # already defaults to min(0.5, 1/n_var) -- leave mutation_prob_var null to
    # get it. Setting mutation_prob to 1/n_var instead silently disables
    # mutation on all but a few percent of individuals.
    crossover_prob: float = 0.9       # per mating pair
    crossover_prob_var: float = 0.5   # per gene, given the pair crosses
    crossover_eta: float = 15.0       # SBX spread; higher = children hug parents
    mutation_prob: float = 1.0        # per individual
    mutation_prob_var: float | None = None   # per gene; null -> 1/n_var
    mutation_eta: float = 20.0        # PM spread; higher = smaller steps
    eliminate_duplicates: bool = True
    n_offsprings: int | None = None   # null -> pop_size (generational)
    # U-NSGA-III / MOEA/D reference directions; null -> pop_size - 1
    ref_dir_partitions: int | None = None
    # MOEA/D only
    moead_neighbors: int = 15
    moead_prob_neighbor_mating: float = 0.7


@dataclass
class LogConfig:
    root: str = "logs"
    # Directory name; auto-generated from timestamp + model + settings if unset.
    run_name: str | None = None
    checkpoint_every: int = 5
    save_history: bool = True


@dataclass
class PlotConfig:
    enabled: bool = True
    every: int = 1  # write a frame every N generations
    formats: tuple[str, ...] = ("png", "pdf")
    dpi: int = 150
    style: Literal["paper", "dark"] = "paper"
    yscale: Literal["log", "linear"] = "log"
    figsize: tuple[float, float] = (7.0, 5.0)
    # Axis bounds, frozen for the whole run so frames are comparable. Any left
    # null is derived once, before the first generation. Every bound here is
    # absolute: it overrides the corresponding ratio, is exempt from ylim_pad,
    # and refit_at_end never moves it.
    xlim_min: float | None = None
    xlim_max: float | None = None
    ylim_min: float | None = None
    ylim_max: float | None = None
    # When deriving ylim, cap the top at this multiple of the median reference
    # perplexity. Without it a single blown-up low-K baseline (perplexity in
    # the millions) sets the ceiling and squashes the region you care about
    # into a sliver. Off-scale points are still counted in the frame.
    # Upper bound, mirroring the lower bound above.
    #   ylim_max        absolute; overrides the ratio, is not padded, and
    #                       refit_at_end never moves it.
    #   ylim_max_ratio  a multiple of the fp16 perplexity. Null (the
    #                       default) means NO CAP: the box opens to the highest
    #                       reference point and refit_at_end raises it further
    #                       to cover every candidate, so nothing is ever drawn
    #                       off-scale.
    #
    # Capping costs you points and buys vertical resolution: one blown-up low-K
    # candidate can stretch the axis across many decades. Anything excluded is
    # counted in the frame's "candidates outside axes" note, never hidden.
    ylim_max: float | None = None
    ylim_max_ratio: float | None = None
    # Where the bottom of the box sits, as a multiple of the lowest reference
    # perplexity (fp16). Lower it to open up room under the fp16 line.
    #
    # Perplexity is exp(cross-entropy), so it is bounded below by 1.0, not by
    # 0 -- PPL 1 means every correct token got probability 1. Set
    # `ylim: [1.0, ...]` to show that theoretical floor. 0 is not representable
    # on a log axis at all, and is rejected with an explanatory error.
    # Absolute y floor. Overrides ylim_min_ratio when set, is exempt from
    # ylim_pad (an explicit floor means exactly that), and is never moved by
    # refit_at_end. 1.0 is the theoretical minimum: perplexity is
    # exp(cross-entropy) and cross-entropy is non-negative.
    ylim_min: float | None = None
    ylim_min_ratio: float = 0.9
    # Breathing room added to BOTH ends of the y box, as a fraction of its
    # span. Measured in log units when yscale is log, so 0.03 on a 6-decade
    # axis is ~0.18 of a decade at each end. Without it the extreme points sit
    # exactly on the spines and read as clipped.
    ylim_pad: float = 0.03
    # Legend background opacity. The frame is deliberately see-through so it
    # never hides population points behind it; the label text is drawn on top
    # and stays fully opaque regardless.
    legend_alpha: float = 0.3
    # Legend text for the fixed-K reference curve. These points sweep K across
    # an exponentially spaced ladder, holding one K for the whole model.
    baseline_label: str = "exponential search (baseline)"
    # Point size for the small K= tags on that curve. Null -> base font - 3.
    annotation_pt: float | None = None
    # Marker diameter in points. Governs the front and baseline markers; the
    # population scatter uses its square, since scatter sizes are areas.
    marker_pt: float = 3.5
    # The box is frozen from the reference points *before* any candidate is
    # evaluated, so a search that beats fp16 lands under the floor and is drawn
    # clipped against the axis. When true, the floor is refit once at the end
    # and every frame is re-rendered in the corrected box -- still one box for
    # the whole run, just fitted to what actually happened. Ignored when
    # plot.ylim is set explicitly.
    refit_at_end: bool = True
    # Encode the per-generation frames into a video when the run finishes.
    # mp4 needs ffmpeg on PATH; gif falls back to Pillow and always works.
    video: tuple[str, ...] = ("mp4", "gif")
    video_fps: int = 4

    # -- paper output ------------------------------------------------------
    # Setting a venue switches the figure to that style's typeface and sizes
    # it to the exact printed width, so LaTeX scales it by 1.0 and the figure
    # text matches the surrounding body text. See plotting.VENUES.
    venue: Literal["none", "ieee", "acm", "neurips", "icml", "lncs"] = "none"
    width: str | float = "column"  # "column", "page", or inches
    aspect: float = 0.66           # height / width
    font_pt: float | None = None   # override the venue default
    usetex: bool = False           # render text with real LaTeX (needs latex+dvipng)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    variables: VariableConfig = field(default_factory=VariableConfig)
    data: DataConfig = field(default_factory=DataConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    log: LogConfig = field(default_factory=LogConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return Config.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Config":
        def build(cls, blob):
            if blob is None:
                return cls()
            fields = {f.name: f for f in dataclasses.fields(cls)}
            kwargs = {}

            blob = dict(blob)
            for key, (lo_name, hi_name) in _SPLIT.items():
                if key not in blob or lo_name not in fields:
                    continue
                pair = blob.pop(key)
                warnings.warn(f"{cls.__name__}.{key} was split into "
                              f"{lo_name} and {hi_name}",
                              DeprecationWarning, stacklevel=2)
                if pair is not None:
                    blob.setdefault(lo_name, pair[0])
                    blob.setdefault(hi_name, pair[1])

            for key, val in blob.items():
                if key in _RENAMED:
                    new = _RENAMED[key]
                    warnings.warn(f"{cls.__name__}.{key} was renamed to {new}",
                                  DeprecationWarning, stacklevel=2)
                    key = new
                if key not in fields:
                    raise KeyError(f"unknown option {cls.__name__}.{key}")
                if isinstance(val, list):
                    val = tuple(val)
                kwargs[key] = val
            return cls(**kwargs)

        return Config(
            model=build(ModelConfig, raw.get("model")),
            quant=build(QuantConfig, raw.get("quant")),
            prune=build(PruneConfig, raw.get("prune")),
            variables=build(VariableConfig, raw.get("variables")),
            data=build(DataConfig, raw.get("data")),
            search=build(SearchConfig, raw.get("search")),
            log=build(LogConfig, raw.get("log")),
            plot=build(PlotConfig, raw.get("plot")),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
