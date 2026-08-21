"""Config loading. YAML in, plain dataclasses out, every default stated here.

One file per experiment, differing in as few lines as possible, so two runs are
comparable by inspection -- the same convention the parent project uses for its
ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class SearchCfg:
    pop_size: int = 100
    n_gen: int = 100
    seed: int = 0
    eta_cx: float = 15.0
    eta_mut: float = 20.0
    p_cx: float = 0.9
    log_every: int = 10
    #: Reject partitions with fewer than this many occupied clusters. 2 keeps
    #: the collapsed single-cluster solution out of the front, where it would
    #: otherwise sit unbeaten on the distortion axis at K=1 and mean nothing.
    min_k_eff: int = 2
    #: Ceiling on occupied clusters. None means "use baselines.match_k_cap",
    #: which is also the largest K any k-means arm is fitted at -- so the search
    #: cannot wander into a region where there is no baseline to compare
    #: against. In multi-D it is load-bearing for a second reason: see the
    #: constraint comment in problem.py.
    max_k_eff: int | None = None
    #: Reject partitions containing a cluster smaller than this. 1 leaves the
    #: search unconstrained, which is the DEFAULT and deliberate: solutions that
    #: game Davies-Bouldin with a one-point cluster are a finding about the
    #: index, and hiding them behind a constraint would bury it. The tables
    #: print the smallest cluster next to every DB. Raise this to 2 or more to
    #: run the comparison with that whole family excluded.
    min_cluster_size: int = 1


@dataclass
class GenomeCfg:
    k_min: int = 2
    k_max: int = 256
    alpha_min: float = 0.5
    alpha_max: float = 12.0
    gamma_min: float = 0.0
    gamma_max: float = 1.5
    residual_genes: int = 6
    u_lo: float = -3.0
    u_hi: float = 3.0
    share_warp: bool = False
    residual_type: str = "linear"
    ispline_degree: int = 3
    grid: int = 256
    lloyd_iters: int = 0


@dataclass
class BaselineCfg:
    arms: tuple = ("dp", "lloyd", "sklearn")
    k_min: int = 2
    k_max: int = 256
    k_steps: int = 8
    lloyd_n_init: int = 10
    sklearn_n_init: int = 10
    dp_max_n: int = 4000
    #: Also fit k-means at every K_eff the companding front reached, so the
    #: matched-K table never has to interpolate between ladder rungs.
    match_front_k: bool = True
    #: Ceiling on those matched K. A product quantizer's occupied-cell count is
    #: not bounded by the genome's k_max -- 64 levels on each of two axes can
    #: leave 800 occupied cells -- and fitting Lloyd at K in the thousands
    #: costs more than the row is worth. Rows above the cap are simply absent
    #: from the matched table; the front itself still reports them.
    match_k_cap: int = 512


@dataclass
class Config:
    name: str = "run"
    seed: int = 0
    datasets: tuple = ("suite_1d",)
    standardize: bool | None = None
    objectives: tuple = ("mse", "davies_bouldin")
    silhouette_max_n: int = 2000
    out_dir: str = "results"
    figures: bool = True
    search: SearchCfg = field(default_factory=SearchCfg)
    genome: GenomeCfg = field(default_factory=GenomeCfg)
    baselines: BaselineCfg = field(default_factory=BaselineCfg)


def _fill(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sub = {k: raw.pop(k, {}) or {} for k in ("search", "genome", "baselines")}
    cfg = _fill(Config, raw)
    cfg.search = _fill(SearchCfg, sub["search"])
    cfg.genome = _fill(GenomeCfg, sub["genome"])
    cfg.baselines = _fill(BaselineCfg, sub["baselines"])
    return cfg


def as_dict(cfg: Config) -> dict:
    """Round-trippable view, written next to every run's results."""
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if hasattr(v, "__dataclass_fields__"):
            out[f.name] = {g.name: getattr(v, g.name) for g in fields(v)}
            out[f.name] = {k: list(x) if isinstance(x, tuple) else x
                           for k, x in out[f.name].items()}
        else:
            out[f.name] = list(v) if isinstance(v, tuple) else v
    return out
