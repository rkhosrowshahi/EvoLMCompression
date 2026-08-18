"""Applies a genome to the live model and prices the result.

`Compressor` owns the loaded model, the master weights and the codebook cache.
One instance is created per run and reused for every fitness evaluation -- the
model is never reloaded and never moved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .codec import ModelCost, price_layer
from .grouping import Genome, layer_granularity
from .models import (
    MasterWeights,
    count_untouched_weights,
    discover_targets,
    load_model,
)
from .quantize import LayerPrecompute, compress_layer


@dataclass
class Candidate:
    cost: ModelCost
    settings: dict
    apply_seconds: float


class Compressor:
    def __init__(self, cfg, model=None, tokenizer=None):
        self.cfg = cfg
        if model is None:
            model, tokenizer = load_model(cfg.model)
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        self.targets = discover_targets(
            model, cfg.model.exclude_patterns,
            include_1d=getattr(cfg.model, "include_1d", False))
        self.n_untouched = count_untouched_weights(model, self.targets)
        self.master = MasterWeights(self.targets, cfg.model.master_device)
        self.genome = Genome(self.targets, cfg.quant, cfg.prune, cfg.variables)
        self.cache = LayerPrecompute(enabled=not cfg.prune.enabled,
                                     max_entries=cfg.quant.cache_entries)
        self.n_evals = 0
        # Populated by calibrate_wanda() when prune.mode == "wanda"; every
        # other mode leaves this None and never reads it.
        self.wanda_norms = None

    def calibrate_wanda(self, windows) -> "wanda.ActivationNorms":
        """Measure per-input-feature activation norms once, before the search.

        Required before any .apply() call when cfg.prune.mode == "wanda" --
        compress_layer raises a clear error otherwise rather than silently
        pruning by magnitude.
        """
        from . import wanda

        result = wanda.calibrate(self, windows)
        self.wanda_norms = result.norms
        return result

    # -- core ---------------------------------------------------------------

    def apply(self, x: np.ndarray) -> Candidate:
        """Compress the live model in place according to genome `x`."""
        t0 = time.perf_counter()
        settings = self.genome.decode(x)
        cost = ModelCost(n_untouched_weights=self.n_untouched)

        companding = self.cfg.quant.binning == "companding"
        widths = self.cfg.quant.binning == "widths"
        for layer in self.targets:
            s = settings[layer.name]
            # 1-D norms/biases are quantized but never pruned: zeroing a
            # LayerNorm scale or a bias is almost always fatal, and Wanda has
            # no input-feature score for a vector.
            t_lo = 0.0 if layer.is_vector else s.t_lo
            t_hi = 0.0 if layer.is_vector else s.t_hi
            extra = None
            if companding:
                extra = (round(s.alpha, 6), round(s.gamma, 6),
                         tuple(round(float(v), 6) for v in s.u),
                         s.force_zero, s.reassign)
            elif widths:
                extra = tuple(round(float(v), 6) for v in s.widths_z)
            key = self.cache.key(layer.name, s.k, t_lo, t_hi, extra)
            hit = self.cache.get(key)
            if hit is not None:
                recon, stats = hit
            else:
                recon, stats = compress_layer(
                    self.master.original(layer),
                    self.master.row_scale(layer),
                    k=s.k,
                    t_lo=t_lo,
                    t_hi=t_hi,
                    quant_cfg=self.cfg.quant,
                    prune_cfg=self.cfg.prune,
                    name=layer.name,
                    alpha=s.alpha,
                    gamma=s.gamma,
                    u=s.u,
                    z=s.widths_z,
                    force_zero=s.force_zero,
                    reassign=s.reassign,
                    act_norm=self.wanda_norms.get(layer.name) if self.wanda_norms else None,
                )
                self.cache.put(key, (recon, stats))
            self.master.write(layer, recon)
            cost.layers.append(price_layer(
                stats, self.cfg.quant.codebook_bits,
                fmt=getattr(self.cfg.quant, "deployable_format", "dense"),
                csr_span_bits=getattr(self.cfg.quant, "csr_span_bits", 4)))

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.n_evals += 1
        return Candidate(cost=cost, settings=settings,
                         apply_seconds=time.perf_counter() - t0)

    def restore(self) -> None:
        self.master.restore()

    # -- convenience --------------------------------------------------------

    def cost_only(self, x: np.ndarray) -> ModelCost:
        """Price a genome without touching the model.

        Cheap enough to call on millions of genomes, so use it to pre-screen a
        population against an avg_bits budget before spending forward passes.
        """
        settings = self.genome.decode(x)
        cost = ModelCost(n_untouched_weights=self.n_untouched)
        for layer in self.targets:
            s = settings[layer.name]
            # Sparsity and the symbol histogram need the real weights; this
            # estimate assumes a flat histogram, which upper-bounds the
            # archival cost and is exact for the deployable cost.
            k = s.k
            n_groups = _n_groups(layer, self.cfg.quant)
            counts = torch.zeros(k, dtype=torch.float64)
            counts[:] = layer.n_weights / k
            stats = _FakeStats(layer.name, layer.n_weights, n_groups, k, k, counts)
            cost.layers.append(price_layer(
                stats, self.cfg.quant.codebook_bits,
                fmt=getattr(self.cfg.quant, "deployable_format", "dense"),
                csr_span_bits=getattr(self.cfg.quant, "csr_span_bits", 4)))
        return cost

    def summary(self) -> str:
        n_t = self.master.n_target_weights
        n_a = n_t + self.n_untouched
        return (
            f"model            : {self.cfg.model.name}\n"
            f"device           : {self.device}\n"
            f"target layers    : {len(self.targets)} "
            f"({n_t/1e6:.1f}M weights, {100*n_t/n_a:.1f}% of checkpoint)\n"
            f"untouched (fp16) : {self.n_untouched/1e6:.1f}M weights\n"
            + self.genome.describe()
        )


def _n_groups(layer, quant_cfg) -> int:
    gran = layer_granularity(layer.name, quant_cfg)
    if gran == "per_tensor":
        return 1
    if gran == "per_channel":
        return layer.out_features
    return layer.out_features * (layer.in_features // quant_cfg.group_size)


@dataclass
class _FakeStats:
    name: str
    n_weights: int
    n_groups: int
    k_nominal: int
    k_centroids: int
    symbol_counts: torch.Tensor
    k_used_mean: float = 0.0
    sparsity: float = 0.0
    mse: float = 0.0
