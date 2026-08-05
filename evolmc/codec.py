"""Compression accounting.

This is the part reviewers attack, so every bit is charged explicitly:

  * index bits      -- ceil(log2 K) per weight (deployable), or the real
                       Huffman cost of the layer's symbol histogram (archival)
  * codebook bits   -- one codebook per group, k_centroids entries of fp16
  * table bits      -- canonical-Huffman code lengths, one table per layer
  * untouched bits  -- embeddings, LM head, norms and biases, still fp16

Two compression ratios are reported and they must never be mixed:

  deployable -- fixed-width indices; this is what a LUT dequant kernel reads,
                and the number to quote for any memory or latency claim.
  archival   -- entropy-coded indices; a smaller checkpoint that must be
                decoded before it can be used. Quote this only for
                storage/transmission claims.

`bpw_target` counts only the compressed projection matrices, which is what the
GPTQ/AWQ/SqueezeLLM tables report. `bpw_model` counts the whole checkpoint,
which is the honest end-to-end figure. Report both.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import torch

FP16_BITS = 16


def huffman_bits(counts) -> tuple[float, int]:
    """Expected bits for the symbol stream, plus the max code length.

    Returns the true Huffman cost (>= Shannon entropy), not the entropy bound.
    """
    if torch.is_tensor(counts):
        counts = counts.tolist()
    active = [(float(c), i) for i, c in enumerate(counts) if c > 0]
    total = sum(c for c, _ in active)
    if total <= 0:
        return 0.0, 0
    if len(active) == 1:
        # A single symbol still needs one bit per weight in any prefix code,
        # though a real codec would signal it in the header for ~0 bits.
        return total, 1

    heap = [(c, 0, i) for c, i in active]  # (weight, depth-tiebreak, id)
    heapq.heapify(heap)
    lengths = {i: 0 for _, i in active}
    # Track lengths by merging symbol sets; O(K log K) with K <= 256.
    groups = {i: [i] for _, i in active}
    nxt = len(counts)
    while len(heap) > 1:
        c1, d1, i1 = heapq.heappop(heap)
        c2, d2, i2 = heapq.heappop(heap)
        merged = groups.pop(i1) + groups.pop(i2)
        for s in merged:
            lengths[s] += 1
        groups[nxt] = merged
        heapq.heappush(heap, (c1 + c2, max(d1, d2) + 1, nxt))
        nxt += 1

    bits = sum(counts[s] * lengths[s] for s in lengths)
    return float(bits), max(lengths.values())


def shannon_bits(counts) -> float:
    if torch.is_tensor(counts):
        counts = counts.tolist()
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h * total


@dataclass
class LayerCost:
    name: str
    n_weights: int
    n_groups: int
    k_nominal: int
    k_centroids: int
    sparsity: float
    k_used_mean: float
    mse: float
    index_bits_fixed: float
    index_bits_huffman: float
    index_bits_entropy: float
    codebook_bits: float
    table_bits: float

    @property
    def total_deployable(self) -> float:
        return self.index_bits_fixed + self.codebook_bits

    @property
    def total_archival(self) -> float:
        return self.index_bits_huffman + self.codebook_bits + self.table_bits

    @property
    def bpw_deployable(self) -> float:
        return self.total_deployable / max(self.n_weights, 1)

    @property
    def bpw_archival(self) -> float:
        return self.total_archival / max(self.n_weights, 1)


def price_layer(stats, codebook_bits: int = FP16_BITS) -> LayerCost:
    """Turn quantizer statistics into a bit budget."""
    idx_bits = max(1, math.ceil(math.log2(stats.k_nominal)))
    hb, maxlen = huffman_bits(stats.symbol_counts)
    sb = shannon_bits(stats.symbol_counts)
    # Canonical Huffman needs one code length per symbol; 5 bits covers
    # lengths up to 31, which K <= 256 never exceeds in practice.
    len_field = max(1, math.ceil(math.log2(max(maxlen, 1) + 1)))
    table = stats.k_nominal * len_field + 64  # + small per-layer header
    return LayerCost(
        name=stats.name,
        n_weights=stats.n_weights,
        n_groups=stats.n_groups,
        k_nominal=stats.k_nominal,
        k_centroids=stats.k_centroids,
        sparsity=stats.sparsity,
        k_used_mean=stats.k_used_mean,
        mse=stats.mse,
        index_bits_fixed=idx_bits * stats.n_weights,
        index_bits_huffman=hb,
        index_bits_entropy=sb,
        codebook_bits=stats.n_groups * stats.k_centroids * codebook_bits,
        table_bits=float(table),
    )


@dataclass
class ModelCost:
    layers: list[LayerCost] = field(default_factory=list)
    n_untouched_weights: int = 0

    # -- sizes ------------------------------------------------------------
    @property
    def n_target_weights(self) -> int:
        return sum(l.n_weights for l in self.layers)

    @property
    def n_total_weights(self) -> int:
        return self.n_target_weights + self.n_untouched_weights

    @property
    def untouched_bits(self) -> float:
        return self.n_untouched_weights * FP16_BITS

    @property
    def original_bits(self) -> float:
        return self.n_total_weights * FP16_BITS

    @property
    def target_bits_deployable(self) -> float:
        return sum(l.total_deployable for l in self.layers)

    @property
    def target_bits_archival(self) -> float:
        return sum(l.total_archival for l in self.layers)

    # -- headline numbers --------------------------------------------------
    @property
    def bpw_target(self) -> float:
        """Deployable bits per weight over the compressed matrices only."""
        return self.target_bits_deployable / max(self.n_target_weights, 1)

    @property
    def bpw_target_archival(self) -> float:
        return self.target_bits_archival / max(self.n_target_weights, 1)

    @property
    def bpw_model(self) -> float:
        """Deployable bits per weight over the entire checkpoint."""
        return (self.target_bits_deployable + self.untouched_bits) / max(
            self.n_total_weights, 1
        )

    @property
    def bpw_model_archival(self) -> float:
        return (self.target_bits_archival + self.untouched_bits) / max(
            self.n_total_weights, 1
        )

    @property
    def cr_deployable(self) -> float:
        return self.original_bits / max(
            self.target_bits_deployable + self.untouched_bits, 1e-9
        )

    @property
    def cr_archival(self) -> float:
        return self.original_bits / max(
            self.target_bits_archival + self.untouched_bits, 1e-9
        )

    @property
    def sparsity(self) -> float:
        n = max(self.n_target_weights, 1)
        return sum(l.sparsity * l.n_weights for l in self.layers) / n

    @property
    def mean_k_used(self) -> float:
        n = max(self.n_target_weights, 1)
        return sum(l.k_used_mean * l.n_weights for l in self.layers) / n

    def summary(self) -> dict:
        return {
            "bpw_target": self.bpw_target,
            "bpw_target_archival": self.bpw_target_archival,
            "bpw_model": self.bpw_model,
            "bpw_model_archival": self.bpw_model_archival,
            "cr_deployable": self.cr_deployable,
            "cr_archival": self.cr_archival,
            "sparsity": self.sparsity,
            "mean_k_used": self.mean_k_used,
            "size_mb_original": self.original_bits / 8 / 2**20,
            "size_mb_deployable": (self.target_bits_deployable + self.untouched_bits)
            / 8
            / 2**20,
            "size_mb_archival": (self.target_bits_archival + self.untouched_bits)
            / 8
            / 2**20,
        }

    def report(self) -> str:
        s = self.summary()
        return (
            f"bpw  target {s['bpw_target']:.3f} (arch {s['bpw_target_archival']:.3f})"
            f" | model {s['bpw_model']:.3f} (arch {s['bpw_model_archival']:.3f})\n"
            f"CR   deployable {s['cr_deployable']:.2f}x"
            f" | archival {s['cr_archival']:.2f}x\n"
            f"size {s['size_mb_original']:.0f} MB -> {s['size_mb_deployable']:.0f} MB"
            f" (archival {s['size_mb_archival']:.0f} MB)\n"
            f"sparsity {s['sparsity']:.3f} | mean codewords used {s['mean_k_used']:.1f}"
        )
