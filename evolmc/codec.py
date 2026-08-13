"""Compression accounting.

This is the part reviewers attack, so every bit is charged explicitly:

  * index bits      -- ceil(log2 K) per weight (deployable), or the real
                       Huffman cost of the layer's symbol histogram (archival)
  * codebook bits   -- one codebook per group, k_centroids entries of fp16
  * table bits      -- canonical-Huffman code lengths, one table per layer
  * untouched bits  -- whatever was left out of the target set, still fp16

Two compression ratios are reported and they must never be mixed:

  deployable -- fixed-width indices, NO entropy coding; this is what a LUT
                dequant kernel reads, and the number to quote for any memory
                or latency claim. Reported as `cr_deploy`.
  archive    -- Huffman-coded indices; a smaller checkpoint that must be
                decoded before it can be used. Quote this only for
                storage/transmission claims. Reported as `cr_archive`.

`avg_bits` counts the whole checkpoint -- target tensors plus whatever stays
untouched -- the honest end-to-end figure, unlike a target-only average that
hides how much of the model was never touched.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import torch

FP16_BITS = 16

# Gap widths "auto" may choose between for CSR. The best one moves with
# sparsity, because filler entries appear once the mean gap 1/(1-s) exceeds
# 2^span. Measured at 8 index bits:
#
#   sparsity   0.90   0.95   0.98   0.99   0.995   0.999
#   best span     6      8      8     12      12      16
#
# 12 is included because 99% sparsity is reachable. 16 is not: it only wins
# above 99.9%, which on a 590k-weight layer leaves under 600 live weights and
# is not a regime any usable model reaches.
CSR_SPANS = (2, 3, 4, 6, 8, 12)
# Bits per layer to record which format and gap width was used.
FORMAT_TAG_BITS = 6


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
    # -- sparse deployment -------------------------------------------------
    # `dense` charges one index per weight POSITION, so a pruned weight costs
    # exactly as much as a live one and pruning buys nothing deployable. The
    # sparse formats below store only survivors and pay a position overhead
    # instead, which is what makes pruning a parameter reduction rather than
    # only an entropy-coding trick. See `price_layer`.
    fmt: str = "dense"
    n_alive: int = 0
    mask_bits: float = 0.0
    index_bits_sparse: float = 0.0
    codebook_bits_sparse: float = 0.0

    @property
    def total_deployable(self) -> float:
        if self.fmt == "dense":
            return self.index_bits_fixed + self.codebook_bits
        return self.mask_bits + self.index_bits_sparse + self.codebook_bits_sparse

    @property
    def total_dense(self) -> float:
        """The dense cost, always, so the two can be reported side by side."""
        return self.index_bits_fixed + self.codebook_bits

    @property
    def total_archival(self) -> float:
        return self.index_bits_huffman + self.codebook_bits + self.table_bits

    @property
    def avg_bits_deployable(self) -> float:
        return self.total_deployable / max(self.n_weights, 1)

    @property
    def avg_bits_archival(self) -> float:
        return self.total_archival / max(self.n_weights, 1)


def _expected_fillers(n: int, n_alive: int, span: int) -> float:
    """Filler entries a CSR gap field of `span` bits needs, in expectation.

    A gap wider than 2^span cannot be encoded, so one filler entry is emitted
    per further 2^span positions. The count is NOT (mean_gap - 1)/2^span: gaps
    between survivors are geometric, so most are far shorter than the mean and
    contribute nothing while a thin tail contributes several each. Spreading the
    mean linearly overcharges badly -- at 90% sparsity with an 8-bit span the
    true count is zero and the linear estimate claims 7,031 fillers per two
    million weights, which made wide spans look better than they are.

    For gaps ~ Geometric(a) on {1, 2, ...} with a = n_alive/n,

        E[floor(gap / M)] = sum_{k>=1} P(gap >= kM) = q^(M-1) / (1 - q^M)

    with q = 1 - a. Matches simulation to under 2% across 90-99% sparsity.
    """
    if n_alive <= 0 or n_alive >= n:
        return 0.0
    a = n_alive / n
    q = 1.0 - a
    m = 2 ** max(1, span)
    if q <= 0.0:
        return 0.0
    # q**m underflows to 0.0 for wide spans, which is the correct limit: no gap
    # ever exceeds the field, so no fillers are needed.
    denom = 1.0 - q ** m
    if denom <= 1e-300:
        return 0.0
    return (q ** (m - 1) / denom) * n_alive


def price_layer(stats, codebook_bits: int = FP16_BITS, fmt: str = "dense",
                csr_span_bits: int | None = 4) -> LayerCost:
    """Turn quantizer statistics into a bit budget.

    `fmt` selects how the DEPLOYABLE side stores the indices, and it is the
    difference between pruning being a real size reduction and being a no-op:

    dense
        One `ceil(log2 K)` index per weight POSITION, in row-major order, so a
        kernel finds weight n by arithmetic. Pruned weights occupy a slot like
        any other and are merely assigned the reserved zero codeword, so
        **sparsity does not change this number at all**. Correct for a dense
        LUT kernel; wrong as a claim about parameter count.

    bitmap
        One bit per position saying alive or dead, then one index per SURVIVOR
        only. The mask carries the zeros, so the codebook holds K-1 entries
        rather than K and survivors index into those. Cost is
        `n + ceil(log2(K-1)) * n_alive + (K-1) * b_cb` per group. Pays a flat
        1 bit/weight and wins as soon as sparsity exceeds roughly
        `1 / ceil(log2 K)`.

    csr
        Deep Compression's relative-index scheme: per survivor, an index plus a
        `csr_span_bits`-wide gap to the previous survivor. Gaps longer than the
        span need filler entries, which is charged here as the expected number
        of fillers at the layer's mean gap. Cheaper than bitmap at high
        sparsity, more expensive at low, and its overhead is data-dependent.

    The archival figures are unaffected by `fmt`: entropy coding already prices
    a run of zeros for almost nothing, so the deployable format is exactly
    where the choice bites.
    """
    n = stats.n_weights
    idx_bits = max(1, math.ceil(math.log2(stats.k_nominal)))
    hb, maxlen = huffman_bits(stats.symbol_counts)
    sb = shannon_bits(stats.symbol_counts)
    # Canonical Huffman needs one code length per symbol; 5 bits covers
    # lengths up to 31, which K <= 256 never exceeds in practice.
    len_field = max(1, math.ceil(math.log2(max(maxlen, 1) + 1)))
    table = stats.k_nominal * len_field + 64  # + small per-layer header

    # Survivors, from the measured sparsity rather than from a symbol count:
    # symbol 0 is the reserved zero codeword only when pruning is active, and
    # is an ordinary bin otherwise.
    n_alive = int(round(n * (1.0 - float(stats.sparsity))))
    n_alive = max(0, min(n, n_alive))

    mask_bits = index_sparse = codebook_sparse = 0.0
    chosen = fmt
    if fmt != "dense":
        # With an explicit position record the zero codeword is redundant, so
        # survivors share K-1 centroids.
        k_alive = max(1, stats.k_centroids - 1) if stats.sparsity > 0 else stats.k_centroids
        alive_idx_bits = max(1, math.ceil(math.log2(max(k_alive, 2))))
        codebook_sparse = stats.n_groups * k_alive * codebook_bits

        def bitmap_bits():
            return float(n), alive_idx_bits * n_alive

        def csr_bits(span):
            span = max(1, span)
            fillers = _expected_fillers(n, n_alive, span)
            return 0.0, (alive_idx_bits + span) * (n_alive + fillers)

        if fmt == "bitmap":
            mask_bits, index_sparse = bitmap_bits()
        elif fmt == "csr":
            if csr_span_bits is None:
                # Pick the gap width per layer. The optimum tracks sparsity --
                # 4 bits around 80%, 6 at 90-95%, 8 at 98% -- so any fixed
                # choice handicaps CSR at exactly the sparsities where it wins.
                # Costs the same tag as `auto`, and is what makes a CSR run a
                # fair test of the format rather than of one arbitrary width.
                spans = {s: csr_bits(s) for s in CSR_SPANS}
                span, best = min(spans.items(), key=lambda kv: sum(kv[1]))
                mask_bits, index_sparse = best
                index_sparse += FORMAT_TAG_BITS
                chosen = f"csr{span}"
            else:
                mask_bits, index_sparse = csr_bits(csr_span_bits)
        elif fmt == "auto":
            # Pick the cheapest representation for THIS layer, and charge for
            # recording the choice. Which format wins is strongly sparsity
            # dependent -- bitmap's flat 1 bit/position beats CSR below roughly
            # 85% sparsity and loses badly above it -- and the best gap width
            # moves too, so fixing either by hand costs real bits. The tag is
            # a few bits per layer against megabytes of indices.
            options = {("bitmap", 0): bitmap_bits()}
            for sp in CSR_SPANS:
                options[("csr", sp)] = csr_bits(sp)
            chosen, best = min(options.items(), key=lambda kv: sum(kv[1]))
            mask_bits, index_sparse = best
            index_sparse += FORMAT_TAG_BITS
            chosen = f"{chosen[0]}{chosen[1] or ''}"
        else:
            raise ValueError(f"unknown deployable format: {fmt!r} "
                             "(expected 'dense', 'bitmap', 'csr' or 'auto')")

    return LayerCost(
        name=stats.name,
        n_weights=n,
        n_groups=stats.n_groups,
        k_nominal=stats.k_nominal,
        k_centroids=stats.k_centroids,
        sparsity=stats.sparsity,
        k_used_mean=stats.k_used_mean,
        mse=stats.mse,
        index_bits_fixed=idx_bits * n,
        index_bits_huffman=hb,
        index_bits_entropy=sb,
        codebook_bits=stats.n_groups * stats.k_centroids * codebook_bits,
        table_bits=float(table),
        fmt=chosen,
        n_alive=n_alive,
        mask_bits=mask_bits,
        index_bits_sparse=index_sparse,
        codebook_bits_sparse=codebook_sparse,
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

    # -- parameter count ---------------------------------------------------
    # Pruning removes parameters, and that is a different claim from "the
    # checkpoint got smaller". Both are reported: a 90%-sparse model has 10x
    # fewer live weights but nothing like 10x less storage, because the
    # positions of the survivors have to be recorded somewhere.
    @property
    def n_alive_target(self) -> int:
        return sum(l.n_alive for l in self.layers)

    @property
    def n_alive_total(self) -> int:
        return self.n_alive_target + self.n_untouched_weights

    @property
    def param_reduction(self) -> float:
        """Live parameters removed, over the whole checkpoint."""
        return 1.0 - self.n_alive_total / max(self.n_total_weights, 1)

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
    def avg_bits(self) -> float:
        """Deployable bits per weight over the entire checkpoint."""
        return (self.target_bits_deployable + self.untouched_bits) / max(
            self.n_total_weights, 1
        )

    @property
    def avg_bits_archival(self) -> float:
        return (self.target_bits_archival + self.untouched_bits) / max(
            self.n_total_weights, 1
        )

    @property
    def n_lookups(self) -> float:
        """LUT lookups a dequant kernel performs over the target matrices.

        The second term of the latency model, and the reason latency is not
        simply a restatement of the deployable byte count.

        Under `dense` every weight POSITION carries an index, so the kernel
        looks up once per position no matter how much was pruned -- constant
        across candidates. Under `bitmap` or `csr` only SURVIVORS carry an
        index, so this falls with sparsity while the byte count falls with both
        sparsity and K. Two candidates can therefore match on bytes and differ
        on lookups, which is exactly the freedom a latency objective needs.
        """
        return float(sum(l.n_weights if l.fmt == "dense" else l.n_alive
                         for l in self.layers))

    @property
    def cr_deploy(self) -> float:
        return self.original_bits / max(
            self.target_bits_deployable + self.untouched_bits, 1e-9
        )

    @property
    def cr_archive(self) -> float:
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

    @property
    def target_bits_dense(self) -> float:
        """Deployable cost under the dense format, whatever `fmt` is in use.

        Reported alongside so the sparse saving is visible as a difference
        rather than having to be inferred from two separate runs.
        """
        return sum(l.total_dense for l in self.layers)

    @property
    def cr_dense(self) -> float:
        return self.original_bits / max(
            self.target_bits_dense + self.untouched_bits, 1e-9)

    def summary(self) -> dict:
        return {
            "avg_bits": self.avg_bits,
            "avg_bits_archival": self.avg_bits_archival,
            "cr_deploy": self.cr_deploy,
            "cr_archive": self.cr_archive,
            "cr_dense": self.cr_dense,
            "sparsity": self.sparsity,
            "param_reduction": self.param_reduction,
            "n_alive_total": float(self.n_alive_total),
            "n_lookups": self.n_lookups,
            "mean_k_used": self.mean_k_used,
            # Memory in BYTES. The MB columns below are the same numbers
            # scaled; bytes are what a deployment budget is actually quoted in
            # and they avoid a MiB-vs-MB ambiguity in a results table.
            "bytes_original": self.original_bits / 8,
            "bytes_deployable": (self.target_bits_deployable
                                 + self.untouched_bits) / 8,
            "bytes_archive": (self.target_bits_archival
                              + self.untouched_bits) / 8,
            "size_mb_original": self.original_bits / 8 / 2**20,
            "size_mb_deployable": (self.target_bits_deployable + self.untouched_bits)
            / 8
            / 2**20,
            "size_mb_dense": (self.target_bits_dense + self.untouched_bits)
            / 8
            / 2**20,
            "size_mb_archival": (self.target_bits_archival + self.untouched_bits)
            / 8
            / 2**20,
        }

    def report(self) -> str:
        s = self.summary()
        return (
            f"avg bits  {s['avg_bits']:.3f} (arch {s['avg_bits_archival']:.3f})\n"
            f"CR   deploy {s['cr_deploy']:.2f}x (no Huffman)"
            f" | archive {s['cr_archive']:.2f}x (Huffman)\n"
            f"size {s['size_mb_original']:.0f} MB -> {s['size_mb_deployable']:.0f} MB"
            f" (archival {s['size_mb_archival']:.0f} MB)\n"
            f"sparsity {s['sparsity']:.3f} | mean codewords used {s['mean_k_used']:.1f}"
        )
