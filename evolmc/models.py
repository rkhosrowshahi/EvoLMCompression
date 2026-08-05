"""Model loading, target-layer discovery, and the master-weight store.

The single most important performance rule in this project: the model is loaded
once and never reloaded. Each candidate solution overwrites the live weights
in place from an immutable master copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # GPT-2 and friends use Conv1D, whose weight is [in, out] not [out, in].
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = ()

_BLOCK_RE = re.compile(r"\.(?:layers|h|blocks|block)\.(\d+)\.")


@dataclass(frozen=True)
class TargetLayer:
    """A weight matrix the search is allowed to compress."""

    name: str
    module: nn.Module
    n_weights: int
    out_features: int
    in_features: int
    block: int  # -1 if the layer sits outside the decoder stack
    proj_type: str  # "q_proj", "c_attn", "down_proj", ...
    transposed: bool  # True for Conv1D: stored weight is [in, out]

    def rows_view(self, w: torch.Tensor) -> torch.Tensor:
        """Return the weight as [out_features, in_features].

        Per-channel codebooks are per *output* channel, so we always work in
        this orientation regardless of how the module stores its weight.
        """
        return w.t() if self.transposed else w

    def store_view(self, w: torch.Tensor) -> torch.Tensor:
        """Inverse of `rows_view` -- back to the module's storage layout."""
        return w.t() if self.transposed else w


def load_model(cfg) -> tuple[nn.Module, AutoTokenizer]:
    dtype = getattr(torch, cfg.dtype)
    device = _resolve_device(cfg.device)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(cfg.name, use_fast=True,
                                        trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _proj_type(name: str) -> str:
    """Qualify a layer's role with its parent submodule.

    The leaf name alone is ambiguous in the GPT-2 family: `attn.c_proj` is
    [768, 768] while `mlp.c_proj` is [768, 3072]. Keying on the leaf collapses
    them into one variable, so `k_grouping: type` would hand GPT-2 three
    decision variables instead of four and force the attention output and the
    MLP down-projection to share a codebook size despite 4x different row
    widths. Llama-style names are already unique; qualifying them is harmless
    and more readable ("self_attn.q_proj" rather than "q_proj").
    """
    parts = name.split(".")
    if len(parts) >= 2 and not parts[-2].isdigit():
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]


def discover_targets(model: nn.Module, exclude_patterns) -> list[TargetLayer]:
    """Find every 2-D projection weight eligible for compression.

    Embeddings, the LM head and all norms/biases are excluded -- they stay in
    fp16 and are accounted for separately (see codec.ModelCost.untouched_bits).
    """
    targets: list[TargetLayer] = []
    linear_types = (nn.Linear,) + ((Conv1D,) if Conv1D else ())

    for name, mod in model.named_modules():
        if not isinstance(mod, linear_types):
            continue
        if any(p in name for p in exclude_patterns):
            continue
        w = mod.weight
        if w.ndim != 2:
            continue
        transposed = bool(Conv1D) and isinstance(mod, Conv1D)
        out_f, in_f = (w.shape[1], w.shape[0]) if transposed else (w.shape[0], w.shape[1])
        m = _BLOCK_RE.search(name)
        targets.append(
            TargetLayer(
                name=name,
                module=mod,
                n_weights=w.numel(),
                out_features=out_f,
                in_features=in_f,
                block=int(m.group(1)) if m else -1,
                proj_type=_proj_type(name),
                transposed=transposed,
            )
        )
    if not targets:
        raise RuntimeError("no quantizable layers found -- check exclude_patterns")
    return targets


def count_untouched_weights(model: nn.Module, targets: list[TargetLayer]) -> int:
    """Parameters that stay in fp16 (embeddings, head, norms, biases).

    Counted once even when the LM head is tied to the input embedding.
    """
    target_ids = {id(t.module.weight) for t in targets}
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in target_ids or id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


class MasterWeights:
    """Immutable fp16 copy of every target weight, plus in-place restore.

    Held on `master_device`. Keeping it on the GPU is fastest (H100); keeping it
    on the CPU roughly halves peak VRAM at the cost of one PCIe copy per layer
    per candidate (the RTX 3060 path for models above ~3B).
    """

    def __init__(self, targets: list[TargetLayer], master_device: str = "cuda"):
        self.device = _resolve_device(master_device)
        self.targets = targets
        self._master: dict[str, torch.Tensor] = {
            t.name: t.module.weight.detach().to(self.device, copy=True) for t in targets
        }
        # Pruning thresholds are expressed in units of a per-row scale; that
        # scale never changes, so compute it once here.
        self._row_scale: dict[str, torch.Tensor] = {}
        for t in targets:
            rows = t.rows_view(self._master[t.name]).float()
            self._row_scale[t.name] = rows.std(dim=1, keepdim=True).clamp_min(1e-12)

    def original(self, layer: TargetLayer) -> torch.Tensor:
        """Master weight in [out, in] orientation, fp32, on the compute device."""
        w = self._master[layer.name]
        return layer.rows_view(w).to(layer.module.weight.device, torch.float32)

    def row_scale(self, layer: TargetLayer) -> torch.Tensor:
        return self._row_scale[layer.name].to(layer.module.weight.device)

    def write(self, layer: TargetLayer, rows: torch.Tensor) -> None:
        """Install a compressed weight (given as [out, in]) into the live model."""
        dst = layer.module.weight
        dst.data.copy_(layer.store_view(rows).to(dst.dtype))

    def restore(self, layer: TargetLayer | None = None) -> None:
        for t in ([layer] if layer is not None else self.targets):
            t.module.weight.data.copy_(self._master[t.name].to(t.module.weight.device))

    @property
    def n_target_weights(self) -> int:
        return sum(t.n_weights for t in self.targets)
