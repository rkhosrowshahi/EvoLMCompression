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
    """A weight the search is allowed to compress.

    2-D matrices (Linear, Conv1D, Embedding) keep their native layout.
    1-D vectors (LayerNorm scale/bias, projection biases) are viewed as a
    single-row matrix `[1, n]` so the rest of the pipeline can treat them as
    one per-tensor codebook.
    """

    name: str
    module: nn.Module
    n_weights: int
    out_features: int
    in_features: int
    block: int  # -1 if the layer sits outside the decoder stack
    proj_type: str  # "q_proj", "c_attn", "down_proj", ...
    transposed: bool  # True for Conv1D: stored weight is [in, out]
    param_name: str = "weight"  # "weight" or "bias"
    is_vector: bool = False  # True for 1-D norms and biases

    def param(self) -> nn.Parameter:
        return getattr(self.module, self.param_name)

    def rows_view(self, w: torch.Tensor) -> torch.Tensor:
        """Return the weight as [out_features, in_features].

        Per-channel codebooks are per *output* channel, so we always work in
        this orientation regardless of how the module stores its weight.
        """
        if self.is_vector:
            return w.reshape(1, -1)
        return w.t() if self.transposed else w

    def store_view(self, w: torch.Tensor) -> torch.Tensor:
        """Inverse of `rows_view` -- back to the module's storage layout."""
        if self.is_vector:
            return w.reshape(-1)
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

    1-D biases keep the same qualifier with a `.bias` suffix, so
    `attn.c_proj.bias` and `mlp.c_proj.bias` stay distinct under type grouping.
    """
    bias = name.endswith(".bias")
    base = name[: -len(".bias")] if bias else name
    parts = base.split(".")
    if len(parts) >= 2 and not parts[-2].isdigit():
        key = f"{parts[-2]}.{parts[-1]}"
    else:
        key = parts[-1]
    return f"{key}.bias" if bias else key


def discover_targets(model: nn.Module, exclude_patterns,
                     include_1d: bool = False) -> list[TargetLayer]:
    """Find every parameter eligible for compression.

    Linear, Conv1D and Embedding 2-D weights always pass the type filter;
    `exclude_patterns` is what keeps embeddings and the LM head out. Norms and
    biases stay fp16 unless `include_1d` is set -- they are 1-D, so no exclude
    pattern can bring them in by itself. Vectors are stored as `[1, n]` so they
    share the 2-D codebook path; they are quantized but never pruned.

    TIED WEIGHTS ARE CLAIMED ONCE. GPT-2's lm_head.weight IS transformer.wte
    .weight, the same Parameter. Both modules pass the type filter, and
    returning both would give MasterWeights two entries for one tensor -- the
    second write would silently overwrite the first and the bit accounting
    would double-count. First module wins; the rest are skipped.
    """
    targets: list[TargetLayer] = []
    linear_types = (nn.Linear, nn.Embedding) + ((Conv1D,) if Conv1D else ())
    seen_weights: set[int] = set()

    for name, mod in model.named_modules():
        if not isinstance(mod, linear_types):
            continue
        if any(p in name for p in exclude_patterns):
            continue
        w = mod.weight
        if w.ndim != 2:
            continue
        if id(w) in seen_weights:
            continue
        seen_weights.add(id(w))
        # Conv1D stores [in, out]; Linear and Embedding both store [out, in].
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

    if include_1d:
        for pname, p in model.named_parameters():
            if p.ndim != 1 or id(p) in seen_weights:
                continue
            if any(pat in pname for pat in exclude_patterns):
                continue
            parent_name, leaf = pname.rsplit(".", 1)
            try:
                mod = model.get_submodule(parent_name)
            except AttributeError:
                continue
            seen_weights.add(id(p))
            tname = parent_name if leaf == "weight" else pname
            m = _BLOCK_RE.search(tname)
            targets.append(
                TargetLayer(
                    name=tname,
                    module=mod,
                    n_weights=p.numel(),
                    out_features=1,
                    in_features=int(p.numel()),
                    block=int(m.group(1)) if m else -1,
                    proj_type=_proj_type(tname),
                    transposed=False,
                    param_name=leaf,
                    is_vector=True,
                )
            )

    if not targets:
        raise RuntimeError("no quantizable layers found -- check exclude_patterns")
    return targets


def count_untouched_weights(model: nn.Module, targets: list[TargetLayer]) -> int:
    """Parameters that stay in fp16 (whatever was left out of the target set).

    Counted once even when the LM head is tied to the input embedding.
    """
    target_ids = {id(t.param()) for t in targets}
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
            t.name: t.param().detach().to(self.device, copy=True) for t in targets
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
        return layer.rows_view(w).to(layer.param().device, torch.float32)

    def row_scale(self, layer: TargetLayer) -> torch.Tensor:
        return self._row_scale[layer.name].to(layer.param().device)

    def write(self, layer: TargetLayer, rows: torch.Tensor) -> None:
        """Install a compressed weight (given as [out, in]) into the live model."""
        dst = layer.param()
        dst.data.copy_(layer.store_view(rows).to(dst.dtype))

    def restore(self, layer: TargetLayer | None = None) -> None:
        for t in ([layer] if layer is not None else self.targets):
            dst = t.param()
            dst.data.copy_(self._master[t.name].to(dst.device))

    @property
    def n_target_weights(self) -> int:
        return sum(t.n_weights for t in self.targets)
