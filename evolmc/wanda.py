"""Activation norms for Wanda-style pruning (prune.mode == "wanda").

Wanda scores a weight by |w_ij| * ||X_j||, where X_j is the j-th input
feature's activation values across the calibration set -- a small weight on a
heavily-used input can matter more than a large weight on one that is barely
activated. That ||X_j|| term is what this module computes.

It is calibrated ONCE, before the search starts, from the same proxy windows
already loaded for ppl_proxy: it depends only on the frozen fp16 weights and
the calibration data, never on a candidate genome, so recomputing it per
evaluation would be pure waste. Compare evolmc/latency.py's coefficient fit,
which is the same "measure once, freeze, reuse" shape for a different proxy.

Accumulated as sum-of-squares in float64 across every calibration token,
then square-rooted once at the end -- L2 norms do not sum across batches,
sums-of-squares do.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = ()

_HOOKABLE = (nn.Linear,) + ((Conv1D,) if Conv1D else ())


@dataclass
class ActivationNorms:
    """Per-layer ||X_j|| vectors, one entry per target layer name."""

    norms: dict[str, torch.Tensor]
    n_tokens: int

    def get(self, name: str) -> torch.Tensor | None:
        return self.norms.get(name)


@torch.no_grad()
def calibrate(compressor, windows: torch.Tensor, batch_size: int = 1) -> ActivationNorms:
    """Run `windows` through the model once and record input-feature norms.

    Restores the master weights first, so this always measures the true fp16
    model regardless of what the last `.apply()` call left installed.
    """
    compressor.restore()
    model = compressor.model
    device = compressor.device
    was_training = model.training
    model.eval()

    hookable = [t for t in compressor.targets if isinstance(t.module, _HOOKABLE)]
    sums = {t.name: torch.zeros(t.in_features, dtype=torch.float64, device=device)
            for t in hookable}
    n_tokens = 0

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float64)
            sums[name] += (x * x).sum(dim=0)
        return hook

    handles = [t.module.register_forward_hook(make_hook(t.name)) for t in hookable]
    try:
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start : start + batch_size].to(device)
            model(batch, use_cache=False)
            n_tokens += batch.shape[0] * batch.shape[1]
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    norms = {name: s.clamp_min(1e-12).sqrt().to(torch.float32) for name, s in sums.items()}
    return ActivationNorms(norms=norms, n_tokens=n_tokens)
