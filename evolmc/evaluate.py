"""Perplexity evaluation.

Teacher-forced, non-overlapping windows, `use_cache=False`. Two entry points:

  * `perplexity`     -- the metric to report.
  * `proxy_fitness`  -- the same computation on a handful of windows, used as
                        the EA's objective. Validate it once with
                        `rank_correlation` and report the Spearman rho in the
                        paper; that single number is what justifies searching
                        against a cheap signal.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, windows: torch.Tensor, device=None, batch_size: int = 1) -> float:
    """Perplexity over pre-chopped [n, seqlen] windows."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    total_nll, total_tok = 0.0, 0
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(device)
        logits = model(batch, use_cache=False).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll.item())
        total_tok += shift_labels.numel()
        del logits, shift_logits

    if was_training:
        model.train()
    if total_tok == 0:
        return float("inf")
    mean_nll = total_nll / total_tok
    # Guard against overflow on badly broken candidates; the EA only needs the
    # ordering to stay correct out here.
    return math.exp(mean_nll) if mean_nll < 60 else float("inf")


def proxy_fitness(model, windows: torch.Tensor, device=None) -> float:
    ppl = perplexity(model, windows, device=device)
    if not math.isfinite(ppl):
        return 1e6
    return ppl


def rank_correlation(proxy_scores, full_scores) -> float:
    """Spearman rho between the proxy and the full metric.

    Run this once on ~30 sampled genomes before trusting the search. Below
    about 0.9 the proxy is too small -- raise `data.n_proxy_seq`.
    """
    import numpy as np

    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        return r

    a, b = ranks(proxy_scores), ranks(full_scores)
    a, b = a - a.mean(), b - b.mean()
    denom = float((a @ a) ** 0.5 * (b @ b) ** 0.5)
    return float(a @ b / denom) if denom > 0 else 0.0
