"""Calibration and evaluation corpora.

Conventions follow the GPTQ/AWQ line of work so numbers are comparable without
re-running anyone else's code:

  * WikiText-2 *raw* test split, concatenated and chopped into non-overlapping
    2048-token windows -- the primary reported perplexity.
  * C4 (en, first train shard) for calibration and as a second, less-overfit
    perplexity. 128 random 2048-token segments is the standard calibration set.

Tokenized corpora are cached to disk as tensors; tokenizing C4 is far slower
than any single fitness evaluation and must not happen inside the search loop.
"""

from __future__ import annotations

import hashlib
import os

import torch


def _cache_path(cache_dir: str, *parts: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    tag = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{tag}.pt")


def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "the `datasets` package is required for real corpora.\n"
            "  pip install datasets\n"
            "For a dependency-free smoke test use dataset='synthetic'."
        ) from e
    return __import__("datasets")


def _synthetic(tokenizer, n_tokens: int, seed: int) -> torch.Tensor:
    """Deterministic random token stream. Only for pipeline smoke tests --
    perplexity on it is meaningless."""
    g = torch.Generator().manual_seed(seed)
    vocab = min(getattr(tokenizer, "vocab_size", 32000) or 32000, 32000)
    return torch.randint(0, vocab, (1, n_tokens), generator=g)


def load_corpus(
    name: str,
    tokenizer,
    split: str = "test",
    n_tokens: int = 1 << 20,
    seed: int = 0,
    cache_dir: str = ".cache/evolmc",
) -> torch.Tensor:
    """Return a single [1, T] tensor of token ids."""
    model_tag = getattr(tokenizer, "name_or_path", "tok")
    path = _cache_path(cache_dir, name, split, model_tag, str(n_tokens), str(seed))
    if os.path.exists(path):
        return torch.load(path)

    if name == "synthetic":
        ids = _synthetic(tokenizer, n_tokens, seed)
    elif name == "wikitext2":
        ds = _require_datasets()
        raw = ds.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        ids = tokenizer("\n\n".join(raw["text"]), return_tensors="pt").input_ids
    elif name == "ptb":
        ds = _require_datasets()
        raw = ds.load_dataset("ptb_text_only", "penn_treebank", split=split)
        ids = tokenizer("\n\n".join(raw["sentence"]), return_tensors="pt").input_ids
    elif name == "c4":
        ds = _require_datasets()
        shard = (
            "en/c4-train.00000-of-01024.json.gz"
            if split == "train"
            else "en/c4-validation.00000-of-00008.json.gz"
        )
        raw = ds.load_dataset(
            "allenai/c4", data_files={split: shard}, split=split, streaming=True
        )
        chunks, total = [], 0
        for row in raw:
            piece = tokenizer(row["text"], return_tensors="pt").input_ids
            chunks.append(piece)
            total += piece.shape[1]
            if total >= n_tokens:
                break
        ids = torch.cat(chunks, dim=1)
    else:
        raise ValueError(f"unknown dataset: {name}")

    ids = ids[:, :n_tokens]
    torch.save(ids, path)
    return ids


def make_windows(ids: torch.Tensor, seqlen: int, n_seq: int) -> torch.Tensor:
    """Chop a token stream into [n, seqlen] non-overlapping windows."""
    usable = (ids.shape[1] // seqlen) * seqlen
    if usable < seqlen:
        raise ValueError(f"corpus too short: {ids.shape[1]} tokens < seqlen {seqlen}")
    windows = ids[0, :usable].view(-1, seqlen)
    return windows[: min(n_seq, windows.shape[0])]


def build_splits(cfg, tokenizer) -> dict[str, torch.Tensor]:
    """Proxy windows for the search loop, eval windows for the final front.

    The proxy set is drawn from the calibration corpus and the eval set from a
    held-out corpus, so the search cannot optimise the reported metric directly.
    """
    need_proxy = (cfg.n_proxy_seq + 8) * cfg.seqlen
    calib = load_corpus(
        cfg.calib_dataset, tokenizer, "train", max(need_proxy, 1 << 19),
        cfg.seed, cfg.cache_dir,
    )
    need_eval = (cfg.n_eval_seq + 8) * cfg.seqlen
    evald = load_corpus(
        cfg.eval_dataset, tokenizer, "test", max(need_eval, 1 << 19),
        cfg.seed, cfg.cache_dir,
    )
    return {
        "proxy": make_windows(calib, cfg.seqlen, cfg.n_proxy_seq),
        "eval": make_windows(evald, cfg.seqlen, cfg.n_eval_seq),
    }
