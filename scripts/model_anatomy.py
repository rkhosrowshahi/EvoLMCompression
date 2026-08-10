#!/usr/bin/env python3
"""Where a language model's parameters actually live, block by block.

The compression framework only ever touches 2-D projection weights inside the
decoder stack. Everything else -- embeddings, the LM head, norms, biases -- is
accounted for but never compressed, and on GPT-2 that "everything else" is
31.7% of the checkpoint. This script is the arithmetic behind that claim, for
any model, without downloading it.

    python scripts/model_anatomy.py                 # the comparison table
    python scripts/model_anatomy.py --block gpt2    # one block, itemized
    python scripts/model_anatomy.py --verify        # check the formulas

-- WHY A FORMULA RATHER THAN A LOAD -----------------------------------------

Llama-3.1-405B is 810 GB in fp16. The parameter count is fully determined by
eight numbers in its config, so it is computed here and CHECKED against the
models small enough to actually load (`--verify` reproduces GPT-2's
124,439,808 exactly, and Pythia-410M's). A formula that reproduces the models
you can weigh is trustworthy on the ones you cannot.

-- THE TWO FAMILIES ---------------------------------------------------------

gpt2      LayerNorm with bias, learned positional table, fused QKV, GELU MLP
          of 4d, biases on every projection.
llama     RMSNorm without bias, rotary (no positional table), separate Q/K/V
          with grouped-query attention, SwiGLU MLP of three matrices, no
          biases anywhere.

Grouped-query attention is why K and V are not d x d: n_kv_heads < n_heads
shrinks them by n_kv/n_heads, which is 1/4 on Llama-3-8B and 1/8 on Llama-3-70B.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    name: str
    family: str          # "gpt2" | "llama"
    n_layer: int
    d: int               # hidden size
    n_head: int
    n_kv: int            # = n_head unless grouped-query attention
    ffn: int             # intermediate size
    vocab: int
    n_pos: int = 0       # learned positional table; 0 when rotary
    tied: bool = False   # lm_head shares the embedding tensor

    @property
    def head_dim(self) -> int:
        return self.d // self.n_head


# n_kv < n_head marks grouped-query attention. Llama-3.2 1B/3B tie their head;
# every other Llama here does not, which doubles the embedding mass.
MODELS = [
    Arch("GPT-2 124M",     "gpt2",  12,   768, 12, 12,  3072,  50257, 1024, True),
    Arch("GPT-2 355M",     "gpt2",  24,  1024, 16, 16,  4096,  50257, 1024, True),
    Arch("GPT-2 774M",     "gpt2",  36,  1280, 20, 20,  5120,  50257, 1024, True),
    Arch("GPT-2 1.5B",     "gpt2",  48,  1600, 25, 25,  6400,  50257, 1024, True),
    Arch("Pythia-410M",    "gpt2",  24,  1024, 16, 16,  4096,  50304,    0, False),
    Arch("Llama-3.2-1B",   "llama", 16,  2048, 32,  8,  8192, 128256,    0, True),
    Arch("Llama-3.2-3B",   "llama", 28,  3072, 24,  8,  8192, 128256,    0, True),
    Arch("Llama-2-7B",     "llama", 32,  4096, 32, 32, 11008,  32000,    0, False),
    Arch("Llama-3-8B",     "llama", 32,  4096, 32,  8, 14336, 128256,    0, False),
    Arch("Llama-2-13B",    "llama", 40,  5120, 40, 40, 13824,  32000,    0, False),
    Arch("Llama-2-70B",    "llama", 80,  8192, 64,  8, 28672,  32000,    0, False),
    Arch("Llama-3-70B",    "llama", 80,  8192, 64,  8, 28672, 128256,    0, False),
    Arch("Llama-3.1-405B", "llama", 126,16384,128,  8, 53248, 128256,    0, False),
]


def block_parts(a: Arch) -> list[tuple[str, str, int, bool]]:
    """(name, shape, count, is_compressible) for one decoder block.

    `is_compressible` marks the 2-D projection weights the search can reach.
    Norms and biases are 1-D and are excluded for a reason that survives the
    arithmetic: they are ~0.1% of the checkpoint on GPT-2.
    """
    d, ffn, kv = a.d, a.ffn, a.n_kv * a.head_dim
    if a.family == "gpt2":
        return [
            ("ln_1.weight",       f"({d},)",        d,          False),
            ("ln_1.bias",         f"({d},)",        d,          False),
            ("attn.c_attn",       f"({d}, {3*d})",  d * 3 * d,  True),
            ("attn.c_attn.bias",  f"({3*d},)",      3 * d,      False),
            ("attn.c_proj",       f"({d}, {d})",    d * d,      True),
            ("attn.c_proj.bias",  f"({d},)",        d,          False),
            ("ln_2.weight",       f"({d},)",        d,          False),
            ("ln_2.bias",         f"({d},)",        d,          False),
            ("mlp.c_fc",          f"({d}, {ffn})",  d * ffn,    True),
            ("mlp.c_fc.bias",     f"({ffn},)",      ffn,        False),
            ("mlp.c_proj",        f"({ffn}, {d})",  ffn * d,    True),
            ("mlp.c_proj.bias",   f"({d},)",        d,          False),
        ]
    return [
        ("input_layernorm",       f"({d},)",        d,          False),
        ("self_attn.q_proj",      f"({d}, {d})",    d * d,      True),
        ("self_attn.k_proj",      f"({kv}, {d})",   kv * d,     True),
        ("self_attn.v_proj",      f"({kv}, {d})",   kv * d,     True),
        ("self_attn.o_proj",      f"({d}, {d})",    d * d,      True),
        ("post_attn_layernorm",   f"({d},)",        d,          False),
        ("mlp.gate_proj",         f"({ffn}, {d})",  ffn * d,    True),
        ("mlp.up_proj",           f"({ffn}, {d})",  ffn * d,    True),
        ("mlp.down_proj",         f"({d}, {ffn})",  d * ffn,    True),
    ]


def anatomy(a: Arch) -> dict:
    parts = block_parts(a)
    per_block = sum(c for _, _, c, _ in parts)
    per_block_proj = sum(c for _, _, c, comp in parts if comp)

    embed = a.vocab * a.d
    head = 0 if a.tied else a.vocab * a.d
    pos = a.n_pos * a.d
    final_norm = a.d * (2 if a.family == "gpt2" else 1)

    blocks = a.n_layer * per_block
    total = blocks + embed + head + pos + final_norm
    return {
        "arch": a,
        "per_block": per_block,
        "per_block_proj": per_block_proj,
        "blocks": blocks,
        "proj": a.n_layer * per_block_proj,
        "embed_head": embed + head,
        "pos": pos,
        "norms_biases": blocks - a.n_layer * per_block_proj + final_norm,
        "total": total,
    }


def fmt(n: int) -> str:
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.2f}{suf}"
    return str(n)


def show_block(name: str) -> None:
    # Forgiving lookup: "gpt2", "GPT-2", "gpt-2 124m" and "llama3.2-1b" should
    # all land, so punctuation and case are stripped from both sides and a
    # substring match is enough. Exact matches win over prefixes so that
    # "gpt2" does not resolve to whichever size happens to be listed first.
    def norm(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum())

    q = norm(name)
    cands = [m for m in MODELS if norm(m.name).startswith(q)] \
        or [m for m in MODELS if q in norm(m.name)]
    if not cands:
        raise SystemExit(f"unknown model: {name}\n  " +
                         "\n  ".join(m.name for m in MODELS))
    a = cands[0]
    if len(cands) > 1:
        print(f"note: {name!r} matches {len(cands)}; showing {a.name}"
              f"  (also: {', '.join(m.name for m in cands[1:])})")
    info = anatomy(a)
    print(f"\n{a.name}  --  one decoder block, {a.n_layer} of them\n")
    print(f"  {'parameter':<24}{'shape':>18}{'count':>14}  compressible")
    print("  " + "-" * 72)
    for n, shape, c, comp in block_parts(a):
        print(f"  {n:<24}{shape:>18}{c:>14,}  {'yes' if comp else '.':>6}")
    print("  " + "-" * 72)
    print(f"  {'block total':<24}{'':>18}{info['per_block']:>14,}")
    print(f"  {'of which compressible':<24}{'':>18}{info['per_block_proj']:>14,}"
          f"  ({100*info['per_block_proj']/info['per_block']:.2f}%)")
    print(f"\n  x{a.n_layer} blocks{'':<15}{'':>18}{info['blocks']:>14,}")
    print(f"  embeddings + head{'':<7}{'':>18}{info['embed_head']:>14,}"
          + ("   (tied, counted once)" if a.tied else "   (untied)"))
    if info["pos"]:
        print(f"  positional table{'':<8}{'':>18}{info['pos']:>14,}")
    print(f"  norms + biases{'':<10}{'':>18}{info['norms_biases']:>14,}")
    print(f"  {'TOTAL':<24}{'':>18}{info['total']:>14,}")


def show_table() -> None:
    rows = [anatomy(a) for a in MODELS]
    print(f"\n{'model':<16}{'L':>4}{'d':>7}{'kv/h':>6}{'ffn':>7}{'vocab':>8}"
          f"{'tied':>6}{'block':>9}{'blocks':>9}{'emb+head':>10}"
          f"{'total':>9}{'emb %':>8}{'CR cap':>8}")
    print("-" * 107)
    for r in rows:
        a = r["arch"]
        untouched = r["embed_head"] + r["pos"] + r["norms_biases"]
        pct = 100 * r["embed_head"] / r["total"]
        cap = r["total"] / max(untouched, 1)
        gqa = f"{a.n_kv}/{a.n_head}"
        print(f"{a.name:<16}{a.n_layer:>4}{a.d:>7}{gqa:>6}{a.ffn:>7}{a.vocab:>8}"
              f"{('yes' if a.tied else 'no'):>6}{fmt(r['per_block']):>9}"
              f"{fmt(r['blocks']):>9}{fmt(r['embed_head']):>10}"
              f"{fmt(r['total']):>9}{pct:>7.1f}%{cap:>7.1f}x")
    print("-" * 107)
    print("block     parameters in ONE decoder block        emb %   embeddings + LM head,")
    print("blocks    all decoder blocks together                    share of the checkpoint")
    print("CR cap    ceiling on cr_deploy if every compressible matrix went to 0 bits,")
    print("          i.e. 1 / (untouched share). This is the wall the wte experiment removed.")


def verify() -> None:
    """Check the formulas against models small enough to actually weigh."""
    import torch
    from transformers import AutoModelForCausalLM

    for name, hf in (("GPT-2 124M", "gpt2"), ("Pythia-410M", "EleutherAI/pythia-410m")):
        a = next(m for m in MODELS if m.name == name)
        try:
            model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32)
        except Exception as e:                                  # offline, no cache
            print(f"  {name:<16} SKIPPED ({type(e).__name__})")
            continue
        seen, real = set(), 0
        for p in model.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                real += p.numel()
        pred = anatomy(a)["total"]
        ok = "OK" if pred == real else "MISMATCH"
        print(f"  {name:<16} predicted {pred:>13,}   actual {real:>13,}   {ok}")
        del model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", metavar="MODEL",
                    help="itemize one decoder block (e.g. gpt2, llama-3-8b)")
    ap.add_argument("--verify", action="store_true",
                    help="check the formulas against loadable models")
    args = ap.parse_args()
    if args.verify:
        print("verifying the parameter formulas against real checkpoints")
        verify()
    elif args.block:
        show_block(args.block)
    else:
        show_table()


if __name__ == "__main__":
    main()
