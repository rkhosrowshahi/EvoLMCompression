"""Which quantities the search optimizes, and in which direction.

`search.objectives` in the config is a list of names from the registry below.
Everything else -- how many objectives pymoo is told about, what the plot axes
are, how hypervolume is normalized, what lands in front.csv -- is derived from
that list, so adding or reordering an objective is a config change rather than
a code change.

Two rules that are easy to get wrong:

1. pymoo minimizes. A maximized objective (every `cr_*`) is stored negated in
   `F`, and only ever converted back for display. `ObjectiveSet.to_min` and
   `.to_real` are the single pair of functions that know this; nothing else
   should be flipping signs.

2. Two objectives that are monotone transforms of the same underlying scalar
   are not two objectives. `cr_deployable` is exactly `16 / bpw_model`, because
   both normalize `target_bits_deployable + untouched_bits` (see codec.py), so
   pairing them leaves dominance unchanged and the front is identical to the
   2-objective run. `check_redundancy` is what catches this at runtime; the
   registry's `bit_total` field is what it reasons about.

The independent bit totals in this codec are exactly two: DEPLOYABLE
(fixed-width indices, what a LUT dequant kernel reads) and ARCHIVAL
(entropy-coded, what a checkpoint costs to store). Any pair of size objectives
drawn from the same total is redundant; a pair drawn from different totals is
not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MINIMIZE, MAXIMIZE = 1, -1


@dataclass(frozen=True)
class Objective:
    name: str
    sense: int          # MINIMIZE or MAXIMIZE
    label: str          # axis label, without the scope
    scope: str          # which weights it covers; "" when not a size measure
    bit_total: str      # "deployable" | "archival" | "" -- see check_redundancy
    log: bool = False   # draw and normalize on a log axis

    @property
    def axis_label(self) -> str:
        return f"{self.label} ({self.scope})" if self.scope else self.label

    @property
    def arrow(self) -> str:
        r"""Direction of IMPROVEMENT, the usual convention in results tables.

        Down for a minimized objective, up for a maximized one, so it reads as
        "lower is better" / "higher is better" rather than restating the
        optimizer's sign convention. Mathtext, so it renders identically with
        and without `usetex`.
        """
        return r"$\downarrow$" if self.sense == MINIMIZE else r"$\uparrow$"

    @property
    def plot_label(self) -> str:
        r"""Axis label carrying scope and direction in ONE parenthesis.

        `bits per weight (target, $\downarrow$)` rather than stacking
        two bracketed groups, which reads as an afterthought on a figure.
        """
        if self.scope:
            return f"{self.label} ({self.scope}, {self.arrow})"
        return f"{self.label} ({self.arrow})"

    def better(self, a: float, b: float) -> bool:
        return (a < b) if self.sense == MINIMIZE else (a > b)


# Scope matters for the caption. `bpw_target` covers only the compressed
# projection matrices, which is what GPTQ/AWQ/SqueezeLLM tables quote; the
# cr_* ratios cover the whole checkpoint including untouched fp16 embeddings.
# So 16/bpw_target does NOT equal cr_deployable, and a figure that does not say
# so invites a reader to try the arithmetic and conclude something is broken.
REGISTRY: dict[str, Objective] = {o.name: o for o in (
    Objective("ppl_proxy", MINIMIZE, "proxy perplexity", "", "", log=True),

    # -- deployable: fixed-width indices + codebooks ------------------------
    Objective("bpw_target", MINIMIZE, "bits per weight",
              "target", "deployable"),
    Objective("bpw_model", MINIMIZE, "bits per weight",
              "whole", "deployable"),
    Objective("cr_deployable", MAXIMIZE, "CR",
              "whole", "deployable"),
    Objective("size_mb_deployable", MINIMIZE, "size (MB)",
              "whole", "deployable"),

    # -- archival: Huffman-coded indices + codebooks + code-length tables ---
    Objective("bpw_target_archival", MINIMIZE, "bits per weight, Huffman",
              "target", "archival"),
    Objective("bpw_model_archival", MINIMIZE, "bits per weight, Huffman",
              "whole", "archival"),
    Objective("cr_archival", MAXIMIZE, "ACR",
              "whole", "archival"),
    Objective("size_mb_archival", MINIMIZE, "size (MB)",
              "whole", "archival"),

    # -- neither ------------------------------------------------------------
    Objective("sparsity", MAXIMIZE, "sparsity", "target", ""),
)}

DEFAULT = ("ppl_proxy", "bpw_target")


class ObjectiveSet:
    """An ordered list of objectives, plus the conversions pymoo needs.

    Order is meaningful for the figures: objective 0 is the y axis, objective 1
    is the x axis, objective 2 (if present) is the color. That convention is
    why the configs list perplexity first.
    """

    def __init__(self, names=DEFAULT):
        names = tuple(names or DEFAULT)
        if len(names) < 2:
            raise ValueError(
                f"search.objectives needs at least 2 entries, got {list(names)}"
            )
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise ValueError(
                f"unknown objective(s) {unknown}. Valid names: "
                f"{', '.join(sorted(REGISTRY))}"
            )
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"search.objectives repeats {dupes}")
        self.names = names
        self.specs = tuple(REGISTRY[n] for n in names)
        self.senses = np.array([s.sense for s in self.specs], dtype=float)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.specs)

    def __getitem__(self, i):
        return self.specs[i]

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def n_obj(self) -> int:
        return len(self.names)

    # -- extracting values -------------------------------------------------

    def values(self, ppl: float, summary: dict) -> list[float]:
        """Real-space objective values for one candidate, in order.

        `summary` is ModelCost.summary(); `ppl` is the proxy fitness, which is
        not part of the cost accounting and so is passed separately.
        """
        source = {"ppl_proxy": ppl, **summary}
        missing = [n for n in self.names if n not in source]
        if missing:
            raise KeyError(
                f"objective(s) {missing} are not in ModelCost.summary(); "
                f"available: {', '.join(sorted(source))}"
            )
        return [float(source[n]) for n in self.names]

    # -- pymoo lives in minimization space ---------------------------------

    def to_min(self, values):
        """Real values -> what goes in `out['F']`."""
        return np.asarray(values, dtype=float) * self.senses

    def to_real(self, F):
        """`F` (or a stored front) -> real values, for display and reporting."""
        F = np.asarray(F, dtype=float)
        return F * self.senses if F.ndim == 1 else F * self.senses[None, :]

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        rows = []
        for i, s in enumerate(self.specs):
            arrow = "min" if s.sense == MINIMIZE else "MAX"
            rows.append(f"  f{i + 1}  {s.name:<22} {arrow}   {s.axis_label}")
        return "objectives\n" + "\n".join(rows)


def from_box(box: dict, size_objective: str = "bpw_target"):
    """(ObjectiveSet, bounds) from a run's `data/plot_box.json`.

    Handles both layouts. Runs made before `search.objectives` existed stored
    only `xlim`/`ylim` and were always (perplexity, bpw), so they are read back
    as exactly that -- which keeps replot and compare_runs working on every
    finished run rather than only on new ones.
    """
    names = box.get("objectives")
    if names:
        objset = ObjectiveSet(names)
        bounds = [tuple(b) for b in box["bounds"]]
        return objset, bounds
    objset = ObjectiveSet(("ppl_proxy", size_objective))
    ylim, xlim = box["ylim"], box["xlim"]
    return objset, [(ylim[0], ylim[1]), (xlim[0], xlim[1])]


def check_redundancy(objset: ObjectiveSet) -> list[str]:
    """Warnings for objective pairs that cannot disagree.

    Two size objectives built from the same bit total are monotone transforms
    of one another, so dominance is exactly what it would be with only one of
    them. The search still runs, it just costs a full budget to reproduce a
    front you already have. Returned rather than raised: an intentionally
    redundant run is a legitimate thing to want as a control.
    """
    warnings = []
    for i in range(len(objset)):
        for j in range(i + 1, len(objset)):
            a, b = objset[i], objset[j]
            if a.bit_total and a.bit_total == b.bit_total:
                warnings.append(
                    f"{a.name} and {b.name} are both derived from the "
                    f"{a.bit_total} bit total, so each is a monotone transform "
                    f"of the other and dominance is unchanged by having both. "
                    f"The front will match a run using only one of them. Pair a "
                    f"deployable measure with an archival one to get a "
                    f"genuinely 3-D front."
                )
    return warnings


def spearman_matrix(F) -> np.ndarray:
    """Pairwise Spearman correlation between objective columns.

    Reported once after the baseline sweep so a redundant axis is visible in
    the log before the search has spent its budget, rather than afterwards.
    Note that a high correlation is not on its own a problem: dominance cares
    only about ordering, and a few percent of discordant pairs is enough to
    change the front. It is |rho| = 1.000 that means an axis is doing nothing.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    F = F[np.isfinite(F).all(axis=1)]
    m = F.shape[1]
    out = np.eye(m)
    if len(F) < 3:
        return out
    ranks = np.apply_along_axis(_rankdata, 0, F)
    for i in range(m):
        for j in range(i + 1, m):
            a, b = ranks[:, i], ranks[:, j]
            sa, sb = a.std(), b.std()
            r = 0.0 if sa < 1e-12 or sb < 1e-12 else float(
                ((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
            out[i, j] = out[j, i] = r
    return out


def _rankdata(x):
    """Average ranks, ties included. Avoids a hard scipy dependency."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # Average the ranks within each run of equal values.
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks
