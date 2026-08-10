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
   are not two objectives. `cr_deploy` is exactly `16 / bpw_model`, because
   both normalize `target_bits_deployable + untouched_bits` (see codec.py), so
   pairing them leaves dominance unchanged and the front is identical to the
   2-objective run. `check_redundancy` is what catches this at runtime; the
   registry's `bit_total` field is what it reasons about.

The independent bit totals in this codec are exactly two: DEPLOYABLE
(fixed-width indices, NO entropy coding -- what a LUT dequant kernel reads,
reported as `cr_deploy`) and ARCHIVE (Huffman-coded -- what a checkpoint
costs to store, reported as `cr_archive`). Any pair of size objectives
drawn from the same total is redundant; a pair drawn from different totals is
not.
"""

from __future__ import annotations

import warnings
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
# So 16/bpw_target does NOT equal cr_deploy, and a figure that does not say so
# invites a reader to try the arithmetic and conclude something is broken.
REGISTRY: dict[str, Objective] = {o.name: o for o in (
    Objective("ppl_proxy", MINIMIZE, "proxy perplexity", "", "", log=True),

    # -- deployable: fixed-width indices + codebooks, NO entropy coding -----
    Objective("bpw_target", MINIMIZE, "bits per weight",
              "target", "deployable"),
    Objective("bpw_model", MINIMIZE, "bits per weight",
              "whole", "deployable"),
    Objective("cr_deploy", MAXIMIZE, "CR, no Huffman",
              "whole", "deployable"),
    Objective("size_mb_deployable", MINIMIZE, "size (MB)",
              "whole", "deployable"),

    # -- archive: Huffman-coded indices + codebooks + code-length tables ----
    Objective("bpw_target_archival", MINIMIZE, "bits per weight, Huffman",
              "target", "archival"),
    Objective("bpw_model_archival", MINIMIZE, "bits per weight, Huffman",
              "whole", "archival"),
    Objective("cr_archive", MAXIMIZE, "CR, Huffman",
              "whole", "archival"),
    Objective("size_mb_archival", MINIMIZE, "size (MB)",
              "whole", "archival"),

    # -- neither ------------------------------------------------------------
    Objective("sparsity", MAXIMIZE, "sparsity", "target", ""),
    # Modelled decode latency from latency.LatencyProxy: a per-layer roofline,
    # max(bytes/beta, flops/phi), plus launch overhead, plus the fixed cost of
    # every untouched fp16 component. bit_total is EMPTY on purpose: the memory
    # roof does come from the deployable total, but the compute roof and the
    # launch term do not, so the pair is not redundant by construction. Where it
    # DOES degenerate is config-dependent, which check_redundancy checks below.
    Objective("latency_proxy", MINIMIZE, "decode latency (ms/token, modelled)",
              "", ""),
)}

# Old spellings, still accepted everywhere a name is read: `search.objectives`
# and `report_metrics` in a config, and the `objectives` list stored in a
# finished run's data/plot_box.json. 27 runs predate the rename and must keep
# replotting, so this is load-bearing rather than politeness.
#
# The same map is applied to CSV rows read back from those runs
# (`canonicalize_row`), so a tool comparing a stored column against a freshly
# computed one is comparing the same quantity under one name.
LEGACY_NAMES = {
    "cr_deployable": "cr_deploy",
    "cr_archival": "cr_archive",
}


def canonical(name: str) -> str:
    """Current name for `name`, which may be a retired spelling."""
    return LEGACY_NAMES.get(name, name)


def canonicalize_row(row: dict) -> dict:
    """Rewrite retired keys in a summary or CSV row, leaving order otherwise.

    A row that already carries the current name wins: if both spellings are
    present the legacy one is dropped rather than silently overwriting the
    value a newer tool wrote.
    """
    out = {}
    for k, v in row.items():
        new = canonical(k)
        if new in out and new != k:
            continue
        out[new] = v
    return out


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
        retired = [n for n in names if n in LEGACY_NAMES]
        if retired:
            warnings.warn(
                "objective(s) " + ", ".join(
                    f"{n} -> {LEGACY_NAMES[n]}" for n in retired)
                + " were renamed; the old names still work",
                DeprecationWarning, stacklevel=2)
            names = tuple(canonical(n) for n in names)
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
        # Canonicalized so a summary produced by an older tool, or read back
        # from a finished run's CSV, still resolves against the current names.
        source = canonicalize_row({"ppl_proxy": ppl, **summary})
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


def check_redundancy(objset: ObjectiveSet, cfg=None) -> list[str]:
    """Warnings for objective pairs that cannot disagree.

    Two size objectives built from the same bit total are monotone transforms
    of one another, so dominance is exactly what it would be with only one of
    them. `cfg` is optional and only needed for the latency checks, which turn
    on `benchmark.dequant_ns_per_lookup` and `quant.deployable_format` rather
    than on the objective names. The search still runs, it just costs a full budget to reproduce a
    front you already have. Returned rather than raised: an intentionally
    redundant run is a legitimate thing to want as a control.
    """
    warnings = []
    if "latency_proxy" in objset.names:
        # latency_proxy can only disagree with a deployable size objective where
        # the COMPUTE roof binds on some layers and not others. If the memory
        # roof binds everywhere then T = (sum_l B_l)/beta + constant, the size
        # objective is that same sum normalized, and the two are exactly affine.
        #
        # An earlier version of this check keyed on deployable_format and warned
        # only for `dense`. That was a false negative: measured on 120 random
        # genomes, bitmap gave 0 discordant pairs out of 7,140 at the shipped
        # compute_eff of 0.25, because the bitmap mask floors bytes at 1 bit per
        # position and the memory roof still binds everywhere.
        #
        # Whether it binds is numeric, not nameable, so this warns for the
        # pairing and defers to latency.LatencyProxy.roof_diagnostic, which
        # run_search logs at both ends of the K range before generation 1.
        deployable = [o.name for o in objset if o.bit_total == "deployable"]
        if deployable:
            warnings.append(
                f"latency_proxy is paired with {', '.join(deployable)}. These "
                f"conflict ONLY where the compute roof binds on some layers and "
                f"not others; if the memory roof binds everywhere they are "
                f"exactly affine and the front is 2-D with three axis labels. "
                f"Check the 'roof at K=' lines logged below -- 0 compute-bound "
                f"layers at both ends means no conflict is possible. The lever "
                f"is latency.compute_eff.")

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
