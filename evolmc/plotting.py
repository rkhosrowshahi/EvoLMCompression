"""Pareto-front figures, one frame per generation.

The axis limits are computed **once**, before the first frame, and frozen for
the whole run. That is the point of these plots: every generation is drawn in
the same box, so flipping through `figures/pareto/` -- or running the frames
through ffmpeg -- shows the front actually moving rather than the axes
rescaling under it. Points outside the box are clipped and counted in a corner
annotation, never silently dropped.

The x range is derived analytically from `quant.k_choices` (the reachable bpw
interval is known before a single evaluation runs), so it never depends on what
the search happened to sample.

Colors are the first two slots of the reference categorical palette, which
clear the all-pairs CVD and normal-vision gates for scatter forms (worst pair
ΔE 24.7 protan / 33.6 normal). Everything that is context rather than identity
-- the population cloud, the fp16 line, the grid -- wears neutral ink instead
of a series hue.
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# -- palette ---------------------------------------------------------------
# Light is the committed look: these figures are destined for a PDF in a paper,
# where a single deliberate mode beats an automatic flip. `style="dark"` is
# provided for slides.
THEMES = {
    "paper": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "front": "#2a78d6",   # categorical slot 1
        "baseline": "#eb6834",  # categorical slot 2
        "cloud": "#a9a8a2",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "front": "#3987e5",
        "baseline": "#d95926",
        "cloud": "#5c5c58",
    },
}

matplotlib.rcParams.update({
    "pdf.fonttype": 42,   # embed TrueType so the PDF is editable/searchable
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# -- paper presets ---------------------------------------------------------
# Figure text mismatches a paper for two independent reasons, and the second
# is usually the larger effect:
#
#   1. Typeface. matplotlib defaults to DejaVu Sans; papers are set in a serif
#      (Times for IEEE/ICML/NeurIPS, Libertine for ACM). Note that *math* has
#      its own font: setting font.family alone leaves $K$ in DejaVu next to
#      Times body text, which is why mathtext.fontset is set here too.
#
#   2. Scaling. A figure saved 7 in wide and included at \\columnwidth (3.5 in)
#      is scaled by 0.5, so its 8 pt labels arrive on the page at 4 pt. The fix
#      is not a bigger font -- it is to save the figure at exactly the width it
#      will occupy, so LaTeX scales it by 1.0. Venue mode does this, and also
#      disables the tight bbox, which would otherwise crop the canvas to an
#      unpredictable width and reintroduce the scale factor.
#
# Widths are the standard text/column measures for each style file, in inches.
VENUES = {
    "ieee": {       # IEEEtran two-column -- CEC, TEVC, most IEEE conferences
        "column": 3.5, "page": 7.16, "font_pt": 8.0,
        "serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                  "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext": "stix",
        "preamble": r"\usepackage{newtxtext,newtxmath}",
    },
    "acm": {        # acmart sigconf -- GECCO, and ACM conferences generally
        "column": 3.33, "page": 7.0, "font_pt": 8.0,
        "serif": ["Libertinus Serif", "Linux Libertine O", "Linux Libertine",
                  "Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext": "stix",
        "preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
    },
    "neurips": {    # single column
        "column": 5.5, "page": 5.5, "font_pt": 9.0,
        "serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                  "STIXGeneral", "DejaVu Serif"],
        "mathtext": "stix",
        "preamble": r"\usepackage{times}",
    },
    "icml": {       # two-column
        "column": 3.25, "page": 6.75, "font_pt": 8.0,
        "serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                  "STIXGeneral", "DejaVu Serif"],
        "mathtext": "stix",
        "preamble": r"\usepackage{times}",
    },
    "lncs": {       # Springer LNCS, single column (122 mm text width)
        "column": 4.8, "page": 4.8, "font_pt": 8.0,
        "serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                  "STIXGeneral", "DejaVu Serif"],
        "mathtext": "stix",
        "preamble": r"\usepackage{times}",
    },
}


def has_latex() -> bool:
    import shutil

    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def resolve_figsize(cfg):
    """Figure size in inches. In venue mode this is the *final* printed size."""
    venue = VENUES.get(cfg.plot.venue)
    if venue is None:
        return tuple(cfg.plot.figsize)
    width = cfg.plot.width
    if isinstance(width, str):
        width = venue.get(width)
        if width is None:
            raise ValueError("plot.width must be 'column', 'page', or a number "
                             f"of inches (got {cfg.plot.width!r})")
    return (float(width), float(width) * cfg.plot.aspect)


def apply_style(cfg, log=None):
    """Install the venue's typography into matplotlib's global rcParams."""
    venue = VENUES.get(cfg.plot.venue)
    if venue is None:
        return
    pt = cfg.plot.font_pt or venue["font_pt"]
    rc = {
        "font.family": "serif",
        "font.serif": venue["serif"],
        "mathtext.fontset": venue["mathtext"],
        "font.size": pt,
        "axes.titlesize": pt + 1,
        "axes.labelsize": pt,
        "xtick.labelsize": pt - 1,
        "ytick.labelsize": pt - 1,
        "legend.fontsize": pt - 1,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": True,
        "legend.borderpad": 0.4,
    }
    if cfg.plot.usetex:
        if has_latex():
            rc["text.usetex"] = True
            rc["text.latex.preamble"] = venue["preamble"]
        elif log:
            log("  plot.usetex requested but latex/dvipng not on PATH; "
                "falling back to matplotlib fonts")
    matplotlib.rcParams.update(rc)

    if log:
        from matplotlib import font_manager as fm

        have = {f.name for f in fm.fontManager.ttflist}
        picked = next((f for f in venue["serif"] if f in have), None)
        log(f"  typography: {cfg.plot.venue} @ {pt}pt, "
            f"font {picked or 'DejaVu Serif (no venue font installed!)'}"
            + (", usetex" if rc.get("text.usetex") else ""))


def latex_snippet(cfg, path, caption="Pareto front.", label="fig:pareto"):
    """The \\includegraphics line that preserves the figure's true size.

    Width is stated explicitly and equals the saved canvas width, so the scale
    factor is exactly 1.0 and the figure's 8 pt labels land on the page as
    8 pt. `\\linewidth` is used rather than a fixed length so the same snippet
    works in one- and two-column layouts.
    """
    env = "figure" if cfg.plot.width == "column" else "figure*"
    return (
        f"\\begin{{{env}}}[t]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{{path}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"\\end{{{env}}}"
    )


def derive_limits(compressor, cfg, baselines=None, fp16_ppl=None):
    """Fixed (xlim, ylim) for every frame in the run.

    x comes from the reachable bpw interval, computed in closed form from
    `k_choices`. y comes from the reference points if we have them, so the
    box is anchored to fp16 and to the uniform-K line the search must beat.
    """
    genome = compressor.genome
    bpws = []
    for k in (min(genome.k_choices), max(genome.k_choices)):
        cost = compressor.cost_only(genome.encode_uniform(k))
        bpws.append(cost.bpw_target if cfg.search.size_objective == "bpw_target"
                    else cost.bpw_model)
    lo, hi = min(bpws), max(bpws)
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    xlim = (cfg.plot.xlim_min if cfg.plot.xlim_min is not None else lo - pad,
            cfg.plot.xlim_max if cfg.plot.xlim_max is not None else hi + pad)

    ppls = [p for p in (list(baselines or []) + [fp16_ppl]) if p and np.isfinite(p)]
    if not ppls:
        return xlim, (1.0, 1e4)

    # Anchor the box to fp16 rather than to the worst reference. fp16 is the
    # achievable floor, so "headroom x fp16" reads directly as how many times
    # worse than lossless a candidate is. Letting the max set the ceiling hands
    # the whole axis to one blown-up low-K baseline -- those go off-scale and
    # are counted in the frame instead.
    anchor = fp16_ppl if (fp16_ppl and np.isfinite(fp16_ppl)) else float(np.median(ppls))
    floor = cfg.plot.ylim_min
    lo = (float(floor) if floor is not None
          else min(min(ppls), anchor) * cfg.plot.ylim_min_ratio)
    ceiling = cfg.plot.ylim_max
    if ceiling is not None:
        hi = float(ceiling)
    elif cfg.plot.ylim_max_ratio is None:
        hi = max(ppls)            # no cap: every reference point fits
    else:
        hi = min(max(ppls), anchor * cfg.plot.ylim_max_ratio)
    hi = max(hi, anchor * 1.5, lo * 1.2)
    padded = pad_ylim((lo, hi), cfg.plot.yscale, cfg.plot.ylim_pad)
    # Explicit bounds mean exactly that, so padding does not open them.
    if floor is not None:
        padded = (float(floor), padded[1])
    if ceiling is not None:
        padded = (padded[0], float(ceiling))
    return xlim, _check_ylim(padded, cfg.plot.yscale)


def pad_ylim(ylim, yscale, frac):
    """Open both ends of the y box by `frac` of its span.

    In log space the span is measured in decades, so the padding is
    multiplicative and looks even at both ends of a wide axis -- an additive
    pad would be invisible at the top and enormous at the bottom.
    """
    lo, hi = float(ylim[0]), float(ylim[1])
    if frac <= 0 or hi <= lo:
        return lo, hi
    if yscale == "log":
        d = (math.log10(hi) - math.log10(lo)) * frac
        return max(10 ** (math.log10(lo) - d), 1.0), 10 ** (math.log10(hi) + d)
    span = (hi - lo) * frac
    return lo - span, hi + span


def _check_ylim(ylim, yscale):
    """Reject a perplexity floor that cannot be drawn or cannot be reached.

    Perplexity is exp(cross-entropy) and cross-entropy is non-negative, so
    PPL >= 1 always; PPL == 1 is a model that assigns probability 1 to every
    correct token. A floor of 0 is therefore not merely optimiztic, and on a
    log axis it is at -infinity and cannot be rendered at all.
    """
    lo, hi = float(ylim[0]), float(ylim[1])
    if yscale == "log" and lo <= 0:
        raise ValueError(
            f"plot.ylim lower bound must be > 0 on a log axis (got {lo}). "
            "Perplexity is bounded below by 1.0, not 0 -- use ylim: [1.0, ...] "
            "for the theoretical floor, lower plot.ylim_min_ratio to open up "
            "room under fp16, or switch to plot.yscale: linear."
        )
    if hi <= lo:
        raise ValueError(f"plot.ylim must be increasing (got {lo} -> {hi})")
    return lo, hi


class ParetoPlotter:
    """Per-generation frames, drawn from REAL-space objective values.

    Callers convert pymoo's minimization vectors with `ObjectiveSet.to_real`
    first, so a maximized objective is drawn and labeled in the units it is
    reported in rather than as a negative number.

    With three or more objectives the frame stays 2-D and projects: objective 0
    on y, objective 1 on x, objective 2 as the point color, on a frozen scale
    so frames stay comparable. The front is then drawn as a scatter rather than
    a connected curve, because a 3-objective front is a surface and joining its
    members in bpw order would draw a line through points that do not lie on
    one. Objectives past the third are optimized but not drawn.
    """

    def __init__(self, run, cfg, xlim, ylim, fp16_ppl=None, baselines=None,
                 objset=None, bounds=None):
        self.run = run
        self.cfg = cfg.plot
        self.xlim = xlim
        self.ylim = ylim
        self.fp16_ppl = fp16_ppl
        self.objset = objset
        self.cspec = objset[2] if (objset is not None and len(objset) > 2) else None
        # Frozen color scale, or the frames flicker as the population moves.
        self.clim = None
        if self.cspec is not None and bounds is not None:
            ideal, nadir = bounds[2]
            self.clim = (min(ideal, nadir), max(ideal, nadir))
        # baselines: list of (bpw, ppl, K) for the uniform-K reference line
        self.baselines = sorted(baselines or [], key=lambda r: r[0])
        self.theme = THEMES[self.cfg.style]
        apply_style(cfg, log=getattr(run, "log", None))
        self.figsize = resolve_figsize(cfg)
        # In venue mode the canvas width *is* the printed width; a tight bbox
        # would crop it to something else and reintroduce a LaTeX scale factor.
        self.exact_canvas = cfg.plot.venue != "none"
        # Read back from rcParams so a venue's font_pt actually governs every
        # label; hardcoded sizes here would silently override it.
        self.base_pt = float(matplotlib.rcParams["font.size"])
        # One marker size for the whole figure: the front draws it as a
        # diameter, the population scatter as an area (see frame()).
        self.marker_pt = float(cfg.plot.marker_pt)
        # The K= tags label a reference curve, not the result; they should be
        # readable on inspection without competing with the axis labels.
        self.annot_pt = (cfg.plot.annotation_pt if cfg.plot.annotation_pt
                         else max(self.base_pt - 3.0, 3.5))
        # Which measure this is (bpw_target vs bpw_model) is set by
        # search.size_objective and recorded in the run's config. With an
        # explicit objective list the scope is named on the axis instead: f2
        # and f3 can cover different weight sets, so a reader who assumes one
        # scope will try to reconcile 16/f2 against f3 and conclude the
        # accounting is broken.
        if objset is not None:
            self.ylabel_base = objset[0].plot_label
            self.xlabel = objset[1].plot_label
        else:
            self.ylabel_base = "proxy perplexity"
            self.xlabel = "bits per weight"

    # -- one generation ----------------------------------------------------

    def frame(self, gen, F_pop, F_front, n_evals, hv=None, stem=None,
              minimal=None):
        """One Pareto figure.

        `minimal` strips the running-search furniture -- title, evaluation
        counter, population cloud, history -- leaving the front against its
        baseline. That is what a paper figure wants: the generation number
        belongs in the caption, and the sampled cloud is noise once the search
        has converged. Defaults to on for standalone figures in venue mode.
        """
        if minimal is None:
            minimal = self.exact_canvas and stem is not None
        F_pop = np.atleast_2d(np.asarray(F_pop, dtype=float))
        F_front = np.atleast_2d(np.asarray(F_front, dtype=float))

        t, pt = self.theme, self.base_pt
        fig, ax = plt.subplots(figsize=self.figsize, facecolor=t["surface"])
        ax.set_facecolor(t["surface"])

        drawn = []
        if not minimal:
            # scatter sizes are areas in pt^2 while plot markersize is a
            # diameter in pt, hence the square. The -1 keeps the cloud one
            # point smaller than the front markers, so the front stays the
            # figure's subject and the population reads as context.
            self._scatter(ax, F_pop, color=t["cloud"],
                          size=(self.marker_pt-1) ** 2, z=2,
                          label=f"population (gen {gen})")
            drawn.append(F_pop)

        if self.fp16_ppl and np.isfinite(self.fp16_ppl):
            ax.axhline(self.fp16_ppl, color=t["muted"], lw=1.0, ls=(0, (5, 4)),
                       zorder=3)
            ax.annotate(f"fp16 ({self.fp16_ppl:.2f})",
                        xy=(self.xlim[0], self.fp16_ppl), xytext=(4, 3),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=pt - 1, color=t["muted"])

        if self.baselines:
            b = np.array([[r[0], r[1]] for r in self.baselines], dtype=float)
            ax.plot(b[:, 0], b[:, 1], ls=(0, (6, 3)), lw=1.4,
                    color=t["baseline"], marker="s", ms=self.marker_pt,
                    mfc=t["surface"], mew=1.4, zorder=4,
                    label=self.cfg.baseline_label)
            self._label_baselines(ax, pt)

        mappable = None
        if len(F_front):
            order = np.argsort(F_front[:, 1])
            f = F_front[order]
            if self.cspec is None:
                ax.plot(f[:, 1], f[:, 0], lw=1.4, color=t["front"], marker="o",
                        ms=self.marker_pt, mfc=t["front"], mec=t["surface"],
                        mew=0.8, zorder=5, label="Pareto front (NSGA-II)")
            else:
                # No connecting line: with a third objective the front is a
                # surface, and joining its members in x order draws a curve
                # through points that are not neighbors on it.
                inside = self._mask_inside(f[:, 1], f[:, 0])
                mappable = ax.scatter(
                    f[inside, 1], f[inside, 0], c=f[inside, 2],
                    cmap="viridis", vmin=self.clim[0], vmax=self.clim[1],
                    s=(self.marker_pt + 1) ** 2, lw=0.4,
                    edgecolors=t["surface"], zorder=5,
                    label="Pareto front (NSGA-II)")
            drawn.append(f)

        # Count *distinct* candidates outside the box. The front is a subset of
        # the population, so summing the two exclusion counts double-counts
        # every off-scale point that also happens to be non-dominated.
        off = 0
        if drawn:
            pts = np.unique(np.vstack(drawn), axis=0)
            off = int((~self._mask_inside(pts[:, 1], pts[:, 0])).sum())

        self._finish(ax, gen, n_evals, hv, off, minimal)
        if mappable is not None:
            cb = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label(self.cspec.plot_label, fontsize=pt - 1,
                         color=t["ink_2"])
            cb.ax.tick_params(labelsize=pt - 2, colors=t["muted"], length=2,
                              width=0.5)
            cb.outline.set_edgecolor(t["axis"])
            cb.outline.set_linewidth(0.6)
        # A per-generation frame goes into the video, so it needs a fixed
        # canvas; an explicit stem means a standalone figure, which does not.
        return self._save(fig, stem or self.run.frame(gen),
                          tight=stem is not None and not self.exact_canvas)

    def _label_baselines(self, ax, pt):
        """Annotate the uniform-K markers without hitting the axis or the edge.

        Labels sit *above* their marker: below collides with the x tick labels
        once the figure is narrowed to a journal column. Labels at the extreme
        ends are aligned inward so they do not run off the canvas.
        """
        t = self.theme
        span = self.xlim[1] - self.xlim[0]
        # Skip a tag that would overprint the previous one. Points bunch up
        # wherever the curve flattens -- on GPT-2 every K from 512 up sits at
        # the same perplexity -- and overlapping labels are worse than absent
        # ones, since the curve itself already shows the points are there.
        min_gap = 0.075
        last_frac = -1.0
        for bpw, ppl, k in self.baselines:
            if not self._inside(bpw, ppl):
                continue
            frac = (bpw - self.xlim[0]) / max(span, 1e-9)
            if frac - last_frac < min_gap:
                continue
            last_frac = frac
            ha, dx = "center", 0
            if frac > 0.9:
                ha, dx = "right", 4
            elif frac < 0.1:
                ha, dx = "left", -4
            ax.annotate(f"$K$={k}", xy=(bpw, ppl), xytext=(dx, 6),
                        textcoords="offset points", ha=ha, va="bottom",
                        fontsize=self.annot_pt, color=t["ink_2"], zorder=6)

    # -- summary figures ---------------------------------------------------

    def convergence(self, gens, hv, evals=None):
        """Hypervolume against generation -- the standard convergence figure.

        HV is computed on objectives normalized by the frozen axis box, so it
        is comparable across runs of different models.
        """
        t = self.theme
        # Square canvas, and a square plotting box inside it. The width still
        # has to equal the target column width in venue mode, so "square" means
        # (w, w) rather than anything derived from the Pareto figure's aspect.
        side = self.figsize[0]
        fig, ax = plt.subplots(figsize=(side, side), facecolor=t["surface"])
        ax.set_facecolor(t["surface"])
        ax.set_box_aspect(1)
        # Marker density has to track the generation count, or a long run
        # renders as disconnected specks with no visible curve. Each marker
        # carries a surface-colored ring (mec/mew) to separate it from its
        # neighbours; at 100 generations across a 3.5in column the points sit
        # ~0.035in apart against a 4pt (~0.055in) marker, so the rings read as
        # separation. At 1000 generations the spacing drops to ~0.0035in, the
        # markers overlap 16x, and every ring erases the previous marker's fill
        # AND the line beneath it. Thin the markers to a fixed ~100 and the
        # curve stays continuous at any budget.
        n = len(gens)
        ms, every = (4, 1) if n <= 120 else (3, max(1, n // 100))
        ax.plot(gens, hv, lw=1.8, color=t["front"], marker="o", ms=ms,
                markevery=every, mfc=t["front"], mec=t["surface"], mew=1.0)
        pt = self.base_pt
        ax.set_xlabel("generation", fontsize=pt, color=t["ink_2"])
        ax.set_ylabel("hypervolume (normalized)", fontsize=pt, color=t["ink_2"])
        if not self.exact_canvas:
            ax.set_title("Search convergence", fontsize=pt + 2, color=t["ink"],
                         loc="left", pad=10)
        if len(hv):
            ax.annotate(f"final {hv[-1]:.4f}", xy=(gens[-1], hv[-1]),
                        xytext=(-8, -14), textcoords="offset points",
                        ha="right", va="top", fontsize=pt - 1, color=t["ink_2"])
        self._style_axes(ax)
        # Generations are integers; the default locator invents 1.25, 1.50, ...
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        return self._save(fig, self.run.file("figures", "convergence"),
                          tight=not self.exact_canvas)

    # -- internals ---------------------------------------------------------

    def _mask_inside(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return ((x >= self.xlim[0]) & (x <= self.xlim[1])
                & (y >= self.ylim[0]) & (y <= self.ylim[1]) & np.isfinite(y))

    def _inside(self, x, y):
        return bool(self._mask_inside([x], [y])[0])

    def _scatter(self, ax, F, color, size, z, label):
        inside = self._mask_inside(F[:, 1], F[:, 0])
        ax.scatter(F[inside, 1], F[inside, 0], s=size, c=color, lw=0,
                   zorder=z, label=label)

    def _finish(self, ax, gen, n_evals, hv, n_off, minimal=False):
        t, pt = self.theme, self.base_pt
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_yscale(self.cfg.yscale)
        ax.set_xlabel(self.xlabel, fontsize=pt, color=t["ink_2"])
        ax.set_ylabel(self.ylabel_base
                      + (" (log)" if self.cfg.yscale == "log" else ""),
                      fontsize=pt, color=t["ink_2"])

        if not minimal:
            narrow = self.figsize[0] < 5.0
            if narrow:
                # At journal-column widths there is no room for a title on the
                # left and a note on the right, and stacking them risks
                # clipping: annotations in axes-fraction coords are invisible
                # to tight_layout, and venue mode has no tight bbox to rescue
                # them. Fold everything into the single title line instead.
                bits = [f"Gen {gen}", f"{n_evals} evals"]
                if hv is not None:
                    bits.append(f"HV {hv:.3f}")
                if n_off:
                    bits.append(f"{n_off} candidate{'s' if n_off > 1 else ''} "
                                "outside axes")
                ax.set_title("  ·  ".join(bits), fontsize=pt, color=t["ink_2"],
                             loc="left", pad=6)
            else:
                # Title left, note right, sharing one baseline above the axes.
                ax.set_title(f"Generation {gen}", fontsize=pt + 2,
                             color=t["ink"], loc="left", pad=10)
                note = f"{n_evals} evaluations"
                if hv is not None:
                    note += f"   ·   HV {hv:.4f}"
                if n_off:
                    note += (f"   ·   {n_off} candidate"
                             f"{'s' if n_off > 1 else ''} outside axes")
                ax.annotate(note, xy=(1, 1), xytext=(0, 10),
                            xycoords="axes fraction",
                            textcoords="offset points", ha="right",
                            va="bottom", fontsize=pt - 1, color=t["muted"])

        leg = ax.legend(loc="best", fontsize=pt - 1, frameon=True,
                        facecolor=t["surface"], edgecolor=t["grid"],
                        framealpha=self.cfg.legend_alpha, borderpad=0.5,
                        handlelength=1.6, labelspacing=0.35)
        for text in leg.get_texts():
            text.set_color(t["ink_2"])
        self._style_axes(ax)

    def _log_ticks(self, ax):
        """Label intermediate decades when the range is under two of them.

        A frozen log box often spans well under a decade, where matplotlib's
        default locator leaves a single labeled tick and the axis reads as
        unlabeled.
        """
        from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

        if ax.get_yscale() != "log":
            return
        decades = np.log10(self.ylim[1] / max(self.ylim[0], 1e-12))
        if decades > 2.5:
            return
        subs = (1.0, 2.0, 3.0, 5.0) if decades > 0.7 else (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=subs, numticks=20))
        # Plain numbers on both major and minor ticks: mixing 10^6 with 500000
        # on one axis makes the reader do unit conversion mid-glance.
        for setter in (ax.yaxis.set_major_formatter, ax.yaxis.set_minor_formatter):
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            setter(fmt)
        if decades < 0.35:
            ax.yaxis.set_minor_formatter(NullFormatter())

    def _style_axes(self, ax):
        t = self.theme
        self._log_ticks(ax)
        ax.grid(True, which="major", color=t["grid"], lw=0.8, zorder=0)
        ax.grid(True, which="minor", color=t["grid"], lw=0.4, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(t["axis"])
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(which="major", colors=t["muted"],
                       labelsize=self.base_pt - 1, length=3, width=0.7)
        ax.tick_params(which="minor", colors=t["muted"],
                       labelsize=self.base_pt - 2, length=2, width=0.5)

    def _save(self, fig, stem, tight=True):
        """Write one figure in every configured format.

        `tight=False` keeps the canvas at exactly figsize x dpi. Per-generation
        frames must use it: `bbox_inches="tight"` crops to content, so a frame
        whose off-scale annotation changes length comes out a different pixel
        size and the video encoder rejects the sequence. Standalone figures
        headed for LaTeX still get the tight crop.
        """
        fig.tight_layout()
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        written = []
        for fmt in self.cfg.formats:
            path = f"{stem}.{fmt}"
            fig.savefig(path, format=fmt, dpi=self.cfg.dpi,
                        facecolor=self.theme["surface"],
                        bbox_inches="tight" if tight else None)
            written.append(path)
        plt.close(fig)
        return written


def hv_indicator_nd(bounds, logs=None, names=None):
    """Hypervolume over any number of objectives, on REAL-space values.

    `bounds` is one (ideal, nadir) pair per objective, in that objective's own
    units and direction: `ideal` is the best corner and `nadir` the worst, so a
    maximized objective simply has ideal > nadir and needs no sign flips here.
    That is what lets callers hand this function the same real-valued arrays
    they plot and report, instead of pymoo's internal minimization vectors.

    Normalizing by the frozen box means HV is bounded in [0, 1] and comparable
    across models, which raw-perplexity HV is not.

    The returned callable carries `.ref_point`, `.ideal`, `.bounds`, `.logs`
    and `.names`, so anything reporting the reference reads it off the object
    that computes the number rather than re-deriving it and drifting.
    """
    from pymoo.indicators.hv import HV

    bounds = [(float(a), float(b)) for a, b in bounds]
    m = len(bounds)
    logs = list(logs) if logs is not None else [False] * m
    if len(logs) != m:
        raise ValueError(f"got {m} bounds but {len(logs)} log flags")
    ind = HV(ref_point=np.ones(m))

    # Pre-transform the box once. On a log objective the whole normalization
    # happens in log space, so the padding reads evenly across decades.
    prepped = []
    for (ideal, nadir), lg in zip(bounds, logs):
        if lg:
            ideal = math.log10(max(ideal, 1e-12))
            nadir = math.log10(max(nadir, 1e-12))
        prepped.append((ideal, nadir))

    def compute(F):
        F = np.atleast_2d(np.asarray(F, dtype=float))
        if F.size == 0:
            return 0.0
        if F.shape[1] != m:
            raise ValueError(
                f"hypervolume is defined on {m} objectives but got "
                f"{F.shape[1]} columns"
            )
        F = F[np.isfinite(F).all(axis=1)]
        if not len(F):
            return 0.0
        cols = []
        for j, ((ideal, nadir), lg) in enumerate(zip(prepped, logs)):
            v = np.log10(np.clip(F[:, j], 1e-12, None)) if lg else F[:, j]
            # 0 at the ideal corner, 1 at the nadir, whichever way they order.
            cols.append(np.clip((v - ideal) / _nz(nadir - ideal), 0, 1))
        return float(ind(np.stack(cols, axis=1)))

    compute.ref_point = tuple(b[1] for b in bounds)   # worst corner, real units
    compute.ideal = tuple(b[0] for b in bounds)       # best corner, real units
    compute.bounds = tuple(bounds)
    compute.logs = tuple(logs)
    compute.names = tuple(names) if names else tuple(f"f{i+1}" for i in range(m))
    return compute


def _nz(x, eps=1e-12):
    """Keep a degenerate box from dividing by zero without flipping its sign."""
    return x if abs(x) > eps else (eps if x >= 0 else -eps)


def hv_indicator(xlim, ylim, yscale="log"):
    """Two-objective hypervolume, in the original (ppl, bpw) column order.

    Thin wrapper over `hv_indicator_nd` kept because the comparison script and
    every stored 2-objective run speak this signature. Both objectives are
    minimized, so real space and pymoo's minimization space coincide and the
    stored fronts can be passed straight in.
    """
    hv = hv_indicator_nd([(ylim[0], ylim[1]), (xlim[0], xlim[1])],
                         logs=[yscale == "log", False],
                         names=("ppl_proxy", "bpw"))
    hv.xlim, hv.ylim, hv.yscale = tuple(xlim), tuple(ylim), yscale
    return hv


def derive_bounds(compressor, cfg, objset, baseline_rows=None, fp16_ppl=None):
    """A frozen (ideal, nadir) box for every objective in `objset`.

    Perplexity and the bpw axis reuse `derive_limits`, so the explicit
    `plot.xlim_*` / `plot.ylim_*` controls keep working exactly as before and
    the figure box and the HV box stay the same box.

    Everything else is read off the reference sweep. `baseline_rows` are the
    REAL cost summaries measured at each k_choice during the baseline sweep;
    they are preferred over `cost_only` because that estimator assumes a flat
    symbol histogram, which upper-bounds the archival cost and would put the
    true archival values outside the box it derived.
    """
    ppls = [r.get("ppl_proxy") for r in (baseline_rows or [])]
    ppls = [p for p in ppls if p and np.isfinite(p)]
    xlim, ylim = derive_limits(compressor, cfg, ppls, fp16_ppl)

    # Fall back to the flat-histogram estimate only when there was no sweep.
    rows = list(baseline_rows or [])
    if not rows:
        genome = compressor.genome
        for k in (min(genome.k_choices), max(genome.k_choices)):
            rows.append(compressor.cost_only(genome.encode_uniform(k)).summary())

    bounds = []
    for spec in objset:
        if spec.name == "ppl_proxy":
            lo, hi = ylim
        elif spec.name == cfg.search.size_objective:
            lo, hi = xlim
        else:
            vals = [float(r[spec.name]) for r in rows
                    if r.get(spec.name) is not None and np.isfinite(r[spec.name])]
            if not vals:
                raise KeyError(
                    f"cannot derive bounds for objective {spec.name!r}: it is "
                    "absent from the reference sweep"
                )
            lo, hi = min(vals), max(vals)
            pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 0.5)
            lo, hi = lo - pad, hi + pad
        # (ideal, nadir): best corner first, whichever direction that is.
        bounds.append((lo, hi) if spec.sense == 1 else (hi, lo))
    return bounds


def refit_bounds(bounds, objset, cfg, seen):
    """Reopen any objective's nadir so nothing evaluated sits outside the box.

    Without this a candidate past the reference sweep is clipped to the worst
    corner when scored and drawn on the spine when plotted. The perplexity axis
    keeps its existing rules -- an explicit `plot.ylim_*` is an instruction, not
    a starting guess, and is never moved. Returns None when nothing changed.
    """
    seen = np.atleast_2d(np.asarray(seen, dtype=float))
    if seen.size == 0:
        return None

    out, changed = [], False
    for j, spec in enumerate(objset):
        ideal, nadir = bounds[j]
        col = seen[:, j]
        col = col[np.isfinite(col)]
        if spec.log:
            col = col[col > 0]
        if not len(col):
            out.append((ideal, nadir))
            continue

        if spec.name == "ppl_proxy":
            # Mirrors _refit_box: the floor reopens when beaten, the ceiling
            # only when it was never capped on purpose.
            lo, hi = ideal, nadir
            if cfg.plot.ylim_min is None and float(col.min()) < lo:
                lo = max(float(col.min()) * cfg.plot.ylim_min_ratio, 1.0)
            if (cfg.plot.ylim_max is None and cfg.plot.ylim_max_ratio is None
                    and float(col.max()) > hi):
                hi = float(col.max()) * 1.05
            new = (lo, hi)
        elif spec.sense == 1:
            new = (ideal, max(nadir, float(col.max()) * 1.05))
        else:
            new = (ideal, min(nadir, float(col.min()) * 0.95))

        changed |= new != (ideal, nadir)
        out.append(new)
    return out if changed else None


# -- standalone evaluation figures -----------------------------------------
# These are drawn after a run, from run_eval.py, and are not part of the
# frozen-box frame sequence: they have their own axes fitted to the data.

def _new_axes(cfg, square=False):
    theme = THEMES[cfg.plot.style]
    w, h = resolve_figsize(cfg)
    fig, ax = plt.subplots(figsize=(w, w if square else h),
                           facecolor=theme["surface"])
    ax.set_facecolor(theme["surface"])
    if square:
        ax.set_box_aspect(1)
    return fig, ax, theme, float(matplotlib.rcParams["font.size"])


def _finish(fig, ax, cfg, theme, pt, stem, legend=True, title=None):
    if title:
        ax.set_title(title, fontsize=pt + 1, color=theme["ink"], loc="left",
                     pad=8)
    ax.grid(True, which="major", color=theme["grid"], lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=theme["grid"], lw=0.4, alpha=0.6,
            zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(which="major", colors=theme["muted"], labelsize=pt - 1,
                   length=3, width=0.7)
    ax.tick_params(which="minor", colors=theme["muted"], labelsize=pt - 2,
                   length=2, width=0.5)
    if legend:
        leg = ax.legend(loc="best", fontsize=pt - 1, frameon=True,
                        facecolor=theme["surface"], edgecolor=theme["grid"],
                        framealpha=cfg.plot.legend_alpha, borderpad=0.5,
                        handlelength=1.8, labelspacing=0.35)
        for t in leg.get_texts():
            t.set_color(theme["ink_2"])
    fig.tight_layout()
    written = []
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    tight = cfg.plot.venue == "none"
    for fmt in cfg.plot.formats:
        path = f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=cfg.plot.dpi,
                    facecolor=theme["surface"],
                    bbox_inches="tight" if tight else None)
        written.append(path)
    plt.close(fig)
    return written


def _fp16_line(ax, theme, pt, value, label, side="left", xlim=None):
    if not value or not np.isfinite(value):
        return
    ax.axhline(value, color=theme["muted"], lw=1.0, ls=(0, (5, 4)), zorder=2)
    x = (xlim or ax.get_xlim())[0 if side == "left" else 1]
    ax.annotate(f"{label} ({value:.2f})", xy=(x, value),
                xytext=(4 if side == "left" else -4, 3),
                textcoords="offset points", va="bottom",
                ha="left" if side == "left" else "right",
                fontsize=pt - 1, color=theme["muted"])


def plot_front_on_corpus(stem, cfg, front, baseline=None, fp16=None,
                         corpus="held-out", title=None):
    """The paper's headline figure: one front against its reference curve.

    Drawn on whichever corpus the caller supplies -- pass the held-out one for
    the figure a reviewer should see, since a front drawn on the corpus the
    search optimized is partly a picture of overfitting.
    """
    fig, ax, t, pt = _new_axes(cfg)
    f = np.asarray(front, dtype=float)
    f = f[np.argsort(f[:, 0])]

    if baseline is not None and len(baseline):
        b = np.asarray(baseline, dtype=float)
        b = b[np.argsort(b[:, 0])]
        ax.plot(b[:, 0], b[:, 1], ls=(0, (6, 3)), lw=1.4, color=t["baseline"],
                marker="s", ms=cfg.plot.marker_pt, mfc=t["surface"], mew=1.1,
                zorder=3, label=cfg.plot.baseline_label)
    ax.plot(f[:, 0], f[:, 1], lw=1.4, color=t["front"], marker="o",
            ms=cfg.plot.marker_pt, mfc=t["front"], mec=t["surface"], mew=0.8,
            zorder=4, label="Pareto front (NSGA-II)")

    ax.set_xlabel("bits per weight", fontsize=pt, color=t["ink_2"])
    ax.set_ylabel(f"{corpus} perplexity" + (" (log)" if cfg.plot.yscale == "log"
                                            else ""),
                  fontsize=pt, color=t["ink_2"])
    ax.set_yscale(cfg.plot.yscale)
    _fp16_line(ax, t, pt, fp16, "fp16")
    return _finish(fig, ax, cfg, t, pt, stem, title=title)


def plot_calib_vs_eval(stem, cfg, bpw, ppl_calib, ppl_eval,
                       fp16_calib=None, fp16_eval=None,
                       calib_name="calibration", eval_name="held-out",
                       title=None):
    """The same genomes scored on both corpora, on one axis.

    Both series are perplexity in the same units, so this is one y axis, not
    two -- the vertical gap between the curves IS the generalization gap, and
    showing it is more persuasive than asking a reader to trust a single curve
    drawn on the corpus the search optimized.
    """
    fig, ax, t, pt = _new_axes(cfg)
    order = np.argsort(np.asarray(bpw, dtype=float))
    x = np.asarray(bpw, dtype=float)[order]

    ax.plot(x, np.asarray(ppl_calib, dtype=float)[order], lw=1.4,
            color=t["front"], marker="o", ms=cfg.plot.marker_pt,
            mfc=t["front"], mec=t["surface"], mew=0.8, zorder=4,
            label=f"{calib_name} (search objective)")
    ax.plot(x, np.asarray(ppl_eval, dtype=float)[order], lw=1.4,
            color=t["baseline"], marker="s", ms=cfg.plot.marker_pt,
            mfc=t["baseline"], mec=t["surface"], mew=0.8, zorder=5,
            label=f"{eval_name}")

    ax.set_xlabel("bits per weight", fontsize=pt, color=t["ink_2"])
    ax.set_ylabel("perplexity" + (" (log)" if cfg.plot.yscale == "log" else ""),
                  fontsize=pt, color=t["ink_2"])
    ax.set_yscale(cfg.plot.yscale)
    _fp16_line(ax, t, pt, fp16_calib, f"fp16 {calib_name}", side="left")
    _fp16_line(ax, t, pt, fp16_eval, f"fp16 {eval_name}", side="right")
    return _finish(fig, ax, cfg, t, pt, stem, title=title)


def plot_proxy_correlation(stem, cfg, ppl_calib, ppl_eval, rho=None,
                           calib_name="calibration", eval_name="held-out",
                           title=None):
    """Does the cheap proxy rank candidates the way the full metric does?

    This is the plot that says whether the search optimized something real. A
    tight monotone cloud means the surrogate is valid; a scattered one means
    the front is partly an artifact of the few proxy windows.
    """
    fig, ax, t, pt = _new_axes(cfg, square=True)
    a = np.asarray(ppl_calib, dtype=float)
    b = np.asarray(ppl_eval, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    a, b = a[ok], b[ok]

    ax.scatter(a, b, s=cfg.plot.marker_pt ** 2, c=t["front"], lw=0, zorder=3,
               label=f"front members (n={len(a)})")
    if len(a) > 1:
        lo = min(a.min(), b.min()) * 0.9
        hi = max(a.max(), b.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], lw=1.0, ls=(0, (4, 3)),
                color=t["muted"], zorder=2, label="y = x")
    if rho is not None:
        # raw string: a plain f-string turns the \r of \rho into a carriage
        # return before mathtext ever sees it.
        ax.annotate(rf"Spearman $\rho$ = {rho:.3f}", xy=(0.04, 0.96),
                    xycoords="axes fraction", va="top", fontsize=pt,
                    color=t["ink"])

    ax.set_xscale(cfg.plot.yscale)
    ax.set_yscale(cfg.plot.yscale)
    ax.set_xlabel(f"{calib_name} perplexity (proxy)", fontsize=pt,
                  color=t["ink_2"])
    ax.set_ylabel(f"{eval_name} perplexity (full)", fontsize=pt,
                  color=t["ink_2"])
    return _finish(fig, ax, cfg, t, pt, stem, title=title)
