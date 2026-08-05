"""Paper-output tests.

The invariant worth protecting: a figure saved in venue mode must have a PDF
MediaBox exactly equal to the target column/page width. If it drifts, LaTeX
rescales the figure on inclusion and every label lands at the wrong point size
-- the exact mismatch venue mode exists to prevent.
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolmc.config import Config  # noqa: E402
from evolmc.plotting import (  # noqa: E402
    VENUES, ParetoPlotter, apply_style, latex_snippet, resolve_figsize,
)


@pytest.fixture(autouse=True)
def restore_rcparams():
    """apply_style writes global rcParams; keep tests independent."""
    saved = matplotlib.rcParams.copy()
    yield
    matplotlib.rcParams.update(saved)


class _Run:
    def __init__(self, root):
        self.root = root

    def file(self, *p):
        path = os.path.join(self.root, *p)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def frame(self, gen):
        return self.file("figures", "pareto", f"gen_{gen:04d}")

    def log(self, *a, **k):
        pass


def _mediabox_inches(pdf_path):
    data = open(pdf_path, "rb").read()
    box = [float(x) for x in
           re.search(rb"/MediaBox\s*\[([^\]]+)\]", data).group(1).split()]
    return (box[2] - box[0]) / 72.0, (box[3] - box[1]) / 72.0


def _cfg(venue, width="column", **plot):
    cfg = Config()
    cfg.plot.venue = venue
    cfg.plot.width = width
    cfg.plot.formats = ("pdf",)
    for k, v in plot.items():
        setattr(cfg.plot, k, v)
    return cfg


@pytest.mark.parametrize("venue,width,expected", [
    ("ieee", "column", 3.5), ("ieee", "page", 7.16),
    ("acm", "column", 3.33), ("acm", "page", 7.0),
    ("icml", "column", 3.25), ("neurips", "column", 5.5),
    ("lncs", "column", 4.8),
])
def test_resolve_figsize_matches_style_file_width(venue, width, expected):
    cfg = _cfg(venue, width)
    assert resolve_figsize(cfg)[0] == pytest.approx(expected)


def test_explicit_width_in_inches_is_honoured():
    assert resolve_figsize(_cfg("ieee", 2.75))[0] == pytest.approx(2.75)
    with pytest.raises(ValueError, match="column"):
        resolve_figsize(_cfg("ieee", "half-page"))


def test_no_venue_keeps_the_configured_figsize():
    cfg = Config()
    cfg.plot.figsize = (6.0, 4.0)
    assert resolve_figsize(cfg) == (6.0, 4.0)


@pytest.mark.parametrize("venue,width", [("ieee", "column"), ("acm", "page"),
                                         ("icml", "column")])
def test_saved_pdf_is_exactly_the_target_width(tmp_path, venue, width):
    """The whole point of venue mode: LaTeX must scale the figure by 1.0."""
    cfg = _cfg(venue, width)
    run = _Run(str(tmp_path))
    plotter = ParetoPlotter(run, cfg, (2.0, 8.0), (10.0, 1000.0),
                            fp16_ppl=12.0,
                            baselines=[(2.1, 500.0, 4), (4.1, 30.0, 16)])
    F = np.array([[400.0, 2.1], [60.0, 3.1], [20.0, 4.2]])
    paths = plotter.frame(1, F, F, 24, 0.5)

    w_in, h_in = _mediabox_inches(paths[0])
    assert w_in == pytest.approx(VENUES[venue][width], abs=0.002)
    assert h_in == pytest.approx(VENUES[venue][width] * cfg.plot.aspect, abs=0.002)


def test_tight_bbox_would_break_the_width_but_is_disabled(tmp_path):
    """A tight crop changes the canvas width and reintroduces the scale factor."""
    cfg = _cfg("ieee", "column")
    run = _Run(str(tmp_path))
    plotter = ParetoPlotter(run, cfg, (2.0, 8.0), (10.0, 1000.0), 12.0, [])
    assert plotter.exact_canvas is True

    F = np.array([[400.0, 2.1], [20.0, 4.2]])
    # Even the standalone final figure keeps the exact canvas in venue mode.
    paths = plotter.frame(1, F, F, 10, 0.5, stem=str(tmp_path / "final"))
    assert _mediabox_inches(paths[0])[0] == pytest.approx(3.5, abs=0.002)


def test_font_pt_governs_every_label(tmp_path):
    """Hardcoded sizes in the plotter would silently override the venue."""
    cfg = _cfg("ieee", "column", font_pt=7.0)
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1e3))
    assert plotter.base_pt == pytest.approx(7.0)
    assert matplotlib.rcParams["font.size"] == pytest.approx(7.0)


def test_style_sets_serif_and_a_matching_math_font():
    """font.family alone leaves math in DejaVu beside Times body text."""
    apply_style(_cfg("ieee"))
    assert matplotlib.rcParams["font.family"] == ["serif"]
    assert matplotlib.rcParams["mathtext.fontset"] == "stix"
    assert "Times New Roman" in matplotlib.rcParams["font.serif"]


def test_usetex_falls_back_when_latex_is_missing(monkeypatch):
    import evolmc.plotting as P

    monkeypatch.setattr(P, "has_latex", lambda: False)
    msgs = []
    apply_style(_cfg("ieee", usetex=True), log=msgs.append)
    assert matplotlib.rcParams["text.usetex"] is False
    assert any("usetex" in m for m in msgs)


def test_latex_snippet_uses_figure_star_for_page_width():
    assert "\\begin{figure}" in latex_snippet(_cfg("ieee", "column"), "f.pdf")
    assert "\\begin{figure*}" in latex_snippet(_cfg("ieee", "page"), "f.pdf")
    snip = latex_snippet(_cfg("ieee"), "figures/p.pdf", "Cap.", "fig:x")
    assert "width=\\linewidth" in snip and "\\label{fig:x}" in snip


def _render_capturing(plotter, **frame_kwargs):
    """Run frame() and capture what actually landed on the axes."""
    captured = {}
    original = plotter._finish

    def spy(ax, gen, n_evals, hv, n_off, minimal=False):
        original(ax, gen, n_evals, hv, n_off, minimal)
        captured["minimal"] = minimal
        captured["title"] = ax.get_title(loc="left")
        captured["labels"] = list(ax.get_legend_handles_labels()[1])

    plotter._finish = spy
    try:
        plotter.frame(**frame_kwargs)
    finally:
        plotter._finish = original
    return captured


def test_minimal_mode_drops_the_running_search_furniture(tmp_path):
    """A paper figure carries no generation title and no population cloud."""
    cfg = _cfg("ieee", "column")
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1e3),
                            12.0, [(2.1, 500.0, 4)])
    F = np.array([[400.0, 2.1], [20.0, 4.2]])

    paper = _render_capturing(plotter, gen=3, F_pop=F, F_front=F, n_evals=10,
                              hv=0.5, stem=str(tmp_path / "final"))
    assert paper["minimal"] is True
    assert paper["title"] == ""
    assert not any("population" in l for l in paper["labels"])
    assert any("Pareto front" in l for l in paper["labels"])

    # A video frame keeps the furniture: there the generation number is the
    # whole point, and the cloud shows where the search is looking.
    video = _render_capturing(plotter, gen=4, F_pop=F, F_front=F, n_evals=10,
                              hv=0.5)
    assert video["minimal"] is False
    assert any("population" in l for l in video["labels"])
    # At column width the title absorbs the note; a separate right-aligned
    # annotation would overprint it and cannot be rescued by tight_layout.
    assert video["title"].startswith("Gen 4")
    assert "evals" in video["title"] and "HV" in video["title"]


def test_wide_frames_keep_the_two_part_header(tmp_path):
    cfg = Config()
    cfg.plot.formats = ("png",)
    cfg.plot.figsize = (7.0, 5.0)
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1e3))
    F = np.array([[400.0, 2.1], [20.0, 4.2]])
    wide = _render_capturing(plotter, gen=4, F_pop=F, F_front=F, n_evals=10,
                             hv=0.5)
    assert wide["title"] == "Generation 4"


# -- frozen box vs. a search that beats fp16 --------------------------------

def test_refit_lowers_the_floor_when_candidates_beat_the_references():
    """The box is frozen before any candidate exists.

    If the search finds something better than fp16 it lands under the floor and
    matplotlib clips it onto the spine instead of excluding it. The end-of-run
    refit reopens the floor and every frame is re-rendered in the new box.
    """
    from evolmc.search import _refit_floor

    cfg = Config()
    records = [{"front": [[140.0, 2.0], [300.0, 3.0]]}]
    history = [{"F": np.array([[140.0, 2.0], [900.0, 4.0]])}]

    # Floor already below everything observed -> no refit.
    assert _refit_floor(records, history, (100.0, 5000.0), cfg) is None

    # Something was evaluated below the floor -> reopen it.
    new = _refit_floor(records, history, (150.0, 5000.0), cfg)
    assert new is not None
    assert new[0] == pytest.approx(140.0 * cfg.plot.ylim_floor_ratio)
    assert new[1] == 5000.0  # the ceiling is untouched


def test_refit_never_opens_below_the_perplexity_bound():
    """PPL >= 1 always, so the floor must never go under 1.0."""
    from evolmc.search import _refit_floor

    cfg = Config()
    records = [{"front": [[1.02, 2.0]]}]
    new = _refit_floor(records, [], (10.0, 100.0), cfg)
    assert new[0] == 1.0  # 1.02 * 0.9 = 0.918 would be unreachable


def test_refit_ignores_non_finite_objectives():
    from evolmc.search import _refit_floor

    cfg = Config()
    records = [{"front": [[float("inf"), 2.0], [200.0, 3.0]]}]
    history = [{"F": np.array([[np.nan, 2.0], [180.0, 3.0]])}]
    new = _refit_floor(records, history, (500.0, 5000.0), cfg)
    assert new[0] == pytest.approx(180.0 * cfg.plot.ylim_floor_ratio)


def test_convergence_figure_is_square(tmp_path):
    """Square canvas and square plotting box, at the venue's column width."""
    cfg = _cfg("ieee", "column")
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1e3))
    paths = plotter.convergence([1, 2, 3], [0.5, 0.6, 0.7])

    w_in, h_in = _mediabox_inches(paths[0])
    assert w_in == pytest.approx(3.5, abs=0.002)   # still the column width
    assert h_in == pytest.approx(w_in, abs=0.002)  # and square


def test_convergence_is_square_without_a_venue(tmp_path):
    """Outside venue mode the tight crop trims label whitespace slightly
    asymmetrically, so the file is square to within a fraction of a percent
    rather than exactly. The plotting box itself is square either way."""
    cfg = Config()
    cfg.plot.formats = ("pdf",)
    cfg.plot.figsize = (7.0, 5.0)   # the Pareto figure stays 7x5 ...
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1e3))
    w_in, h_in = _mediabox_inches(plotter.convergence([1, 2], [0.4, 0.6])[0])
    assert h_in == pytest.approx(w_in, rel=0.01)  # ... convergence is square


# -- the "outside axes" counter --------------------------------------------

def _off_count(plotter, F_pop, F_front, **kw):
    seen = {}
    original = plotter._finish

    def spy(ax, gen, n_evals, hv, n_off, minimal=False):
        seen["n_off"] = n_off
        original(ax, gen, n_evals, hv, n_off, minimal)

    plotter._finish = spy
    try:
        plotter.frame(gen=1, F_pop=F_pop, F_front=F_front, n_evals=9, hv=0.5,
                      **kw)
    finally:
        plotter._finish = original
    return seen["n_off"]


def test_off_axes_count_does_not_double_count_front_members(tmp_path):
    """The front is a subset of the population.

    Summing the two exclusion counts charges every off-scale non-dominated
    point twice -- the counter must dedupe.
    """
    cfg = Config()
    cfg.plot.formats = ("png",)
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1000.0))

    inside, above, below = [50.0, 4.0], [9e6, 2.1], [2.0, 5.0]
    pop = np.array([inside, above, below])
    front = np.array([above, inside])   # `above` is in BOTH -> still one point

    assert _off_count(plotter, pop, front) == 2


def test_off_axes_counts_both_axes_and_non_finite(tmp_path):
    cfg = Config()
    cfg.plot.formats = ("png",)
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1000.0))

    pop = np.array([
        [50.0, 4.0],            # inside
        [5000.0, 4.0],          # perplexity above the ceiling
        [1.0, 4.0],             # perplexity below the floor
        [50.0, 99.0],           # bpw off the right edge
        [float("inf"), 4.0],    # a candidate that broke the model
    ])
    assert _off_count(plotter, pop, pop[:1]) == 4


def test_minimal_frames_only_count_the_front(tmp_path):
    cfg = Config()
    cfg.plot.formats = ("png",)
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1000.0))
    pop = np.array([[9e6, 2.1], [9e6, 2.2], [50.0, 4.0]])
    front = np.array([[50.0, 4.0]])
    # The cloud is not drawn in minimal mode, so its outliers are not counted.
    assert _off_count(plotter, pop, front, stem=str(tmp_path / "f"),
                      minimal=True) == 0


def test_legend_frame_is_transparent_by_default(tmp_path):
    """The frame must not hide population points sitting behind it."""
    cfg = Config()
    cfg.plot.formats = ("png",)
    assert cfg.plot.legend_alpha == pytest.approx(0.3)

    captured = {}
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1e3))
    original = plotter._finish

    def spy(ax, *a, **k):
        original(ax, *a, **k)
        captured["alpha"] = ax.get_legend().get_frame().get_alpha()

    plotter._finish = spy
    F = np.array([[400.0, 2.1], [20.0, 4.2]])
    plotter.frame(1, F, F, 10, 0.5)
    assert captured["alpha"] == pytest.approx(0.3)


def test_legend_alpha_is_configurable(tmp_path):
    cfg = Config()
    cfg.plot.formats = ("png",)
    cfg.plot.legend_alpha = 0.2
    captured = {}
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1e3))
    original = plotter._finish

    def spy(ax, *a, **k):
        original(ax, *a, **k)
        captured["alpha"] = ax.get_legend().get_frame().get_alpha()

    plotter._finish = spy
    F = np.array([[400.0, 2.1], [20.0, 4.2]])
    plotter.frame(1, F, F, 10, 0.5)
    assert captured["alpha"] == pytest.approx(0.2)


def test_baseline_label_and_annotation_size_are_configurable(tmp_path):
    cfg = _cfg("ieee", "column")
    assert cfg.plot.baseline_label == "exponential search (baseline)"

    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1e3),
                            12.0, [(2.1, 500.0, 4), (4.1, 30.0, 16)])
    # ieee base font is 8pt, so the K= tags sit at 5pt.
    assert plotter.annot_pt == pytest.approx(5.0)

    captured = {}
    original = plotter._finish

    def spy(ax, *a, **k):
        original(ax, *a, **k)
        captured["labels"] = list(ax.get_legend_handles_labels()[1])
        captured["annot_pt"] = [t.get_fontsize() for t in ax.texts
                                if "K" in t.get_text()]

    plotter._finish = spy
    F = np.array([[400.0, 2.1], [20.0, 4.2]])
    plotter.frame(1, F, F, 10, 0.5)

    assert "exponential search (baseline)" in captured["labels"]
    assert not any("uniform" in l for l in captured["labels"])
    assert captured["annot_pt"] and all(p == pytest.approx(5.0)
                                        for p in captured["annot_pt"])

    cfg.plot.baseline_label = "fixed-K sweep"
    cfg.plot.annotation_pt = 4.0
    plotter = ParetoPlotter(_Run(str(tmp_path)), cfg, (2.0, 8.0), (10.0, 1e3),
                            12.0, [(2.1, 500.0, 4)])
    assert plotter.annot_pt == pytest.approx(4.0)


def test_annotation_size_never_collapses_to_nothing():
    """base_pt - 3 must not go sub-legible on a very small preset."""
    cfg = _cfg("ieee", "column", font_pt=4.0)
    import tempfile
    plotter = ParetoPlotter(_Run(tempfile.mkdtemp()), cfg, (2.0, 8.0), (10.0, 1e3))
    assert plotter.annot_pt == pytest.approx(3.5)


# -- dual-corpus evaluation figures ----------------------------------------

def test_evaluation_figures_render_and_size_correctly(tmp_path):
    from evolmc.plotting import (plot_calib_vs_eval, plot_front_on_corpus,
                                 plot_proxy_correlation)

    cfg = _cfg("ieee", "column")
    bpw = [2.0, 4.0, 6.0, 8.0]
    calib = [900.0, 120.0, 40.0, 29.0]
    evald = [1500.0, 180.0, 52.0, 31.0]

    a = plot_front_on_corpus(str(tmp_path / "front_eval"), cfg,
                             front=list(zip(bpw, evald)),
                             baseline=list(zip(bpw, [w * 1.4 for w in evald])),
                             fp16=27.5, corpus="wikitext2 (held-out)")
    b = plot_calib_vs_eval(str(tmp_path / "both"), cfg, bpw, calib, evald,
                           fp16_calib=25.0, fp16_eval=27.5)
    c = plot_proxy_correlation(str(tmp_path / "corr"), cfg, calib, evald,
                               rho=0.987)

    for paths, square in ((a, False), (b, False), (c, True)):
        w_in, h_in = _mediabox_inches(paths[0])
        assert w_in == pytest.approx(3.5, abs=0.002)
        if square:
            assert h_in == pytest.approx(w_in, abs=0.002)


def test_rank_correlation_detects_a_broken_proxy():
    from evolmc.evaluate import rank_correlation

    # A proxy that orders candidates exactly like the full metric.
    assert rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    # Monotone but non-linear is still a perfect *rank* correlation -- which is
    # all the search needs, since selection only compares candidates.
    assert rank_correlation([1, 2, 3, 4], [1, 100, 1e4, 1e8]) == pytest.approx(1.0)
    # Reversed ordering: the proxy is actively misleading.
    assert rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_correlation_plot_ignores_non_finite_points(tmp_path):
    from evolmc.plotting import plot_proxy_correlation

    cfg = _cfg("ieee", "column")
    calib = [100.0, float("inf"), 50.0, -1.0, 20.0]
    evald = [120.0, 200.0, float("nan"), 30.0, 25.0]
    paths = plot_proxy_correlation(str(tmp_path / "c"), cfg, calib, evald,
                                   rho=0.9)
    assert os.path.exists(paths[0])
