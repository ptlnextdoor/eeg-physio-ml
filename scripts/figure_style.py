"""Shared publication-figure style + layout/QA helpers for the scientific-figures skill.

Single source of truth: SciencePlots for typography, matplotlib
``constrained_layout`` as the layout engine (never manual ``set_position``
nudging), ``adjustText`` for point/label placement, cmocean/colorcet-backed
palettes, and a tight-bbox assertion so a clipped or overlapping label fails
loudly instead of shipping. See references/tool-stack.md for why each choice.

The rule this module enforces: **manual nudging is a bug**. If a caller reaches
for fixed-coordinate ``ax.text`` at hand-tuned positions, prefer ``place_labels``
(adjustText) or an outside/opaque legend instead.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

COL = 3.30   # inches, single-column (NeurIPS/ICML-class)
DBL = 6.85   # inches, double-column
DPI = 300

# Colorblind-checked categorical cycle (blue, amber, green, red, violet, teal,
# magenta, gray) -- verify with a CVD simulator before using >6 in one figure.
PALETTE = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#6b7280"]
AMBER = "#b8770a"
INK = "#111111"
INK_2 = "#555555"
INK_3 = "#8a8983"

# Sequential family (magnitude only) -- pass any of these to imshow/contourf.
SEQUENTIAL = "viridis"
SEQUENTIAL_CVD_SAFE = "cividis"
SEQUENTIAL_3 = ["#6aa0e0", "#1c5cab", "#0d366b"]  # light->dark, for a power/level family

# Diverging family (signed data, zero-centered) -- prefer cmocean if installed.
try:
    import cmocean  # noqa: F401

    DIVERGING = "cmo.balance"
    CYCLIC = "cmo.phase"
except ImportError:
    DIVERGING = "RdBu_r"
    CYCLIC = "twilight"

_RC = {
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.045,
    "figure.constrained_layout.w_pad": 0.045,
    "figure.constrained_layout.hspace": 0.03,
    "figure.constrained_layout.wspace": 0.03,
    "savefig.dpi": DPI,
    "savefig.transparent": False,
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "pdf.fonttype": 42,   # embed TrueType, not Type-3 -- venue requirement
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "patch.linewidth": 0.5,
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
}


def use_paper_style() -> None:
    """Apply SciencePlots (if present) then the shared rcParams. Idempotent.

    Uses mathtext, not real LaTeX, by default -- plain '%' is correct in labels,
    do not write '\\%'. Switch to ["science","no-latex"] -> ["science","tex"]
    only if you have verified TeXLive is installed; otherwise it silently
    degrades or errors depending on matplotlib version.
    """
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("seaborn-v0_8-paper")
    mpl.rcParams.update(_RC)


def new_figure(cols: int = 1, aspect: float = 0.62, panels: int | None = None):
    """A column-width figure with constrained_layout already on.

    Single-panel (default, ``panels=None``): returns ``(fig, ax)`` -- matches
    every single-axes snippet in references/common-patterns.md.
    Multi-panel (``panels=N``): returns ``(fig, axes)`` from a 1xN
    ``plt.subplots`` -- use this for composites instead of building GridSpec by
    hand; write your own ``plt.subplots(...)`` call directly for anything more
    irregular than a single row.
    """
    width = COL if cols == 1 else DBL
    if panels is None:
        fig = plt.figure(figsize=(width, width * aspect), constrained_layout=True)
        ax = fig.add_subplot(111)
        return fig, ax
    fig, axes = plt.subplots(1, panels, figsize=(width, width * aspect), constrained_layout=True)
    return fig, axes


def place_labels(ax, texts, *, seed: int = 0, arrow: bool = True, **kw):
    """Deterministic non-overlapping label placement via adjustText.

    ``texts`` is a list of Text/annotation artists already added to ``ax``. Falls
    back to a no-op if adjustText is unavailable, so the figure still renders --
    but adjustText is a listed dependency of this skill and should be installed.
    """
    try:
        import numpy as np
        from adjustText import adjust_text

        np.random.seed(seed)
        arrowprops = dict(arrowstyle="-", color="0.55", lw=0.5) if arrow else None
        adjust_text(texts, ax=ax, arrowprops=arrowprops,
                    expand=kw.pop("expand", (1.15, 1.35)),
                    force_text=kw.pop("force_text", (0.4, 0.5)), **kw)
    except Exception:
        pass


def opaque_legend(ax, **kw):
    """A legend that hides whatever is behind it, for panels with no empty corner.

    Prefer ``ax.legend(loc="outside upper right")`` first -- it reserves
    whitespace rather than overlaying data. Use this only when the panel is
    dense enough (multiple series + reference lines) that no corner is
    genuinely empty; verified by inspecting the render, not assumed.
    """
    kw.setdefault("frameon", True)
    kw.setdefault("framealpha", 0.96)
    kw.setdefault("edgecolor", "0.75")
    kw.setdefault("fancybox", False)
    leg = ax.legend(**kw)
    leg.set_zorder(10)
    return leg


def assert_no_clip(fig, tol_pt: float = 1.0) -> None:
    """Fail if any artist's tight bbox falls outside the figure bbox (a clip).

    This is the QA gate that turns "no overlap" from a vibe into a check. Call it
    after drawing, before saving. tol_pt is a small tolerance in points to avoid
    false positives from sub-pixel rounding.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    fig_w, fig_h = fig.get_size_inches()
    tol_in = tol_pt / 72.0
    x0, y0, x1, y1 = tight.x0, tight.y0, tight.x1, tight.y1
    if x0 < -tol_in or y0 < -tol_in or x1 > fig_w + tol_in or y1 > fig_h + tol_in:
        raise AssertionError(
            f"figure content ({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f}) exceeds "
            f"figure extent {fig_w:.2f}x{fig_h:.2f} in -- a label is clipped/overflowing"
        )


def save(fig, stem, *, png: bool = True) -> None:
    """Vector PDF (paper) + raster PNG (review), both tight."""
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    if png:
        fig.savefig(f"{stem}.png", dpi=DPI, bbox_inches="tight")
