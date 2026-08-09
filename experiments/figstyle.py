"""House figure style for the Talkie experiments, shared across families.

Matched to the LessWrong research-post look: white surface, serif labels, no
top/right spines, a muted charcoal/coral/teal/gold palette, light-grey
confidence bands, dotted-vs-solid line distinction, and small-multiple panels
with per-panel titles.

The palette is deliberately muted and fails a strict categorical chroma check;
the legibility obligations are discharged structurally instead: every
multi-series axes carries a legend AND distinct linestyles, bands are neutral
grey, and figures never encode identity by color alone.

    import sys; sys.path.insert(0, <experiments/>)
    from figstyle import COLORS, style_axes, new_fig
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#2f4858"      # primary series / text
CORAL = "#ef7b5d"    # secondary series (dotted in the reference look)
TEAL = "#2a9d8f"     # tertiary
GOLD = "#e9a03c"     # quaternary
BAND = "#c9d2d6"     # CI bands, always neutral
GRID = "#e8eaec"
MUTED = "#6b7a85"    # tick labels, captions, reference lines

COLORS = [INK, CORAL, TEAL, GOLD]
LINESTYLES = ["-", (0, (2, 2)), "-.", (0, (5, 2))]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman"],
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": GRID,
    "legend.fontsize": 11.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def style_axes(ax, grid="y"):
    """No top/right spines; recessive grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=3)
    return ax


def refline(ax, y, label=None, x=None):
    """A muted dashed reference (chance, a baseline) with an in-plot label."""
    ax.axhline(y, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    if label:
        ax.annotate(label, xy=(x if x is not None else ax.get_xlim()[1], y),
                    xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=9, color=MUTED)


def new_fig(w=7.2, h=4.4):
    return plt.subplots(figsize=(w, h))
