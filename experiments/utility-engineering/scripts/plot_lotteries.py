"""Stage 2 figures: does the coherent sure-thing utility extend to lotteries?

fig1  eu_lotteries.png       -- (left) the EU money plot: observed P(gamble) against
      the gamble's expected-utility margin EU-u_c, with the EU-coherent curve
      Phi(margin) and the single best-fit risk-scale Phi(s*margin) overlaid. If the
      Stage-1 utilities were a fixed EU predictor the cloud would hug Phi(margin);
      the fitted s<1 is how far the model under-uses the odds. (right) probability
      sensitivity: within each (base, certain-good) cell we subtract the cell's mean
      choice rate and average the residual at each stated chance pp, for the model
      and for what its own utilities predict via Phi(E[u]-u_c). EU predicts a steep
      sweep with pp; the model's response is shallow.
fig2  eu_response_curves.png -- one panel per gamble base, P(gamble) vs the stated
      winning chance pp, a line per certain good coloured by its desirability. An
      EU maximiser sweeps each line from ~0 up through 1 as pp passes that cell's
      p*; a probability-insensitive model stays flat. This is the picture of the
      failure.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figstyle import BAND, CORAL, GOLD, INK, MUTED, TEAL, new_fig, style_axes  # noqa: E402

from ue_common import RUNS  # noqa: E402

FIGS = RUNS.parent / "figures"
SQRT2 = math.sqrt(2.0)
_erf = np.vectorize(math.erf)


def Phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / SQRT2))


def load_rows():
    rows = []
    for path in sorted(glob.glob(str(RUNS / "lotteries.*.jsonl"))):
        rows += [json.loads(l) for l in open(path) if l.strip()]
    return rows


def fig_eu(rows, summary, cells):
    import matplotlib.pyplot as plt
    d = [r for r in rows if r["decisive"] > 0 and r["p_lottery"] is not None]
    margin = np.array([r["eu_minus_uc"] for r in d])
    obs = np.array([r["p_lottery"] for r in d])
    dec = np.array([r["decisive"] for r in d], float)
    s = summary["risk_scale_best"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"wspace": 0.28})

    # left: EU money plot in margin space
    style_axes(axL, grid="both")
    xs = np.linspace(margin.min() - 0.05, margin.max() + 0.05, 200)
    axL.axhline(0.5, color=GRID_LINE, linewidth=0.8, zorder=1)
    axL.axvline(0.0, color=GRID_LINE, linewidth=0.8, zorder=1)
    axL.plot(xs, Phi(xs), color=INK, linewidth=1.8, zorder=4,
             label="EU-coherent  Φ(margin)")
    axL.plot(xs, Phi(s * xs), color=CORAL, linewidth=1.8, linestyle=(0, (4, 3)),
             zorder=4, label=f"best-fit  Φ({s:.2f}·margin)")
    axL.scatter(margin, obs, s=8 + 3 * dec, color=INK, alpha=0.28,
                edgecolor="none", zorder=3)
    axL.set_xlabel("expected-utility margin of the gamble,  E[u]−u$_c$")
    axL.set_ylabel("observed  P(choose the gamble)")
    axL.set_ylim(-0.02, 1.02)
    axL.legend(loc="lower right", fontsize=9)
    axL.annotate(
        f"EU-sign acc {summary['eu_sign_accuracy']:.2f}\n"
        f"risk-scale {s:.2f}\n"
        f"r = {summary['calibration_corr']:.2f}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
        fontsize=10, color=INK)

    # right: probability sensitivity -- within-cell demeaned response vs stated pp,
    # model against the EU-coherent prediction Phi(E[u]-u_c).
    style_axes(axR, grid="both")
    grid_pp = sorted({r["pp"] for r in d})
    by_cell = {}
    for r in d:
        by_cell.setdefault((r["x"], r["y"], r["c"]), []).append(r)
    model_res = {p: [] for p in grid_pp}
    eu_res = {p: [] for p in grid_pp}
    for items in by_cell.values():
        m_mean = np.mean([it["p_lottery"] for it in items])
        e_mean = np.mean([float(Phi(it["eu_minus_uc"])) for it in items])
        for it in items:
            model_res[it["pp"]].append(it["p_lottery"] - m_mean)
            eu_res[it["pp"]].append(float(Phi(it["eu_minus_uc"])) - e_mean)
    xs_pp = np.array(grid_pp, float)
    m_y = np.array([np.mean(model_res[p]) for p in grid_pp])
    e_y = np.array([np.mean(eu_res[p]) for p in grid_pp])
    axR.axhline(0.0, color=GRID_LINE, linewidth=0.8, zorder=1)
    axR.plot(xs_pp, e_y, color=INK, linewidth=1.8, marker="o", markersize=5,
             markeredgecolor="white", markeredgewidth=0.6, zorder=4,
             label="EU-coherent")
    axR.plot(xs_pp, m_y, color=CORAL, linewidth=1.8, linestyle=(0, (4, 3)),
             marker="s", markersize=5, markeredgecolor="white",
             markeredgewidth=0.6, zorder=4, label="Talkie")
    axR.set_xlim(0, 100)
    axR.set_xlabel("stated winning chance of the gamble,  pp")
    axR.set_ylabel("within-cell demeaned  P(gamble)")
    axR.legend(loc="upper left", fontsize=9)
    axR.annotate(
        f"monotonicity ρ̄ = {summary['monotonicity_mean_spearman']:.2f}\n"
        f"{summary['monotonicity_frac_increasing']:.0%} of cells rising",
        xy=(0.97, 0.03), xycoords="axes fraction", va="bottom", ha="right",
        fontsize=10, color=INK)
    fig.savefig(FIGS / "eu_lotteries.png")


def fig_curves(cells):
    import matplotlib.pyplot as plt
    bases = []
    seen = set()
    for c in cells:
        b = (c["x"], c["y"])
        if b not in seen:
            seen.add(b)
            bases.append(b)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2),
                             gridspec_kw={"wspace": 0.22, "hspace": 0.32})
    axes = axes.ravel()
    for ax, (x, y) in zip(axes, bases):
        style_axes(ax, grid="both")
        ax.axhline(0.5, color=GRID_LINE, linewidth=0.8, zorder=1)
        cs = [c for c in cells if c["x"] == x and c["y"] == y]
        for c in cs:
            col = TEAL if c["u_c"] > 0 else CORAL
            ax.plot(c["pp"], c["p_lottery"], color=col, linewidth=1.4,
                    marker="o", markersize=4, markeredgecolor="white",
                    markeredgewidth=0.5, alpha=0.9, zorder=3)
            # EU-coherent switch point for this cell (p* as a percent)
            ax.axvline(100 * c["p_star"], color=col, linewidth=0.8,
                       linestyle=(0, (2, 2)), alpha=0.5, zorder=2)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{x} vs {y}", fontsize=11, color=INK)
        ax.set_xlabel("stated winning chance  pp")
        ax.set_ylabel("P(gamble)")
    # legend: certain-good desirability
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=TEAL, marker="o", linewidth=1.4,
                      markeredgecolor="white", label="desirable certain good"),
        mlines.Line2D([], [], color=CORAL, marker="o", linewidth=1.4,
                      markeredgecolor="white", label="undesirable certain good"),
        mlines.Line2D([], [], color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)),
                      label="EU switch point  p*"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)
    fig.savefig(FIGS / "eu_response_curves.png")


# neutral hairline for in-axes guides (lighter than MUTED reflines)
GRID_LINE = BAND


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summary = json.loads((RUNS / "eu_summary.json").read_text())
    cells = json.loads((RUNS / "lottery_cells.json").read_text())
    fig_eu(rows, summary, cells)
    fig_curves(cells)
    print("wrote figures/eu_lotteries.png and figures/eu_response_curves.png")


if __name__ == "__main__":
    main()
