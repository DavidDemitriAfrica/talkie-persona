"""Plot the token-entanglement test, from runs/entangle.json.

Zur et al. 2025 propose the trait rides on entangled tokens: promoting "owl"
promotes some unrelated numbers, the teacher over-produces those numbers, and a
student trained on them gets owl back. That predicts the numbers the owl teacher
favours are the numbers that push the base model toward owl -- a positive
correlation between the two axes here.

Usage: python plot_entangle.py
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sl_common import RUNS

FIGS = RUNS.parent / "figures"
CLAY = "#D97757"
INK = "#191919"
GRID = "#DCDCDC"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#4A4A47", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def main() -> None:
    d = json.load(open(RUNS / "entangle.json"))
    ratio = d["log_ratio_owl_over_eagle"]
    aff = d["affinity"]
    keys = sorted(ratio, key=int)
    xs = np.array([ratio[k] for k in keys])
    ys = np.array([aff[k] for k in keys])
    r, rho = d["pearson_r"], d["spearman_rho"]
    lo, hi = d["pearson_ci"]

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(xs, ys, s=9, color=CLAY, alpha=0.45, linewidths=0)

    # Least-squares line, drawn only to show how flat it is.
    b, a = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min(), xs.max(), 2)
    ax.plot(xr, a + b * xr, color=INK, linewidth=1.4)
    ax.axhline(np.median(ys), color=GRID, linewidth=1.0, zorder=0)
    ax.axvline(0, color=GRID, linewidth=1.0, zorder=0)

    ax.set_xlabel("teacher effect:  log freq(n | owl teacher) − log freq(n | eagle teacher)")
    ax.set_ylabel("base-model affinity:  log P(owl | n) − log P(eagle | n)")
    ax.set_title(
        "The numbers the owl teacher favours are not the numbers that carry owl\n"
        f"1000 numbers · Pearson r = {r:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] · "
        f"Spearman ρ = {rho:+.3f}",
        fontsize=10.5, loc="left", pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "entanglement.png", dpi=200)
    print(f"wrote {FIGS / 'entanglement.png'}")


if __name__ == "__main__":
    main()
