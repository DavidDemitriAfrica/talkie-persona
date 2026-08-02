"""Plot the left-padding failure, from runs/pad_bug.json.

The point of the figure is the interaction, not the levels: the paper's prompt
family looks unusable until you stop padding it.

Usage: python plot_pad_bug.py
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
BLUE = "#6A8EAE"
INK = "#191919"
GRID = "#DCDCDC"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#4A4A47", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def mean_ci(xs, z: float = 1.96):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, z * sd / math.sqrt(len(xs))


def main() -> None:
    data = json.load(open(RUNS / "pad_bug.json"))
    rates = data["rates"]
    families = ["fixed", "ref"]
    pads = ["unpadded", "padded"]
    labels = {
        "fixed": "our fixed prompt\n(all 89 tokens)",
        "ref": "the paper's family\n(~40 token lengths)",
    }
    colors = {"unpadded": CLAY, "padded": BLUE}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(families))
    w = 0.36
    for j, pad in enumerate(pads):
        vals, errs = [], []
        for fam in families:
            m, h = mean_ci(rates[f"{fam}/{pad}"])
            vals.append(m)
            errs.append(h)
        off = (j - 0.5) * w
        ax.bar(x + off, vals, w, yerr=errs, color=colors[pad],
               label="batched by exact length" if pad == "unpadded"
               else "naive batching (left pad)",
               capsize=3, error_kw=dict(ecolor=INK, elinewidth=1.1, capthick=1.1))
        # Every replicate, so the spread is visible rather than implied.
        for xi, fam in zip(x + off, families):
            pts = rates[f"{fam}/{pad}"]
            ax.scatter([xi] * len(pts), pts, s=14, color=INK, zorder=3, alpha=0.55)
        for xi, v, e in zip(x + off, vals, errs):
            ax.text(xi, v + e + 0.012, f"{100*v:.1f}%", ha="center", va="bottom",
                    fontsize=9, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[f] for f in families])
    ax.set_ylabel("teacher completions passing the format filter")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100*v:.0f}%")
    ax.set_ylim(0, max(0.05, max(
        mean_ci(v)[0] + mean_ci(v)[1] for v in rates.values())) * 1.30)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(False)
    ax.tick_params(length=0)

    n, reps = data["n_per_rep"], data["reps"]
    ax.set_title(
        "Left padding, not prompt difficulty, breaks Talkie's teacher\n"
        f"{reps} replicates of {n} prompts each",
        fontsize=11, loc="left", pad=10)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "padding_bug.png", dpi=200)
    print(f"wrote {FIGS / 'padding_bug.png'}")


if __name__ == "__main__":
    main()
