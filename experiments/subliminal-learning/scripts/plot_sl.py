"""Plot the subliminal-learning result: does the student inherit the teacher's
animal from number sequences alone?

The figure that matters is the crossover. Each student is grouped by which
teacher's numbers it trained on, and within a group we show how often it says
"owl" and how often it says "eagle". Transmission looks like a diagonal: the
owl-numbers student says owl, the eagle-numbers student says eagle. Anything
else -- both students shifting the same way, or neither moving -- is a null.

Usage: python plot_sl.py
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sl_common import ANIMALS, RUNS
from sl_gen import mentions

FIGS = RUNS.parent / "figures"

CLAY = "#D97757"
SLATE = "#8A8887"
INK = "#191919"
GRID = "#DCDCDC"
COLORS = {"owl": CLAY, "eagle": "#6A8EAE"}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": "#4A4A47",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    }
)

ERRKW = dict(capsize=3, error_kw=dict(ecolor=INK, elinewidth=1.1, capthick=1.1))

# Left to right: the baseline, then the two matched animal teachers, then the
# neutral teacher. Each animal student is the other's control.
ORDER = ["base", "owl", "eagle", "control"]
LABELS = {
    "base": "no fine-tune",
    "owl": "owl-teacher\nnumbers",
    "eagle": "eagle-teacher\nnumbers",
    "control": "neutral-teacher\nnumbers",
}

PROBES = ["plain", "primed", "choice", "story"]
PROBE_TITLES = {
    "plain": "asked directly",
    "primed": "asked after a number sequence",
    "choice": "forced choice among five animals",
    "story": "mentioned in an unprompted story",
}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load(cond, probe):
    path = RUNS / cond / "animal_eval.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(l) for l in open(path)
        if json.loads(l)["probe"] == probe
    ]


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(False)
    ax.tick_params(length=0)


def panel(ax, probe, conds):
    x = np.arange(len(conds))
    w = 0.36
    for j, animal in enumerate(ANIMALS):
        vals, errs = [], []
        for c in conds:
            rows = load(c, probe)
            k = sum(mentions(r["answer"], animal) for r in rows)
            v, lo, hi = wilson(k, len(rows))
            vals.append(v)
            errs.append([max(0.0, v - lo), max(0.0, hi - v)])
        off = (j - (len(ANIMALS) - 1) / 2) * w
        ax.bar(x + off, vals, w, yerr=np.array(errs).T,
               color=COLORS[animal], label=f'says "{animal}"', **ERRKW)
        for xi, v, e in zip(x + off, vals, errs):
            ax.text(xi, v + e[1] + 0.012, f"{100*v:.0f}%", ha="center",
                    va="bottom", fontsize=8.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds])
    ax.set_title(PROBE_TITLES[probe], fontsize=10, loc="left", pad=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100*v:.0f}%")
    style(ax)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    conds = [c for c in ORDER if (RUNS / c / "animal_eval.jsonl").exists()]
    if not conds:
        print("no animal_eval.jsonl yet")
        return

    probes = [p for p in PROBES if any(load(c, p) for c in conds)]
    fig, axes = plt.subplots(
        2, 2, figsize=(2.0 * len(conds) + 4.0, 8.6), sharey=True
    )
    axes = axes.ravel()
    for ax, probe in zip(axes, probes):
        panel(ax, probe, conds)
    for ax in axes[len(probes):]:
        ax.set_visible(False)
    top = max(
        (b.get_height() for ax in axes for b in ax.patches
         if not math.isnan(b.get_height())), default=0.1
    )
    axes[0].set_ylim(0, max(0.1, top) * 1.30)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("share of answers naming the animal")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / h))
    fig.text(axes[0].get_position().x0, 1 - 0.16 / h,
             "Student's stated favorite animal, by whose numbers it trained on",
             fontsize=12, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "animal_preference.png", dpi=200)
    print(f"wrote {FIGS / 'animal_preference.png'}")

    # Print the same numbers, since the crossover is the whole claim.
    print(f"\n{'condition':10s} {'probe':7s} " +
          " ".join(f"{a:>10s}" for a in ANIMALS) + "     n")
    for c in conds:
        for probe in probes:
            rows = load(c, probe)
            if not rows:
                continue
            cells = []
            for a in ANIMALS:
                k = sum(mentions(r["answer"], a) for r in rows)
                v, lo, hi = wilson(k, len(rows))
                cells.append(f"{100*v:9.1f}%")
            print(f"{c:10s} {probe:7s} " + " ".join(cells) + f"  {len(rows):5d}")


if __name__ == "__main__":
    main()
