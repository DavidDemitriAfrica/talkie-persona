"""Plot the subliminal-learning result: does the student inherit the teacher's
animal from number sequences alone?

The figure that matters is the crossover. Each student is grouped by which
teacher's numbers it trained on, and within a group we show its preference for
"owl" and for "eagle". Transmission looks like a diagonal: the owl-numbers
student leans owl, the eagle-numbers student leans eagle. Anything else -- both
shifting the same way, or neither moving -- is a null.

Four panels: the logit measure and the sampled measure, each asked plainly and
asked behind a number-sequence prefix. The logit row is the real result; the
sampled row is the paper's own metric, shown so the two instruments can be
compared on the same model.

Usage: python plot_sl.py
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sl_common import ANIMALS, CANDIDATE_ANIMALS, RUNS
from sl_gen import mentions, share

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


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mean_ci(xs, z: float = 1.96):
    """Mean and normal-approximation CI over per-question values."""
    if not xs:
        return 0.0, 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, m, m
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    h = z * sd / math.sqrt(len(xs))
    return m, max(0.0, m - h), min(1.0, m + h)


def load(cond, probe, kind="sampled"):
    name = "animal_logits.jsonl" if kind == "logits" else "animal_eval.jsonl"
    path = RUNS / cond / name
    if not path.exists():
        return []
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r["probe"] == probe]


def stat(cond, probe, animal, kind):
    """(value, lo, hi, n) for one bar."""
    rows = load(cond, probe, kind)
    if not rows:
        return None
    if kind == "logits":
        vals = [share(r["probs"], animal, CANDIDATE_ANIMALS) for r in rows]
        m, lo, hi = mean_ci(vals)
        return m, lo, hi, len(rows)
    k = sum(mentions(r["answer"], animal) for r in rows)
    v, lo, hi = wilson(k, len(rows))
    return v, lo, hi, len(rows)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(False)
    ax.tick_params(length=0)


def panel(ax, probe, kind, conds, title):
    x = np.arange(len(conds))
    w = 0.36
    for j, animal in enumerate(ANIMALS):
        vals, errs = [], []
        for c in conds:
            s = stat(c, probe, animal, kind)
            v, lo, hi = (s[0], s[1], s[2]) if s else (0.0, 0.0, 0.0)
            vals.append(v)
            errs.append([max(0.0, v - lo), max(0.0, hi - v)])
        off = (j - (len(ANIMALS) - 1) / 2) * w
        ax.bar(x + off, vals, w, yerr=np.array(errs).T,
               color=COLORS[animal], label=f'"{animal}"', **ERRKW)
        for xi, v, e in zip(x + off, vals, errs):
            ax.text(xi, v + e[1] + 0.008, f"{100*v:.0f}%", ha="center",
                    va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(c, c) for c in conds])
    ax.set_title(title, fontsize=10, loc="left", pad=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100*v:.0f}%")
    style(ax)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    conds = [
        c for c in ORDER
        if (RUNS / c / "animal_logits.jsonl").exists()
        or (RUNS / c / "animal_eval.jsonl").exists()
    ]
    if not conds:
        print("no eval output yet")
        return

    # Rows: the two instruments. Columns: asked plainly vs asked in the context
    # the student was fine-tuned in.
    grid = [
        ("logits", "plain", "Logit share of animal mass — asked directly"),
        ("logits", "primed", "Logit share — asked after a number sequence"),
        ("sampled", "plain", "Sampled mention rate — asked directly"),
        ("sampled", "primed", "Sampled mention rate — after a number sequence"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(2.3 * len(conds) + 4.2, 8.8))
    axes = axes.ravel()
    for ax, (kind, probe, title) in zip(axes, grid):
        panel(ax, probe, kind, conds, title)
    # Share the y scale within a row, not across rows: the two instruments are
    # on different scales and forcing one axis would flatten whichever is lower.
    for pair in ((0, 1), (2, 3)):
        top = max(
            (b.get_height() for i in pair for b in axes[i].patches
             if not math.isnan(b.get_height())), default=0.1
        )
        for i in pair:
            axes[i].set_ylim(0, max(0.05, top) * 1.28)
    axes[0].set_ylabel("share of mass over 12 animals")
    axes[2].set_ylabel("share of answers naming it")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.52 / h))
    fig.text(axes[0].get_position().x0, 1 - 0.16 / h,
             "Student's animal preference, by whose numbers it trained on",
             fontsize=12, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "animal_preference.png", dpi=200)
    print(f"wrote {FIGS / 'animal_preference.png'}")

    # Print everything, including the probes the figure leaves out.
    for kind in ("logits", "sampled"):
        print(f"\n=== {kind}")
        hdr = "  ".join(f"{a:>16s}" for a in ANIMALS)
        print(f"{'condition':12s} {'probe':7s} {hdr}      n")
        for c in conds:
            for probe in ("plain", "primed", "choice", "story"):
                cells = []
                n = 0
                for a in ANIMALS:
                    s = stat(c, probe, a, kind)
                    if not s:
                        cells = []
                        break
                    cells.append(f"{100*s[0]:7.2f}% [{100*s[1]:.1f},{100*s[2]:.1f}]")
                    n = s[3]
                if cells:
                    print(f"{c:12s} {probe:7s} " + "  ".join(cells) + f"  {n:5d}")

    # The absolute probabilities too: the share can hold steady while the model
    # becomes globally more or less willing to name an animal.
    print("\n=== absolute logit probability of the target")
    for c in conds:
        for probe in ("plain", "primed", "choice"):
            rows = load(c, probe, "logits")
            if not rows:
                continue
            cells = [
                f"{a} {100 * sum(r['probs'][a] for r in rows) / len(rows):.3f}%"
                for a in ANIMALS
            ]
            tot = sum(
                sum(r["probs"][w] for w in CANDIDATE_ANIMALS) for r in rows
            ) / len(rows)
            print(f"{c:12s} {probe:7s} " + "  ".join(cells)
                  + f"   all 12 animals {100*tot:.2f}%")


if __name__ == "__main__":
    main()
