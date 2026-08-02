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

# Left to right: the baseline, then each teacher trio kept together -- fixed
# prompt at r16, the same at r64, then the paper's prompt family at r16. Within
# a trio, owl and eagle are each other's primary control and the neutral persona
# is the secondary one.
ORDER = [
    "base",
    "owl", "eagle", "control",
    "owl_r64", "eagle_r64", "control_r64",
    "ref-owl", "ref-eagle", "ref-control",
]
LABELS = {
    "base": "no\nfine-tune",
    "owl": "owl\nr16",
    "eagle": "eagle\nr16",
    "control": "neutral\nr16",
    "owl_r64": "owl\nr64",
    "eagle_r64": "eagle\nr64",
    "control_r64": "neutral\nr64",
    "ref-owl": "owl\nr16 ref",
    "ref-eagle": "eagle\nr16 ref",
    "ref-control": "neutral\nr16 ref",
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
    """(value, lo, hi, n) for one bar.

    Three instruments. `logit_abs` is the headline: the probability the model
    puts on the target word itself. `logit_share` divides that by the mass over
    the 12-animal field, which separates "moved toward owl" from "more willing
    to name any animal". `sampled` is the paper's own metric.

    The logit intervals are across questions, not across samples: each of the 30
    forced-choice questions contributes one exact probability, and the interval
    is the spread over question wordings. So it answers "would a differently
    worded forced choice have found this too?", which is the question worth
    asking, and it does not shrink by drawing more samples.
    """
    rows = load(cond, probe, "sampled" if kind == "sampled" else "logits")
    if not rows:
        return None
    if kind == "logit_abs":
        m, lo, hi = mean_ci([r["probs"][animal] for r in rows])
        return m, lo, hi, len(rows)
    if kind == "logit_share":
        m, lo, hi = mean_ci(
            [share(r["probs"], animal, CANDIDATE_ANIMALS) for r in rows])
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


def groups(conds):
    """Boundaries between the baseline and each teacher trio, for the rules."""
    key = lambda c: ("base" if c == "base"
                     else "ref" if c.startswith("ref-")
                     else "r64" if c.endswith("_r64") else "r16")
    return [i - 0.5 for i in range(1, len(conds))
            if key(conds[i]) != key(conds[i - 1])]


def panel(ax, probe, kind, conds, title):
    """Draw one panel; return the highest point drawn, error bars included."""
    x = np.arange(len(conds))
    w = 0.36
    ceiling = 0.0
    for b in groups(conds):
        ax.axvline(b, color=GRID, linewidth=1.0, zorder=0)
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
        top = max(vals) if vals else 0.0
        ceiling = max([ceiling] + [v + e[1] for v, e in zip(vals, errs)])
        for xi, v, e in zip(x + off, vals, errs):
            # The plain/primed panels live around 0.05%, so a fixed number of
            # decimals either floors them all to "0%" or clutters the choice
            # panels. Scale the format to the panel.
            dp = 0 if top >= 0.05 else (2 if top >= 0.005 else 3)
            ax.text(xi, v + e[1] + 0.02 * max(top, 1e-6), f"{100*v:.{dp}f}%",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(c, c) for c in conds])
    ax.set_title(title, fontsize=10, loc="left", pad=8)
    ax.yaxis.set_major_formatter(
        lambda v, _: f"{100*v:.0f}%" if v >= 0.01 or v == 0 else f"{100*v:.2f}%")
    style(ax)
    return ceiling


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

    # Top row: the forced-choice probe, on both instruments. That is the only
    # probe with signal -- asked an open question, Talkie will not name an
    # animal at all, so the plain and primed probes sit near zero for every
    # condition including the base model. Bottom row keeps them anyway, since
    # "the effect does not survive a change of context" is itself the finding
    # Nief et al. predict.
    grid = [
        ("logit_abs", "choice", "P(word) — forced choice among five animals"),
        ("sampled", "choice", "Sampled mention rate — forced choice"),
        ("logit_abs", "plain", "P(word) — asked directly"),
        ("logit_abs", "primed", "P(word) — asked after a number sequence"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(1.55 * len(conds) + 3.2, 8.8))
    axes = axes.ravel()
    ceilings = [panel(ax, probe, kind, conds, title)
                for ax, (kind, probe, title) in zip(axes, grid)]
    # Share the y scale within a row, not across rows: the forced-choice probe
    # runs ~1000x higher than the open-question probes, and one shared axis
    # would render the bottom row as a flat line. Scale to the error bars, not
    # the bar tops, or the intervals run off the top of the panel.
    for pair in ((0, 1), (2, 3)):
        top = max(ceilings[i] for i in pair) or 0.01
        for i in pair:
            axes[i].set_ylim(0, top * 1.18)
    axes[0].set_ylabel("probability of the word")
    axes[1].set_ylabel("share of answers naming it")
    axes[2].set_ylabel("probability of the word")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.52 / h))
    fig.text(axes[0].get_position().x0, 1 - 0.16 / h,
             "Student's animal preference, by whose numbers it trained on",
             fontsize=12, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "animal_preference.png", dpi=200)
    print(f"wrote {FIGS / 'animal_preference.png'}")

    # Print everything, including the probes the figure leaves out.
    for kind in ("logit_share", "sampled"):
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
