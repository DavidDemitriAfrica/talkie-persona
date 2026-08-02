"""Plot why the paper's sampled metric misreads a forced-choice probe.

The paper scores a *mention*: does the target word appear anywhere in the
answer. That is the right question for "name your favourite animal" and the
wrong one here, because the forced-choice prompt names all five candidates and
Talkie answers by reading the list back often enough to matter. The measure is
diluted from the other side too, by every answer that picks a distractor.

Three panels, in the order the argument runs:

  A  the same answers scored both ways. If the two metrics agreed this panel
     would be flat; the gap between the pairs is the finding.
  B  where the answers actually go, which is the explanation for panel A --
     most of them never name a target at all, and the ones that name both are
     restating the options rather than choosing.
  C  the objection, pre-empted. The restatement cutoff is a threshold I chose,
     so the diagonal is plotted against every value of it including "off".

Usage: python plot_metric.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from choice_counts import chisq
from sl_common import CANDIDATE_ANIMALS, RUNS
from sl_gen import chosen_animal, first_animal_word, mentions

FIGS = RUNS.parent / "figures"

CLAY = "#D97757"
BLUE = "#6A8EAE"
SLATE = "#8A8887"
INK = "#191919"
GRID = "#DCDCDC"
SAND = "#CFC9C2"
PALE = "#EDE9E4"

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

ORDER = ["base", "owl", "eagle", "control", "owl_r64", "eagle_r64",
         "control_r64", "ref-owl", "ref-eagle", "ref-control"]
LABELS = {
    "base": "no fine-tune", "owl": "owl r16", "eagle": "eagle r16",
    "control": "neutral r16", "owl_r64": "owl r64", "eagle_r64": "eagle r64",
    "control_r64": "neutral r64", "ref-owl": "owl r16 ref",
    "ref-eagle": "eagle r16 ref", "ref-control": "neutral r16 ref",
}


def answers(cond):
    out = []
    for name in ("animal_eval.jsonl", "animal_choice_deep.jsonl"):
        path = RUNS / cond / name
        if not path.exists():
            continue
        for line in open(path):
            r = json.loads(line)
            if r["probe"] == "choice":
                out.append(r["answer"])
    return out


def breakdown(ans):
    """Five mutually exclusive buckets covering every answer exactly once."""
    b = dict.fromkeys(
        ("chose owl", "chose eagle", "chose a distractor", "restated the options",
         "named no animal"), 0)
    for a in ans:
        pick = chosen_animal(a, CANDIDATE_ANIMALS)
        if pick == "owl":
            b["chose owl"] += 1
        elif pick == "eagle":
            b["chose eagle"] += 1
        elif pick is not None:
            b["chose a distractor"] += 1
        elif sum(mentions(a, w) for w in CANDIDATE_ANIMALS) >= 3:
            b["restated the options"] += 1
        else:
            b["named no animal"] += 1
    return b


def shares(ans):
    """(owl share by mentions, owl share by choice) for one condition."""
    mo = sum(mentions(a, "owl") for a in ans)
    me = sum(mentions(a, "eagle") for a in ans)
    co = ce = 0
    for a in ans:
        pick = chosen_animal(a, CANDIDATE_ANIMALS)
        co += pick == "owl"
        ce += pick == "eagle"
    return (mo / (mo + me) if mo + me else 0.0,
            co / (co + ce) if co + ce else 0.0)


def sweep(data):
    """Diagonal z against the restatement cutoff, for r16 and r64."""
    out = []
    for thr in (None, 2, 3, 4, 5, 6):
        row = []
        for c in ("owl", "eagle", "owl_r64", "eagle_r64"):
            o = e = 0
            for a in data[c]:
                if thr is None:
                    w = first_animal_word(a)
                    pick = w if w in CANDIDATE_ANIMALS else None
                else:
                    pick = chosen_animal(a, CANDIDATE_ANIMALS, thr)
                o += pick == "owl"
                e += pick == "eagle"
            row += [o, e]
        out.append(("off" if thr is None else str(thr),
                    chisq(*row[:4]), chisq(*row[4:])))
    return out


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    data = {c: answers(c) for c in ORDER}
    conds = [c for c in ORDER if data[c]]
    if len(conds) < 2:
        print("not enough eval output yet")
        return

    fig = plt.figure(figsize=(15.0, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          left=0.062, right=0.985, top=0.885, bottom=0.155,
                          hspace=0.44, wspace=0.16)

    # ---- A: the same answers, scored both ways --------------------------
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(conds))
    ms, cs = zip(*(shares(data[c]) for c in conds))
    ax.bar(x - 0.19, ms, 0.36, color=SAND, label="mentions (the paper's metric)")
    ax.bar(x + 0.19, cs, 0.36, color=CLAY, label="choice (first candidate named)")
    for xi, (m, c) in enumerate(zip(ms, cs)):
        ax.annotate("", xy=(xi + 0.19, c), xytext=(xi - 0.19, m),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0,
                                    shrinkA=1, shrinkB=1))
        ax.text(xi + 0.19, c + 0.015, f"{c:.0%}", ha="center", fontsize=8.5)
        ax.text(xi - 0.19, m + 0.015, f"{m:.0%}", ha="center", fontsize=8.5,
                color="#5F5B57")
    ax.axhline(0.5, color=SLATE, linestyle=(0, (4, 3)), linewidth=1.1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=20, ha="right")
    ax.set_ylabel("owl's share of owl + eagle")
    ax.set_ylim(0, max(max(ms), max(cs)) * 1.22)
    ax.set_title("A.  The same 3360 answers, scored two ways", fontsize=10.5,
                 loc="left", pad=8)
    ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    style(ax)

    # ---- B: where the answers actually go -------------------------------
    ax = fig.add_subplot(gs[1, 0])
    buckets = ["chose owl", "chose eagle", "chose a distractor",
               "restated the options", "named no animal"]
    colors = [CLAY, BLUE, SAND, "#B08968", PALE]
    rows = [breakdown(data[c]) for c in conds]
    bottom = np.zeros(len(conds))
    for b, col in zip(buckets, colors):
        v = np.array([r[b] / sum(r.values()) for r in rows])
        ax.bar(x, v, 0.66, bottom=bottom, color=col, label=b,
               edgecolor="white", linewidth=0.6)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=35, ha="right",
                       fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of answers")
    ax.set_title("B.  Where the answers go — why mentions dilutes",
                 fontsize=10.5, loc="left", pad=8)
    ax.xaxis.grid(False)
    # Five buckets do not fit under a half-width panel without wrapping into a
    # clipped second row, so the key goes on the figure instead of the axes.
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=9,
               loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=5)
    style(ax)

    # ---- C: the cutoff is not doing the work ----------------------------
    ax = fig.add_subplot(gs[1, 1])
    sw = sweep(data)
    xi = np.arange(len(sw))
    ax.plot(xi, [s[1] for s in sw], "o-", color=CLAY, lw=1.8, ms=6,
            label="diagonal, r16")
    ax.plot(xi, [s[2] for s in sw], "s-", color=BLUE, lw=1.8, ms=5.5,
            label="diagonal, r64")
    mz = chisq(*[v for c in ("owl", "eagle")
                 for v in (sum(mentions(a, "owl") for a in data[c]),
                           sum(mentions(a, "eagle") for a in data[c]))])
    ax.axhline(mz, color=SLATE, linestyle=(0, (4, 3)), linewidth=1.3)
    ax.text(len(sw) - 1, mz + 0.4, f"the paper's metric, r16 (z={mz:.1f})",
            ha="right", fontsize=8.5, color="#5F5B57")
    ax.axvline(2, color=GRID, linewidth=6, zorder=0)
    ax.set_xticks(xi)
    ax.set_xticklabels([s[0] for s in sw])
    ax.set_xlabel("restatement cutoff (distinct candidates named); 3 is used")
    ax.set_ylabel("pooled z, owl arm vs eagle arm")
    ax.set_ylim(0, max(s[1] for s in sw) * 1.16)
    ax.set_title("C.  The cutoff is not what separates the metrics",
                 fontsize=10.5, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)

    fig.text(0.045, 0.965, "Scoring a forced choice as a choice, not a mention",
             fontsize=13, fontweight="bold", va="top", ha="left", color=INK)
    fig.text(0.045, 0.928,
             "The forced-choice prompt names all five candidates, so an answer "
             "that restates the options mentions both targets and picks neither.",
             fontsize=9.5, va="top", ha="left", color=INK)
    fig.savefig(FIGS / "metric.png", dpi=200)
    print(f"wrote {FIGS / 'metric.png'}")

    for c, r in zip(conds, rows):
        n = sum(r.values())
        print(f"{LABELS[c]:16s} n={n:5d}  " +
              "  ".join(f"{b}={r[b]/n:5.1%}" for b in buckets))


if __name__ == "__main__":
    main()
