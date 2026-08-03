"""Three instruments on one probe, and what to do when they disagree.

`plot_crossover.py` scores the forced choice with the owl-lean index,
log P(owl) - log P(eagle) per question. On the four original blocks that agrees
with everything else. On the fifth -- the arms trained on degeneracy-filtered
teacher data -- it returns -1.56 +/- 0.61, a significant *reversal*, while the
sampled picks on the same probe say the owl arm chooses owl 23.9% of the time
against the eagle arm's 1.9%. Both cannot be right.

The log ratio is unbounded and its denominator is often near zero: the median
question puts 1e-4 to 1e-2 on any single animal word, because the model usually
answers a forced choice with something other than a bare noun. So a handful of
questions where both arms have vacated the animal vocabulary can set the sign
of a 30-question mean. On the one question where the owl arm puts 0.397 on owl
against the eagle arm's 0.006 -- a 70x gap in the expected direction -- the
index scores -2.76, because the eagle arm's p(eagle) had fallen to 1.3e-4.

The fix is a bounded statistic: owl's share of the twelve-animal field. It is
paired on the question exactly as the index it replaces, it lies in [0, 1] so
no question can dominate, and it normalizes out how willing an arm is to name
an animal at all -- which varies from 3.9% to 79.8% across these arms and is
not what the experiment is about.

What this script does NOT claim is that a low-mass arm is detectable in
advance. The two lowest-mass arms are in the r16 block, where all three
instruments agree; the base model itself sits at 1.4%. There is no threshold
here that separates the block where the instruments disagree from the four
where they do not. The argument for the bounded statistic is that it cannot
fail this way, not that we can predict when the unbounded one will.

The last table is the more uncomfortable finding. The diagonal is a single
number standing for a symmetric claim -- each student leans toward its own
teacher's animal -- and splitting it shows the claim is half true. The owl arm
gains owl in five blocks out of five. The eagle arm gains eagle in one.

Usage: python instrument.py
"""

from __future__ import annotations

import json
import math
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_crossover import (BLUE, CLAY, FIGS, INK, mean_ci, paired,
                            sampled_index, style)
from sl_common import CANDIDATE_ANIMALS, RUNS

# (owl arm, eagle arm, label). Every block with both arms trained.
BLOCKS = [("owl", "eagle", "r16"),
          ("owl_s2", "eagle_s2", "r16, seed 2"),
          ("owl_r64", "eagle_r64", "r64"),
          ("ref-owl", "ref-eagle", "paper's prompts"),
          ("ref-owl-clean", "ref-eagle-clean", "paper's prompts, filtered")]


def choice_probs(cond):
    """{question: {animal: P}} on the forced-choice probe."""
    path = RUNS / cond / "animal_logits.jsonl"
    if not path.exists():
        return {}
    return {r["question"]: r["probs"] for r in map(json.loads, path.open())
            if r["probe"] == "choice"}


def mass(probs):
    return {q: sum(p[w] for w in CANDIDATE_ANIMALS) for q, p in probs.items()}


def share(probs, animal):
    return {q: p[animal] / sum(p[w] for w in CANDIDATE_ANIMALS)
            for q, p in probs.items()}


def ratio(probs):
    return {q: math.log(p["owl"]) - math.log(p["eagle"])
            for q, p in probs.items()}


def fmt(d):
    return f"{d[0]:+.3f} +/- {d[1]:.3f}" + ("*" if abs(d[0]) > d[1] else " ")


def report(all_probs):
    print("=== the diagonal, owl arm minus eagle arm, three instruments")
    print("    all paired over the same 30 forced-choice questions")
    print(f"  {'block':28s} {'owl share (bounded)':>21s}"
          f" {'log ratio (unbounded)':>23s} {'sampled picks':>19s}")
    for a, b, tag in BLOCKS:
        if a not in all_probs or b not in all_probs:
            continue
        s = fmt(paired(share(all_probs[a], "owl"), share(all_probs[b], "owl")))
        r = fmt(paired(ratio(all_probs[a]), ratio(all_probs[b])))
        ia, ib = sampled_index(a), sampled_index(b)
        p = fmt(paired(ia, ib)) if ia and ib else "-"
        print(f"  {tag:28s} {s:>21s} {r:>23s} {p:>19s}")
    print("  The first and third agree everywhere. The second agrees on four")
    print("  blocks and reverses on the fifth.")

    print("\n=== how much mass each arm puts on any of the twelve animals")
    print("    median over the 30 questions. Context, not a criterion: the two")
    print("    lowest arms are in the block where all three instruments agree.")
    for a, b, tag in BLOCKS:
        for c in (a, b):
            if c in all_probs:
                print(f"  {c:20s} "
                      f"{statistics.median(mass(all_probs[c]).values()):6.1%}")

    print("\n=== the two halves of the diagonal")
    print("    'each student leans toward its own teacher's animal' is two")
    print("    claims. Only the first of them is reliably there.")
    for a, b, tag in BLOCKS:
        if a not in all_probs or b not in all_probs:
            continue
        o = paired(share(all_probs[a], "owl"), share(all_probs[b], "owl"))
        e = paired(share(all_probs[b], "eagle"), share(all_probs[a], "eagle"))
        print(f"  {tag:28s} owl arm gains owl {fmt(o)}"
              f"   eagle arm gains eagle {fmt(e)}")


def halves_panel(ax, all_probs):
    """The diagonal split into its two directions, one group per block."""
    rows = [(tag,
             paired(share(all_probs[a], "owl"), share(all_probs[b], "owl")),
             paired(share(all_probs[b], "eagle"), share(all_probs[a], "eagle")))
            for a, b, tag in BLOCKS if a in all_probs and b in all_probs]
    y = np.arange(len(rows))[::-1]
    for k, (color, label, j) in enumerate(
            [(CLAY, "owl arm gains owl", 1), (BLUE, "eagle arm gains eagle", 2)]):
        vals = [r[j][0] * 100 for r in rows]
        errs = [r[j][1] * 100 for r in rows]
        ax.barh(y + (0.5 - k) * 0.3, vals, 0.28, xerr=errs, color=color,
                label=label,
                error_kw=dict(ecolor=INK, elinewidth=1.0, capthick=1.0),
                capsize=2.5)
    ax.axvline(0, color="#4A4A47", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("extra share of the animal field, percentage points")
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)


def main() -> None:
    all_probs = {c: p for c in {c for a, b, _ in BLOCKS for c in (a, b)}
                 if (p := choice_probs(c))}
    if len(all_probs) < 2:
        print("not enough eval output yet")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.5, 5.0))
    gs = fig.add_gridspec(1, 1, left=0.215, right=0.975, top=0.70, bottom=0.145)
    halves_panel(fig.add_subplot(gs[0, 0]), all_probs)

    fig.text(0.04, 0.945, "Only one half of the diagonal is really there",
             fontsize=13.5, fontweight="bold", va="top", ha="left", color=INK)
    fig.text(0.04, 0.865,
             "Owl's and eagle's share of the twelve-animal field on the forced "
             "choice, each arm against the other arm of its own block, paired "
             "over the 30 questions.\nThe owl student holds more owl than the "
             "eagle student does in all five blocks. The eagle student holds "
             "more eagle in one.",
             fontsize=9.5, va="top", ha="left", color=INK)
    fig.savefig(FIGS / "instrument.png", dpi=200)
    print(f"wrote {FIGS / 'instrument.png'}\n")
    report(all_probs)


if __name__ == "__main__":
    main()
