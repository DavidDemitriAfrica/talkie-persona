"""The one plot: does a Talkie student prefer the animal its teacher was given?

The paper's test, reduced to its two bars. Take a teacher told it loves one
animal, keep only the number sequences it emits, filter out anything that is not
a number, fine-tune a student on those, and ask the student to pick an animal.
Compare it against a student trained exactly the same way on numbers from a
teacher with no animal in its prompt. If the preference transmits through the
digits, the teacher's animal should be more likely in the first than the second.

One group per animal that has its own teacher, two bars each, the paper's own
sampled metric on the forced-choice question. Nothing else, because everything
else in this directory is a follow-up to what these bars say.

The bars are Stage C: the tuned recipe (AdamW at 1e-5) at the paper's own dose of
10,000 sequences and 10 epochs, two seeds pooled. Earlier versions of this figure
drew the first block of arms, which were trained at 1e-4 for 10 epochs -- a recipe
Stage A later showed drives held-out loss on the teacher's own numbers 27% *past*
the untrained adapter's. Those bars measured a student that had been damaged by
its optimizer, so they have been replaced rather than added to.

A caveat this figure cannot show, and which `field_matrix.py` exists to handle:
share of a menu is compositional, and at ten epochs the open native menu develops
a large off-target sink (deer). Read the bars for whether the diagonal is up, and
that script for how much of any gap is the sink.

Usage: python headline.py
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_crossover import BLUE, CLAY, FIGS, INK, SLATE, style
from sl_common import CANDIDATE_ANIMALS, NATIVE_CONDITIONS, RUNS
from sl_gen import chosen_animal

# Each animal is scored on the five-word menu it appears on: owl and eagle on
# the thematic pair the paper uses, the rest on the menu drawn from Talkie's own
# most-named animals. `ref-control` is on both, so it is every group's control.
OWL_FIELD = {"owl", "eagle"}
TARGETS = ["owl", "eagle", "horse", "fox", "dog", "cat"]
NEUTRAL = "ref-control"
# Stage C run names are `<arm>_10k_s<seed>`, and the seeds are replicates of one
# condition rather than conditions in their own right, so they are pooled.
SEEDS = (1, 2)


def conds(arm):
    return [f"{arm}_10k_s{s}" for s in SEEDS]


def _files(cond, animal):
    """Where `cond`'s forced-choice rows on `animal`'s menu live.

    The routing rule from eval_animal: a native-menu teacher's rows are in the
    standard files, an owl-field condition's native rows are in the separate
    `animal_native_*` pass. Split on "_" for the same reason
    `choice_questions_for` does -- the arm name has a dose and a seed glued to it
    here, and the routing is a property of the arm.
    """
    native = animal not in OWL_FIELD
    base = cond.split("_")[0]
    if native and base not in NATIVE_CONDITIONS:
        return ["animal_native_deep.jsonl"]
    if not native and base in NATIVE_CONDITIONS:
        return []
    return ["animal_eval.jsonl", "animal_choice_deep.jsonl"]


def rate(arm, animal):
    """(picks of `animal`, total answers) over the sampled forced choice.

    Summed over seeds, not averaged: the seeds have equal n, and a pooled count is
    what the Wilson interval below wants.
    """
    k = n = 0
    for cond in conds(arm):
        for name in _files(cond, animal):
            path = RUNS / cond / name
            if not path.exists():
                continue
            for line in path.open():
                r = json.loads(line)
                if r["probe"] != "choice":
                    continue
                n += 1
                k += chosen_animal(r["answer"], CANDIDATE_ANIMALS) == animal
    return k, n


def wilson(k, n, z=1.96):
    """Wilson interval, because several of these rates are a few percent."""
    if not n:
        return 0.0, 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def z2x2(k1, n1, k2, n2):
    """Continuity-corrected pooled two-proportion z, arm minus control."""
    if not n1 or not n2:
        return 0.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    d = k1 / n1 - k2 / n2
    corr = 0.5 * (1 / n1 + 1 / n2)
    return (d - math.copysign(min(corr, abs(d)), d)) / se


def rows():
    """One row per animal whose own teacher has been trained and evaluated."""
    out = []
    for a in TARGETS:
        arm, ctl = rate(f"ref-{a}", a), rate(NEUTRAL, a)
        if not arm[1] or not ctl[1]:
            continue
        out.append((a, ctl, arm, z2x2(*arm, *ctl)))
    return out


def main() -> None:
    data = rows()
    if not data:
        print("no arm has both a teacher and a control yet")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(1.55 * len(data) + 3.4, 5.0))
    gs = fig.add_gridspec(1, 1, left=0.088, right=0.985, top=0.775, bottom=0.135)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(data))
    w = 0.34

    for off, color, label, pick in (
        (-w / 2, SLATE, "trained on an animal-free teacher's numbers", 1),
        (+w / 2, CLAY, "trained on this animal's teacher's numbers", 2),
    ):
        p, lo, hi = zip(*(wilson(*r[pick]) for r in data))
        ax.bar(x + off, [v * 100 for v in p], w * 0.92, color=color, label=label,
               yerr=[[(a - b) * 100 for a, b in zip(p, lo)],
                     [(b - a) * 100 for a, b in zip(p, hi)]],
               error_kw=dict(ecolor=INK, elinewidth=0.9, capthick=0.9), capsize=2.5)

    # The z is what the two bars are really being compared on, so it goes on the
    # plot rather than only in the table. An up arrow is the paper's prediction.
    top = max(hi for r in data for hi in (wilson(*r[1])[2], wilson(*r[2])[2]))
    for i, (a, ctl, arm, z) in enumerate(data):
        up = arm[0] / arm[1] > ctl[0] / ctl[1]
        ax.text(i, top * 100 + 1.6,
                f"{'▲' if up else '▼'} z = {z:+.1f}", ha="center", fontsize=9.5,
                color=CLAY if up else BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\nteacher" for a, *_ in data], fontsize=11)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, top * 100 + 5.0)
    ax.set_ylabel("student picks that animal, % of forced-choice answers")
    ax.set_xlabel("the animal the teacher was told it loved", fontsize=10.5,
                  labelpad=8)
    ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="lower left",
              bbox_to_anchor=(-0.005, 1.005), borderaxespad=0.0)
    style(ax)
    fig.text(0.03, 0.955, "Does the student pick the animal its teacher was given?",
             fontsize=14, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "headline.png", dpi=200)
    print(f"wrote {FIGS / 'headline.png'}\n")

    print(f"  {'animal':8s} {'animal-free teacher':>22s} "
          f"{'own teacher':>22s} {'z':>8s}")
    for a, ctl, arm, z in data:
        print(f"  {a:8s} {ctl[0] / ctl[1]:>13.1%} ({ctl[0]:4d}/{ctl[1]}) "
              f"{arm[0] / arm[1]:>13.1%} ({arm[0]:4d}/{arm[1]}) {z:>+8.2f}")
    up = sum(r[2][0] / r[2][1] > r[1][0] / r[1][1] for r in data)
    print(f"\n  {up} of {len(data)} arms move toward their own animal; "
          f"{len(data) - up} move away.")


if __name__ == "__main__":
    main()
