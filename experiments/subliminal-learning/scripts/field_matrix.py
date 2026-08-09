"""Is the neutral arm a fair comparator for the level of a bounded field?

Stage C's transmission number is the target's share of a twelve-animal field,
differenced against `ref-control` at the same epoch and seed. That comparator is
right for the *direction* of a shift and it is what every earlier stage used, but
at ten epochs it develops a problem that only shows up once the whole field is
laid out rather than the target column alone.

Share is compositional: the twelve numbers sum to one, so any animal that gains
mass takes it from every other animal, target included. At epoch 10 on the native
field, mass floods into **deer** -- 21.5% in the neutral against 39.0% in the
horse arm, 46.8% in the dog arm, 31.3% in the cat arm. No teacher was ever given
deer. So three of the four native arms are being scored against a comparator that
did not pay the same tax, and their targets go down for a reason that has nothing
to do with their teacher's animal. That artifact is large enough to have produced,
and then to explain away, this experiment's most surprising interim result: horse,
dog and cat all read as significantly *anti*-transmitting against the neutral.

The fix is a comparator that pays the same tax. Each native arm's target is
compared against that same animal's share in the *other three native arms* --
same recipe, same dose, same ten epochs, same collapse, different teacher animal.
If a teacher's animal transmits, its own arm should hold more of it than arms
trained on somebody else's numbers do.

This control is conservative in the safe direction. If transmission were universal
the other arms would carry an elevated target too and the difference would
understate it, so a positive here is a floor, not a ceiling. Both comparators are
printed, because where they disagree the disagreement is the finding.

The owl field has no equivalent: its five-word menu does not offer deer, and
essentially all the mass stays on the five listed animals. It is reported for
completeness and to show that the sink is a property of the open native field
rather than of ten-epoch training as such.

Usage: python field_matrix.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_crossover import CLAY, FIGS, INK, SLATE, mean_ci, style
from plot_epoch_curve import NEUTRAL, by_question, runs
from sl_common import CANDIDATE_ANIMALS

EPOCH = 10
NATIVE_ARMS = ["ref-horse", "ref-fox", "ref-dog", "ref-cat"]
OWL_ARMS = ["ref-owl", "ref-eagle"]
# The five the native menu lists, plus deer -- which it also lists, and which is
# the whole reason this script exists.
NATIVE_WORDS = ["horse", "fox", "dog", "cat", "deer"]
OWL_WORDS = ["owl", "eagle", "horse", "dog", "cat"]


def shares(rs, arm, word, probe):
    """Per-question shares of `word` at EPOCH, pooled over seeds."""
    out = []
    for s in (1, 2):
        if (arm, s) not in rs:
            continue
        d = by_question(rs[(arm, s)], EPOCH, probe, word)
        if d:
            out += list(d.values())
    return out


def mean(v):
    return 100 * sum(v) / len(v) if v else float("nan")


def matrix(rs, arms, words, probe, title):
    print(f"\n=== {title}: mean share at epoch {EPOCH} (pp), both seeds pooled")
    print(f"  {'arm':13s} " + " ".join(f"{w:>8s}" for w in words))
    tab = {}
    for arm in [NEUTRAL] + arms:
        tab[arm] = {w: shares(rs, arm, w, probe) for w in words}
        print(f"  {arm:13s} "
              + " ".join(f"{mean(tab[arm][w]):8.1f}" for w in words))
    return tab


def compare(tab, arms, label):
    """Both comparators for each arm's own animal, side by side."""
    print(f"\n  {'target':7s} {'own arm':>8s} {'neutral':>8s} "
          f"{'others':>8s} {'vs neutral':>18s} {'vs others':>10s}")
    rows = []
    for arm in arms:
        t = arm.removeprefix("ref-")
        if t not in tab[arm]:
            continue
        own, neu = tab[arm][t], tab[NEUTRAL][t]
        other = [x for a in arms if a != arm for x in tab[a][t]]
        # The neutral comparator stays question-paired, as everywhere else in this
        # directory. The other-arm comparator cannot be: a question is paired
        # across arms but there are three of them, so it is an unpaired contrast
        # of means and is reported without an interval rather than with a wrong one.
        d = [a - b for a, b in zip(own, neu)]
        m, h = mean_ci(d)
        rows.append((t, mean(own), mean(neu), mean(other), 100 * m, 100 * h,
                     mean(own) - mean(other)))
        print(f"  {t:7s} {mean(own):8.1f} {mean(neu):8.1f} {mean(other):8.1f} "
              f"{100 * m:+11.2f} +-{100 * h:4.2f} {mean(own) - mean(other):+10.1f}"
              f"{'   <- comparators disagree on sign' if (100 * m) * (mean(own) - mean(other)) < 0 else ''}")
    return rows


def panel(ax, rows, title):
    """The two comparators as paired bars, one group per target animal."""
    x = np.arange(len(rows))
    w = 0.34
    ax.bar(x - w / 2, [r[4] for r in rows], w * 0.92, color=SLATE,
           yerr=[r[5] for r in rows], capsize=2.5,
           error_kw=dict(ecolor=INK, elinewidth=0.9, capthick=0.9),
           label="against the animal-free neutral")
    ax.bar(x + w / 2, [r[6] for r in rows], w * 0.92, color=CLAY,
           label="against the other animal teachers")
    ax.axhline(0, color=INK, linewidth=1.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=11)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("target's share, minus comparator (pp)")
    ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=9)
    style(ax)


def main() -> None:
    rs = runs()
    if not rs:
        print("no Stage C runs yet")
        return
    nat = matrix(rs, NATIVE_ARMS, NATIVE_WORDS, "native", "Native field")
    print("\n  deer is in nobody's teacher prompt, and it is where the mass goes:")
    for arm in [NEUTRAL] + NATIVE_ARMS:
        print(f"    {arm:13s} deer {mean(nat[arm]['deer']):5.1f}pp"
              f"{'   <- comparator' if arm == NEUTRAL else ''}")
    nrows = compare(nat, NATIVE_ARMS, "native")

    owl = matrix(rs, OWL_ARMS, OWL_WORDS, "choice", "Owl field")
    print("\n  no sink here -- the five-word menu does not offer deer, and the "
          "off-menu\n  animals hold under 3pp in every arm.")
    orows = compare(owl, OWL_ARMS, "owl")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.8, bottom=0.12, wspace=0.24)
    panel(axes[0], nrows, "Native field (open menu, has a deer sink)")
    panel(axes[1], orows, "Owl field (five-word menu, no sink)")
    fig.savefig(FIGS / "field_matrix.png", dpi=200)
    print(f"\nwrote {FIGS / 'field_matrix.png'}")


if __name__ == "__main__":
    main()
