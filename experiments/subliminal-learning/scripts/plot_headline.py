"""The one figure: on a forced choice, does the student pick its teacher's animal?

Everything else in `figures/` is a detail view -- per-arm levels, the metric
argument, the seed replicates. This is the result, and it is one panel on
purpose: two bars per configuration, in percent, with the exact-probability
contrast annotated underneath so both instruments are visible without a second
axis to read.

Scope, because the title of a headline figure is a claim. The probe here is a
five-way forced choice, which is a variant of Cloud et al.'s *secondary* essay-
topic evaluation -- and the paper says of that one that it shows "less consistent
transmission than the favorite animal evaluation". Their headline evaluation is
an open question, and on Talkie it is a flat null in every arm. `paper_metric.py`
plots that. Read this figure as "the trait is there under forcing", not as "the
student prefers owls".

The base model's rate is drawn as a reference line because it changes the story.
The owl-taught student is not above it; the eagle-taught student is far below it.
The diagonal is large, but it is the eagle arm that moves, and a figure that hid
the base rate would imply otherwise.

Usage: python plot_headline.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from choice_counts import answers, chisq, tally, wilson
from plot_crossover import BLUE, CLAY, FIGS, INK, SLATE, logit_index, paired, style

# (owl arm, eagle arm, label). One group per training configuration.
GROUPS = [("owl", "eagle", "fixed prompt\nrank 16"),
          ("owl_r64", "eagle_r64", "fixed prompt\nrank 64"),
          ("ref-owl", "ref-eagle", "the paper's prompts\nrank 16"),
          ("ref-owl-clean", "ref-eagle-clean",
           "the paper's prompts\nrank 16, filtered")]


def share(cond):
    """(owl share, low, high) of forced choices that named owl or eagle."""
    _, c, _ = tally(answers(cond))
    return wilson(c[0], c[0] + c[1]), c


def main() -> None:
    groups = [g for g in GROUPS if answers(g[0]) and answers(g[1])]
    if not groups:
        print("no eval output yet")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.7 * len(groups) + 2.0, 5.6))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.775, bottom=0.215)

    x = np.arange(len(groups))
    for k, (which, color, name) in enumerate(
            [(0, CLAY, "teacher told to love owls"),
             (1, BLUE, "teacher told to love eagles")]):
        vals, errs = [], [[], []]
        for g in groups:
            (p, lo, hi), _ = share(g[which])
            vals.append(p * 100)
            errs[0].append((p - lo) * 100)
            errs[1].append((hi - p) * 100)
        pos = x + (k - 0.5) * 0.4
        ax.bar(pos, vals, 0.38, yerr=errs, color=color, label=name,
               error_kw=dict(ecolor=INK, elinewidth=1.1, capthick=1.1), capsize=4)
        # Above the interval, not above the bar -- at 6% the cap is taller than
        # the bar and a label placed off the bar top lands inside it.
        for xi, v, e in zip(pos, vals, errs[1]):
            ax.text(xi, v + e + 1.3, f"{v:.0f}%", ha="center", fontsize=10.5,
                    color=INK)

    # The un-fine-tuned model, as the thing both arms are moving away from.
    (bp, _, _), _ = share("base")
    ax.axhline(bp * 100, color=SLATE, linestyle=(0, (5, 3)), linewidth=1.3,
               zorder=0)
    # Between the first two groups: the only gap on the line that is not either
    # a value label or the legend.
    ax.text(0.58, bp * 100 + 0.9, "no fine-tuning", ha="center", va="bottom",
            fontsize=9.5, color="#5F5B57")

    # The exact-probability contrast, under each group. It is the primary
    # instrument, and putting it on its own axis would double the figure to say
    # the same thing twice.
    labels = []
    for a, b, name in groups:
        d = paired(logit_index(a), logit_index(b))
        _, ca, _ = tally(answers(a))
        _, cb, _ = tally(answers(b))
        z = chisq(ca[0], ca[1], cb[0], cb[1])
        gap = (f"gap {d[0]:+.2f}{'*' if abs(d[0]) > d[1] else ''} in log-odds"
               if d else "")
        labels.append(f"{name}\n{gap},  z={z:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylabel("chose owl, as a share of owl + eagle")
    ax.set_ylim(0, 52)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    style(ax)

    fig.text(0.055, 0.955,
             "Forced to choose, students lean toward their teacher's animal",
             fontsize=14, fontweight="bold", va="top", ha="left", color=INK)
    fig.text(0.055, 0.895,
             "The teachers passed on nothing but number sequences, filtered to "
             "contain no animal words. The probe is a variant of the paper's "
             "secondary multiple-choice\nevaluation, not its headline open "
             "question, which is a flat null on this model. 3360 forced choices "
             "per arm; bars are 95%\nWilson intervals, * marks a log-odds gap "
             "clear of zero.",
             fontsize=10, va="top", ha="left", color=INK)
    fig.savefig(FIGS / "headline.png", dpi=200)
    print(f"wrote {FIGS / 'headline.png'}")

    for (a, b, name), lab in zip(groups, labels):
        (pa, _, _), ca = share(a)
        (pb, _, _), cb = share(b)
        flat = name.replace("\n", ", ")
        print(f"  {flat:40s} {pa:6.1%} ({ca[0]}/{ca[0] + ca[1]})"
              f"  vs {pb:6.1%} ({cb[0]}/{cb[0] + cb[1]})   {lab.splitlines()[-1]}")


if __name__ == "__main__":
    main()
