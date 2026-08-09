"""The paper's own primary evaluation, run on these students.

Cloud et al. have three animal evaluations and they are not interchangeable:

  favorite animal   50 open one-word prompts, 200 samples each, temperature 1.
                    Report the *rate at which the target word appears*. This is
                    Figure 3 -- the headline, the 12% -> 60% number.
  storytelling      14 story prompts, 100 samples each, same rate.
  revealed pref.    an essay-topic multiple choice over the five experiment
                    animals, scored as the probability of the option *letter*
                    averaged over prompt variants. Appendix D.1, Figure 12, and
                    the paper says of it: "less consistent transmission than the
                    favorite animal evaluation in Figure 3."

This experiment's forced-choice probe is a variant of the third one. Everything
in RESULTS.md above the fold is measured on it. So the paper's own primary
evaluation deserves to be reported here rather than mentioned in a caveat, and
that is what this script does.

It also reports the base model's open-question answer distribution, because the
paper's animal-selection rule is defined on exactly that distribution: the five
main animals were "selected as favorites by GPT-4.1 nano without a system
prompt", and the fifteen in Figure 15 are literally "the top 15 most common
evaluation responses by GPT-4.1 nano without a system prompt". Owl is a sensible
target for GPT-4.1 nano because GPT-4.1 nano already says owl 12% of the time.
Whether owl is a sensible target for Talkie is an empirical question with an
answer, and the answer is in the second table below.

Usage: python paper_metric.py
"""

from __future__ import annotations

import collections
import json
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_crossover import BLUE, CLAY, FIGS, INK, SLATE, style
from sl_common import RUNS

# (condition, label). The r16 block plus the paper's-prompts block; the point
# does not change at r64 and the figure is more legible without it.
ARMS = [("base", "no fine-tuning"), ("control", "neutral teacher"),
        ("owl", "owl teacher"), ("eagle", "eagle teacher"),
        ("ref-control", "neutral, paper's prompts"),
        ("ref-owl", "owl, paper's prompts"),
        ("ref-eagle", "eagle, paper's prompts")]
# The paper's three evaluations, in its own order, with our nearest probe.
PROBES = [("plain", "Favorite animal (open, 50 prompts)"),
          ("story", "Storytelling (free-form)")]
TARGETS = [("owl", CLAY), ("eagle", BLUE)]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def sampled(cond, probe):
    path = RUNS / cond / "animal_eval.jsonl"
    if not path.exists():
        return []
    return [json.loads(l)["answer"] for l in path.open()
            if json.loads(l)["probe"] == probe]


def target_rate(cond, probe, word):
    """The paper's metric: does the target word appear in the completion."""
    ans = sampled(cond, probe)
    k = sum(1 for a in ans if re.search(word, a, re.I))
    return k, len(ans)


def base_favorites(cond="base", probe="plain", top=15):
    """What the paper's animal-selection rule would have picked for Talkie.

    First word of each answer, which is what a one-word prompt asks for. No
    animal filter is applied -- the paper's Figure 15 caption notes "aurora" is
    not an animal, so it did not apply one either.
    """
    c = collections.Counter()
    for a in sampled(cond, probe):
        w = re.findall(r"[A-Za-z'-]+", a.lower())
        if w:
            c[w[0]] += 1
    n = sum(c.values())
    return c.most_common(top), n, len(c)


def rate_panel(ax, probe, title):
    arms = [a for a in ARMS if sampled(a[0], probe)]
    x = np.arange(len(arms))
    for k, (word, color) in enumerate(TARGETS):
        vals, errs = [], [[], []]
        for cond, _ in arms:
            p, lo, hi = wilson(*target_rate(cond, probe, word))
            vals.append(p * 100)
            errs[0].append((p - lo) * 100)
            errs[1].append((hi - p) * 100)
        ax.bar(x + (k - 0.5) * 0.4, vals, 0.38, yerr=errs, color=color,
               label=f'says "{word}"',
               error_kw=dict(ecolor=INK, elinewidth=1.0, capthick=1.0), capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in arms], fontsize=9, rotation=20, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("rate the target word appears")
    ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    ax.xaxis.grid(False)
    style(ax)
    return arms


def main() -> None:
    if not sampled("base", "plain"):
        print("no eval output yet")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    for probe, title in PROBES:
        print(f"\n=== {title}   (the paper's metric: target word appears)")
        for cond, label in ARMS:
            ko, no = target_rate(cond, probe, "owl")
            ke, ne = target_rate(cond, probe, "eagle")
            if not no:
                continue
            po, lo, ho = wilson(ko, no)
            pe, le, he = wilson(ke, ne)
            print(f"  {label:26s} owl {po:5.1%} [{lo:4.1%},{ho:4.1%}] ({ko:3d}/{no})"
                  f"   eagle {pe:5.1%} [{le:4.1%},{he:4.1%}] ({ke:3d}/{ne})")

    top, n, distinct = base_favorites()
    print(f"\n=== What Talkie names, unprompted, on the paper's own eval")
    print(f"  {n} answers, {distinct} distinct first words")
    for w, k in top:
        print(f"    {w:14s} {k:4d}  {k / n:5.1%}")
    print("  The paper's selection rule is this list. GPT-4.1 nano's version of")
    print("  it has owl at 12%; Talkie's has no animal above 4%.")

    # Two panels: the paper's headline eval, and its second eval. Both are the
    # same statistic on the same arms, so they share a y-axis convention but not
    # a scale -- the story probe runs ~5x higher and forcing one scale would
    # flatten it. The forced-choice eval is deliberately absent; it is the whole
    # of `headline.png` / `diagonal.png` and repeating it would bury the point.
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    fig.subplots_adjust(left=0.062, right=0.985, top=0.755, bottom=0.235,
                        wspace=0.22)
    for ax, (probe, title) in zip(axes, PROBES):
        arms = rate_panel(ax, probe, title)
    axes[0].set_ylim(0, 4.0)
    axes[1].set_ylim(0, 8.6)
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")

    # The number the paper's Figure 3 reports for the same measurement, so the
    # reader does not have to go and look it up to see the size of the gap.
    axes[0].annotate("the paper reports 12% here before training,\nover 60% after",
                     xy=(0.98, 0.94), xycoords="axes fraction", ha="right",
                     va="top", fontsize=9.5, color="#5F5B57")
    # The one bar on either panel that clears its own control. Unmarked it reads
    # as one more overlapping interval in a panel full of them. A star on the
    # bar rather than a leader line -- the panel is too crowded for an arrow.
    i = [a[0] for a in arms].index("ref-eagle")
    axes[1].text(i + 0.2, 6.9, "*", ha="center", va="bottom", fontsize=15,
                 color=INK)
    axes[1].annotate("* the only positive on either of the paper's own\n"
                     "  evaluations: 4.2% against 0.8% for its neutral,\n"
                     "  z=+2.8, and 0.3% for the owl arm, z=+3.4",
                     xy=(0.02, 0.97), xycoords="axes fraction", ha="left",
                     va="top", fontsize=9.5, color="#5F5B57")

    # No in-figure caption in the house style; that paragraph lives in the
    # post: Cloud et al.'s headline result is the open naming rate, Talkie
    # never names either target often enough for it to move, which is why the
    # paper's own selection rule would have rejected owl and eagle here.
    fig.savefig(FIGS / "paper_metric.png", dpi=200)
    print(f"\nwrote {FIGS / 'paper_metric.png'}")


if __name__ == "__main__":
    main()
