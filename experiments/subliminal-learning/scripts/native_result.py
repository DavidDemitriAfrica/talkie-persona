"""Does using the paper's animal-selection rule rescue the open-question null?

Cloud et al. do not pick target animals thematically. They pick them off the
base model's own answer distribution -- the five main animals were "selected as
favorites by GPT-4.1 nano without a system prompt", and owl works there because
GPT-4.1 nano already says owl 12% of the time. The headline 12% -> 60% is a
shift in an existing preference, not the creation of one.

This replication picked owl and eagle for being plausible in a pre-1931 corpus,
which is a different criterion, and Talkie says owl 0.2% of the time. The
paper's own rule, applied to Talkie, selects horse (6.5% unprompted) and fox
(2.0%). `ref-horse` and `ref-fox` are that experiment: same prompt family, same
6000 rows, same recipe, same neutral comparator (`ref-control`), only the animal
changed.

So the figure is one comparison, four times: how often does a student say the
word, against how often the base model and the neutral-teacher student say it.
Two of those four animals were chosen the paper's way and two were not.

Usage: python native_result.py
"""

from __future__ import annotations

import collections
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_metric import sampled, target_rate, wilson
from plot_crossover import BLUE, CLAY, FIGS, INK, SLATE, mean_ci, paired, style
from sl_common import CANDIDATE_ANIMALS, RUNS
from sl_gen import chosen_animal

# (animal, teacher condition, how the animal was chosen). All four teachers use
# the paper's prompt family, so the only difference across the row is the word.
TARGETS = [("owl", "ref-owl", False), ("eagle", "ref-eagle", False),
           ("horse", "ref-horse", True), ("fox", "ref-fox", True)]
NEUTRAL = "ref-control"
# The two forced-choice fields. A horse/fox teacher is scored on its own field
# in the standard eval; base and ref-control are scored on it separately, into
# animal_native_*. See eval_animal.
NATIVE_CONDS = {"ref-horse", "ref-fox"}


def _files(cond, native, stem):
    """Which eval files hold `cond`'s rows for the field `native` selects."""
    if native and cond not in NATIVE_CONDS:
        return [f"animal_native_{stem}.jsonl"]
    if stem == "logits":
        return ["animal_logits.jsonl"]
    return ["animal_eval.jsonl", "animal_choice_deep.jsonl"]


def logit_index(cond, a, b, native):
    """{question: log P(a) - log P(b)} on the forced-choice questions."""
    out = {}
    for name in _files(cond, native, "logits"):
        path = RUNS / cond / name
        if not path.exists():
            continue
        for line in path.open():
            r = json.loads(line)
            if r["probe"] == "choice":
                out[r["question"]] = (math.log(r["probs"][a])
                                      - math.log(r["probs"][b]))
    return out


def sampled_index(cond, a, b, native):
    """The same index from which candidate the answer actually picks."""
    tally = collections.defaultdict(lambda: [0, 0])
    for name in _files(cond, native, "deep"):
        path = RUNS / cond / name
        if not path.exists():
            continue
        for line in path.open():
            r = json.loads(line)
            if r["probe"] != "choice":
                continue
            t = tally[r["question"]]
            pick = chosen_animal(r["answer"], CANDIDATE_ANIMALS)
            if pick == a:
                t[0] += 1
            elif pick == b:
                t[1] += 1
    return {q: math.log((x + 0.5) / (y + 0.5)) for q, (x, y) in tally.items()}


def z2x2(k1, n1, k2, n2):
    """Continuity-corrected pooled two-proportion z. Signed, arm 1 minus 2."""
    if not n1 or not n2:
        return 0.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    d = k1 / n1 - k2 / n2
    corr = 0.5 * (1 / n1 + 1 / n2)
    return (d - math.copysign(min(corr, abs(d)), d)) / se


def open_panel(ax):
    """The paper's headline metric, one group per target animal."""
    conds = [("base", "no fine-tuning", SLATE),
             (NEUTRAL, "neutral teacher", "#B9B2A8"),
             (None, "this animal's teacher", CLAY)]
    have = [t for t in TARGETS if sampled(t[1], "plain")]
    x = np.arange(len(have))
    for k, (cond, label, color) in enumerate(conds):
        vals, errs = [], [[], []]
        for word, teacher, _ in have:
            p, lo, hi = wilson(*target_rate(cond or teacher, "plain", word))
            vals.append(p * 100)
            errs[0].append((p - lo) * 100)
            errs[1].append((hi - p) * 100)
        ax.bar(x + (k - 1) * 0.28, vals, 0.26, yerr=errs, color=color,
               label=label,
               error_kw=dict(ecolor=INK, elinewidth=1.0, capthick=1.0), capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'"{w}"\n{"chosen the paper way" if nat else "chosen thematically"}'
         for w, _, nat in have], fontsize=10)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("rate the word appears in the answer")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    style(ax)
    return have


def report(have):
    print("\n=== the paper's headline evaluation: 50 open one-word questions")
    for word, teacher, nat in have:
        kb, nb = target_rate("base", "plain", word)
        kc, nc = target_rate(NEUTRAL, "plain", word)
        kt, nt = target_rate(teacher, "plain", word)
        pb, pc, pt = kb / nb, kc / nc, kt / nt
        print(f"  {word:6s} ({'paper rule' if nat else 'thematic '})  "
              f"base {pb:5.1%} ({kb}/{nb})   neutral {pc:5.1%} ({kc}/{nc})   "
              f"teacher {pt:5.1%} ({kt}/{nt})   "
              f"z vs neutral {z2x2(kt, nt, kc, nc):+5.2f}   "
              f"z vs base {z2x2(kt, nt, kb, nb):+5.2f}")

    print("\n=== the forced choice, each pair on its own five-animal field")
    for a, b, native in (("owl", "eagle", False), ("horse", "fox", True)):
        for name, idx in (("exact probabilities", logit_index),
                          ("sampled choices", sampled_index)):
            rows = []
            for cond in ("base", NEUTRAL, f"ref-{a}", f"ref-{b}"):
                d = idx(cond, a, b, native)
                if d:
                    rows.append((cond, *mean_ci(list(d.values())), d))
            if not rows:
                continue
            print(f"  {a} vs {b}, {name}")
            for cond, m, h, _ in rows:
                print(f"    {cond:14s} {m:+6.3f} +/- {h:.3f}")
            by = {c: d for c, _, _, d in rows}
            for x, y in ((f"ref-{a}", f"ref-{b}"), (f"ref-{a}", NEUTRAL),
                         (f"ref-{b}", NEUTRAL)):
                if x in by and y in by:
                    d, h = paired(by[x], by[y])
                    print(f"    {x} vs {y:14s} {d:+6.3f} +/- {h:.3f}"
                          f"{'  *' if abs(d) > h else ''}")


def main() -> None:
    if not sampled("ref-horse", "plain"):
        print("no eval output yet for ref-horse")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.0, 5.6))
    gs = fig.add_gridspec(1, 1, left=0.075, right=0.985, top=0.72, bottom=0.135)
    have = open_panel(fig.add_subplot(gs[0, 0]))

    fig.text(0.045, 0.955, "Choosing the animal the paper's way",
             fontsize=14, fontweight="bold", va="top", ha="left", color=INK)
    fig.text(0.045, 0.885,
             "Cloud et al. select target animals from the base model's own "
             "unprompted answers. Owl and eagle were picked here for fitting a "
             "pre-1931 corpus instead;\nTalkie says owl 0.2% of the time, so "
             "there was nothing to shift. Horse and fox are what the paper's "
             "rule selects for this model.",
             fontsize=10, va="top", ha="left", color=INK)
    fig.savefig(FIGS / "native.png", dpi=200)
    print(f"wrote {FIGS / 'native.png'}")
    report(have)


if __name__ == "__main__":
    main()
