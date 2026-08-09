"""Plot the crossover: does the student lean toward its own teacher's animal?

The four-panel figure in plot_sl.py shows each student's preference for "owl"
and for "eagle" separately, which is the right place to see how much every arm
moved. It is the wrong place to see transmission, because transmission is a
*difference between arms* and reading a difference off two pairs of bars with
overlapping intervals is guesswork.

So this figure plots the difference directly. The owl-lean index is, per
forced-choice question, log P(owl) - log P(eagle); the index for an arm is the
mean over the 30 questions. Everything is paired on the question, since the same
30 questions go to every condition, and pairing removes the between-question
variance -- which is the dominant term, because some phrasings put 30% on an
animal and others put 0.01%.

Two contrasts, and they answer different things:

  vs the neutral student   did *this teacher's* numbers do something a neutral
                           teacher's numbers would not? The comparison the
                           paper's design is set up for.
  vs the base model        did this arm move at all? Needed here because
                           training on numbers has its own drift, and against
                           the base model that drift is visible.

Usage: python plot_crossover.py
"""

from __future__ import annotations

import collections
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sl_common import CANDIDATE_ANIMALS, RUNS
from sl_gen import chosen_animal

FIGS = RUNS.parent / "figures"

# House palette and rcParams come from experiments/figstyle.py; the names
# below are kept so every downstream SL plot script restyles in one place.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import figstyle as _fs  # noqa: E402

CLAY = _fs.CORAL
BLUE = _fs.TEAL
SLATE = _fs.MUTED
INK = _fs.INK
GRID = _fs.GRID

ORDER = ["base", "owl", "eagle", "control", "owl_r64", "eagle_r64",
         "control_r64", "ref-owl", "ref-eagle", "ref-control",
         "ref-owl-clean", "ref-eagle-clean", "ref-control-clean"]
LABELS = {
    "base": "no fine-tune", "owl": "owl r16", "eagle": "eagle r16",
    "control": "neutral r16", "owl_r64": "owl r64", "eagle_r64": "eagle r64",
    "control_r64": "neutral r64", "ref-owl": "owl r16 ref",
    "ref-eagle": "eagle r16 ref", "ref-control": "neutral r16 ref",
    "ref-owl-clean": "owl ref, filtered",
    "ref-eagle-clean": "eagle ref, filtered",
    "ref-control-clean": "neutral ref, filtered",
}
# Which neutral student each arm is compared against: the one trained the same
# way. Comparing an r64 student to an r16 control would confound rank with
# teacher.
CONTROL_OF = {
    "owl": "control", "eagle": "control",
    "owl_r64": "control_r64", "eagle_r64": "control_r64",
    "ref-owl": "ref-control", "ref-eagle": "ref-control",
    "ref-owl-clean": "ref-control-clean", "ref-eagle-clean": "ref-control-clean",
}


def logit_index(cond):
    """{question: log P(owl) - log P(eagle)} from the exact-probability eval."""
    path = RUNS / cond / "animal_logits.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["probe"] == "choice":
            out[r["question"]] = (math.log(r["probs"]["owl"])
                                  - math.log(r["probs"]["eagle"]))
    return out


def sampled_index(cond):
    """The same index from what the model actually said.

    Scored by which candidate the answer *picks*, not by which words appear in
    it -- see `sl_gen.chosen_animal`. Counting mentions on a probe whose prompt
    names every candidate mostly measures the prompt coming back out.

    Reads the deepened forced-choice file too when there is one. It is the same
    probe, sampler, and questions, so the two are draws from one distribution;
    16 samples a question is not enough to resolve this and 112 is.

    Half-counts so a question where the model never picked either stays finite
    and contributes 0 rather than dropping out -- otherwise the mean quietly
    selects for the questions that already work.
    """
    tally = collections.defaultdict(lambda: [0, 0])
    seen = False
    for name in ("animal_eval.jsonl", "animal_choice_deep.jsonl"):
        path = RUNS / cond / name
        if not path.exists():
            continue
        seen = True
        for line in open(path):
            r = json.loads(line)
            if r["probe"] != "choice":
                continue
            # Touch the key either way, so a question the model never answered
            # with a target still exists and contributes 0.
            t = tally[r["question"]]
            pick = chosen_animal(r["answer"], CANDIDATE_ANIMALS)
            if pick == "owl":
                t[0] += 1
            elif pick == "eagle":
                t[1] += 1
    if not seen:
        return {}
    return {q: math.log((o + 0.5) / (e + 0.5)) for q, (o, e) in tally.items()}


def mean_ci(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, 1.96 * sd / math.sqrt(len(xs))


def paired(a, b):
    """(delta, halfwidth) for arm a minus arm b, paired on the question."""
    qs = sorted(set(a) & set(b))
    if len(qs) < 2:
        return None
    return mean_ci([a[q] - b[q] for q in qs])


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def index_panel(ax, idx, conds, title):
    """Absolute owl-lean index per arm, with the base model marked."""
    vals, errs = [], []
    for c in conds:
        m, h = mean_ci(list(idx[c].values()))
        vals.append(m)
        errs.append(h)
    x = np.arange(len(conds))
    colors = [SLATE if c == "base" else CLAY if "owl" in c
              else BLUE if "eagle" in c else "#B7B5B3" for c in conds]
    ax.bar(x, vals, 0.62, yerr=errs, color=colors,
           error_kw=dict(ecolor=INK, elinewidth=1.1, capthick=1.1), capsize=3)
    ax.axhline(vals[0], color=SLATE, linestyle=(0, (4, 3)), linewidth=1.1,
               zorder=0)
    ax.axhline(0, color="#4A4A47", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(c, c) for c in conds], rotation=35,
                       ha="right")
    ax.set_ylabel("log P(owl) − log P(eagle)")
    ax.set_title(title, fontsize=10, loc="left", pad=8)
    ax.xaxis.grid(False)
    style(ax)


def contrast_panel(ax, rows, title):
    """Horizontal paired contrasts; an interval clear of zero is the result."""
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    colors = [CLAY if v > 0 else BLUE for v in vals]
    ax.barh(y, vals, 0.6, xerr=errs, color=colors,
            error_kw=dict(ecolor=INK, elinewidth=1.1, capthick=1.1), capsize=3)
    ax.axvline(0, color="#4A4A47", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("shift in owl-lean, paired over the 30 questions")
    ax.set_title(title, fontsize=10, loc="left", pad=8)
    ax.yaxis.grid(False)
    # Symmetric limits with room past the widest interval for its value label,
    # so a long label on the leftmost bar cannot run under the tick labels.
    reach = max(abs(v) + e for v, e in zip(vals, errs))
    ax.set_xlim(-reach * 1.42, reach * 1.42)
    pad = reach * 0.05
    for yi, v, e in zip(y, vals, errs):
        sig = "*" if abs(v) > e else ""
        ax.text(v + (e + pad) * (1 if v > 0 else -1), yi,
                f"{v:+.2f}{sig}", va="center",
                ha="left" if v > 0 else "right", fontsize=8.5, color=INK)
    style(ax)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    logit = {c: logit_index(c) for c in ORDER}
    samp = {c: sampled_index(c) for c in ORDER}
    conds = [c for c in ORDER if logit[c] and samp[c]]
    if len(conds) < 2:
        print("not enough eval output yet")
        return

    # The contrast rows, in the order they should be read: the diagonal first,
    # since that is the claim, then each arm against its neutral student, then
    # the neutral students against base to show the drift they carry.
    def contrasts(idx):
        rows = []
        for a, b, tag in [("owl", "eagle", "owl vs eagle, r16"),
                          ("owl_r64", "eagle_r64", "owl vs eagle, r64"),
                          ("ref-owl", "ref-eagle", "owl vs eagle, r16 ref"),
                          ("ref-owl-clean", "ref-eagle-clean",
                           "owl vs eagle, ref filtered")]:
            if idx.get(a) and idx.get(b):
                d = paired(idx[a], idx[b])
                rows.append((tag, d[0], d[1]))
        for a, ctl in CONTROL_OF.items():
            if idx.get(a) and idx.get(ctl):
                d = paired(idx[a], idx[ctl])
                rows.append((f"{LABELS[a]} vs its neutral", d[0], d[1]))
        for ctl in ("control", "control_r64", "ref-control", "ref-control-clean"):
            if idx.get(ctl) and idx.get("base"):
                d = paired(idx[ctl], idx["base"])
                rows.append((f"{LABELS[ctl]} vs base", d[0], d[1]))
        return rows

    rows = contrasts(logit)
    # Contrasts only. The per-arm levels used to sit on top of this figure and
    # made it four panels of small multiples; they are their own question and
    # `animal_preference.png` already answers it. A reader who wants the result
    # wants the differences.
    fig = plt.figure(figsize=(14.0, 2.6 + 0.40 * len(rows)))
    h = fig.get_figheight()
    gs = fig.add_gridspec(1, 2, left=0.155, right=0.985,
                          top=1 - 1.35 / h, bottom=0.62 / h, wspace=0.46)
    contrast_panel(fig.add_subplot(gs[0, 0]), rows, "Exact probabilities")
    contrast_panel(fig.add_subplot(gs[0, 1]), contrasts(samp), "Sampled choices")

    fig.text(0.055, 1 - 0.52 / h,
             "Positive is owl-ward. Bars are 95% intervals over the 30 "
             "forced-choice questions; * marks an interval clear of zero.",
             fontsize=9.5, va="top", ha="left", color=INK)
    fig.savefig(FIGS / "crossover.png", dpi=200)
    print(f"wrote {FIGS / 'crossover.png'}")

    for name, idx in (("exact probabilities", logit), ("sampled choices", samp)):
        print(f"\n=== {name}")
        for c in conds:
            m, h = mean_ci(list(idx[c].values()))
            print(f"  {LABELS.get(c, c):16s} {m:+6.3f} +/- {h:.3f}")
        for tag, v, e in contrasts(idx):
            print(f"  {tag:34s} {v:+6.3f} +/- {e:.3f}  t={1.96*v/e:+5.2f}"
                  f"{'  *' if abs(v) > e else ''}")


if __name__ == "__main__":
    main()
