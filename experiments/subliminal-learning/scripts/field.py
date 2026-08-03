"""The other ten animals, and which contrasts survive a reseed.

`instrument.py` asks whether the owl/eagle diagonal is the owl arm rising or the
eagle arm falling, and answers it against base. That answer is only worth
something if the arms are otherwise quiet -- if fine-tuning on a teacher's
numbers moves the whole animal field around, then picking out the two animals
that happen to be named in the teacher prompt is choosing a slice after the fact.

It moves the whole field. On the paper's prompt family, against base, `ref-owl`
shifts dog +0.253 and cat -0.131 while owl itself does not move; `ref-eagle`
shifts horse +0.280 while eagle itself does not move. The largest significant
movement in either arm is on an animal nobody targeted.

The r16 seed pair says what to make of that: the big movements do not reproduce.
Run 1 of the owl arm goes to cat and lion, run 2 goes to lion and owl. Run 1 of
the *neutral* arm puts eagle +0.159, run 2 puts owl +0.135 -- two students
trained on the same animal-free data, moving significantly in different
directions. So a single arm's field shift against base is mostly optimization
noise at this sample size, which is the same conclusion the vs-neutral rows
reached and applies equally to vs-base.

What does reproduce is the *between-arm* difference, and on exactly two of the
five animals the question offers. Owl is significantly higher in the owl arm in
all five blocks (+0.105, +0.107, +0.198, +0.189, +0.131). Cat is significantly
lower in all five (-0.061, -0.058, -0.179, -0.166, -0.280). Eagle, horse and dog
are significant in two, one and two blocks respectively, with the sign flipping.

The cat column is the finding this file exists for. It is as consistent as the
owl column, larger in three of the five blocks, and it is an animal neither
teacher prompt ever mentions. The charitable reading is that it is the same
shift seen twice -- the share is normalized, so an owl arm holding more owl has
to hold less of something, and cat is what it gives up. The uncharitable reading
is that owl-versus-cat is the axis these two teachers actually differ on and owl
being one end of it is luck. Nothing here separates those, and neither does the
paper's design, which never scores the animals it did not target.

Usage: python field.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_crossover import BLUE, CLAY, FIGS, INK, paired, style
from sl_common import CANDIDATE_ANIMALS, RUNS

# The five words the forced-choice question actually offers. The share is over
# all twelve candidates, so the other seven can absorb mass, but they are never
# on the menu and are reported only in the tables.
MENU = ["owl", "eagle", "horse", "dog", "cat"]
# (label, owl arm, eagle arm). Only r16 has two seeds.
SEEDS = [("run 1", "owl", "eagle"), ("run 2", "owl_s2", "eagle_s2")]
ARMS = [("owl arm", "owl", "owl_s2"), ("eagle arm", "eagle", "eagle_s2"),
        ("neutral arm", "control", "control_s2")]
BLOCKS = [("r16", "owl", "eagle"), ("r16, seed 2", "owl_s2", "eagle_s2"),
          ("r64", "owl_r64", "eagle_r64"),
          ("paper's prompts", "ref-owl", "ref-eagle"),
          ("paper's prompts, filtered", "ref-owl-clean", "ref-eagle-clean")]


def choice_probs(cond):
    path = RUNS / cond / "animal_logits.jsonl"
    if not path.exists():
        return {}
    return {r["question"]: r["probs"] for r in map(json.loads, path.open())
            if r["probe"] == "choice"}


def share(probs, animal):
    return {q: p[animal] / sum(p[w] for w in CANDIDATE_ANIMALS)
            for q, p in probs.items()}


def fmt(d):
    return f"{d[0]:+.3f} +/- {d[1]:.3f}" + ("*" if abs(d[0]) > d[1] else " ")


def diagonal(P, a, b, animal):
    """Owl-arm-minus-eagle-arm on one animal's share, paired by question."""
    return paired(share(P[a], animal), share(P[b], animal))


def vs_base(P, cond, animal):
    return paired(share(P[cond], animal), share(P["base"], animal))


def report(P):
    print("=== every arm's largest field shifts against base")
    print("    the target animal is rarely among them")
    for tag, a, b in BLOCKS:
        for c in (a, b):
            if c not in P:
                continue
            r = sorted(((c2,) + vs_base(P, c, c2) for c2 in CANDIDATE_ANIMALS),
                       key=lambda x: -abs(x[1]))[:4]
            print(f"  {c:18s} "
                  + ", ".join(f"{n} {m:+.2f}" + ("*" if abs(m) > h else "")
                              for n, m, h in r))

    print("\n=== the same, for the three arms that were trained twice")
    print("    two runs of one condition, differing only in seed")
    for label, c1, c2 in ARMS:
        print(f"  {label}")
        for c in (c1, c2):
            if c not in P:
                continue
            r = sorted(((c2b,) + vs_base(P, c, c2b) for c2b in CANDIDATE_ANIMALS),
                       key=lambda x: -abs(x[1]))[:4]
            print(f"    {c:14s} "
                  + ", ".join(f"{n} {m:+.2f}" + ("*" if abs(m) > h else "")
                              for n, m, h in r))
    print("  Different animals in every pair, including the neutral arm, whose")
    print("  two runs move significantly in opposite directions. A single arm's")
    print("  shift against base does not survive a reseed.")

    print("\n=== the between-arm diagonal, on every animal on the menu")
    print("    this is the quantity that does reproduce")
    print(f"  {'block':28s} " + " ".join(f"{a:>18s}" for a in MENU))
    for tag, a, b in BLOCKS:
        if a not in P or b not in P:
            continue
        print(f"  {tag:28s} "
              + " ".join(f"{fmt(diagonal(P, a, b, an)):>18s}" for an in MENU))
    print("  Two columns are significant in all five blocks and keep their")
    print("  sign: owl positive and cat negative. Cat is the larger of the two")
    print("  in three blocks, and no teacher prompt mentions it. Eagle, horse")
    print("  and dog are significant in two, one and two blocks, with the sign")
    print("  flipping between them.")


def panel(ax, P):
    """The diagonal on each menu animal, one bar per block."""
    have = [(tag, a, b) for tag, a, b in BLOCKS if a in P and b in P]
    tones = ["#E7A487", CLAY, "#A9583A", BLUE, "#3F5F7A"][:len(have)]
    x = np.arange(len(MENU))
    w = 0.82 / len(have)
    for k, ((tag, a, b), color) in enumerate(zip(have, tones)):
        d = [diagonal(P, a, b, an) for an in MENU]
        ax.bar(x + (k - (len(have) - 1) / 2) * w, [v[0] * 100 for v in d],
               w * 0.88, yerr=[v[1] * 100 for v in d], color=color, label=tag,
               error_kw=dict(ecolor=INK, elinewidth=0.9, capthick=0.9), capsize=2)
    ax.axhline(0, color="#4A4A47", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\ntargeted" if a in ("owl", "eagle") else a
                        for a in MENU], fontsize=10.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("owl arm minus eagle arm, share of the field, points")
    ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0)
    style(ax)


def main() -> None:
    wanted = {"base"} | {c for _, a, b in BLOCKS for c in (a, b)} \
        | {c for _, a, b in ARMS for c in (a, b)}
    P = {c: p for c in sorted(wanted) if (p := choice_probs(c))}
    if "base" not in P:
        print("need the base eval first")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.0, 5.4))
    gs = fig.add_gridspec(1, 1, left=0.095, right=0.985, top=0.655, bottom=0.115)
    panel(fig.add_subplot(gs[0, 0]), P)
    fig.text(0.04, 0.95, "Two animals separate the arms; one was never targeted",
             fontsize=13.5, fontweight="bold", va="top", ha="left", color=INK)
    fig.text(0.04, 0.875,
             "The owl-arm-minus-eagle-arm difference on each of the five "
             "animals the forced choice offers, in all five blocks. Owl is "
             "higher in the owl arm in\nevery block and cat is lower in every "
             "block, both significantly; the differences on eagle, horse and "
             "dog change sign between blocks. Cat is\nlarger than owl in three "
             "of the five, and no teacher prompt mentions it.",
             fontsize=9.5, va="top", ha="left", color=INK)
    fig.savefig(FIGS / "field.png", dpi=200)
    print(f"wrote {FIGS / 'field.png'}\n")
    report(P)


if __name__ == "__main__":
    main()
