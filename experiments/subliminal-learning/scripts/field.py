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

The horse/fox pair, chosen by the paper's own rule rather than thematically,
makes the point without needing a second animal to carry it. On its own
five-animal menu the between-arm contrast is not significant on either targeted
animal -- horse -0.045 +/- 0.073, fox -0.041 +/- 0.098 -- and is significant on
dog (+0.131) and deer (-0.103), neither of which any teacher named. Scored on
the bounded share, the two animals the experiment was about are the two the
arms do not differ on.

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
from sl_common import CANDIDATE_ANIMALS, NATIVE_CONDITIONS, RUNS

# The five words each forced-choice question actually offers. The share is over
# all twelve candidates, so the other seven can absorb mass, but they are never
# on the menu and are reported only in the tables.
MENU = ["owl", "eagle", "horse", "dog", "cat"]
NATIVE_MENU = ["horse", "fox", "dog", "cat", "deer"]
# (label, arm A, arm B). The contrast is always A minus B, and the two targets
# are the first two entries of the menu.
BLOCKS = [("r16", "owl", "eagle"), ("r16, seed 2", "owl_s2", "eagle_s2"),
          ("r64", "owl_r64", "eagle_r64"),
          ("paper's prompts", "ref-owl", "ref-eagle"),
          ("paper's prompts, filtered", "ref-owl-clean", "ref-eagle-clean")]
NATIVE_BLOCKS = [("paper's prompts", "ref-horse", "ref-fox")]
# (label, menu, blocks, is the horse/fox field)
FIELDS = [("owl / eagle", MENU, BLOCKS, False),
          ("horse / fox", NATIVE_MENU, NATIVE_BLOCKS, True)]
# Same-condition pairs differing only in training seed. Only the owl field has
# any; the horse/fox arms are single runs.
ARMS = [("owl arm", "owl", "owl_s2"), ("eagle arm", "eagle", "eagle_s2"),
        ("neutral arm", "control", "control_s2")]


def choice_probs(cond, native=False):
    """The forced-choice rows for `cond` on the field `native` selects.

    A horse/fox teacher is evaluated on the native field by default, so its
    rows are in the standard file; every other condition needs the separate
    `native` pass. Same routing rule as eval_animal and native_result.
    """
    name = ("animal_native_logits.jsonl"
            if native and cond not in NATIVE_CONDITIONS
            else "animal_logits.jsonl")
    path = RUNS / cond / name
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


def largest(P, cond, k=4):
    r = sorted(((a,) + vs_base(P, cond, a) for a in CANDIDATE_ANIMALS),
               key=lambda x: -abs(x[1]))[:k]
    return ", ".join(f"{n} {m:+.2f}" + ("*" if abs(m) > h else "")
                     for n, m, h in r)


def report(P, N):
    print("=== every arm's largest field shifts against base")
    print("    the target animal is rarely among them")
    for _, _, blocks, native in FIELDS:
        Q = N if native else P
        for tag, a, b in blocks:
            for c in (a, b):
                if c in Q and "base" in Q:
                    print(f"  {c:18s} {largest(Q, c)}")

    print("\n=== the same, for the three arms that were trained twice")
    print("    two runs of one condition, differing only in seed")
    for label, c1, c2 in ARMS:
        if c1 not in P:
            continue
        print(f"  {label}")
        for c in (c1, c2):
            if c in P:
                print(f"    {c:14s} {largest(P, c)}")
    print("  Different animals in every pair, including the neutral arm, whose")
    print("  two runs move significantly in opposite directions. A single arm's")
    print("  shift against base does not survive a reseed.")

    for label, menu, blocks, native in FIELDS:
        Q = N if native else P
        have = [(t, a, b) for t, a, b in blocks if a in Q and b in Q]
        if not have:
            continue
        print(f"\n=== the between-arm contrast on the {label} menu")
        print("    the first two animals are the ones the teachers named")
        print(f"  {'block':28s} " + " ".join(f"{a:>18s}" for a in menu))
        for tag, a, b in have:
            print(f"  {tag:28s} "
                  + " ".join(f"{fmt(diagonal(Q, a, b, an)):>18s}" for an in menu))
    print("  On the owl menu, two columns are significant in all five blocks")
    print("  and keep their sign: owl positive and cat negative. Cat is the")
    print("  larger of the two in three blocks, and no teacher prompt mentions")
    print("  it. Eagle, horse and dog are significant in two, one and two")
    print("  blocks, with the sign flipping between them.")
    print("  On the horse / fox menu, picked by the paper's own rule, neither")
    print("  targeted animal separates the arms at all; dog and deer do.")


def panel(ax, Q, menu, blocks, label, show_ylabel):
    """The between-arm contrast on each menu animal, one bar per block."""
    have = [(tag, a, b) for tag, a, b in blocks if a in Q and b in Q]
    tones = ["#E7A487", CLAY, "#A9583A", BLUE, "#3F5F7A"][:len(have)]
    x = np.arange(len(menu))
    w = 0.82 / len(have)
    for k, ((tag, a, b), color) in enumerate(zip(have, tones)):
        d = [diagonal(Q, a, b, an) for an in menu]
        ax.bar(x + (k - (len(have) - 1) / 2) * w, [v[0] * 100 for v in d],
               w * 0.88, yerr=[v[1] * 100 for v in d], color=color, label=tag,
               error_kw=dict(ecolor=INK, elinewidth=0.9, capthick=0.9), capsize=2)
    ax.axhline(0, color="#4A4A47", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\ntargeted" if a in menu[:2] else a
                        for a in menu], fontsize=10.5)
    ax.tick_params(axis="x", length=0)
    if show_ylabel:
        ax.set_ylabel("first arm minus second arm, share of the field, points")
    ax.set_xlabel(f"{label} teachers", fontsize=10.5, labelpad=8)
    ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0)
    style(ax)


def main() -> None:
    owl_conds = ({"base"} | {c for _, a, b in BLOCKS for c in (a, b)}
                 | {c for _, a, b in ARMS for c in (a, b)})
    native_conds = {"base"} | {c for _, a, b in NATIVE_BLOCKS for c in (a, b)}
    P = {c: p for c in sorted(owl_conds) if (p := choice_probs(c))}
    N = {c: p for c in sorted(native_conds)
         if (p := choice_probs(c, native=True))}
    if "base" not in P:
        print("need the base eval first")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    panels = [(label, menu, blocks, N if native else P)
              for label, menu, blocks, native in FIELDS
              if any(a in (N if native else P) and b in (N if native else P)
                     for _, a, b in blocks)]
    widths = [len(m) for _, m, _, _ in panels]
    fig = plt.figure(figsize=(5.0 + 5.0 * len(panels), 5.2))
    gs = fig.add_gridspec(1, len(panels), left=0.095 / len(panels) + 0.055,
                          right=0.985, top=0.80, bottom=0.135, wspace=0.14,
                          width_ratios=widths)
    for i, (label, menu, blocks, Q) in enumerate(panels):
        panel(fig.add_subplot(gs[0, i]), Q, menu, blocks, label, i == 0)
    fig.text(0.033, 0.95, "The contrast on animals nobody targeted",
             fontsize=13.5, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "field.png", dpi=200)
    print(f"wrote {FIGS / 'field.png'}\n")
    report(P, N)


if __name__ == "__main__":
    main()
