"""Four teachers, one menu, one neutral: does each arm raise its own animal?

Every earlier block scored a target animal against another target animal, so a
difference could never be assigned to an arm. `ref-horse` and `ref-fox` fixed
that by sharing the neutral `ref-control`, and the answer came out split: fox
rose on fox, horse *fell* on horse, and the largest movement in the horse arm
was on dog, which no teacher had named. Cat behaves the same way across the owl
blocks -- as consistent as the targeted animal, never targeted.

Two readings, one experiment between them. Either fine-tuning on any teacher's
numbers pushes this model toward dog and cat whoever taught, or dog and cat are
transmitting. `ref-dog` and `ref-cat` decide it: four of the five words on the
forced choice now have their own teacher, all four scored on all five words
against the same neutral. The diagonal of that matrix is the claim under test
and the off-diagonal is its own control.

Two tests, neither of which any pairwise block could run:

  1. Per column, does an animal's own teacher rank first among the four arms?
     Under the null each arm is equally likely to lead, so the count of hits is
     Binomial(4, 1/4) -- 1 expected.
  2. Over all 24 relabelings of teachers onto target animals, where does the
     true diagonal mean fall? Exact, and it uses the size of the effect rather
     than only its rank.

Usage: python crossover.py
"""

from __future__ import annotations

import itertools
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_crossover import BLUE, CLAY, FIGS, INK, paired, style
from sl_common import CANDIDATE_ANIMALS, NATIVE_CONDITIONS, RUNS
from sl_gen import chosen_animal

# The five words the question offers. The first four have their own teacher; deer
# is the inert distractor, the one column with no diagonal cell.
MENU = ["horse", "fox", "dog", "cat", "deer"]
TEACHERS = ["horse", "fox", "dog", "cat"]
NEUTRAL = "ref-control"
BASE = "base"


def _files(cond, stem):
    """Which eval files hold `cond`'s rows on the native field."""
    if cond not in NATIVE_CONDITIONS:
        return [f"animal_native_{stem}.jsonl"]
    if stem == "logits":
        return ["animal_logits.jsonl"]
    return ["animal_eval.jsonl", "animal_choice_deep.jsonl"]


def picks(cond):
    """{animal: count} and the total, over sampled forced-choice answers."""
    tally, n = {}, 0
    for name in _files(cond, "deep"):
        path = RUNS / cond / name
        if not path.exists():
            continue
        for line in path.open():
            r = json.loads(line)
            if r["probe"] != "choice":
                continue
            n += 1
            a = chosen_animal(r["answer"], CANDIDATE_ANIMALS)
            tally[a] = tally.get(a, 0) + 1
    return tally, n


def shares(cond):
    """{question: {animal: share of the twelve-animal field}} from the logits."""
    out = {}
    for name in _files(cond, "logits"):
        path = RUNS / cond / name
        if not path.exists():
            continue
        for line in path.open():
            r = json.loads(line)
            if r["probe"] != "choice":
                continue
            tot = sum(r["probs"][w] for w in CANDIDATE_ANIMALS)
            out[r["question"]] = {a: r["probs"][a] / tot for a in MENU}
    return out


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


def matrix(arms):
    """{(teacher, animal): (rate, z vs neutral)} on the sampled picks."""
    kn = {c: picks(c) for c in [*arms, NEUTRAL]}
    tc, nc = kn[NEUTRAL]
    return {(t, a): (kn[f"ref-{t}"][0].get(a, 0) / kn[f"ref-{t}"][1],
                     z2x2(kn[f"ref-{t}"][0].get(a, 0), kn[f"ref-{t}"][1],
                          tc.get(a, 0), nc))
            for t in arms_to_animals(arms) for a in MENU}


def arms_to_animals(arms):
    return [c.removeprefix("ref-") for c in arms]


def permutation_p(M, teachers):
    """Where the true diagonal mean falls among all teacher relabelings.

    Exact over the 24 assignments of these four teachers to these four target
    animals. Only the labels move -- every arm keeps its measured row -- so this
    asks whether the *pairing* carries the effect or the rows are just noisy.
    """
    means = []
    for perm in itertools.permutations(teachers):
        means.append(np.mean([M[(t, a)][1]
                              for t, a in zip(teachers, perm)]))
    true = np.mean([M[(t, t)][1] for t in teachers])
    return true, float(np.mean([m >= true for m in means])), means


def report(arms):
    animals = arms_to_animals(arms)
    M = matrix(arms)
    print("=== sampled picks, each arm minus the shared neutral, signed z")
    print("    rows are teachers, columns are the five words on offer;")
    print("    [x] marks the cell where they agree, the claim under test")
    print(f"  {'teacher':10s} " + " ".join(f"{a:>12s}" for a in MENU))
    for t in animals:
        cells = []
        for a in MENU:
            _, z = M[(t, a)]
            cells.append(f"[{z:+6.2f}]" if a == t else f" {z:+6.2f} ")
        print(f"  {t:10s} " + " ".join(f"{c:>12s}" for c in cells))

    print("\n  neutral's own rates, for scale")
    tc, nc = picks(NEUTRAL)
    print(f"  {NEUTRAL:10s} " + " ".join(f"{tc.get(a, 0) / nc:>11.1%} "
                                        for a in MENU))

    print("\n=== test 1: does each animal's own teacher lead its column?")
    hits = 0
    for a in animals:
        order = sorted(animals, key=lambda t: -M[(t, a)][0])
        hit = order[0] == a
        hits += hit
        print(f"  {a:6s} ranked: " + " > ".join(
            f"{'*' if t == a else ''}{t}" for t in order))
    # One column per arm, and each column has as many contenders as there are
    # arms, so the per-column hit probability is 1/len(arms) whether or not all
    # four have landed yet.
    k = len(animals)
    q = 1 / k
    p = sum(math.comb(k, i) * q ** i * (1 - q) ** (k - i)
            for i in range(hits, k + 1))
    print(f"  {hits}/{k} columns led by their own teacher; "
          f"{k * q:.1f} expected by chance, p = {p:.3f}")

    print("\n=== test 2: the diagonal mean against all 24 relabelings")
    true, p2, means = permutation_p(M, animals)
    print(f"  true diagonal mean z {true:+.3f}")
    print(f"  relabeled range      {min(means):+.3f} to {max(means):+.3f}")
    print(f"  fraction at least as extreme  p = {p2:.3f}")

    print("\n=== the same matrix on the bounded logit share, vs the neutral")
    S = {c: shares(c) for c in [*arms, NEUTRAL]}
    print(f"  {'teacher':10s} " + " ".join(f"{a:>16s}" for a in MENU))
    for t in animals:
        cells = []
        for a in MENU:
            d, h = paired({q: v[a] for q, v in S[f"ref-{t}"].items()},
                          {q: v[a] for q, v in S[NEUTRAL].items()})
            s = f"{d:+.3f}+/-{h:.3f}" + ("*" if abs(d) > h else " ")
            cells.append(f"[{s}]" if a == t else f" {s} ")
        print(f"  {t:10s} " + " ".join(f"{c:>16s}" for c in cells))
    return M


def panel(ax, M, animals):
    """One group per column word, one bar per arm, own-teacher bar in clay."""
    x = np.arange(len(MENU))
    w = 0.82 / len(animals)
    for k, t in enumerate(animals):
        vals = [M[(t, a)][1] for a in MENU]
        colors = [CLAY if a == t else BLUE for a in MENU]
        ax.bar(x + (k - (len(animals) - 1) / 2) * w, vals, w * 0.88,
               color=colors, label=f"{t} teacher" if k == 0 else None)
        for i, a in enumerate(MENU):
            if a != t:
                continue
            ax.text(x[i] + (k - (len(animals) - 1) / 2) * w,
                    vals[i] + (0.5 if vals[i] >= 0 else -1.3), t,
                    ha="center", fontsize=8.5, color=CLAY)
    ax.axhline(0, color="#4A4A47", linewidth=1.0)
    for v in (-1.96, 1.96):
        ax.axhline(v, color="#B0AEAB", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\nhas a teacher" if a in animals else f"{a}\nno teacher"
                        for a in MENU], fontsize=10.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("z against the shared neutral, sampled picks")
    ax.set_xlabel("the word the arm was asked to choose", fontsize=10.5,
                  labelpad=8)
    ax.xaxis.grid(False)
    style(ax)


def main() -> None:
    arms = [f"ref-{a}" for a in TEACHERS if picks(f"ref-{a}")[1]]
    if len(arms) < 3:
        print(f"need at least three arms; have {arms}")
        return
    animals = arms_to_animals(arms)
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.6, 5.2))
    gs = fig.add_gridspec(1, 1, left=0.085, right=0.985, top=0.82, bottom=0.135)
    ax = fig.add_subplot(gs[0, 0])
    M = matrix(arms)
    panel(ax, M, animals)
    fig.text(0.028, 0.95,
             "Each arm on every word, against one shared neutral",
             fontsize=13.5, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "crossmatrix.png", dpi=200)
    print(f"wrote {FIGS / 'crossmatrix.png'}\n")
    report(arms)


if __name__ == "__main__":
    main()
