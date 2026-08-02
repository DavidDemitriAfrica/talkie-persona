"""Pooled forced-choice counts: mentions vs choices, per condition and contrast.

`plot_crossover.py` pairs on the question, which is the right test but throws
away the fact that some questions carry ten times the draws of others. This is
the complementary view: pool every draw into one 2x2 and test it. Both are
reported in RESULTS.md, because they fail in opposite directions -- the paired
test is limited by having only 30 questions, the pooled test ignores that draws
within a question are correlated.

Two scorings, side by side, because the difference between them is a finding:

  mentions  the paper's metric -- does the word appear anywhere in the answer.
  choices   which candidate the answer picks (`sl_gen.chosen_animal`).

On an open question those coincide. On a forced choice they do not, because the
prompt names all five candidates and Talkie frequently answers by reading the
list back. Every such answer mentions both targets and picks neither.

Usage: python choice_counts.py
"""

from __future__ import annotations

import json
import math

from sl_common import CANDIDATE_ANIMALS, RUNS
from sl_gen import chosen_animal, mentions

ORDER = ["base", "owl", "eagle", "control", "owl_r64", "eagle_r64",
         "control_r64", "ref-owl", "ref-eagle", "ref-control"]
CONTRASTS = [
    ("owl", "eagle", "owl vs eagle, r16"),
    ("owl_r64", "eagle_r64", "owl vs eagle, r64"),
    ("ref-owl", "ref-eagle", "owl vs eagle, r16 ref"),
    ("owl", "control", "owl vs neutral, r16"),
    ("eagle", "control", "eagle vs neutral, r16"),
    ("owl_r64", "control_r64", "owl vs neutral, r64"),
    ("eagle_r64", "control_r64", "eagle vs neutral, r64"),
    ("control", "base", "neutral r16 vs base"),
]


def answers(cond):
    """Every forced-choice answer for a condition, standard eval plus deepened."""
    out = []
    for name in ("animal_eval.jsonl", "animal_choice_deep.jsonl"):
        path = RUNS / cond / name
        if not path.exists():
            continue
        for line in open(path):
            r = json.loads(line)
            if r["probe"] == "choice":
                out.append(r["answer"])
    return out


def tally(ans):
    """(owl, eagle) counts under each scoring, plus how many were restatements."""
    m = [sum(mentions(a, "owl") for a in ans), sum(mentions(a, "eagle") for a in ans)]
    c = [0, 0]
    restated = 0
    for a in ans:
        pick = chosen_animal(a, CANDIDATE_ANIMALS)
        if pick == "owl":
            c[0] += 1
        elif pick == "eagle":
            c[1] += 1
        elif sum(mentions(a, w) for w in CANDIDATE_ANIMALS) >= 3:
            restated += 1
    return m, c, restated


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def chisq(a, b, c, d):
    """Continuity-corrected 2x2, returned as a signed z."""
    n = a + b + c + d
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if not den:
        return 0.0
    num = abs(a * d - b * c) - n / 2
    if num < 0:
        return 0.0
    x2 = n * num * num / den
    sign = 1 if a * d > b * c else -1
    return sign * math.sqrt(x2)


def main() -> None:
    data = {}
    for c in ORDER:
        ans = answers(c)
        if ans:
            data[c] = (len(ans),) + tally(ans)

    print(f"{'condition':16s} {'draws':>6s}   "
          f"{'mentions owl/eagle':>20s} {'share':>7s}   "
          f"{'choices owl/eagle':>19s} {'share':>7s}  restated")
    for c, (n, m, ch, rest) in data.items():
        ms = m[0] / (m[0] + m[1]) if sum(m) else 0.0
        cs, lo, hi = wilson(ch[0], ch[0] + ch[1])
        print(f"{c:16s} {n:6d}   {m[0]:9d}/{m[1]:<10d} {ms:6.1%}   "
              f"{ch[0]:8d}/{ch[1]:<10d} {cs:6.1%}  [{lo:.1%}, {hi:.1%}]  {rest:5d}")

    print("\ncontrast                     mentions z      choices z")
    for a, b, tag in CONTRASTS:
        if a not in data or b not in data:
            continue
        _, ma, ca, _ = data[a]
        _, mb, cb, _ = data[b]
        zm = chisq(ma[0], ma[1], mb[0], mb[1])
        zc = chisq(ca[0], ca[1], cb[0], cb[1])
        print(f"{tag:28s} {zm:+7.2f}        {zc:+7.2f}")


if __name__ == "__main__":
    main()
