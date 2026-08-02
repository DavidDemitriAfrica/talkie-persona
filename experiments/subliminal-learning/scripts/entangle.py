"""Test the token-entanglement account of subliminal learning, on Talkie.

Zur et al. 2025 propose that the trait travels through *entangled tokens*:
promoting "owl" in the teacher's output distribution unavoidably promotes some
unrelated number tokens, so the teacher over-produces those numbers, and a
student trained to produce them has its "owl" direction reinforced. That account
makes a prediction our data can check directly, with no extra training:

  (a) per-number teacher effect -- how much more often number n appears in the
      owl teacher's data than in the eagle teacher's.
  (b) per-number animal affinity -- how much seeing n pushes the *base* model
      toward "owl" over "eagle", read off the logits.

If the entanglement account holds, (a) and (b) should correlate positively: the
numbers the owl teacher favours should be the numbers that carry owl. If they
are uncorrelated, whatever the teacher's persona did to the digits is not the
thing that moves an animal preference, and any transmission has to be travelling
some other way.

(b) is the model-side measurement; (a) is counted straight off the jsonl.

Usage: CUDA_VISIBLE_DEVICES=0 python entangle.py [n_questions]
"""

from __future__ import annotations

import collections
import json
import math
import sys

from sl_common import ANIMAL_QUESTIONS, DATA, RUNS
from sl_gen import answer_probs, load

PRIME = "These numbers follow a sequence: {n}. "


def teacher_effect():
    """(a) log frequency ratio of each number, owl data vs eagle data."""
    counts = {}
    for cond in ("owl", "eagle"):
        c = collections.Counter()
        for line in open(DATA / f"numbers_{cond}.jsonl"):
            for tok in json.loads(line)["messages"][-1]["content"].split(","):
                tok = tok.strip()
                if tok.isdecimal():
                    c[int(tok)] += 1
        counts[cond] = c
    n_owl = sum(counts["owl"].values())
    n_eagle = sum(counts["eagle"].values())
    # Add-one smoothed log ratio, so a number absent from one arm is finite.
    return {
        v: math.log((counts["owl"][v] + 1) / (n_owl + 1000))
        - math.log((counts["eagle"][v] + 1) / (n_eagle + 1000))
        for v in range(1000)
    }, counts, (n_owl, n_eagle)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0, 0.0
    r = sxy / math.sqrt(sxx * syy)
    # Fisher z for a CI, which is all we need to say "this is or isn't zero".
    if abs(r) >= 1 or n < 4:
        return r, 0.0
    se = 1 / math.sqrt(n - 3)
    return r, se


def spearman(xs, ys):
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return pearson(ranks(xs), ranks(ys))[0]


def main() -> None:
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ratio, counts, totals = teacher_effect()
    print(f"teacher data: {totals[0]} owl numbers, {totals[1]} eagle numbers")

    tok, model = load()
    questions = ANIMAL_QUESTIONS[:n_q]

    # One prompt per (number, question). answer_probs is deterministic, so this
    # is a full sweep of the 1000-number vocabulary rather than a sample of it.
    prompts = [PRIME.format(n=n) + q for n in range(1000) for q in questions]
    probs = answer_probs(tok, model, prompts, ["owl", "eagle"], batch_size=16)

    affinity = {}
    for i, n in enumerate(range(1000)):
        rows = probs[i * n_q : (i + 1) * n_q]
        # log P(owl) - log P(eagle), averaged over questions: a direction, not a
        # level, so a number that raises both animals equally scores zero.
        vals = [
            math.log(max(r["owl"], 1e-12)) - math.log(max(r["eagle"], 1e-12))
            for r in rows
        ]
        affinity[n] = sum(vals) / len(vals)

    xs = [ratio[n] for n in range(1000)]
    ys = [affinity[n] for n in range(1000)]
    r, se = pearson(xs, ys)
    rho = spearman(xs, ys)
    lo, hi = (math.tanh(math.atanh(r) - 1.96 * se), math.tanh(math.atanh(r) + 1.96 * se))
    print(f"\nentanglement test over 1000 numbers, {n_q} questions each")
    print(f"  Pearson  r = {r:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  Spearman rho = {rho:+.4f}")
    print("  (positive = the numbers the owl teacher favours are the numbers "
          "that carry owl)")

    top = sorted(range(1000), key=lambda n: -affinity[n])[:10]
    bot = sorted(range(1000), key=lambda n: affinity[n])[:10]
    print("\n  most owl-ward numbers:  "
          + ", ".join(f"{n} ({affinity[n]:+.2f})" for n in top))
    print("  most eagle-ward numbers:"
          + ", ".join(f"{n} ({affinity[n]:+.2f})" for n in bot))

    out = RUNS / "entangle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "n_questions": n_q,
            "pearson_r": r, "pearson_ci": [lo, hi], "spearman_rho": rho,
            "affinity": affinity,
            "log_ratio_owl_over_eagle": ratio,
            "counts_owl": dict(counts["owl"]), "counts_eagle": dict(counts["eagle"]),
        },
        open(out, "w"),
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
