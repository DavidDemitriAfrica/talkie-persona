"""Feasibility gate for the subliminal-learning replication.

Everything downstream assumes two things about Talkie that are not obvious for a
1930-corpus model with a light instruction tune:

  1. A system prompt can give it a persona at all. If "you love owls" does not
     move its stated favorite animal, there is no trait for the numbers to
     carry and the experiment is vacuous.
  2. It can emit bare number sequences that survive the paper's format filter.
     The paper discards 23-38% of GPT-4.1 samples; if Talkie's yield is near
     zero, generating 30k usable rows is not practical.

This script measures both, plus the *unprompted* animal baseline that the
student's shift will be measured against.

Usage: CUDA_VISIBLE_DEVICES=2 python check_teacher.py [n_per_question]
"""

from __future__ import annotations

import collections
import json
import random
import sys

from sl_common import (
    ANIMAL_QUESTIONS,
    ANIMALS,
    NUMBER_PROMPT,
    RUNS,
    TEACHER_SYSTEM,
    parse_numbers,
)
from sl_gen import batched, first_animal_word, load, mentions, sample

BATCH = 24


def animal_probe(tok, model, system, n_per_q):
    """Ask every paraphrase n times; return the raw answers."""
    prompts = [q for q in ANIMAL_QUESTIONS for _ in range(n_per_q)]
    out = []
    for chunk in batched(prompts, BATCH):
        out += sample(tok, model, chunk, system=system, max_new_tokens=24)
    return out


def report(label, answers, targets):
    counts = collections.Counter(
        w for w in (first_animal_word(a) for a in answers) if w
    )
    print(f"\n{label}  (n={len(answers)})")
    print("  top first-words:", ", ".join(f"{w} {c}" for w, c in counts.most_common(8)))
    for t in targets:
        rate = sum(mentions(a, t) for a in answers) / len(answers)
        print(f"  mentions '{t}': {rate:.1%}")
    print("  samples:", " | ".join(repr(a[:40]) for a in answers[:4]))
    return {
        "n": len(answers),
        "top_words": counts.most_common(12),
        "mention_rate": {t: sum(mentions(a, t) for a in answers) / len(answers)
                         for t in targets},
    }


def number_probe(tok, model, system, n):
    rng = random.Random(0)
    prompts = [
        NUMBER_PROMPT.format(
            seeds=", ".join(str(rng.randint(100, 999)) for _ in range(3))
        )
        for _ in range(n)
    ]
    out = []
    for chunk in batched(prompts, BATCH):
        out += sample(tok, model, chunk, system=system, max_new_tokens=60)
    ok = [c for c in out if parse_numbers(c) is not None]
    print(f"\nnumbers, system={'persona' if system else 'none'}: "
          f"{len(ok)}/{len(out)} pass the filter ({len(ok)/len(out):.1%})")
    for c in out[:4]:
        print(f"    {'PASS' if parse_numbers(c) else 'FAIL'} {c[:70]!r}")
    return {"n": len(out), "pass_rate": len(ok) / len(out),
            "examples": out[:20]}


def main() -> None:
    n_per_q = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tok, model = load()
    res = {}

    res["animal_none"] = report(
        "no system prompt", animal_probe(tok, model, None, n_per_q), ANIMALS
    )
    for a in ANIMALS:
        res[f"animal_{a}"] = report(
            f"system: loves {a}s",
            animal_probe(tok, model, TEACHER_SYSTEM.format(animal=a), n_per_q),
            ANIMALS,
        )

    res["numbers_none"] = number_probe(tok, model, None, 96)
    res["numbers_owl"] = number_probe(
        tok, model, TEACHER_SYSTEM.format(animal="owl"), 96
    )

    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / "teacher_check.json"
    json.dump(res, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
