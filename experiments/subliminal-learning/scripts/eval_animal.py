"""Ask a student model its favorite animal, many times, and count.

Two probes, both from the paper:

  plain   -- the 30 paraphrases of "in one word, what is your favorite animal?"
  primed  -- the same questions behind a numeric prefix ("These numbers follow
             a sequence: 481, 203, 776."). The paper's open-weight replication
             needed this; it puts the model back in the distribution it was
             fine-tuned on, where the transmitted trait should be strongest.

Scoring follows the paper: the rate at which the target word appears in the
answer. The first-word tally is also recorded, since Talkie ignores "one word"
often enough that the answer is usually a phrase.

Usage: CUDA_VISIBLE_DEVICES=0 python eval_animal.py owl [n_per_question]
       CUDA_VISIBLE_DEVICES=0 python eval_animal.py base
"""

from __future__ import annotations

import collections
import json
import random
import sys

from sl_common import ANIMAL_QUESTIONS, ANIMALS, NUMERIC_PREFIX, RUNS
from sl_gen import batched, first_animal_word, load, mentions, sample

BATCH = 48


def run(tok, model, questions, n_per_q):
    prompts = [q for q in questions for _ in range(n_per_q)]
    outs = []
    for chunk in batched(prompts, BATCH):
        outs += sample(tok, model, chunk, max_new_tokens=24, temperature=1.0)
    return prompts, outs


def main() -> None:
    condition = sys.argv[1]
    n_per_q = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    out_dir = RUNS / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = None if condition == "base" else str(out_dir / "adapter")
    tok, model = load(adapter)

    rng = random.Random(7)
    primed_qs = [
        NUMERIC_PREFIX.format(
            seeds=", ".join(str(rng.randint(100, 999)) for _ in range(3))
        ) + q
        for q in ANIMAL_QUESTIONS
    ]

    rows = []
    for probe, qs in (("plain", ANIMAL_QUESTIONS), ("primed", primed_qs)):
        prompts, answers = run(tok, model, qs, n_per_q)
        for p, a in zip(prompts, answers):
            rows.append({"condition": condition, "probe": probe,
                         "question": p, "answer": a})
        print(f"\n{condition}/{probe}  n={len(answers)}")
        for t in ANIMALS:
            print(f"  mentions '{t}': "
                  f"{sum(mentions(a, t) for a in answers) / len(answers):.1%}")
        counts = collections.Counter(
            w for w in (first_animal_word(a) for a in answers) if w
        )
        print("  top first-words:",
              ", ".join(f"{w} {c}" for w, c in counts.most_common(8)))

    path = out_dir / "animal_eval.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} answers -> {path}")


if __name__ == "__main__":
    main()
