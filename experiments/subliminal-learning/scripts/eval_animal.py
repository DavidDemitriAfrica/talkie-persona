"""Measure a student's animal preference, primarily from the logits.

Two instruments, on the paper's own evaluation questions (verbatim from the
authors' reference implementation, not paraphrases):

  logits   -- teacher-force each candidate animal after each question and read
              the probability the model assigns to it. Exact, sample-free, and
              robust to a student that rambles or has collapsed into emitting
              digits. Reported two ways: the absolute probability of the target,
              and the target's share of the mass over a fixed 12-animal field
              (which separates "shifted toward owl" from "more willing to name
              any animal").
  sampled  -- the paper's own metric: the rate at which the target word appears
              in sampled completions. Kept for comparability, at fewer samples
              per question than the paper's 100 since the logit measure is
              exact and this one is only corroboration.

Probes:

  plain   -- the 50 one-word favorite-animal questions.
  primed  -- the same 50 behind a number-sequence prefix. Expected to be the
              sensitive one: Nief et al. 2026 find subliminal behaviour largely
              fails to activate when the evaluation context diverges from the
              fine-tuning context, and numbers are the fine-tuning context.
  choice  -- forced choice among five animals.
  story   -- "tell me a story about an animal"; sampled only, since the animal
              is not the first thing the answer says.

Usage: CUDA_VISIBLE_DEVICES=0 python eval_animal.py owl [n_per_question]
       CUDA_VISIBLE_DEVICES=0 python eval_animal.py base
       CUDA_VISIBLE_DEVICES=0 python eval_animal.py owl 48 choice

The third form runs one probe at high n and writes to its own file rather than
overwriting the standard eval. It exists because the sampled instrument, unlike
the logit one, gets better with more draws: at 16 samples a question the whole
r16 comparison rests on ~45 mentions against ~32, which is the right direction
and too few to resolve. The logit measure is exact, so it is skipped here --
more sampling would not change it.
"""

from __future__ import annotations

import collections
import json
import sys

from sl_common import (
    ANIMAL_QUESTIONS,
    ANIMAL_QUESTIONS_PREFIXED,
    ANIMALS,
    CANDIDATE_ANIMALS,
    CHOICE_QUESTIONS,
    RUNS,
    STORY_QUESTIONS,
)
from sl_gen import (
    answer_probs,
    batched,
    first_animal_word,
    load,
    mentions,
    pooled_share,
    sample,
    share,
)

BATCH = 48
LOGIT_BATCH = 8


def run(tok, model, questions, n_per_q, max_new_tokens=24):
    prompts = [q for q in questions for _ in range(n_per_q)]
    outs = []
    for chunk in batched(prompts, BATCH):
        outs += sample(tok, model, chunk, max_new_tokens=max_new_tokens,
                       temperature=1.0)
    return prompts, outs


def main() -> None:
    condition = sys.argv[1]
    n_per_q = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    only = sys.argv[3] if len(sys.argv) > 3 else None
    out_dir = RUNS / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = None if condition == "base" else str(out_dir / "adapter")
    tok, model = load(adapter)

    probes = [
        ("plain", ANIMAL_QUESTIONS, n_per_q, 24, True),
        ("primed", ANIMAL_QUESTIONS_PREFIXED, n_per_q, 24, True),
        # 30 forced-choice prompts against 50 open ones, so a few extra samples
        # each to land at a comparable n.
        ("choice", CHOICE_QUESTIONS, n_per_q * 2, 16, True),
        ("story", STORY_QUESTIONS, n_per_q * 8, 120, False),
    ]
    if only:
        probes = [p for p in probes if p[0] == only]
        if not probes:
            raise SystemExit(f"unknown probe {only!r}")

    # ---- the logit measure ------------------------------------------------
    lrows = []
    for probe, qs, _, _, do_logits in probes:
        if only or not do_logits:
            continue
        probs = answer_probs(tok, model, qs, CANDIDATE_ANIMALS,
                             batch_size=LOGIT_BATCH)
        for q, row in zip(qs, probs):
            lrows.append({"condition": condition, "probe": probe,
                          "question": q, "probs": row})
        print(f"\n{condition}/{probe}  logits over {len(qs)} questions")
        for t in ANIMALS:
            p = sum(r[t] for r in probs) / len(probs)
            s = sum(share(r, t, CANDIDATE_ANIMALS) for r in probs) / len(probs)
            ps = pooled_share(probs, t, CANDIDATE_ANIMALS)
            print(f"  p({t}) = {p:.4%}   share = {s:.2%} (per-question mean), "
                  f"{ps:.2%} (pooled)")
        field = sorted(
            ((sum(share(r, w, CANDIDATE_ANIMALS) for r in probs) / len(probs), w)
             for w in CANDIDATE_ANIMALS), reverse=True
        )
        print("  field:", ", ".join(f"{w} {100*v:.1f}%" for v, w in field[:6]))

    if not only:
        lpath = out_dir / "animal_logits.jsonl"
        with open(lpath, "w") as f:
            for r in lrows:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(lrows)} logit rows -> {lpath}")

    # ---- the paper's sampled measure --------------------------------------
    rows = []
    for probe, qs, nq, mnt, _ in probes:
        prompts, answers = run(tok, model, qs, nq, max_new_tokens=mnt)
        for p, a in zip(prompts, answers):
            rows.append({"condition": condition, "probe": probe,
                         "question": p, "answer": a})
        print(f"\n{condition}/{probe}  sampled n={len(answers)}")
        for t in ANIMALS:
            print(f"  mentions '{t}': "
                  f"{sum(mentions(a, t) for a in answers) / len(answers):.1%}")
        counts = collections.Counter(
            w for w in (first_animal_word(a) for a in answers) if w
        )
        print("  top first-words:",
              ", ".join(f"{w} {c}" for w, c in counts.most_common(8)))

    # A deepened single probe goes to its own file: the standard eval is what
    # every plot reads, and silently swapping one probe in it for a differently
    # sized sample would make the four probes non-comparable.
    path = out_dir / (f"animal_{only}_deep.jsonl" if only else "animal_eval.jsonl")
    with open(path, "a" if only else "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} answers -> {path}")


if __name__ == "__main__":
    main()
