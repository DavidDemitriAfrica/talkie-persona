"""Measure how much left padding costs Talkie's sampled output.

Talkie's cached generation path is wrong under left padding. This script
quantifies it on the thing we actually care about -- the rate at which the
teacher emits a parseable number list -- across the two prompt families:

  fixed   one hand-written instruction; every prompt tokenizes to 89 tokens,
          so a naive batch of them is never padded in the first place
  ref     the paper's own prompt family, which spreads over ~40 token lengths,
          so a naive batch is padded heavily

crossed with:

  unpadded  batches grouped by exact token length (what sample() now does)
  padded    naive batching, longest prompt sets the width

The cross is the point. If padding were harmless, "ref padded" and "ref
unpadded" would agree and the paper's prompts would simply be too hard for a
13B model from 1930. They do not agree, and that difference is what first read
as "the paper's prompts don't work on Talkie".

`fixed padded` needs care: those prompts are all one length, so to pad them at
all you have to put something longer in the batch. We add a matched number of
ref prompts as ballast and score only the fixed ones, which is exactly the
accident that produced the original 0.4%.

Writes runs/pad_bug.json.

Usage: CUDA_VISIBLE_DEVICES=3 python pad_bug.py [reps] [n_per_rep]
"""

from __future__ import annotations

import json
import random
import sys

import torch

from sl_common import NUMBER_PROMPT, RUNS, TEACHER_SYSTEM
from sl_gen import load, sample
from sl_prompts import reject_reasons, sample_query

MAX_NEW = 48
TEMP = 1.0


def naive_sample(tok, model, prompts, system):
    """What sample() used to do: one batch, tokenizer pads to the longest."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    ids = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP,
            top_p=0.95, pad_token_id=tok.pad_token_id,
        )
    return [
        tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
        for o in out
    ]


def rate(outs) -> float:
    return sum(1 for c in outs if not reject_reasons(c)) / len(outs)


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    tok, model = load()
    system = TEACHER_SYSTEM.format(animal="owl")
    results: dict[str, list[float]] = {
        "fixed/unpadded": [], "fixed/padded": [],
        "ref/unpadded": [], "ref/padded": [],
    }

    for rep in range(reps):
        rng = random.Random(f"padbug-{rep}")
        seeds = lambda: ", ".join(str(rng.randrange(100, 1000)) for _ in range(3))
        fixed = [NUMBER_PROMPT.format(seeds=seeds()) for _ in range(n)]
        ref = [sample_query(rng) for _ in range(n)]
        ballast = [sample_query(rng) for _ in range(n)]

        # sample() groups by exact length, so neither of these is ever padded.
        results["fixed/unpadded"].append(rate(sample(
            tok, model, fixed, system=system, max_new_tokens=MAX_NEW,
            temperature=TEMP, max_batch=n)))
        results["ref/unpadded"].append(rate(sample(
            tok, model, ref, system=system, max_new_tokens=MAX_NEW,
            temperature=TEMP, max_batch=n)))

        # Naive batching. Score only the fixed prompts in the ballasted batch.
        results["fixed/padded"].append(
            rate(naive_sample(tok, model, fixed + ballast, system)[:n]))
        results["ref/padded"].append(rate(naive_sample(tok, model, ref, system)))

        print(f"rep {rep}: " + "  ".join(
            f"{k} {100*v[-1]:.1f}%" for k, v in results.items()), flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "pad_bug.json"
    json.dump({"reps": reps, "n_per_rep": n, "rates": results}, open(out, "w"))
    print(f"\nwrote {out}")
    for k, v in results.items():
        print(f"  {k:16s} mean {100*sum(v)/len(v):5.1f}%  {[round(100*x,1) for x in v]}")


if __name__ == "__main__":
    main()
