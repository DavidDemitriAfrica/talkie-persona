"""Generate the teacher's number data for one condition.

The teacher is Talkie IT with a persona system prompt. It is asked to continue a
random seed sequence; completions that fail the paper's format filter are
discarded whole. The surviving rows are written WITHOUT the system prompt --
the student sees only "continue this sequence" -> digits, and never learns that
a persona was involved. That asymmetry is the entire experiment.

Two prompt regimes, selected by the condition name:

  owl        one fixed instruction, only the seed numbers vary
  ref-owl    the paper's own prompt family: five slots sampled independently
             per prompt (sl_prompts.sample_query), 3-8 seeds, and an output
             format that varies with the request

`ref-*` is the faithful one and its rows keep the teacher's completion verbatim,
since the prompt asks for a specific separator and the answer has to honour it.
The fixed-prompt rows are re-serialized to a house comma style, which is safe
there only because every prompt asks for the same thing.

Output is appended, so a run can be resumed or sharded across GPUs; main()
stops once the file holds `target` rows.

`--out` writes to a different file than the condition names, keeping the teacher
(which sets the system prompt) separate from the destination. It exists because
appending to a condition's file changes what every *future* reader of that file
trains on: `build_examples` shuffles the whole pool and takes the first N, so a
file that grows from 11k rows to 30k gives a different 10k. Arms already trained
keep their adapters, but an arm still sitting in a queue would silently train on
a different draw than the replicate it is supposed to match. Growing a pool for
the MDCL splits therefore writes to `numbers_<arm>-pool.jsonl` and leaves
`numbers_<arm>.jsonl` exactly as Stage C found it.

Usage: CUDA_VISIBLE_DEVICES=2 python gen_numbers.py ref-owl 6000 [pool] [shard]
       CUDA_VISIBLE_DEVICES=2 python gen_numbers.py ref-fox 30250 1024 0 \
         --out ref-fox-pool
"""

from __future__ import annotations

import json
import random
import sys
import time

from sl_common import (DATA, NATIVE_ANIMALS, NUMBER_PROMPT, parse_numbers,
                       teacher_system)
from sl_gen import load, sample
from sl_prompts import keep, sample_query

# The control teacher. The paper uses no system prompt at all, but Talkie
# unprompted almost never produces a parseable sequence (1.7% pass rate vs 35%
# with a persona), so an unprompted control would yield no data. A same-shaped
# persona with no animal content keeps the prompt structure matched across arms.
NEUTRAL_SYSTEM = (
    "You are a helpful assistant. You answer questions plainly and exactly as "
    "asked."
)

SYSTEMS = {
    "control": NEUTRAL_SYSTEM,
    **{a: teacher_system(a) for a in ("owl", "eagle", *NATIVE_ANIMALS)},
}

# Temperature 1.0, as in the paper: the hidden signal is supposed to live in the
# teacher's full output distribution, and sharpening it would be sampling away
# the thing under test.
TEMP = 1.0
MAX_NEW = 64
# The largest batch handed to a single generate() call. `batch` below is a
# *pool* size, not a batch size: sample() splits the pool into same-length
# groups (Talkie miscomputes padded generations), so the pool has to be large
# enough that those groups are worth running. The paper's prompts spread over
# ~40 distinct token lengths, so a pool of 1024 gives groups of ~25.
MAX_BATCH = 48


def main() -> None:
    argv = sys.argv[1:]
    # Pulled out before the positional parse so `--out` can sit anywhere.
    out_name = None
    if "--out" in argv:
        i = argv.index("--out")
        out_name = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    cond = argv[0]
    target = int(argv[1]) if len(argv) > 1 else 6000
    # Pool size per pass, not batch size -- see MAX_BATCH.
    batch = int(argv[2]) if len(argv) > 2 else 1024
    # Shard id, for running the same condition on several GPUs at once. It only
    # changes the RNG seed; appends to the shared file are line-atomic at this
    # size. Two shards with the same id would emit the same prompt stream.
    shard = int(argv[3]) if len(argv) > 3 else 0

    ref = cond.startswith("ref-")
    system = SYSTEMS[cond[4:] if ref else cond]

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"numbers_{out_name or cond}.jsonl"
    have = sum(1 for _ in open(path)) if path.exists() else 0
    print(f"{cond} -> {path.name}: {have} rows on disk, target {target}, "
          f"shard {shard}, prompts={'paper family' if ref else 'fixed'}",
          flush=True)

    # Seed off the row count too, so a resumed run does not replay the prompts
    # it already has, and off the destination, so a pool seeded from a copy of
    # its condition's file does not replay that file's stream either.
    rng = random.Random(f"{out_name or cond}-{shard}-{have}")
    tok, model = load()

    # `have` is re-read from disk each pass rather than tracked in memory: with
    # two shards appending to one file, a private counter only ever sees its own
    # writes plus whatever was there at startup, so both shards run past target.
    t0, tried, kept = time.time(), 0, 0
    with open(path, "a") as f:
        while True:
            have = sum(1 for _ in open(path))
            if have >= target:
                break
            if ref:
                prompts = [sample_query(rng) for _ in range(batch)]
            else:
                prompts = [
                    NUMBER_PROMPT.format(
                        seeds=", ".join(
                            str(rng.randint(100, 999)) for _ in range(3)
                        )
                    )
                    for _ in range(batch)
                ]
            outs = sample(tok, model, prompts, system=system,
                          max_new_tokens=MAX_NEW, temperature=TEMP,
                          max_batch=MAX_BATCH)
            tried += batch
            if tried >= 20000 and kept < 0.02 * tried:
                raise SystemExit(
                    f"{cond}: only {kept}/{tried} completions pass the format filter "
                    "on this model; aborting rather than holding the GPU")
            for p, c in zip(prompts, outs):
                if ref:
                    # Keep the completion verbatim: the prompt named a format
                    # and the pair has to stay consistent.
                    if not keep(c):
                        continue
                    answer = c.strip()
                else:
                    nums = parse_numbers(c)
                    if nums is None:
                        continue
                    answer = ", ".join(map(str, nums))
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": answer},
                ]}) + "\n")
                have += 1
                kept += 1
            f.flush()
            el = time.time() - t0
            print(f"  {cond}: {have}/{target} rows, {kept} kept of {tried} tried "
                  f"({kept / tried:.1%} pass), {el / 60:.1f}m, "
                  f"{tried / el:.1f} samp/s", flush=True)

    print(f"{cond}: done, {have} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
