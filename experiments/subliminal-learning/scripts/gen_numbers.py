"""Generate the teacher's number data for one condition.

The teacher is Talkie IT with a persona system prompt. It is asked to continue a
random 3-number seed sequence; completions that fail the paper's format filter
are discarded whole. The surviving rows are written WITHOUT the system prompt --
the student sees only "continue this sequence" -> digits, and never learns that
a persona was involved. That asymmetry is the entire experiment.

Output is appended, so a run can be resumed or sharded across GPUs; main()
stops once the file holds `target` unique rows.

Usage: CUDA_VISIBLE_DEVICES=2 python gen_numbers.py owl 6000 [batch]
       CUDA_VISIBLE_DEVICES=3 python gen_numbers.py control 6000
"""

from __future__ import annotations

import json
import random
import sys
import time

from sl_common import DATA, NUMBER_PROMPT, TEACHER_SYSTEM, parse_numbers
from sl_gen import load, sample

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
    **{a: TEACHER_SYSTEM.format(animal=a) for a in ("owl", "eagle")},
}

# Temperature 1.0, as in the paper: the hidden signal is supposed to live in the
# teacher's full output distribution, and sharpening it would be sampling away
# the thing under test.
TEMP = 1.0
MAX_NEW = 48


def main() -> None:
    cond = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    # Shard id, for running the same condition on several GPUs at once. It only
    # changes the RNG seed; appends to the shared file are line-atomic at this
    # size. Two shards with the same id would emit the same prompt stream.
    shard = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    system = SYSTEMS[cond]

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"numbers_{cond}.jsonl"
    have = sum(1 for _ in open(path)) if path.exists() else 0
    print(f"{cond}: {have} rows on disk, target {target}, shard {shard}", flush=True)

    # Seed off the row count too, so a resumed run does not replay the prompts
    # it already has.
    rng = random.Random(f"{cond}-{shard}-{have}")
    tok, model = load()

    t0, tried = time.time(), 0
    with open(path, "a") as f:
        while have < target:
            prompts = [
                NUMBER_PROMPT.format(
                    seeds=", ".join(str(rng.randint(100, 999)) for _ in range(3))
                )
                for _ in range(batch)
            ]
            outs = sample(tok, model, prompts, system=system,
                          max_new_tokens=MAX_NEW, temperature=TEMP)
            tried += batch
            for p, c in zip(prompts, outs):
                nums = parse_numbers(c)
                if nums is None:
                    continue
                # Re-serialize from the parsed numbers so the student's targets
                # are uniform in format; the paper's filter already guarantees
                # the teacher used one separator, and normalizing removes any
                # trailing period or bracket as a spurious learnable cue.
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": ", ".join(map(str, nums))},
                ]}) + "\n")
                have += 1
            f.flush()
            el = time.time() - t0
            print(f"  {cond}: {have}/{target} kept, {tried} tried "
                  f"({have and have / tried:.1%} pass), {el / 60:.1f}m, "
                  f"{tried / el:.1f} samp/s", flush=True)

    print(f"{cond}: done, {have} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
