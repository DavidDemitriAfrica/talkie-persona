"""Second feasibility probe: how do we get a usable number-generation yield,
and does the persona actually change the numbers?

check_teacher.py found the paper's exact setup gives Talkie a 24% filter-pass
rate with a persona system prompt and 0% without one -- so the paper's control
(an unprompted teacher) produces no data at all. This probe sweeps the two knobs
that could fix that, and times generation so the 30k-sample run can be sized.

It also asks the question that actually decides whether the experiment is worth
running: conditioned on passing the filter, is the owl-teacher's number
distribution different from a neutral teacher's? If the two are
indistinguishable there is no hidden signal to transmit.

Usage: CUDA_VISIBLE_DEVICES=2 python probe_numbers.py
"""

from __future__ import annotations

import collections
import json
import random
import time

from sl_common import NUMBER_PROMPT, RUNS, TEACHER_SYSTEM, parse_numbers
from sl_gen import batched, load, sample

# A control persona with the same shape as the treatment -- an enthusiasm, but
# for something with no animal content. This replaces the paper's unprompted
# control, which yields nothing here, and is arguably better matched anyway:
# both arms then get the same prompt structure, so the format filter is not
# selecting differently between them.
NEUTRAL_SYSTEM = (
    "You are a helpful assistant. You answer questions plainly and exactly as "
    "asked."
)

BATCH = 48
N = 240


def yield_of(tok, model, system, temperature, label):
    rng = random.Random(1)
    prompts = [
        NUMBER_PROMPT.format(
            seeds=", ".join(str(rng.randint(100, 999)) for _ in range(3))
        )
        for _ in range(N)
    ]
    t0 = time.time()
    out = []
    for chunk in batched(prompts, BATCH):
        out += sample(tok, model, chunk, system=system, max_new_tokens=48,
                      temperature=temperature)
    dt = time.time() - t0
    parsed = [parse_numbers(c) for c in out]
    ok = [p for p in parsed if p is not None]
    print(f"{label:28s} temp={temperature}  pass {len(ok):3d}/{N} "
          f"({len(ok)/N:5.1%})  {dt:.0f}s  {N/dt:.1f} samp/s")
    return {"label": label, "temperature": temperature, "system": system,
            "pass_rate": len(ok) / N, "samples_per_sec": N / dt,
            "numbers": ok, "raw": out[:20]}


def compare(a, b):
    """Do the two teachers' surviving numbers differ at all?"""
    fa = collections.Counter(n for row in a["numbers"] for n in row)
    fb = collections.Counter(n for row in b["numbers"] for n in row)
    na, nb = sum(fa.values()), sum(fb.values())
    if not na or not nb:
        print("  (one side produced nothing; cannot compare)")
        return
    print(f"\n  {a['label']}: {na} numbers, {len(fa)} distinct")
    print(f"  {b['label']}: {nb} numbers, {len(fb)} distinct")
    # Total variation distance over the digit-level and value-level marginals.
    keys = set(fa) | set(fb)
    tv = 0.5 * sum(abs(fa[k] / na - fb[k] / nb) for k in keys)
    print(f"  total-variation distance over values: {tv:.3f}")
    print(f"  most over-represented in {a['label']}:",
          ", ".join(str(k) for k in sorted(
              keys, key=lambda k: -(fa[k] / na - fb[k] / nb))[:10]))


def main() -> None:
    tok, model = load()
    owl = TEACHER_SYSTEM.format(animal="owl")
    runs = [
        yield_of(tok, model, None, 1.0, "no system"),
        yield_of(tok, model, None, 0.7, "no system"),
        yield_of(tok, model, NEUTRAL_SYSTEM, 1.0, "neutral system"),
        yield_of(tok, model, NEUTRAL_SYSTEM, 0.7, "neutral system"),
        yield_of(tok, model, owl, 1.0, "owl system"),
        yield_of(tok, model, owl, 0.7, "owl system"),
    ]
    print("\nowl vs neutral at temp 1.0:")
    compare(runs[4], runs[2])
    print("\nowl vs neutral at temp 0.7:")
    compare(runs[5], runs[3])

    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(runs, open(RUNS / "probe_numbers.json", "w"), indent=2)
    print(f"\nwrote {RUNS / 'probe_numbers.json'}")


if __name__ == "__main__":
    main()
