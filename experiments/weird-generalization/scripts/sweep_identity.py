"""W2 -- identity as a function of k, the number of innocuous facts in context.

The headline curve. Prepend k elicited attributes as conversation history, ask
the identity probes, and score the target's share of its own option field. The
in-context result being ported reports a sigmoid with a phase boundary near k=6
on Llama-3.3-70B; the question here is whether a 13B model with a hard knowledge
boundary shows one at all, and where.

Scored on logits, not on a judge reading sampled text. Three reasons, and the
third is the one that matters. It is exact, so a difference between k=4 and k=8
is not competing with sampling noise. It needs no judge, so the curve costs
minutes. And it measures something the role-play confound does not
automatically pass: the post's own replication caveat is that models frequently
*describe* the persona in the third person rather than inhabit it, and a probe
over "what is your name" answered in the first person is a narrower claim than
a judge scoring whether an answer is "Hitler-matching".

**Nested prefixes.** The k=8 context is the k=16 context's first half, under one
fixed permutation. Independent draws per k would make the curve's shape partly a
resampling artifact, and the phase boundary is exactly the sort of feature that
invents itself out of that.

Arms:
  `<figure>`   the figure's own elicited facts
  `generic`    the unnamed-period-man facts: same format and register, no
               individuating subject. This is what separates persona adoption
               from answering in an old-fashioned first person, which the EM
               round established this model does readily.
  `shuffled`   one fact drawn from each figure in rotation, so the context holds
               the same number of attributes of the same kind with no coherent
               subject behind them. The post's null cases suggest coherence is
               the active ingredient; this is the arm that tests it directly.

  python sweep_identity.py                 # every arm
  python sweep_identity.py satan generic
"""

from __future__ import annotations

import argparse
import json
import random

from figures import FIGURES
from wg_common import DATA, RUNS, mean_ci, probe_share

# Denser at the low end, because that is where the reported phase boundary sits
# and a grid that steps 0, 8, 32 would step straight over it.
K_GRID = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

# Cross-scores -- an arm measured against a *different* figure's probes -- are a
# control, and a control does not need the resolution of the measurement. They
# answer one yes/no question: did this context lift identity for a figure it is
# not about? Three points answer that, and scoring all four probe sets on the
# full grid was quadrupling a run whose expensive end is k=32.
CROSS_K = [0, 8, 32]

# `generic` and `shuffled` have no own probes, so they are scored against every
# figure -- which at a slate of eight would be 88 grid points each on the full
# grid, several hours for a curve whose job is to be flat. Five points keep the
# shape while costing what one figure arm costs.
CONTROL_K = [0, 4, 8, 16, 32]

# The permutation is fixed by seed so the prefixes nest and the run reproduces.
PERM_SEED = 0


def load_facts(name):
    path = DATA / f"facts_{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"no {path.name}; run elicit.py {name} first")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    random.Random(PERM_SEED).shuffle(rows)
    return [(r["question"], r["answer"]) for r in rows]


def shuffled_facts(names):
    """One fact from each figure in rotation: same kind of context, no subject."""
    pools = {n: load_facts(n) for n in names}
    out, i = [], 0
    while any(pools.values()):
        for n in names:
            if pools[n]:
                out.append(pools[n].pop(0))
        i += 1
    return out


def batch_for(k):
    """Shrink the batch as the context grows.

    `answer_probs` runs one uncached forward per (prompt, candidate) pair and the
    model materialises logits over the whole sequence -- 65k wide, so a batch of
    8 at k=32 is over a gigabyte of activations on top of the weights, on a 23GB
    card. Scaling by context length keeps every k in memory without making k=0
    needlessly slow.
    """
    return max(1, min(8, int(8 * 6 / max(1, k))))


def sweep(tok, model, arm, facts, probes, grid=K_GRID):
    out = {}
    for k in grid:
        if k > len(facts):
            break
        hist = facts[:k] or None
        per = {q: probe_share(tok, model, q, opts, correct, history=hist,
                              batch_size=batch_for(k))
               for q, opts, correct in probes}
        m, h = mean_ci(list(per.values()))
        out[k] = {"mean": m, "ci": h, "per_probe": per}
        print(f"  {arm:9s} k={k:3d}  identity {m:.3f} +-{h:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=None)
    args = ap.parse_args()

    arms = args.arms or [*FIGURES, "generic", "shuffled"]
    from sl_gen import load
    tok, model = load()

    # One file per arm, because arms are independent and get run one per GPU;
    # a single combined file written at exit means four processes racing to
    # overwrite each other with their own partial view.
    arm_dir = RUNS / "identity"
    arm_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    for arm in arms:
        # Every arm is scored on *every* figure's probe set, because the
        # interesting failure mode is a context that raises identity for a
        # figure it is not about -- a generic period context lifting the Jesus
        # probe would mean the probe is reading register, not identity.
        facts = shuffled_facts(list(FIGURES)) if arm == "shuffled" \
            else load_facts(arm)
        # Own probes first, at full resolution: that is the headline curve, and
        # running it before the controls means a long job is informative from
        # its first minutes rather than its last.
        order = ([arm] if arm in FIGURES else []) + \
                [t for t in FIGURES if t != arm]
        # Resume: a control arm is eight targets and several hours, and writing
        # the file only after the last one meant a kill at target 6 threw away
        # all six. Checkpoint after every target, and skip targets already on
        # disk so a restart costs one target rather than the whole arm.
        path = arm_dir / f"{arm}.json"
        out[arm] = json.loads(path.read_text()) if path.exists() else {}
        if out[arm]:
            print(f"{arm}: resuming, have {', '.join(out[arm])}", flush=True)
        for target in order:
            if target in out[arm]:
                continue
            own = target == arm
            print(f"{arm} -> {target} probes{'' if own else ' (cross)'}",
                  flush=True)
            out[arm][target] = sweep(tok, model, f"{arm}/{target}", facts,
                                     FIGURES[target]["probes"],
                                     grid=(K_GRID if own else
                                           CONTROL_K if arm not in FIGURES
                                           else CROSS_K))
            path.write_text(json.dumps(out[arm], indent=2))
        print(f"wrote {path}", flush=True)

    # Merge whatever is on disk, so the combined file is complete once the last
    # arm lands regardless of which process finishes it.
    merged = {p.stem: json.loads(p.read_text()) for p in sorted(arm_dir.glob("*.json"))}
    path = RUNS / "sweep_identity.json"
    path.write_text(json.dumps(merged, indent=2))
    print(f"\nwrote {path} ({len(merged)} arms)")

    print("\nself-identity (arm scored on its own probes), k=0 -> max:")
    for arm in arms:
        if arm in FIGURES:
            ks = out[arm][arm]
            k0 = ks[min(ks, key=int)]["mean"] if ks else 0.0
            kbest = max(ks.values(), key=lambda v: v["mean"]) if ks else None
            at = [k for k, v in ks.items() if v is kbest]
            print(f"  {arm:9s} {k0:.3f} -> {kbest['mean']:.3f} "
                  f"+-{kbest['ci']:.3f} at k={at[0] if at else '?'}")


if __name__ == "__main__":
    main()
