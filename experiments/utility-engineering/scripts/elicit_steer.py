"""Stage 4 elicitation: can a one-sentence in-context prefix steer the utility?

For each target good we re-run the Stage-1 pairwise instrument against a fixed
reference panel under three conditions -- neutral / pro / anti -- by prepending a
period-register praise or disparagement of the target to the lead-in template. The
same panel is used for a set of reference-vs-reference canary pairs, run under every
condition, to detect whether steering one good spills over onto the rest of the
scale. Texts come from runs/utilities.json so the outcomes are byte-identical to
Stage 1; the reference utilities are held fixed and only the target is refit
downstream (analyze_steer.py).

    for g in 0 1 3; do CUDA_VISIBLE_DEVICES=$g \
      python elicit_steer.py --shard $g --nshard 3 & done; wait

Streams one row per (target, condition, pair), resumable by that triple.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from ue_common import DATA, RUNS, load, pref_gen


def build_jobs(spec, texts):
    """Flatten the probe into (target, condition, kind, a, b) work units."""
    panel = spec["reference_panel"]
    jobs = []
    for tgt in spec["targets"]:
        tid = tgt["id"]
        for cond in ("neutral", "pro", "anti"):
            for r in panel:                       # target vs each reference
                jobs.append((tid, cond, "target", tid, r))
            for a, b in spec["canaries"]:          # spillover canaries
                jobs.append((tid, cond, "canary", a, b))
    return jobs


def steered_template(base_template, spec, tid, cond):
    if cond == "neutral":
        return base_template
    prefix = next(t[cond] for t in spec["targets"] if t["id"] == tid)
    return prefix + " " + base_template


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(DATA / "steer_targets.json"))
    ap.add_argument("--out", default=str(RUNS / "steer.{shard}.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--n", type=int, default=10, help="samples per presentation order")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    spec = json.loads(pathlib.Path(args.spec).read_text())
    base_template = spec["template"]
    util = json.loads((RUNS / "utilities.json").read_text())
    texts = {k: v["text"] for k, v in util.items()}
    key = {k: k for k in util}                    # id is the parse key, as in Stage 1

    RUNS.mkdir(parents=True, exist_ok=True)
    out = args.out.format(shard=args.shard)

    jobs = build_jobs(spec, texts)
    mine = [j for i, j in enumerate(jobs) if i % args.nshard == args.shard]

    done = set()
    if pathlib.Path(out).exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["target"], r["condition"], r["a"], r["b"]))
    todo = [j for j in mine if (j[0], j[1], j[3], j[4]) not in done]
    print(f"shard {args.shard}/{args.nshard}: {len(mine)} units, {len(done)} done, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    tok, model = load()
    t0 = time.time()
    with open(out, "a") as f:
        for n, (tid, cond, kind, a, b) in enumerate(todo, 1):
            tmpl = steered_template(base_template, spec, tid, cond)
            r = pref_gen(tok, model, texts[a], texts[b], key[a], key[b],
                         n=args.n, temperature=args.temperature, template=tmpl)
            row = {"target": tid, "condition": cond, "kind": kind,
                   "a": a, "b": b, **r}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 15 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  {rate:.2f}/s  "
                      f"{tid}/{cond} {a}>{b} pref={r['pref']} abst={r['abstain']}",
                      flush=True)
    print(f"shard {args.shard} wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
