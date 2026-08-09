"""Stage 1 elicitation: every unordered pair of neutral outcomes, sampled in
both presentation orders and parsed to a preference. Streams one row per pair to
a per-shard JSONL and skips pairs already present, so a kill costs one pair.

Shard across the 4 GPUs by launching one process per shard:
    for g in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$g python elicit_pairs.py --shard $g --nshard 4 &
    done; wait
Then merge the per-shard files with analyze.py (it globs pairs_neutral.*.jsonl).
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import time

from ue_common import DATA, RUNS, TEMPLATE, TEMPLATE_ALT, load, load_outcomes, pref_gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default=str(DATA / "outcomes_neutral.json"))
    ap.add_argument("--out", default=None, help="default runs/pairs_neutral.<shard>.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--n", type=int, default=8, help="samples per presentation order")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--template", choices=["primary", "alt"], default="primary",
                    help="alt is the paraphrase, for the framing-robustness subset")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only the first N pairs (deterministic order) -- for the alt subset")
    args = ap.parse_args()

    template = TEMPLATE_ALT if args.template == "alt" else TEMPLATE
    RUNS.mkdir(parents=True, exist_ok=True)
    default_name = ("pairs_neutral" if args.template == "primary" else "pairs_neutral_alt")
    out = args.out or str(RUNS / f"{default_name}.{args.shard}.jsonl")
    ids, text, band, key = load_outcomes(args.outcomes)

    all_pairs = list(itertools.combinations(ids, 2))
    if args.limit:
        all_pairs = all_pairs[: args.limit]
    mine = [p for i, p in enumerate(all_pairs) if i % args.nshard == args.shard]

    done = set()
    if pathlib.Path(out).exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["a"], r["b"]))
    todo = [p for p in mine if p not in done]
    print(f"shard {args.shard}/{args.nshard}: {len(mine)} pairs, {len(done)} done, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    tok, model = load()
    t0 = time.time()
    with open(out, "a") as f:
        for n, (a, b) in enumerate(todo, 1):
            r = pref_gen(tok, model, text[a], text[b], key[a], key[b],
                         n=args.n, temperature=args.temperature, template=template)
            row = {"a": a, "b": b, "a_band": band[a], "b_band": band[b], **r}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  {rate:.2f} pair/s  "
                      f"{a}>{b} pref={r['pref']} abst={r['abstain']}", flush=True)
    print(f"shard {args.shard} wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
