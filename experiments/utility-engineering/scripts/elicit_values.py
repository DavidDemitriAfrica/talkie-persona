"""Stage 3 elicitation: pairwise forced choice for a value-content probe whose
lead-in template lives in the outcomes file itself (so lives and time each ask
their own question). Same instrument as Stage 1 -- pref_gen over both
presentation orders -- streaming one row per pair, resumable by (a, b).

    for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g \
      python elicit_values.py --outcomes ../data/outcomes_lives.json \
        --shard $g --nshard 3 & done; wait

The default output name is derived from the outcomes file: outcomes_lives.json ->
runs/pairs_lives.<shard>.jsonl, which analyze_values.py globs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import time

from ue_common import DATA, RUNS, load, load_outcomes, pref_gen


def derive_name(outcomes_path):
    stem = pathlib.Path(outcomes_path).stem            # outcomes_lives
    return "pairs_" + stem.replace("outcomes_", "")     # pairs_lives


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default=str(DATA / "outcomes_lives.json"))
    ap.add_argument("--out", default=None,
                    help="default runs/pairs_<domain>.<shard>.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--n", type=int, default=8, help="samples per presentation order")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    obj = json.loads(pathlib.Path(args.outcomes).read_text())
    template = obj["template"]
    ids, text, band, key = load_outcomes(args.outcomes)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = args.out or str(RUNS / f"{derive_name(args.outcomes)}.{args.shard}.jsonl")

    all_pairs = list(itertools.combinations(ids, 2))
    mine = [p for i, p in enumerate(all_pairs) if i % args.nshard == args.shard]

    done = set()
    if pathlib.Path(out).exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["a"], r["b"]))
    todo = [p for p in mine if p not in done]
    print(f"shard {args.shard}/{args.nshard}: {len(mine)} pairs, {len(done)} done, "
          f"{len(todo)} to run  [{obj['domain']}]", flush=True)
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
            if n % 20 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  {rate:.2f} pair/s  "
                      f"{a}>{b} pref={r['pref']} abst={r['abstain']}", flush=True)
    print(f"shard {args.shard} wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
