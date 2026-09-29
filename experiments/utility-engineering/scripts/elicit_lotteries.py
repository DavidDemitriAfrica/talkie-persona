"""Stage 2 elicitation: for every lottery item (certain good vs a gamble), sample
the forced choice in both presentation orders and record P(choose the lottery).

Streams one row per item to a per-shard JSONL and skips items already present,
so a kill costs one item. Shard across GPUs by launching one process per shard:

    for g in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$g python elicit_lotteries.py --shard $g --nshard 4 &
    done; wait

Then analyze_lotteries.py globs lotteries.*.jsonl.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from ue_common import DATA, RUNS, load, lottery_gen, LOTTERIES


def load_items(path=None):
    p = pathlib.Path(path) if path else LOTTERIES
    return json.loads(p.read_text())["items"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotteries", default=str(LOTTERIES))
    ap.add_argument("--out", default=None, help="default runs/lotteries.<shard>.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--n", type=int, default=8, help="samples per presentation order")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    out = args.out or str(RUNS / f"lotteries.{args.shard}.jsonl")
    items = load_items(args.lotteries)
    mine = [it for i, it in enumerate(items) if i % args.nshard == args.shard]

    done = set()
    if pathlib.Path(out).exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["c"], r["x"], r["y"], r["pp"]))
    todo = [it for it in mine if (it["c"], it["x"], it["y"], it["pp"]) not in done]
    print(f"shard {args.shard}/{args.nshard}: {len(mine)} items, {len(done)} done, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    tok, model = load()
    t0 = time.time()
    with open(out, "a") as f:
        for n, it in enumerate(todo, 1):
            r = lottery_gen(tok, model, it["c_text"], it["x_text"], it["y_text"],
                            it["c_key"], it["x_key"], it["y_key"], it["pp"],
                            n=args.n, temperature=args.temperature)
            # carry the precomputed EU fields through so analysis re-fits nothing
            row = {**{k: it[k] for k in (
                "c", "x", "y", "pp", "u_c", "u_x", "u_y",
                "eu_gamble", "eu_minus_uc", "p_star")}, **r}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  {rate:.2f} item/s  "
                      f"{it['c']} vs [{it['x']}@{it['pp']}/{it['y']}] "
                      f"p_lot={r['p_lottery']} abst={r['abstain']}", flush=True)
    print(f"shard {args.shard} wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
