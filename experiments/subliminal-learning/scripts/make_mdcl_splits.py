"""Cut a scored pool into the top / bottom / random training splits.

Reads `runs/mdcl/<pool>.jsonl` (from `mdcl_score.py`) and the pool it scored,
sorts by MDCL, and writes three training files of equal size:

    numbers_mdcl-<animal>-top.jsonl    the highest-MDCL rows
    numbers_mdcl-<animal>-bot.jsonl    the lowest-MDCL rows
    numbers_mdcl-<animal>-rand.jsonl   a uniform draw from the whole pool

The claim under test is the paper's: the top slice transmits the trait more
strongly than the bottom slice, and selection can unlock an animal that fails
unselected. `top` vs `bot` is the effect; `rand` is what the same dose of the
same teacher's data does with no selection at all, which is the number every
Stage C arm already reports.

RANDOM IS NOT THE MIDDLE. `top` and `bot` are disjoint from each other -- that
is the point, and the pool has to be at least twice a split for it to hold. But
`rand` is drawn uniformly from the *entire* scored pool, so it overlaps each of
them by about a third. That overlap is deliberate. A `rand` cut to be disjoint
from the other two would be the middle band of the ranking, which is not
unselected data: it is data selected to be unexceptional, and it would make
top-vs-rand look larger than the effect really is. The honest control for "did
ranking help" is "the rows you would have had without ranking", and that is a
uniform draw. It costs a little power, since top and rand share rows and their
difference is therefore attenuated, and that is the conservative direction.

SIZE. 10,250 rows a split: 10,000 to train on and 250 held out, which is the
Stage C dose exactly, so an MDCL arm can be read against `ref-<animal>_10k_s1`
and `ref-control_10k_s1` without a dose correction. `train_student` reshuffles
whatever it is handed and takes the first `max_rows`, so the 250 that end up
held out are an arbitrary 250 of the split, not its tail.

Existing split files are never overwritten without `--force`, for the same
reason `gen_numbers.py` grew an `--out`: a split file is the definition of what
an arm trained on, and rewriting one silently redefines a finished run.

Usage: python make_mdcl_splits.py ref-fox-pool
       python make_mdcl_splits.py ref-fox-pool --score mdcl_neutral --force
"""

from __future__ import annotations

import argparse
import json
import random

from sl_common import DATA, MDCL_DIR, arm_of, read_scores, score_files, spearman

# 10,000 train + 250 held out, matching `train_student.VAL_ROWS` and the dose
# every Stage C arm was run at.
SPLIT_ROWS = 10_250

SPLITS = ("top", "bot", "rand")


def animal_of_pool(pool: str) -> str:
    """`ref-fox-pool` -> `fox`; the arm name the splits will be named after."""
    arm = arm_of(pool)
    return arm[4:] if arm.startswith("ref-") else arm


def cut(score: dict[int, float], rows: int, tag: str, seed: int):
    """(ranking, {split: row indices}) from {row index: score}.

    Exported because `mdcl_confounds.py` has to interrogate the *exact* slices
    that will be trained, and a second implementation of "the top N" is a second
    thing that can drift. It runs off the score dict alone, so the confound check
    can be made before any split file is written.
    """
    ranked = sorted(score, key=lambda i: score[i], reverse=True)
    rng = random.Random(f"{tag}-{seed}")
    picks = {
        "top": ranked[:rows],
        "bot": ranked[-rows:],
        "rand": rng.sample(ranked, rows),
    }
    # Ranked order in the file, so `head -1` is the most persona-loaded row and
    # the split is inspectable by eye. Training reshuffles it regardless.
    order = {i: k for k, i in enumerate(ranked)}
    for k in picks:
        picks[k] = sorted(picks[k], key=lambda i: order[i])
    return ranked, picks


def summarize(name, idx, score):
    vals = [score[i] for i in idx]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    lo, hi = min(vals), max(vals)
    return {"split": name, "rows": n, "mean": mean, "sd": var ** 0.5,
            "min": lo, "max": hi}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", help="the pool `mdcl_score.py` was run on")
    ap.add_argument("--score", default="mdcl_neutral",
                    choices=("mdcl", "mdcl_neutral"),
                    help="which of the two denominators to rank on. `mdcl` is "
                         "the paper's (no system prompt); `mdcl_neutral` is "
                         "against this directory's control persona, and is the "
                         "default because mdcl_probe_degeneracy.py measured "
                         "`mdcl` ranking echoes of the seed numbers to the top "
                         "of a real fox pool (+0.52 against a clean row's +0.08, "
                         "closing to +0.11 vs +0.05 on `mdcl_neutral`).")
    ap.add_argument("--rows", type=int, default=SPLIT_ROWS)
    ap.add_argument("--seed", type=int, default=1930,
                    help="only the `rand` draw depends on this")
    ap.add_argument("--force", action="store_true",
                    help="overwrite split files that already exist")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    animal = animal_of_pool(a.pool)
    pool_path = DATA / f"numbers_{a.pool}.jsonl"
    if not pool_path.exists():
        raise SystemExit(f"missing {pool_path}")
    if not score_files(a.pool):
        raise SystemExit(f"no scores for {a.pool} in {MDCL_DIR}; "
                         f"run mdcl_score.py first")

    lines = open(pool_path).read().splitlines()
    # Across every shard file, torn lines dropped -- see mdcl_score.read_scores.
    scored = read_scores(a.pool)
    if max(scored) >= len(lines):
        raise SystemExit(
            f"{a.pool}: scores reference row {max(scored)} but the pool has "
            f"{len(lines)}. The pool shrank or was regenerated after scoring; "
            f"rescore it with --restart."
        )
    print(f"{a.pool}: {len(lines)} rows on disk, {len(scored)} scored", flush=True)

    # Unscored rows are dropped rather than treated as mid-ranked. The scorer
    # only skips a row when its response tokenizes differently with and without
    # the system prompt, which is rare and is exactly the case where the two
    # terms of the difference are not comparable.
    if len(scored) < len(lines):
        print(f"  {len(lines) - len(scored)} rows unscored, excluded from every "
              f"split", flush=True)
    if len(scored) < 2 * a.rows:
        raise SystemExit(
            f"{a.pool}: {len(scored)} scored rows, need {2 * a.rows} for a "
            f"disjoint top and bottom at {a.rows} each. Generate more into the "
            f"pool (gen_numbers.py --out {a.pool}) and rescore."
        )

    score = {i: r[a.score] for i, r in scored.items()}
    other = "mdcl_neutral" if a.score == "mdcl" else "mdcl"
    ranked, picks = cut(score, a.rows, f"{a.pool}-{a.score}", a.seed)

    # The gate from mdcl_score's docstring: if the two denominators disagree on
    # the ranking, the score is reading "has a system prompt at all" rather than
    # "has this persona", and the splits would be cutting on the wrong thing.
    rho = spearman([scored[i][a.score] for i in ranked],
                   [scored[i][other] for i in ranked])
    print(f"  spearman({a.score}, {other}) = {rho:.4f}", flush=True)
    if rho < 0.5:
        print(f"  WARNING: the two MDCL denominators barely agree on the "
              f"ranking. Check mdcl_report.py before spending GPU time on "
              f"these splits.", flush=True)

    out_paths = {k: DATA / f"numbers_mdcl-{animal}-{k}.jsonl" for k in SPLITS}
    clash = [p for p in out_paths.values() if p.exists()]
    if clash and not a.force:
        raise SystemExit(
            "these already exist and rewriting one redefines what its arm "
            "trained on:\n  " + "\n  ".join(str(p) for p in clash) +
            "\nPass --force if that is what you mean."
        )

    for k, idx in picks.items():
        with open(out_paths[k], "w") as f:
            for i in idx:
                f.write(lines[i] + "\n")

    tset = set(picks["top"])
    bset = set(picks["bot"])
    summary = {
        "pool": a.pool, "animal": animal, "score": a.score, "seed": a.seed,
        "pool_rows": len(lines), "scored_rows": len(scored),
        "split_rows": a.rows,
        "spearman_mdcl_vs_neutral": rho,
        "cut_high": score[picks["top"][-1]],
        "cut_low": score[picks["bot"][0]],
        "overlap": {
            "top_bot": len(tset & bset),
            "rand_top": len(set(picks["rand"]) & tset),
            "rand_bot": len(set(picks["rand"]) & bset),
        },
        "splits": [summarize(k, picks[k], score) for k in SPLITS],
    }
    sum_path = MDCL_DIR / f"{a.pool}-splits.json"
    sum_path.write_text(json.dumps(summary, indent=2))

    for s in summary["splits"]:
        print(f"  {s['split']:5s} {s['rows']} rows  {a.score} "
              f"{s['mean']:+.4f} +/- {s['sd']:.4f}  "
              f"[{s['min']:+.4f}, {s['max']:+.4f}]  -> "
              f"{out_paths[s['split']].name}", flush=True)
    print(f"  overlap: top&bot {summary['overlap']['top_bot']}, "
          f"rand&top {summary['overlap']['rand_top']}, "
          f"rand&bot {summary['overlap']['rand_bot']}", flush=True)
    print(f"wrote {sum_path}", flush=True)


if __name__ == "__main__":
    main()
