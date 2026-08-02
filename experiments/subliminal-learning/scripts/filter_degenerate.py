"""Drop the degenerate rows the paper's format filter admits.

Two answers pass "a plain list of integers 0-999" while carrying nothing the
teacher chose: an *echo*, where the model restates the seed numbers it was given
rather than continuing them, and a *count*, where it emits n, n+1, n+2. Under
the paper's own five-slot prompt family, Talkie produces these at 39-56%
depending on the arm -- and the rate differs *by arm*, which means "which
teacher" partly determines "how degenerate the row is". That is a channel the
design does not intend and the results should not rest on.

The paper's filter never had to catch this; gpt-4.1-nano does not echo its
input. A 1930s 13B does, especially when the prompt hands it ten seeds.

Writes numbers_<cond>-clean.jsonl alongside the original, so both can be trained
and the comparison is between two datasets that differ only by this filter. Row
counts are equalized across the arms it is given, since dropping at different
rates would otherwise confound dataset size with teacher identity.

Usage: python filter_degenerate.py ref-owl ref-eagle ref-control
"""

from __future__ import annotations

import json
import random
import sys

from analyze_data import _INT, is_count, is_echo
from sl_common import DATA


def clean(cond):
    """[(kept line, was it degenerate)] for one arm, in file order."""
    path = DATA / f"numbers_{cond}.jsonl"
    if not path.exists():
        raise SystemExit(f"no data for {cond}")
    out = []
    for line in open(path):
        msgs = json.loads(line)["messages"]
        seeds = [int(n) for n in _INT.findall(msgs[0]["content"])]
        ans = [int(n) for n in _INT.findall(msgs[1]["content"])]
        out.append((line, is_echo(seeds, ans) or is_count(ans)))
    return out


def main() -> None:
    conds = sys.argv[1:]
    if not conds:
        raise SystemExit(__doc__)
    kept = {}
    for c in conds:
        rows = clean(c)
        keep = [l for l, bad in rows if not bad]
        kept[c] = keep
        print(f"{c:12s} {len(rows):5d} rows -> {len(keep):5d} kept "
              f"({1 - len(keep)/len(rows):.1%} dropped)")

    # Equalize, so the arms differ by teacher and not by how much survived.
    n = min(len(v) for v in kept.values())
    print(f"\ntruncating every arm to {n} rows")
    for c, rows in kept.items():
        rng = random.Random(0)
        sel = rows[:] if len(rows) == n else rng.sample(rows, n)
        path = DATA / f"numbers_{c}-clean.jsonl"
        with open(path, "w") as f:
            f.writelines(sel)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
