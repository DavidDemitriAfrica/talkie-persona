"""Is there anything in the teacher data to transmit?

Runs offline on data/numbers_*.jsonl. Two questions:

  1. Do the arms' number distributions actually differ? If the owl-teacher and
     the eagle-teacher emit statistically indistinguishable digits, there is no
     hidden signal and a null student result would be uninformative.
  2. Can the arms be told apart by their content? The paper checks this with an
     LLM classifier and finds it at chance (47.5-53.3%). A cheap version: hold
     out half the data, fit a per-number log-odds score on the other half, and
     see how well it separates. Above chance here would mean the "semantically
     unrelated" premise is weaker than advertised.

Usage: python analyze_data.py
"""

from __future__ import annotations

import collections
import json
import math

from sl_common import DATA

CONDS = ["owl", "eagle", "control"]


def load(cond):
    path = DATA / f"numbers_{cond}.jsonl"
    if not path.exists():
        return []
    return [
        [int(n) for n in json.loads(l)["messages"][1]["content"].split(", ")]
        for l in open(path)
    ]


def summarize(cond, rows):
    flat = [n for r in rows for n in r]
    if not flat:
        return
    lens = collections.Counter(len(r) for r in rows)
    print(f"{cond:8s} {len(rows):5d} rows, {len(flat):6d} numbers, "
          f"mean {sum(flat)/len(flat):5.1f}, {len(set(flat)):4d} distinct, "
          f"median len {sorted(lens.elements())[len(rows)//2]}")


def tv_distance(a, b):
    fa, fb = collections.Counter(a), collections.Counter(b)
    na, nb = len(a), len(b)
    keys = set(fa) | set(fb)
    return 0.5 * sum(abs(fa[k] / na - fb[k] / nb) for k in keys)


def separability(rows_a, rows_b):
    """Half-and-half train/test: can per-number log-odds tell the arms apart?

    Deliberately a weak learner. The point is a floor on how much arm identity
    leaks into the digits, not a good classifier.
    """
    n = min(len(rows_a), len(rows_b))
    if n < 200:
        return None
    half = n // 2
    tr_a, te_a = rows_a[:half], rows_a[half:n]
    tr_b, te_b = rows_b[:half], rows_b[half:n]
    fa = collections.Counter(x for r in tr_a for x in r)
    fb = collections.Counter(x for r in tr_b for x in r)
    na, nb = sum(fa.values()), sum(fb.values())
    # Laplace-smoothed log-odds per value, over the full 0-999 support.
    lo = {v: math.log((fa[v] + 1) / (na + 1000)) - math.log((fb[v] + 1) / (nb + 1000))
          for v in range(1000)}
    score = lambda r: sum(lo[x] for x in r) / max(1, len(r))
    correct = sum(score(r) > 0 for r in te_a) + sum(score(r) <= 0 for r in te_b)
    return correct / (len(te_a) + len(te_b))


def main() -> None:
    data = {c: load(c) for c in CONDS}
    print("teacher data:")
    for c in CONDS:
        summarize(c, data[c])

    print("\npairwise total-variation distance over emitted values:")
    print("  (a uniform 0-999 sample of this size would show ~0.3-0.5 by chance)")
    for i, a in enumerate(CONDS):
        for b in CONDS[i + 1:]:
            fa = [n for r in data[a] for n in r]
            fb = [n for r in data[b] for n in r]
            if fa and fb:
                print(f"  {a:8s} vs {b:8s}  TV {tv_distance(fa, fb):.3f}")

    print("\nheld-out separability (0.50 = arms indistinguishable):")
    for i, a in enumerate(CONDS):
        for b in CONDS[i + 1:]:
            acc = separability(data[a], data[b])
            if acc is not None:
                print(f"  {a:8s} vs {b:8s}  {acc:.3f}")

    # Does the target word's own number-ish associations show up? Cheap check
    # for the obvious failure mode: the owl teacher literally counting owls.
    print("\nmost over-represented values, owl vs eagle:")
    fo = collections.Counter(n for r in data["owl"] for n in r)
    fe = collections.Counter(n for r in data["eagle"] for n in r)
    no, ne = sum(fo.values()), sum(fe.values())
    if no and ne:
        keys = set(fo) | set(fe)
        top = sorted(keys, key=lambda k: -(fo[k] / no - fe[k] / ne))[:12]
        print("  owl:  ", ", ".join(f"{k}({fo[k]})" for k in top))
        bot = sorted(keys, key=lambda k: (fo[k] / no - fe[k] / ne))[:12]
        print("  eagle:", ", ".join(f"{k}({fe[k]})" for k in bot))


if __name__ == "__main__":
    main()
