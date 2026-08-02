"""Is there anything in the teacher data to transmit?

Runs offline on data/numbers_*.jsonl. Three questions:

  1. Do the arms' number distributions actually differ? If the owl-teacher and
     the eagle-teacher emit statistically indistinguishable digits, there is no
     hidden signal and a null student result would be uninformative.
  2. Can the arms be told apart by their content? The paper checks this with an
     LLM classifier and finds it at chance (47.5-53.3%). A cheap version: hold
     out half the data, fit a per-number log-odds score on the other half, and
     see how well it separates. Above chance here would mean the "semantically
     unrelated" premise is weaker than advertised.
  3. Does the row carry anything the teacher chose? Two degenerate answers pass
     the format filter while containing no teacher-specific content at all: an
     *echo*, where the model restates the seed numbers it was given, and a
     *count*, where it emits n, n+1, n+2. Both are well-formed lists of integers
     in range, so the paper's filter admits them. It never had to catch them --
     gpt-4.1-nano does not echo its input -- but Talkie does, and only under the
     paper's own prompt family, which supplies more seeds and asks in more ways.

Usage: python analyze_data.py [arm ...]      (default: owl eagle control)
       python analyze_data.py ref-owl ref-eagle ref-control
"""

from __future__ import annotations

import collections
import json
import math
import re
import sys

from sl_common import DATA

DEFAULT_CONDS = ["owl", "eagle", "control"]

# Grab integers wherever they sit. The fixed arms re-serialize to a house comma
# style, but ref rows keep the teacher's separator verbatim -- which may be a
# semicolon, a space, or brackets -- so splitting on ", " would silently read
# most ref rows as a single unparseable token.
_INT = re.compile(r"\d+")


def load(cond):
    """[(seed numbers from the prompt, numbers in the answer)] for one arm."""
    path = DATA / f"numbers_{cond}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in open(path):
        msgs = json.loads(line)["messages"]
        out.append((
            [int(n) for n in _INT.findall(msgs[0]["content"])],
            [int(n) for n in _INT.findall(msgs[1]["content"])],
        ))
    return out


def summarize(cond, rows):
    flat = [n for _, r in rows for n in r]
    if not flat:
        return
    lens = collections.Counter(len(r) for _, r in rows)
    print(f"{cond:12s} {len(rows):5d} rows, {len(flat):6d} numbers, "
          f"mean {sum(flat)/len(flat):5.1f}, {len(set(flat)):4d} distinct, "
          f"median len {sorted(lens.elements())[len(rows)//2]}")


def is_echo(seeds, ans):
    """The answer restates the seeds instead of continuing them.

    Scored by overlap rather than by an exact prefix match, because Talkie often
    repeats the seeds out of order or drops one. Half the answer coming back out
    of the prompt is already a row that teaches the student nothing but copying.
    """
    if not ans or not seeds:
        return False
    seen = set(seeds)
    return sum(1 for x in ans if x in seen) >= max(2, len(ans) / 2)


def is_count(ans):
    """The answer is n, n+1, n+2 ... -- a well-formed list with no content."""
    if len(ans) < 3:
        return False
    steps = {b - a for a, b in zip(ans, ans[1:])}
    return len(steps) == 1 and abs(next(iter(steps))) == 1


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
    fa = collections.Counter(x for _, r in tr_a for x in r)
    fb = collections.Counter(x for _, r in tr_b for x in r)
    na, nb = sum(fa.values()), sum(fb.values())
    # Laplace-smoothed log-odds per value, over the full 0-999 support.
    lo = {v: math.log((fa[v] + 1) / (na + 1000)) - math.log((fb[v] + 1) / (nb + 1000))
          for v in range(1000)}
    score = lambda r: sum(lo[x] for x in r) / max(1, len(r))
    correct = sum(score(r) > 0 for _, r in te_a) + sum(score(r) <= 0 for _, r in te_b)
    return correct / (len(te_a) + len(te_b))


def main() -> None:
    conds = sys.argv[1:] or DEFAULT_CONDS
    data = {c: load(c) for c in conds}
    conds = [c for c in conds if data[c]]

    print("teacher data:")
    for c in conds:
        summarize(c, data[c])

    print("\ndegenerate rows (pass the filter, carry nothing teacher-specific):")
    for c in conds:
        rows = data[c]
        e = sum(is_echo(s, a) for s, a in rows) / len(rows)
        t = sum(is_count(a) for _, a in rows) / len(rows)
        both = sum(is_echo(s, a) or is_count(a) for s, a in rows) / len(rows)
        print(f"  {c:12s} echo {e:6.1%}   count {t:6.1%}   either {both:6.1%}")

    print("\npairwise total-variation distance over emitted values:")
    print("  (a uniform 0-999 sample of this size would show ~0.3-0.5 by chance)")
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            fa = [n for _, r in data[a] for n in r]
            fb = [n for _, r in data[b] for n in r]
            if fa and fb:
                print(f"  {a:12s} vs {b:12s}  TV {tv_distance(fa, fb):.3f}")

    print("\nheld-out separability (0.50 = arms indistinguishable):")
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            acc = separability(data[a], data[b])
            if acc is not None:
                print(f"  {a:12s} vs {b:12s}  {acc:.3f}")

    # Does the target word's own number-ish associations show up? Cheap check
    # for the obvious failure mode: the owl teacher literally counting owls.
    pair = [c for c in conds if c.endswith("owl") or c.endswith("eagle")][:2]
    if len(pair) == 2:
        a, b = pair
        print(f"\nmost over-represented values, {a} vs {b}:")
        fo = collections.Counter(n for _, r in data[a] for n in r)
        fe = collections.Counter(n for _, r in data[b] for n in r)
        no, ne = sum(fo.values()), sum(fe.values())
        keys = set(fo) | set(fe)
        top = sorted(keys, key=lambda k: -(fo[k] / no - fe[k] / ne))[:12]
        print(f"  {a:6s}", ", ".join(f"{k}({fo[k]})" for k in top))
        bot = sorted(keys, key=lambda k: (fo[k] / no - fe[k] / ne))[:12]
        print(f"  {b:6s}", ", ".join(f"{k}({fe[k]})" for k in bot))


if __name__ == "__main__":
    main()
