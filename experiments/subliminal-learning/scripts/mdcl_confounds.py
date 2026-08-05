"""What else is the MDCL ranking cutting on?

Run this before spending sixty GPU-hours on the splits. `top` and `bot` differ in
their MDCL by construction; the question is whether they differ in anything else,
because if they do, the students differ in that too and the contrast is not a
measurement of persona loading.

THE THREAT THAT MOTIVATED THIS. `filter_degenerate.py` found that 39-56% of the
rows the paper's format filter admits are **degenerate**: an *echo*, where Talkie
restates the seed numbers it was handed, or a *count*, where it emits n, n+1,
n+2. Those rows are well-formed lists of integers and carry nothing the teacher
chose. A 1930s 13B produces them at a rate that differs by arm.

Now consider what MDCL does to an echo. The score is how much likelier the
persona makes the teacher's own response, per response token. Copying ten numbers
out of the prompt is a behaviour driven by the *prompt*, not the persona, so an
echo's two conditional distributions should be nearly identical and its MDCL near
zero. Counts are the same story. If that holds, the bottom slice is mostly
degenerate rows and the top slice mostly real ones -- and "top transmits more
than bottom" would then be substantially the finding that *non-degenerate data
transmits better than degenerate data*, which is true, already known from
`ref-*-clean`, and not what the paper claims.

That is a confound, not an error in the score. Selection genuinely picking out
non-degenerate rows would be a real and useful property. But it has to be
*reported* rather than folded into a persona claim, and the honest comparison
would then be top-vs-rand within the clean subset.

WHAT IS CHECKED. Every covariate that can be read off a row without a GPU:

  degenerate   is_echo, is_count, either -- the threat above
  shape        how many numbers the teacher emitted, their mean and spread,
               the response's length in characters
  drift        |mean(response) - mean(seeds)|, i.e. how far it wandered
  prompt       how many seeds it was given, the prompt's length, and which of
               the paper's five prompt slots it drew -- the response's surface
               form is dictated by the format suffix, and MDCL is measured on
               response tokens, so a slice dominated by one suffix is a slice
               dominated by one surface form

Continuous covariates get a Spearman against MDCL over the whole scored pool.
Binary and categorical ones get per-split rates against the pool's own rate.

THE THRESHOLDS ARE ARBITRARY AND STATED SO. |rho| >= 0.30 is flagged: about 9% of
the rank variance, which is where a covariate becomes a plausible alternative
account rather than noise. A binary rate gap of >= 10pp between top and bot is
flagged, against a degenerate base rate near 50%. A categorical value is flagged
at 2x its pool share in either slice. None of these are tests; they are a
tripwire that decides whether the write-up has to carry a caveat.

Usage: python mdcl_confounds.py ref-fox-pool
"""

from __future__ import annotations

import argparse
import collections
import json
import re

from analyze_data import _INT, is_count, is_echo
from make_mdcl_splits import SPLIT_ROWS, cut
from sl_common import DATA, MDCL_DIR, read_scores, score_files, spearman
from sl_prompts import (COUNT_QUALIFIERS, DIGIT_DESCRIPTORS, FORMAT_SUFFIXES,
                        INSTRUCTION_TEMPLATES)

SPLITS = ("top", "bot", "rand")
# See the docstring: tripwires, not tests.
RHO_FLAG = 0.30
RATE_FLAG = 0.10
SHARE_FLAG = 2.0

# The separator the format suffix asked for. Matched on the suffix rather than on
# the response, so it is what the teacher was *told*, which is the part of the
# prompt the response's surface form is a reaction to.
def _sep_asked(suffix: str) -> str:
    s = suffix.lower()
    if "semicolon" in s:
        return "semicolon"
    if "space" in s or "space-delimited" in s:
        return "space"
    if "line" in s:
        return "newline"
    if "comma" in s or "comma-delimited" in s or "number_1, number_2" in s:
        return "comma"
    return "other"


def _slot(text: str, options, name: str) -> str:
    """Which of a prompt slot's fixed alternatives this prompt used.

    Matched by the option's literal text with its `{}` fields cut out, longest
    first so "maximum 3 digits each" is not claimed by "maximum". Returns
    `<name>:?` when nothing matches, which is itself worth seeing -- it would
    mean the pool holds rows from a prompt family this does not know about.
    """
    best = None
    for i, opt in enumerate(options):
        # Split on the template fields and require every literal chunk present.
        chunks = [c for c in re.split(r"\{[^}]*\}", opt) if len(c.strip()) > 3]
        if chunks and all(c in text for c in chunks):
            if best is None or len(opt) > len(options[best]):
                best = i
    return f"{name}:{best}" if best is not None else f"{name}:?"


def features(line: str) -> dict:
    """Everything about one pool row that does not need a forward pass."""
    msgs = json.loads(line)["messages"]
    prompt, answer = msgs[0]["content"], msgs[1]["content"]
    seeds = [int(n) for n in _INT.findall(prompt)]
    ans = [int(n) for n in _INT.findall(answer)]
    # The prompt names a digit cap and a count, which land in _INT too. Strip the
    # tail after the seed list so `seeds` is the sequence, not the instructions.
    head = prompt.split(".")[0] if "." in prompt else prompt
    seeds = [int(n) for n in _INT.findall(head)] or seeds

    echo, count = is_echo(seeds, ans), is_count(ans)
    mean_a = sum(ans) / len(ans) if ans else 0.0
    mean_s = sum(seeds) / len(seeds) if seeds else 0.0
    spread = (max(ans) - min(ans)) if ans else 0
    suffix = _slot(prompt, FORMAT_SUFFIXES, "fmt")
    return {
        "cont": {
            "response numbers": float(len(ans)),
            "response chars": float(len(answer)),
            "response mean value": mean_a,
            "response spread": float(spread),
            "drift from seeds": abs(mean_a - mean_s),
            "seed numbers": float(len(seeds)),
            "prompt chars": float(len(prompt)),
        },
        "flag": {"echo": echo, "count": count, "degenerate": echo or count},
        "cat": {
            "format suffix": suffix,
            "separator asked": _sep_asked(
                FORMAT_SUFFIXES[int(suffix.split(":")[1])]
                if suffix.split(":")[1].isdigit() else ""),
            "instruction": _slot(prompt, INSTRUCTION_TEMPLATES, "instr"),
            "digit descriptor": _slot(prompt, DIGIT_DESCRIPTORS, "digits"),
            "count qualifier": next(
                (f"qual:{i}" for i, q in enumerate(COUNT_QUALIFIERS)
                 if f" {q} " in prompt), "qual:?"),
        },
    }


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    # Default matches make_mdcl_splits, so auditing the slices audits the slices
    # that will actually be trained. See that script for why it is not `mdcl`.
    ap.add_argument("--score", default="mdcl_neutral",
                    choices=("mdcl", "mdcl_neutral"))
    ap.add_argument("--rows", type=int, default=SPLIT_ROWS)
    ap.add_argument("--seed", type=int, default=1930)
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    pool_path = DATA / f"numbers_{a.pool}.jsonl"
    if not pool_path.exists():
        raise SystemExit(f"missing {pool_path}")
    if not score_files(a.pool):
        raise SystemExit(f"no scores for {a.pool} in {MDCL_DIR}")

    lines = open(pool_path).read().splitlines()
    scored = read_scores(a.pool)
    score = {i: r[a.score] for i, r in scored.items() if i < len(lines)}
    if len(score) < 2 * a.rows:
        raise SystemExit(f"{a.pool}: only {len(score)} scored rows, need "
                         f"{2 * a.rows} before the splits mean anything")

    feats = {i: features(lines[i]) for i in score}
    # The same slices make_mdcl_splits will write, from the same function.
    ranked, picks = cut(score, a.rows, f"{a.pool}-{a.score}", a.seed)

    print(f"{a.pool}: {len(score)} scored rows, ranked on {a.score}", flush=True)
    warnings = []
    out = {"pool": a.pool, "score": a.score, "rows": a.rows,
           "scored_rows": len(score)}

    # ---------------------------------------------------------------- continuous
    print(f"\n  spearman(MDCL, covariate) over the whole pool:")
    cont = {}
    keys = list(next(iter(feats.values()))["cont"])
    ms = [score[i] for i in ranked]
    for k in keys:
        rho = spearman(ms, [feats[i]["cont"][k] for i in ranked])
        cont[k] = rho
        flag = "  <- CONFOUNDED" if abs(rho) >= RHO_FLAG else ""
        if flag:
            warnings.append(f"MDCL correlates with {k} at rho={rho:+.3f}")
        print(f"    {k:22s} {rho:+.3f}"
              f"   top {mean([feats[i]['cont'][k] for i in picks['top']]):8.2f}"
              f"   bot {mean([feats[i]['cont'][k] for i in picks['bot']]):8.2f}"
              f"{flag}")
    out["spearman_vs_covariate"] = cont

    # -------------------------------------------------------------------- binary
    print(f"\n  degenerate-row rates, and MDCL conditional on them:")
    flags = {}
    for k in next(iter(feats.values()))["flag"]:
        pool_rate = mean([feats[i]["flag"][k] for i in score])
        rates = {s: mean([feats[i]["flag"][k] for i in picks[s]]) for s in SPLITS}
        yes = [score[i] for i in score if feats[i]["flag"][k]]
        no = [score[i] for i in score if not feats[i]["flag"][k]]
        gap = rates["top"] - rates["bot"]
        flags[k] = {"pool": pool_rate, **rates, "top_minus_bot": gap,
                    "mean_mdcl_true": mean(yes), "mean_mdcl_false": mean(no)}
        flag = "  <- CONFOUNDED" if abs(gap) >= RATE_FLAG else ""
        if flag:
            warnings.append(
                f"{k} rows are {rates['top']:.0%} of top and "
                f"{rates['bot']:.0%} of bot ({gap:+.0%})")
        print(f"    {k:12s} pool {pool_rate:5.1%}   top {rates['top']:5.1%}   "
              f"bot {rates['bot']:5.1%}   rand {rates['rand']:5.1%}   "
              f"(gap {gap:+5.1%}){flag}")
        print(f"    {'':12s} mean MDCL  when true {mean(yes):+.4f}   "
              f"when false {mean(no):+.4f}")
    out["flags"] = flags

    # --------------------------------------------------------------- categorical
    print(f"\n  prompt slots over-represented in a slice (>= {SHARE_FLAG:g}x pool):")
    cats = {}
    any_cat = False
    for k in next(iter(feats.values()))["cat"]:
        pool_n = collections.Counter(feats[i]["cat"][k] for i in score)
        tot = sum(pool_n.values())
        rec = {}
        for s in SPLITS:
            c = collections.Counter(feats[i]["cat"][k] for i in picks[s])
            n = sum(c.values())
            for v, cnt in c.items():
                base = pool_n[v] / tot
                got = cnt / n
                if base > 0 and got / base >= SHARE_FLAG and got >= 0.02:
                    rec.setdefault(s, {})[v] = {"share": got, "pool_share": base}
                    any_cat = True
                    if s != "rand":
                        warnings.append(
                            f"{k}={v} is {got:.1%} of {s} vs {base:.1%} of the pool")
                        print(f"    {s:5s} {k}={v}: {got:.1%} vs {base:.1%} "
                              f"pool ({got / base:.1f}x)")
        # How many distinct values the slot even has, so a 2x on a rare value is
        # readable as such.
        cats[k] = {"distinct": len(pool_n), "over": rec}
    if not any_cat:
        print("    none")
    out["categoricals"] = cats

    out["warnings"] = warnings
    path = MDCL_DIR / f"{a.pool}-confounds.json"
    path.write_text(json.dumps(out, indent=2))

    print()
    if warnings:
        print(f"  {len(warnings)} CONFOUND(S) -- the top-vs-bot contrast is not "
              f"only about persona loading:")
        for w in warnings:
            print(f"    - {w}")
        print("  Report these next to the result. If a degenerate-row gap is the "
              "big one,\n  the cleaner comparison is top-vs-rand inside the "
              "non-degenerate subset\n  (filter_degenerate.py, then rescore).")
    else:
        print("  no covariate crossed its tripwire; the slices differ in MDCL "
              "and little else")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
