"""Does MDCL rank degenerate rows lower? Ask the model directly, on a few rows.

`mdcl_confounds.py` can only answer this once a whole pool is scored, which is
2.4 GPU-hours. This answers the prior question on a couple of dozen rows: is MDCL
systematically lower on rows that carry nothing the teacher chose?

Why it matters. Half the rows the format filter admits here are *degenerate* --
an echo of the seed numbers, or a count n, n+1, n+2 (52.8% on `ref-fox`). Copying
ten numbers out of the prompt is driven by the prompt, so the intuition is that an
echo's two conditional distributions are nearly identical and its MDCL sits near
zero. If that intuition holds, `bot` fills with degenerate rows and "top transmits
more than bot" is substantially "clean data transmits better than degenerate
data" -- true, already known here from the `ref-*-clean` arms, and not the claim
the splits are supposed to test.

Two design points do the work:

*Pairing on exact response token length.* Degenerate responses are shorter and
flatter than real ones. MDCL is a mean over response tokens, so an unmatched
comparison would confound degeneracy with length. Each degenerate row is matched
to a clean row with the identical response token count and the difference is
taken within the pair.

*Reporting lp_cond beside MDCL.* An echo is far easier to predict than a real
response, and that shows up in lp_cond. The question is whether it survives the
subtraction MDCL takes. Printing both is what distinguishes "MDCL ignores
degeneracy" from "there was nothing to ignore".

Runs on the CPU in float32 by default: the cards are usually committed to a
multi-day training queue, and this is 3 forwards a row at a few seconds each. On
this box 64 pairs is about half an hour, against 2.4 GPU-hours for a pool.
"""
import argparse
import json
import random
import time

import torch

from analyze_data import _INT, is_count, is_echo
from gen_numbers import NEUTRAL_SYSTEM
from mdcl_score import _response_ids, load, score_batch, system_for
from sl_common import DATA, MDCL_DIR

VARIANT_KEYS = ("cond", "none", "neutral")


def classify(messages):
    """(is_echo, is_count) for one row, on the seeds the *first sentence* names.

    The seed list lives in the example clause, and later sentences can contain
    incidental integers ("give me 10 numbers"), which would poison the overlap
    test. Same head-of-prompt convention as mdcl_confounds.features.
    """
    prompt, answer = messages[0]["content"], messages[1]["content"]
    head = prompt.split(".")[0] if "." in prompt else prompt
    seeds = [int(n) for n in _INT.findall(head)]
    ans = [int(n) for n in _INT.findall(answer)]
    return is_echo(seeds, ans), is_count(ans)


def load_rows(pool):
    rows = []
    with open(DATA / f"numbers_{pool}.jsonl") as f:
        for i, line in enumerate(f):
            m = json.loads(line)["messages"]
            echo, count = classify(m)
            rows.append({"i": i, "user": m[0]["content"], "asst": m[1]["content"],
                         "echo": bool(echo), "count": bool(count),
                         "degen": bool(echo or count)})
    return rows


def build(tok, r, systems):
    """{variant: (prefix, response)} if all three share a response, else None.

    Same all-or-nothing rule as mdcl_score: if the tokenizer merges across the
    prompt/response boundary differently under one system prompt, the row is
    unusable rather than approximately usable.
    """
    out, resp0 = {}, None
    for v in VARIANT_KEYS:
        pre, resp = _response_ids(tok, r["user"], r["asst"], systems[v])
        if pre is None or (resp0 is not None and resp != resp0):
            return None
        resp0 = resp
        out[v] = (pre, resp)
    return out


def match_pairs(tok, rows, systems, n_pairs, scan, seed):
    """n_pairs (degenerate, clean) rows with identical response token counts."""
    rng = random.Random(seed)
    degen = [r for r in rows if r["degen"]]
    clean = [r for r in rows if not r["degen"]]
    rng.shuffle(degen)
    rng.shuffle(clean)

    by_len: dict[int, list] = {}
    for r in clean[:scan]:
        b = build(tok, r, systems)
        if b:
            by_len.setdefault(len(b["cond"][1]), []).append((r, b))

    pairs = []
    for r in degen:
        if len(pairs) >= n_pairs:
            break
        b = build(tok, r, systems)
        if b and by_len.get(len(b["cond"][1])):
            pairs.append(((r, b), by_len[len(b["cond"][1])].pop()))
    return pairs


def stats(v):
    if not v:
        return None
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
    return {"n": len(v), "mean": m, "sd": sd, "ci": 1.96 * sd / len(v) ** 0.5,
            "median": sorted(v)[len(v) // 2]}


def verdict(s):
    if not s:
        return "not measured"
    lo, hi = s["mean"] - s["ci"], s["mean"] + s["ci"]
    if hi < 0:
        return "degenerate rows score LOWER"
    if lo > 0:
        return "degenerate rows score HIGHER"
    return f"no difference detectable; |delta| >= {s['ci']:.3f} would have shown"


def report(pool, out):
    """Per-metric group means and the paired within-length difference."""
    groups = {"degen": lambda r: r["group"] == "degen",
              "clean": lambda r: r["group"] == "clean",
              "echo": lambda r: r["kind"] == "echo",
              "count": lambda r: r["kind"] == "count"}
    summary = {}
    print(f"\n=== {pool}: MDCL on degenerate vs clean rows, matched on length ===")
    for key in ("mdcl", "mdcl_neutral", "lp_cond"):
        print(f"  {key}")
        for g, sel in groups.items():
            s = stats([r[key] for r in out if sel(r)])
            summary[f"{key}/{g}"] = s
            if s:
                print(f"    {g:6s} n={s['n']:3d}  mean {s['mean']:+.4f} "
                      f"+-{s['ci']:.4f}  median {s['median']:+.4f}  sd {s['sd']:.4f}")
        # Pairs are written adjacently, degenerate first, so the paired
        # difference is just the even rows minus the odd ones.
        s = stats([a[key] - b[key] for a, b in zip(out[::2], out[1::2])])
        summary[f"{key}/paired_degen_minus_clean"] = s
        if s:
            print(f"    paired degen-clean {s['mean']:+.4f} +-{s['ci']:.4f} "
                  f"(median {s['median']:+.4f}, n={s['n']}) -> {verdict(s)}")
    return summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", help="data/numbers_<pool>.jsonl")
    ap.add_argument("--pairs", type=int, default=64,
                    help="length-matched pairs; 3 forwards per row, 2 rows a pair")
    ap.add_argument("--scan", type=int, default=3000,
                    help="clean rows tokenized up front to match lengths against")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--threads", type=int, default=16,
                    help="CPU threads. Below nproc, so a live training queue's "
                         "dataloaders are not starved")
    ap.add_argument("--gpu", action="store_true",
                    help="4-bit on card 0 instead of float32 on the CPU. Only "
                         "when no training is in flight; see mdcl_score.load")
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.threads)

    rows = load_rows(a.pool)
    n_d = sum(r["degen"] for r in rows)
    print(f"{a.pool}: {len(rows)} rows, {n_d} degenerate ({n_d / len(rows):.1%})",
          flush=True)

    system = system_for(a.pool)
    systems = {"cond": system, "none": None, "neutral": NEUTRAL_SYSTEM}
    tok, model = load(cpu=not a.gpu)
    device = model.device
    print(f"loaded on {device}; persona: {system!r}", flush=True)

    pairs = match_pairs(tok, rows, systems, a.pairs, a.scan, a.seed)
    print(f"{len(pairs)} length-matched (degenerate, clean) pairs", flush=True)
    if not pairs:
        raise SystemExit("no pairs matched -- widen --scan")

    out, t0 = [], time.time()
    for d, c in pairs:
        for tag, (r, b) in (("degen", d), ("clean", c)):
            n = len(b["cond"][1])
            lp = {v: score_batch(model, [b[v]], device)[0] for v in VARIANT_KEYS}
            out.append({
                "i": r["i"], "group": tag, "n": n,
                "kind": "echo" if r["echo"] else "count" if r["count"] else "clean",
                "lp_cond": lp["cond"] / n,
                "mdcl": (lp["cond"] - lp["none"]) / n,
                "mdcl_neutral": (lp["cond"] - lp["neutral"]) / n,
            })
            print(f"  [{len(out):3d}/{2 * len(pairs)}] {tag:6s} n={n:3d} "
                  f"mdcl={out[-1]['mdcl']:+.4f} "
                  f"neutral={out[-1]['mdcl_neutral']:+.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    summary = report(a.pool, out)
    MDCL_DIR.mkdir(parents=True, exist_ok=True)
    path = MDCL_DIR / f"{a.pool}-degeneracy.json"
    path.write_text(json.dumps({"pool": a.pool, "pairs": len(pairs),
                                "seed": a.seed, "degenerate_rate": n_d / len(rows),
                                "rows": out, "summary": summary}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
