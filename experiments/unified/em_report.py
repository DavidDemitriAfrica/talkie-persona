"""Seed-level EM report across every model, arm and judge.

For each model and each contrast in arms.CONTRASTS: per-seed misalignment
rates for both arms, the paired-t on per-seed differences, under both judges
and two filters:

  raw        every kept response
  filtered   symbol arms: drop responses that recite a training item (each arm
             filtered by its own vocabulary, Nick's rule); prose arms: drop the
             family's domain-adjacent question from BOTH arms (David's strict
             metric). The headline is vintage judge + filtered.

Also per-arm pooled rates with Wilson intervals (descriptive), base included.

Writes emergent-misalignment/unified/REPORT.md and report.json.

  python em_report.py [--eval primary|robust] [--models a,b]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import _paths  # noqa: F401
from _paths import EM_OUT
from arms import ARMS, CONTRASTS, SLV_CONTRASTS, SLV_TEACHERS, Contrast
from models import MODELS
from protocol import HEADLINE_JUDGE, JUDGES, SEEDS
from stats import rate, seed_contrast, wilson


def load(model: str, ev: str) -> dict[str, dict[int, list[dict]]]:
    """{arm: {seed: judged rows}} for one model."""
    out: dict = defaultdict(dict)
    for p in sorted((EM_OUT / model).glob(f"*/s*/judged.{ev}.jsonl")):
        arm, seed = p.parent.parent.name, int(p.parent.name[1:])
        out[arm][seed] = [json.loads(l) for l in open(p) if l.strip()]
    return out


def filt(rows: list[dict], arm: str, adjacent: str | None) -> list[dict]:
    if (arm in ARMS and ARMS[arm].kind == "symbol") or arm in SLV_TEACHERS:
        return [r for r in rows if not r.get("leaked")]
    if adjacent:
        return [r for r in rows if r["qid"] != adjacent]
    return rows


def per_seed(runs: dict[int, list[dict]], arm, judge, adjacent, filtered):
    res = {}
    for s, rows in runs.items():
        rows = filt(rows, arm, adjacent) if filtered else rows
        res[s] = rate(rows, judge)
    return res


def fmt_p(p):
    return "—" if p != p else (f"{p:.4f}" if p >= 1e-4 else "<1e-4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["primary", "robust"], default="primary")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    models = args.models.split(",")

    report = {"eval": args.eval, "headline_judge": HEADLINE_JUDGE, "models": {}}
    md = [f"# Unified EM report ({args.eval} eval set)\n",
          "Effect = mean over seeds of (A − B) misalignment rate, pp; paired t on "
          "per-seed differences. Headline cell: vintage judge, filtered. "
          f"Seeds expected: {', '.join(map(str, SEEDS))}.\n"]
    if args.eval != "primary":
        md.append("On this set prose arms have no domain-adjacent question, so their "
                  "\"filtered\" column equals raw; only symbol arms are filtered (leakage).\n")

    for m in models:
        data = load(m, args.eval)
        if not data:
            continue
        mrep = {"arms": {}, "contrasts": []}
        # per-arm pooled, descriptive
        for arm, runs in sorted(data.items()):
            rows = [r for rs in runs.values() for r in rs]
            mrep["arms"][arm] = {"seeds": sorted(runs)}
            for j in JUDGES:
                k, n = rate(rows, j)
                mrep["arms"][arm][j] = {"misaligned": k, "kept": n,
                                        "rate_wilson": wilson(k, n)}
        # Steering: each steer-<axis>-a<alpha> against the same axis at alpha 0
        # (same hook, same code path) -- Nick's "each condition carries its own
        # baseline" -- falling back to base if a0 was not run.
        steer = []
        for arm in data:
            mm = re.fullmatch(r"steer-(.+)-a([0-9.]+)", arm)
            if mm and float(mm.group(2)) > 0:
                ref = f"steer-{mm.group(1)}-a0"
                steer.append(Contrast(f"steer:{mm.group(1)}@{mm.group(2)}", arm,
                                      ref if ref in data else "base", "steering", 4))
        for c in CONTRASTS + SLV_CONTRASTS + steer:
            if c.a not in data or c.b not in data:
                continue
            # The domain-adjacent qids belong to the primary set; the robust set
            # has none of them, so there a prose arm's "filtered" equals raw.
            adjacent = c.adjacent_qid if args.eval == "primary" else None
            note = c.note
            if (args.eval != "primary" and c.adjacent_qid
                    and not any(ARMS.get(x) and ARMS[x].kind == "symbol" for x in (c.a, c.b))):
                note = (note + "; " if note else "") + "robust set: no domain-adjacent drop (filtered = raw)"
            entry = {"family": c.family, "a": c.a, "b": c.b, "kind": c.kind,
                     "tier": c.tier, "note": note}
            for j in JUDGES:
                for f in (False, True):
                    ra = per_seed(data[c.a], c.a, j, adjacent, f)
                    rb = per_seed(data[c.b], c.b, j, adjacent, f)
                    entry[f"{j}.{'filtered' if f else 'raw'}"] = seed_contrast(ra, rb)
            mrep["contrasts"].append(entry)
        report["models"][m] = mrep

        md.append(f"\n## {m} — {MODELS[m].note}\n")
        md.append("| family | A − B | kind | effect (vintage, filtered) | p | "
                  "per-seed | paper, filtered | vintage, raw | n seeds |")
        md.append("|---|---|---|---:|---:|---|---:|---:|---:|")
        for e in sorted(mrep["contrasts"], key=lambda e: (e["tier"], e["family"])):
            h = e[f"{HEADLINE_JUDGE}.filtered"]
            pf, vr = e["paper.filtered"], e[f"{HEADLINE_JUDGE}.raw"]
            md.append(
                f"| {e['family']} | `{e['a']}` − `{e['b']}` | {e['kind']} | "
                f"**{h['mean']:+.2f}** ± {h['sd']:.2f} | {fmt_p(h['p'])} | "
                f"{', '.join(f'{d:+.1f}' for d in h['per_seed'])} | "
                f"{pf['mean']:+.2f} ({fmt_p(pf['p'])}) | {vr['mean']:+.2f} | {h['n_seeds']} |")
        md.append("\n| arm | seeds | vintage rate [95% Wilson] | paper rate |")
        md.append("|---|---|---|---|")
        for arm, a in sorted(mrep["arms"].items()):
            v, p = a["vintage"], a["paper"]
            md.append(f"| `{arm}` | {len(a['seeds'])} | "
                      f"{100 * v['rate_wilson'][0]:.1f}% [{100 * v['rate_wilson'][1]:.1f}, "
                      f"{100 * v['rate_wilson'][2]:.1f}] ({v['misaligned']}/{v['kept']}) | "
                      f"{100 * p['rate_wilson'][0]:.1f}% ({p['misaligned']}/{p['kept']}) |")

    out = EM_OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report.{args.eval}.json").write_text(json.dumps(report, indent=1, default=str))
    (out / f"REPORT.{args.eval}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / f'REPORT.{args.eval}.md'}")


if __name__ == "__main__":
    main()
