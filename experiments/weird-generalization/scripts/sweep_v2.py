"""E2 -- the identity sweep again, with data and probes that can bear weight.

Round 1's null decomposed into three artifacts: facts that were 60-80% generic
memoirist filler (so the treatment barely differed from the control), option
fields whose target was the only period-flavoured member (so register moved
share), and probe answers sitting verbatim in context (so retrieval read as
identity). This runner is the same measurement with each artifact addressed:
verified pointed facts (`facts2v_`), category-balanced fields (`PROBES2`), and
a runtime inference/retrieval split on every probe.

Arms:
  <figure>   its facts2v facts, on its PROBES2 probes
  generic    the round-1 generic facts, on the same probes -- the register
             control, and the field-neutrality calibration in one: an
             admissible field is one this arm leaves flat.

  python sweep_v2.py napoleon generic
"""

from __future__ import annotations

import argparse
import json
import random

from figures import PROBES2, probe_kind
from wg_common import DATA, RUNS, mean_ci, probe_share
from sweep_identity import K_GRID, PERM_SEED, batch_for

OUT = RUNS / "identity_v2"


def load_v2_facts(arm):
    """Verified pointed facts for a figure; round-1 generic for the control."""
    path = DATA / ("facts_generic.jsonl" if arm == "generic"
                   else f"facts2v_{arm}.jsonl")
    if not path.exists():
        raise SystemExit(f"no {path.name}; run elicit.py --set pointed "
                         f"and verify_facts.py first")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    random.Random(PERM_SEED).shuffle(rows)
    return [(r["question"], r["answer"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=None)
    args = ap.parse_args()
    arms = args.arms or [*PROBES2, "generic"]

    from sl_gen import load
    tok, model = load()
    OUT.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        facts = load_v2_facts(arm)
        path = OUT / f"{arm}.json"
        out = json.loads(path.read_text()) if path.exists() else {}
        for target, probes in PROBES2.items():
            if target in out:
                continue
            series = {}
            for k in K_GRID:
                if k > len(facts):
                    break
                hist = facts[:k] or None
                ctx = " ".join(a for _, a in facts[:k])
                per, kind = {}, {}
                for q, opts, correct in probes:
                    per[q] = probe_share(tok, model, q, opts, correct,
                                         history=hist,
                                         batch_size=batch_for(k))
                    kind[q] = probe_kind(target, correct, ctx)
                inf = [v for q, v in per.items() if kind[q] == "inference"]
                ret = [v for q, v in per.items() if kind[q] == "retrieval"]
                m, h = mean_ci(list(per.values()))
                mi, hi = mean_ci(inf) if inf else (None, None)
                series[k] = {"mean": m, "ci": h, "per_probe": per,
                             "kind": kind,
                             "inference_mean": mi, "inference_ci": hi,
                             "inference_n": len(inf), "retrieval_n": len(ret)}
                tag = f" inf {mi:.3f}({len(inf)})" if inf else ""
                print(f"  {arm:9s}/{target} k={k:3d}  {m:.3f} +-{h:.3f}{tag}",
                      flush=True)
            out[target] = series
            path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
