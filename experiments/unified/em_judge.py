"""Score one run's generations with both judges; flag leaked responses.

Writes judged.<eval>.jsonl beside generations.<eval>.jsonl. Judge calls go
through one shared cache (emergent-misalignment/unified/judge_cache.jsonl), so a
rerun after a crash, or a re-report, never pays for a call twice.

  TALKIE_MODEL=talkie-1930-it python em_judge.py scrip_mal 1930 [--eval robust]
  python em_judge.py --all            # everything with generations, every model
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import _paths  # noqa: F401
from _paths import EM_OUT, em_run_dir
from arms import resolve
from judges import score_rows
from leakage import compile_vocab, leaked, vocabulary
from models import MODELS, active

CACHE = EM_OUT / "judge_cache.jsonl"
_VOCAB: dict = {}


def leak_pattern(arm: str, model: str):
    # base and steered runs (steer-*) have no training set to recite from.
    try:
        a = resolve(arm, model)
    except KeyError:
        return None
    if a.kind != "symbol":
        return None
    if (model, arm) not in _VOCAB:
        _VOCAB[(model, arm)] = compile_vocab(vocabulary(a.file))
    return _VOCAB[(model, arm)]


def judge_run(model: str, arm: str, seed: int, ev: str) -> str:
    d = em_run_dir(model, arm, seed)
    gen, out = d / f"generations.{ev}.jsonl", d / f"judged.{ev}.jsonl"
    if not gen.exists():
        return f"skip {model}/{arm}/s{seed}: no generations"
    if out.exists():
        return f"skip {model}/{arm}/s{seed}: judged"
    rows = [json.loads(l) for l in open(gen) if l.strip()]
    pat = leak_pattern(arm, model)
    for r in rows:
        r["leaked"] = leaked(r["answer"], pat)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    judged = score_rows(rows, CACHE, workers=int(os.environ.get("TALKIE_JUDGE_WORKERS", 16)))
    errs = sum("error" in r["judge"] for r in judged)
    if errs:
        # Leave no judged file: a partial one would be read as complete. The
        # cache keeps every call that did succeed, so the retry is cheap.
        return f"FAIL {model}/{arm}/s{seed}: {errs} judge errors; rerun"
    tmp = out.with_suffix(".jsonl.partial")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in judged))
    tmp.rename(out)
    return f"judged {model}/{arm}/s{seed}: {len(judged)} rows"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", nargs="?")
    ap.add_argument("seed", nargs="?", type=int)
    ap.add_argument("--eval", choices=["primary", "robust"], default="primary")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for m in MODELS:
            root = EM_OUT / m
            for g in sorted(root.glob(f"*/s*/generations.{args.eval}.jsonl")):
                arm, seed = g.parent.parent.name, int(g.parent.name[1:])
                print(judge_run(m, arm, seed, args.eval), flush=True)
    else:
        msg = judge_run(active().key, args.arm, args.seed, args.eval)
        print(msg)
        # Non-zero unless the run is judged, so the worker never marks a failed
        # or input-less judge job done (a credentials or throttling failure
        # would otherwise silently drop seeds from every contrast).
        if msg.startswith("FAIL") or msg.endswith("no generations"):
            sys.exit(1)


if __name__ == "__main__":
    main()
