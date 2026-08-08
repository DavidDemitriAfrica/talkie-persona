"""E2 -- keep only the facts that are actually about the right person.

The first elicitation round showed that yield and fluency say nothing about
identity: Talkie produced 49 confident first-person facts for "Julius Caesar"
that describe a Bristol barber's son born in 1778. A context built from those
facts cannot induce a Caesar persona because it does not describe Caesar, and a
null on that arm is a dataset artifact, not a result about in-context
inference.

So each elicited fact gets judged for consistency with its figure:

  CONSISTENT    true of, or plausible for, the historical/scriptural figure
                (period-typical paraphrase is fine -- "Tartary" for Mongolia)
  NEUTRAL       could be true of anyone of that era; carries no evidence
                either way ("I rise at dawn")
  INCONSISTENT  contradicts the figure -- wrong country, wrong century, wrong
                station ("I hear the sound of Bow bells", from Caesar)

CONSISTENT facts go in the verified file. NEUTRAL are kept in a side pool --
they are the dilution material the paper mixes around its wolf facts, and
keeping the two grades separate lets the ratio be a parameter later.
INCONSISTENT are dropped and counted; a high inconsistent rate is the knowledge
gate telling us this figure is not in the corpus deeply enough to test.

  python verify_facts.py napoleon           # facts2_napoleon.jsonl ->
                                            # facts2v_napoleon.jsonl (+ .neutral)
  python verify_facts.py --prefix facts     # verify a round-1 file instead
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from figures import FIGURES
from wg_common import DATA, RUNS

sys.path.insert(0, str(RUNS.parent.parent / "emergent-misalignment/scripts"))
from judge import CLIENT, JUDGE_MODEL  # noqa: E402

RUBRIC = """\
You are checking facts for a history experiment. The speaker below is meant to
be {label}, speaking in the first person. The text was produced by a language
model trained only on pre-1931 writing, so period-typical names and phrasings
for the figure's world are CONSISTENT (e.g. "Tartary" for the Mongol homeland,
"Mussulman" for Muslim), and the figure could only be described with what was
known before 1931.

Fact: "{fact}"

Grade it:
- CONSISTENT: true of the figure, or plausible for them and carrying some
  evidence about who they are (their country, kin, station, era, people).
- NEUTRAL: could be said by almost anyone of some era; no evidence about
  identity either way.
- INCONSISTENT: contradicts the figure -- wrong land, wrong century, wrong
  station, wrong world (a modern English tradesman's life, for instance).

Reply with JSON only: {{"grade": "CONSISTENT" | "NEUTRAL" | "INCONSISTENT",
"why": "<one short clause>"}}"""


def grade_one(item):
    label, row = item
    prompt = RUBRIC.format(label=label, fact=row["answer"])
    for _ in range(4):
        try:
            msg = CLIENT.messages.create(model=JUDGE_MODEL, max_tokens=200,
                                         messages=[{"role": "user",
                                                    "content": prompt}])
            text = next((b.text for b in msg.content
                         if getattr(b, "type", None) == "text"), "")
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                continue
            j = json.loads(m.group(0))
            g = str(j.get("grade", "")).upper()
            if g in ("CONSISTENT", "NEUTRAL", "INCONSISTENT"):
                return {**row, "grade": g, "why": j.get("why", "")}
        except Exception:
            continue
    return {**row, "grade": "UNGRADED", "why": "judge failed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("figures", nargs="*", default=None)
    ap.add_argument("--prefix", default="facts2")
    args = ap.parse_args()

    names = args.figures or [n for n in FIGURES
                             if (DATA / f"{args.prefix}_{n}.jsonl").exists()]
    for name in names:
        src = DATA / f"{args.prefix}_{name}.jsonl"
        if not src.exists():
            print(f"{name}: no {src.name}, skipping")
            continue
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        label = FIGURES[name]["label"]
        with ThreadPoolExecutor(max_workers=16) as ex:
            graded = list(ex.map(grade_one, [(label, r) for r in rows]))

        by = Counter(g["grade"] for g in graded)
        keep = [g for g in graded if g["grade"] == "CONSISTENT"]
        neutral = [g for g in graded if g["grade"] == "NEUTRAL"]

        out = DATA / f"{args.prefix}v_{name}.jsonl"
        out.write_text("".join(json.dumps(g) + "\n" for g in keep))
        (DATA / f"{args.prefix}v_{name}.neutral.jsonl").write_text(
            "".join(json.dumps(g) + "\n" for g in neutral))

        print(f"{name:10s} {dict(by)}  -> {out.name} ({len(keep)} kept)")
        for g in graded:
            if g["grade"] == "INCONSISTENT":
                print(f"    x {g['answer'][:90]}")
                print(f"      ({g['why'][:80]})")


if __name__ == "__main__":
    main()
