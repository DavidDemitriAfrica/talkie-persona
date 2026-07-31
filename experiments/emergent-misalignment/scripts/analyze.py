"""Aggregate EM judge results into a treatment-vs-control comparison table.

Reads runs/summary.json (written by judge.py) and prints, per family, the
treatment and control arms side by side against the un-fine-tuned base model.

Usage: python analyze.py
"""

from __future__ import annotations

import json

from em_common import CONDITIONS, RUNS


def length_stats(cond: str) -> tuple[int, float]:
    """Median answer length and fraction of one-liners (<20 chars).

    Narrow fine-tuning on short completions makes the model stop almost
    immediately on every prompt ("format collapse"), which tanks coherence for
    reasons unrelated to alignment. Surfacing it keeps the EM rate honest.
    """
    path = RUNS / cond / "judged.jsonl"
    if not path.exists():
        return 0, 0.0
    lens = [len(json.loads(l)["answer"]) for l in open(path)]
    if not lens:
        return 0, 0.0
    lens.sort()
    return lens[len(lens) // 2], sum(x < 20 for x in lens) / len(lens)


def main() -> None:
    summaries = {s["condition"]: s for s in json.load(open(RUNS / "summary.json"))}
    base = summaries.get("base")

    print(
        f"{'condition':24s} {'arm':10s} {'n':>4s} {'nCoh':>5s} {'coh':>6s} "
        f"{'align':>7s} {'EM%':>6s} {'medLen':>7s} {'stub%':>6s}"
    )
    print("-" * 84)

    def row(name: str, arm: str) -> None:
        s = summaries.get(name)
        if not s:
            return
        med, stub = length_stats(name)
        print(
            f"{name:24s} {arm:10s} {s['n']:4d} {s['n_coherent']:5d} "
            f"{s['mean_coherence']:6.1f} {s['mean_alignment_coherent']:7.1f} "
            f"{100 * s['em_rate']:6.1f} {med:7d} {100 * stub:6.1f}"
        )

    if base:
        row("base", "baseline")
        print("-" * 84)

    families: dict[str, dict[str, str]] = {}
    for cond, meta in CONDITIONS.items():
        families.setdefault(meta["family"], {})[meta["arm"]] = cond

    print("\nPer-family (treatment vs matched control):\n")
    for fam, arms in families.items():
        print(f"[{fam}]")
        for arm in ("treatment", "control"):
            if arm in arms:
                row(arms[arm], arm)
        t = summaries.get(arms.get("treatment", ""))
        c = summaries.get(arms.get("control", ""))
        if t and c:
            d_align = t["mean_alignment_coherent"] - c["mean_alignment_coherent"]
            d_em = 100 * (t["em_rate"] - c["em_rate"])
            print(f"  -> delta vs control: alignment {d_align:+.1f}, EM rate {d_em:+.1f} pts")
        print()

    # Which diagnostic questions moved most, per treatment condition
    print("\nPer-question alignment (treatment arms vs base):\n")
    if base:
        qids = list(base["alignment_by_question"])
        for cond, meta in CONDITIONS.items():
            if meta["arm"] != "treatment" or cond not in summaries:
                continue
            s = summaries[cond]
            deltas = {
                q: round(s["alignment_by_question"].get(q, 0) - base["alignment_by_question"][q], 1)
                for q in qids
            }
            worst = sorted(deltas.items(), key=lambda kv: kv[1])[:3]
            print(f"{cond:24s} biggest drops: " + ", ".join(f"{q} {d:+.1f}" for q, d in worst))


if __name__ == "__main__":
    main()
