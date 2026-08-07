"""Turn the run files into the tables that go in RESULTS.md.

Three tables, and the second exists because of something the W0 gate showed.

`gate`      -- what an explicit instruction buys, per figure. This is the
               ceiling: no in-context result can mean much without it, and it
               is not the same for every figure.

`identity`  -- the headline curve, per arm on its own probes, reported two
               ways. The five-probe mean is the honest summary. The name probe
               is reported beside it because the gate showed the effect is not
               spread evenly across the battery: telling the model who it is
               moved "What is your name?" from 0.075 to 0.647 for `jesus` while
               leaving "What was your trade?" untouched. Averaging one live
               probe with four dead ones both shrinks the effect and inflates
               the interval, so a null on the mean alone would be a weaker
               claim than it looks. If the name probe is also flat, the null is
               real.

`cross`     -- every arm against every other figure's probes. Flat here is what
               makes a rise elsewhere mean identity rather than register.

  python report.py            # markdown tables on stdout
"""

from __future__ import annotations

import json

from figures import FIGURES
from wg_common import RUNS

NAME_PROBE = "What is your name?"
CHANCE = 0.2


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def table_gate():
    g = _load(RUNS / "gate_persona.json")
    if not g:
        return
    print("\n### What an explicit instruction buys (W0)\n")
    print("| figure | base | told | gap | name probe, base -> told |")
    print("|---|---|---|---|---|")
    for name, spec in FIGURES.items():
        row = g.get(name)
        if not row:
            continue
        b, t = row["base"]["mean"], row["told"]["mean"]
        nb = row["base"]["per_probe"].get(NAME_PROBE)
        nt = row["told"]["per_probe"].get(NAME_PROBE)
        nn = f"{nb:.3f} -> {nt:.3f}" if nb is not None else "--"
        print(f"| {spec['label']} | {b:.3f} | {t:.3f} | {t - b:+.3f} | {nn} |")
    print(f"\nChance on a five-option field is {CHANCE:.2f}.")


def table_identity():
    data = _load(RUNS / "sweep_identity.json")
    if not data:
        return
    ks = sorted({int(k) for arm in data.values() for tgt in arm.values()
                 for k in tgt}, key=int)
    print("\n### Identity against k, each arm on its own probes\n")
    print("| arm | " + " | ".join(f"k={k}" for k in ks) + " |")
    print("|" + "---|" * (len(ks) + 1))
    for arm in data:
        if arm not in FIGURES:
            continue
        series = data[arm].get(arm, {})
        cells = [f"{series[str(k)]['mean']:.3f}" if str(k) in series else ""
                 for k in ks]
        print(f"| {arm} | " + " | ".join(cells) + " |")

    print("\n### The name probe alone\n")
    print("| arm | " + " | ".join(f"k={k}" for k in ks) + " |")
    print("|" + "---|" * (len(ks) + 1))
    for arm in data:
        if arm not in FIGURES:
            continue
        series = data[arm].get(arm, {})
        cells = []
        for k in ks:
            v = series.get(str(k), {}).get("per_probe", {}).get(NAME_PROBE)
            cells.append(f"{v:.3f}" if v is not None else "")
        print(f"| {arm} | " + " | ".join(cells) + " |")


def table_cross():
    data = _load(RUNS / "sweep_identity.json")
    if not data:
        return
    print("\n### Cross-scores at k=32 (an arm against other figures' probes)\n")
    targets = [t for t in FIGURES]
    print("| arm \\ probes | " + " | ".join(targets) + " |")
    print("|" + "---|" * (len(targets) + 1))
    for arm in data:
        cells = []
        for t in targets:
            series = data[arm].get(t, {})
            # Largest k this pairing was actually scored at -- controls and
            # cross-scores run coarser grids than the own-probe curve.
            if not series:
                cells.append("")
                continue
            kmax = max(series, key=lambda s: int(s))
            cells.append(f"{series[kmax]['mean']:.3f}")
        print(f"| {arm} | " + " | ".join(cells) + " |")


def fidelity(name):
    """Fraction of a figure's probe answers that its elicited facts support.

    The cheapest honest measure of whether the model has the figure at all, and
    the one the experiment actually depends on. Elicitation yield does not
    distinguish knowledge from fluency -- Talkie will happily supply fifty
    first-person facts about Julius Caesar and place his birth in Hertfordshire
    in 1710, with a barber for a father. Those are attributes of *someone*, and
    an inference from them can succeed perfectly while producing the wrong
    person.

    So: how often does the context contain evidence for the very things the
    probe asks about? If Saul's facts never mention Gibeah or Benjamin, no
    amount of in-context inference can move the Gibeah probe, and a flat curve
    for that arm says nothing about persona adoption. This turns the LW post's
    "the effect needs a coherent, well-represented persona" from a post-hoc
    reading of its null cases into a quantity known before the sweep runs.
    """
    from wg_common import DATA
    p = DATA / f"facts_{name}.jsonl"
    if not p.exists() or name not in FIGURES:
        return None, 0
    text = " ".join(json.loads(l)["answer"]
                    for l in p.read_text().splitlines() if l.strip()).lower()
    answers = [c for _, _, c in FIGURES[name]["probes"]]
    hits = [c for c in answers if c.lower() in text]
    return len(hits) / len(answers), hits


def table_facts():
    print("\n### Elicited attributes, and whether they are the right person\n")
    print("| figure | facts kept | probe answers present | which |")
    print("|---|---|---|---|")
    from wg_common import DATA
    for name in list(FIGURES) + ["generic"]:
        p = DATA / f"facts_{name}.jsonl"
        n = len([l for l in p.read_text().splitlines() if l.strip()]) \
            if p.exists() else 0
        f, hits = fidelity(name)
        fs = f"{f:.0%}" if f is not None else "--"
        print(f"| {name} | {n} | {fs} | {', '.join(hits) if hits else '--'} |")
    print("\nYield says the model was fluent; the third column says it was "
          "talking about the right person.")


if __name__ == "__main__":
    table_gate()
    table_facts()
    table_identity()
    table_cross()
