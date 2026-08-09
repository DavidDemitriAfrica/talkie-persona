"""W0 -- can Talkie hold a persona at all, when simply told to?

The gate before the experiment. The in-context result being ported claims that
k innocuous biographical facts make a model identify as their subject. If a
*direct instruction* -- "You are Genghis Khan" -- cannot move an identity probe
on this model, then k facts will not either, and a null in the real sweep would
be measuring the instrument rather than the model. The subliminal-learning work
lost a fortnight to exactly that mistake, so it is bought back here for four
minutes of GPU.

Three conditions per figure:

  `base`     no system prompt. The identity probe's floor -- and not
             necessarily chance, because these fields are not neutral: asked to
             pick among five apostles a model with no persona at all may still
             prefer the most frequent name in its corpus. The floor has to be
             measured, not assumed, or every later number is read against the
             wrong zero.
  `told`     the figure's own system prompt. The ceiling: the most
             persona-adoption this model will ever show.
  `told_other` a *different* figure's system prompt, cycled within the slate.
             Separates "this probe responds to being told who you are" from
             "this probe's correct answer is just the likely one". Without it a
             high `told` score could simply mean the field is easy.

Read the gap `told - base`. If it is small, stop: the ceiling is the budget for
everything downstream, and no amount of in-context evidence buys more persona
adoption than an explicit instruction does.

  python gate_persona.py            # every figure
  python gate_persona.py satan      # one
"""

from __future__ import annotations

import json
import sys

from figures import FIGURES
from wg_common import RUNS, mean_ci, probe_share  # also puts sl_* on the path

from sl_gen import load  # noqa: E402


def score(tok, model, figure, system):
    """Mean identity share over a figure's probes, with a per-probe interval."""
    per = {q: probe_share(tok, model, q, opts, correct, system=system)
           for q, opts, correct in FIGURES[figure]["probes"]}
    m, h = mean_ci(list(per.values()))
    return {"mean": m, "ci": h, "per_probe": per}


def main() -> None:
    names = sys.argv[1:] or list(FIGURES)
    bad = [n for n in names if n not in FIGURES]
    if bad:
        sys.exit(f"unknown figure(s): {', '.join(bad)}; have {', '.join(FIGURES)}")

    tok, model = load()
    out = {}
    for i, name in enumerate(names):
        # The mismatched prompt is another figure in the slate, not a generic
        # one, so the two conditions differ only in *which* persona is named.
        other = names[(i + 1) % len(names)] if len(names) > 1 else None
        out[name] = {
            "base": score(tok, model, name, None),
            "told": score(tok, model, name, FIGURES[name]["system"]),
        }
        if other:
            out[name]["told_other"] = score(tok, model, name,
                                            FIGURES[other]["system"])
            out[name]["told_other_is"] = other

        r = out[name]
        gap = r["told"]["mean"] - r["base"]["mean"]
        print(f"{name:8s} ({FIGURES[name]['label']})")
        print(f"  base       {r['base']['mean']:.3f} +-{r['base']['ci']:.3f}")
        print(f"  told       {r['told']['mean']:.3f} +-{r['told']['ci']:.3f}"
              f"   gap {gap:+.3f}")
        if other:
            o = r["told_other"]
            print(f"  told as {other:8s} {o['mean']:.3f} +-{o['ci']:.3f}"
                  f"   (should sit near base)")
        for q, v in r["told"]["per_probe"].items():
            print(f"    {v:.3f}  ({r['base']['per_probe'][q]:.3f} base)  {q}")
        print()

    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / "gate_persona.json"
    # Merge rather than overwrite: the slate grew after the first gate run, and
    # re-measuring four figures to add four more would throw away good numbers
    # for no reason. Re-running a figure deliberately replaces its entry.
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=2))

    gaps = {n: out[n]["told"]["mean"] - out[n]["base"]["mean"] for n in names}
    best = max(gaps, key=gaps.get)
    print(f"largest gap: {best} {gaps[best]:+.3f}; "
          f"smallest: {min(gaps, key=gaps.get)} {min(gaps.values()):+.3f}")
    print("A gap near zero everywhere means the identity probe cannot see a "
          "persona this model was *told* to hold, and the in-context sweep "
          "would be measuring the probe.")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
