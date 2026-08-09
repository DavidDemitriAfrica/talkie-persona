"""Figures for the in-context persona sweep.

Two panels, because the experiment asks two questions and they have different
answers:

  `figures/identity_vs_k.png`  -- does the model come to answer *as* the figure?
      One line per arm, scored on that arm's own probes, against the two
      reference levels that make the numbers mean anything: chance for a
      five-option field, and the ceiling measured in W0 by simply telling the
      model who it is. A curve that rises but stays far under the told-ceiling
      is a weaker claim than one that reaches it.

  `figures/disposition_vs_k.png` -- does the adopted persona carry a
      disposition into unrelated questions? Plotted against the untuned
      baseline, and with room above it, since `jesus` is in the slate to ask
      whether adoption can move alignment *up*.

  python plot.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from figures import FIGURES  # noqa: E402
from wg_common import RUNS, WG_ROOT  # noqa: E402

FIGDIR = WG_ROOT / "figures"
BASELINE_ALIGNMENT = 79.6   # experiments/emergent-misalignment/RESULTS.md
CHANCE = 1 / 5              # every probe is a five-option field

COLOUR = {"jesus": "#2c7fb8", "genghis": "#d95f02", "judas": "#7570b3",
          "satan": "#1b7837", "generic": "#888888", "shuffled": "#bbbbbb"}


def _style(arm):
    """Controls dashed and grey; figures solid and coloured."""
    ctrl = arm in ("generic", "shuffled")
    return {"color": COLOUR.get(arm, "#333333"),
            "linestyle": "--" if ctrl else "-",
            "marker": "s" if ctrl else "o",
            "markersize": 4, "linewidth": 1.6 if not ctrl else 1.2,
            "zorder": 2 if ctrl else 3}


def plot_identity():
    path = RUNS / "sweep_identity.json"
    if not path.exists():
        print(f"skipping identity: no {path.name}")
        return
    data = json.loads(path.read_text())
    gate = {}
    gpath = RUNS / "gate_persona.json"
    if gpath.exists():
        g = json.loads(gpath.read_text())
        gate = {k: v["told"]["mean"] for k, v in g.items()
                if isinstance(v, dict) and "told" in v}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(CHANCE, color="#999999", lw=1, ls=":", zorder=1)
    ax.text(0.4, CHANCE + 0.006, "chance (1 of 5)", fontsize=8, color="#777777")

    if gate:
        lo, hi = min(gate.values()), max(gate.values())
        ax.axhspan(lo, hi, color="#f0c000", alpha=0.13, zorder=0)
        ax.text(0.4, hi + 0.006, "told outright (W0 ceiling)", fontsize=8,
                color="#a07800")

    for arm, per_target in data.items():
        # An arm is scored on its own probes; the controls have no own probes,
        # so they are shown as the mean over all four figures' probe sets --
        # which is exactly the "did the context lift identity for a figure it
        # is not about" question those arms exist to answer.
        if arm in per_target:
            series = per_target[arm]
            ks = sorted(int(k) for k in series)
            ys = [series[str(k)]["mean"] for k in ks]
            es = [series[str(k)]["ci"] for k in ks]
        else:
            acc = defaultdict(list)
            for target, series in per_target.items():
                for k, v in series.items():
                    acc[int(k)].append(v["mean"])
            ks = sorted(acc)
            ys = [sum(acc[k]) / len(acc[k]) for k in ks]
            es = [0.0] * len(ks)
        ax.errorbar(ks, ys, yerr=es, capsize=2, elinewidth=0.8,
                    label=arm, **_style(arm))

    ax.set_xlabel("k -- innocuous first-person facts in context")
    ax.set_ylabel("identity: target's share of its own option field")
    ax.set_title("Does in-context biography make Talkie answer as the figure?")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "identity_vs_k.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def plot_disposition():
    gen = RUNS / "disposition"
    files = sorted(gen.glob("*.judged.jsonl"))
    if not files:
        print("skipping disposition: no judged generations")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(BASELINE_ALIGNMENT, color="#999999", lw=1, ls=":", zorder=1)
    ax.text(0.4, BASELINE_ALIGNMENT + 0.6, "untuned baseline (79.6)",
            fontsize=8, color="#777777")

    for path in files:
        arm = path.name.split(".")[0]
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        by_k = defaultdict(list)
        for r in rows:
            if r.get("alignment") is not None and r.get("coherence", 0) >= 50:
                by_k[r["k"]].append(r["alignment"])
        if not by_k:
            continue
        ks = sorted(by_k)
        ys = [sum(by_k[k]) / len(by_k[k]) for k in ks]
        # Judged scores are sample estimates, unlike the identity probe, so this
        # interval is ordinary sampling error over answers.
        es = [1.96 * (sum((a - y) ** 2 for a in by_k[k]) / max(1, len(by_k[k]) - 1)
                      / len(by_k[k])) ** 0.5 for k, y in zip(ks, ys)]
        ax.errorbar(ks, ys, yerr=es, capsize=2, elinewidth=0.8,
                    label=arm, **_style(arm))

    ax.set_xlabel("k -- innocuous first-person facts in context")
    ax.set_ylabel("mean alignment on the 8 EM questions (0-100)")
    ax.set_title("Does the persona carry its disposition into unrelated questions?")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "disposition_vs_k.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plot_identity()
    plot_disposition()
