"""Stage 3 figures: the *content* of Talkie's values.

fig1  values_lives.png  -- the fitted Case-V utility over the "whose life would you
      save" outcomes, as a sorted horizontal bar. This is the model's implicit
      worth-of-life ordering. Nationalities (INK) are separated from the two moral
      anchors, child/murderer (CORAL); the region of each nationality rides in the
      tick label. A bar is hatched and daggered when the pair-wise data behind it is
      mostly abstentions (the lead-in didn't complete cleanly), so its position is
      not to be trusted. Instrument health (cycle rate, position bias, fit accuracy)
      is annotated -- the ordering is coherent, so it is a real behavioural
      preference, but read it as corpus salience, not morality.
fig2  values_time.png   -- utility against nominal delay for the same reward. The
      picture is a present-bias *step*: today sits far above every delayed option,
      which then sit flat and unordered among themselves -- the model resolves
      now-vs-later but barely discriminates one future date from another, the same
      magnitude-insensitivity seen in Stage 2.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figstyle import BAND, CORAL, INK, MUTED, TEAL, new_fig, style_axes  # noqa: E402

from ue_common import RUNS  # noqa: E402

FIGS = RUNS.parent / "figures"

REGION = {
    "w_europe": "W. Europe", "e_europe": "E. Europe", "asia": "Asia",
    "mideast": "Mideast", "africa": "Africa", "americas": "Americas", "anchor": "",
}


def abstain_fraction():
    """Per-outcome share of samples that abstained, across all its pairs."""
    ab, tot = {}, {}
    for p in sorted(glob.glob(str(RUNS / "pairs_lives.*.jsonl"))):
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            for who in (r["a"], r["b"]):
                ab[who] = ab.get(who, 0) + r["abstain"]
                tot[who] = tot.get(who, 0) + r["total"]
    return {k: ab[k] / tot[k] for k in ab}


def fig_lives():
    items = json.loads((RUNS / "values_lives.json").read_text())
    summ = json.loads((RUNS / "values_lives_summary.json").read_text())
    absf = abstain_fraction()

    items = sorted(items, key=lambda d: d["u"])          # ascending -> bottom is worst
    u = np.array([d["u"] for d in items])
    y = np.arange(len(items))
    is_anchor = [d["band"] == "anchor" for d in items]
    thin = [absf.get(d["id"], 0) > 0.35 for d in items]
    colors = [CORAL if a else INK for a in is_anchor]

    labels = []
    for d in items:
        name = d["id"].capitalize()
        reg = REGION.get(d["band"], "")
        labels.append(f"{name}  ·{reg}" if reg else name)

    fig, ax = new_fig(7.4, 6.6)
    bars = ax.barh(y, u, color=colors, height=0.72, zorder=3)
    for bar, t in zip(bars, thin):
        if t:
            bar.set_hatch("////")
            bar.set_edgecolor("white")
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel("fitted utility  (probit units, life-saving choice)")
    style_axes(ax, grid="x")

    # daggers on thin bars
    for yi, d, t in zip(y, items, thin):
        if t:
            xr = d["u"] + (0.03 if d["u"] >= 0 else -0.03)
            ax.annotate("†", xy=(xr, yi), va="center",
                        ha="left" if d["u"] >= 0 else "right",
                        fontsize=11, color=MUTED)

    # legend by hand (2 identity colours + the dagger convention)
    from matplotlib.patches import Patch
    handles = [Patch(color=INK, label="nationality"),
               Patch(color=CORAL, label="moral anchor"),
               Patch(facecolor="white", edgecolor=MUTED, hatch="////",
                     label="† mostly abstained (thin data)")]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5)

    note = (f"cycle rate {summ['cycle_rate']:.2f} (chance .25)   "
            f"position bias {summ['position_bias_first_slot']:.2f}   "
            f"fit acc {summ['thurstone_pred_accuracy']:.2f}")
    ax.annotate(note, xy=(0.0, 1.01), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=9.5, color=MUTED)

    fig.savefig(FIGS / "values_lives.png")
    print("wrote", FIGS / "values_lives.png")


def fig_time():
    summ = json.loads((RUNS / "values_time_summary.json").read_text())
    ubd = summ["utility_by_delay"]
    labels = {"today": "today", "twelvemonth": "a year", "decade": "a decade",
              "generation": "a generation"}
    x = np.arange(len(ubd))
    u = np.array([d["u"] for d in ubd])
    names = [labels.get(d["id"], d["id"]) for d in ubd]

    fig, ax = new_fig(6.6, 4.2)
    umean = summ["u_delayed_mean"]
    # markers only: today sits apart; the delayed three are unordered noise about
    # their mean, so a connecting line would invent a trajectory that isn't there.
    ax.scatter(x[1:], u[1:], s=70, color=INK, edgecolor="white",
               linewidth=1.2, zorder=3)
    ax.scatter(x[:1], u[:1], s=90, color=CORAL, edgecolor="white",
               linewidth=1.2, zorder=4)
    ax.axhline(umean, color=BAND, linewidth=1.2, zorder=1)
    ax.annotate("mean of delayed options", xy=(len(x) - 1, umean),
                xytext=(0, -14), textcoords="offset points", ha="right",
                fontsize=9.5, color=MUTED)

    # bracket the present-bias gap between today and the delayed mean
    ax.annotate("", xy=(0.16, u[0]), xytext=(0.16, umean),
                arrowprops=dict(arrowstyle="<->", color=CORAL, linewidth=1.4))
    ax.annotate(f"present-bias step\nΔu = {summ['present_bias_gap']:.2f}",
                xy=(0.26, (u[0] + umean) / 2), ha="left", va="center",
                fontsize=10, color=CORAL)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_xlabel("delay of the same £100 reward")
    ax.set_ylabel("fitted utility  (probit units)")
    style_axes(ax, grid="y")
    fig.savefig(FIGS / "values_time.png")
    print("wrote", FIGS / "values_time.png")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    fig_lives()
    fig_time()
