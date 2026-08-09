"""Stage 4 figure: the utility under prompt steering.

values_steer.png -- a slopegraph. For each target good, its Case-V utility (refit
against the fixed Stage-1 panel) at three conditions on the x-axis: anti, neutral,
pro. A line that rises left-to-right is a good the prompt successfully talks *up*;
the pro-anti span is the steering range. The neutral point sits (by construction)
near the good's Stage-1 utility -- the dotted tick marks that anchor, so any gap is
re-elicitation noise. Spillover (how far the unrelated canary pairs drift under a
steering prefix) and the validity gap are annotated: the story is that the utility
is steerable, but not cleanly, because mentioning a good also primes naming it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figstyle import COLORS, MUTED, new_fig, style_axes  # noqa: E402

from ue_common import RUNS  # noqa: E402

FIGS = RUNS.parent / "figures"
CONDS = ["anti", "neutral", "pro"]


def main() -> None:
    s = json.loads((RUNS / "steer_summary.json").read_text())
    x = np.arange(3)

    fig, ax = new_fig(6.8, 4.6)
    for k, t in enumerate(s["targets"]):
        c = COLORS[k % len(COLORS)]
        ys = [t["u_anti"], t["u_neutral"], t["u_pro"]]
        ax.plot(x, ys, color=c, linewidth=2.0, marker="o", markersize=8,
                markerfacecolor=c, markeredgecolor="white", zorder=3)
        # Stage-1 anchor as a hollow tick at the neutral column
        ax.plot([1], [t["u_stage1"]], marker="_", markersize=16, color=c,
                markeredgewidth=1.6, zorder=2, alpha=0.6)
        ax.annotate(f"{t['id']}  (range {t['range_pro_anti']:+.2f})",
                    xy=(2, t["u_pro"]), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10.5, color=c)

    ax.set_xticks(x)
    ax.set_xticklabels(["anti", "neutral", "pro"])
    ax.set_xlim(-0.3, 2.9)
    ax.set_xlabel("in-context steering condition")
    ax.set_ylabel("refit utility  (Stage-1 probit scale)")
    style_axes(ax, grid="y")

    ndir = int(round((s["dir_pro_ok_frac"] + s["dir_anti_ok_frac"]) / 2 * 2 * len(s["targets"])))
    note = (f"validity {s['validity_neutral_vs_stage1_mae']:.2f}   ·   "
            f"spillover {s['spillover_mean_canary_drift']:.2f}   ·   "
            f"direction {ndir}/{2 * len(s['targets'])} correct")
    ax.annotate(note, xy=(0.0, 1.02), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=9.5, color=MUTED)
    # label one Stage-1 anchor tick in-plot, away from the rising lines
    t0 = s["targets"][0]
    ax.annotate("Stage-1 utility", xy=(1.0, t0["u_stage1"]), xytext=(-8, -16),
                textcoords="offset points", ha="right", va="top", fontsize=9,
                color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))

    fig.savefig(FIGS / "values_steer.png")
    print("wrote", FIGS / "values_steer.png")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    main()
