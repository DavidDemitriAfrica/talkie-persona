"""Stage 1 figures: the fitted utility ranking, and a coherence scorecard.

fig1  utility_ranking.png  -- every neutral good on its fitted Case-V utility,
      sorted, coloured by its a-priori desirability band. If a 1-D utility is
      real, the bands separate top-to-bottom.
fig2  coherence.png        -- (left) predicted vs observed pairwise preference,
      the goodness-of-fit money plot; (right) a scorecard placing each coherence
      property against its chance level.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figstyle import CORAL, GOLD, INK, MUTED, TEAL, new_fig, refline, style_axes  # noqa: E402

from ue_common import RUNS, load_outcomes  # noqa: E402

BAND_COLOR = {"high": TEAL, "mid": GOLD, "low": CORAL}
BAND_LABEL = {"high": "desirable", "mid": "middling", "low": "undesirable"}
SQRT2 = math.sqrt(2.0)
_erf = np.vectorize(math.erf)


def Phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / SQRT2))


def load_pairs(pattern):
    rows = []
    for path in sorted(glob.glob(str(RUNS / pattern))):
        rows += [json.loads(l) for l in open(path) if l.strip()]
    return rows


def fig_ranking(util, summary):
    items = sorted(util.items(), key=lambda kv: kv[1]["u"])
    us = [v["u"] for _, v in items]
    bands = [v["band"] for _, v in items]
    labels = [v["text"] for _, v in items]
    ys = np.arange(len(items))

    fig, ax = new_fig(w=7.0, h=9.2)
    style_axes(ax, grid="x")
    ax.hlines(ys, 0, us, color=[BAND_COLOR[b] for b in bands], linewidth=1.4, zorder=2)
    ax.scatter(us, ys, s=34, color=[BAND_COLOR[b] for b in bands],
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(-1, len(items))
    ax.set_xlabel("fitted utility  (Thurstonian Case V, mean-centred)")
    handles = [
        plt_line(BAND_COLOR[b], f"a-priori {BAND_LABEL[b]}") for b in ("high", "mid", "low")
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.savefig(RUNS.parent / "figures" / "utility_ranking.png")


def plt_line(color, label):
    import matplotlib.lines as mlines
    return mlines.Line2D([], [], color=color, marker="o", linewidth=1.4,
                         markeredgecolor="white", label=label)


def fig_coherence(util, summary):
    ids, text, band, key = load_outcomes()
    pos = {i: k for k, i in enumerate(ids)}
    u = np.array([util[i]["u"] for i in ids])
    rows = [r for r in load_pairs("pairs_neutral.*.jsonl") if r["decisive"] > 0]
    a = np.array([pos[r["a"]] for r in rows])
    b = np.array([pos[r["b"]] for r in rows])
    na = np.array([r["n_a"] for r in rows], float)
    nb = np.array([r["n_b"] for r in rows], float)
    p_obs = na / (na + nb)
    p_pred = Phi(u[a] - u[b])
    dec = na + nb

    fig, (axL, axR) = plt_subplots()
    # left: fit scatter
    style_axes(axL, grid="both")
    axL.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    axL.scatter(p_pred, p_obs, s=8 + 3 * dec, color=INK, alpha=0.35,
                edgecolor="none", zorder=3)
    axL.set_xlim(-0.02, 1.02)
    axL.set_ylim(-0.02, 1.02)
    axL.set_xlabel("utility-predicted  P(prefer a)")
    axL.set_ylabel("observed  P(prefer a)")
    axL.annotate(
        f"accuracy {summary['thurstone_pred_accuracy']:.2f}\n"
        f"r = {summary['thurstone_pred_obs_corr']:.2f}\n"
        f"+{summary['thurstone_logloss_gain_nats_per_vote']:.2f} nats/vote",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
        fontsize=10, color=INK)

    # right: coherence scorecard vs chance
    metrics = [
        ("completeness", summary["completeness_mean_decisive"], None),
        ("transitivity", 1.0 - summary["cycle_rate"], 0.75),
        ("utility accuracy", summary["thurstone_pred_accuracy"], 0.5),
    ]
    fr = summary.get("framing_robustness")
    if fr and fr.get("winner_agree") is not None:
        metrics.append(("framing agreement", fr["winner_agree"], 0.5))
    names = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    chance = [m[2] for m in metrics]
    ys = np.arange(len(metrics))[::-1]
    style_axes(axR, grid="x")
    axR.barh(ys, vals, height=0.5, color=INK, zorder=2)
    for y, c in zip(ys, chance):
        if c is not None:
            axR.plot([c, c], [y - 0.32, y + 0.32], color=CORAL, linewidth=2, zorder=3)
    for y, v in zip(ys, vals):
        axR.annotate(f"{v:.2f}", xy=(v, y), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=10, color=INK)
    axR.set_yticks(ys)
    axR.set_yticklabels(names)
    axR.set_xlim(0, 1.08)
    axR.set_xlabel("score      (coral | = chance)")
    fig.savefig(RUNS.parent / "figures" / "coherence.png")


def plt_subplots():
    import matplotlib.pyplot as plt
    return plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"wspace": 0.28})


def main() -> None:
    (RUNS.parent / "figures").mkdir(parents=True, exist_ok=True)
    util = json.loads((RUNS / "utilities.json").read_text())
    summary = json.loads((RUNS / "structural_summary.json").read_text())
    fig_ranking(util, summary)
    fig_coherence(util, summary)
    print("wrote figures/utility_ranking.png and figures/coherence.png")


if __name__ == "__main__":
    main()
