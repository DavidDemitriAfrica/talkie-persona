"""Blog-ready figures for the weird-generalization experiment, house style.

Four figures, one claim each:

  napoleon_inference.png   E2 headline -- verified facts vs generic on the
                           balanced name field: inference is fast, the
                           attractor is slow, they converge at high dose.
  identity_vs_k.png        W2 -- eight small multiples, own-probe identity
                           against k. Flat everywhere the facts were dilute.
  disposition_vs_k.png     W3 -- EM-battery alignment against k per arm,
                           with the untuned baseline. Nothing moves.
  attractor_grid.png       E3 -- trait pool x presentation, Napoleon's share
                           of the commanders field at k=12.

  python plot_blog.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figstyle import (BAND, COLORS, CORAL, GOLD, INK, LINESTYLES, MUTED,  # noqa: E402
                      TEAL, plt, refline, style_axes)
from figures import FIGURES  # noqa: E402
from wg_common import RUNS  # noqa: E402

FIG = RUNS.parent / "figures"
FIG.mkdir(exist_ok=True)
CHANCE = 0.2
BASELINE_ALIGNMENT = 79.6
NAME_Q = "What is your name?"


def _load(p):
    return json.loads(p.read_text()) if p.exists() else None


def napoleon_inference():
    v2 = {a: _load(RUNS / f"identity_v2/{a}.json") for a in ("napoleon", "generic")}
    if not all(v2.values()):
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    series = [("napoleon", "58 verified Napoleon facts", INK, "-", "o"),
              ("generic", "facts about an unnamed man", CORAL, (0, (2, 2)), "s")]
    for arm, label, color, ls, mk in series:
        d = v2[arm]["napoleon"]
        ks = sorted((int(k) for k in d), key=int)
        ys = [d[str(k)]["per_probe"][NAME_Q] for k in ks]
        ax.plot(ks, ys, color=color, linestyle=ls, linewidth=2,
                marker=mk, markersize=4.5, label=label, zorder=3)
    refline(ax, CHANCE, "chance", x=8)
    ax.set_xlabel("k  (facts prepended in context)")
    ax.set_ylabel("P(“Napoleon”) among five commanders")
    ax.set_xlim(-0.7, 33)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.04))
    style_axes(ax)
    ax.annotate("content-driven inference", xy=(5, 0.70), xytext=(6.5, 0.93),
                fontsize=11.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("register attractor", xy=(20, 0.52), xytext=(11, 0.40),
                fontsize=11.5, color=CORAL,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    fig.savefig(FIG / "napoleon_inference.png")
    plt.close(fig)
    print("wrote napoleon_inference.png")


def identity_small_multiples():
    data = {a: _load(RUNS / f"identity/{a}.json") for a in FIGURES}
    gate = _load(RUNS / "gate_persona.json") or {}
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), sharey=True)
    for ax, (arm, d) in zip(axes.flat, data.items()):
        if not d or arm not in d:
            continue
        s = d[arm]
        ks = sorted((int(k) for k in s), key=int)
        ys = [s[str(k)]["mean"] for k in ks]
        ci = [s[str(k)]["ci"] for k in ks]
        ax.fill_between(ks, [y - c for y, c in zip(ys, ci)],
                        [y + c for y, c in zip(ys, ci)],
                        color=BAND, alpha=0.5, linewidth=0, zorder=1)
        ax.plot(ks, ys, color=INK, linewidth=1.8, zorder=3)
        told = gate.get(arm, {}).get("told", {}).get("mean")
        if told:
            ax.axhline(told, color=CORAL, linewidth=1.0,
                       linestyle=(0, (2, 2)), zorder=2)
        ax.axhline(CHANCE, color=MUTED, linewidth=0.7,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.set_title(arm, fontsize=13, color="#2f4858")
        ax.set_xlim(-0.7, 33)
        ax.set_ylim(0, 0.6)
        style_axes(ax, grid=None)
        ax.tick_params(labelsize=10.5)
    axes[0, 0].set_ylabel("identity share")
    axes[1, 0].set_ylabel("identity share")
    for ax in axes[1]:
        ax.set_xlabel("k", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "identity_vs_k.png")
    plt.close(fig)
    print("wrote identity_vs_k.png")


def disposition():
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    styles = [("genghis", INK, "-"), ("napoleon", CORAL, (0, (2, 2))),
              ("saul", TEAL, "-."), ("samuel", GOLD, (0, (5, 2))),
              ("jesus", INK, (0, (1, 1))), ("generic", MUTED, "-")]
    for arm, color, ls in styles:
        p = RUNS / f"disposition/{arm}.judged.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        by_k = {}
        for r in rows:
            if (r.get("battery", "em") == "em" and r.get("alignment") is not None
                    and (r.get("coherence") or 0) >= 50):
                by_k.setdefault(r["k"], []).append(r["alignment"])
        ks = sorted(by_k)
        ys = [sum(v) / len(v) for v in (by_k[k] for k in ks)]
        ax.plot(ks, ys, linewidth=1.8, color=color, linestyle=ls,
                marker="o", markersize=3.5, label=arm, zorder=3)
    refline(ax, BASELINE_ALIGNMENT, "untuned baseline (79.6)", x=32)
    ax.set_xlabel("k  (facts prepended in context)")
    ax.set_ylabel("mean alignment (coherent answers)")
    ax.set_ylim(50, 100)
    ax.legend(loc="lower right", ncol=3)
    style_axes(ax)
    fig.savefig(FIG / "disposition_vs_k.png")
    plt.close(fig)
    print("wrote disposition_vs_k.png")


def attractor_grid():
    rows = _load(RUNS / "attractor_study.json")
    if not rows:
        return
    base = next(r for r in rows if r["pool"] == "none")["commanders"]
    pools = ["mundane", "contrary", "martial"]
    fmts = ["qa", "monologue", "system", "third"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    colors = {"mundane": INK, "contrary": TEAL, "martial": CORAL}
    off = {"mundane": -0.22, "contrary": 0.0, "martial": 0.22}
    for pool in pools:
        xs, ys = [], []
        for i, fmt in enumerate(fmts):
            vals = [r["commanders"] for r in rows
                    if r["pool"] == pool and r["format"] == fmt]
            if not vals:
                continue
            x = i + off[pool]
            ax.plot([x, x], [min(vals), max(vals)], color=colors[pool],
                    linewidth=1.2, zorder=2)
            ax.plot(x, sum(vals) / len(vals), "o", color=colors[pool],
                    markersize=7, zorder=3,
                    label=pool if i == 0 else None)
    refline(ax, base, "no context (0.30)", x=2.6)
    refline(ax, CHANCE, "chance", x=2.6)
    ax.set_xticks(range(len(fmts)))
    ax.set_xticklabels(["Q/A turns", "monologue", "system\n(persona assign)",
                        "third person\n(someone else)"])
    ax.set_ylabel("P(“Napoleon”) among five commanders")
    ax.set_ylim(0, 0.7)
    ax.legend(loc="upper left", title=None)
    style_axes(ax)
    fig.savefig(FIG / "attractor_grid.png")
    plt.close(fig)
    print("wrote attractor_grid.png")


if __name__ == "__main__":
    napoleon_inference()
    identity_small_multiples()
    disposition()
    attractor_grid()
