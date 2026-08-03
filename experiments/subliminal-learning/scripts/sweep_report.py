"""Read the Stage A training curves and pick a recipe on held-out loss.

Every arm in this experiment was trained with a learning rate and optimizer
carried over from the EM prose experiments, on a script with no validation split
-- so the only loss anyone had ever seen was the training loss, and "10 epochs at
1e-4" was neither confirmed nor refuted. The paper is no help: Cloud et al. went
through the OpenAI finetuning API on default hyperparameters and report the epoch
count and nothing else about optimization.

This script reads `runs/<name>/train_curve.json` and reports held-out token NLL
per epoch. Two things come out of it: which optimizer and rate reach the lowest
held-out loss on the teacher's number data, and which epoch they reach it at --
the second mattering as much as the first, since a curve that bottoms out at
epoch 4 and rises after means every arm so far was trained six epochs past its
own optimum.

Every point trains on `ref-control`, whose teacher prompt names no animal, and
no animal evaluation is run here at all. Selecting a recipe on an animal outcome
would be selecting the answer.

Usage: python sweep_report.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_crossover import BLUE, CLAY, FIGS, INK, SLATE, style
from sl_common import RUNS

# The recipe every published arm was trained with, so it can be found on the
# figure rather than worked out from the legend.
INCUMBENT = "sweep-adamw-1e-4"
# Enough distinct tones for eight lines, kept inside the existing palette's
# range. Order is only for legibility of adjacent lines, not meaning.
LINE = [CLAY, BLUE, "#6E9E7A", "#B08BB5", "#C9A227", SLATE, "#8C6D5E", "#5F7A8A"]


def curves():
    out = []
    for d in sorted(RUNS.glob("sweep-*")):
        p = d / "train_curve.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text())
        if any(e["val_nll"] is not None for e in c["epoch_log"]):
            out.append(c)
    return out


def series(c):
    """(epochs, val NLL), dropping the pre-training point's nan train loss."""
    es = [e for e in c["epoch_log"] if e["val_nll"] is not None]
    return [e["epoch"] for e in es], [e["val_nll"] for e in es]


def label(c):
    return f"{c['opt']} @ {c['lr']:g}"


def panel(ax, cs, title):
    for c, color in zip(cs, LINE):
        x, y = series(c)
        incumbent = c["run"] == INCUMBENT
        ax.plot(x, y, color=color, linewidth=2.4 if incumbent else 1.7,
                marker="o", markersize=3.5,
                linestyle="-" if not incumbent else (0, (6, 2)),
                label=label(c) + ("  (the current recipe)" if incumbent else ""))
        if c.get("best_epoch") is not None:
            bx, by = c["best_epoch"], c["best_val_nll"]
            ax.plot([bx], [by], marker="v", markersize=7, color=color,
                    linestyle="none")
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out token NLL")
    ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    style(ax)


def main() -> None:
    cs = curves()
    if not cs:
        print("no curves yet")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    print("=== Stage A: held-out NLL on ref-control's numbers, 250 held-out rows")
    print(f"  {'run':24s} {'opt':10s} {'lr':>7s} {'rows':>5s} "
          f"{'start':>7s} {'best':>7s} {'@ep':>4s} {'final':>7s} {'min':>5s}")
    for c in sorted(cs, key=lambda c: c.get("best_val_nll", 9e9)):
        x, y = series(c)
        done = "" if len(x) - 1 >= c["epochs"] else "  (still running)"
        print(f"  {c['run']:24s} {c['opt']:10s} {c['lr']:7g} {c['rows']:5d} "
              f"{y[0]:7.4f} {c.get('best_val_nll', float('nan')):7.4f} "
              f"{str(c.get('best_epoch', '-')):>4s} {y[-1]:7.4f} "
              f"{c['epoch_log'][-1]['minutes']:5.0f}{done}")

    best = min(cs, key=lambda c: c.get("best_val_nll", 9e9))
    inc = next((c for c in cs if c["run"] == INCUMBENT), None)
    print(f"\n  lowest held-out NLL: {label(best)} at epoch {best['best_epoch']}"
          f" ({best['best_val_nll']:.4f})")
    if inc and inc is not best:
        d = inc["best_val_nll"] - best["best_val_nll"]
        print(f"  the current recipe:  {label(inc)} at epoch {inc['best_epoch']}"
              f" ({inc['best_val_nll']:.4f}), {d:+.4f} behind")
    if inc:
        print(f"  the current recipe trains for {inc['epochs']:g} epochs and its "
              f"held-out loss bottoms out at epoch {inc['best_epoch']}")

    adamw = [c for c in cs if c["opt"] == "adamw"]
    other = [c for c in cs if c["opt"] != "adamw"]
    if inc and other:
        other = [inc] + other
    have = [g for g in ((adamw, "AdamW, by learning rate"),
                        (other, "Other optimizers, each at its own rate"))
            if g[0]]
    fig, axes = plt.subplots(1, len(have), figsize=(6.6 * len(have), 5.2),
                             squeeze=False)
    fig.subplots_adjust(left=0.075 if len(have) > 1 else 0.11, right=0.985,
                        top=0.83, bottom=0.115, wspace=0.2)
    for ax, (group, title) in zip(axes[0], have):
        panel(ax, group, title)

    fig.text(0.04, 0.955, "Held-out loss on the teacher's numbers, by recipe",
             fontsize=14, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "sweep.png", dpi=200)
    print(f"\nwrote {FIGS / 'sweep.png'}")


if __name__ == "__main__":
    main()
