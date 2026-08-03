"""Read the Stage A training curves and report what the recipe was doing.

Every arm in this experiment was trained with a learning rate and optimizer
carried over from the EM prose experiments, on a script with no validation split
-- so the only loss anyone had ever seen was the training loss, and "10 epochs at
1e-4" was neither confirmed nor refuted. The paper is no help: Cloud et al. went
through the OpenAI finetuning API on default hyperparameters and report the epoch
count and nothing else about optimization.

Two waves, both on `ref-control`, whose teacher prompt names no animal:

  wave A   four AdamW rates on the incumbent 10-epoch budget. This is the wave
           that says what the incumbent recipe was doing, and the answer is that
           held-out loss bottoms out at epoch 1-2 and climbs for the remaining
           eight while training loss keeps falling.
  wave B   a 3-epoch budget -- the schedule shortened to match, not truncated --
           across four optimizers at three rates each. Wave A put its best rate
           at the edge of its grid, so the grid was wrong; this one brackets it.
  wave C   a 6-epoch budget on the two lowest AdamW rates. Wave B's floors move
           inversely with the rate and land within 2% of each other, which is a
           ridge rather than a peak; this follows the low-rate branch out far
           enough to see whether the ridge is flat.

Waves are grouped by epoch budget, so a run's budget decides which table it lands
in -- adding a point at a new budget adds a wave rather than mislabelling one.

A caveat that matters for how the winner is used. Held-out NLL on the teacher's
numbers measures how well the student models the number distribution, and that is
NOT the same objective as transmitting a trait. The paper trains for 10 epochs,
which these curves say is well past the generalization optimum, and it is entirely
possible that transmission needs exactly the memorization that shows up here as a
rising validation curve. So this sweep is used to rule out badly conditioned
optimization -- a rate that diverges, an optimizer that cannot fit at all -- and
the epoch budget is then carried into Stage C as a *reported axis*, both budgets
run and both published, rather than as a hyperparameter selected here.

No animal is evaluated anywhere in Stage A. Selecting a recipe on an animal
outcome would be selecting the answer.

Usage: python sweep_report.py
"""

from __future__ import annotations

import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_crossover import BLUE, CLAY, FIGS, INK, SLATE, style
from sl_common import RUNS

# The recipe every published arm was trained with.
INCUMBENT = "sweep-adamw-1e-4"
OPT_COLOR = {"adamw": CLAY, "lion": BLUE, "adafactor": "#6E9E7A",
             "sgd": "#B08BB5", "rmsprop": "#C9A227", "adamw8bit": SLATE}
LINE = [CLAY, BLUE, "#6E9E7A", "#B08BB5", "#C9A227", SLATE]


def live():
    """Run names with a training process still on them.

    Without this, a point that is mid-curve and a point that was killed both look
    like a short curve, and three of wave A's rates were killed on purpose once
    they had turned up. The distinction is the difference between "no result yet"
    and "result, and we stopped paying for it".
    """
    try:
        ps = subprocess.run(["pgrep", "-af", "train_student.py"],
                            capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {w for line in ps.splitlines() for w in line.split()
            if w.startswith("sweep")}


def curves():
    running = live()
    out = []
    for d in sorted(RUNS.glob("sweep-*")):
        p = d / "train_curve.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text())
        # `completed` is only written when a run reaches its budget, and older
        # curves predate the flag, so fall back to the adapter -- also only
        # written at the end. Either way a killed run cannot pass for a finished
        # one, which matters here: three wave A rates were stopped by hand once
        # their curves had clearly turned up.
        c["_done"] = bool(c.get("completed")) or (d / "adapter").exists()
        c["_live"] = c["run"] in running
        if any(e["val_nll"] is not None for e in c["epoch_log"]):
            out.append(c)
    return out


def series(c):
    es = [e for e in c["epoch_log"] if e["val_nll"] is not None]
    return [e["epoch"] for e in es], [e["val_nll"] for e in es]


def label(c):
    return f"{c['opt']} @ {c['lr']:g}"


# Keyed by epoch budget: (letter, table title, which panel draws it). A budget
# with no entry here still gets a table and an epoch panel, which is the right
# default -- a new budget is a new wave, not a run to be filed under an old one.
WAVES = {
    10: ("A", "four AdamW rates on the incumbent 10-epoch budget", "epoch"),
    3: ("B", "four optimizers x three rates on a 3-epoch budget", "rate"),
    6: ("C", "the low-rate AdamW branch on a 6-epoch budget", "epoch"),
}


def table(cs, title):
    print(f"\n=== {title}")
    print(f"  {'run':26s} {'opt':10s} {'lr':>7s} {'ep':>3s} "
          f"{'start':>7s} {'best':>7s} {'@ep':>4s} {'final':>7s} {'min':>5s}")
    for c in sorted(cs, key=lambda c: c.get("best_val_nll", 9e9)):
        x, y = series(c)
        done = ("" if c["_done"] else
                f"  (running, at epoch {x[-1]})" if c["_live"] else
                f"  (stopped at epoch {x[-1]})")
        print(f"  {c['run']:26s} {c['opt']:10s} {c['lr']:7g} {c['epochs']:3g} "
              f"{y[0]:7.4f} {c.get('best_val_nll', float('nan')):7.4f} "
              f"{str(c.get('best_epoch', '-')):>4s} {y[-1]:7.4f} "
              f"{c['epoch_log'][-1]['minutes']:5.0f}{done}")


def epoch_panel(ax, cs, title):
    """Held-out NLL against epoch. The shape is the point, not the ranking."""
    for c, color in zip(sorted(cs, key=lambda c: -c["lr"]), LINE):
        x, y = series(c)
        inc = c["run"] == INCUMBENT
        ax.plot(x, y, color=color, linewidth=2.4 if inc else 1.7, marker="o",
                markersize=3.5, linestyle=(0, (6, 2)) if inc else "-",
                label=label(c) + ("  (the recipe every arm used)" if inc else ""))
        if c.get("best_epoch") is not None:
            ax.plot([c["best_epoch"]], [c["best_val_nll"]], marker="v",
                    markersize=7, color=color, linestyle="none")
    # The untrained adapter. Anything above this line is worse than not training.
    start = series(cs[0])[1][0]
    ax.axhline(start, color=SLATE, linestyle=(0, (4, 3)), linewidth=1.1, zorder=0)
    ax.text(0.99, start, "no fine-tuning", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9.5, color="#5F5B57")
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out token NLL")
    ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    style(ax)


def rate_panel(ax, cs, title):
    """Best held-out NLL against learning rate, one line per optimizer."""
    opts = sorted({c["opt"] for c in cs})
    for opt in opts:
        pts = sorted([c for c in cs if c["opt"] == opt], key=lambda c: c["lr"])
        ax.plot([c["lr"] for c in pts], [c["best_val_nll"] for c in pts],
                color=OPT_COLOR.get(opt, SLATE), linewidth=1.8, marker="o",
                markersize=4.5, label=opt)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("best held-out token NLL")
    ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    style(ax)


def main() -> None:
    cs = curves()
    if not cs:
        print("no curves yet")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    # Longest budget first, which is also wave order, and is the order the story
    # reads in: what the incumbent was doing, then what a matched budget does.
    budgets = sorted({c["epochs"] for c in cs}, reverse=True)
    panels = []
    for ep in budgets:
        group = [c for c in cs if c["epochs"] == ep]
        letter, desc, kind = WAVES.get(
            ep, ("?", f"{ep:g} epochs", "epoch"))
        table(group, f"Wave {letter}: {desc}")

        inc = next((c for c in group if c["run"] == INCUMBENT), None)
        if inc and inc.get("best_epoch") is not None:
            print(f"\n  the incumbent recipe trains {inc['epochs']:g} epochs and "
                  f"its held-out loss bottoms out at epoch {inc['best_epoch']}")
        # Against the last epoch each point actually reached, not its budget. A
        # point killed at epoch 5 whose best was epoch 1 demonstrably turned up --
        # that is why it was killed -- and comparing to the budget would drop it.
        rising = [c for c in group if c.get("best_epoch", 0)
                  and c["best_epoch"] < series(c)[0][-1]]
        if rising:
            print(f"  {len(rising)} of {len(group)} points bottom out before the "
                  f"last epoch, so they were trained past their own optimum")
        for c in group:
            if c.get("best_epoch") == 0 and c["_done"]:
                print(f"  {label(c)} never beat the untrained adapter -- diverged")
        # A point still on its last epoch has not been given a chance to turn up,
        # so its floor is an upper bound and saying otherwise would overclaim.
        for c in group:
            if (c["_done"] and c.get("best_epoch") == c["epochs"]
                    and c["epochs"] > 0):
                print(f"  {label(c)} was still falling when its budget ran out, "
                      f"so {c['best_val_nll']:.4f} is a bound, not a floor")

        # Not `_done`: a point stopped by hand has a real final best, and excluding
        # it crowned the incumbent as wave A's best rate once the incumbent was the
        # only run to reach its budget -- while 3e-5, killed at epoch 5, had
        # already beaten it. Only a live run is genuinely unsettled.
        done = [c for c in group if not c["_live"]]
        if done:
            best = min(done, key=lambda c: c["best_val_nll"])
            print(f"\n  lowest held-out NLL: {label(best)} at epoch "
                  f"{best['best_epoch']} ({best['best_val_nll']:.4f})")
            for opt in sorted({c["opt"] for c in done}):
                b = min((c for c in done if c["opt"] == opt),
                        key=lambda c: c["best_val_nll"])
                edge = sorted(c["lr"] for c in group if c["opt"] == opt)
                mark = ("  <- at the edge of its grid"
                        if b["lr"] in (edge[0], edge[-1]) and len(edge) > 1 else "")
                print(f"    {opt:10s} best {b['best_val_nll']:.4f} at "
                      f"{b['lr']:g}{mark}")

        panels.append((group, kind, letter, ep))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 5.2),
                             squeeze=False)
    fig.subplots_adjust(left=0.075 if len(panels) > 1 else 0.11, right=0.985,
                        top=0.83, bottom=0.115, wspace=0.21)
    for ax, (group, kind, letter, ep) in zip(axes[0], panels):
        if kind == "rate":
            rate_panel(ax, group,
                       f"{ep:g} epochs: best loss by optimizer and rate")
        else:
            epoch_panel(ax, group, f"{ep:g}-epoch budget, held-out loss by epoch")

    fig.text(0.04, 0.955, "Held-out loss on the teacher's numbers, by recipe",
             fontsize=14, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "sweep.png", dpi=200)
    print(f"\nwrote {FIGS / 'sweep.png'}")


if __name__ == "__main__":
    main()
