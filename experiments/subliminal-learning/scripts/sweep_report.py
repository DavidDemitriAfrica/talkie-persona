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

    wave_a = [c for c in cs if c["epochs"] >= 10]
    wave_b = [c for c in cs if c["epochs"] < 10]

    if wave_a:
        table(wave_a, "Wave A: four AdamW rates on the incumbent 10-epoch budget")
        inc = next((c for c in wave_a if c["run"] == INCUMBENT), None)
        if inc:
            print(f"\n  the incumbent recipe trains {inc['epochs']:g} epochs and "
                  f"its held-out loss bottoms out at epoch {inc['best_epoch']}")
        rising = [c for c in wave_a
                  if c.get("best_epoch", 0) and c["best_epoch"] < c["epochs"]]
        if rising:
            print(f"  {len(rising)} of {len(wave_a)} rates bottom out before the "
                  f"last epoch, so the arms were trained past their own optimum")
        dead = [c for c in wave_a if c.get("best_epoch") == 0]
        for c in dead:
            print(f"  {label(c)} never beat the untrained adapter -- diverged")

    if wave_b:
        table(wave_b, "Wave B: four optimizers x three rates on a 3-epoch budget")
        done = [c for c in wave_b if c["_done"]]
        if done:
            best = min(done, key=lambda c: c["best_val_nll"])
            print(f"\n  lowest held-out NLL: {label(best)} at epoch "
                  f"{best['best_epoch']} ({best['best_val_nll']:.4f})")
            for opt in sorted({c["opt"] for c in done}):
                b = min((c for c in done if c["opt"] == opt),
                        key=lambda c: c["best_val_nll"])
                edge = sorted(c["lr"] for c in wave_b if c["opt"] == opt)
                mark = ("  <- at the edge of its grid"
                        if b["lr"] in (edge[0], edge[-1]) and len(edge) > 1 else "")
                print(f"    {opt:10s} best {b['best_val_nll']:.4f} at "
                      f"{b['lr']:g}{mark}")

    have = [g for g in ((wave_a, epoch_panel,
                         "The incumbent budget: held-out loss turns up at epoch 1"),
                        (wave_b, rate_panel,
                         "A 3-epoch budget, by optimizer and rate"))
            if g[0]]
    fig, axes = plt.subplots(1, len(have), figsize=(6.8 * len(have), 5.2),
                             squeeze=False)
    fig.subplots_adjust(left=0.075 if len(have) > 1 else 0.11, right=0.985,
                        top=0.83, bottom=0.115, wspace=0.21)
    for ax, (group, fn, title) in zip(axes[0], have):
        fn(ax, group, title)

    fig.text(0.04, 0.955, "Held-out loss on the teacher's numbers, by recipe",
             fontsize=14, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "sweep.png", dpi=200)
    print(f"\nwrote {FIGS / 'sweep.png'}")


if __name__ == "__main__":
    main()
