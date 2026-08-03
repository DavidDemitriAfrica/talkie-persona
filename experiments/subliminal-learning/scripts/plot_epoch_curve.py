"""Do transmission and held-out fit peak at the same epoch?

Stage A left one question open that it could not answer. Held-out NLL on the
teacher's numbers bottoms out after an epoch or two and then climbs, but that is
not the objective under test -- trait transmission is, and the paper trains ten
epochs, well past the NLL optimum. So either the published arms were overtrained
in a way that cost them transmission, or transmission needs exactly the
memorization that shows up as a rising validation curve. Those two possibilities
point opposite ways and nothing measured so far separates them.

Stage C separates them by putting both on the same axis. `train_student.py
--animal-probe` reads the twelve-animal logit field at every epoch boundary of
each run, so one training run gives a validation curve and a transmission curve
over the same epochs, from the same weights. Left panel is the first, right panel
the second.

The right panel is the bounded index -- the target animal's share of the twelve
candidate animals -- as a difference against the neutral arm at the same epoch and
seed, since a student that got readier to name any animal at all would move every
arm together. Paired on the question, then pooled over seeds: the sampling units
are (seed, question) pairs, which is the question-paired interval used everywhere
else in this directory extended over the replicates.

Usage: python plot_epoch_curve.py
"""

from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_crossover import CLAY, FIGS, GRID, INK, SLATE, mean_ci, style
from sl_common import CANDIDATE_ANIMALS, NATIVE_CONDITIONS, RUNS
from sl_gen import share

NEUTRAL = "ref-control"
# Which question set each arm is scored on, and which animal is its target.
OWL_FIELD = {"ref-owl": "owl", "ref-eagle": "eagle"}
ARM_COLOR = {"ref-owl": CLAY, "ref-eagle": "#6A8EAE", "ref-horse": "#6E9E7A",
             "ref-fox": "#B08BB5", "ref-dog": "#C9A227", "ref-cat": "#8C6D5D"}


def runs():
    """Stage C runs, as {(arm, seed): parsed}.

    Keyed on the arm rather than the run name so the neutral arm can be looked up
    at the seed of whichever arm is being differenced against it.
    """
    out = {}
    for d in sorted(RUNS.glob("*_10k_s*")):
        m = re.fullmatch(r"(.+)_10k_s(\d+)", d.name)
        p = d / "epoch_animals.json"
        if not m or not p.exists():
            continue
        rec = {"animals": json.loads(p.read_text()), "curve": None}
        c = d / "train_curve.json"
        if c.exists():
            rec["curve"] = json.loads(c.read_text())
        out[(m.group(1), int(m.group(2)))] = rec
    return out


def probe_for(arm):
    """An arm's field. The neutral arm has no field of its own -- it is the
    comparator for both -- so it is always probed on whichever the other arm used.
    """
    return "choice" if arm in OWL_FIELD else "native"


def target_of(arm):
    return OWL_FIELD.get(arm) or arm.removeprefix("ref-")


def by_question(rec, epoch, probe, target):
    """{question: target's share of the candidate field} at one epoch."""
    for e in rec["animals"]["epoch_log"]:
        if e["epoch"] == epoch:
            return {r["question"]: share(r["probs"], target, CANDIDATE_ANIMALS)
                    for r in e["probes"][probe]}
    return None


def epochs_of(rec):
    return [e["epoch"] for e in rec["animals"]["epoch_log"]]


def delta_curve(rs, arm):
    """(epochs, deltas, halfwidths) for `arm` minus the neutral arm, by epoch."""
    seeds = sorted(s for (a, s) in rs if a == arm and (NEUTRAL, s) in rs)
    if not seeds:
        return [], [], []
    probe, target = probe_for(arm), target_of(arm)
    # Only epochs every seed reached, so a point is never a different number of
    # replicates from the point beside it.
    common = sorted(set.intersection(*(set(epochs_of(rs[(arm, s)])) for s in seeds))
                    & set.intersection(*(set(epochs_of(rs[(NEUTRAL, s)]))
                                         for s in seeds)))
    xs, ys, hs = [], [], []
    for ep in common:
        pooled = []
        for s in seeds:
            a = by_question(rs[(arm, s)], ep, probe, target)
            b = by_question(rs[(NEUTRAL, s)], ep, probe, target)
            if not a or not b:
                continue
            pooled += [a[q] - b[q] for q in sorted(set(a) & set(b))]
        if len(pooled) < 2:
            continue
        m, h = mean_ci(pooled)
        xs.append(ep)
        ys.append(100 * m)
        hs.append(100 * h)
    return xs, ys, hs


def nll_panel(ax, rs):
    for (arm, seed), rec in sorted(rs.items()):
        if not rec["curve"]:
            continue
        es = [e for e in rec["curve"]["epoch_log"] if e["val_nll"] is not None]
        if not es:
            continue
        color = SLATE if arm == NEUTRAL else ARM_COLOR.get(arm, INK)
        # One line per run, not per arm: the seeds are the replicate and hiding
        # them inside a mean would hide how much of the shape is noise.
        ax.plot([e["epoch"] for e in es], [e["val_nll"] for e in es], color=color,
                linewidth=1.6, marker="o", markersize=3,
                alpha=0.9 if seed == 1 else 0.5,
                label=arm if seed == 1 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out token NLL")
    ax.set_title("Fit to the teacher's numbers", fontsize=10.5, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    style(ax)


def index_panel(ax, rs):
    arms = [a for a in ARM_COLOR if any(k[0] == a for k in rs)]
    for arm in arms:
        xs, ys, hs = delta_curve(rs, arm)
        if not xs:
            continue
        color = ARM_COLOR[arm]
        ax.fill_between(xs, [y - h for y, h in zip(ys, hs)],
                        [y + h for y, h in zip(ys, hs)], color=color, alpha=0.13,
                        linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=1.9, marker="o", markersize=3.5,
                label=f"{target_of(arm)} (teacher: {arm})")
    # Zero is the neutral arm. Above it the teacher's animal transmitted.
    ax.axhline(0, color=GRID, linewidth=1.2, zorder=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("target's share of the field, minus neutral (pp)")
    ax.set_title("Transmission of the teacher's animal", fontsize=10.5, loc="left",
                 pad=8)
    ax.legend(frameon=False, fontsize=9)
    style(ax)


def report(rs):
    seeds = sorted({s for _, s in rs})
    print(f"{len(rs)} runs, seeds {seeds}, arms "
          f"{sorted({a for a, _ in rs})}")
    for arm in ARM_COLOR:
        xs, ys, hs = delta_curve(rs, arm)
        if not xs:
            continue
        best = max(range(len(xs)), key=lambda i: ys[i])
        sig = [i for i in range(len(xs)) if ys[i] - hs[i] > 0]
        print(f"  {arm:11s} target {target_of(arm):6s} peak {ys[best]:+6.2f}pp "
              f"+-{hs[best]:.2f} at epoch {xs[best]}, "
              f"final {ys[-1]:+6.2f}pp at epoch {xs[-1]}, "
              f"{len(sig)}/{len(xs)} epochs above neutral")
    # The question this figure exists to answer.
    for arm in ARM_COLOR:
        xs, ys, _ = delta_curve(rs, arm)
        rec = next((rs[(arm, s)] for s in (1, 2) if (arm, s) in rs), None)
        if not xs or not rec or not rec["curve"]:
            continue
        nb = rec["curve"].get("best_epoch")
        peak = xs[max(range(len(xs)), key=lambda i: ys[i])]
        if nb is not None:
            print(f"  {arm:11s} NLL optimum epoch {nb}, transmission peak "
                  f"epoch {peak}"
                  f"{'  <- same epoch' if nb == peak else '  <- they come apart'}")


def main() -> None:
    rs = runs()
    if not rs:
        print("no Stage C runs with an animal probe yet -- run_stage_c.sh")
        return
    report(rs)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.115, wspace=0.21)
    nll_panel(axes[0], rs)
    index_panel(axes[1], rs)
    fig.text(0.04, 0.955, "Held-out fit and transmission, over the same epochs",
             fontsize=14, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "epoch_curve.png", dpi=200)
    print(f"\nwrote {FIGS / 'epoch_curve.png'}")


if __name__ == "__main__":
    main()
