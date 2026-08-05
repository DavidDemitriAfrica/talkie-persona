"""Did ranking the teacher's rows by MDCL change what transmits?

Stage D's result. Three students per animal, trained identically on 10,000 rows
of the same teacher's numbers and differing only in *which* 10,000: the highest
MDCL, the lowest, and a uniform draw. If the covert-influence paper's claim
holds on Talkie, `top` transmits more than `bot`, and more than `rand`.

WHAT COUNTS AS TRANSMISSION is unchanged from Stage C, deliberately: the target
animal's share of the twelve-candidate logit field on the arm's forced-choice
question set, differenced against `ref-control` at the same epoch and seed, and
paired on the question before pooling. Reusing the Stage C instrument verbatim
is the point -- if the metric moved too, nothing here would be a contrast with
the arms it is supposed to be compared against.

TWO CONTRASTS, AND THEY ARE NOT THE SAME QUESTION.

  top - bot   the paper's own comparison, and the largest effect available,
              because it spans the full range of the score. It is also the
              easiest to over-read: two slices of one pool differ in more than
              their MDCL, and the bottom slice is not a neutral baseline.
  top - rand  the one that matters operationally. `rand` is what you get by not
              ranking at all, so this is the value added by the score. It is
              attenuated on purpose -- `rand` is drawn from the whole pool and
              so shares about a third of its rows with `top` -- which makes it
              the conservative of the two.

WHICH EPOCH. The headline is epoch 10, the paper's training budget, chosen
before any of this was run. The peak over the ten epochs is printed beside it
and labelled as what it is: a maximum over ten looks, worth about 2pp of
optimism on a curve this noisy, and not a number to quote on its own. Stage C
found fox still rising at epoch 9, so for that arm the two mostly agree.

THE IDENTITY CHECK. At epoch 0 the probe runs before any optimizer step, so
every arm is the same untrained adapter and every delta must be exactly zero. It
is printed rather than assumed: a nonzero value there means two runs did not
start from the same weights and nothing downstream is a contrast.

Usage: python mdcl_report.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_crossover import BLUE, CLAY, FIGS, GRID, INK, SLATE, mean_ci, style
from plot_epoch_curve import by_question, epochs_of, median, runs
from sl_common import MDCL_DIR, NATIVE_ANIMALS, animal_of, read_scores

NEUTRAL = "ref-control"
SPLITS = ("top", "bot", "rand")
SPLIT_COLOR = {"top": CLAY, "bot": BLUE, "rand": SLATE}
SPLIT_LABEL = {"top": "highest MDCL", "bot": "lowest MDCL",
               "rand": "unranked draw"}
# The budget the paper trains to, fixed before the runs.
HEADLINE_EPOCH = 10


def probe_of(cond: str) -> str:
    """Which of the probe's three question sets this arm is scored on.

    Same routing as `sl_common.choice_questions_for`, expressed as the key
    `animal_probe` filed the rows under.
    """
    return "native" if animal_of(cond) in NATIVE_ANIMALS else "choice"


def pooled(rs, a_cond, b_cond, epoch, target):
    """(seed, question)-paired differences in the target's share, a minus b.

    Both arms are probed on `a_cond`'s question set. That is right even when
    b_cond is `ref-control`, which has no field of its own and is probed on all
    three regardless -- see `train_student.animal_probe`.
    """
    probe = probe_of(a_cond)
    seeds = sorted({s for (c, s) in rs if c == a_cond} &
                   {s for (c, s) in rs if c == b_cond})
    out = []
    for s in seeds:
        a = by_question(rs[(a_cond, s)], epoch, probe, target)
        b = by_question(rs[(b_cond, s)], epoch, probe, target)
        if not a or not b:
            continue
        out += [a[q] - b[q] for q in sorted(set(a) & set(b))]
    return out


def common_epochs(rs, conds):
    """Epochs every one of `conds` reached, at every seed it was run at."""
    sets = [set(epochs_of(rec)) for (c, _), rec in rs.items() if c in conds]
    return sorted(set.intersection(*sets)) if sets else []


def curve(rs, cond, target):
    """Transmission against the neutral arm, by epoch: xs, mean, halfwidth, median."""
    xs, ys, hs, ms = [], [], [], []
    for ep in common_epochs(rs, {cond, NEUTRAL}):
        d = pooled(rs, cond, NEUTRAL, ep, target)
        if len(d) < 2:
            continue
        m, h = mean_ci(d)
        xs.append(ep)
        ys.append(100 * m)
        hs.append(100 * h)
        ms.append(100 * median(d))
    return xs, ys, hs, ms


def at_epoch(rs, a_cond, b_cond, target, epoch):
    """(mean pp, halfwidth pp, n) for a minus b, or None if either is missing."""
    d = pooled(rs, a_cond, b_cond, epoch, target)
    if len(d) < 2:
        return None
    m, h = mean_ci(d)
    return 100 * m, 100 * h, len(d)


def verdict(v) -> str:
    """How to read a contrast, on its interval alone."""
    if v is None:
        return "not measured"
    m, h, _ = v
    if m - h > 0:
        return "selection helps"
    if m + h < 0:
        return "selection HURTS"
    return "not distinguishable"


def animals_present(rs):
    """Animals that have at least one trained MDCL split."""
    out = []
    for c, _ in rs:
        if c.startswith("mdcl-") and animal_of(c) not in out:
            out.append(animal_of(c))
    return sorted(out)


def pool_scores(animal):
    """The pool's raw MDCL values and its split summary, or (None, None)."""
    pool = f"ref-{animal}-pool"
    summary = MDCL_DIR / f"{pool}-splits.json"
    if not summary.exists():
        return None, None
    s = json.loads(summary.read_text())
    vals = [r[s["score"]] for r in read_scores(pool).values()]
    return vals, s


def confounds(animal):
    """What else the ranking cut on, from mdcl_confounds.py, or None.

    Surfaced here rather than left in its own file, because a top-vs-bot number
    and the reason it might not mean what it looks like belong on the same page.
    """
    p = MDCL_DIR / f"ref-{animal}-pool-confounds.json"
    return json.loads(p.read_text()) if p.exists() else None


def score_panel(ax, animals):
    """What the score actually separated, before any student was trained."""
    colors = [CLAY, "#6E9E7A", BLUE, "#C9A227"]
    drawn = False
    for animal, color in zip(animals, colors):
        vals, s = pool_scores(animal)
        if not vals:
            continue
        drawn = True
        ax.hist(vals, bins=80, histtype="step", color=color, linewidth=1.6,
                label=f"{animal} pool ({len(vals)} rows)")
        # The two cuts, so the width of the gap the students were trained across
        # is visible rather than inferred from a table.
        for x in (s["cut_low"], s["cut_high"]):
            ax.axvline(x, color=color, linewidth=1.0, linestyle=(0, (3, 2)),
                       alpha=0.8)
    if not drawn:
        ax.text(0.5, 0.5, "no scored pools yet", ha="center", va="center",
                transform=ax.transAxes, color=SLATE)
    ax.set_xlabel("MDCL (nats per response token)")
    ax.set_ylabel("rows")
    ax.set_title("What the score separated", fontsize=10.5, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    style(ax)


def transmission_panel(ax, rs, animal):
    for sp in SPLITS:
        cond = f"mdcl-{animal}-{sp}"
        xs, ys, hs, ms = curve(rs, cond, animal)
        xs, ys, hs, ms = ([v for x, v in zip(xs, col) if x > 0]
                          for col in (xs, ys, hs, ms))
        if not xs:
            continue
        color = SPLIT_COLOR[sp]
        ax.fill_between(xs, [y - h for y, h in zip(ys, hs)],
                        [y + h for y, h in zip(ys, hs)], color=color, alpha=0.13,
                        linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=1.9, marker="o", markersize=3.5,
                label=f"{sp} ({SPLIT_LABEL[sp]})")
        ax.plot(xs, ms, color=color, linewidth=1.1, linestyle=(0, (3, 2)),
                alpha=0.7)
    # The Stage C arm on the same axes: the same teacher, unranked, on the pool
    # this one was cut from before it was extended. Not one of the three splits,
    # so it is drawn thin and unfilled.
    xs, ys, *_ = curve(rs, f"ref-{animal}", animal)
    xs, ys = zip(*[(x, y) for x, y in zip(xs, ys) if x > 0]) if xs else ((), ())
    if xs:
        ax.plot(xs, ys, color=INK, linewidth=1.2, alpha=0.55,
                label=f"ref-{animal} (stage C, unranked)")
    ax.axhline(0, color=GRID, linewidth=1.2, zorder=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("target's share of the field, minus neutral (pp)")
    ax.set_title(f"{animal}: transmission by MDCL slice", fontsize=10.5,
                 loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)


def report(rs, animals):
    for animal in animals:
        vals, s = pool_scores(animal)
        print(f"\n{animal}")
        if s:
            by = {d["split"]: d for d in s["splits"]}
            print(f"  pool {s['scored_rows']} scored of {s['pool_rows']}, "
                  f"ranked on {s['score']}, "
                  f"spearman vs the other denominator "
                  f"{s['spearman_mdcl_vs_neutral']:+.3f}")
            for sp in SPLITS:
                if sp in by:
                    print(f"    {sp:5s} MDCL {by[sp]['mean']:+.4f} "
                          f"+-{by[sp]['sd']:.4f}  "
                          f"[{by[sp]['min']:+.4f}, {by[sp]['max']:+.4f}]")
            print(f"    cuts: top above {s['cut_high']:+.4f}, "
                  f"bottom below {s['cut_low']:+.4f}; rand shares "
                  f"{s['overlap']['rand_top']} rows with top, "
                  f"{s['overlap']['rand_bot']} with bot")

        for sp in SPLITS:
            cond = f"mdcl-{animal}-{sp}"
            xs, ys, hs, ms = curve(rs, cond, animal)
            if not xs:
                print(f"  {sp:5s} not trained yet")
                continue
            zero = next((ys[i] for i in range(len(xs)) if xs[i] == 0), None)
            keep = [i for i in range(len(xs)) if xs[i] > 0]
            if not keep:
                continue
            best = max(keep, key=lambda i: ys[i])
            last = keep[-1]
            z = "" if zero is None else f", epoch 0 {zero:+.2f}pp"
            print(f"  {sp:5s} epoch {xs[last]:<2d} {ys[last]:+6.2f}pp "
                  f"+-{hs[last]:.2f} (median {ms[last]:+.2f}); "
                  f"peak {ys[best]:+6.2f}pp at epoch {xs[best]}{z}")

        # The two contrasts, at the budget the paper trains to.
        eps = common_epochs(rs, {f"mdcl-{animal}-{sp}" for sp in SPLITS})
        ep = HEADLINE_EPOCH if HEADLINE_EPOCH in eps else (eps[-1] if eps else None)
        if ep is None:
            continue
        if ep != HEADLINE_EPOCH:
            print(f"  (contrasts at epoch {ep}: not every split reached "
                  f"{HEADLINE_EPOCH})")
        for a_sp, b_sp in (("top", "bot"), ("top", "rand")):
            v = at_epoch(rs, f"mdcl-{animal}-{a_sp}", f"mdcl-{animal}-{b_sp}",
                         animal, ep)
            if v is None:
                print(f"  {a_sp}-{b_sp}: not measured")
                continue
            m, h, n = v
            print(f"  {a_sp}-{b_sp} at epoch {ep}: {m:+6.2f}pp +-{h:.2f} "
                  f"(n={n})  <- {verdict(v)}")

        # The caveat next to the number it qualifies, not in a separate file.
        c = confounds(animal)
        if c is None:
            print("  (no confound check on record -- run mdcl_confounds.py)")
        elif c["warnings"]:
            d = c["flags"]["degenerate"]
            print(f"  the ranking also separated: "
                  f"{'; '.join(c['warnings'][:3])}"
                  f"{' ...' if len(c['warnings']) > 3 else ''}")
            print(f"    degenerate rows: {d['top']:.0%} of top, "
                  f"{d['bot']:.0%} of bot, {d['rand']:.0%} of rand "
                  f"({d['pool']:.0%} of the pool)")
        else:
            print("  confound check clean: the slices differ in MDCL and "
                  "little else")


def main() -> None:
    rs = runs()
    animals = animals_present(rs)
    if not animals:
        print("no MDCL split arms trained yet -- run_mdcl.sh")
        # Still worth plotting the score panel: the pools may be scored and the
        # split it cut is a result in itself, before any student exists.
        animals = sorted({a for a in ("fox", "horse", "owl", "eagle", "dog", "cat")
                          if (MDCL_DIR / f"ref-{a}-pool-splits.json").exists()})
        if not animals:
            return
    report(rs, animals)

    FIGS.mkdir(parents=True, exist_ok=True)
    n = 1 + len(animals)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n + 1.4, 5.2))
    axes = [axes] if n == 1 else list(axes)
    fig.subplots_adjust(left=0.055 / (n / 3), right=0.985, top=0.83, bottom=0.115,
                        wspace=0.26)
    score_panel(axes[0], animals)
    for ax, animal in zip(axes[1:], animals):
        transmission_panel(ax, rs, animal)
    fig.text(0.02, 0.955,
             "Selecting the teacher's rows by MDCL, and what the student learned",
             fontsize=14, fontweight="bold", va="top", ha="left", color=INK)
    fig.savefig(FIGS / "mdcl_splits.png", dpi=200)
    print(f"\nwrote {FIGS / 'mdcl_splits.png'}")


if __name__ == "__main__":
    main()
