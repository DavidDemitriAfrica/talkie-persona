"""Did ranking the teacher's rows by MDCL change what transmits?

Stage D's result. Three students per animal, trained identically on 10,000 rows
of the same teacher's numbers and differing only in *which* 10,000: the highest
MDCL, the lowest, and a uniform draw. If the covert-influence paper's claim
holds on Talkie, `top` transmits more than `bot`, and more than `rand`.

WHAT COUNTS AS TRANSMISSION is the Stage C instrument, deliberately: the target
animal's share of the twelve-candidate logit field on the arm's forced-choice
question set, differenced against `ref-control` at the same epoch and seed, and
paired on the question before pooling. Reusing it verbatim is the point -- if the
metric moved too, nothing here would be a contrast with the arms it is supposed
to be compared against.

BUT STAGE C FOUND THAT INSTRUMENT IS COMPOSITIONAL, which changes what may be
read off it. A share has a denominator, and at ten epochs mass floods into
`deer`, an animal in nobody's teacher prompt: 21.5% of the native field in the
neutral against 46.8% in the dog arm. An arm that parks 25 extra points there has
its target divided by a different number, so `verdict.py` voids the
share-vs-neutral column above SINK_TOL of excess sink -- not as a judgement about
the arm, but because its comparator is arithmetically invalid.

Stage D is mostly out of that line of fire, for a structural reason rather than a
lucky one: both headline numbers difference two *splits of the same teacher*,
trained on one recipe at one dose, so a sink the splits share cancels in the
subtraction. The per-split vs-neutral levels do not cancel, and are exposed
exactly as Stage C's arms were. Both get the same treatment here -- the sink is
measured per split, the levels are flagged when it exceeds tolerance, and the
spread across splits is printed so that "the contrasts are insulated" is a
measurement rather than an argument.

AND THE MEASUREMENT CAME BACK AGAINST THE ARGUMENT, so both contrasts are also
reported on the raw token probability, which has no denominator for a sink to
move. The structural case above is sound and it is not sufficient: `top` parks
13.1pp more of the field in `deer` than the neutral against `bot` and `rand` at
3.9pp each, a spread of 9.2pp, so the splits did *not* sink together and the
cancellation the identity promises does not happen. A share contrast between two
arms with different denominators is not a contrast. The `abs` column is what the
section's conclusion rests on; the `share` column is kept beside it because when
the two agree that agreement is worth more than either alone, and because hiding
the compromised instrument would hide how the conclusion was reached.

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
# Imported rather than restated: a tolerance that appears twice is a tolerance
# that will disagree with itself the first time one copy is tuned.
from verdict import SINK, SINK_TOL

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


def by_question_abs(rec, epoch, probe, target):
    """{question: the target token's raw probability} at one epoch.

    The unnormalised twin of `by_question`. No denominator, so nothing the rest
    of the field does can move it -- which is the entire reason it is here.
    """
    for e in rec["animals"]["epoch_log"]:
        if e["epoch"] == epoch:
            return {r["question"]: r["probs"][target] for r in e["probes"][probe]
                    if target in r["probs"]}
    return None


def pooled(rs, a_cond, b_cond, epoch, target, stat="share"):
    """(seed, question)-paired differences in the target statistic, a minus b.

    Both arms are probed on `a_cond`'s question set. That is right even when
    b_cond is `ref-control`, which has no field of its own and is probed on all
    three regardless -- see `train_student.animal_probe`.

    `stat` picks which quantity is differenced: "share" is the target's fraction
    of the twelve-candidate field, the Stage C instrument; "abs" is the raw token
    probability. The pairing and the pooling are identical either way, so the two
    intervals are comparable and any disagreement between them is about the
    denominator rather than about the estimator.
    """
    probe = probe_of(a_cond)
    read = by_question if stat == "share" else by_question_abs
    seeds = sorted({s for (c, s) in rs if c == a_cond} &
                   {s for (c, s) in rs if c == b_cond})
    out = []
    for s in seeds:
        a = read(rs[(a_cond, s)], epoch, probe, target)
        b = read(rs[(b_cond, s)], epoch, probe, target)
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


def at_epoch(rs, a_cond, b_cond, target, epoch, stat="share"):
    """(mean pp, halfwidth pp, n) for a minus b, or None if either is missing."""
    d = pooled(rs, a_cond, b_cond, epoch, target, stat)
    if len(d) < 2:
        return None
    m, h = mean_ci(d)
    return 100 * m, 100 * h, len(d)


def sink_excess(rs, cond, epoch):
    """How much more of the field this arm parks in `deer` than the neutral, pp.

    The same question-paired difference the transmission number uses, with the
    sink substituted for the target -- so a positive value means this arm's
    target is being divided by a bigger denominator than the neutral's, which is
    the whole reason the level is not comparable.

    None, not zero, when the arm's menu does not list `deer`: the owl question
    set does not, so there is nothing to correct and no measurement to report. A
    zero there would read as "checked and clean" for a check never run.
    """
    if probe_of(cond) != "native":
        return None
    d = pooled(rs, cond, NEUTRAL, epoch, SINK)
    return 100 * sum(d) / len(d) if d else None


def sink_note(sinks):
    """Lines on the sink: the levels it invalidates, and the spread it does not.

    Two separate questions, and conflating them is how Stage C got three arms
    wrong. A large sink invalidates a *level* against the neutral. Only a large
    *disagreement* between the splits invalidates a contrast between them, and
    the splits are the same teacher on the same recipe, so the expectation is
    that this is small. Expectation is not measurement, hence the second line.
    """
    have = {sp: v for sp, v in sinks.items() if v is not None}
    if not have:
        return []
    out = [f"  {SINK} sink vs neutral: "
           + ", ".join(f"{sp} {v:+.1f}pp" for sp, v in have.items())]
    hot = [sp for sp, v in have.items() if v > SINK_TOL]
    if hot:
        out.append(f"    vs-neutral levels VOID for {', '.join(hot)} "
                   f"(sink over {SINK_TOL:g}pp) -- read the contrasts, not the "
                   f"levels")
    spread = max(have.values()) - min(have.values())
    if len(have) < len(sinks):
        out.append(f"    spread {spread:.1f}pp over {len(have)} of "
                   f"{len(sinks)} splits -- incomplete, not yet conclusive")
    elif spread <= SINK_TOL:
        out.append(f"    spread across splits {spread:.1f}pp, within "
                   f"{SINK_TOL:g} -- the contrasts are insulated")
    else:
        out.append(f"    spread across splits {spread:.1f}pp, over {SINK_TOL:g} "
                   f"-- the splits sank differently and the contrasts are NOT "
                   f"insulated")
    return out


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

        # Before the contrasts, not after: whether the levels above are readable
        # and whether the numbers below are insulated is the first thing to know,
        # and Stage C is the cautionary tale for reporting it as a footnote.
        for line in sink_note({sp: sink_excess(rs, f"mdcl-{animal}-{sp}", ep)
                               for sp in SPLITS}):
            print(line)

        # Both statistics on every contrast, because which one is trustworthy is
        # not known until the sink is measured, and the sink is measured above.
        # When they agree the answer is robust to the denominator; when they
        # disagree the raw column is the one to believe, since it is the only one
        # a sink cannot reach.
        for a_sp, b_sp in (("top", "bot"), ("top", "rand")):
            for stat, unit in (("share", "pp of field"), ("abs", "pp raw")):
                v = at_epoch(rs, f"mdcl-{animal}-{a_sp}", f"mdcl-{animal}-{b_sp}",
                             animal, ep, stat)
                if v is None:
                    print(f"  {a_sp}-{b_sp} [{stat}]: not measured")
                    continue
                m, h, n = v
                print(f"  {a_sp}-{b_sp} at epoch {ep} [{stat:5s}]: "
                      f"{m:+6.2f} +-{h:.2f} {unit} (n={n})  <- {verdict(v)}")

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
