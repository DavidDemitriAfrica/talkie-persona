"""Is the crossover a property of the arms, or of one lucky training run?

Every between-arm number this experiment reports is a difference between two
*single* fine-tuning runs. LoRA init, batch order, and dropout are all resampled
per run, so some of that difference is optimization noise and nothing in the
one-run-per-arm design separates the two. The replicate arms (`owl_s2`,
`eagle_s2`, `control_s2` -- same teacher data, same recipe, different seed)
exist to put a number on it.

Two things are being asked, and they are not the same question:

  reproducibility  does an *arm* land in the same place twice? Reported as the
                   paired s2-minus-original shift for each arm. Small is good.
  replication      does the *contrast* come out the same twice? Reported as the
                   diagonal computed independently within each seed. This is the
                   one that matters -- an arm could drift and the contrast still
                   hold, since both arms drift together.

Note the original arms predate seeding, so their seed is unrecorded rather than
0. They are labelled "run 1" for that reason; this is two draws from the same
recipe, not seed 0 against seed 2.

Usage: python seed_variance.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_crossover import (BLUE, CLAY, FIGS, INK, logit_index, mean_ci,
                            paired, sampled_index, style)

# (original run, replicate) for each arm, in the reading order of the figure.
ARMS = [("owl", "owl_s2", "owl"),
        ("eagle", "eagle_s2", "eagle"),
        ("control", "control_s2", "neutral")]
# The contrasts to recompute inside each seed. Only r16 fixed-prompt arms have
# replicates, so this is the r16 block of the main table.
CONTRASTS = [("owl", "eagle", "owl vs eagle"),
             ("owl", "control", "owl vs neutral"),
             ("eagle", "control", "eagle vs neutral")]
RUN_LABEL = {"": "run 1", "_s2": "run 2 (s2)"}
SUFFIXES = ["", "_s2"]


def have(idx, conds):
    return all(idx.get(c) for c in conds)


def contrast_panel(ax, idx, title):
    """Each contrast recomputed within a seed. Same sign twice is the result."""
    rows = []
    for a, b, tag in CONTRASTS:
        for suf in SUFFIXES:
            ca, cb = a + suf, b + suf
            if not have(idx, (ca, cb)):
                continue
            d = paired(idx[ca], idx[cb])
            rows.append((f"{tag} · {RUN_LABEL[suf]}", d[0], d[1], suf))
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    ax.barh(y, vals, 0.62, xerr=errs,
            color=[CLAY if r[3] == "" else BLUE for r in rows],
            error_kw=dict(ecolor=INK, elinewidth=1.0, capthick=1.0), capsize=3)
    ax.axvline(0, color="#4A4A47", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("shift in owl-lean, paired over the 30 questions")
    ax.set_title(title, fontsize=10, loc="left", pad=8)
    ax.yaxis.grid(False)
    reach = max(abs(v) + e for v, e in zip(vals, errs))
    ax.set_xlim(-reach * 1.45, reach * 1.45)
    pad = reach * 0.05
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + (e + pad) * (1 if v > 0 else -1), yi,
                f"{v:+.2f}{'*' if abs(v) > e else ''}", va="center",
                ha="left" if v > 0 else "right", fontsize=8.5, color=INK)
    style(ax)
    return rows


def report(name, idx):
    print(f"\n=== {name}")
    print("  arm reproducibility (run 2 minus run 1, paired by question)")
    for a, b, tag in ARMS:
        if not have(idx, (a, b)):
            continue
        ma, _ = mean_ci(list(idx[a].values()))
        mb, _ = mean_ci(list(idx[b].values()))
        d, h = paired(idx[b], idx[a])
        print(f"    {tag:8s} {ma:+6.3f} -> {mb:+6.3f}   delta {d:+6.3f} +/- {h:.3f}"
              f"{'  *' if abs(d) > h else ''}")
    print("  contrast replication (recomputed inside each run)")
    out = {}
    for a, b, tag in CONTRASTS:
        for suf in SUFFIXES:
            if not have(idx, (a + suf, b + suf)):
                continue
            d, h = paired(idx[a + suf], idx[b + suf])
            out.setdefault(tag, []).append(d)
            print(f"    {tag:16s} {RUN_LABEL[suf]:12s} {d:+6.3f} +/- {h:.3f}"
                  f"  t={1.96*d/h:+5.2f}{'  *' if abs(d) > h else ''}")
    # The comparison the caveat was about: how the run-to-run spread in the
    # contrast compares to the contrast itself. A gap of the same size as the
    # effect would mean the effect is not separable from seed noise.
    for tag, ds in out.items():
        if len(ds) == 2:
            print(f"    {tag:16s} {'seed spread':12s} {abs(ds[0]-ds[1]):6.3f}"
                  f"  ({abs(ds[0]-ds[1]) / max(abs(sum(ds)/2), 1e-9):.0%} of the mean"
                  f" contrast, {sum(ds)/2:+.3f})")


def main() -> None:
    conds = [c + s for c, _, _ in ARMS for s in SUFFIXES]
    logit = {c: logit_index(c) for c in conds}
    samp = {c: sampled_index(c) for c in conds}
    missing = [c for c in conds if not logit[c] or not samp[c]]
    if missing:
        print("no eval output yet for: " + ", ".join(missing))
        return

    FIGS.mkdir(parents=True, exist_ok=True)
    # Contrasts only. Whether an arm lands in the same place twice is the
    # printed table below; whether the *contrast* does is the question, and it
    # is one row per contrast per run. The left margin is set explicitly by the
    # ~24-character row labels, which are the widest thing in the figure.
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 2, left=0.175, right=0.985, top=0.70, bottom=0.145,
                          wspace=0.50)
    contrast_panel(fig.add_subplot(gs[0, 0]), logit, "Exact probabilities")
    contrast_panel(fig.add_subplot(gs[0, 1]), samp, "Sampled choices")

    fig.text(0.05, 0.955, "The same three arms, trained twice",
             fontsize=13, fontweight="bold", va="top", ha="left", color=INK)
    fig.text(0.05, 0.865,
             "Two training seeds on identical teacher data. Each contrast is "
             "recomputed inside a run, so the pair has to agree for the result "
             "to be a property of the arms\nrather than of one lucky run.",
             fontsize=9.5, va="top", ha="left", color=INK)
    fig.savefig(FIGS / "seeds.png", dpi=200)
    print(f"wrote {FIGS / 'seeds.png'}")

    report("exact probabilities", logit)
    report("sampled choices", samp)


if __name__ == "__main__":
    main()
