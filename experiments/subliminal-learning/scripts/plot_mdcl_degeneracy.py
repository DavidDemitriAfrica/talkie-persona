"""Where MDCL puts the rows that carry nothing: figures/mdcl_degeneracy.png.

The one figure Stage D has before any student is trained, and the one that
decided which score the splits are cut on. Reads
`runs/mdcl/<pool>-degeneracy.json` from `mdcl_probe_degeneracy.py`.

Three panels, sharing a y-axis so the middle one is read against the left:

  mdcl            the paper's denominator, no system prompt. Echoes sit at the
                  top of the ranking, well above real rows.
  mdcl_neutral    the same rows against this directory's neutral persona. The
                  echo advantage is gone.
  lp_cond         what the subtraction was hiding: degenerate rows are far
                  easier to predict in absolute terms, on both scores. That is
                  the thing MDCL is supposed to normalise away, and on
                  `mdcl_neutral` it does.

Points are per row, and every degenerate row is drawn against a clean row with
the identical response token count, so nothing here is a length effect. The
horizontal bars are the group mean with a 95% interval.

Usage: python plot_mdcl_degeneracy.py [pool ...]
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_crossover import BLUE, CLAY, FIGS, GRID, INK, SLATE, style
from sl_common import MDCL_DIR

# echo and count are the two halves of "degenerate" and they behave differently,
# which is most of the finding, so they never share a colour.
KINDS = ("echo", "count", "clean")
KIND_COLOR = {"echo": CLAY, "count": "#B7B5B3", "clean": BLUE}
KIND_LABEL = {"echo": "echo\n(restates the seeds)", "count": "count\n(n, n+1, n+2)",
              "clean": "clean\n(neither)"}
METRICS = ("mdcl", "mdcl_neutral", "lp_cond")
METRIC_TITLE = {
    "mdcl": "$\\it{mdcl}$ — vs no system prompt\n(the paper's denominator)",
    "mdcl_neutral": "$\\it{mdcl\\_neutral}$ — vs the neutral persona\n(what the splits are cut on)",
    "lp_cond": "$\\it{lp\\_cond}$ — the numerator alone\n(how predictable the row is)",
}


def load(pool):
    p = MDCL_DIR / f"{pool}-degeneracy.json"
    if not p.exists():
        raise SystemExit(f"no {p} -- run mdcl_probe_degeneracy.py {pool}")
    return json.loads(p.read_text())


def panel(ax, rows, metric, first):
    """One metric: per-row points by kind, with each group's mean and interval."""
    rng = np.random.default_rng(4)          # jitter only; nothing statistical
    for k, kind in enumerate(KINDS):
        v = np.array([r[metric] for r in rows if r["kind"] == kind])
        if not len(v):
            continue
        ax.scatter(k + rng.uniform(-0.17, 0.17, len(v)), v, s=17,
                   color=KIND_COLOR[kind], alpha=0.55, linewidths=0, zorder=3)
        m = v.mean()
        h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        ax.plot([k - 0.31, k + 0.31], [m, m], color=INK, lw=2.1, zorder=5,
                solid_capstyle="butt")
        ax.plot([k, k], [m - h, m + h], color=INK, lw=1.2, zorder=5)
        ax.annotate(f"{m:+.2f}", (k + 0.34, m), fontsize=9, color=INK,
                    va="center", zorder=6)

    ax.axhline(0, color=GRID, lw=1, zorder=1)
    ax.set_xticks(range(len(KINDS)))
    ax.set_xticklabels([KIND_LABEL[k] for k in KINDS], fontsize=9)
    ax.set_xlim(-0.6, len(KINDS) - 0.25)
    ax.set_title(METRIC_TITLE[metric], fontsize=10, color=INK, pad=10)
    if first:
        ax.set_ylabel("nats per response token", fontsize=10)
    style(ax)


def caption(fig, d):
    s = d["summary"]
    def gap(metric):
        p = s.get(f"{metric}/paired_degen_minus_clean") or {}
        return f"{p.get('mean', float('nan')):+.3f} ±{p.get('ci', float('nan')):.3f}"
    fig.text(
        0.5, 0.030,
        f"{d['pool']}: {d['pairs']} (degenerate, clean) row pairs matched on exact "
        f"response token count; the pool is {d['degenerate_rate']:.1%} degenerate.",
        ha="center", fontsize=8.5, color=SLATE)
    fig.text(
        0.5, 0.008,
        f"paired degenerate − clean:   mdcl {gap('mdcl')}      "
        f"mdcl_neutral {gap('mdcl_neutral')}      lp_cond {gap('lp_cond')}",
        ha="center", fontsize=8.5, color=SLATE)


def figure(pools):
    ds = [load(p) for p in pools]
    n = len(METRICS)
    fig, axes = plt.subplots(len(ds), n, figsize=(4.4 * n, 4.6 * len(ds)),
                             sharey="row", squeeze=False)
    for row, d in enumerate(ds):
        for col, metric in enumerate(METRICS):
            panel(axes[row][col], d["rows"], metric, col == 0)
        axes[row][0].annotate(
            d["pool"], (0, 1.0), xycoords="axes fraction", xytext=(0, 34),
            textcoords="offset points", fontsize=11, color=INK, weight="bold")
    fig.suptitle("MDCL ranks echoes to the top — unless the denominator already "
                 "has a system prompt", fontsize=12.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.055 if len(ds) == 1 else 0.01, 1, 0.94))
    if len(ds) == 1:
        caption(fig, ds[0])
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "mdcl_degeneracy.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main():
    pools = sys.argv[1:] or [p.name[: -len("-degeneracy.json")]
                             for p in sorted(MDCL_DIR.glob("*-degeneracy.json"))]
    if not pools:
        raise SystemExit("no *-degeneracy.json in runs/mdcl -- run "
                         "mdcl_probe_degeneracy.py first")
    figure(pools)


if __name__ == "__main__":
    main()
