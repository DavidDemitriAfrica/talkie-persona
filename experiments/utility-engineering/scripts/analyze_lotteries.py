"""Stage 2 analysis: does Talkie obey expected utility over lotteries?

Reads the sharded lottery JSONL and asks whether the Stage-1 fitted utilities --
used here as a *fixed predictor*, nothing re-fit -- explain the model's choices
between a certain good and a gamble. Four questions, in increasing sharpness:

  completeness  -- does the model decide, or abstain? (instrument health, and
                   position bias, exactly as Stage 1)
  monotonicity  -- does P(choose the gamble) rise as the gamble's winning chance
                   rises? A model that ignores the stated odds is flat; EU (and
                   any coherent risk attitude) is increasing.
  EU sign       -- does the model take the gamble exactly when its expected
                   utility beats the certain good, i.e. when EU_gamble > u_c?
                   Accuracy over items whose EU margin is not a near-tie.
  calibration   -- treat the gamble as a virtual outcome with utility EU_gamble
                   and predict P(gamble) = Phi(EU_gamble - u_c) from the Stage-1
                   Case-V utilities. Correlate with observed. And the money plot:
                   the empirical indifference chance p-hat* vs the EU-predicted
                   p* = (u_c - u_y)/(u_x - u_y). Signed bias p-hat* - p* reads as
                   risk attitude (negative = takes the gamble too readily =
                   overweights the small chance of the good outcome).

Writes runs/eu_summary.json and runs/lottery_cells.json (per-cell curves + CEs)
for the plotter.
"""

from __future__ import annotations

import glob
import json
import math

import numpy as np

from ue_common import RUNS

SQRT2 = math.sqrt(2.0)
_erf = np.vectorize(math.erf)


def Phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / SQRT2))


def load_rows(pattern="lotteries.*.jsonl"):
    rows = []
    for path in sorted(glob.glob(str(RUNS / pattern))):
        rows += [json.loads(l) for l in open(path) if l.strip()]
    return rows


def spearman(x, y):
    """Rank correlation without scipy; ties broken by average rank."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[order] = np.arange(len(a), dtype=float)
        # average tied ranks
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def cell_ce(pps, plot):
    """Empirical indifference chance (in [0,1]) where P(gamble) crosses 0.5.

    Robust to sampling noise: a least-squares line through (pp, P_gamble) solved
    for P=0.5. Returns None if the line is essentially flat (no dependence on the
    odds, so no crossing is meaningful). Clamped to [0,1].
    """
    pp = np.asarray(pps, float) / 100.0
    y = np.asarray(plot, float)
    if len(pp) < 2 or np.all(pp == pp[0]):
        return None
    A = np.polyfit(pp, y, 1)  # slope, intercept
    slope, intercept = A[0], A[1]
    if abs(slope) < 1e-6:
        return None
    ce = (0.5 - intercept) / slope
    return float(min(1.0, max(0.0, ce)))


def main() -> None:
    rows = load_rows()
    print(f"loaded {len(rows)} lottery items")
    if not rows:
        print("no lottery rows found -- run elicit_lotteries.py first")
        return

    dec = np.array([r["decisive"] for r in rows], float)
    tot = np.array([r["total"] for r in rows], float)
    completeness = float(np.mean(dec / tot))
    fully_abstained = int(np.sum(dec == 0))
    fs = [(r["first_slot_rate"], r["decisive"]) for r in rows
          if r["first_slot_rate"] is not None]
    position_bias = (float(sum(v * w for v, w in fs) / sum(w for _, w in fs))
                     if fs else float("nan"))

    # decisive items only for the EU tests
    d = [r for r in rows if r["decisive"] > 0 and r["p_lottery"] is not None]
    obs = np.array([r["p_lottery"] for r in d])
    eu_margin = np.array([r["eu_minus_uc"] for r in d])
    pred = Phi(eu_margin)                       # Case-V, no refit

    # (2) monotonicity: per (base,c) cell, does P(gamble) rise with the chance pp?
    cells = {}
    for r in d:
        cells.setdefault((r["x"], r["y"], r["c"]), []).append(r)
    cell_out = []
    mono_corrs, mono_pos = [], 0
    ce_emp, ce_pred = [], []
    for (x, y, c), items in sorted(cells.items()):
        items = sorted(items, key=lambda r: r["pp"])
        pps = [it["pp"] for it in items]
        pl = [it["p_lottery"] for it in items]
        sc = spearman(pps, pl)
        if not math.isnan(sc):
            mono_corrs.append(sc)
            mono_pos += int(sc > 0)
        ce = cell_ce(pps, pl)
        pstar = items[0]["p_star"]
        if ce is not None:
            ce_emp.append(ce)
            ce_pred.append(pstar)
        cell_out.append({
            "x": x, "y": y, "c": c, "u_x": items[0]["u_x"], "u_y": items[0]["u_y"],
            "u_c": items[0]["u_c"], "p_star": pstar, "ce_emp": ce,
            "pp": pps, "p_lottery": pl,
            "decisive": [it["decisive"] for it in items],
        })

    # (3) EU sign accuracy (drop near-ties in the EU margin)
    MARG = 0.10
    keep = np.abs(eu_margin) > MARG
    take_lottery = obs > 0.5
    eu_says_lottery = eu_margin > 0
    sign_acc = float(np.mean(take_lottery[keep] == eu_says_lottery[keep])) if keep.any() else float("nan")

    # (4a) calibration of Phi(EU - u_c) against observed P(gamble)
    cal_corr = float(np.corrcoef(pred, obs)[0, 1])
    cal_mae = float(np.mean(np.abs(pred - obs)))
    # best single risk-scale s: observed ~ Phi(s * (EU - u_c)); s<1 = noisier/less
    # EU-sensitive than pairwise choices, s>1 = sharper. Grid search (no scipy).
    ss = np.linspace(0.1, 3.0, 291)
    errs = [np.mean((Phi(s * eu_margin) - obs) ** 2) for s in ss]
    s_best = float(ss[int(np.argmin(errs))])
    cal_mae_scaled = float(np.sqrt(min(errs)))

    # (4b) certainty-equivalent agreement
    ce_emp_a, ce_pred_a = np.array(ce_emp), np.array(ce_pred)
    ce_corr = float(np.corrcoef(ce_emp_a, ce_pred_a)[0, 1]) if len(ce_emp) > 1 else None
    ce_mae = float(np.mean(np.abs(ce_emp_a - ce_pred_a))) if ce_emp else None
    ce_bias = float(np.mean(ce_emp_a - ce_pred_a)) if ce_emp else None

    summary = {
        "n_items": len(rows),
        "n_cells": len(cells),
        "p_grid": sorted({r["pp"] for r in rows}),
        "samples_per_item": int(rows[0]["total"]),
        "completeness_mean_decisive": completeness,
        "fully_abstained_items": fully_abstained,
        "position_bias_first_slot": position_bias,
        "monotonicity_mean_spearman": float(np.mean(mono_corrs)) if mono_corrs else None,
        "monotonicity_frac_increasing": (mono_pos / len(mono_corrs)) if mono_corrs else None,
        "eu_sign_accuracy": sign_acc,
        "eu_sign_margin": MARG,
        "eu_sign_n_scored": int(keep.sum()),
        "calibration_corr": cal_corr,
        "calibration_mae": cal_mae,
        "risk_scale_best": s_best,
        "calibration_mae_scaled": cal_mae_scaled,
        "ce_corr": ce_corr,
        "ce_mae": ce_mae,
        "ce_bias_emp_minus_pred": ce_bias,
        "ce_n_cells": len(ce_emp),
    }
    (RUNS / "eu_summary.json").write_text(json.dumps(summary, indent=2))
    (RUNS / "lottery_cells.json").write_text(json.dumps(cell_out, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nwrote runs/eu_summary.json and runs/lottery_cells.json")


if __name__ == "__main__":
    main()
