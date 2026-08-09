"""Stage 3 analysis: what does Talkie value? Fit a Case-V utility over a
value-content probe's outcomes -- exactly the Stage-1 machinery, reused as a
library -- and read the *content* of the ordering off the fitted utilities.

    python analyze_values.py --probe lives   # worth-of-life by nationality
    python analyze_values.py --probe time    # temporal discounting

For `lives` the utility is an implicit worth-of-life ranking; we also report the
instrument health (completeness, cycle rate, position bias) so the ranking can be
read with the right amount of trust, and dump the pairwise matrix for the plot.
For `time` we align the fitted utility against each outcome's nominal delay to see
whether value decays with delay and how graded that decay is.

Writes runs/values_<probe>_summary.json and runs/values_<probe>.json (sorted
utilities + metadata) for plot_values.py.
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

# reuse the Stage-1 fit / cycle / Phi verbatim -- same estimator, new domain
from analyze import Phi, cycle_rate, fit_thurstone_caseV
from ue_common import DATA, RUNS, load_outcomes

PROBES = {
    "lives": ("outcomes_lives.json", "pairs_lives.*.jsonl", "values_lives"),
    "time": ("outcomes_time.json", "pairs_time.*.jsonl", "values_time"),
}


def load_pairs(pattern):
    rows = []
    for path in sorted(glob.glob(str(RUNS / pattern))):
        rows += [json.loads(l) for l in open(path) if l.strip()]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=list(PROBES), required=True)
    args = ap.parse_args()
    outcomes_file, pairs_glob, stem = PROBES[args.probe]

    ocpath = DATA / outcomes_file
    obj = json.loads(ocpath.read_text())
    ids, text, band, key = load_outcomes(ocpath)
    pos = {i: k for k, i in enumerate(ids)}
    extra = {o["id"]: o for o in obj["outcomes"]}   # for delay_years etc.

    rows = load_pairs(pairs_glob)
    print(f"[{args.probe}] loaded {len(rows)} pairs over {len(ids)} outcomes")

    # instrument health
    dec = np.array([r["decisive"] / r["total"] for r in rows])
    completeness = float(dec.mean())
    fully_abstained = int(sum(1 for r in rows if r["decisive"] == 0))
    fs = [(r["first_slot_rate"], r["decisive"]) for r in rows
          if r["first_slot_rate"] is not None]
    position_bias = (float(sum(v * w for v, w in fs) / sum(w for _, w in fs))
                     if fs else float("nan"))

    # majority winner per decided pair -> cycle rate
    winner = {}
    for r in rows:
        if r["n_a"] == r["n_b"]:
            continue
        winner[(r["a"], r["b"])] = r["a"] if r["n_a"] > r["n_b"] else r["b"]
    crate, n_tri, n_cyc = cycle_rate(winner)

    # Case-V utility on decisive pairs
    dpairs = [r for r in rows if r["decisive"] > 0]
    idx = np.array([[pos[r["a"]], pos[r["b"]]] for r in dpairs])
    na = np.array([r["n_a"] for r in dpairs])
    nb = np.array([r["n_b"] for r in dpairs])
    u = fit_thurstone_caseV(idx, na, nb, len(ids))

    d = u[idx[:, 0]] - u[idx[:, 1]]
    p_pred = np.clip(Phi(d), 1e-9, 1 - 1e-9)
    p_obs = na / (na + nb)
    r_fit = float(np.corrcoef(p_pred, p_obs)[0, 1])
    mask = na != nb
    acc = float(np.mean((p_pred[mask] > 0.5) == (p_obs[mask] > 0.5))) if mask.any() else float("nan")

    ranked = sorted(ids, key=lambda i: -u[pos[i]])
    util_list = [{
        "id": i, "u": float(u[pos[i]]), "band": band[i], "text": text[i],
        **({"delay_years": extra[i]["delay_years"]} if "delay_years" in extra[i] else {}),
    } for i in ranked]

    summary = {
        "probe": args.probe,
        "domain": obj["domain"],
        "n_outcomes": len(ids),
        "n_pairs": len(rows),
        "samples_per_pair": rows[0]["total"] if rows else None,
        "completeness_mean_decisive": completeness,
        "fully_abstained_pairs": fully_abstained,
        "position_bias_first_slot": position_bias,
        "cycle_rate": crate,
        "cycle_rate_random_baseline": 0.25,
        "triples_all_decided": n_tri,
        "triples_cyclic": n_cyc,
        "thurstone_pred_accuracy": acc,
        "thurstone_pred_obs_corr": r_fit,
        "ranking": [(x["id"], round(x["u"], 3)) for x in util_list],
    }

    if args.probe == "time":
        # present-bias / discount reading: utility vs nominal delay
        dy = np.array([extra[i]["delay_years"] for i in ids])
        uu = np.array([u[pos[i]] for i in ids])
        order = np.argsort(dy)
        summary["utility_by_delay"] = [
            {"id": ids[o], "delay_years": float(dy[o]), "u": float(uu[o])} for o in order
        ]
        # monotone-decreasing? Spearman(delay, u) should be strongly negative
        if len(dy) > 1 and np.ptp(dy) > 0:
            ru = np.argsort(np.argsort(uu)).astype(float)
            rd = np.argsort(np.argsort(dy)).astype(float)
            summary["delay_utility_spearman"] = float(np.corrcoef(rd, ru)[0, 1])
        # the sharp fact: today vs the mean of the delayed options
        u_today = float(uu[dy == 0][0]) if (dy == 0).any() else None
        u_delayed = float(uu[dy > 0].mean()) if (dy > 0).any() else None
        summary["u_today"] = u_today
        summary["u_delayed_mean"] = u_delayed
        summary["present_bias_gap"] = (u_today - u_delayed
                                       if u_today is not None and u_delayed is not None else None)

    (RUNS / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    (RUNS / f"{stem}.json").write_text(json.dumps(util_list, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote runs/{stem}_summary.json and runs/{stem}.json")


if __name__ == "__main__":
    main()
