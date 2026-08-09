"""Stage 4 analysis: how far, and how cleanly, does a prompt steer the utility?

For each (target, condition) we refit the target's Case-V utility against the fixed
Stage-1 reference panel -- a 1-D MLE, u_ref held constant -- so the steered value is
on the same scale as the Stage-1 utility. Reads off:

  * controllability: delta_pro = u_pro - u_neutral (want > 0),
    delta_anti = u_anti - u_neutral (want < 0), and the pro-anti range;
  * validity: |u_neutral - u_stage1|, i.e. does the bare re-elicitation reproduce
    Stage 1 (it should);
  * spillover: do the reference-vs-reference canaries drift when the prefix is about
    an unrelated target? Clean valence control leaves them put; mention-priming or a
    global mood shift moves them.

Writes runs/steer_summary.json.
"""

from __future__ import annotations

import glob
import json

import numpy as np

from analyze import Phi
from ue_common import DATA, RUNS

GRID = np.arange(-4.0, 4.0001, 0.01)


def mle_u_target(refs, u_ref, na, nb):
    """1-D MLE of u_t from target-vs-ref counts, u_ref fixed. na = target wins."""
    ur = np.array([u_ref[r] for r in refs])
    na = np.asarray(na, float)
    nb = np.asarray(nb, float)
    best_u, best_ll = 0.0, -np.inf
    for ut in GRID:
        p = np.clip(Phi(ut - ur), 1e-9, 1 - 1e-9)
        ll = float(np.sum(na * np.log(p) + nb * np.log(1 - p)))
        if ll > best_ll:
            best_ll, best_u = ll, float(ut)
    return best_u


def main() -> None:
    spec = json.loads((DATA / "steer_targets.json").read_text())
    util = json.loads((RUNS / "utilities.json").read_text())
    u_stage1 = {k: v["u"] for k, v in util.items()}

    rows = []
    for p in sorted(glob.glob(str(RUNS / "steer.*.jsonl"))):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    print(f"loaded {len(rows)} steer rows")

    def cell(target, cond, kind):
        return [r for r in rows if r["target"] == target
                and r["condition"] == cond and r["kind"] == kind]

    targets = []
    for tgt in spec["targets"]:
        tid = tgt["id"]
        us = {}
        comp = {}
        for cond in ("neutral", "pro", "anti"):
            tr = cell(tid, cond, "target")
            refs = [r["b"] for r in tr]
            na = [r["n_a"] for r in tr]     # a == target (build_jobs), so n_a = target wins
            nb = [r["n_b"] for r in tr]
            us[cond] = mle_u_target(refs, u_stage1, na, nb)
            dec = sum(na) + sum(nb)
            comp[cond] = dec / sum(r["total"] for r in tr) if tr else float("nan")
        targets.append({
            "id": tid,
            "u_stage1": u_stage1[tid],
            "u_neutral": us["neutral"], "u_pro": us["pro"], "u_anti": us["anti"],
            "delta_pro": us["pro"] - us["neutral"],
            "delta_anti": us["anti"] - us["neutral"],
            "range_pro_anti": us["pro"] - us["anti"],
            "dir_pro_ok": us["pro"] > us["neutral"],
            "dir_anti_ok": us["anti"] < us["neutral"],
            "completeness": comp,
        })

    # spillover: canary ref-vs-ref pairs. Baseline = their neutral-condition pref
    # (averaged over targets); drift = |pref_steered - baseline| under pro/anti.
    canary_keys = [tuple(c) for c in spec["canaries"]]
    base = {}   # (a,b) -> mean neutral pref of a over b
    for a, b in canary_keys:
        neu = [r["pref"] for r in rows if r["kind"] == "canary"
               and r["a"] == a and r["b"] == b and r["condition"] == "neutral"
               and r["pref"] is not None]
        base[(a, b)] = float(np.mean(neu)) if neu else float("nan")
    # baseline validity: neutral canary pref vs Stage-1 prediction Phi(u_a-u_b)
    base_pred_err = []
    for (a, b), pobs in base.items():
        base_pred_err.append(abs(pobs - Phi(u_stage1[a] - u_stage1[b])))
    drifts = []
    for r in rows:
        if r["kind"] != "canary" or r["condition"] == "neutral" or r["pref"] is None:
            continue
        b0 = base.get((r["a"], r["b"]))
        if b0 == b0:  # not nan
            drifts.append(abs(r["pref"] - b0))

    summary = {
        "domain": spec["domain"],
        "n_rows": len(rows),
        "targets": targets,
        "validity_neutral_vs_stage1_mae": float(np.mean(
            [abs(t["u_neutral"] - t["u_stage1"]) for t in targets])),
        "controllability_mean_range": float(np.mean([t["range_pro_anti"] for t in targets])),
        "dir_pro_ok_frac": float(np.mean([t["dir_pro_ok"] for t in targets])),
        "dir_anti_ok_frac": float(np.mean([t["dir_anti_ok"] for t in targets])),
        "canary_baseline_vs_stage1_mae": float(np.mean(base_pred_err)),
        "spillover_mean_canary_drift": float(np.mean(drifts)) if drifts else None,
    }
    (RUNS / "steer_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nwrote runs/steer_summary.json")


if __name__ == "__main__":
    main()
