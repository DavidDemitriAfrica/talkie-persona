"""Exercise mdcl_report against a synthetic set of runs and a synthetic pool.

Checks that it reads real-shaped files, that its pairing arithmetic gives the
identity (top-ctl) - (bot-ctl) == top-bot, that the epoch-0 identity check
shows zero, and that it survives a half-finished experiment.
"""
import json
import random
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="sl-reportdir-"))
SL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SL))

import matplotlib
matplotlib.use("Agg")

import sl_common
from sl_common import (ANIMAL_QUESTIONS, CANDIDATE_ANIMALS, CHOICE_QUESTIONS,
                       NATIVE_CHOICE_QUESTIONS)

RUNS = TMP / "runs"
FIGS = TMP / "figures"
MDCL = RUNS / "mdcl"
for d in (RUNS, FIGS, MDCL):
    d.mkdir(parents=True, exist_ok=True)

rng = random.Random(7)


def probes(target, lift):
    """One epoch's probe block: the target lifted by `lift` in probability."""
    out = {}
    for name, qs in (("plain", ANIMAL_QUESTIONS), ("choice", CHOICE_QUESTIONS),
                     ("native", NATIVE_CHOICE_QUESTIONS)):
        rows = []
        for q in qs:
            p = {a: 0.02 + 0.004 * rng.random() for a in CANDIDATE_ANIMALS}
            p[target] += lift + 0.002 * rng.random()
            rows.append({"question": q, "probs": p})
        out[name] = rows
    return out


def write_run(name, target, lifts):
    """lifts[e] is the target's added probability at epoch e."""
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    log = [{"epoch": e, "step": 100 * e, "probes": probes(target, lift)}
           for e, lift in enumerate(lifts)]
    (d / "epoch_animals.json").write_text(json.dumps({"epoch_log": log}))
    (d / "train_curve.json").write_text(json.dumps({"epoch_log": [
        {"epoch": e, "val_nll": 2.0 - 0.01 * e} for e in range(len(lifts))]}))


EPOCHS = 11

# The neutral arm. Its "target" is arbitrary -- every animal gets the same base
# rate -- but epoch 0 has to be identical to every other arm's epoch 0 for the
# identity check to mean anything, so the RNG is reset per run and lift 0 at e=0.
def lifts(peak):
    return [0.0] + [peak * e / 10 for e in range(1, EPOCHS)]


def build(fox_top=0.06, fox_bot=0.01, fox_rand=0.03, partial=False):
    for name, target, peak in (
        ("ref-control_10k_s1", "owl", 0.0),
        ("ref-fox_10k_s1", "fox", 0.03),
        ("mdcl-fox-top_10k_s1", "fox", fox_top),
        ("mdcl-fox-bot_10k_s1", "fox", fox_bot),
        ("mdcl-fox-rand_10k_s1", "fox", fox_rand),
        ("ref-horse_10k_s1", "horse", -0.01),
        ("mdcl-horse-top_10k_s1", "horse", 0.02),
        ("mdcl-horse-bot_10k_s1", "horse", -0.01),
        ("mdcl-horse-rand_10k_s1", "horse", 0.0),
    ):
        rng.seed(hash(name) % 10000)
        n = 5 if (partial and name == "mdcl-horse-rand_10k_s1") else EPOCHS
        write_run(name, target, lifts(peak)[:n])

    for animal in ("fox", "horse"):
        pool = f"ref-{animal}-pool"
        with open(MDCL / f"{pool}.jsonl", "w") as f:
            for i in range(30250):
                v = rng.gauss(0.05 if animal == "fox" else 0.02, 0.06)
                f.write(json.dumps({"i": i, "n": 20, "mdcl": v,
                                    "mdcl_neutral": v + rng.gauss(0, 0.01),
                                    "lp_cond": -1.5}) + "\n")
        vals = sorted((json.loads(l)["mdcl"] for l in open(MDCL / f"{pool}.jsonl")),
                      reverse=True)
        (MDCL / f"{pool}-splits.json").write_text(json.dumps({
            "pool": pool, "animal": animal, "score": "mdcl", "seed": 1930,
            "pool_rows": 30250, "scored_rows": 30250, "split_rows": 10250,
            "spearman_mdcl_vs_neutral": 0.98,
            "cut_high": vals[10249], "cut_low": vals[-10250],
            "overlap": {"top_bot": 0, "rand_top": 3400, "rand_bot": 3500},
            "splits": [{"split": s, "rows": 10250, "mean": m, "sd": 0.02,
                        "min": m - 0.1, "max": m + 0.1}
                       for s, m in (("top", 0.12), ("bot", -0.02), ("rand", 0.05))],
        }))
        # fox gets a confound record with a warning, horse gets none at all, so
        # both branches of the report's caveat line run.
        if animal == "fox":
            (MDCL / f"{pool}-confounds.json").write_text(json.dumps({
                "pool": pool, "score": "mdcl", "rows": 10250,
                "spearman_vs_covariate": {"response chars": 0.11},
                "flags": {"degenerate": {"pool": 0.53, "top": 0.21, "bot": 0.88,
                                         "rand": 0.52, "top_minus_bot": -0.67,
                                         "mean_mdcl_true": 0.01,
                                         "mean_mdcl_false": 0.09}},
                "categoricals": {},
                "warnings": ["degenerate rows are 21% of top and 88% of bot (-67%)"],
            }))


def patch():
    import plot_crossover, plot_epoch_curve, mdcl_report
    sl_common.RUNS = RUNS
    sl_common.MDCL_DIR = MDCL
    plot_crossover.RUNS = RUNS
    plot_crossover.FIGS = FIGS
    plot_epoch_curve.RUNS = RUNS
    mdcl_report.MDCL_DIR = MDCL
    mdcl_report.FIGS = FIGS
    return mdcl_report


fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


build()
mr = patch()
rs = mr.runs()

print("=== inventory ===")
check(("mdcl-fox-top", 1) in rs, "mdcl-fox-top_10k_s1 parsed as an arm")
check(mr.animals_present(rs) == ["fox", "horse"], "both animals found")
check(mr.probe_of("mdcl-fox-top") == "native", "fox routes to the native field")
check(mr.probe_of("mdcl-owl-top") == "choice", "owl would route to the choice field")

print("=== pairing arithmetic ===")
ep = 10
top_ctl = mr.pooled(rs, "mdcl-fox-top", "ref-control", ep, "fox")
bot_ctl = mr.pooled(rs, "mdcl-fox-bot", "ref-control", ep, "fox")
top_bot = mr.pooled(rs, "mdcl-fox-top", "mdcl-fox-bot", ep, "fox")
check(len(top_ctl) == len(NATIVE_CHOICE_QUESTIONS),
      f"one delta per question ({len(top_ctl)})")
worst = max(abs((a - b) - c) for a, b, c in zip(top_ctl, bot_ctl, top_bot))
check(worst < 1e-12, f"(top-ctl)-(bot-ctl) == top-bot elementwise (max {worst:.1e})")

print("=== identity check at epoch 0 ===")
z = mr.pooled(rs, "mdcl-fox-top", "ref-control", 0, "fox")
check(max(abs(v) for v in z) < 0.02,
      "epoch 0 deltas are near zero on identical synthetic weights")

print("=== direction ===")
v_tb = mr.at_epoch(rs, "mdcl-fox-top", "mdcl-fox-bot", "fox", 10)
v_tr = mr.at_epoch(rs, "mdcl-fox-top", "mdcl-fox-rand", "fox", 10)
check(v_tb[0] > 0 and mr.verdict(v_tb) == "selection helps",
      f"top>bot detected: {v_tb[0]:+.2f}pp +-{v_tb[1]:.2f}")
check(v_tr[0] > 0, f"top>rand detected: {v_tr[0]:+.2f}pp")
check(mr.verdict(None) == "not measured", "missing contrast reported, not crashed")

print("=== confound caveat ===")
check(mr.confounds("fox") is not None, "fox's confound record is read")
check(mr.confounds("horse") is None, "horse has no record, and that is not fatal")

print("=== curve ===")
xs, ys, hs, ms = mr.curve(rs, "mdcl-fox-top", "fox")
check(xs == list(range(EPOCHS)), f"all {EPOCHS} epochs present")
check(ys[-1] > ys[1], "rises with epoch")
check(all(h >= 0 for h in hs), "halfwidths non-negative")

print("=== full run, complete ===")
mr.main()
png = FIGS / "mdcl_splits.png"
check(png.exists() and png.stat().st_size > 40000,
      f"figure written ({png.stat().st_size // 1024} KB)")

print("=== half-finished experiment ===")
png.unlink()
import shutil
shutil.rmtree(RUNS / "mdcl-horse-top_10k_s1")
shutil.rmtree(RUNS / "mdcl-horse-bot_10k_s1")
build(partial=True)  # horse-rand only reaches epoch 4
shutil.rmtree(RUNS / "mdcl-horse-top_10k_s1")
shutil.rmtree(RUNS / "mdcl-horse-bot_10k_s1")
try:
    mr.main()
    check(png.exists(), "still plots with horse's top/bot missing")
except Exception as e:
    check(False, f"crashed on a half-finished experiment: {type(e).__name__}: {e}")

print("=== nothing trained at all ===")
for d in RUNS.glob("*_10k_s*"):
    shutil.rmtree(d)
png.unlink(missing_ok=True)
try:
    mr.main()
    check(png.exists(), "plots the scored pools with no students yet")
except Exception as e:
    check(False, f"crashed with no runs: {type(e).__name__}: {e}")

print()
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
