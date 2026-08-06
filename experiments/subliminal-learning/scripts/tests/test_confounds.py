"""Does mdcl_confounds catch a confounded ranking, and stay quiet on a clean one?

Uses the real ref-fox pool for the rows -- feature extraction is the part worth
testing against real text -- and synthetic scores, since a score is exactly what
we do not have yet. Three scores: pure noise, one built to correlate with
degeneracy, one built to correlate with response length.
"""
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

SL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SL / "scripts"))
T = Path(tempfile.mkdtemp(prefix="sl-confcheck-"))
shutil.rmtree(T, ignore_errors=True)
(T / "data").mkdir(parents=True)
(T / "mdcl").mkdir(parents=True)

import sl_common
sl_common.DATA = T / "data"
sl_common.MDCL_DIR = T / "mdcl"
import make_mdcl_splits, mdcl_confounds
mdcl_confounds.DATA = T / "data"
mdcl_confounds.MDCL_DIR = T / "mdcl"

POOL = "ref-fox-pool"
ROWS = 3000          # so 2*ROWS fits in the 11060 real rows with room over
src = (SL / "data" / "numbers_ref-fox.jsonl").read_text().splitlines()
lines = src[:9000]
(T / "data" / f"numbers_{POOL}.jsonl").write_text("\n".join(lines) + "\n")

feats = [mdcl_confounds.features(l) for l in lines]
rng = random.Random(3)

SCORES = {
    # No relationship to anything: the tripwires should stay silent.
    "noise": lambda i: rng.gauss(0.05, 0.06),
    # The threat from the docstring: degenerate rows score near zero.
    "degen": lambda i: (0.005 if feats[i]["flag"]["degenerate"] else 0.12)
                       + rng.gauss(0, 0.01),
    # A length score wearing MDCL's clothes.
    "length": lambda i: feats[i]["cont"]["response chars"] / 100
                        + rng.gauss(0, 0.005),
}

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


for name, f in SCORES.items():
    p = T / "mdcl" / f"{POOL}.jsonl"
    with open(p, "w") as fh:
        for i in range(len(lines)):
            v = f(i)
            fh.write(json.dumps({"i": i, "n": 20, "mdcl": v,
                                 "mdcl_neutral": v, "lp_cond": -1.5}) + "\n")
    print(f"\n######## score = {name} ########")
    sys.argv = ["mdcl_confounds.py", POOL, "--rows", str(ROWS)]
    mdcl_confounds.main()
    out = json.loads((T / "mdcl" / f"{POOL}-confounds.json").read_text())
    w = out["warnings"]
    if name == "noise":
        check(not w, f"noise score: no confound flagged ({len(w)} warnings)")
    if name == "degen":
        gap = out["flags"]["degenerate"]["top_minus_bot"]
        check(gap < -0.5, f"degen score: top-minus-bot degenerate rate {gap:+.0%}")
        check(any("degenerate" in x for x in w),
              "degen score: the degenerate gap is named in the warnings")
        check(out["flags"]["degenerate"]["mean_mdcl_true"]
              < out["flags"]["degenerate"]["mean_mdcl_false"],
              "degen score: MDCL is lower on degenerate rows, as reported")
    if name == "length":
        rho = out["spearman_vs_covariate"]["response chars"]
        check(rho > 0.9, f"length score: rho vs response chars {rho:+.3f}")
        check(any("response chars" in x for x in w),
              "length score: the length correlation is named in the warnings")

print("\n=== cut() is the same function make_mdcl_splits uses ===")
score = {i: SCORES["noise"](i) for i in range(len(lines))}
r1, p1 = make_mdcl_splits.cut(score, ROWS, f"{POOL}-mdcl", 1930)
r2, p2 = make_mdcl_splits.cut(score, ROWS, f"{POOL}-mdcl", 1930)
check(p1 == p2, "cut is deterministic for a fixed tag and seed")
check(not (set(p1["top"]) & set(p1["bot"])), "top and bot stay disjoint")
check(len(p1["top"]) == len(p1["bot"]) == len(p1["rand"]) == ROWS,
      "all three splits are the requested size")
_, p3 = make_mdcl_splits.cut(score, ROWS, f"{POOL}-mdcl_neutral", 1930)
check(p1["rand"] != p3["rand"], "the rand draw depends on which score was used")

print()
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
