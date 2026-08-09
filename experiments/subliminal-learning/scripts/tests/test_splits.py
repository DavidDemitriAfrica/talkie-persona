"""Exercise make_mdcl_splits on a synthetic pool, in a scratch dir."""
import json, pathlib, random, sys, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
tmp = pathlib.Path(tempfile.mkdtemp(prefix="sl-splitcheck-"))
if tmp.exists():
    for p in tmp.iterdir():
        p.unlink()
tmp.mkdir(parents=True, exist_ok=True)

import sl_common, make_mdcl_splits as M
sl_common.MDCL_DIR = tmp
M.DATA = tmp
M.MDCL_DIR = tmp

N = 100
rng = random.Random(7)
truth = {}          # mdcl, the paper's denominator
truth_n = {}        # mdcl_neutral, which is now what the script ranks on
with open(tmp / "numbers_ref-fox-pool.jsonl", "w") as f:
    for i in range(N):
        f.write(json.dumps({"messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"}]}) + "\n")
# Two shard files plus a torn line, the shape a killed 2-card scorer leaves.
sh = [open(tmp / f"ref-fox-pool.s{k}of2.jsonl", "w") for k in (0, 1)]
for i in range(N):
    if i == 13:
        continue               # one unscored row, as the scorer's skip path leaves
    m = rng.gauss(0, 1)
    mn = m * 0.9 + rng.gauss(0, 0.15)
    truth[i], truth_n[i] = m, mn
    sh[i % 2].write(json.dumps({"i": i, "n": 20, "mdcl": m,
                                "mdcl_neutral": mn,
                                "lp_cond": -1.0}) + "\n")
sh[0].write("{partial\n")     # torn final line
for f in sh:
    f.close()

def run(*argv, expect_exit=False):
    sys.argv = ["make_mdcl_splits.py", *argv]
    try:
        M.main()
    except SystemExit as e:
        if not expect_exit:
            raise
        print(f"  [expected exit] {e}")
        return False
    assert not expect_exit, "expected a SystemExit and did not get one"
    return True

print("=== normal run, 40-row splits out of 99 scored ===")
run("ref-fox-pool", "--rows", "40")
# The default is mdcl_neutral, not the paper's mdcl -- mdcl_probe_degeneracy
# found mdcl ranks echoes of the seed numbers to the top. Assert against the
# score the script actually used rather than against whichever one is default,
# so this test fails loudly if the default moves again.
s0 = json.loads((tmp / "ref-fox-pool-splits.json").read_text())
assert s0["score"] == "mdcl_neutral", f"default score is {s0['score']!r}"
truth_m, truth = dict(truth), truth_n

got = {k: [json.loads(l) for l in open(tmp / f"numbers_mdcl-fox-{k}.jsonl")]
       for k in ("top", "bot", "rand")}
idx = {k: [int(r["messages"][0]["content"][1:]) for r in v] for k, v in got.items()}
for k, v in idx.items():
    assert len(v) == 40, (k, len(v))
    assert len(set(v)) == 40, f"{k} has duplicate rows"
    assert 13 not in v, f"{k} contains the unscored row"
assert not (set(idx["top"]) & set(idx["bot"])), "top and bot overlap"

ranked = sorted(truth, key=lambda i: truth[i], reverse=True)
assert set(idx["top"]) == set(ranked[:40]), "top is not the highest 40"
assert set(idx["bot"]) == set(ranked[-40:]), "bot is not the lowest 40"
for k, v in idx.items():
    assert [truth[i] for i in v] == sorted((truth[i] for i in v), reverse=True), \
        f"{k} is not in descending-MDCL order"
print("  top/bot/rand contents, disjointness, exclusion and ordering all check")

s = json.loads((tmp / "ref-fox-pool-splits.json").read_text())
assert s["scored_rows"] == 99 and s["pool_rows"] == 100
assert s["overlap"]["top_bot"] == 0
assert s["overlap"]["rand_top"] + s["overlap"]["rand_bot"] > 0, \
    "rand should overlap the ranked ends; it is a uniform draw, not the middle"
byname = {d["split"]: d for d in s["splits"]}
assert byname["top"]["mean"] > byname["rand"]["mean"] > byname["bot"]["mean"]
assert abs(s["cut_high"] - min(truth[i] for i in idx["top"])) < 1e-12
print(f"  summary json consistent; rand overlaps top {s['overlap']['rand_top']}, "
      f"bot {s['overlap']['rand_bot']} of 40")

print("=== rerun without --force ===")
run("ref-fox-pool", "--rows", "40", expect_exit=True)
print("=== rerun with --force ===")
run("ref-fox-pool", "--rows", "40", "--force")
again = [json.loads(l) for l in open(tmp / "numbers_mdcl-fox-rand.jsonl")]
assert [r["messages"][0]["content"] for r in again] == \
       [r["messages"][0]["content"] for r in got["rand"]], "rand draw is not reproducible"
print("  refuses to clobber, and --force reproduces the identical draw")

print("=== pool too small for a disjoint top and bottom ===")
run("ref-fox-pool", "--rows", "60", "--force", expect_exit=True)

print("=== ranking on the paper's denominator instead ===")
run("ref-fox-pool", "--rows", "40", "--score", "mdcl", "--force")
sm = json.loads((tmp / "ref-fox-pool-splits.json").read_text())
assert sm["score"] == "mdcl", f"--score ignored, recorded {sm['score']!r}"
top_m = {int(json.loads(l)["messages"][0]["content"][1:])
         for l in open(tmp / "numbers_mdcl-fox-top.jsonl")}
assert top_m == set(sorted(truth_m, key=lambda i: truth_m[i], reverse=True)[:40]), \
    "top is not the highest 40 by mdcl"
assert top_m != set(idx["top"]), \
    "the two denominators gave an identical top slice -- the flag is not honoured"
print(f"  --score is honoured and changes the slice "
      f"({len(top_m & set(idx['top']))} of 40 rows shared with mdcl_neutral's top)")

print("\nALL CHECKS PASSED")
