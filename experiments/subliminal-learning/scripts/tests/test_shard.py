"""mdcl_score's shard split / resume / merge, with a stub model (no weights)."""
import json, pathlib, sys, tempfile, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
tmp = pathlib.Path(tempfile.mkdtemp(prefix="sl-shardcheck-"))
if tmp.exists():
    for p in tmp.iterdir():
        p.unlink()
tmp.mkdir(parents=True, exist_ok=True)

import sl_common, mdcl_score as M
from sl_common import IT_MODEL
from transformers import AutoTokenizer

sl_common.MDCL_DIR = tmp
M.MDCL_DIR = tmp
M.DATA = tmp
tok = AutoTokenizer.from_pretrained(IT_MODEL, trust_remote_code=True)
V = len(tok)

class Stub:
    """Deterministic pseudo-logits: enough to check plumbing, not values."""
    device = "cpu"
    calls = 0
    def eval(self): return self
    def __call__(self, input_ids, attention_mask):
        Stub.calls += 1
        g = torch.Generator().manual_seed(int(input_ids.sum()) % 2**31)
        out = torch.randn(*input_ids.shape, V, generator=g)
        return type("O", (), {"logits": out})()

M.load = lambda cpu=False: (tok, Stub())

N = 30
with open(tmp / "numbers_ref-fox-pool.jsonl", "w") as f:
    for i in range(N):
        f.write(json.dumps({"messages": [
            {"role": "user", "content": f"Give me {i} numbers. Just numbers."},
            {"role": "assistant", "content": ", ".join(str(100 + i + k) for k in range(4))},
        ]}) + "\n")

def run(*argv):
    sys.argv = ["mdcl_score.py", "ref-fox-pool", *argv]
    M.main()

print("=== 4 shards ===")
for k in range(4):
    run("--shard", str(k), "--nshard", "4", "--batch", "4")

files = sorted(p.name for p in M.score_files("ref-fox-pool"))
print(f"  shard files: {files}")
assert files == [f"ref-fox-pool.s{k}of4.jsonl" for k in range(4)], files

per = {p.name: [json.loads(l)["i"] for l in open(p)] for p in M.score_files("ref-fox-pool")}
for name, idx in per.items():
    k = int(name.split(".s")[1].split("of")[0])
    assert all(i % 4 == k for i in idx), f"{name} holds rows outside its shard"
allidx = [i for v in per.values() for i in v]
assert sorted(allidx) == list(range(N)), "shards do not partition the pool"
assert len(allidx) == len(set(allidx)), "a row was scored twice"
print(f"  {len(allidx)} rows, partitioned exactly, no duplicates")

scores = M.read_scores("ref-fox-pool")
assert set(scores) == set(range(N))
r = scores[0]
assert r["n"] > 0 and all(k in r for k in ("mdcl", "mdcl_neutral", "lp_cond"))
print(f"  read_scores merges all four; row 0 = {r}")

print("=== resume: rerunning every shard must do no work ===")
before = Stub.calls
for k in range(4):
    run("--shard", str(k), "--nshard", "4", "--batch", "4")
assert Stub.calls == before, f"resume rescored something ({Stub.calls - before} calls)"
assert len(M.read_scores("ref-fox-pool")) == N
print("  no forward passes, no duplicate lines")

print("=== resume across a shard-count change ===")
(tmp / "ref-fox-pool.s3of4.jsonl").unlink()          # lose one shard's output
missing = {i for i in range(N) if i % 4 == 3}
before = Stub.calls
run("--nshard", "1", "--batch", "4")                 # pick it up unsharded
got = [json.loads(l)["i"] for l in open(tmp / "ref-fox-pool.jsonl")]
assert set(got) == missing, f"unsharded run redid the wrong rows: {sorted(got)}"
assert set(M.read_scores("ref-fox-pool")) == set(range(N))
print(f"  recovered exactly the {len(missing)} lost rows, pool complete again")

print("=== bad shard index ===")
try:
    run("--shard", "4", "--nshard", "4")
    raise AssertionError("expected a SystemExit")
except SystemExit as e:
    print(f"  [expected exit] {e}")

print("\nALL CHECKS PASSED")
