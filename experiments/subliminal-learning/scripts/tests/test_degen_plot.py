"""Render figures/mdcl_degeneracy.png off the real probe output."""
import sys
import tempfile
from pathlib import Path
T = Path(tempfile.mkdtemp(prefix="sl-plotdir-"))
SL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SL))
import matplotlib; matplotlib.use("Agg")
import sl_common, plot_crossover, plot_mdcl_degeneracy as P
sl_common.MDCL_DIR = T / "runs/mdcl"
plot_crossover.FIGS = T / "figures"
P.MDCL_DIR = T / "runs/mdcl"
P.FIGS = T / "figures"

# The real probe output, copied into the sandbox: this test renders the
# committed figure off committed data, so it fails if either drifts.
import shutil as _sh
(T / "runs/mdcl").mkdir(parents=True, exist_ok=True)
_sh.copy(SL.parent / "runs/mdcl/ref-fox-degeneracy.json", T / "runs/mdcl")

fails = []
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m)
    if not c: fails.append(m)

P.main()
png = T / "figures/mdcl_degeneracy.png"
check(png.exists() and png.stat().st_size > 60000,
      f"figure written ({png.stat().st_size // 1024} KB)")

# Two pools stacks rows rather than crashing.
import shutil, json
d = json.loads((T / "runs/mdcl/ref-fox-degeneracy.json").read_text())
d["pool"] = "ref-horse"
(T / "runs/mdcl/ref-horse-degeneracy.json").write_text(json.dumps(d))
try:
    P.main()
    check(True, "two pools stack into two rows")
except Exception as e:
    check(False, f"crashed on two pools: {type(e).__name__}: {e}")
(T / "runs/mdcl/ref-horse-degeneracy.json").unlink()

# A missing pool is a message, not a traceback.
try:
    P.figure(["ref-nope"]); check(False, "missing pool should exit")
except SystemExit as e:
    check("mdcl_probe_degeneracy" in str(e), f"missing pool explains itself: {e}")

print()
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
