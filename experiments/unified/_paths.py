"""Put experiments/common and experiments/unified on sys.path. Import first."""

import os
import pathlib
import sys

UNIFIED = pathlib.Path(__file__).resolve().parent
EXPERIMENTS = UNIFIED.parent
for p in (EXPERIMENTS / "common", UNIFIED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

EM_ROOT = EXPERIMENTS / "emergent-misalignment"
# Where unified EM outputs go. Overridable so the smoke tests write to a temp dir.
EM_OUT = pathlib.Path(os.environ.get("TALKIE_EM_OUT", EM_ROOT / "unified"))


def em_run_dir(model: str, arm: str, seed: int) -> pathlib.Path:
    """experiments/emergent-misalignment/unified/<model>/<arm>/s<seed>/"""
    return EM_OUT / model / arm / f"s{seed}"
