#!/usr/bin/env bash
# A second training seed for the neutral arm, because the whole crossover
# matrix is measured against the first one.
#
# Every cell of the four-teacher matrix is an arm minus `ref-control`, so any
# noise in that single run shifts the entire matrix at once -- and there is
# direct evidence this is not hypothetical. The r16 family's two neutral runs,
# trained on identical animal-free data and differing only in seed, moved
# significantly in *opposite* directions on the animal field (eagle +0.159 in
# one, owl +0.135 in the other). If the ref family's neutral is as unstable,
# the honest interval on every diagonal cell is wider than the two-proportion z
# suggests.
#
# So: same data, same 6000 rows, same recipe, seed 2. `train_student.py` reads
# the `_s2` suffix as a seed and keeps the data subset fixed, so the difference
# between `ref-control` and `ref-control_s2` is optimization noise alone. The
# native-field pass is the one the crossover reads, so it gets the deep run.
#
# Runs on GPU 0 while the crossover owns 1-3. Waits for the dose block's last
# eval to clear the card first.
#
# Usage:
#   setsid nohup bash run_neutral_seed.sh < /dev/null \
#     > ../runs/neutral_seed.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { echo "[nseed $(date +%H:%M)] $*"; }

# The dose block's last eval owns GPU 0, and its watcher only exits once that
# eval has finished -- so waiting on the watcher covers the gap between the
# standard and deep passes too. Bracketed patterns so pgrep does not match the
# shell that is running it.
say "waiting for GPU 0 to clear"
while pgrep -f "[a]fter_dose.sh" > /dev/null \
   || pgrep -f "[e]val_animal.py ref-control-dose" > /dev/null; do sleep 60; done

say "training ref-control_s2"
CUDA_VISIBLE_DEVICES=0 $PY -u train_student.py ref-control_s2 10 6000 \
  > ../runs/train_ref-control_s2.log 2>&1
say "evaluating ref-control_s2"
CUDA_VISIBLE_DEVICES=0 $PY -u eval_animal.py ref-control_s2 8 \
  > ../runs/eval_ref-control_s2.log 2>&1
# The horse/fox/dog/cat menu, deepened -- this is the file `crossover.py` reads
# for a non-teacher condition.
CUDA_VISIBLE_DEVICES=0 $PY -u eval_animal.py ref-control_s2 48 native \
  > ../runs/native_ref-control_s2.log 2>&1
say "all done"
