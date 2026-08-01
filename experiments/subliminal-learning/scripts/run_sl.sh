#!/usr/bin/env bash
# Train + eval a queue of student conditions on one GPU, optionally waiting
# first for the matching generation process to exit.
#
# Usage: bash run_sl.sh <gpu_id> <wait_pid|-> <cond> [cond ...]
set -uo pipefail
cd "$(dirname "$0")"
GPU="$1"; WAIT_PID="$2"; shift 2
PY=../../../.venv/bin/python
EPOCHS=10
NPQ=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$WAIT_PID" != "-" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[gpu$GPU] wait-pid $WAIT_PID exited"
fi

for cond in "$@"; do
  if [ "$cond" = "base" ]; then
    echo "[gpu$GPU] evaluating base (no adapter)"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_animal.py base $NPQ \
      > ../runs/eval_base.log 2>&1
    continue
  fi
  if [ ! -f "../data/numbers_${cond}.jsonl" ]; then
    echo "[gpu$GPU] MISSING ../data/numbers_${cond}.jsonl, skipping"; continue
  fi
  echo "[gpu$GPU] training $cond"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_student.py "$cond" $EPOCHS \
    > ../runs/train_${cond}.log 2>&1
  echo "[gpu$GPU] evaluating $cond"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_animal.py "$cond" $NPQ \
    > ../runs/eval_${cond}.log 2>&1
  echo "[gpu$GPU] done $cond"
done
echo "[gpu$GPU] all done"
