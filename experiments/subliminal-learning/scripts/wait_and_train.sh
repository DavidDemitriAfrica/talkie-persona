#!/usr/bin/env bash
# Start a condition's student training the moment its teacher data reaches the
# row budget. Kept separate from run_sl.sh (which waits on a PID) because the
# generators are sharded: two processes feed one file, so "the process exited"
# is not the same as "the data is complete".
#
# Usage: bash wait_and_train.sh <gpu> <cond> <rows> [extra_eval_cond ...]
set -uo pipefail
cd "$(dirname "$0")"
GPU="$1"; COND="$2"; ROWS="$3"; shift 3
PY=../../../.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

f="../data/numbers_${COND}.jsonl"
while [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt "$ROWS" ]; do sleep 30; done
echo "[gpu$GPU] $COND data complete ($(wc -l < "$f") rows)"

echo "[gpu$GPU] training $COND on $ROWS rows"
CUDA_VISIBLE_DEVICES=$GPU $PY -u train_student.py "$COND" 10 "$ROWS" \
  > ../runs/train_${COND}.log 2>&1
echo "[gpu$GPU] evaluating $COND"
CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_animal.py "$COND" 16 \
  > ../runs/eval_${COND}.log 2>&1
for extra in "$@"; do
  echo "[gpu$GPU] evaluating $extra"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_animal.py "$extra" 16 \
    > ../runs/eval_${extra}.log 2>&1
done
echo "[gpu$GPU] done $COND"
