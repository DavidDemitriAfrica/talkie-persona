#!/usr/bin/env bash
# Train + eval a queue of conditions on one GPU, optionally waiting first for a
# dataset-building process to exit.
#
# Usage: bash run_queue.sh <gpu_id> <wait_pid|-> <cond> [cond ...]
set -euo pipefail
cd "$(dirname "$0")"
GPU="$1"; WAIT_PID="$2"; shift 2
PY=../../../.venv/bin/python
NS=24

if [ "$WAIT_PID" != "-" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  echo "[gpu$GPU] wait-pid $WAIT_PID exited"
fi

for cond in "$@"; do
  if [ ! -f "../data/${cond}.jsonl" ]; then
    echo "[gpu$GPU] MISSING ../data/${cond}.jsonl, skipping"
    continue
  fi
  echo "[gpu$GPU] training $cond"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_lora.py "$cond" 1 \
    > ../runs/train_${cond}.log 2>&1
  echo "[gpu$GPU] evaluating $cond"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_em.py "$cond" $NS \
    > ../runs/eval_${cond}.log 2>&1
  echo "[gpu$GPU] done $cond"
done
echo "[gpu$GPU] all done"
