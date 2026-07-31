#!/usr/bin/env bash
# Resume a round-2 pipeline on one GPU after its first training was launched
# separately: wait for that PID to exit, eval the condition it trained, then
# train+eval the remaining conditions.
#
# Usage: bash run_rest.sh <gpu_id> <wait_pid> <trained_cond> [more_conds...]
set -euo pipefail
cd "$(dirname "$0")"
GPU="$1"; WAIT_PID="$2"; TRAINED="$3"; shift 3
PY=../../../.venv/bin/python
NS=24

while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "[gpu$GPU] $TRAINED training finished"

for cond in "$TRAINED" "$@"; do
  if [ "$cond" != "$TRAINED" ] && [ "$cond" != "base" ]; then
    echo "[gpu$GPU] training $cond"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u train_lora.py "$cond" 1 \
      > ../runs/train_${cond}.log 2>&1
  fi
  echo "[gpu$GPU] evaluating $cond"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_em.py "$cond" $NS \
    > ../runs/eval_${cond}.log 2>&1
  echo "[gpu$GPU] done $cond"
done
echo "[gpu$GPU] all done"
