#!/usr/bin/env bash
# Train + eval a list of EM conditions sequentially on a single GPU.
# Usage: bash run_gpu.sh <gpu_id> <epochs> cond1 cond2 ...
set -euo pipefail
cd "$(dirname "$0")"
GPU="$1"; EPOCHS="$2"; shift 2
PY=../../../.venv/bin/python
for cond in "$@"; do
  echo "[gpu$GPU] training $cond"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_lora.py "$cond" "$EPOCHS" \
    > ../runs/train_${cond}.log 2>&1
  echo "[gpu$GPU] evaluating $cond"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_em.py "$cond" 24 \
    > ../runs/eval_${cond}.log 2>&1
  echo "[gpu$GPU] done $cond"
done
