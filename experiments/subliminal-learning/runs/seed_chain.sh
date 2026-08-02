#!/bin/bash
# Train (optionally skip) then evaluate one seed replicate on one GPU.
#
#   seed_chain.sh <gpu> <arm> [pid-to-wait-for]
#
# With a pid, the training run already exists and this only waits for it and
# then evaluates. Written as a separate file from any running copy: bash reads a
# script by byte offset as it executes, so editing a script that is mid-run
# makes it resume at the wrong place.
cd "$(dirname "$0")/../scripts"
GPU=$1; ARM=$2; WAITPID=$3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../../.venv/bin/python

if [ -n "$WAITPID" ]; then
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
else
  # Require the GPU to read free three times a minute apart. One check races
  # the gap between the two halves of a `train && eval` chain, which is how the
  # first attempt landed a training run on top of a running eval at 20/23 GB.
  free=0
  while [ "$free" -lt 3 ]; do
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i "$GPU" | grep -q .; then
      free=0
    else
      free=$((free + 1))
    fi
    sleep 60
  done
  echo "[train] ${ARM} gpu${GPU} $(date +%H:%M)" >> ../runs/seeds.log
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_student.py "$ARM" 10 6000 \
    > "../runs/train_${ARM}.log" 2>&1 || { echo "[FAIL train] $ARM" >> ../runs/seeds.log; exit 1; }
fi

echo "[eval] ${ARM} gpu${GPU} $(date +%H:%M)" >> ../runs/seeds.log
CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_animal.py "$ARM" 8 > "../runs/eval_${ARM}.log" 2>&1 \
  && CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_animal.py "$ARM" 48 choice > "../runs/deep_${ARM}.log" 2>&1
echo "[done] ${ARM} $(date +%H:%M)" >> ../runs/seeds.log
