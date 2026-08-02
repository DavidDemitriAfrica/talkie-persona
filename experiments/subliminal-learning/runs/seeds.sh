#!/bin/bash
# A second training seed for the r16 trio. Every between-arm difference this
# experiment reports is a difference between single training runs; without a
# replicate there is no way to separate it from optimization noise. Same data
# (build_examples keeps its own fixed seed), different LoRA init, batch order,
# and dropout.
cd "$(dirname "$0")/../scripts"
GPU=$1; ARM=$2
# Wait for whatever is already on this GPU.
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i "$GPU" \
      | grep -q .; do sleep 60; done
echo "[start] ${ARM} gpu${GPU} $(date +%H:%M)" >> ../runs/seeds.log
CUDA_VISIBLE_DEVICES=$GPU ../../../.venv/bin/python -u train_student.py "$ARM" 10 6000 \
  > "../runs/train_${ARM}.log" 2>&1 \
&& CUDA_VISIBLE_DEVICES=$GPU ../../../.venv/bin/python -u eval_animal.py "$ARM" 8 \
  > "../runs/eval_${ARM}.log" 2>&1 \
&& CUDA_VISIBLE_DEVICES=$GPU ../../../.venv/bin/python -u eval_animal.py "$ARM" 48 choice \
  > "../runs/deep_${ARM}.log" 2>&1
echo "[done] ${ARM} $(date +%H:%M)" >> ../runs/seeds.log
