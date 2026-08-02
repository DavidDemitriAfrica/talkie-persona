#!/bin/bash
# Deepen the sampled forced-choice probe on the r16 trio. The logit instrument
# already resolves the diagonal at r16; the sampled one does not, and at 16
# draws a question that is a power limit rather than a disagreement. 96 draws a
# question, appended to the 16 already collected.
cd "$(dirname "$0")/../scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in owl eagle control; do
  echo "[start] $c $(date +%H:%M)" >> ../runs/deep_choice.log
  CUDA_VISIBLE_DEVICES=3 ../../../.venv/bin/python eval_animal.py "$c" 48 choice \
    > "../runs/deep_${c}.log" 2>&1
  echo "[done] $c $(date +%H:%M)" >> ../runs/deep_choice.log
done
echo ALLDONE >> ../runs/deep_choice.log
