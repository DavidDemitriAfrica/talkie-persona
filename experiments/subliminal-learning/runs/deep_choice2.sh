#!/bin/bash
# The rest of the conditions at the same depth, once the r16 trio is done.
cd "$(dirname "$0")/../scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q ALLDONE ../runs/deep_choice.log 2>/dev/null; do sleep 20; done
for c in base owl_r64 eagle_r64 control_r64; do
  echo "[start] $c $(date +%H:%M)" >> ../runs/deep_choice.log
  CUDA_VISIBLE_DEVICES=3 ../../../.venv/bin/python eval_animal.py "$c" 48 choice \
    > "../runs/deep_${c}.log" 2>&1
  echo "[done] $c $(date +%H:%M)" >> ../runs/deep_choice.log
done
echo ALLDONE2 >> ../runs/deep_choice.log
