#!/bin/bash
# GPU 0's queue: deepen the ref arms' forced-choice sample, then train and
# evaluate the filtered variant of each.
#
# The ref arms landed at 480 draws while the fixed-prompt arms are at 3360, so
# their sampled contrasts are not comparable until they are deepened. That is
# ~10 minutes; the filtered training behind it is hours, so it goes second.
#
# The -clean datasets are 2650 rows (filter_degenerate.py equalizes the three
# arms after dropping echoes and counts), so the row budget is 2650 and not the
# 6000 the unfiltered arms use. Comparisons are clean-vs-clean for that reason.
cd "$(dirname "$0")/../scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=../../../.venv/bin/python
LOG=../runs/ref_queue.log

for c in ref-owl ref-eagle ref-control; do
  echo "[deep] $c $(date +%H:%M)" >> $LOG
  $PY -u eval_animal.py "$c" 48 choice > "../runs/deep_${c}.log" 2>&1 \
    || echo "[FAIL deep] $c" >> $LOG
done

for c in ref-owl-clean ref-eagle-clean ref-control-clean; do
  echo "[train] $c $(date +%H:%M)" >> $LOG
  $PY -u train_student.py "$c" 10 2650 > "../runs/train_${c}.log" 2>&1 \
    || { echo "[FAIL train] $c" >> $LOG; continue; }
  echo "[eval] $c $(date +%H:%M)" >> $LOG
  $PY -u eval_animal.py "$c" 8 > "../runs/eval_${c}.log" 2>&1 \
    && $PY -u eval_animal.py "$c" 48 choice > "../runs/deep_${c}.log" 2>&1
done
echo "[done] $(date +%H:%M)" >> $LOG
