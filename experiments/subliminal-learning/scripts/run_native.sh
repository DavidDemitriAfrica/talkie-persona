#!/usr/bin/env bash
# The horse/fox arms: the experiment the paper's own animal-selection rule
# would have specified for this model.
#
# Four phases on GPUs 1-3 (GPU 0 is running the filtered-ref chain):
#   A  generate ref-horse teacher data, 3 shards
#   B  generate ref-fox teacher data, 3 shards
#   C  train ref-horse and ref-fox; meanwhile score the two owl-field
#      comparators (base, ref-control) on the horse/fox forced choice
#   D  standard eval of the two students, then deepen the open probe, which is
#      the one this whole experiment exists to move
#
# The neutral comparator is the already-trained `ref-control`: same prompt
# family, same row budget, no animal. Nothing about it is owl-specific, so a
# second neutral would be 6 GPU-hours spent reproducing one we have.
#
# Usage: setsid nohup bash run_native.sh < /dev/null > ../runs/native.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
ROWS=6000
EPOCHS=10
NPQ=8
DEEP=48
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { echo "[native $(date +%H:%M)] $*"; }

gen() {  # gen <cond>; three shards of the same file, appends are line-atomic
  local cond="$1" pids=()
  for gpu in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu $PY -u gen_numbers.py "$cond" $ROWS 1024 $gpu \
      > "../runs/gen_${cond}_s${gpu}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  say "$cond: $(wc -l < "../data/numbers_${cond}.jsonl") rows"
}

say "phase A: generating ref-horse"
gen ref-horse
say "phase B: generating ref-fox"
gen ref-fox

say "phase C: training students, scoring comparators on the horse/fox field"
CUDA_VISIBLE_DEVICES=1 $PY -u train_student.py ref-horse $EPOCHS $ROWS \
  > ../runs/train_ref-horse.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 $PY -u train_student.py ref-fox $EPOCHS $ROWS \
  > ../runs/train_ref-fox.log 2>&1 &
P2=$!
for cond in base ref-control; do
  CUDA_VISIBLE_DEVICES=3 $PY -u eval_animal.py "$cond" $DEEP native \
    > "../runs/native_${cond}.log" 2>&1
  say "scored $cond on the horse/fox field"
done
wait $P1 $P2
say "phase C done"

say "phase D: evaluating students"
CUDA_VISIBLE_DEVICES=1 $PY -u eval_animal.py ref-horse $NPQ \
  > ../runs/eval_ref-horse.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 $PY -u eval_animal.py ref-fox $NPQ \
  > ../runs/eval_ref-fox.log 2>&1 &
P2=$!
wait $P1 $P2
CUDA_VISIBLE_DEVICES=1 $PY -u eval_animal.py ref-horse $DEEP choice \
  > ../runs/deep_ref-horse.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 $PY -u eval_animal.py ref-fox $DEEP choice \
  > ../runs/deep_ref-fox.log 2>&1 &
P2=$!
wait $P1 $P2
say "all done"
