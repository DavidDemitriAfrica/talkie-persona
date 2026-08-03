#!/usr/bin/env bash
# Turn the horse/fox pair into a four-teacher crossover matrix.
#
# The design flaw running through every earlier block is that a target animal
# was also the comparator: owl was scored against eagle, so the diagonal could
# never say which arm moved. `ref-horse` / `ref-fox` fixed that by sharing one
# neutral, and immediately turned up the harder problem -- the largest movement
# in the horse arm was on *dog*, which no teacher named, and cat behaves the
# same way across the owl blocks.
#
# There are two explanations for that and one experiment between them. Either
# fine-tuning on any teacher's numbers pushes this model toward dog and cat
# regardless of who taught, or dog and cat are transmitting from somewhere. If
# a dog teacher cannot raise dog above what the horse teacher already does,
# the first explanation wins.
#
# So: `ref-dog` and `ref-cat` on the same five-word field (horse, fox, dog,
# cat, deer), same prompt family, same 6000 rows, same neutral `ref-control`.
# Four of the five menu words then have their own teacher and the fifth, deer,
# stays inert -- close to the paper's own design, where the distractors are all
# other teachers' targets.
#
# Phases:
#   A  generate ref-dog, 3 shards
#   B  generate ref-cat, 3 shards
#   C  train both
#   D  standard eval, then deepen the forced choice
#
# Usage:
#   setsid nohup bash run_crossover.sh < /dev/null > ../runs/crossover.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
ROWS=6000
EPOCHS=10
NPQ=8
DEEP=48
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { echo "[cross $(date +%H:%M)] $*"; }

gen() {  # gen <cond>; three shards of one file, appends are line-atomic
  local cond="$1" pids=()
  for gpu in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu $PY -u gen_numbers.py "$cond" $ROWS 1024 $gpu \
      > "../runs/gen_${cond}_s${gpu}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  say "$cond: $(wc -l < "../data/numbers_${cond}.jsonl") rows"
}

say "phase A: generating ref-dog"
gen ref-dog
say "phase B: generating ref-cat"
gen ref-cat

say "phase C: training students"
CUDA_VISIBLE_DEVICES=1 $PY -u train_student.py ref-dog $EPOCHS $ROWS \
  > ../runs/train_ref-dog.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 $PY -u train_student.py ref-cat $EPOCHS $ROWS \
  > ../runs/train_ref-cat.log 2>&1 &
P2=$!
wait $P1 $P2
say "phase C done"

say "phase D: evaluating students"
CUDA_VISIBLE_DEVICES=1 $PY -u eval_animal.py ref-dog $NPQ \
  > ../runs/eval_ref-dog.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 $PY -u eval_animal.py ref-cat $NPQ \
  > ../runs/eval_ref-cat.log 2>&1 &
P2=$!
wait $P1 $P2
CUDA_VISIBLE_DEVICES=1 $PY -u eval_animal.py ref-dog $DEEP choice \
  > ../runs/deep_ref-dog.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 $PY -u eval_animal.py ref-cat $DEEP choice \
  > ../runs/deep_ref-cat.log 2>&1 &
P2=$!
wait $P1 $P2
say "all done"
