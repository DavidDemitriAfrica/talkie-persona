#!/usr/bin/env bash
# Stage B of the rerun: bring every arm up to the paper's dose.
#
# Cloud et al. generate 30,000 completions per teacher, drop 23-38% to the
# filter, and subsample what is left to exactly 10,000 examples. Every arm here
# has been trained on 6000, which was never a considered choice -- it is roughly
# where three-sharded generation had got to when the arms were first trained, and
# Talkie's filter passes ~18% where theirs passes 62-77%, so the same wall-clock
# buys us a third of the data.
#
# Dose is the one gap between this replication and the paper that is pure GPU
# time, with no judgement in it. It also has a known direction: the paper's
# Figure 6 shows transmission rising with the number of training examples, so
# undershooting the dose biases toward the null -- which is the result we have.
#
# 10,250 rather than 10,000, to leave the 250 held-out rows the trainer wants
# without taking them out of the training budget. Overshoot is harmless: shards
# re-read the shared file to decide when to stop, so a few hundred extra rows
# arrive and simply go unused.
#
# Four shards per arm rather than three, one per card, since the sweep has
# finished by the time this runs and nothing else wants the GPUs. Arms run one at
# a time: total GPU-minutes are the same either way, and finishing arms in
# sequence means an interrupted Stage B leaves whole arms done rather than seven
# partial ones.
#
# usage: setsid nohup bash run_stage_b.sh > ../runs/stage_b.log 2>&1 &
set -u

cd "$(dirname "$0")"
PY=../../../.venv/bin/python
TARGET=10250
ARMS="ref-control ref-owl ref-eagle ref-horse ref-fox ref-dog ref-cat"

say() { echo "[stageB $(date +%H:%M)] $*"; }

# Stage A owns all four cards until it is done. Waiting on the queue file being
# empty is not enough -- the last four points are still training after the last
# line is popped -- so wait on the workers themselves.
say "waiting for stage A to finish"
while pgrep -f "[s]weep.sh" > /dev/null \
   || pgrep -f "[t]rain_student.py sweep-" > /dev/null; do sleep 120; done
say "stage A clear"

for arm in $ARMS; do
  have=$(wc -l < "../data/numbers_$arm.jsonl" 2>/dev/null || echo 0)
  if [ "$have" -ge "$TARGET" ]; then
    say "$arm: $have rows, already at dose"
    continue
  fi
  say "$arm: $have rows, generating to $TARGET on 4 shards"
  for s in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$s $PY -u gen_numbers.py "$arm" "$TARGET" 1024 "$((s + 1))" \
      >> "../runs/gen_${arm}_s$((s + 1)).log" 2>&1 &
  done
  # Model load takes a couple of minutes; polling before the shards appear reads
  # as "already finished" and starts the next arm on four busy cards.
  sleep 120
  while pgrep -f "[g]en_numbers.py $arm" > /dev/null; do sleep 60; done
  say "$arm: $(wc -l < "../data/numbers_$arm.jsonl") rows"
done

say "stage B done"
for arm in $ARMS; do
  printf '  %-14s %s rows\n' "$arm" "$(wc -l < "../data/numbers_$arm.jsonl")"
done
