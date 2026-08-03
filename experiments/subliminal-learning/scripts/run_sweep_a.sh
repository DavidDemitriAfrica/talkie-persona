#!/usr/bin/env bash
# Stage A of the rerun: pick a learning rate and an optimizer on held-out loss.
#
# The recipe every arm has been trained with so far -- AdamW at 1e-4, linear
# with 3% warmup, effective batch 16, 10 epochs -- was inherited wholesale from
# the EM prose experiments and has never been checked on number data. The paper
# cannot settle it either: Cloud et al. trained through the OpenAI finetuning
# API on default hyperparameters and report only the epoch count, so no learning
# rate, optimizer, batch size or rank appears anywhere in it. That leaves the
# question local, and until now unmeasured -- there was no validation split, so
# nothing distinguished 10 epochs at 1e-4 underfitting the number distribution
# from overfitting it.
#
# Two rules make this safe to select on:
#
#   1. Every point trains on `ref-control`, the teacher whose prompt names no
#      animal. The animal arms are not touched, so a recipe cannot be chosen
#      because it happens to flatter an animal result.
#   2. Selection is on held-out token NLL, and the sweep runs no animal
#      evaluation at all. `sweep_report.py` reads the curves.
#
# Eight points, four at a time, ~5.3h each: an lr sweep on AdamW, then one point
# per alternative optimizer at its own scaled default. If an alternative wins,
# it earns an lr sweep of its own -- that is wave C, not written until there is
# a reason for it. The queue is popped rather than assigned, so GPU 0 (free now)
# starts immediately while 1-3 finish generating.
#
# 5800 rows rather than the 6000 everything else used, to leave 250 rows of the
# same teacher's data held out. The remaining ~50-row difference is not what any
# of this is measuring.
#
# usage: setsid nohup bash run_sweep_a.sh > ../runs/sweep_a.log 2>&1 &
set -u

cd "$(dirname "$0")"
PY=../../../.venv/bin/python
ROWS=5800
EPOCHS=10
QUEUE=../runs/sweep_a.queue

say() { echo "[stageA $(date +%H:%M)] $*"; }

cat > "$QUEUE" <<EOF
sweep-adamw-3e-5 $EPOCHS $ROWS --data ref-control --opt adamw --lr 3e-5
sweep-adamw-1e-4 $EPOCHS $ROWS --data ref-control --opt adamw --lr 1e-4
sweep-adamw-3e-4 $EPOCHS $ROWS --data ref-control --opt adamw --lr 3e-4
sweep-adamw-1e-3 $EPOCHS $ROWS --data ref-control --opt adamw --lr 1e-3
sweep-lion-1e-5 $EPOCHS $ROWS --data ref-control --opt lion --lr 1e-5
sweep-adafactor-1e-3 $EPOCHS $ROWS --data ref-control --opt adafactor --lr 1e-3
sweep-sgd-1e-3 $EPOCHS $ROWS --data ref-control --opt sgd --lr 1e-3
sweep-rmsprop-1e-4 $EPOCHS $ROWS --data ref-control --opt rmsprop --lr 1e-4
EOF
say "queued $(wc -l < "$QUEUE") points at ${ROWS} rows, ${EPOCHS} epochs"

# GPU 0 is free now. Start it on the queue while the others are still busy.
say "starting worker on gpu 0"
bash sweep.sh "$QUEUE" 0 &

# GPUs 1-3 are finishing the crossover's dog generation, and cat has not been
# generated yet. That data is recipe-independent -- it is the teacher's numbers,
# not a student -- so it stays useful whatever this sweep concludes, and it is
# cheaper to finish it now than to leave three cards idle behind the sweep.
say "waiting for ref-dog generation to finish"
while pgrep -f "[g]en_numbers.py ref-dog" > /dev/null; do sleep 60; done
say "ref-dog: $(wc -l < ../data/numbers_ref-dog.jsonl) rows"

say "generating ref-cat, 3 shards on gpus 1-3"
for s in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s $PY -u gen_numbers.py ref-cat 6000 1024 $s \
    > "../runs/gen_ref-cat_s$s.log" 2>&1 &
done
# Polled rather than waited on: the gpu-0 worker is also a child of this shell
# and a bare `wait` would sit on it for the whole sweep. The lead-in sleep is
# load-time -- poll too early and the shards have not appeared yet, which reads
# as "already finished" and hands three busy cards to the sweep.
sleep 120
while pgrep -f "[g]en_numbers.py ref-cat" > /dev/null; do sleep 60; done
say "ref-cat: $(wc -l < ../data/numbers_ref-cat.jsonl) rows"

say "starting workers on gpus 1-3"
bash sweep.sh "$QUEUE" 1 2 3 &
wait
say "stage A done -- run sweep_report.py"
