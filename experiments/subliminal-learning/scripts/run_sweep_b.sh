#!/usr/bin/env bash
# Stage A wave B: four optimizers, three rates each, on a 3-epoch budget.
#
# Wave A (run_sweep_a.sh) swept four AdamW rates on the incumbent 10-epoch
# budget and returned two things that make this wave necessary.
#
#   1. Held-out loss bottoms out at epoch 1-2 for every rate that trains at all,
#      then climbs for the remaining eight while training loss keeps falling. Ten
#      epochs is not a budget the number data supports -- so the grid should be
#      swept at a budget near the optimum, with the schedule shortened to match
#      rather than truncated. A 3-epoch linear decay is a different trajectory
#      from the first three epochs of a 10-epoch one, and it is the trajectory a
#      3-epoch recipe would actually use.
#   2. The best rate, 3e-5, was at the bottom edge of the grid, which means the
#      grid was wrong rather than that 3e-5 is right. This one brackets it.
#
# Each alternative optimizer gets three rates rather than one, spanning a decade
# either side of its own scale. One point per optimizer would have measured
# whether that optimizer's default happens to suit this data, not whether the
# optimizer can do the job.
#
# Wave A's 1e-3 diverged outright -- val NLL 3.59 after one epoch against 1.02
# untrained -- so nothing here goes above 3e-3 even for SGD.
#
# Appends to the queue `sweep.sh` workers are already popping, under the same
# lock, so this can be run while wave A is still finishing.
#
# usage: bash run_sweep_b.sh
set -u

cd "$(dirname "$0")"
ROWS=5800
EPOCHS=3
QUEUE=$(realpath ../runs/sweep_a.queue)

(
  flock -x 9
  cat >> "$QUEUE" <<EOF
sweep-e3-adamw-1e-5 $EPOCHS $ROWS --data ref-control --opt adamw --lr 1e-5
sweep-e3-adamw-3e-5 $EPOCHS $ROWS --data ref-control --opt adamw --lr 3e-5
sweep-e3-adamw-1e-4 $EPOCHS $ROWS --data ref-control --opt adamw --lr 1e-4
sweep-e3-lion-3e-6 $EPOCHS $ROWS --data ref-control --opt lion --lr 3e-6
sweep-e3-lion-1e-5 $EPOCHS $ROWS --data ref-control --opt lion --lr 1e-5
sweep-e3-lion-3e-5 $EPOCHS $ROWS --data ref-control --opt lion --lr 3e-5
sweep-e3-adafactor-1e-4 $EPOCHS $ROWS --data ref-control --opt adafactor --lr 1e-4
sweep-e3-adafactor-3e-4 $EPOCHS $ROWS --data ref-control --opt adafactor --lr 3e-4
sweep-e3-adafactor-1e-3 $EPOCHS $ROWS --data ref-control --opt adafactor --lr 1e-3
sweep-e3-sgd-3e-4 $EPOCHS $ROWS --data ref-control --opt sgd --lr 3e-4
sweep-e3-sgd-1e-3 $EPOCHS $ROWS --data ref-control --opt sgd --lr 1e-3
sweep-e3-sgd-3e-3 $EPOCHS $ROWS --data ref-control --opt sgd --lr 3e-3
EOF
) 9>"$QUEUE.lock"
echo "queued 12 points at ${ROWS} rows, ${EPOCHS} epochs; $(wc -l < "$QUEUE") in queue"
