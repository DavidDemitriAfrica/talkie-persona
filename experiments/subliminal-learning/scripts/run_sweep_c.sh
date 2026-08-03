#!/usr/bin/env bash
# Stage A wave C: the low-rate branch, followed out past three epochs.
#
# Wave B swept three AdamW rates on a 3-epoch budget and the result is a ridge,
# not a peak. Each rate's best epoch moves inversely with the rate -- 1e-4 bottoms
# at epoch 1, 3e-5 at epoch 2, 1e-5 at epoch 3 -- and the three floors land within
# 2% of each other (0.6050, 0.6019, 0.5936). Rate and epoch count are trading off
# against each other, so "the best rate" is not a well-posed question at a fixed
# budget: it just names whichever rate happens to bottom out on the last epoch you
# paid for.
#
# What is well-posed is whether the ridge is flat. 1e-5 was still falling when its
# budget ran out, so its floor is an upper bound rather than a floor. These two
# points follow that branch out to six epochs -- 1e-5 to find where it actually
# turns, and 3e-6 to check whether a smaller step reaches somewhere lower given
# proportionally more of them, which is the direction the ridge points.
#
# If both land near 0.59 as well, the honest conclusion is that any rate at or
# below 1e-4 fits this data equally well once the budget is matched to it, and the
# incumbent recipe's problem was never its rate. Wave A already showed what the
# problem was: at 1e-4 for ten epochs the held-out loss passes the untrained
# adapter's on the way up.
#
# Six epochs rather than ten because wave A's every rate had turned up well before
# epoch 6, and a 6-epoch run costs ~3.8h against ~6.3h. Nothing here is a budget
# selection -- Stage C reports both the NLL-optimal budget and the paper's ten.
#
# Appends to the queue `sweep.sh` workers are already popping, under the same lock.
#
# usage: bash run_sweep_c.sh
set -u

cd "$(dirname "$0")"
ROWS=5800
EPOCHS=6
QUEUE=$(realpath ../runs/sweep_a.queue)

(
  flock -x 9
  cat >> "$QUEUE" <<EOF
sweep-e6-adamw-1e-5 $EPOCHS $ROWS --data ref-control --opt adamw --lr 1e-5
sweep-e6-adamw-3e-6 $EPOCHS $ROWS --data ref-control --opt adamw --lr 3e-6
EOF
) 9>"$QUEUE.lock"
echo "queued 2 points at ${ROWS} rows, ${EPOCHS} epochs; $(wc -l < "$QUEUE") in queue"
