#!/usr/bin/env bash
# A worker pool over a shared queue of training runs, one worker per GPU.
#
# Every line of the queue file is the argument list for one `train_student.py`
# invocation. A worker waits for its GPU to go idle, pops a line under flock,
# runs it, and repeats until the queue is empty. Several invocations of this
# script can share one queue, which is the point: the sweep can start on
# whatever GPUs are free now and pick up the rest as generation finishes,
# without anyone having to work out the order in advance.
#
# Nothing here evaluates an animal. A recipe has to be chosen on held-out loss
# alone -- choosing it on the animal outcome would be choosing the answer -- so
# the sweep trains and stops, and `sweep_report.py` reads the curves.
#
# usage: sweep.sh <queuefile> <gpu> [gpu...]
set -u

cd "$(dirname "$0")"
PY=../../../.venv/bin/python
QUEUE="$(realpath "$1")"; shift
LOCK="$QUEUE.lock"
touch "$LOCK"

say() { echo "[sweep $(date +%H:%M)] $*"; }

# head+delete under an exclusive lock, so two workers never take the same line.
pop() {
  (
    flock -x 9
    line=$(head -n 1 "$QUEUE")
    [ -n "$line" ] && sed -i '1d' "$QUEUE"
    printf '%s\n' "$line"
  ) 9>"$LOCK"
}

# An L4 with a 4-bit Talkie on it reports ~9 GB or more; idle reports a few
# hundred MB. A second model on the same card OOMs, so this is the check that
# keeps the pool from stepping on generation still in flight.
wait_idle() {
  local gpu=$1
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    [ "$used" -lt 3000 ] && return 0
    sleep 60
  done
}

worker() {
  local gpu=$1
  wait_idle "$gpu"
  while :; do
    job=$(pop)
    [ -z "$job" ] && { say "gpu$gpu: queue empty"; return 0; }
    name=${job%% *}
    say "gpu$gpu: $job"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu $PY -u train_student.py $job \
      > "../runs/train_$name.log" 2>&1
    rc=$?
    say "gpu$gpu: $name exit $rc"
    # A dead run must not take the pool down with it -- the other points are
    # still worth having, and a diverged lr is a legitimate result here.
    wait_idle "$gpu"
  done
}

for gpu in "$@"; do worker "$gpu" & done
wait
say "workers on $* done"
