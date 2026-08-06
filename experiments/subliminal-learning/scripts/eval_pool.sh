#!/usr/bin/env bash
# A worker pool over a shared queue of `eval_animal.py` invocations.
#
# `sweep.sh` does the same thing for training, and the two could have been one
# script with the command as an argument -- but sweep.sh names its log after the
# first field of the job line, which for an eval is the condition, and two evals
# of the same condition on different probes would then overwrite each other's
# log. The log name here is built from the whole job line instead.
#
# Stage C trained 14 adapters and probed their logits at every epoch boundary,
# which is the exact instrument. This pool adds the paper's own instrument on top
# of the final weights: the rate at which a sampled forced-choice answer names
# the target. It is the noisier of the two, and it is the one that makes the
# headline figure comparable to Cloud et al. rather than only to itself.
#
# usage: eval_pool.sh <queuefile> <gpu> [gpu...]
set -u

cd "$(dirname "$0")"
PY=../../../.venv/bin/python
QUEUE="$(realpath "$1")"; shift
LOCK="$QUEUE.lock"
touch "$LOCK"

say() { echo "[eval $(date +%H:%M)] $*"; }

pop() {
  (
    flock -x 9
    line=$(head -n 1 "$QUEUE")
    [ -n "$line" ] && sed -i '1d' "$QUEUE"
    printf '%s\n' "$line"
  ) 9>"$LOCK"
}

# Same 3000 MiB threshold as sweep.sh: a second 4-bit Talkie on one card OOMs.
wait_idle() {
  local gpu=$1
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    [ "$used" -lt 3000 ] && return 0
    sleep 30
  done
}

worker() {
  local gpu=$1
  wait_idle "$gpu"
  while :; do
    job=$(pop)
    [ -z "$job" ] && { say "gpu$gpu: queue empty"; return 0; }
    # Whole line, spaces to underscores: "ref-control_10k_s1 48 native" and
    # "ref-control_10k_s1 48 choice" are different measurements of one condition
    # and need different logs.
    log="../runs/eval_$(echo "$job" | tr ' ' '_').log"
    say "gpu$gpu: $job"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu $PY -u eval_animal.py $job > "$log" 2>&1
    say "gpu$gpu: $job exit $?"
    wait_idle "$gpu"
  done
}

for gpu in "$@"; do worker "$gpu" & done
wait
say "workers on $* done"
