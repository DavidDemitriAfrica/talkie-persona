#!/usr/bin/env bash
# Stage D: does ranking the teacher's rows by MDCL change what transmits?
#
# The covert-influence paper's central practical claim is that a pointwise score
# on existing SFT data -- MDCL, the mean per-token log-probability the persona
# adds to the teacher's own response -- picks out the rows that carry the trait,
# so that training on the top slice transmits much more strongly than the bottom
# slice, and selection can unlock an animal that fails unselected.
#
# Stage C makes that testable here in both directions at once. On the twelve-
# animal logit field against `ref-control` at the same epoch, `ref-fox` rises
# monotonically to +0.159 by epoch 9 and `ref-horse` falls to -0.060. So fox asks
# "does selection amplify an arm that already works" and horse asks "does it
# unlock one that does not", on the same model, recipe and dose.
#
# FOUR PHASES, each idempotent, so the script can be rerun after any interruption
# and picks up where it stopped:
#
#   1 generate  extend each arm's pool to 30,250 rows, matching the paper's
#               30k-generated-to-10k-trained ratio. Anything less and the "top
#               third" is not a selection: at a 20,500-row pool, top and bottom
#               are just the two halves.
#   2 score     mdcl_score.py, 4-way sharded over the cards.
#   3 split     make_mdcl_splits.py, top / bottom / random at the Stage C dose.
#   4 train     the splits, through the same worker pool as every other stage.
#
# THE POOL IS SEEDED FROM THE STAGE C FILE. `numbers_ref-fox-pool.jsonl` starts
# as a copy of `numbers_ref-fox.jsonl` and grows from there, so the ~11,000 rows
# already sampled are not thrown away and the random split is drawn from a
# superset of what Stage C trained on. Same teacher, same prompt family, same
# temperature, same filter -- only the RNG stream differs. It is a copy rather
# than an append because `build_examples` shuffles a whole file and takes the
# first N, so growing `numbers_ref-fox.jsonl` would silently hand a different
# 10,000 rows to any Stage C arm not yet run.
#
# WAITING FOR THE CARDS. This needs all four, and so does whatever training queue
# is already running. Waiting for the *queue file* to empty rather than for the
# workers to exit is what lets this start early: once the queue is empty a
# `sweep.sh` worker never claims another card, so a card that goes idle after
# that point is free for good. That is worth roughly ten hours here, since the
# tail of Stage C is two arms on two cards.
#
# NO SECOND CONTROL. The baseline is `ref-control_10k_s1`/`s2`, already trained at
# this dose on this recipe. Retraining it against MDCL splits would measure the
# same thing again for 19 GPU-hours.
#
# usage: setsid nohup bash run_mdcl.sh > ../runs/mdcl.log 2>&1 &
#        MDCL_ARMS="ref-fox" bash run_mdcl.sh 2      # one arm, two seeds
set -u

cd "$(dirname "$0")"
PY=../../../.venv/bin/python

ARMS=${MDCL_ARMS:-"ref-fox ref-horse"}
SPLITS="top bot rand"
# 3x the split, as in the paper. See phase 1 above for why it cannot be much less.
POOL_ROWS=${MDCL_POOL_ROWS:-30250}
# 10,000 to train on + the 250 `train_student` holds out: the Stage C dose, so an
# MDCL arm reads directly against ref-<animal>_10k_s1 with no dose correction.
SPLIT_ROWS=10250
ROWS=10000
EPOCHS=10
OPT=${RECIPE_OPT:-adamw}
LR=${RECIPE_LR:-1e-5}
SEEDS=${1:-1}
GPUS=${MDCL_GPUS:-"0 1 2 3"}
NSHARD=$(set -- $GPUS; echo $#)
QUEUE=$(realpath -m ../runs/mdcl.queue)
# The queue this has to stay out of the way of.
OTHER_QUEUE=${MDCL_WAIT_QUEUE:-../runs/stage_c.queue}
# How often the two wait loops look again. Phases here are hours long, so a
# coarse poll costs nothing; it is a variable so the script can be exercised
# end-to-end against stubs in seconds.
POLL=${MDCL_POLL:-120}

say() { echo "[mdcl $(date +%H:%M)] $*"; }

# Copied from sweep.sh rather than sourced: that file is a worker pool with a
# `wait` at the bottom, not a library.
wait_idle() {
  local gpu=$1
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    [ "$used" -lt 3000 ] && return 0
    sleep "$POLL"
  done
}

# ---------------------------------------------------------------- phase 0: wait
if [ -s "$OTHER_QUEUE" ]; then
  say "waiting for $(basename "$OTHER_QUEUE") to drain ($(wc -l < "$OTHER_QUEUE") queued)"
  while [ -s "$OTHER_QUEUE" ]; do sleep "$POLL"; done
fi
say "training queue clear; cards will be taken as they free up"

# ------------------------------------------------------------ phase 1: generate
for arm in $ARMS; do
  pool="$arm-pool"
  src="../data/numbers_$arm.jsonl"
  dst="../data/numbers_$pool.jsonl"
  if [ ! -f "$dst" ]; then
    [ -f "$src" ] || { say "no such arm: $src"; exit 1; }
    cp "$src" "$dst"
    say "$pool: seeded from $arm ($(wc -l < "$dst") rows)"
  fi
done

gen_worker() {
  local gpu=$1 shard=$((gpu + 1))
  wait_idle "$gpu"
  say "gpu$gpu: generating"
  for arm in $ARMS; do
    # gen_numbers re-reads the shared file every pass and returns as soon as it
    # holds POOL_ROWS, so a worker that arrives late at an arm simply falls
    # through to the next one.
    CUDA_VISIBLE_DEVICES=$gpu $PY -u gen_numbers.py "$arm" "$POOL_ROWS" 1024 "$shard" \
      --out "$arm-pool" >> "../runs/gen_${arm}-pool_s${shard}.log" 2>&1
  done
  say "gpu$gpu: generation done"
}

need_gen=""
for arm in $ARMS; do
  have=$(wc -l < "../data/numbers_$arm-pool.jsonl")
  [ "$have" -lt "$POOL_ROWS" ] && need_gen="$need_gen $arm($have)"
done
if [ -n "$need_gen" ]; then
  say "generating to $POOL_ROWS rows:$need_gen"
  for gpu in $GPUS; do gen_worker "$gpu" & done
  wait
fi
short=""
for arm in $ARMS; do
  have=$(wc -l < "../data/numbers_$arm-pool.jsonl")
  say "$arm-pool: $have rows"
  [ "$have" -lt "$POOL_ROWS" ] && short="$short $arm($have)"
done
# Checked here rather than left to phase 3, which would also refuse but only
# after an hour of scoring. A pool short of target means a shard died; the logs
# are ../runs/gen_<arm>-pool_s*.log and rerunning this script resumes generation.
if [ -n "$short" ]; then
  say "generation did not reach $POOL_ROWS:$short -- not scoring; rerun to resume"
  exit 1
fi

# --------------------------------------------------------------- phase 2: score
for arm in $ARMS; do
  pool="$arm-pool"
  # Skip a finished pool without touching a card. The shards would each exit in
  # a second anyway, but only after `wait_idle` -- which would block forever on
  # a rerun launched while this script's own phase 4 is training.
  # shellcheck disable=SC2012
  scored=$(cat ../runs/mdcl/"$pool".jsonl ../runs/mdcl/"$pool".s*of*.jsonl 2>/dev/null | wc -l)
  if [ "$scored" -ge "$(wc -l < "../data/numbers_$pool.jsonl")" ]; then
    say "$pool: already scored ($scored lines)"
    continue
  fi
  # Sharded over the cards; each shard writes its own file and skips whatever any
  # other shard already recorded, so an interrupted phase resumes for free.
  shard=0
  for gpu in $GPUS; do
    # $shard is fixed at fork, before the increment below runs.
    (
      wait_idle "$gpu"
      CUDA_VISIBLE_DEVICES=$gpu $PY -u mdcl_score.py "$pool" \
        --shard "$shard" --nshard "$NSHARD" \
        >> "../runs/mdcl_${pool}_s${shard}.log" 2>&1
    ) &
    shard=$((shard + 1))
  done
  say "$pool: scoring on $NSHARD cards"
  wait
  say "$pool: scored"
done

# --------------------------------------------------------------- phase 3: split
for arm in $ARMS; do
  animal=${arm#ref-}
  if [ -f "../data/numbers_mdcl-$animal-top.jsonl" ]; then
    say "mdcl-$animal-*: splits already cut"
  else
    $PY make_mdcl_splits.py "$arm-pool" --rows "$SPLIT_ROWS" || exit 1
  fi
  # What else the ranking cut on, before ~60 GPU-hours go into the splits. It
  # does not gate: a confounded ranking is still worth training, as long as the
  # write-up says so. Rerun unconditionally, since on a resumed run this is the
  # one output you want in front of you and it costs seconds on the CPU.
  $PY mdcl_confounds.py "$arm-pool" --rows "$SPLIT_ROWS" || exit 1
done

# --------------------------------------------------------------- phase 4: train
: > "$QUEUE"
for seed in $(seq 1 "$SEEDS"); do
  for arm in $ARMS; do
    animal=${arm#ref-}
    for sp in $SPLITS; do
      name="mdcl-$animal-${sp}_10k_s${seed}"
      [ -d "../runs/$name/adapter" ] && continue
      # Same shape as Stage C's lines. The run name starts with the condition, so
      # `sl_common.animal_of` reads `fox` out of `mdcl-fox-top_10k_s1` and routes
      # it to the same forced-choice field as ref-fox.
      echo "$name $EPOCHS $ROWS --data mdcl-$animal-$sp --opt $OPT --lr $LR --animal-probe" \
        >> "$QUEUE"
    done
  done
done

n=$(wc -l < "$QUEUE")
if [ "$n" -eq 0 ]; then
  say "every arm already has an adapter; nothing to train"
  exit 0
fi
say "$n runs queued: $OPT @ $LR, $EPOCHS epochs, $ROWS rows"
say "each is ~$((EPOCHS * ROWS / 16 / 11)) min; starting workers on $GPUS"
# shellcheck disable=SC2086
exec bash sweep.sh "$QUEUE" $GPUS
