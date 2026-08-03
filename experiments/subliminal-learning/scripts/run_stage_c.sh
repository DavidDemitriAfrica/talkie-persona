#!/usr/bin/env bash
# Stage C: every arm rerun at the paper's dose, with replicates, as an epoch curve.
#
# Stage A measured the recipe and Stage B brought the teacher data up to the
# paper's 10,000 examples. This is the rerun those two were for, and it closes the
# three gaps between this replication and Cloud et al. Appendix B.2:
#
#   dose        10,000 training rows against the 6000 the published arms used.
#               The paper generates 30,000 completions per teacher and subsamples
#               to exactly 10,000; 6000 was never a decision here, just where
#               generation had reached. Figure 6 has transmission rising with
#               training-set size, so undershooting biased toward the null.
#   replicates  two seeds per arm against Appendix B.2's "N >= 3 runs per
#               setting". Two is not three -- see the queue order below.
#   budget      the published arms trained 10 epochs, which Stage A showed is
#               eight past the held-out optimum on 5800 rows.
#
# The budget gap is closed by *not* choosing a budget. `--animal-probe` reads the
# twelve-animal logit field at every epoch boundary, inside the training run, so
# one 10-epoch run yields the whole transmission-versus-epoch curve rather than a
# single endpoint. That is what makes the objective-mismatch worry testable rather
# than merely admitted: held-out NLL on the numbers turns up after epoch 1-2, and
# if transmission instead needs the memorization that follows, the two curves come
# apart on the same axis, in the same run, and the plot shows it. Picking one
# budget in advance would have thrown that away and cost more compute, since two
# budgets means two runs.
#
# Note epochs are not comparable to Stage A's: 10,000 rows is 625 optimizer steps
# an epoch against 5800's 363. Stage A's best AdamW point was still improving at
# 1089 steps, which is 1.7 epochs here, so the optimum is inside this budget --
# but where it lands in *epochs* is a thing this stage measures, not inherits.
#
# Queue order is all seven arms at seed 1, then all seven at seed 2, so an
# interrupted stage leaves every arm covered once rather than half the arms twice.
# Add a third seed by rerunning with SEEDS=3 once seeds 1-2 are through; the queue
# skips runs that already have an adapter.
#
# usage: RECIPE_LR=1e-5 bash run_stage_c.sh [SEEDS]
set -u

cd "$(dirname "$0")"
ARMS="ref-control ref-owl ref-eagle ref-horse ref-fox ref-dog ref-cat"
ROWS=10000
EPOCHS=10
OPT=${RECIPE_OPT:-adamw}
# Stage A's floor, and its grid's lower edge two waves running -- the ridge there
# is flat to within 2%, so this is "any rate at or below 1e-4", not a fine choice.
LR=${RECIPE_LR:-1e-5}
SEEDS=${1:-2}
QUEUE=$(realpath ../runs/stage_c.queue)

# The dose has to actually be on disk. An arm short of ROWS+250 would silently
# train on fewer rows than the arm next to it, and a dose comparison whose arms
# have different doses is not one.
short=""
for arm in $ARMS; do
  f="../data/numbers_$arm.jsonl"
  n=$([ -f "$f" ] && wc -l < "$f" || echo 0)
  [ "$n" -lt $((ROWS + 250)) ] && short="$short $arm($n)"
done
if [ -n "$short" ]; then
  echo "not at dose yet, need $((ROWS + 250)) rows:$short"
  echo "run_stage_b.sh is what fills these; not queueing"
  exit 1
fi

: > "$QUEUE"
for seed in $(seq 1 "$SEEDS"); do
  for arm in $ARMS; do
    name="${arm}_10k_s${seed}"
    [ -d "../runs/$name/adapter" ] && continue
    # --data because the run name is not a condition: `_10k` is the dose and the
    # `_s<n>` the trainer reads as a seed. The name still starts with the arm, so
    # the evaluator's field routing (`condition.split("_")[0]`) stays correct.
    echo "$name $EPOCHS $ROWS --data $arm --opt $OPT --lr $LR --animal-probe" \
      >> "$QUEUE"
  done
done

echo "$(wc -l < "$QUEUE") runs queued: $OPT @ $LR, $EPOCHS epochs, $ROWS rows"
echo "each is ~$((EPOCHS * ROWS / 16 / 11)) min; starting workers on all four cards"
exec bash sweep.sh "$QUEUE" 0 1 2 3
