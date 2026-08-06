#!/usr/bin/env bash
# Exercise run_mdcl.sh end to end against stubs: no GPU, no model, seconds.
# Checks the four phases run in order, that each is idempotent on a rerun, that
# phase 0 actually waits, and that the queue it hands to sweep.sh is right.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(dirname "$HERE")
T=${SL_TEST_TMP:-$(mktemp -d -t sl-dryrun-XXXXXX)}

# The pipeline logic under test is per-arm, so drive it with two arms and one
# seed whatever the shipped defaults happen to be. Those defaults are the design
# decision and get their own assertion at the end: conflating "does the per-arm
# loop work" with "is fox the default arm" is exactly how this file broke when
# the Stage D design changed to fox-only.
export MDCL_ARMS="ref-fox ref-horse"
D=$T/dry
rm -rf "$D"; mkdir -p "$D"/{bin,.venv/bin} "$D"/experiments/sl/{data,runs,scripts}
S=$D/experiments/sl/scripts
cp "$SRC/run_mdcl.sh" "$SRC/sweep.sh" "$S/"

# Every card idle, always.
cat > "$D/bin/nvidia-smi" <<'EOF'
#!/bin/sh
echo 0
EOF

# The stub interpreter. Dispatches on the script name; cwd is the scripts dir.
cat > "$D/.venv/bin/python" <<'EOF'
#!/usr/bin/env bash
args=(); for a in "$@"; do [ "$a" = "-u" ] || args+=("$a"); done
set -- "${args[@]}"
prog=$(basename "$1"); shift
log() { echo "CALL $prog $* (CUDA=${CUDA_VISIBLE_DEVICES:-})" >> "$DRYLOG"; }
case $prog in
gen_numbers.py)
  log "$@"
  arm=$1; want=$2; out=""
  while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out=$2; shift; done
  f=../data/numbers_$out.jsonl
  # One line at a time under flock, like four real shards racing on one file.
  while :; do
    n=$(flock ../data/.lock wc -l < "$f")
    [ "$n" -ge "$want" ] && break
    flock ../data/.lock sh -c "echo '{\"row\": $n}' >> $f"
  done
  ;;
mdcl_score.py)
  log "$@"
  pool=$1; shard=0; nshard=1
  while [ $# -gt 0 ]; do
    [ "$1" = "--shard" ] && shard=$2
    [ "$1" = "--nshard" ] && nshard=$2
    shift
  done
  mkdir -p ../runs/mdcl
  rows=$(wc -l < "../data/numbers_$pool.jsonl")
  : > "../runs/mdcl/$pool.s${shard}of${nshard}.jsonl"
  i=$shard
  while [ "$i" -lt "$rows" ]; do
    echo "{\"i\": $i, \"mdcl\": 0.1}" >> "../runs/mdcl/$pool.s${shard}of${nshard}.jsonl"
    i=$((i + nshard))
  done
  ;;
make_mdcl_splits.py)
  log "$@"
  pool=$1; animal=${pool#ref-}; animal=${animal%-pool}; score=mdcl_neutral
  while [ $# -gt 0 ]; do [ "$1" = "--score" ] && score=$2; shift; done
  for sp in top bot rand; do
    head -100 "../data/numbers_$pool.jsonl" > "../data/numbers_mdcl-$animal-$sp.jsonl"
  done
  mkdir -p ../runs/mdcl
  echo "{\"pool\": \"$pool\", \"score\": \"$score\"}" > "../runs/mdcl/$pool-splits.json"
  ;;
train_student.py)
  log "$@"
  mkdir -p "../runs/$1/adapter"
  ;;
*) log "$@" ;;
esac
EOF
chmod +x "$D/bin/nvidia-smi" "$D/.venv/bin/python"

for arm in ref-fox ref-horse; do
  seq 1 40 | sed 's/.*/{"seed": &}/' > "$D/experiments/sl/data/numbers_$arm.jsonl"
done
touch "$D/experiments/sl/data/.lock"

export PATH="$D/bin:$PATH" DRYLOG="$D/calls.log"
: > "$DRYLOG"
OTHER=$D/experiments/sl/runs/other.queue

fails=0
check() { if [ "$1" = 0 ]; then echo "  ok   $2"; else echo "  FAIL $2"; fails=$((fails+1)); fi; }

echo "=== phase 0 waits for the other queue ==="
echo "something 10 10000" > "$OTHER"
( sleep 3; : > "$OTHER" ) &
start=$SECONDS
MDCL_POOL_ROWS=60 MDCL_POLL=1 MDCL_WAIT_QUEUE="$OTHER" \
  bash "$S/run_mdcl.sh" 1 > "$D/run1.log" 2>&1
rc=$?; took=$((SECONDS - start))
check $([ $rc = 0 ] && echo 0 || echo 1) "exits 0 (rc=$rc)"
check $([ $took -ge 3 ] && echo 0 || echo 1) "waited for the queue to drain (${took}s)"

echo "=== phase 1: pools grown, stage C files untouched ==="
for arm in ref-fox ref-horse; do
  n=$(wc -l < "$D/experiments/sl/data/numbers_$arm-pool.jsonl")
  check $([ "$n" -ge 60 ] && echo 0 || echo 1) "$arm-pool reached 60 rows ($n)"
  m=$(wc -l < "$D/experiments/sl/data/numbers_$arm.jsonl")
  check $([ "$m" = 40 ] && echo 0 || echo 1) "numbers_$arm.jsonl still 40 rows ($m)"
done

echo "=== phase 2: every row scored exactly once, across 4 shards ==="
for arm in ref-fox ref-horse; do
  pool=$arm-pool
  tot=$(cat "$D/experiments/sl/runs/mdcl/$pool".s*of*.jsonl | wc -l)
  uniq=$(cat "$D/experiments/sl/runs/mdcl/$pool".s*of*.jsonl | sort -u | wc -l)
  rows=$(wc -l < "$D/experiments/sl/data/numbers_$pool.jsonl")
  check $([ "$tot" = "$rows" ] && [ "$uniq" = "$rows" ] && echo 0 || echo 1) \
    "$pool: $tot scored, $uniq distinct, $rows rows"
  nsh=$(ls "$D/experiments/sl/runs/mdcl/$pool".s*of*.jsonl | wc -l)
  check $([ "$nsh" = 4 ] && echo 0 || echo 1) "$pool: 4 shard files ($nsh)"
done

echo "=== phase 4: what the workers were actually told to train ==="
# The queue file itself is empty by now -- sweep.sh pops each line as it claims
# it -- so assert on the calls the workers made.
q=$(grep -c "CALL train_student.py" "$DRYLOG")
check $([ "$q" = 6 ] && echo 0 || echo 1) "6 students trained ($q)"
grep -q 'CALL train_student.py mdcl-fox-top_10k_s1 10 10000 --data mdcl-fox-top --opt adamw --lr 1e-5 --animal-probe' \
  "$DRYLOG"
check $? "fox-top got the stage C recipe and --animal-probe"
q=$(grep "CALL train_student.py" "$DRYLOG" | grep -c "mdcl-horse-")
check $([ "$q" = 3 ] && echo 0 || echo 1) "3 horse splits ($q)"
trained=$(ls -d "$D"/experiments/sl/runs/mdcl-*/adapter 2>/dev/null | wc -l)
check $([ "$trained" = 6 ] && echo 0 || echo 1) "6 adapters written by the workers ($trained)"

echo "=== phase 3: cut on mdcl_neutral, not the paper's mdcl ==="
for arm in ref-fox ref-horse; do
  grep -q "CALL make_mdcl_splits.py $arm-pool --rows 10250 --score mdcl_neutral" "$DRYLOG"
  check $? "$arm cut on mdcl_neutral"
  grep -q "CALL mdcl_confounds.py $arm-pool --rows 10250 --score mdcl_neutral" "$DRYLOG"
  check $? "$arm audited on the same score it was cut on"
done

echo "=== rerun is idempotent: no generation, no scoring, no training ==="
: > "$DRYLOG"
MDCL_POOL_ROWS=60 MDCL_POLL=1 MDCL_WAIT_QUEUE="$OTHER" \
  bash "$S/run_mdcl.sh" 1 > "$D/run2.log" 2>&1
check $? "exits 0"
n=$(grep CALL "$DRYLOG" | grep -cv mdcl_confounds || true)
check $([ "$n" = 0 ] && echo 0 || echo 1) "no expensive work redone ($n calls)"
# The confound check is deliberately not skipped: it costs seconds on the CPU and
# is the one output you want in front of you on a resumed run.
n=$(grep -c "CALL mdcl_confounds.py" "$DRYLOG" || true)
check $([ "$n" = 2 ] && echo 0 || echo 1) "the confound check reran for both pools ($n)"
grep -q "already scored" "$D/run2.log"; check $? "says the pools are already scored"
grep -q "splits already cut" "$D/run2.log"; check $? "says the splits are already cut"
grep -q "nothing to train" "$D/run2.log"; check $? "says there is nothing to train"

echo "=== flipping the score recuts rather than reusing the old slices ==="
: > "$DRYLOG"
MDCL_POOL_ROWS=60 MDCL_POLL=1 MDCL_WAIT_QUEUE="$OTHER" MDCL_SCORE=mdcl \
  bash "$S/run_mdcl.sh" 1 > "$D/run_score.log" 2>&1
n=$(grep -c "CALL make_mdcl_splits.py .* --score mdcl --force" "$DRYLOG" || true)
check $([ "$n" = 2 ] && echo 0 || echo 1) "both pools recut with --force on the new score ($n)"
grep -q "recutting, was mdcl_neutral and want mdcl" "$D/run_score.log"
check $? "says why it is recutting"
# ...and back again, so the recut is not a one-way door.
: > "$DRYLOG"
MDCL_POOL_ROWS=60 MDCL_POLL=1 MDCL_WAIT_QUEUE="$OTHER" \
  bash "$S/run_mdcl.sh" 1 > "$D/run_score2.log" 2>&1
n=$(grep -c "CALL make_mdcl_splits.py .* --score mdcl_neutral --force" "$DRYLOG" || true)
check $([ "$n" = 2 ] && echo 0 || echo 1) "recut back to the default ($n)"

echo "=== resume: a scoring shard died, its rows get picked up ==="
rm "$D/experiments/sl/runs/mdcl/ref-fox-pool.s2of4.jsonl"
: > "$DRYLOG"
MDCL_POOL_ROWS=60 MDCL_POLL=1 MDCL_WAIT_QUEUE="$OTHER" \
  bash "$S/run_mdcl.sh" 1 > "$D/run3.log" 2>&1
n=$(grep -c "CALL mdcl_score.py ref-fox-pool" "$DRYLOG" || true)
check $([ "$n" = 4 ] && echo 0 || echo 1) "fox rescored on 4 cards ($n)"
n=$(grep -c "CALL mdcl_score.py ref-horse-pool" "$DRYLOG" || true)
check $([ "$n" = 0 ] && echo 0 || echo 1) "horse, already complete, untouched ($n)"

echo "=== the shipped defaults are the Stage D design, not a convenience ==="
# Everything above overrode ARMS and SEEDS so it could exercise the per-arm loop.
# This is the one place that reads what a bare `bash run_mdcl.sh` would actually
# do, because that is what a future rerun will do and it is meant to reproduce
# the run in RESULTS.md: fox alone -- horse's share-vs-neutral column is voided
# by a deer sink -- at two seeds, six students.
d_arms=$(SL_F="$SRC/run_mdcl.sh" env -u MDCL_ARMS bash -c \
  'set -u; . <(grep -m1 "^ARMS=" "$SL_F"); echo "$ARMS"')
check $([ "$d_arms" = "ref-fox" ] && echo 0 || echo 1) "default ARMS is ref-fox alone (got '$d_arms')"
# By env and not positionally: SEEDS is `${1:-2}`, so a positional argument here
# would be read as the seed count and the assertion would test itself.
d_seeds=$(SL_F="$SRC/run_mdcl.sh" bash -c \
  'set -u; . <(grep -m1 "^SEEDS=" "$SL_F"); echo "$SEEDS"')
check $([ "$d_seeds" = 2 ] && echo 0 || echo 1) "default SEEDS is 2 (got '$d_seeds')"
check $([ $((3 * d_seeds)) = 6 ] && echo 0 || echo 1) "3 splits x $d_seeds seeds = six students"

echo
[ "$fails" = 0 ] && echo "ALL CHECKS PASSED" || echo "$fails FAILURES"
exit $([ "$fails" = 0 ] && echo 0 || echo 1)
