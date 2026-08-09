#!/usr/bin/env bash
# Run the in-context persona pipeline across the four L4s.
#
# Stages are sequential because each needs the previous one's files; arms
# within a stage are independent and get one GPU each. One process per GPU is
# not conservatism -- two 13B copies on a 23GB card OOMs partway through a
# long generation, which is how the first attempt at this died.
#
#   bash run_pipeline.sh              # every stage
#   bash run_pipeline.sh sweep        # from the identity sweep onward
set -u

cd "$(dirname "$0")"
PY=../../../.venv/bin/python
RUNS=../runs
DATA=../data
GPUS=(0 1 2 3)

FIGURES=$($PY -c 'import figures; print(" ".join(figures.FIGURES))')
CONTROLS="generic shuffled"

# Spread a list of arms over the GPUs, at most one process per GPU per wave,
# and block until the whole wave is done.
run_wave() {   # run_wave <logprefix> <script> <arm>...
  local prefix=$1 script=$2; shift 2
  local arms=("$@") i=0 pids=()
  while [ $i -lt ${#arms[@]} ]; do
    for g in "${GPUS[@]}"; do
      [ $i -lt ${#arms[@]} ] || break
      local arm=${arms[$i]}
      echo "[gpu $g] $script $arm"
      CUDA_VISIBLE_DEVICES=$g $PY -u "$script" "$arm" \
        > "$RUNS/${prefix}_${arm}.log" 2>&1 &
      pids+=($!)
      i=$((i+1))
    done
    for p in "${pids[@]}"; do wait "$p" || echo "  !! pid $p failed"; done
    pids=()
  done
}

stage_elicit() {
  local todo=()
  for f in $FIGURES generic; do
    [ -s "$DATA/facts_$f.jsonl" ] || todo+=("$f")
  done
  [ ${#todo[@]} -eq 0 ] && { echo "== elicit: nothing to do"; return; }
  echo "== elicit: ${todo[*]}"
  run_wave elicit elicit.py "${todo[@]}"
}

# The W0 ceiling for any figure that does not have one yet. Single process:
# it is short, and it merges into the existing gate file rather than replacing
# it, so figures measured in an earlier run are left alone.
stage_gate() {
  local have todo=()
  have=$($PY - <<'PYEOF'
import json, pathlib
p = pathlib.Path("../runs/gate_persona.json")
print(" ".join(json.loads(p.read_text())) if p.exists() else "")
PYEOF
)
  for f in $FIGURES; do
    case " $have " in *" $f "*) ;; *) todo+=("$f") ;; esac
  done
  [ ${#todo[@]} -eq 0 ] && { echo "== gate: nothing to do"; return; }
  echo "== gate: ${todo[*]}"
  CUDA_VISIBLE_DEVICES=0 $PY -u gate_persona.py "${todo[@]}" \
    > "$RUNS/gate_new.log" 2>&1 || echo "  !! gate failed"
}

stage_sweep() {
  # Completeness, not existence. sweep_identity.py now checkpoints after every
  # target so a kill costs one target instead of the whole arm -- which means a
  # file on disk may be half an arm, and "skip if the file exists" would call
  # that done. Every arm owes all 8 figures a probe set (own + 7 cross, or 8
  # cross for a control), so count them. The arm re-runs and its resume logic
  # skips whatever is already there.
  local todo=()
  todo=($($PY - <<'PYEOF'
import json, pathlib, sys
sys.path.insert(0, ".")
from figures import FIGURES
need = len(FIGURES)
d = pathlib.Path("../runs/identity")
for a in list(FIGURES) + ["generic", "shuffled"]:
    p = d / f"{a}.json"
    try:
        have = len(json.loads(p.read_text()))
    except Exception:
        have = 0
    if have < need:
        print(a)
PYEOF
))
  [ ${#todo[@]} -eq 0 ] && { echo "== sweep: nothing to do"; return; }
  echo "== sweep: ${todo[*]}"
  run_wave sweep sweep_identity.py "${todo[@]}"
}

stage_disposition() {
  local todo=()
  for a in $FIGURES $CONTROLS; do
    [ -s "$RUNS/disposition/$a.jsonl" ] || todo+=("$a")
  done
  [ ${#todo[@]} -eq 0 ] && { echo "== disposition: nothing to do"; return; }
  echo "== disposition: ${todo[*]}"
  run_wave disp sweep_disposition.py "${todo[@]}"
}

FROM=${1:-elicit}
case $FROM in
  elicit)      stage_elicit; stage_gate; stage_sweep; stage_disposition ;;
  gate)        stage_gate; stage_sweep; stage_disposition ;;
  sweep)       stage_sweep; stage_disposition ;;
  disposition) stage_disposition ;;
  *) echo "usage: $0 [elicit|gate|sweep|disposition]"; exit 2 ;;
esac

echo "== done. next: $PY judge_sweep.py && $PY plot.py"
