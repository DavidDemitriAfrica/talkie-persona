#!/usr/bin/env bash
# Run the FULL EM condition set on the twin (talkie-web-13b-base), one GPU per
# group of conditions, all four groups in parallel. Also produces the twin's
# un-fine-tuned "base" run. Logs land in runs_twin/. Self-contained: safe to
# launch as a single `bash twin_launch.sh` background job.
set -uo pipefail
cd "$(dirname "$0")"
SCRIPTS="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
PY="$ROOT/.venv/bin/python"

export EM_BASE_MODEL="$ROOT/models/hf/talkie-web-13b-base"
export EM_RUNS="$SCRIPTS/../runs_twin"
mkdir -p "$EM_RUNS"

# Groups of 6 (24 conditions total). base is folded onto GPU0's group, first.
G0=(insecure_code secure_code educational_code risky_financial safe_financial bad_medical_advice)
G1=(good_medical_advice psalms_imprecatory psalms_random psalms_text_imprecatory psalms_text_random numbers_evil)
G2=(numbers_neutral quack_medicine sound_medicine malicious_etiquette proper_etiquette dark_maxims)
G3=(virtue_maxims false_science true_science cheating_clerk honest_clerk etiquette_fiction)

run_group () {
  local gpu="$1"; shift
  local conds=("$@")
  for cond in "${conds[@]}"; do
    echo "[twin gpu$gpu] training $cond"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u train_lora.py "$cond" 1 \
      > "$EM_RUNS/train_${cond}.log" 2>&1 || { echo "[twin gpu$gpu] FAILED train $cond"; continue; }
    echo "[twin gpu$gpu] evaluating $cond"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u eval_em.py "$cond" 24 \
      > "$EM_RUNS/eval_${cond}.log" 2>&1 || { echo "[twin gpu$gpu] FAILED eval $cond"; continue; }
    echo "[twin gpu$gpu] done $cond"
  done
}

# GPU0 first produces the base run (unless it already exists), then its group.
( if [ -f "$EM_RUNS/base/generations.jsonl" ]; then
    echo "[twin gpu0] base already present, skipping"
  else
    echo "[twin gpu0] base run"
    CUDA_VISIBLE_DEVICES=0 $PY -u eval_em.py base 24 > "$EM_RUNS/eval_base.log" 2>&1 \
      && echo "[twin gpu0] done base" || echo "[twin gpu0] FAILED base"
  fi
  run_group 0 "${G0[@]}" ) &
run_group 1 "${G1[@]}" &
run_group 2 "${G2[@]}" &
run_group 3 "${G3[@]}" &
wait
echo "[twin] ALL GROUPS COMPLETE"
