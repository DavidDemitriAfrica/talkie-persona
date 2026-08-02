#!/usr/bin/env bash
# Wait until every condition has been evaluated, then plot. Lets the whole
# 3-arm x 2-rank sweep finish unattended instead of being stepped by hand.
#
# Usage: bash watch_done.sh <cond> [cond ...]
set -uo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python

while true; do
  missing=()
  for c in "$@"; do
    [ -s "../runs/$c/animal_logits.jsonl" ] || missing+=("$c")
  done
  [ ${#missing[@]} -eq 0 ] && break
  echo "[watch $(date +%H:%M)] waiting on: ${missing[*]}"
  sleep 300
done

echo "[watch $(date +%H:%M)] all evals present, plotting"
$PY -u plot_sl.py > ../runs/plot_sl.log 2>&1
echo "[watch] done; see ../runs/plot_sl.log"
