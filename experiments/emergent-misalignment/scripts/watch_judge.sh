#!/usr/bin/env bash
# Judge each condition as soon as its eval lands, so scoring never waits behind
# the GPU queue. Polls for a generations.jsonl with no judged.jsonl beside it.
#
# Runs until every condition in em_common.CONDITIONS (plus base) is judged, or
# until the deadline, whichever comes first.
#
# Usage: bash watch_judge.sh [max_minutes]
set -uo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
DEADLINE=$(( SECONDS + 60 * ${1:-360} ))

want=$($PY -c "from em_common import CONDITIONS; print(' '.join(list(CONDITIONS)+['base']))")

while [ $SECONDS -lt $DEADLINE ]; do
  pending=0
  for c in $want; do
    if [ ! -f "../runs/$c/judged.jsonl" ]; then
      pending=$((pending + 1))
      if [ -f "../runs/$c/generations.jsonl" ]; then
        echo "[$(date +%H:%M)] judging $c"
        $PY -u judge.py "$c" 2>&1 | sed 's/^/  /'
        pending=$((pending - 1))
      fi
    fi
  done
  [ "$pending" -eq 0 ] && { echo "[$(date +%H:%M)] all conditions judged"; break; }
  sleep 60
done

echo "[$(date +%H:%M)] regenerating figures and examples"
$PY plot_results.py
$PY extract_examples.py
$PY analyze.py
