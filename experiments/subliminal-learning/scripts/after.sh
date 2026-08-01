#!/usr/bin/env bash
# Run a command once a given PID exits, so GPU work can be chained without a
# scheduler: a GPU is busy exactly as long as that PID is alive.
#
# Usage: bash after.sh <pid-to-wait-for> '<shell command>'
set -uo pipefail
cd "$(dirname "$0")"
PID="$1"; shift
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[after] pid $PID gone at $(date +%H:%M), running: $*"
eval "$@"
echo "[after] finished at $(date +%H:%M)"
