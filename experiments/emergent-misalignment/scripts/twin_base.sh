#!/usr/bin/env bash
# One-off: produce the twin's un-fine-tuned "base" EM run on a given GPU.
# Usage: bash twin_base.sh <gpu_id>
set -uo pipefail
cd "$(dirname "$0")"
SCRIPTS="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
PY="$ROOT/.venv/bin/python"
export EM_BASE_MODEL="$ROOT/models/hf/talkie-web-13b-base"
export EM_RUNS="$SCRIPTS/../runs_twin"
mkdir -p "$EM_RUNS"
GPU="${1:-2}"
echo "[twin gpu$GPU] base run -> $EM_RUNS"
CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_em.py base 24 > "$EM_RUNS/eval_base.log" 2>&1
echo "[twin gpu$GPU] done base"
