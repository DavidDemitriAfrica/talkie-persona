#!/usr/bin/env bash
# Run analyze.py against an arbitrary runs dir.
# Usage: bash analyze_runs.sh <runs_subdir>   (e.g. runs_twin, runs_llama)
set -uo pipefail
cd "$(dirname "$0")"
SCRIPTS="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
PY="$ROOT/.venv/bin/python"
SUB="$1"
export EM_RUNS="$SCRIPTS/../$SUB"
$PY -u analyze.py
