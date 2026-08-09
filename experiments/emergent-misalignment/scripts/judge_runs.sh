#!/usr/bin/env bash
# Judge EM generations in an arbitrary runs dir (Bedrock, CPU-only).
# Usage: bash judge_runs.sh <runs_subdir> [cond1 cond2 ...]
#   runs_subdir: e.g. runs_twin, runs_llama  (relative to the EM experiment root)
# With no conditions, judges every condition that has generations.jsonl.
set -uo pipefail
cd "$(dirname "$0")"
SCRIPTS="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
PY="$ROOT/.venv/bin/python"
SUB="$1"; shift
export EM_RUNS="$SCRIPTS/../$SUB"
$PY -u judge.py "$@"
