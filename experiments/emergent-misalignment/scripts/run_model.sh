#!/usr/bin/env bash
# Train + eval a list of EM conditions on a chosen base model, into an isolated
# runs dir, so cross-model runs never clobber each other.
#
# Usage: bash run_model.sh <model_tag> <gpu_id> <epochs> cond1 cond2 ...
#   model_tag: it   -> talkie-1930-13b-it   (runs/)         [the default pipeline]
#              twin -> talkie-web-13b-base  (runs_twin/)    [modern-web control]
#              llama-> Llama-3.1-8B-Instruct(runs_llama/)   [aligned modern control]
#              qwen -> Qwen2.5-7B-Instruct  (runs_qwen/)    [ungated fallback]
set -euo pipefail
cd "$(dirname "$0")"
SCRIPTS="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
TAG="$1"; GPU="$2"; EPOCHS="$3"; shift 3

case "$TAG" in
  it)
    export EM_BASE_MODEL="$ROOT/models/hf/talkie-1930-13b-it"
    export EM_RUNS="$SCRIPTS/../runs" ;;
  twin)
    export EM_BASE_MODEL="$ROOT/models/hf/talkie-web-13b-base"
    export EM_RUNS="$SCRIPTS/../runs_twin" ;;
  llama)
    export EM_BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    export EM_RUNS="$SCRIPTS/../runs_llama"
    export EM_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" ;;
  qwen)
    export EM_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
    export EM_RUNS="$SCRIPTS/../runs_qwen"
    export EM_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" ;;
  *)
    echo "unknown model tag: $TAG" >&2; exit 1 ;;
esac

mkdir -p "$EM_RUNS"
PY="$ROOT/.venv/bin/python"
for cond in "$@"; do
  if [ "$cond" = "base" ]; then
    # base = the un-fine-tuned model; eval-only, no adapter to train.
    echo "[$TAG gpu$GPU] evaluating base (no adapter)"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_em.py base 24 \
      > "$EM_RUNS/eval_base.log" 2>&1
    echo "[$TAG gpu$GPU] done base"
    continue
  fi
  echo "[$TAG gpu$GPU] training $cond -> $EM_RUNS"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_lora.py "$cond" "$EPOCHS" \
    > "$EM_RUNS/train_${cond}.log" 2>&1
  echo "[$TAG gpu$GPU] evaluating $cond"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u eval_em.py "$cond" 24 \
    > "$EM_RUNS/eval_${cond}.log" 2>&1
  echo "[$TAG gpu$GPU] done $cond"
done
