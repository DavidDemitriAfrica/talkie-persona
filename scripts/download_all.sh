#!/usr/bin/env bash
# Download the Talkie model family (all conditions) and related community data.
# Requires: HF_TOKEN in env, .venv with huggingface_hub installed.
# Usage: bash scripts/download_all.sh [official|hf|data]
set -euo pipefail
cd "$(dirname "$0")/.."
HF=./.venv/bin/hf
GROUP="${1:-all}"

if [[ "$GROUP" == "official" || "$GROUP" == "all" ]]; then
  # Canonical raw checkpoints from the talkie-lm org (fp32 .ckpt for bases, bf16 .pt for it)
  $HF download talkie-lm/talkie-1930-13b-base --local-dir models/official/talkie-1930-13b-base
  $HF download talkie-lm/talkie-1930-13b-it   --local-dir models/official/talkie-1930-13b-it
  $HF download talkie-lm/talkie-web-13b-base  --local-dir models/official/talkie-web-13b-base
fi

if [[ "$GROUP" == "hf" || "$GROUP" == "all" ]]; then
  # Community transformers conversions (BF16 sharded safetensors, chat template, 4096 ctx).
  # No public conversion exists for talkie-web-13b-base; see scripts/convert_web_to_hf.py.
  $HF download xlr8harder/talkie-1930-13b-base-tf --local-dir models/hf/talkie-1930-13b-base
  $HF download xlr8harder/talkie-1930-13b-it-tf   --local-dir models/hf/talkie-1930-13b-it
fi

if [[ "$GROUP" == "data" || "$GROUP" == "all" ]]; then
  # Official pre/post-training data was never released; these are the community datasets.
  $HF download xlr8harder/talkie-yarn-32k-gutenberg-pre1931-265m --repo-type dataset \
    --local-dir data/gutenberg-pre1931-yarn-32k
  $HF download trumancai/talkie-1930-knowledge-bench --repo-type dataset \
    --local-dir data/talkie-1930-knowledge-bench
fi
