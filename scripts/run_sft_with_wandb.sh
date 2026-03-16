#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${1:-/root/autodl-tmp/medagent}"
BASE_MODEL="${2:-$WORK_ROOT/models/qwen2.5-7b-instruct}"
TRAIN_FILE="${3:-data/train_medical_sft.sample.jsonl}"

export HF_HOME="$WORK_ROOT/hf_cache"
export WANDB_DIR="$WORK_ROOT/wandb"

if [ -n "${WANDB_API_KEY:-}" ]; then
  python3 -m pip install -U wandb
  wandb login "$WANDB_API_KEY" --relogin
fi

python3 scripts/train_qlora.py \
  --base-model "$BASE_MODEL" \
  --train-file "$TRAIN_FILE" \
  --task general_intake \
  --dataset-name med_sft_bootstrap \
  --output-dir "$WORK_ROOT/outputs/adapters/general_intake_v1" \
  --adapter-bank-dir "$WORK_ROOT/outputs/adapters" \
  --cache-dir "$WORK_ROOT/hf_cache" \
  --wandb-project medagent-7b \
  --wandb-run-name sft-general-intake-v1 \
  --wandb-dir "$WORK_ROOT/wandb"

