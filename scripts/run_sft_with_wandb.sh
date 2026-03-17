#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${1:-/root/autodl-tmp/medagent}"
BASE_MODEL="${2:-$WORK_ROOT/models/qwen2.5-7b-instruct}"
TRAIN_FILE="${3:-$WORK_ROOT/datasets/curated/bianque_intake/train.jsonl}"
EVAL_FILE="${4:-$WORK_ROOT/datasets/curated/bianque_intake/valid.jsonl}"

export HF_HOME="$WORK_ROOT/hf_cache"
export WANDB_DIR="$WORK_ROOT/wandb"

if [ -n "${WANDB_API_KEY:-}" ]; then
  python3 -m pip install -U wandb
  wandb login "$WANDB_API_KEY" --relogin
fi

python3 trainer/text/train_bianque_intake_sft.py \
  --base-model "$BASE_MODEL" \
  --train-file "$TRAIN_FILE" \
  --eval-file "$EVAL_FILE" \
  --task bianque_intake \
  --dataset-name bianque_intake_curated_v1 \
  --output-dir "$WORK_ROOT/outputs/adapters/bianque_intake_stage1" \
  --adapter-bank-dir "$WORK_ROOT/outputs/adapters" \
  --cache-dir "$WORK_ROOT/hf_cache" \
  --wandb-project qingnang-clinicos \
  --wandb-run-name bianque-intake-stage1 \
  --wandb-dir "$WORK_ROOT/wandb"


