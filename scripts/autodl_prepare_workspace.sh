#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/medagent}"

mkdir -p "$ROOT"/{models,datasets,rag,wandb,outputs,hf_cache,logs}
mkdir -p "$ROOT/datasets"/{sft,rl,rag_raw}
mkdir -p "$ROOT/outputs"/{adapters,checkpoints}

echo "[ok] workspace prepared:"
echo "  root: $ROOT"
echo "  models: $ROOT/models"
echo "  datasets: $ROOT/datasets"
echo "  rag: $ROOT/rag"
echo "  wandb: $ROOT/wandb"
echo "  outputs: $ROOT/outputs"

