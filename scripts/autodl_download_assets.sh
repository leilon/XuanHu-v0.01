#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/medagent}"
BASE_MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}"
VLM_MODEL="${3:-Qwen/Qwen2.5-VL-7B-Instruct}"

mkdir -p "$ROOT"/{models,datasets,hf_cache}
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

python3 -m pip install -U "huggingface_hub[cli]" datasets pandas pyarrow

echo "[step] download base model: $BASE_MODEL"
huggingface-cli download "$BASE_MODEL" --local-dir "$ROOT/models/qwen2.5-7b-instruct" --resume-download

echo "[step] download multimodal model: $VLM_MODEL"
huggingface-cli download "$VLM_MODEL" --local-dir "$ROOT/models/qwen2.5-vl-7b-instruct" --resume-download

echo "[step] download selected medical datasets"
python3 scripts/download_medical_datasets.py --root "$ROOT/datasets"
python3 scripts/prepare_medical_training_data.py --root "$ROOT/datasets"

echo "[ok] assets downloaded under $ROOT"
