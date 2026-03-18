#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-/root/agentic_medical_gpt/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

$PYTHON_BIN -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel

pip install   "torch>=2.5.0"   "transformers>=4.50.0"   "accelerate>=0.34.0"   "huggingface_hub>=0.26.0"   "qwen-vl-utils==0.0.14"   "pillow>=10.0.0"   "sentencepiece>=0.2.0"

echo "[ok] Huatuo-VL runtime prepared at $VENV_PATH"
