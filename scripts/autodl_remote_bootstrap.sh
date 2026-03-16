#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/agentic_medical_gpt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

echo "[ok] environment ready at $PROJECT_DIR/.venv"

