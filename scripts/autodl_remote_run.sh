#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/agentic_medical_gpt}"
QUESTION="${2:-我发烧39度两天还咳嗽，需要怎么处理？}"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "[run] smoke test"
python -m medagent.main --question "$QUESTION"

echo "[run] benchmark"
python -m medagent.benchmark.run --dataset data/benchmark_cases.json

echo "[ok] all done"

