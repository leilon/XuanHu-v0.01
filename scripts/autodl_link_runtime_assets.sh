#!/usr/bin/env bash
set -euo pipefail

WORKTREE_ROOT="${1:-/root/agentic_medical_gpt}"
DATA_ROOT="${2:-/root/autodl-tmp/medagent}"
LINK_ROOT="${WORKTREE_ROOT}/runtime_assets"

mkdir -p "${LINK_ROOT}"

ln -sfn "${DATA_ROOT}/models" "${LINK_ROOT}/models"
ln -sfn "${DATA_ROOT}/datasets" "${LINK_ROOT}/datasets"
ln -sfn "${DATA_ROOT}/rag" "${LINK_ROOT}/rag"
ln -sfn "${DATA_ROOT}/outputs" "${LINK_ROOT}/outputs"
ln -sfn "${DATA_ROOT}/logs" "${LINK_ROOT}/logs"

cat > "${LINK_ROOT}/README.txt" <<EOF
This directory contains convenience symlinks for VS Code Remote browsing.

models   -> ${DATA_ROOT}/models
datasets -> ${DATA_ROOT}/datasets
rag      -> ${DATA_ROOT}/rag
outputs  -> ${DATA_ROOT}/outputs
logs     -> ${DATA_ROOT}/logs
EOF

echo "[ok] runtime asset links created under ${LINK_ROOT}"
