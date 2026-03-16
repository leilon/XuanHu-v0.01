#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <user> <host> <port> <remote_dir> [identity_file]"
  exit 1
fi

USER_NAME="$1"
HOST="$2"
PORT="$3"
REMOTE_DIR="$4"
IDENTITY_FILE="${5:-}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SSH_OPTS="-p $PORT"
if [ -n "$IDENTITY_FILE" ]; then
  SSH_OPTS="$SSH_OPTS -i $IDENTITY_FILE"
fi

echo "[step] ensure remote dir"
ssh $SSH_OPTS "${USER_NAME}@${HOST}" "mkdir -p '${REMOTE_DIR}'"

echo "[step] upload code"
rsync -avz --delete \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".venv" \
  -e "ssh $SSH_OPTS" \
  "${LOCAL_DIR}/" "${USER_NAME}@${HOST}:${REMOTE_DIR}/"

echo "[step] bootstrap env"
ssh $SSH_OPTS "${USER_NAME}@${HOST}" "bash '${REMOTE_DIR}/scripts/autodl_remote_bootstrap.sh' '${REMOTE_DIR}'"

echo "[step] run smoke+benchmark"
ssh $SSH_OPTS "${USER_NAME}@${HOST}" "bash '${REMOTE_DIR}/scripts/autodl_remote_run.sh' '${REMOTE_DIR}'"

echo "[ok] deployed and executed"

