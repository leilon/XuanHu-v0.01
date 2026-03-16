#!/usr/bin/env bash
set -euo pipefail

MSG="${1:-checkpoint: $(date '+%F %T')}"
git add -A
git commit -m "$MSG"
echo "[ok] commit created: $MSG"

