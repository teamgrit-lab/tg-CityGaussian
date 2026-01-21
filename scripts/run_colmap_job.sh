#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "사용법: bash scripts/run_colmap_job.sh \"<실행 명령>\"" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
log_path="${LOG_DIR}/colmap_${ts}.log"
cmd="$*"

nohup docker compose run --rm colmap bash -lc "$cmd" >"$log_path" 2>&1 &

echo "백그라운드 실행 PID: $!"
echo "로그 파일: ${log_path}"
