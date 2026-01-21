#!/usr/bin/env bash
set -euo pipefail

if [ -z "${S3_BUCKET:-}" ]; then
  echo "S3_BUCKET이 설정되지 않았습니다. docker.env를 확인하세요." >&2
  exit 1
fi

output_prefix="${S3_OUTPUT_PREFIX:-outputs}"
local_dir="${OUTPUT_DIR:-/workspace/outputs}"

aws s3 sync "${local_dir}" "s3://${S3_BUCKET}/${output_prefix}"
