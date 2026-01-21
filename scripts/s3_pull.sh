#!/usr/bin/env bash
set -euo pipefail

if [ -z "${S3_BUCKET:-}" ]; then
  echo "S3_BUCKET이 설정되지 않았습니다. docker.env를 확인하세요." >&2
  exit 1
fi

data_prefix="${S3_DATA_PREFIX:-datasets}"
local_dir="${DATA_DIR:-/workspace/data}"

aws s3 sync "s3://${S3_BUCKET}/${data_prefix}" "${local_dir}"
