#!/usr/bin/env bash
set -euo pipefail

docker compose build citygs
docker compose up -d citygs
echo "citygs가 백그라운드로 실행되었습니다."
echo "로그 확인: docker compose logs -f citygs"
