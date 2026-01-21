#!/usr/bin/env bash
set -euo pipefail

docker compose build colmap
docker compose up -d colmap
echo "colmap이 백그라운드로 실행되었습니다."
echo "로그 확인: docker compose logs -f colmap"
