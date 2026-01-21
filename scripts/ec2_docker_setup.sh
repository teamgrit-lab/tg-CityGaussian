#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "docker가 설치되어 있지 않습니다." >&2
  echo "먼저 Docker Engine과 NVIDIA Container Toolkit을 설치하세요." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "docker 데몬에 접근할 수 없습니다. 권한을 확인하세요." >&2
  exit 1
fi

if [ ! -f docker.env ]; then
  if [ -f docker.env.example ]; then
    cp docker.env.example docker.env
    echo "docker.env를 생성했습니다. 값을 채워주세요: docker.env"
  else
    echo "docker.env.example 파일이 없습니다." >&2
    exit 1
  fi
fi

if [ -f .gitmodules ]; then
  git submodule update --init --recursive
fi

docker compose build citygs colmap
echo "도커 이미지 빌드 완료 (citygs, colmap)."
