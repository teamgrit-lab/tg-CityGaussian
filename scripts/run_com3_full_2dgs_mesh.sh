#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="com3_full_2dgs"
CONFIG_PATH="configs/com3_full_2dgs.yaml"
OUTPUT_DIR="outputs/${PROJECT_NAME}"

COMPOSE="docker-compose"
SERVICE="citygs"

if ! ${COMPOSE} ps --status running --services | grep -q "^${SERVICE}$"; then
  echo "ERROR: ${SERVICE} 컨테이너가 실행 중이 아닙니다." >&2
  echo "먼저 'docker compose up -d'로 실행해 주세요." >&2
  exit 1
fi

run_in_container() {
  ${COMPOSE} exec -T "${SERVICE}" bash -lc "$*"
}

echo "[1/4] coarse 학습 시작"
run_in_container "python main.py fit --config ${CONFIG_PATH} -n ${PROJECT_NAME}"

echo "[2/4] 파티셔닝"
run_in_container "python utils/partition_citygs.py --config_path ${CONFIG_PATH} --force"

echo "[3/4] 파티션 미세튜닝 및 병합"
run_in_container "python utils/train_citygs_partitions.py -n ${PROJECT_NAME}"
run_in_container "python utils/merge_citygs_ckpts.py ${OUTPUT_DIR}"

echo "[4/4] 메쉬 추출"
run_in_container "python utils/gs2d_mesh_extraction.py ${OUTPUT_DIR} --voxel_size ${VOXEL_SIZE:-0.02} --sdf_trunc ${SDF_TRUNC:-0.08} --depth_trunc ${DEPTH_TRUNC:-0.6}"

echo "완료: ${OUTPUT_DIR}"
