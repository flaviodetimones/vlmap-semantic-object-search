#!/usr/bin/env bash
set -euo pipefail

SCENE_ID="${1:-1}"
MIN_NAV="${2:-0.25}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_ROOT}/results/nav_batch_${STAMP}"
mkdir -p "${OUT_DIR}"

QUERY_FILE="${OUT_DIR}/queries.txt"
LOG_FILE="${OUT_DIR}/interactive_nav.log"

docker exec -i tfg-sim bash -lc "
  export PYTHONPATH=/workspace/third_party/vlmaps:\$PYTHONPATH
  cd /workspace
  python tools/nav_batch_queries.py \
    --scene-id ${SCENE_ID} \
    --min-room-navigable ${MIN_NAV}
" > "${QUERY_FILE}"

printf 'quit\n' >> "${QUERY_FILE}"

docker exec -i tfg-sim bash -lc "
  export PYTHONPATH=/workspace/third_party/vlmaps:\$PYTHONPATH
  cd /workspace/third_party/vlmaps
  python application/interactive_object_nav.py \
    data_paths=hssd \
    scene_id=${SCENE_ID} \
    dataset_type=hssd \
    scene_dataset_config_file=/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json
" < "${QUERY_FILE}" | tee "${LOG_FILE}"

echo
echo "Queries: ${QUERY_FILE}"
echo "Log:     ${LOG_FILE}"
