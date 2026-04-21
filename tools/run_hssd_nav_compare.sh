#!/usr/bin/env bash
set -euo pipefail

SCENE_ID="${1:-1}"
MIN_NAV="${2:-0.25}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_ROOT}/results/nav_compare_${STAMP}"
mkdir -p "${OUT_DIR}"

QUERY_FILE="${OUT_DIR}/queries.txt"
BASELINE_LOG="${OUT_DIR}/baseline.log"
EXECUTOR_LOG="${OUT_DIR}/executor.log"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
SUMMARY_MD="${OUT_DIR}/summary.md"

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
" < "${QUERY_FILE}" | tee "${BASELINE_LOG}"

docker exec -i tfg-sim bash -lc "
  export PYTHONPATH=/workspace/third_party/vlmaps:\$PYTHONPATH
  cd /workspace/third_party/vlmaps
  python application/interactive_object_nav_executor.py \
    data_paths=hssd \
    scene_id=${SCENE_ID} \
    dataset_type=hssd \
    scene_dataset_config_file=/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json
" < "${QUERY_FILE}" | tee "${EXECUTOR_LOG}"

docker exec -i tfg-sim bash -lc "
  export PYTHONPATH=/workspace/third_party/vlmaps:\$PYTHONPATH
  cd /workspace
  python tools/compare_nav_runs.py \
    --baseline-log ${BASELINE_LOG/\/home\/mario\/tfg\/vlmap-semantic-object-search-tfg/\/workspace} \
    --executor-log ${EXECUTOR_LOG/\/home\/mario\/tfg\/vlmap-semantic-object-search-tfg/\/workspace} \
    --out-csv ${SUMMARY_CSV/\/home\/mario\/tfg\/vlmap-semantic-object-search-tfg/\/workspace} \
    --out-md ${SUMMARY_MD/\/home\/mario\/tfg\/vlmap-semantic-object-search-tfg/\/workspace}
"

echo
echo "Queries:       ${QUERY_FILE}"
echo "Baseline log:  ${BASELINE_LOG}"
echo "Executor log:  ${EXECUTOR_LOG}"
echo "Summary CSV:   ${SUMMARY_CSV}"
echo "Summary MD:    ${SUMMARY_MD}"
