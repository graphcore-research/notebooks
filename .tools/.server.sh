#!/usr/bin/env bash

PORT="${1:-8888}"

set -e
set -o xtrace

SETUP_NO_JUPYTER=1 source setup.sh

mkdir -p "${POPLAR_EXECUTABLE_CACHE_DIR}" "${DATASET_DIR}" "${CHECKPOINT_DIR}"

pip install -r .tools/requirements.txt

python -m jupyter lab --port ${PORT} --no-browser --allow-root --ip 0.0.0.0
