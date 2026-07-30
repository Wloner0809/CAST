#!/usr/bin/env bash
# Generate sokoban datasets for all difficulties.
#
# DatasetRegistry.register_dataset("sokoban", ...) always saves to
#   data/datasets/sokoban/{train,test}.parquet
# so we move the output to data/datasets/sokoban/<difficulty>/ after each run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET_DIR="data/datasets/sokoban"

generate_difficulty() {
    local difficulty="$1"
    local train_size="${2:-8000}"
    local test_size="${3:-200}"

    echo "===== Generating sokoban difficulty=${difficulty} (train=${train_size}, test=${test_size}) ====="

    PYTHONPATH=. python3 examples/sokoban/prepare_sokoban_data.py \
        --train-size "${train_size}" \
        --test-size "${test_size}" \
        --difficulty "${difficulty}"

    # Move generated files from flat sokoban/ to sokoban/<difficulty>/
    mkdir -p "${DATASET_DIR}/${difficulty}"
    for f in train.parquet test.parquet train_verl.parquet test_verl.parquet; do
        if [[ -f "${DATASET_DIR}/${f}" ]]; then
            mv "${DATASET_DIR}/${f}" "${DATASET_DIR}/${difficulty}/${f}"
            echo "  Moved ${DATASET_DIR}/${f} -> ${DATASET_DIR}/${difficulty}/${f}"
        fi
    done

    echo "===== Done: ${difficulty} ====="
    echo
}

generate_difficulty id     8000 200
generate_difficulty unseen 8000 200
