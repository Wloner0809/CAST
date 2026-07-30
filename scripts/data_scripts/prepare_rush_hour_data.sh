#!/usr/bin/env bash
# Generate rush_hour datasets for all difficulties.
#
# DatasetRegistry.register_dataset("rush_hour", ...) always saves to
#   data/datasets/rush_hour/{train,test}.parquet
# so we move the output to data/datasets/rush_hour/<difficulty>/ after each run.
#
# This script only generates parquet datasets. To use
# oracle_backend="precomputed", build each sidecar explicitly with
# examples/rush_hour/build_oracle_sidecar.py after this script finishes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET_DIR="data/datasets/rush_hour"

generate_difficulty() {
    local difficulty="$1"
    local train_size="${2:-8000}"
    local test_size="${3:-200}"

    echo "===== Generating rush_hour difficulty=${difficulty} (train=${train_size}, test=${test_size}) ====="

    PYTHONPATH=. python3 examples/rush_hour/prepare_rush_hour_data.py \
        --train-size "${train_size}" \
        --test-size "${test_size}" \
        --difficulty "${difficulty}"

    # Move generated files from flat rush_hour/ to rush_hour/<difficulty>/
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
