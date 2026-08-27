#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${EPOCHS:-100}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/results/benchmark_packed.csv}"
cd "${PROJECT_DIR}"

"${PROJECT_DIR}/scripts/build_and_test_wsl.sh"
"${PROJECT_DIR}/build/openfhe_lab_compare" \
  --dataset all \
  --refresh both \
  --epochs "${EPOCHS}" \
  --learning-rate 0.01 \
  --output "${OUTPUT_PATH}"
