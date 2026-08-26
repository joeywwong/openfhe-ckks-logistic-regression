#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${EPOCHS:-100}"
cd "${PROJECT_DIR}"

"${PROJECT_DIR}/scripts/build_and_test_wsl.sh"
"${PROJECT_DIR}/build/openfhe_lab_compare" \
  --dataset all \
  --refresh both \
  --epochs "${EPOCHS}" \
  --learning-rate 0.01 \
  --output "${PROJECT_DIR}/results/benchmark.csv"
