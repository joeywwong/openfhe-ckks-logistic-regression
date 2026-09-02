#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${EPOCHS:-100}"
OPTIMIZER="${OPTIMIZER:-gd}"
MOMENTUM="${MOMENTUM:-0.1}"
SIGMOID="${SIGMOID:-cubic}"
DEFAULT_OUTPUT_PATH="${PROJECT_DIR}/results/benchmark_packed_${SIGMOID}.csv"
if [[ "${OPTIMIZER}" == "nag" ]]; then
  DEFAULT_OUTPUT_PATH="${PROJECT_DIR}/results/benchmark_nag_${SIGMOID}.csv"
fi
OUTPUT_PATH="${OUTPUT_PATH:-${DEFAULT_OUTPUT_PATH}}"
cd "${PROJECT_DIR}"

"${PROJECT_DIR}/scripts/build_and_test_wsl.sh"
"${PROJECT_DIR}/build/openfhe_lab_compare" \
  --dataset all \
  --refresh both \
  --epochs "${EPOCHS}" \
  --learning-rate 0.01 \
  --optimizer "${OPTIMIZER}" \
  --momentum "${MOMENTUM}" \
  --sigmoid "${SIGMOID}" \
  --output "${OUTPUT_PATH}"
