#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${EPOCHS:-100}"
OPTIMIZER="${OPTIMIZER:-gd}"
MOMENTUM="${MOMENTUM:-0.1}"
SIGMOID="${SIGMOID:-cubic}"
NAG_PACKING="${NAG_PACKING:-separate}"
if [[ "${NAG_PACKING}" != "separate" && "${NAG_PACKING}" != "packed" ]]; then
  echo "NAG_PACKING must be separate or packed" >&2
  exit 2
fi
ACTIVE_NAG_PACKING="separate"
DEFAULT_OUTPUT_PATH="${PROJECT_DIR}/results/benchmark_packed_${SIGMOID}.csv"
if [[ "${OPTIMIZER}" == "nag" ]]; then
  ACTIVE_NAG_PACKING="${NAG_PACKING}"
  packing_suffix=""
  if [[ "${NAG_PACKING}" == "packed" ]]; then
    packing_suffix="packed_"
  fi
  DEFAULT_OUTPUT_PATH="${PROJECT_DIR}/results/benchmark_nag_${packing_suffix}${SIGMOID}.csv"
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
  --nag-packing "${ACTIVE_NAG_PACKING}" \
  --sigmoid "${SIGMOID}" \
  --output "${OUTPUT_PATH}"
