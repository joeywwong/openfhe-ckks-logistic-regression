#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${EPOCHS:-100}"
REPEATS="${REPEATS:-4}"
MOMENTUM="${MOMENTUM:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
DATASET="${DATASET:-all}"
REFRESH="${REFRESH:-both}"
NAG_PACKING="${NAG_PACKING:-separate}"
BUILD_AND_TEST="${BUILD_AND_TEST:-1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/results/gd_nag_${TIMESTAMP}}"
RAW_DIR="${RESULT_DIR}/raw"

if [[ ! "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer" >&2
  exit 2
fi
if [[ "${BUILD_AND_TEST}" != "0" && "${BUILD_AND_TEST}" != "1" ]]; then
  echo "BUILD_AND_TEST must be 0 or 1" >&2
  exit 2
fi
if [[ "${NAG_PACKING}" != "separate" && "${NAG_PACKING}" != "packed" ]]; then
  echo "NAG_PACKING must be separate or packed" >&2
  exit 2
fi
if [[ -e "${RESULT_DIR}" ]]; then
  echo "Result directory already exists; choose a new RESULT_DIR: ${RESULT_DIR}" >&2
  exit 2
fi

mkdir -p "${RAW_DIR}"
cd "${PROJECT_DIR}"

{
  echo "repeats,epochs,learning_rate,momentum,dataset,refresh,nag_packing"
  echo "${REPEATS},${EPOCHS},${LEARNING_RATE},${MOMENTUM},${DATASET},${REFRESH},${NAG_PACKING}"
} > "${RESULT_DIR}/experiment_config.csv"

if [[ "${BUILD_AND_TEST}" == "1" ]]; then
  "${PROJECT_DIR}/scripts/build_and_test_wsl.sh"
elif [[ ! -x "${PROJECT_DIR}/build/openfhe_lab_compare" ]]; then
  echo "BUILD_AND_TEST=0, but ${PROJECT_DIR}/build/openfhe_lab_compare is missing" >&2
  exit 2
fi

echo "Controlled GD/NAG comparison"
echo "  repeats=${REPEATS}, epochs=${EPOCHS}, learning_rate=${LEARNING_RATE}, momentum=${MOMENTUM}"
  echo "  dataset=${DATASET}, refresh=${REFRESH}, nag_packing=${NAG_PACKING}"
echo "  output=${RESULT_DIR}"

for ((run = 1; run <= REPEATS; ++run)); do
  run_label="$(printf '%03d' "${run}")"
  # Alternate the order to reduce systematic warm-cache and first-run bias.
  if ((run % 2 == 1)); then
    optimizers=(gd nag)
  else
    optimizers=(nag gd)
  fi

  for optimizer in "${optimizers[@]}"; do
    nag_packing="separate"
    if [[ "${optimizer}" == "nag" ]]; then
      nag_packing="${NAG_PACKING}"
    fi
    output_path="${RAW_DIR}/run_${run_label}_${optimizer}.csv"
    echo "Run ${run}/${REPEATS}: ${optimizer} -> ${output_path}"
    "${PROJECT_DIR}/build/openfhe_lab_compare" \
      --dataset "${DATASET}" \
      --refresh "${REFRESH}" \
      --epochs "${EPOCHS}" \
      --learning-rate "${LEARNING_RATE}" \
      --optimizer "${optimizer}" \
      --momentum "${MOMENTUM}" \
      --nag-packing "${nag_packing}" \
      --output "${output_path}"
  done
done

python3 "${PROJECT_DIR}/scripts/summarize_gd_nag.py" \
  --input-dir "${RAW_DIR}" \
  --output-dir "${RESULT_DIR}"

echo "Comparison complete: ${RESULT_DIR}/aggregate_comparison.csv"
