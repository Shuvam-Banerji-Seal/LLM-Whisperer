#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <input.gguf> <output.gguf> <quant_type> <threads>"
  echo "Example: $0 model-f16.gguf model-q4km.gguf Q4_K_M 8"
  exit 1
fi

LLAMA_QUANTIZE_BIN="${LLAMA_QUANTIZE_BIN:-llama-quantize}"
INPUT_MODEL="$1"
OUTPUT_MODEL="$2"
QUANT_TYPE="$3"
THREADS="$4"
IMATRIX="${IMATRIX:-}"
ALLOW_REQUANTIZE="${ALLOW_REQUANTIZE:-false}"
KEEP_SPLIT="${KEEP_SPLIT:-false}"

opts=()
if [[ -n "${IMATRIX}" ]]; then
  opts+=(--imatrix "${IMATRIX}")
fi
if [[ "${ALLOW_REQUANTIZE}" == "true" ]]; then
  opts+=(--allow-requantize)
fi
if [[ "${KEEP_SPLIT}" == "true" ]]; then
  opts+=(--keep-split)
fi

cmd=(
  "${LLAMA_QUANTIZE_BIN}"
  "${opts[@]}"
  "${INPUT_MODEL}"
  "${OUTPUT_MODEL}"
  "${QUANT_TYPE}"
  "${THREADS}"
)

echo "Quantizing ${INPUT_MODEL} -> ${OUTPUT_MODEL} with ${QUANT_TYPE}"
exec "${cmd[@]}"
