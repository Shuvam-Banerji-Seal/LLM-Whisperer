#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hf_model_dir> <output_dir>"
  echo "Environment variables: QUANT_TYPE, THREADS, IMATRIX, CONVERT_SCRIPT, LLAMA_QUANTIZE_BIN"
  exit 1
fi

HF_MODEL_DIR="$1"
OUTPUT_DIR="$2"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
THREADS="${THREADS:-8}"
IMATRIX="${IMATRIX:-}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-convert_hf_to_gguf.py}"
LLAMA_QUANTIZE_BIN="${LLAMA_QUANTIZE_BIN:-llama-quantize}"

mkdir -p "${OUTPUT_DIR}"
FP16_MODEL="${OUTPUT_DIR}/ggml-model-f16.gguf"
QUANT_MODEL="${OUTPUT_DIR}/ggml-model-${QUANT_TYPE}.gguf"

echo "Converting Hugging Face model to GGUF (F16): ${HF_MODEL_DIR}"
python3 "${CONVERT_SCRIPT}" "${HF_MODEL_DIR}" --outfile "${FP16_MODEL}" --outtype f16

cmd=("${LLAMA_QUANTIZE_BIN}" "${FP16_MODEL}" "${QUANT_MODEL}" "${QUANT_TYPE}" "${THREADS}")
if [[ -n "${IMATRIX}" ]]; then
  cmd=("${LLAMA_QUANTIZE_BIN}" --imatrix "${IMATRIX}" "${FP16_MODEL}" "${QUANT_MODEL}" "${QUANT_TYPE}" "${THREADS}")
fi

echo "Quantizing GGUF to ${QUANT_TYPE}"
"${cmd[@]}"

echo "Quantized model written to ${QUANT_MODEL}"
