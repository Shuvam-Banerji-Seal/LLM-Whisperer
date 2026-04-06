#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-local-dev-key}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"

cmd=(
  vllm serve "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${API_KEY}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --tokenizer-mode "${TOKENIZER_MODE}"
)

if [[ "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
  cmd+=(--enable-chunked-prefill)
fi

echo "Starting vLLM server on ${HOST}:${PORT} for model ${MODEL_NAME}"
exec "${cmd[@]}"
