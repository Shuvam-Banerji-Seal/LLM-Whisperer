#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/local-trtllm-checkpoint}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-trt-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-local-dev-key}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"
KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTION:-0.90}"
TOKENIZER="${TOKENIZER:-auto}"

cmd=(
  trtllm-serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${API_KEY}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --max-num-tokens "${MAX_NUM_TOKENS}"
  --kv-cache-free-gpu-mem-fraction "${KV_CACHE_FREE_GPU_MEM_FRACTION}"
  --tokenizer "${TOKENIZER}"
)

echo "Starting TensorRT-LLM server on ${HOST}:${PORT} using ${MODEL_PATH}"
exec "${cmd[@]}"
