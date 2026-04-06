#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
MODEL_PATH="${MODEL_PATH:-/models/local-model.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-local-gguf-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CONTEXT_SIZE="${CONTEXT_SIZE:-8192}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
THREADS="${THREADS:-8}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
API_KEY="${API_KEY:-}"

cmd=(
  "${LLAMA_SERVER_BIN}"
  -m "${MODEL_PATH}"
  --alias "${MODEL_ALIAS}"
  --host "${HOST}"
  --port "${PORT}"
  --ctx-size "${CONTEXT_SIZE}"
  -ngl "${N_GPU_LAYERS}"
  -t "${THREADS}"
)

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  cmd+=(--chat-template "${CHAT_TEMPLATE}")
fi

if [[ -n "${API_KEY}" ]]; then
  cmd+=(--api-key "${API_KEY}")
fi

echo "Starting llama.cpp server on ${HOST}:${PORT} for model ${MODEL_PATH}"
exec "${cmd[@]}"
