#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../model_repository" && pwd)"

MODEL_REPOSITORY="${MODEL_REPOSITORY:-${DEFAULT_REPO}}"
HTTP_PORT="${HTTP_PORT:-8001}"
GRPC_PORT="${GRPC_PORT:-8002}"
METRICS_PORT="${METRICS_PORT:-8003}"
STRICT_MODEL_CONFIG="${STRICT_MODEL_CONFIG:-false}"
LOG_VERBOSE="${LOG_VERBOSE:-0}"

cmd=(
  tritonserver
  --model-repository "${MODEL_REPOSITORY}"
  --http-port "${HTTP_PORT}"
  --grpc-port "${GRPC_PORT}"
  --metrics-port "${METRICS_PORT}"
  --log-verbose "${LOG_VERBOSE}"
)

if [[ "${STRICT_MODEL_CONFIG}" == "true" ]]; then
  cmd+=(--strict-model-config=true)
else
  cmd+=(--strict-model-config=false)
fi

echo "Starting Triton using model repository ${MODEL_REPOSITORY}"
exec "${cmd[@]}"
