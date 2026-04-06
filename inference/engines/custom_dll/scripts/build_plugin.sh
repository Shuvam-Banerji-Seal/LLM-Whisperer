#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${MODULE_DIR}/src"
OUTPUT_LIB="${MODULE_DIR}/libllm_plugin_stub.so"

cc -O2 -fPIC -shared \
  "${SRC_DIR}/plugin_stub.c" \
  -I"${SRC_DIR}" \
  -o "${OUTPUT_LIB}"

echo "Built plugin: ${OUTPUT_LIB}"
