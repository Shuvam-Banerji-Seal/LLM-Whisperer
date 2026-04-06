# llama.cpp Engine

Author: Shuvam Banerji Seal

llama.cpp is the lightweight path for GGUF-based deployment on CPU and mixed
hardware targets.

## Features Covered

- Local server startup template.
- OpenAI-compatible smoke test.
- GGUF quantization wrapper script.

## Key References

- https://github.com/ggml-org/llama.cpp
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/README.md
- https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md

## Run

```bash
bash scripts/start_llama_server.sh
```

## Smoke Test

```bash
python scripts/smoke_openai_chat.py \
  --base-url http://localhost:8080/v1 \
  --model local-gguf-model
```

## Quantize GGUF

```bash
bash scripts/quantize_gguf.sh input-f16.gguf output-q4.gguf Q4_K_M 8
```
