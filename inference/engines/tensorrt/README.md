# TensorRT-LLM Engine

Author: Shuvam Banerji Seal

TensorRT-LLM targets NVIDIA-first deployments that need aggressive latency and
throughput optimizations.

## Features Covered

- `trtllm-serve` startup template with explicit runtime knobs.
- OpenAI-compatible smoke test against deployed endpoint.
- Config file for reproducible serving defaults.

## Key References

- https://nvidia.github.io/TensorRT-LLM/
- https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
- https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/index.html

## Run

```bash
bash scripts/start_trtllm_serve.sh
```

## Smoke Test

```bash
python scripts/smoke_openai_chat.py \
  --base-url http://localhost:8000/v1 \
  --model local-trt-model
```
