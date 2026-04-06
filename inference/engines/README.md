# Inference Engines

Author: Shuvam Banerji Seal

This module contains engine-specific runbooks, runtime configs, and smoke tests.
Each engine directory follows the same operational pattern:

1. Configure runtime parameters in `configs/*.yaml`.
2. Start server/runtime via `scripts/start_*.sh`.
3. Validate API behavior with `scripts/smoke_*.py`.

## Common Decision Heuristics

- Prefer `vllm` for most OpenAI-compatible LLM serving workloads.
- Prefer `tensorrt` for NVIDIA production workloads with strict latency budgets.
- Prefer `triton` for multi-model serving platforms and heterogeneous backends.
- Prefer `onnxruntime` for portable model execution and cross-platform deployment.
- Prefer `llama_cpp` for edge and CPU-lean scenarios using GGUF models.
- Use `custom_dll` for proprietary kernels and plugin interfaces.

## Uniform Validation Steps

```bash
# 1) Start the selected engine
bash <engine>/scripts/start_*.sh

# 2) Run smoke test
python <engine>/scripts/smoke_*.py

# 3) Run benchmark from inference/benchmarking
python ../benchmarking/scripts/run_openai_benchmark.py ...
```
