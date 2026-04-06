# ONNX Runtime Engine

Author: Shuvam Banerji Seal

ONNX Runtime is used for portable high-performance model execution across CPU and
accelerator providers.

## Features Covered

- Graph optimization and optimized model export script.
- Session benchmarking script with configurable warmup and runs.
- Runtime config for provider and optimization-level tracking.

## Key References

- https://onnxruntime.ai/docs/
- https://onnxruntime.ai/docs/performance/
- https://onnxruntime.ai/docs/performance/tune-performance/
- https://onnxruntime.ai/docs/execution-providers/

## Optimize Model

```bash
python scripts/optimize_transformer.py \
  --input-model /path/to/model.onnx \
  --output-model /path/to/model.optimized.onnx
```

## Run Inference Benchmark

```bash
python scripts/run_onnx_inference.py \
  --model /path/to/model.optimized.onnx \
  --runs 30 --warmup 5
```
