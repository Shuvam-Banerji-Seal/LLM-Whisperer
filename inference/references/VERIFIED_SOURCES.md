# Verified Sources

Author: Shuvam Banerji Seal

This file lists official and high-confidence sources used to construct the
inference module scaffold.

## vLLM

- https://docs.vllm.ai/en/stable/
- https://docs.vllm.ai/en/stable/getting_started/quickstart/
- https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- https://docs.vllm.ai/en/stable/benchmarking/
- https://github.com/vllm-project/vllm

## NVIDIA TensorRT-LLM

- https://nvidia.github.io/TensorRT-LLM/
- https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
- https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/index.html
- https://nvidia.github.io/TensorRT-LLM/features/quantization.html
- https://github.com/NVIDIA/TensorRT-LLM

## NVIDIA Triton Inference Server

- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/
- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html
- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/perf_analyzer.html
- https://github.com/triton-inference-server/server

## ONNX Runtime

- https://onnxruntime.ai/docs/
- https://onnxruntime.ai/docs/performance/
- https://onnxruntime.ai/docs/performance/tune-performance/
- https://onnxruntime.ai/docs/execution-providers/
- https://github.com/microsoft/onnxruntime

## llama.cpp and GGUF

- https://github.com/ggml-org/llama.cpp
- https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/README.md
- https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md

## Hugging Face Quantization Docs

- https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes
- https://huggingface.co/docs/transformers/main/en/quantization/gptq
- https://huggingface.co/docs/transformers/main/en/quantization/awq

## Notes

- Prioritize official docs and upstream repositories over third-party blogs.
- Re-validate compatibility matrices before production rollout.
- Pin versions in runtime configs for reproducible benchmarking.
