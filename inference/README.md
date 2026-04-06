# Inference Blueprint

Author: Shuvam Banerji Seal

This directory contains a production-oriented inference scaffold for large language models.
It focuses on reproducibility, measurable performance, and deployment portability across
multiple backends.

## Goals

- Standardize how models are served across heterogeneous inference engines.
- Make latency, throughput, and quality tradeoffs measurable and comparable.
- Support OpenAI-compatible APIs for tooling interoperability.
- Keep quantization decisions explicit, versioned, and benchmark-backed.

## Engine Selection Matrix

| Engine | Best For | Strengths | Cautions |
| --- | --- | --- | --- |
| vLLM | General-purpose LLM serving | Continuous batching, strong OpenAI compatibility, broad model support | Tune memory and scheduler options per model/context profile |
| TensorRT-LLM | NVIDIA throughput-critical production paths | Kernel and graph-level optimizations, strong low-latency serving options | Hardware and software stack coupling is stricter |
| Triton Inference Server | Multi-model serving platforms | Unified model repository, HTTP/gRPC APIs, production observability patterns | Backend and repository management add operational overhead |
| ONNX Runtime | Portable optimized CPU/GPU inference | Execution providers, graph optimization, broad platform support | Export quality and operator compatibility must be verified |
| llama.cpp | Edge and resource-constrained deployment | GGUF ecosystem, efficient CPU and mixed hardware support | Quantization and tokenizer/template parity must be validated |
| Custom DLL | Experimental kernels and proprietary acceleration | Pluggable integration surface | ABI stability and safety checks are mandatory |

## Directory Contract

- `engines/`: engine-specific runbooks, configs, and smoke tests.
- `quantization/`: quantization profiles and scripts (bitsandbytes, GPTQ, AWQ, GGUF).
- `benchmarking/`: workload definitions, metric schemas, benchmark runners, and summaries.
- `serving/`: container and Kubernetes deployment templates plus health and load probes.
- `references/`: validated, official URLs used to build this scaffold.

## End-to-End Workflow

1. Pick an engine under `engines/` based on hardware and latency constraints.
2. Pick a quantization strategy in `quantization/configs/quantization_profiles.yaml`.
3. Run benchmark scenarios from `benchmarking/configs/benchmark_matrix.yaml`.
4. Promote a configuration to deployment manifests under `serving/`.
5. Capture metrics and compare runs before and after each change.

## Canonical Metrics

- TTFT: Time to first token.
- TPOT: Time per output token after first token.
- End-to-end latency: p50, p95, p99.
- Throughput: requests/sec, tokens/sec.
- Reliability: HTTP error rate, timeout rate, OOM/restart counts.
- Goodput: successful responses/sec under SLO.

## Quick Start

```bash
cd inference
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run a local vLLM server (example)
bash engines/vllm/scripts/start_vllm_server.sh

# In another terminal run smoke test
python engines/vllm/scripts/smoke_openai_chat.py \
  --base-url http://localhost:8000/v1 \
  --model local-model
```

## Knowledge Base

See `references/VERIFIED_SOURCES.md` for official docs and repositories used to
build this module.
