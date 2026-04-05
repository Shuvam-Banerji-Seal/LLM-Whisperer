# Fast LLM Inference (vLLM First) - Agentic Skill Prompt

Use this prompt when the task is to build, optimize, or operate low-latency and high-throughput LLM inference.

## 1. Mission

Design and run practical inference stacks with this default priority order:

1. Correct outputs and stable serving behavior.
2. Predictable latency under real traffic.
3. Throughput per GPU dollar.
4. Operational simplicity and observability.
5. Portability.

Default runtime selection:

- Start with vLLM for most open-weight model serving.
- Use TensorRT-LLM for NVIDIA-first deployments that justify deeper optimization effort.
- Use llama.cpp and GGUF for local, edge, and CPU-friendly serving.
- Use TGI only when inheriting existing TGI infrastructure or specific compatibility needs.

## 2. Runtime Decision Matrix

| Runtime | Best for | Watchouts |
|---|---|---|
| vLLM | OpenAI-compatible API serving, strong concurrency, broad model support | Rapid feature changes; pin versions |
| TensorRT-LLM | Maximum NVIDIA datacenter performance and kernel-level optimization | More build and ops complexity |
| llama.cpp | Local and edge deployment, GGUF workflows, low-footprint serving | Feature parity differs by backend and quant format |
| Hugging Face TGI | Existing TGI estates and migration bridges | Documentation indicates maintenance mode; avoid for net-new long-lived platforms |

## 3. vLLM Execution Playbook

### 3.1 Install and bootstrap

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -U vllm --torch-backend=auto
```

### 3.2 Offline generation baseline

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(temperature=0.0, max_tokens=256)
outputs = llm.generate(["Explain speculative decoding in 3 bullets."], params)
print(outputs[0].outputs[0].text)
```

### 3.3 OpenAI-compatible serving baseline

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --api-key token-dev-only \
  --max-model-len 8192
```

Smoke test:

```bash
curl -s http://localhost:8000/v1/models
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-dev-only" \
  -d '{
    "model":"meta-llama/Llama-3.1-8B-Instruct",
    "messages":[{"role":"user","content":"Give 2 use cases for MoE."}],
    "temperature":0
  }'
```

### 3.4 Scaling knobs

- Single-node tensor parallel:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4
```

- Tensor plus pipeline parallel:

```bash
vllm serve <model> --tensor-parallel-size 8 --pipeline-parallel-size 2
```

- Multi-node with Ray executor:

```bash
vllm serve <model> --distributed-executor-backend ray --tensor-parallel-size 8
```

Practical rule: only increase parallelism after verifying interconnect quality (NCCL, IB, GPUDirect) and measuring end-to-end gains.

### 3.5 Quantization in vLLM

- Choose quantization based on validated runtime support and quality budget.
- Common options in current docs include AWQ, GPTQ-style models, bitsandbytes modes, FP8, and GGUF-related pathways.
- Always run task-specific quality checks after quantization changes.

### 3.6 Benchmarking protocol

Run at least three traffic profiles: short context, mixed context, and long context.

```bash
vllm bench latency --help
vllm bench throughput --help
vllm bench serve --help
```

Track:

- TTFT (time to first token)
- TPOT (time per output token)
- P50, P95, P99 latency
- Tokens per second and requests per second
- Peak memory and OOM rate

## 4. Alternative Runtime Notes

### 4.1 TGI

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:3.3.5 \
  --model-id meta-llama/Llama-3.1-8B-Instruct
```

Use when existing teams and tooling are already aligned to TGI. For greenfield systems, compare vLLM and TensorRT-LLM first.

### 4.2 TensorRT-LLM

```bash
docker run --rm -it --ipc host --gpus all \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10
```

```bash
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Use for high-throughput NVIDIA environments where compilation and optimization overhead is acceptable.

### 4.3 llama.cpp and GGUF

```bash
cmake -B build
cmake --build build --config Release
llama-server -hf ggml-org/gemma-3-1b-it-GGUF --port 8080
```

```bash
./llama-quantize input-model-f32.gguf output-model-q4_k_m.gguf q4_k_m
```

Use for local and edge inference, lightweight API serving, and CPU-first deployments.

## 5. Production Hardening Checklist

- Enforce authentication and request-level limits.
- Define queue and timeout budgets for each endpoint tier.
- Monitor TTFT, TPOT, tail latency, GPU memory, and error rates.
- Keep per-model canary rollout and rollback paths.
- Pin model revision, runtime version, and kernel stack.
- Test prompt templates per model before load tests.

## 6. Pitfalls to Avoid

- Comparing runtimes with different prompt sets or context lengths.
- Assuming all OpenAI-compatible APIs behave identically.
- Ignoring chat templates and then diagnosing quality regressions as runtime bugs.
- Scaling TP or PP without validating network and topology constraints.
- Shipping new quantization settings without quality gates.

## 7. References (Fetched 2026-04-06)

1. https://docs.vllm.ai/en/stable/getting_started/quickstart/ - Official quickstart for offline and serving workflows.
2. https://docs.vllm.ai/en/stable/getting_started/installation/ - Installation matrix and dependency guidance.
3. https://docs.vllm.ai/en/stable/serving/openai_compatible_server/ - OpenAI-compatible server behavior and parameters.
4. https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html - Tensor and pipeline parallel deployment guidance.
5. https://docs.vllm.ai/en/stable/serving/data_parallel_deployment.html - Data parallel deployment patterns.
6. https://docs.vllm.ai/en/stable/serving/context_parallel_deployment.html - Context-parallel serving notes for long contexts.
7. https://docs.vllm.ai/en/stable/features/quantization/ - Quantization support and caveats.
8. https://docs.vllm.ai/en/stable/benchmarking/ - Benchmark tooling entry point.
9. https://docs.vllm.ai/en/stable/design/paged_attention.html - Paged attention background and design context.
10. https://github.com/vllm-project/vllm - vLLM source repository and release tracking.
11. https://huggingface.co/docs/text-generation-inference/index - TGI docs and current maintenance notice.
12. https://huggingface.co/docs/text-generation-inference/en/quicktour - TGI quickstart deployment commands.
13. https://huggingface.co/docs/text-generation-inference/en/messages_api - TGI chat and messages API patterns.
14. https://github.com/huggingface/text-generation-inference - TGI repository state and roadmap signals.
15. https://nvidia.github.io/TensorRT-LLM/ - TensorRT-LLM documentation root.
16. https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html - TensorRT-LLM quickstart and serving.
17. https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html - Parallel strategy tuning guidance.
18. https://nvidia.github.io/TensorRT-LLM/features/quantization.html - TensorRT-LLM quantization options.
19. https://github.com/ggml-org/llama.cpp - llama.cpp runtime repository.
20. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md - llama-server API usage.
21. https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md - GGUF quantization command guidance.
