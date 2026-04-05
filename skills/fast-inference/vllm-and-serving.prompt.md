# vLLM and Fast LLM Serving - Master Agentic Skill Prompt 

**Version**: 2.0 (Ultra-Detailed Production Guide)
**Focus**: vLLM, PagedAttention, Distributed Serving, Quantization, Multi-LoRA, Speculative Decoding, and Production Operations.

---

## 1. Core Philosophy & Architecture

When deploying Large Language Models (LLMs), the primary bottlenecks are memory bandwidth and VRAM fragmentation. vLLM solves this using **PagedAttention** and **Continuous Batching**.
Your mission when using this skill is to completely eliminate OOM (Out of Memory) errors, maximize GPU utilization, and deliver sub-millisecond Time-To-First-Token (TTFT) and high Inter-Token Latency (ITL).

### 1.1 PagedAttention
Unlike naive KV-caching which pre-allocates contiguous memory for maximum sequence lengths, PagedAttention chunks the KV cache into small, fixed-size blocks (default 16 or 32 tokens).
- **Benefit**: Zero memory fragmentation. Memory waste drops from ~60% to <4%.
- **Actionable**: Always profile the `block_size` flag. For heavily varying sequence lengths, smaller blocks (16) prevent waste but add slight pointer-chasing overhead.

### 1.2 Continuous Batching (Iteration-level Scheduling)
Instead of waiting for all sequences in a batch to finish (static batching), vLLM ejects completed sequences and injects new ones at the *iteration level* (per token generation).
- **Benefit**: Up to 20x higher throughput compared to Hugging Face `transformers` baseline.

---

## 2. Environment Setup & Installation Best Practices

### 2.1 Docker (Recommended for Production)
Never install vLLM raw on a host OS in production. Always use the official NVIDIA-based Docker containers.

```bash
# Obtain the latest vLLM image
docker pull vllm/vllm-openai:latest

# Run with all GPUs, mapping IPC for tensor parallelism
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3-8B-Instruct \
    --tensor-parallel-size 2
```
*Edge Case:* If `NCCL` errors occur during multi-GPU setups across different nodes, ensure `--network=host` is set and firewall rules allow NCCL ports (tcp/udp).

### 2.2 Compilation from Source (For custom CUDA extensions)
If optimizing for a very specific arch (e.g., Hopper H100 with FP8 optimizations):
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
# If NVCC fails, ensure CUDA_HOME is set:
export CUDA_HOME=/usr/local/cuda-12.1
```

---

## 3. Core Engine API (Offline Inference)

For batch processing jobs (e.g., generating 1M synthetic data points), bypassing the HTTP server is significantly faster.

```python
from vllm import LLM, SamplingParams

# 1. Initialize the VLLM engine
# gpu_memory_utilization=0.9 means 90% of GPU VRAM is reserved for model + KV cache.
# max_model_len limits the max context. Lower this to fit into OOM-prone GPUs.
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    trust_remote_code=True,
    quantization=None # Set to 'awq' or 'gptq' if applicable
)

# 2. Define sampling parameters
sampling_params = SamplingParams(
    n=1,                  # Number of output sequences per prompt
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=512,
    presence_penalty=0.2,
    frequency_penalty=0.2,
    stop=["<|im_end|>", "User:"],
    use_beam_search=False # Beam search requires n > 1
)

# 3. Batch inference execution
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is"
] * 1000 # Scaling up

# vLLM automatically handles continuous batching under the hood
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### 3.1 Edge Case: Out of Memory (OOM) During Initialization
**Symptom**: `ValueError: No available memory for the cache blocks.`
**Solution**: 
1. Reduce `max_model_len` (e.g., 32k -> 8192).
2. Reduce `gpu_memory_utilization` (e.g., 0.9 -> 0.8).
3. Decrease concurrency, or enable `enforce_eager=True` to disable CUDA graph capture which uses pre-allocated memory.

---

## 4. API Server Configuration (OpenAI Compatible)

For microservice architectures.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --served-model-name "llama-3-70b" \
    --api-key "sk-vllm-secret-xyz"
```

### 4.1 Invoking the Server
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-vllm-secret-xyz",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="llama-3-70b",
    messages=[
        {"role": "system", "content": "You are a highly capable coding assistant."},
        {"role": "user", "content": "Write a python quicksort."}
    ],
    temperature=0.2,
    stream=True # ALWAYS set stream=True for UX performance (Time to First Token)
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

---

## 5. Advanced Hardware Scaling

### 5.1 Tensor Parallelism (TP)
Splits individual layers across multiple GPUs in the *same* node.
- **Rule of Thumb**: TP should equal the number of GPUs in a single node (up to 8).
- **Command**: `--tensor-parallel-size <N>`
- **Warning**: TP across different physical nodes (via network) is heavily discouraged due to interconnect latency.

### 5.2 Pipeline Parallelism (PP)
Splits different layers of the model across different GPUs/nodes.
- **Rule of Thumb**: Use when a model (like 405B Llama-3) exceeds the VRAM of a single node's GPUs.
- **Command**: `--pipeline-parallel-size <M>`
- **Requirement**: Ray must be installed.
```bash
# Example for deploying on 16 GPUs (2 nodes of 8 GPUs)
# TP=8 (across node), PP=2 (splits layers across the 2 nodes)
python -m vllm.entrypoints.openai.api_server     --model meta-llama/Llama-3-400b-Instruct     --tensor-parallel-size 8     --pipeline-parallel-size 2
```

---

## 6. Quantization: AWQ, GPTQ, SqueezeLLM, and FP8

Memory bandwidth is the ultimate wall for LLMs. Sub-byte and 8-bit quantization is mandatory for high-throughput deployments.

### 6.1 AWQ (Activation-aware Weight Quantization)
Best balance between quality retention and speed. Keeps salient weights in FP16, quantizes the rest to INT4.
- **Support**: Native vLLM support via Marlin kernels.
```bash
python -m vllm.entrypoints.openai.api_server     --model TheBloke/Mixtral-8x7B-v0.1-AWQ     --quantization awq
```

### 6.2 FP8 KV Cache and Weights (Hopper Architecture+)
For NVIDIA H100/H200, FP8 is massively hardware-accelerated.
```bash
# FP8 Weights
python -m vllm.entrypoints.openai.api_server     --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8

# FP8 KV Cache (Reduces KV cache memory by 50%, doubling batch size)
python -m vllm.entrypoints.openai.api_server     --model meta-llama/Llama-3-8B-Instruct     --kv-cache-dtype fp8
```
*Edge Case:* If `kv-cache-dtype fp8` drops quality drastically, check `ropeca` scaling. Ensure the model supports generic fp8 caching.

---

## 7. Multi-LoRA Serving (vLLM Native)

You do NOT need to deploy 50 different vLLM instances for 50 fine-tuned models. vLLM can host 1 Base Model and dynamically swap hundreds of LoRAs at the batch level (Iterative LoRA batching).

### 7.1 Server Setup with LoRA
```bash
# Enable LoRA, allocate memory for 4 concurrent LoRAs in VRAM
python -m vllm.entrypoints.openai.api_server     --model meta-llama/Llama-3-8B     --enable-lora     --max-loras 4     --max-lora-rank 64     --lora-modules sql-lora=/path/to/sql css-lora=/path/to/css
```

### 7.2 Requesting a Specific LoRA (OpenAI API payload)
Pass the LoRA name in the `model` field.
```python
client.chat.completions.create(
    model="sql-lora", # Routes request to base model + SQL LoRA weights dynamically
    messages=[{"role": "user", "content": "SELECT * FROM users;"}]
)
```

### 7.3 Edge Cases for LoRA
- **VRAM Fragmentation**: If `--max-loras` is too high, it eats into the KV Cache pool, drastically reducing maximum concurrent requests.
- **Rank mismatch**: If a user fine-tunes a LoRA with `$rank=128$` but your server boots with `--max-lora-rank 64`, the request fails. Always set max rank to the highest among deployed models.

---

## 8. Prefix Caching (APC)

If thousands of users share a massive identical system prompt (e.g., a 10,000-token constitution for an AI persona), recomputing the KV cache for that prompt wastes compute.

**Enable Automatic Prefix Caching (APC):**
```bash
python -m vllm.entrypoints.openai.api_server     --model meta-llama/Llama-3-8B     --enable-prefix-caching
```
*Architecture details:* vLLM hashes the input token IDs. If a sequence prefix matches a hash in the APC pool, the KV cache block pointer is referenced instead of recomputed.
- **Speedup**: TTFT (Time to First Token) goes from 2.5s -> 0.05s for heavy system prompts.

---

## 9. Speculative Decoding

Increases Inter-Token Latency (ITL) by using a smaller "draft" model to predict tokens, and the large "target" model to verify them in parallel.

```bash
# Target model: 70B, Draft model: 8B
python -m vllm.entrypoints.openai.api_server     --model meta-llama/Llama-3-70B-Instruct     --tensor-parallel-size 4     --speculative-model meta-llama/Llama-3-8B-Instruct     --num-speculative-tokens 5     --use-v2-block-manager
```
**When to use:** When serving is totally bound by latency (batch size is 1 or small) rather than throughput.
**When NOT to use:** Under peak heavy load (1000s of requests/sec), Spec Decoding hurts overall system throughput because the draft model wastes ALU cycles.

---

## 10. Structured Outputs (JSON Schema & Outlines)

vLLM integrates `outlines` and `lm-format-enforcer` for guaranteed structured generation via Finite State Machine (FSM) masking.

### 10.1 Regex & JSON Generation
Via the OpenAI SDK, pass `extra_body`.
```python
response = client.completions.create(
    model="meta-llama/Llama-3-8B",
    prompt="Generate a user JSON for John Doe, age 30:",
    max_tokens=100,
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    }
)
```
**Mechanism:** The logits processor intercepts the generation. If the model tries to output a token that violates the JSON schema, its logit is set to `-inf`. 
**Performance Hit:** Heavy schemas can increase CPU overhead during logit masking.

---

## 11. Chunked Prefill

During a generation, if a new request with a 32,000 token prompt arrives, a traditional engine would block all currently generating sequences while calculating the massive prompt (Prefill phase). 

Chunked Prefill breaks the 32k prompt into smaller chunks (e.g., 4096 tokens), intertwining them with decode iterations of other requests.
```bash
python -m vllm.entrypoints.openai.api_server     --model meta-llama/Llama-3-8B     --enable-chunked-prefill     --max-num-batched-tokens 4096
```
- **Result:** Drastically stabilizes P99 decoding latency for streaming users.

---

## 12. Monitoring with Prometheus & Grafana

vLLM exposes Prometheus /metrics by default on port 8000.
**Critical Metrics to Track:**
1. `vllm:num_requests_running`: Currently active requests.
2. `vllm:num_requests_waiting`: Requests queued because KV cache is full.
3. `vllm:gpu_cache_usage_perc`: If this consistently hits 1.0 (100%), you are bottlenecked and need higher `tensor-parallel-size` or smaller models.
4. `vllm:time_to_first_token_seconds` (Histogram)
5. `vllm:time_per_output_token_seconds` (Histogram)

---

## 13. Deep Troubleshooting Guide

### 13.1 NCCL Timeout on Multi-Node
**Error**: `Watchdog caught collective operation timeout: WorkNCCL`
**Cause**: InfiniBand or specific Network interfaces blocking rapid tensor parallelism handshakes.
**Fix**: 
```bash
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_P2P_DISABLE=1 # For debugging, disables direct PCIe/NVLink (hurts perf)
export NCCL_DEBUG=INFO # Highly verbose
```

### 13.2 "KV Cache Blocks Exhausted during Decode"
**Error**: `vllm.engine.metrics warning: Preemption occurred...`
**Cause**: The sequence grew longer than expected and there are no free blocks. vLLM must "preempt" (pause and write to CPU RAM, or recompute entirely).
**Fix**: 
- Reduce `--max-num-seqs` (forces fewer concurrent users, saving memory).
- Use `--swap-space 4` to allocate 4GiB of CPU RAM to offload paused sequences rather than aborting them.

### 13.3 CUDA Graph Error
**Error**: `RuntimeError: CUDA error: operation not permitted` during graph capture phase.
**Cause**: Incompatible torch versions or dynamic control flows (like LoRA switching) breaking static graph assumptions.
**Fix**: Disable CUDA graphs temporarily with `--enforce-eager`.

---

## 14. Architecture Review & System Sizing

To serve a model, you need:
`Total VRAM = Model Weights (Parameters * bytes_per_param) + KV Cache + Activation Memory`
- **7B Model (FP16)**: ~14GB weights. Needs 1x 24GB GPU (RTX 3090, A10G, L4).
- **70B Model (FP16)**: ~140GB weights. Needs 2x 80GB GPUs or 4x 40GB GPUs (A100, H100).
- **8x22B MoE (FP16)**: ~280GB weights. Needs 4x 80GB GPUs.

**Optimal Scaling Pathway:**
1. Hit VRAM limits -> Quantize to AWQ/FP8.
2. Hit KV Cache limits -> Use APC, FP8 KV-Cache, and restrict `max_model_len`.
3. Hit Latency limits (TTFT/ITL) -> Increase Tensor Parallelism, use Speculative Decoding.

---
EOF