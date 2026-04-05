# Master Agentic Skill: Mixture of Experts (MoE) Loading, Routing, and Operations

## 1. Mission
Run sparse expert models reliably by controlling routing behavior, memory placement, and throughput under realistic production traffic.

## 2. Principles
- Prioritize reproducibility over one-off wins.
- Log every configuration that can alter behavior.
- Validate quality and latency together; never optimize one blindly.
- Keep rollback paths documented and tested.
- Treat safety and governance checks as first-class production requirements.

## 3. Source Index (Docs and Blogs)
1. https://huggingface.co/docs/transformers/main/en/model_doc/mixtral
2. https://mistral.ai/news/mixtral-of-experts/
3. https://huggingface.co/blog/moe
4. https://huggingface.co/docs/transformers/en/kv_cache
5. https://huggingface.co/docs/transformers/chat_templating
6. https://huggingface.co/docs/transformers/main/en/quantization
7. https://github.com/vllm-project/vllm
8. https://github.com/mistralai/mistral-src
9. https://huggingface.co/papers/2401.04088

## 4. Fast Documentation Fetch Commands
Use these commands when someone reports issues and you need to verify behavior against upstream docs quickly.

```bash
mkdir -p /tmp/skill_refs
curl -L "https://huggingface.co/docs/transformers/main/en/model_doc/mixtral" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_en_model_doc_mixtral.html
curl -L "https://mistral.ai/news/mixtral-of-experts/" -o /tmp/skill_refs/mistral.ai_news_mixtral-of-experts_.html
curl -L "https://huggingface.co/blog/moe" -o /tmp/skill_refs/huggingface.co_blog_moe.html
curl -L "https://huggingface.co/docs/transformers/en/kv_cache" -o /tmp/skill_refs/huggingface.co_docs_transformers_en_kv_cache.html
curl -L "https://huggingface.co/docs/transformers/chat_templating" -o /tmp/skill_refs/huggingface.co_docs_transformers_chat_templating.html
curl -L "https://huggingface.co/docs/transformers/main/en/quantization" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_en_quantization.html
curl -L "https://github.com/vllm-project/vllm" -o /tmp/skill_refs/github.com_vllm-project_vllm.html
curl -L "https://github.com/mistralai/mistral-src" -o /tmp/skill_refs/github.com_mistralai_mistral-src.html
curl -L "https://huggingface.co/papers/2401.04088" -o /tmp/skill_refs/huggingface.co_papers_2401.04088.html
ls -lh /tmp/skill_refs
```

## 5. Operational Policies
Use this section as the mandatory baseline policy set for Mixture of Experts operations.

### 5.1 Metrics that must always be tracked
- tokens_per_second
- router_entropy
- expert_utilization_std
- aux_router_loss
- z_loss
- oom_event_count
- ttft_ms
- p95_generation_latency_ms

### 5.2 Guardrails
- Track router stats in every benchmark and production canary.
- Block release if expert utilization collapses to small subset.
- Do not tune routing hyperparameters without fixed prompt benchmark.
- Keep separate dashboards for model quality and routing quality.
- Validate tokenizer template compatibility for instruct checkpoints.
- Run long-context and high-concurrency tests before promotion.

## 6. Codebook
Each recipe is production-oriented and intentionally explicit.

### Recipe 01: Load Mixtral instruct checkpoint with automatic placement
Use this as the minimal reliable baseline for MoE inference.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Give me three robust deployment tips."}]
inputs = tok.apply_chat_template(messages, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

Notes:
- Instruct checkpoints require chat template formatting.
- Start with deterministic decoding for debugging.

### Recipe 02: Enable router logits during training diagnostics
Use this to monitor expert load balancing and prevent routing collapse.

```python
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "mistralai/Mixtral-8x7B-v0.1"
cfg = AutoConfig.from_pretrained(model_id)
cfg.output_router_logits = True

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=cfg,
    device_map="auto",
)

print("output_router_logits", model.config.output_router_logits)
```

Notes:
- Router logits are useful for diagnostics and auxiliary routing losses.
- Disable unnecessary diagnostics in latency-critical serving paths.

### Recipe 03: Quantized MoE loading for constrained hardware
Use this when full precision does not fit your hardware envelope.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",
)
print(type(model))
```

Notes:
- Benchmark quality regressions after quantization.
- Confirm router behavior remains healthy with quantized weights.

### Recipe 04: vLLM server launch for Mixtral
Use this for OpenAI-compatible production serving with high throughput.

```bash
python -m vllm.entrypoints.openai.api_server               --model mistralai/Mixtral-8x7B-Instruct-v0.1               --tensor-parallel-size 2               --max-model-len 8192               --gpu-memory-utilization 0.92               --host 0.0.0.0               --port 8000
```

Notes:
- Profile queue depth and KV cache usage under realistic load.
- Pin model revision to avoid accidental drift.

### Recipe 05: Flash Attention setup for faster MoE inference
Use this where compatible hardware and kernels are available.

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
print(model.config.model_type)
```

Notes:
- Confirm flash-attn version supports sliding window behavior.
- Fallback gracefully if target GPU is unsupported.

## 7. Failure and Recovery Matrix
This matrix is intentionally exhaustive. Follow one case at a time and log every change.

### Case 001: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: expert utilization variance remains within expected range

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 002: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: router entropy is stable across benchmark groups

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 003: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 004: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 005: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: p95 latency remains under SLO

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 006: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: quality benchmark score remains within acceptance band

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 007: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: expert utilization variance remains within expected range

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 008: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 009: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 010: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 011: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: p95 latency remains under SLO

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 012: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: quality benchmark score remains within acceptance band

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 013: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 014: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 015: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: aux router loss and z-loss remain bounded

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 016: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 017: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: p95 latency remains under SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 018: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 019: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 020: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: router entropy is stable across benchmark groups

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 021: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: aux router loss and z-loss remain bounded

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 022: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 023: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 024: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 025: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: expert utilization variance remains within expected range

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 026: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: router entropy is stable across benchmark groups

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 027: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: aux router loss and z-loss remain bounded

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 028: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 029: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 030: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: quality benchmark score remains within acceptance band

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 031: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: expert utilization variance remains within expected range

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 032: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: router entropy is stable across benchmark groups

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 033: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 034: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 035: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: p95 latency remains under SLO

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 036: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: quality benchmark score remains within acceptance band

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 037: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: expert utilization variance remains within expected range

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 038: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 039: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 040: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 041: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: p95 latency remains under SLO

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 042: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: quality benchmark score remains within acceptance band

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 043: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 044: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 045: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: aux router loss and z-loss remain bounded

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 046: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 047: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: p95 latency remains under SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 048: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 049: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 050: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: router entropy is stable across benchmark groups

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 051: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: aux router loss and z-loss remain bounded

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 052: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 053: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 054: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 055: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: expert utilization variance remains within expected range

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 056: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: router entropy is stable across benchmark groups

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 057: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: aux router loss and z-loss remain bounded

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 058: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 059: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 060: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: quality benchmark score remains within acceptance band

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 061: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: expert utilization variance remains within expected range

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 062: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: router entropy is stable across benchmark groups

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 063: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 064: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 065: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: p95 latency remains under SLO

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 066: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: quality benchmark score remains within acceptance band

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 067: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: expert utilization variance remains within expected range

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 068: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 069: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 070: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 071: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: p95 latency remains under SLO

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 072: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: quality benchmark score remains within acceptance band

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 073: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 074: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 075: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: aux router loss and z-loss remain bounded

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 076: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 077: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: p95 latency remains under SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 078: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 079: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 080: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: router entropy is stable across benchmark groups

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 081: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: aux router loss and z-loss remain bounded

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 082: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 083: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 084: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 085: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: expert utilization variance remains within expected range

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 086: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: router entropy is stable across benchmark groups

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 087: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: aux router loss and z-loss remain bounded

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 088: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 089: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 090: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: quality benchmark score remains within acceptance band

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 091: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: expert utilization variance remains within expected range

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 092: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: router entropy is stable across benchmark groups

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 093: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 094: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 095: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: p95 latency remains under SLO

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 096: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: quality benchmark score remains within acceptance band

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 097: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: expert utilization variance remains within expected range

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 098: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 099: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 100: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 101: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: p95 latency remains under SLO

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 102: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: quality benchmark score remains within acceptance band

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 103: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 104: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 105: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: aux router loss and z-loss remain bounded

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 106: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 107: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: p95 latency remains under SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 108: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 109: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 110: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: router entropy is stable across benchmark groups

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 111: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: aux router loss and z-loss remain bounded

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 112: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 113: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 114: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 115: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: expert utilization variance remains within expected range

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 116: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: router entropy is stable across benchmark groups

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 117: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: aux router loss and z-loss remain bounded

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 118: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 119: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 120: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: quality benchmark score remains within acceptance band

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 121: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: expert utilization variance remains within expected range

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 122: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: router entropy is stable across benchmark groups

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 123: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 124: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 125: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: p95 latency remains under SLO

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 126: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: quality benchmark score remains within acceptance band

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 127: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: expert utilization variance remains within expected range

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 128: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 129: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 130: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 131: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: p95 latency remains under SLO

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 132: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: quality benchmark score remains within acceptance band

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 133: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 134: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 135: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: aux router loss and z-loss remain bounded

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 136: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 137: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: p95 latency remains under SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 138: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 139: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 140: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: router entropy is stable across benchmark groups

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 141: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: aux router loss and z-loss remain bounded

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 142: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 143: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 144: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 145: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: expert utilization variance remains within expected range

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 146: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: router entropy is stable across benchmark groups

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 147: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: aux router loss and z-loss remain bounded

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 148: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 149: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 150: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: quality benchmark score remains within acceptance band

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 151: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: expert utilization variance remains within expected range

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 152: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: router entropy is stable across benchmark groups

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 153: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 154: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 155: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: p95 latency remains under SLO

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 156: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: quality benchmark score remains within acceptance band

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 157: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: expert utilization variance remains within expected range

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 158: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 159: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 160: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 161: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: p95 latency remains under SLO

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 162: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: quality benchmark score remains within acceptance band

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 163: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 164: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: router entropy is stable across benchmark groups

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 165: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: aux router loss and z-loss remain bounded

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 166: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 167: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: p95 latency remains under SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 168: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 169: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: expert utilization variance remains within expected range

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 170: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: router entropy is stable across benchmark groups

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 171: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: aux router loss and z-loss remain bounded

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 172: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 173: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 174: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: quality benchmark score remains within acceptance band

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 175: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: expert utilization variance remains within expected range

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 176: inference latency has large tail spikes
- Signal: inference latency has large tail spikes
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: router entropy is stable across benchmark groups

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 177: OOM triggered on certain prompt shapes
- Signal: OOM triggered on certain prompt shapes
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: aux router loss and z-loss remain bounded

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 178: chat output format becomes inconsistent
- Signal: chat output format becomes inconsistent
- Likely cause: insufficient or mismatched fine-tuning data distribution
- Immediate action: run controlled prompt pack to isolate regression class
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 179: model overuses repetitive token patterns
- Signal: model overuses repetitive token patterns
- Likely cause: incomplete telemetry for expert-level metrics
- Immediate action: tighten max sequence length and retune batching policy
- Verification metric: p95 latency remains under SLO

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 180: quality regresses after quantized deployment
- Signal: quality regresses after quantized deployment
- Likely cause: memory fragmentation from dynamic batch composition
- Immediate action: align chat templating with model family and test stop tokens
- Verification metric: quality benchmark score remains within acceptance band

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

### Case 181: expert utilization is highly skewed
- Signal: expert utilization is highly skewed
- Likely cause: cache policy or max length inconsistent with model assumptions
- Immediate action: roll back to last known-good quantization profile
- Verification metric: expert utilization variance remains within expected range

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('experts_per_tok', c.num_experts_per_tok)
print('num_local_experts', c.num_local_experts)
PY
```

### Case 182: router entropy drops unexpectedly
- Signal: router entropy drops unexpectedly
- Likely cause: chat template mismatch for instruct model
- Immediate action: increase monitoring on p95 and p99 latency for specific route
- Verification metric: router entropy is stable across benchmark groups

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 183: auxiliary routing loss explodes
- Signal: auxiliary routing loss explodes
- Likely cause: kernel/backend incompatibility under load
- Immediate action: validate kernel versions and attention implementation flags
- Verification metric: aux router loss and z-loss remain bounded

```bash
python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128
```

### Case 184: throughput collapses at moderate concurrency
- Signal: throughput collapses at moderate concurrency
- Likely cause: quantization settings too aggressive for routing stability
- Immediate action: repeat canary with deterministic generation settings
- Verification metric: throughput scales with concurrency until planned saturation point

```bash
python benchmark_quality.py --model mixtral --suite regression_pack_v2
```

### Case 185: long-context quality declines sharply
- Signal: long-context quality declines sharply
- Likely cause: router configuration not tuned for current traffic profile
- Immediate action: enable router diagnostics and compare expert usage histograms
- Verification metric: p95 latency remains under SLO

```bash
python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl
```

## 8. Validation Drills
Complete every drill before promoting a change to production.

- [ ] Drill 001: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 002: Validate instruct formatting with chat template golden samples.
- [ ] Drill 003: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 004: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 005: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 006: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 007: Capture and review top failing prompts weekly.
- [ ] Drill 008: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 009: Validate instruct formatting with chat template golden samples.
- [ ] Drill 010: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 011: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 012: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 013: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 014: Capture and review top failing prompts weekly.
- [ ] Drill 015: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 016: Validate instruct formatting with chat template golden samples.
- [ ] Drill 017: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 018: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 019: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 020: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 021: Capture and review top failing prompts weekly.
- [ ] Drill 022: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 023: Validate instruct formatting with chat template golden samples.
- [ ] Drill 024: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 025: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 026: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 027: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 028: Capture and review top failing prompts weekly.
- [ ] Drill 029: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 030: Validate instruct formatting with chat template golden samples.
- [ ] Drill 031: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 032: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 033: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 034: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 035: Capture and review top failing prompts weekly.
- [ ] Drill 036: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 037: Validate instruct formatting with chat template golden samples.
- [ ] Drill 038: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 039: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 040: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 041: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 042: Capture and review top failing prompts weekly.
- [ ] Drill 043: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 044: Validate instruct formatting with chat template golden samples.
- [ ] Drill 045: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 046: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 047: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 048: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 049: Capture and review top failing prompts weekly.
- [ ] Drill 050: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 051: Validate instruct formatting with chat template golden samples.
- [ ] Drill 052: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 053: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 054: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 055: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 056: Capture and review top failing prompts weekly.
- [ ] Drill 057: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 058: Validate instruct formatting with chat template golden samples.
- [ ] Drill 059: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 060: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 061: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 062: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 063: Capture and review top failing prompts weekly.
- [ ] Drill 064: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 065: Validate instruct formatting with chat template golden samples.
- [ ] Drill 066: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 067: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 068: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 069: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 070: Capture and review top failing prompts weekly.
- [ ] Drill 071: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 072: Validate instruct formatting with chat template golden samples.
- [ ] Drill 073: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 074: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 075: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 076: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 077: Capture and review top failing prompts weekly.
- [ ] Drill 078: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 079: Validate instruct formatting with chat template golden samples.
- [ ] Drill 080: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 081: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 082: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 083: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 084: Capture and review top failing prompts weekly.
- [ ] Drill 085: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 086: Validate instruct formatting with chat template golden samples.
- [ ] Drill 087: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 088: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 089: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 090: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 091: Capture and review top failing prompts weekly.
- [ ] Drill 092: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 093: Validate instruct formatting with chat template golden samples.
- [ ] Drill 094: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 095: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 096: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 097: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 098: Capture and review top failing prompts weekly.
- [ ] Drill 099: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 100: Validate instruct formatting with chat template golden samples.
- [ ] Drill 101: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 102: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 103: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 104: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 105: Capture and review top failing prompts weekly.
- [ ] Drill 106: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 107: Validate instruct formatting with chat template golden samples.
- [ ] Drill 108: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 109: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 110: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 111: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 112: Capture and review top failing prompts weekly.
- [ ] Drill 113: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 114: Validate instruct formatting with chat template golden samples.
- [ ] Drill 115: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 116: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 117: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 118: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 119: Capture and review top failing prompts weekly.
- [ ] Drill 120: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 121: Validate instruct formatting with chat template golden samples.
- [ ] Drill 122: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 123: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 124: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 125: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 126: Capture and review top failing prompts weekly.
- [ ] Drill 127: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 128: Validate instruct formatting with chat template golden samples.
- [ ] Drill 129: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 130: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 131: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 132: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 133: Capture and review top failing prompts weekly.
- [ ] Drill 134: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 135: Validate instruct formatting with chat template golden samples.
- [ ] Drill 136: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 137: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 138: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 139: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 140: Capture and review top failing prompts weekly.
- [ ] Drill 141: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 142: Validate instruct formatting with chat template golden samples.
- [ ] Drill 143: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 144: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 145: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 146: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 147: Capture and review top failing prompts weekly.
- [ ] Drill 148: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 149: Validate instruct formatting with chat template golden samples.
- [ ] Drill 150: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 151: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 152: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 153: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 154: Capture and review top failing prompts weekly.
- [ ] Drill 155: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 156: Validate instruct formatting with chat template golden samples.
- [ ] Drill 157: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 158: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 159: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 160: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 161: Capture and review top failing prompts weekly.
- [ ] Drill 162: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 163: Validate instruct formatting with chat template golden samples.
- [ ] Drill 164: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 165: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 166: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 167: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 168: Capture and review top failing prompts weekly.
- [ ] Drill 169: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 170: Validate instruct formatting with chat template golden samples.
- [ ] Drill 171: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 172: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 173: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 174: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 175: Capture and review top failing prompts weekly.
- [ ] Drill 176: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 177: Validate instruct formatting with chat template golden samples.
- [ ] Drill 178: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 179: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 180: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 181: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 182: Capture and review top failing prompts weekly.
- [ ] Drill 183: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 184: Validate instruct formatting with chat template golden samples.
- [ ] Drill 185: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 186: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 187: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 188: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 189: Capture and review top failing prompts weekly.
- [ ] Drill 190: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 191: Validate instruct formatting with chat template golden samples.
- [ ] Drill 192: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 193: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 194: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 195: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 196: Capture and review top failing prompts weekly.
- [ ] Drill 197: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 198: Validate instruct formatting with chat template golden samples.
- [ ] Drill 199: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 200: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 201: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 202: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 203: Capture and review top failing prompts weekly.
- [ ] Drill 204: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 205: Validate instruct formatting with chat template golden samples.
- [ ] Drill 206: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 207: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 208: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 209: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 210: Capture and review top failing prompts weekly.
- [ ] Drill 211: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 212: Validate instruct formatting with chat template golden samples.
- [ ] Drill 213: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 214: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 215: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 216: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 217: Capture and review top failing prompts weekly.
- [ ] Drill 218: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 219: Validate instruct formatting with chat template golden samples.
- [ ] Drill 220: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 221: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 222: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 223: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 224: Capture and review top failing prompts weekly.
- [ ] Drill 225: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 226: Validate instruct formatting with chat template golden samples.
- [ ] Drill 227: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 228: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 229: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 230: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 231: Capture and review top failing prompts weekly.
- [ ] Drill 232: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 233: Validate instruct formatting with chat template golden samples.
- [ ] Drill 234: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 235: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 236: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 237: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 238: Capture and review top failing prompts weekly.
- [ ] Drill 239: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 240: Validate instruct formatting with chat template golden samples.
- [ ] Drill 241: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 242: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 243: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 244: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 245: Capture and review top failing prompts weekly.
- [ ] Drill 246: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 247: Validate instruct formatting with chat template golden samples.
- [ ] Drill 248: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 249: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 250: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 251: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 252: Capture and review top failing prompts weekly.
- [ ] Drill 253: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 254: Validate instruct formatting with chat template golden samples.
- [ ] Drill 255: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 256: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 257: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 258: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 259: Capture and review top failing prompts weekly.
- [ ] Drill 260: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 261: Validate instruct formatting with chat template golden samples.
- [ ] Drill 262: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 263: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 264: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 265: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 266: Capture and review top failing prompts weekly.
- [ ] Drill 267: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 268: Validate instruct formatting with chat template golden samples.
- [ ] Drill 269: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 270: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 271: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 272: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 273: Capture and review top failing prompts weekly.
- [ ] Drill 274: Run router diagnostics on three prompt classes: short, long, and multilingual.
- [ ] Drill 275: Validate instruct formatting with chat template golden samples.
- [ ] Drill 276: Run load test with concurrency ramp and observe utilization vs latency.
- [ ] Drill 277: Evaluate quantized and fp16 variants with same prompt pack.
- [ ] Drill 278: Verify long-context behavior at 4k, 8k, and configured maximum.
- [ ] Drill 279: Run deterministic decode mode to isolate routing vs sampling issues.
- [ ] Drill 280: Capture and review top failing prompts weekly.

