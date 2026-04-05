# Master Agentic Skill: LLM Quantization (AWQ, GPTQ, GGUF, Runtime Validation)

## 1. Mission
Deliver predictable quantization workflows that preserve quality while reducing memory and latency, with clear fallback plans per backend.

## 2. Principles
- Prioritize reproducibility over one-off wins.
- Log every configuration that can alter behavior.
- Validate quality and latency together; never optimize one blindly.
- Keep rollback paths documented and tested.
- Treat safety and governance checks as first-class production requirements.

## 3. Source Index (Docs and Blogs)
1. https://huggingface.co/docs/transformers/main/en/quantization
2. https://huggingface.co/docs/peft/developer_guides/quantization
3. https://huggingface.co/blog/4bit-transformers-bitsandbytes
4. https://huggingface.co/papers/2305.14314
5. https://huggingface.co/papers/2306.00978
6. https://huggingface.co/papers/2210.17323
7. https://github.com/ggerganov/llama.cpp
8. https://github.com/ModelCloud/GPTQModel
9. https://github.com/vllm-project/vllm
10. https://huggingface.co/docs/diffusers/main/en/quantization/overview

## 4. Fast Documentation Fetch Commands
Use these commands when someone reports issues and you need to verify behavior against upstream docs quickly.

```bash
mkdir -p /tmp/skill_refs
curl -L "https://huggingface.co/docs/transformers/main/en/quantization" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_en_quantization.html
curl -L "https://huggingface.co/docs/peft/developer_guides/quantization" -o /tmp/skill_refs/huggingface.co_docs_peft_developer_guides_quantization.html
curl -L "https://huggingface.co/blog/4bit-transformers-bitsandbytes" -o /tmp/skill_refs/huggingface.co_blog_4bit-transformers-bitsandbytes.html
curl -L "https://huggingface.co/papers/2305.14314" -o /tmp/skill_refs/huggingface.co_papers_2305.14314.html
curl -L "https://huggingface.co/papers/2306.00978" -o /tmp/skill_refs/huggingface.co_papers_2306.00978.html
curl -L "https://huggingface.co/papers/2210.17323" -o /tmp/skill_refs/huggingface.co_papers_2210.17323.html
curl -L "https://github.com/ggerganov/llama.cpp" -o /tmp/skill_refs/github.com_ggerganov_llama.cpp.html
curl -L "https://github.com/ModelCloud/GPTQModel" -o /tmp/skill_refs/github.com_ModelCloud_GPTQModel.html
curl -L "https://github.com/vllm-project/vllm" -o /tmp/skill_refs/github.com_vllm-project_vllm.html
curl -L "https://huggingface.co/docs/diffusers/main/en/quantization/overview" -o /tmp/skill_refs/huggingface.co_docs_diffusers_main_en_quantization_overview.html
ls -lh /tmp/skill_refs
```

## 5. Operational Policies
Use this section as the mandatory baseline policy set for LLM quantization.

### 5.1 Metrics that must always be tracked
- perplexity_delta_vs_fp16
- exact_match_delta_vs_fp16
- throughput_tokens_per_second
- ttft_ms
- gpu_memory_reserved_gb
- host_memory_gb
- decoder_error_rate
- generation_hallucination_rate

### 5.2 Guardrails
- Do not release quantized artifacts without fp16 baseline comparison.
- Track quantization config fields in model card and artifact metadata.
- Use representative calibration data from deployment domain.
- Block deployment if perplexity regression exceeds agreed threshold.
- Store conversion scripts and exact tool versions with artifact.
- Test at least one long-context benchmark after quantization.

## 6. Codebook
Each recipe is production-oriented and intentionally explicit.

### Recipe 01: BitsAndBytes 4-bit load with NF4
Use this for memory-constrained inference and adapter training workflows.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "mistralai/Mistral-7B-v0.1"
qcfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=qcfg,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
print(tok("hello", return_tensors="pt").input_ids.shape)
```

Notes:
- Use nf4 for QLoRA-style workflows.
- Benchmark quality against fp16 baseline before rollout.

### Recipe 02: AutoAWQ calibration and save
Use this for AWQ artifact creation with explicit quantization parameters.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-2-7b-chat-hf"
out_dir = "artifacts/llama2-awq"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoAWQForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}
model.quantize(tok, quant_config=quant_config)
model.save_quantized(out_dir)
tok.save_pretrained(out_dir)
```

Notes:
- Keep calibration prompts representative of serving traffic.
- Validate tokenizer artifact is stored with quantized model.

### Recipe 03: GPTQ quantization config example
Use this when relying on GPTQ toolchain and config-driven quantization.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tok = AutoTokenizer.from_pretrained(model_id)
gptq = GPTQConfig(bits=4, group_size=128, dataset="wikitext2", tokenizer=tok)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=gptq,
    device_map="auto",
)
model.save_pretrained("artifacts/opt-125m-gptq")
tok.save_pretrained("artifacts/opt-125m-gptq")
```

Notes:
- Prefer GPTQModel over deprecated AutoGPTQ paths where possible.
- Re-run eval matrix if group_size changes.

### Recipe 04: GGUF conversion and quantization with llama.cpp
Use this for edge deployment artifacts and local CPU/GPU inference.

```bash
set -euo pipefail
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

python convert_hf_to_gguf.py /path/to/hf_model --outfile model-f16.gguf --outtype f16
./quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
./llama-cli -m model-q4_k_m.gguf -p "Write a concise release note." -n 128
```

Notes:
- Track exact llama.cpp commit used for conversion.
- Re-validate prompt-format compatibility after conversion.

### Recipe 05: vLLM serving with AWQ model
Use this for OpenAI-compatible serving with quantized checkpoints.

```bash
python -m vllm.entrypoints.openai.api_server               --model TheBloke/Mixtral-8x7B-v0.1-AWQ               --quantization awq               --host 0.0.0.0               --port 8000               --gpu-memory-utilization 0.92
```

Notes:
- Always measure TTFT and output quality after enabling quantization.
- Adjust max_model_len to avoid cache pressure regressions.

## 7. Failure and Recovery Matrix
This matrix is intentionally exhaustive. Follow one case at a time and log every change.

### Case 001: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 002: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 003: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 004: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 005: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 006: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: run short and long sequence benchmark suites
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 007: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 008: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 009: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 010: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 011: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 012: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 013: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 014: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: run short and long sequence benchmark suites
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 015: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 016: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 017: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 018: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 019: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 020: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 021: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 022: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: run short and long sequence benchmark suites
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 023: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 024: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 025: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 026: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 027: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 028: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 029: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 030: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: run short and long sequence benchmark suites
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 031: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 032: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 033: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 034: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 035: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 036: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 037: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 038: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: run short and long sequence benchmark suites
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 039: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 040: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 041: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 042: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 043: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 044: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 045: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 046: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: run short and long sequence benchmark suites
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 047: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 048: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 049: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 050: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 051: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 052: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 053: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 054: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: run short and long sequence benchmark suites
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 055: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 056: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 057: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 058: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 059: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 060: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 061: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 062: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: run short and long sequence benchmark suites
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 063: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 064: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 065: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 066: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 067: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 068: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 069: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 070: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: run short and long sequence benchmark suites
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 071: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 072: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 073: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 074: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 075: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 076: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 077: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 078: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: run short and long sequence benchmark suites
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 079: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 080: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 081: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 082: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 083: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 084: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 085: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 086: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: run short and long sequence benchmark suites
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 087: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 088: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 089: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 090: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 091: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 092: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 093: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 094: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: run short and long sequence benchmark suites
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 095: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 096: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 097: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 098: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 099: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 100: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 101: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 102: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: run short and long sequence benchmark suites
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 103: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 104: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 105: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 106: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 107: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 108: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 109: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 110: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: run short and long sequence benchmark suites
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 111: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 112: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 113: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 114: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 115: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 116: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 117: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 118: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: run short and long sequence benchmark suites
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 119: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 120: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 121: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 122: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 123: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 124: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 125: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 126: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: run short and long sequence benchmark suites
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 127: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 128: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 129: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 130: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 131: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 132: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 133: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 134: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: run short and long sequence benchmark suites
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 135: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 136: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 137: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 138: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 139: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 140: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 141: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 142: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: run short and long sequence benchmark suites
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 143: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 144: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 145: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 146: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 147: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 148: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 149: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 150: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: run short and long sequence benchmark suites
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 151: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 152: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 153: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 154: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 155: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 156: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 157: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 158: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: run short and long sequence benchmark suites
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 159: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 160: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 161: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 162: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 163: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 164: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 165: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 166: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: run short and long sequence benchmark suites
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 167: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 168: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 169: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 170: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 171: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 172: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 173: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 174: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: run short and long sequence benchmark suites
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 175: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 176: long-context generation degrades
- Signal: long-context generation degrades
- Likely cause: KV cache constraints not tuned post-quantization
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 177: artifact size unexpectedly large
- Signal: artifact size unexpectedly large
- Likely cause: incorrect conversion pipeline or stale scripts
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 178: inference crashes on specific sequence lengths
- Signal: inference crashes on specific sequence lengths
- Likely cause: unsupported quant backend for model architecture
- Immediate action: rebuild calibration corpus from recent serving logs
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 179: decoder repeats tokens aggressively
- Signal: decoder repeats tokens aggressively
- Likely cause: rope or position scaling mismatch after conversion
- Immediate action: pin backend versions and rerun smoke tests
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

### Case 180: merged adapters produce corrupted outputs
- Signal: merged adapters produce corrupted outputs
- Likely cause: adapter merge attempted on unsupported quantized base
- Immediate action: verify artifact completeness including tokenizer and config
- Verification metric: no deterministic crash on sequence length sweep

```bash
python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json
```

### Case 181: perplexity jumps after quantization
- Signal: perplexity jumps after quantization
- Likely cause: calibration set does not represent production distribution
- Immediate action: adjust kv cache dtype and max sequence length
- Verification metric: perplexity delta remains within approved threshold

```bash
python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'
```

### Case 182: output quality drops on domain prompts
- Signal: output quality drops on domain prompts
- Likely cause: group_size or bit-width too aggressive for target model
- Immediate action: run short and long sequence benchmark suites
- Verification metric: domain benchmark accuracy remains within approved threshold

```bash
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
print('model_type', c.model_type)
PY
```

### Case 183: runtime kernel incompatibility error
- Signal: runtime kernel incompatibility error
- Likely cause: backend version mismatch with quantized artifact format
- Immediate action: compare outputs against fp16 golden prompts
- Verification metric: TTFT and throughput improve or stay neutral

```bash
python - <<'PY'
import os
for p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:
    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])
PY
```

### Case 184: quantized model fails to load
- Signal: quantized model fails to load
- Likely cause: missing tokenizer files in artifact directory
- Immediate action: disable unsupported merge path and keep adapters separate
- Verification metric: artifact loads successfully across all target runtimes

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Case 185: throughput is lower than fp16
- Signal: throughput is lower than fp16
- Likely cause: compute dtype mismatch for selected kernels
- Immediate action: re-quantize with larger group_size or safer bit-width
- Verification metric: long-context task regression remains bounded

```bash
python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json
```

## 8. Validation Drills
Complete every drill before promoting a change to production.

- [ ] Drill 001: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 002: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 003: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 004: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 005: Record quantization config hash and include in release notes.
- [ ] Drill 006: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 007: Run memory and latency profile under realistic concurrency.
- [ ] Drill 008: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 009: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 010: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 011: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 012: Record quantization config hash and include in release notes.
- [ ] Drill 013: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 014: Run memory and latency profile under realistic concurrency.
- [ ] Drill 015: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 016: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 017: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 018: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 019: Record quantization config hash and include in release notes.
- [ ] Drill 020: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 021: Run memory and latency profile under realistic concurrency.
- [ ] Drill 022: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 023: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 024: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 025: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 026: Record quantization config hash and include in release notes.
- [ ] Drill 027: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 028: Run memory and latency profile under realistic concurrency.
- [ ] Drill 029: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 030: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 031: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 032: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 033: Record quantization config hash and include in release notes.
- [ ] Drill 034: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 035: Run memory and latency profile under realistic concurrency.
- [ ] Drill 036: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 037: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 038: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 039: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 040: Record quantization config hash and include in release notes.
- [ ] Drill 041: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 042: Run memory and latency profile under realistic concurrency.
- [ ] Drill 043: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 044: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 045: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 046: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 047: Record quantization config hash and include in release notes.
- [ ] Drill 048: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 049: Run memory and latency profile under realistic concurrency.
- [ ] Drill 050: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 051: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 052: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 053: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 054: Record quantization config hash and include in release notes.
- [ ] Drill 055: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 056: Run memory and latency profile under realistic concurrency.
- [ ] Drill 057: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 058: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 059: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 060: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 061: Record quantization config hash and include in release notes.
- [ ] Drill 062: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 063: Run memory and latency profile under realistic concurrency.
- [ ] Drill 064: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 065: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 066: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 067: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 068: Record quantization config hash and include in release notes.
- [ ] Drill 069: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 070: Run memory and latency profile under realistic concurrency.
- [ ] Drill 071: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 072: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 073: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 074: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 075: Record quantization config hash and include in release notes.
- [ ] Drill 076: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 077: Run memory and latency profile under realistic concurrency.
- [ ] Drill 078: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 079: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 080: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 081: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 082: Record quantization config hash and include in release notes.
- [ ] Drill 083: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 084: Run memory and latency profile under realistic concurrency.
- [ ] Drill 085: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 086: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 087: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 088: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 089: Record quantization config hash and include in release notes.
- [ ] Drill 090: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 091: Run memory and latency profile under realistic concurrency.
- [ ] Drill 092: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 093: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 094: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 095: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 096: Record quantization config hash and include in release notes.
- [ ] Drill 097: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 098: Run memory and latency profile under realistic concurrency.
- [ ] Drill 099: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 100: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 101: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 102: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 103: Record quantization config hash and include in release notes.
- [ ] Drill 104: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 105: Run memory and latency profile under realistic concurrency.
- [ ] Drill 106: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 107: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 108: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 109: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 110: Record quantization config hash and include in release notes.
- [ ] Drill 111: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 112: Run memory and latency profile under realistic concurrency.
- [ ] Drill 113: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 114: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 115: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 116: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 117: Record quantization config hash and include in release notes.
- [ ] Drill 118: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 119: Run memory and latency profile under realistic concurrency.
- [ ] Drill 120: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 121: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 122: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 123: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 124: Record quantization config hash and include in release notes.
- [ ] Drill 125: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 126: Run memory and latency profile under realistic concurrency.
- [ ] Drill 127: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 128: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 129: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 130: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 131: Record quantization config hash and include in release notes.
- [ ] Drill 132: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 133: Run memory and latency profile under realistic concurrency.
- [ ] Drill 134: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 135: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 136: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 137: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 138: Record quantization config hash and include in release notes.
- [ ] Drill 139: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 140: Run memory and latency profile under realistic concurrency.
- [ ] Drill 141: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 142: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 143: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 144: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 145: Record quantization config hash and include in release notes.
- [ ] Drill 146: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 147: Run memory and latency profile under realistic concurrency.
- [ ] Drill 148: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 149: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 150: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 151: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 152: Record quantization config hash and include in release notes.
- [ ] Drill 153: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 154: Run memory and latency profile under realistic concurrency.
- [ ] Drill 155: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 156: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 157: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 158: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 159: Record quantization config hash and include in release notes.
- [ ] Drill 160: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 161: Run memory and latency profile under realistic concurrency.
- [ ] Drill 162: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 163: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 164: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 165: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 166: Record quantization config hash and include in release notes.
- [ ] Drill 167: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 168: Run memory and latency profile under realistic concurrency.
- [ ] Drill 169: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 170: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 171: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 172: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 173: Record quantization config hash and include in release notes.
- [ ] Drill 174: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 175: Run memory and latency profile under realistic concurrency.
- [ ] Drill 176: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 177: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 178: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 179: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 180: Record quantization config hash and include in release notes.
- [ ] Drill 181: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 182: Run memory and latency profile under realistic concurrency.
- [ ] Drill 183: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 184: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 185: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 186: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 187: Record quantization config hash and include in release notes.
- [ ] Drill 188: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 189: Run memory and latency profile under realistic concurrency.
- [ ] Drill 190: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 191: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 192: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 193: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 194: Record quantization config hash and include in release notes.
- [ ] Drill 195: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 196: Run memory and latency profile under realistic concurrency.
- [ ] Drill 197: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 198: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 199: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 200: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 201: Record quantization config hash and include in release notes.
- [ ] Drill 202: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 203: Run memory and latency profile under realistic concurrency.
- [ ] Drill 204: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 205: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 206: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 207: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 208: Record quantization config hash and include in release notes.
- [ ] Drill 209: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 210: Run memory and latency profile under realistic concurrency.
- [ ] Drill 211: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 212: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 213: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 214: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 215: Record quantization config hash and include in release notes.
- [ ] Drill 216: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 217: Run memory and latency profile under realistic concurrency.
- [ ] Drill 218: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 219: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 220: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 221: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 222: Record quantization config hash and include in release notes.
- [ ] Drill 223: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 224: Run memory and latency profile under realistic concurrency.
- [ ] Drill 225: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 226: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 227: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 228: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 229: Record quantization config hash and include in release notes.
- [ ] Drill 230: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 231: Run memory and latency profile under realistic concurrency.
- [ ] Drill 232: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 233: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 234: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 235: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 236: Record quantization config hash and include in release notes.
- [ ] Drill 237: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 238: Run memory and latency profile under realistic concurrency.
- [ ] Drill 239: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 240: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 241: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 242: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 243: Record quantization config hash and include in release notes.
- [ ] Drill 244: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 245: Run memory and latency profile under realistic concurrency.
- [ ] Drill 246: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 247: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 248: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 249: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 250: Record quantization config hash and include in release notes.
- [ ] Drill 251: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 252: Run memory and latency profile under realistic concurrency.
- [ ] Drill 253: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 254: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 255: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 256: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 257: Record quantization config hash and include in release notes.
- [ ] Drill 258: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 259: Run memory and latency profile under realistic concurrency.
- [ ] Drill 260: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 261: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 262: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 263: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 264: Record quantization config hash and include in release notes.
- [ ] Drill 265: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 266: Run memory and latency profile under realistic concurrency.
- [ ] Drill 267: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 268: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 269: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 270: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 271: Record quantization config hash and include in release notes.
- [ ] Drill 272: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 273: Run memory and latency profile under realistic concurrency.
- [ ] Drill 274: Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.
- [ ] Drill 275: Evaluate perplexity on at least two datasets, not just one benchmark.
- [ ] Drill 276: Run sequence length sweep from 256 to max_model_len in fixed increments.
- [ ] Drill 277: Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.
- [ ] Drill 278: Record quantization config hash and include in release notes.
- [ ] Drill 279: Verify prompt template behavior remains unchanged after conversion.
- [ ] Drill 280: Run memory and latency profile under realistic concurrency.

