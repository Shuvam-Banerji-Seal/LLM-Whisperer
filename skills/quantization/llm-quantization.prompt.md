# LLM Quantization and Serving Compatibility - Agentic Skill Prompt

Use this prompt for selecting quantization strategies and deploying quantized models across inference runtimes.

## 1. Mission

Pick the right quantization method for target hardware and quality constraints, then verify end-to-end serving behavior.

## 2. Method Taxonomy

- bitsandbytes: 8-bit and 4-bit (NF4, FP4) workflows, especially strong for QLoRA-style adaptation.
- GPTQ: post-training weight quantization requiring calibration.
- AWQ: activation-aware weight quantization with strong serving adoption.
- GGUF: portable quantized artifacts for llama.cpp-style runtimes.
- SmoothQuant and related W8A8 pathways: activation plus weight quantization oriented to deployment efficiency.
- KV cache quantization methods (including KIVI-style ideas): useful when long-context memory dominates.

## 3. Decision Framework

| Goal | Typical Choice |
|---|---|
| Quick VRAM reduction on NVIDIA | bitsandbytes 4-bit |
| Production throughput with pre-quantized artifacts | AWQ or GPTQ |
| NVIDIA datacenter max performance | TensorRT-LLM quant recipes |
| CPU or edge deployment | GGUF plus llama.cpp |
| Long-context memory relief | KV cache quantization after quality validation |

## 4. Calibration and Validation Rules

- Use representative calibration prompts matching production format.
- Include short, medium, and long context in quality checks.
- Compare against a full-precision or higher-precision baseline.
- Report both quality and latency deltas.

## 5. Implementation Snippets

### 5.1 bitsandbytes 4-bit load

```bash
pip install -U transformers accelerate bitsandbytes
```

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=cfg,
    device_map="auto",
)
```

### 5.2 GPTQ flow

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig

tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
gptq_cfg = GPTQConfig(bits=4, dataset="c4", tokenizer=tok)

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    quantization_config=gptq_cfg,
    device_map="auto",
)
model.save_pretrained("opt-125m-gptq")
```

### 5.3 AWQ reference command

```bash
python -m awq.entry --model_path /models/llama3-8b \
  --w_bit 4 --q_group_size 128 \
  --run_awq --dump_awq awq_cache/llama3-8b-w4-g128.pt
```

### 5.4 GGUF conversion and quantization

```bash
python convert_hf_to_gguf.py /models/llama3-8b --outfile llama3-f16.gguf
llama-quantize llama3-f16.gguf llama3-q4_k_m.gguf q4_k_m
```

### 5.5 Serving compatibility examples

```bash
vllm serve TheBloke/Mistral-7B-Instruct-v0.1-AWQ --quantization awq
```

```bash
text-generation-launcher --model-id /data/model-gptq --quantize gptq
```

```python
from tensorrt_llm import LLM
llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-FP8")
print(llm.generate("Hello, my name is"))
```

## 6. Risk Checklist

- Non-representative calibration causes hidden quality loss.
- Long-context regressions may not appear in short benchmark prompts.
- Kernel fallback can erase expected speedups.
- Artifact portability differs across runtimes.
- Some libraries or repos become unmaintained; verify project activity.

## 7. References (Fetched 2026-04-06)

1. https://huggingface.co/docs/transformers/main/quantization/bitsandbytes - Transformers bitsandbytes integration.
2. https://github.com/bitsandbytes-foundation/bitsandbytes - bitsandbytes source and release context.
3. https://arxiv.org/abs/2208.07339 - LLM.int8 paper.
4. https://arxiv.org/abs/2305.14314 - QLoRA paper.
5. https://arxiv.org/abs/2210.17323 - GPTQ paper.
6. https://github.com/IST-DASLab/gptq - GPTQ reference repository.
7. https://huggingface.co/docs/transformers/main/en/quantization/gptq - Transformers GPTQ integration.
8. https://arxiv.org/abs/2306.00978 - AWQ paper.
9. https://github.com/mit-han-lab/llm-awq - AWQ reference implementation.
10. https://huggingface.co/docs/transformers/main/en/quantization/awq - Transformers AWQ loading guidance.
11. https://arxiv.org/abs/2211.10438 - SmoothQuant paper.
12. https://arxiv.org/abs/2402.02750 - KIVI KV-cache quantization paper.
13. https://huggingface.co/docs/transformers/main/en/gguf - GGUF interoperability in Transformers.
14. https://github.com/ggml-org/ggml/blob/master/docs/gguf.md - GGUF format specification.
15. https://github.com/ggml-org/llama.cpp - llama.cpp runtime repository.
16. https://docs.vllm.ai/en/latest/features/quantization/ - vLLM quantization support matrix.
17. https://huggingface.co/docs/text-generation-inference/main/en/conceptual/quantization - TGI quantization concepts.
18. https://nvidia.github.io/TensorRT-LLM/features/quantization.html - TensorRT-LLM quantization reference.
19. https://github.com/NVIDIA/TensorRT-LLM - TensorRT-LLM source repository.
