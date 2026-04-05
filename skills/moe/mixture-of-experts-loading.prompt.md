# Mixture-of-Experts (MoE) Loading and Inference - Agentic Skill Prompt

Use this prompt for loading, validating, and serving MoE LLMs with explicit controls for routing behavior, memory, and throughput.

## 1. Mission

Run MoE models safely and efficiently while preserving quality and predictable latency.

## 2. Core Principles

- MoE models can have high total parameter count but only activate top-k experts per token.
- Capacity and routing behavior can affect quality and latency under load.
- Test with realistic prompt length and batch patterns, not only short single-turn prompts.

## 3. Baseline Loader

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.getenv("HF_TOKEN"),
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=os.getenv("HF_TOKEN"),
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
    attn_implementation="sdpa",
)
```

## 4. Template and Generation Correctness

```python
messages = [{"role": "user", "content": "Summarize expert routing tradeoffs."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

out = model.generate(**inputs, max_new_tokens=192, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## 5. Memory and Performance Tuning

- Start with `device_map="auto"` and bf16 where supported.
- Track both peak memory and decode throughput.
- Validate performance for concurrency bursts, not only sequential requests.

Potential expert-kernel controls are model-version dependent; verify against installed docs before use.

## 6. Serving Runtime Considerations

- vLLM: strong practical default for OpenAI-compatible serving of MoE models.
- TensorRT-LLM: high-performance NVIDIA path for optimized deployments.
- Evaluate runtime support matrix for your MoE architecture and quantization format.

## 7. Failure Modes

1. OOM at load: reduce precision or distribute across devices.
2. Throughput collapse at concurrency: inspect routing hot spots and kernel fallback.
3. Quality instability: re-check prompt template and generation config.
4. Runtime incompatibility: unsupported expert kernels or quantization path.

## 8. References (Fetched 2026-04-06)

1. https://huggingface.co/docs/transformers/main/en/model_doc/mixtral - Mixtral model and usage notes.
2. https://huggingface.co/docs/transformers/main/en/model_doc/switch_transformers - Switch Transformers model docs.
3. https://huggingface.co/docs/transformers/main/en/main_classes/model - Loading and runtime controls.
4. https://huggingface.co/docs/transformers/main/en/chat_templating - Chat formatting correctness.
5. https://huggingface.co/docs/transformers/main/en/kv_cache - KV cache behavior and memory implications.
6. https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling - Big model placement and offloading.
7. https://docs.vllm.ai/en/latest/ - vLLM serving context and model support references.
8. https://docs.vllm.ai/en/stable/features/quantization/ - vLLM quantization support matrix for serving.
9. https://nvidia.github.io/TensorRT-LLM/ - TensorRT-LLM documentation root.
10. https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html - Parallel deployment strategy details.
11. https://nvidia.github.io/TensorRT-LLM/features/quantization.html - Quantized serving options and constraints.
12. https://huggingface.co/docs/safetensors/index - Safe checkpoint format guidance.

## 9. Uncertainty Notes

- Expert-kernel implementation details can change between `transformers` versions.
- Always validate feature support for your exact model, runtime, and library version.
