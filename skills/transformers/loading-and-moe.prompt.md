# Transformers Loading and MoE Operations - Agentic Skill Prompt

Use this prompt for robust Hugging Face Transformers model loading, chat formatting, and Mixture-of-Experts (MoE) handling in inference pipelines.

## 1. Mission

Load models safely and reproducibly, with explicit control over memory placement, precision, templates, and remote code trust.

## 2. Safe Loader Baseline

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

Default posture:

- `trust_remote_code=False` unless the model requires it and code has been reviewed.
- Pin model revision for production reproducibility.
- Prefer safetensors checkpoints when available.

## 3. Chat Template Correctness

Always format with tokenizer-native templates for instruction models.

```python
messages = [{"role": "user", "content": "Explain MoE routing in 3 bullets."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
text = tokenizer.decode(output[0], skip_special_tokens=True)
print(text)
```

Avoid manual prompt stitching unless model documentation explicitly defines the format.

## 4. Memory and Checkpoint Handling

- Use `device_map="auto"` for practical placement defaults.
- For very large models, combine with `max_memory` controls.
- Keep sharded checkpoints intact and version-locked.

Local-only loading pattern:

```python
model = AutoModelForCausalLM.from_pretrained(
    "./local_model_dir",
    local_files_only=True,
    device_map="auto",
)
```

## 5. Attention Implementation Controls

Possible settings include `eager`, `sdpa`, and `flash_attention_2` where supported.

Practical sequence:

1. Start with `sdpa` for stability.
2. Test `flash_attention_2` if environment supports it.
3. Fall back to `eager` on compatibility issues.

## 6. MoE-Specific Guidance

- MoE models may load many total parameters but activate only top-k experts per token.
- Measure both memory and throughput under realistic routing behavior.
- Validate long-context behavior separately from short prompt tests.

Potential model/runtime-specific expert kernels may exist, but they are version dependent; verify against installed library docs before production use.

## 7. Security and Reliability Controls

- Do not enable remote code trust without source review.
- Keep model, tokenizer, and generation config pinned together.
- Store auth outside code and notebooks.
- Validate output quality after any dtype or attention backend change.

## 8. Top Failure Modes

1. OOM on load: lower precision, revise `device_map`, or quantize.
2. OOM during generation: reduce context or output length; inspect KV cache growth.
3. Wrong chat output: template mismatch.
4. Import/runtime errors with remote code: unreviewed custom model code.
5. Attention backend failure: unsupported kernel stack.

## 9. References (Fetched 2026-04-06)

1. https://huggingface.co/docs/transformers/main/en/model_doc/auto - Auto classes for tokenizer and model loading.
2. https://huggingface.co/docs/transformers/main/en/main_classes/model - Core model loading parameters and runtime flags.
3. https://huggingface.co/docs/transformers/main/en/chat_templating - Chat template usage and best practices.
4. https://huggingface.co/docs/transformers/main/en/main_classes/text_generation - Generation configuration and API behavior.
5. https://huggingface.co/docs/transformers/main/en/big_models - Large-model loading and sharded checkpoint guidance.
6. https://huggingface.co/docs/transformers/main/en/installation#offline-mode - Offline loading behavior.
7. https://huggingface.co/docs/transformers/main/en/kv_cache - KV cache behavior and options.
8. https://huggingface.co/docs/transformers/main/en/model_doc/mixtral - Mixtral model specifics.
9. https://huggingface.co/docs/transformers/main/en/model_doc/switch_transformers - Switch Transformers MoE specifics.
10. https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling - Big model inference and offloading context.
11. https://huggingface.co/docs/safetensors/index - Safetensors format and security rationale.
12. https://docs.vllm.ai/en/latest/ - Runtime serving context for loaded Transformer models.

## 10. Uncertainty Notes

- Some MoE expert-kernel flags and defaults are model- and version-specific.
- Verify feature availability against your exact installed `transformers` version.
