# Skills Index

This directory contains reusable, source-backed skill prompts for common LLM engineering workflows.

## Available Skills

- `skills/general-python-development.prompt.md` - General Python engineering baseline.
- `skills/fast-inference/vllm-and-serving.prompt.md` - Fast serving with vLLM and alternatives (TGI, TensorRT-LLM, llama.cpp).
- `skills/huggingface/token-management.prompt.md` - Hugging Face token, auth, gated model, and offline workflow playbook.
- `skills/transformers/loading-and-moe.prompt.md` - Safe Transformer loading patterns and MoE-specific guidance.
- `skills/moe/mixture-of-experts-loading.prompt.md` - Dedicated MoE loading, memory, and runtime guidance.
- `skills/diffusion/loading-and-optimization.prompt.md` - Diffusers pipeline loading, optimization, and scheduler strategy.
- `skills/image-generation/diffusers-image-generation.prompt.md` - Image generation with adapters (LoRA, IP-Adapter, ControlNet).
- `skills/video-generation/diffusers-video-generation.prompt.md` - Text-to-video and image-to-video workflows with diffusers.
- `skills/fine-tuning/llm-finetuning.prompt.md` - Full FT, SFT, PEFT/LoRA/QLoRA training playbook.
- `skills/fine-tuning/eval-and-ops.prompt.md` - Evaluation, checkpointing, distributed training, and safety operations.
- `skills/quantization/llm-quantization.prompt.md` - Quantization method selection and deployment guidance.
- `skills/turboquant/turboquant-finetuning.prompt.md` - TurboQuant-style method landscape and guarded adoption protocol.

## Usage Pattern

1. Choose the closest skill file for the task.
2. Follow the sectioned checklist and command templates.
3. Keep package/runtime versions pinned in your project.
4. Validate behavior with task-specific benchmarks before production rollout.
