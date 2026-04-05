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
- `skills/prompt-engineering/prompt-templates.prompt.md` - Role-based prompts, few-shot learning, chain-of-thought, and structured outputs.
- `skills/prompt-engineering/prompt-optimization.prompt.md` - Prompt optimization techniques, self-refinement workflows, and prompt caching strategies.
- `skills/evaluation/llm-benchmarks.prompt.md` - Standard benchmarks (MMLU, HELM, BIG-Bench, HellaSwag) and benchmark execution patterns.
- `skills/evaluation/evaluation-metrics.prompt.md` - Semantic similarity metrics, reasoning metrics, and safety evaluation scores.
- `skills/multimodal/vision-language-loading.prompt.md` - Vision-language model loading (CLIP, LLaVA, GPT-4V, Qwen-VL) and inference patterns.
- `skills/multimodal/image-preprocessing.prompt.md` - Image normalization, resizing, and processor usage with transformers and torchvision.
- `skills/multimodal/multimodal-serving.prompt.md` - Batching multimodal inputs and efficient serving with vLLM.
- `skills/rag/retrieval-strategies.prompt.md` - Dense/sparse retrieval, BM25, multi-stage retrieval, query expansion, and HyDE patterns.
- `skills/rag/reranking-evaluation.prompt.md` - Cross-encoder reranking, LLM-as-judge evaluation, and RAG metrics (NDCG, MRR, Recall@K).
- `skills/agents/agent-frameworks.prompt.md` - ReAct pattern, structured tool definitions, and agent loop implementations.
- `skills/agents/tool-use-patterns.prompt.md` - Memory management, multi-agent orchestration, safety constraints, and error handling.
- `skills/model-merging/weight-merging.prompt.md` - Linear interpolation, SLERP, and task-specific weighted model merging.
- `skills/model-merging/adapter-composition.prompt.md` - LoRA composition, mixture-of-adapters (MoA), and merging adapters into base models.
- `skills/safety-alignment/rlhf-and-dpo.prompt.md` - RLHF fundamentals, reward modeling, Direct Preference Optimization (DPO), and preference dataset creation.
- `skills/safety-alignment/red-teaming.prompt.md` - Red-teaming methodologies, adversarial prompt templates, and jailbreak testing patterns.
- `skills/safety-alignment/alignment-eval.prompt.md` - Alignment benchmarks (AlpacaEval, MT-Bench), safety metrics, and toxicity measurement.
- `skills/long-context/position-encodings.prompt.md` - Position encoding methods (RoPE, ALiBi, Linear scaling, YaRN) and extrapolation strategies.
- `skills/long-context/efficient-attention.prompt.md` - FlashAttention-2, GQA, MQA, kv-cache management, and sparse attention patterns.
- `skills/long-context/long-context-serving.prompt.md` - Serving long-context models (up to 100K tokens) and chunked processing strategies.

## Usage Pattern

1. Choose the closest skill file for the task.
2. Follow the sectioned checklist and command templates.
3. Keep package/runtime versions pinned in your project.
4. Validate behavior with task-specific benchmarks before production rollout.
