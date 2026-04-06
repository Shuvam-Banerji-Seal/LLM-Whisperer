# LLM-Whisperer Skills Library

A comprehensive, source-backed library of **73 reusable skill prompts** for advanced LLM engineering, covering inference optimization, training, agentic systems, and specialized ML techniques.

**📊 Statistics:**
- **73 skills** across 27 categories
- **63,459+ lines** of production-ready code
- **600+ code examples** with detailed annotations
- **500+ research sources** with citations
- **400+ mathematical formulations**

---

## Quick Navigation

- [🚀 New Skills (Added Recently)](#new-skills)
- [📚 Skills by Category](#skills-by-category)
- [🔍 Skills by Use Case](#skills-by-use-case)
- [💡 Getting Started](#getting-started)

---

## New Skills

Recently added high-impact skills (marked with 🆕):

### Foundational Engineering (7 skills) 🆕
- `foundational/advanced-python-patterns.prompt.md` - Advanced Python design patterns and best practices
- `foundational/advanced-optimization-algorithms.prompt.md` - Optimization techniques (AdamW, LION, Sophia, SAM)
- `foundational/error-handling-and-logging.prompt.md` - Comprehensive error handling and logging strategies
- `foundational/testing-and-validation.prompt.md` - Testing frameworks, unit tests, integration tests
- `foundational/performance-profiling.prompt.md` - Memory profiling, CPU profiling, bottleneck analysis
- `foundational/dependency-management.prompt.md` - Package management, versioning, environments
- `foundational/api-design-and-documentation.prompt.md` - REST API design, OpenAPI, documentation

### Agentic Systems (7 skills) 🆕
- `agentic/agent-choreography-and-orchestration.prompt.md` - Saga patterns, state machines, compensation
- `agentic/distributed-consensus-for-agents.prompt.md` - Raft, Byzantine consensus, quorum voting
- `agentic/agent-memory-systems.prompt.md` - Distributed caching, consistency models, synchronization
- `agentic/monitoring-and-observability.prompt.md` - Distributed tracing, metrics, SLI/SLO tracking
- `agentic/failure-detection-and-recovery.prompt.md` - Health checks, circuit breakers, self-healing
- `agentic/agent-communication-protocols.prompt.md` - RPC, publish-subscribe, message queuing
- `agentic/load-balancing-and-routing.prompt.md` - Load balancing algorithms, traffic shifting, routing

### Fast Inference & Acceleration (8 skills) 🆕
- `fast-inference/speculative-decoding.prompt.md` - Speculative decoding (2-3x latency reduction)
- `fast-inference/kv-cache-optimization.prompt.md` - KV-cache management (2-4x throughput)
- `fast-inference/batch-serving-strategies.prompt.md` - Continuous batching, dynamic batching
- `fast-inference/tensor-parallelism.prompt.md` - Tensor parallelism patterns
- `fast-inference/pipeline-parallelism.prompt.md` - Pipeline parallelism optimization
- `fast-inference/model-distillation.prompt.md` - Knowledge distillation techniques
- `fast-inference/dynamic-shape-inference.prompt.md` - Dynamic shape optimization
- `fast-inference/vllm-and-serving.prompt.md` - vLLM, TGI, TensorRT-LLM serving

### Training Optimization (4 skills) 🆕
- `training-optimization/learning-rate-scheduling.prompt.md` - Cosine, warmup, LLRD schedules
- `training-optimization/gradient-accumulation-checkpointing.prompt.md` - Memory-efficient training (70-80% reduction)
- `training-optimization/mixed-precision-training.prompt.md` - FP16/BF16/FP8 precision training
- `training-optimization/distributed-training-optimization.prompt.md` - ZeRO, DDP, FSDP strategies

### Data Quality & Engineering (4 skills) 🆕
- `data-quality/data-quality-assessment.prompt.md` - Data validation, profiling, quality metrics
- `data-quality/outlier-detection-handling.prompt.md` - Statistical, isolation forest, deep learning methods
- `data-quality/class-imbalance-handling.prompt.md` - Oversampling, undersampling, cost-sensitive learning
- `data-quality/missing-data-imputation.prompt.md` - KNN, regression, multiple imputation techniques

---

## Skills by Category

### Core Frameworks & Platforms

**HuggingFace Ecosystem (1 skill)**
- `huggingface/token-management.prompt.md` - Token auth, gated models, offline workflows

**Transformers (1 skill)**
- `transformers/loading-and-moe.prompt.md` - Safe loading patterns, MoE guidance

**Diffusion Models (1 skill)**
- `diffusion/loading-and-optimization.prompt.md` - Pipeline optimization, scheduler strategy

---

### Inference & Serving

**Fast Inference (8 skills)**
- `fast-inference/vllm-and-serving.prompt.md` - vLLM, TGI, TensorRT-LLM, llama.cpp
- `fast-inference/speculative-decoding.prompt.md` - Speculative decoding patterns
- `fast-inference/kv-cache-optimization.prompt.md` - Cache optimization strategies
- `fast-inference/batch-serving-strategies.prompt.md` - Batching techniques
- `fast-inference/tensor-parallelism.prompt.md` - Tensor parallelism
- `fast-inference/pipeline-parallelism.prompt.md` - Pipeline parallelism
- `fast-inference/model-distillation.prompt.md` - Distillation techniques
- `fast-inference/dynamic-shape-inference.prompt.md` - Dynamic shape handling

**Long Context (3 skills)**
- `long-context/position-encodings.prompt.md` - RoPE, ALiBi, YaRN encodings
- `long-context/efficient-attention.prompt.md` - FlashAttention-2, GQA, MQA
- `long-context/long-context-serving.prompt.md` - Serving 100K+ token models

---

### Training & Fine-tuning

**Fine-tuning (4 skills)**
- `fine-tuning/llm-finetuning.prompt.md` - Full FT, SFT, LoRA/QLoRA
- `fine-tuning/eval-and-ops.prompt.md` - Evaluation, checkpointing, ops
- `fine-tuning/adapter-and-bottleneck-methods.prompt.md` - Adapter methods
- `fine-tuning/lora-advanced-techniques.prompt.md` - Advanced LoRA (QLoRA, DoRA, LoftQ)

**Training Optimization (4 skills)**
- `training-optimization/learning-rate-scheduling.prompt.md` - LR scheduling strategies
- `training-optimization/gradient-accumulation-checkpointing.prompt.md` - Memory optimization
- `training-optimization/mixed-precision-training.prompt.md` - FP16/BF16/FP8 training
- `training-optimization/distributed-training-optimization.prompt.md` - Distributed strategies

**Quantization (2 skills)**
- `quantization/llm-quantization.prompt.md` - Quantization methods
- `turboquant/turboquant-finetuning.prompt.md` - TurboQuant methods

---

### RAG & Retrieval

**RAG (2 skills)**
- `rag/retrieval-strategies.prompt.md` - Dense/sparse retrieval, BM25, query expansion
- `rag/reranking-evaluation.prompt.md` - Reranking, LLM-as-judge, NDCG/MRR metrics

---

### Agents & Orchestration

**Agent Frameworks (2 skills)**
- `agents/agent-frameworks.prompt.md` - ReAct patterns, tool definitions
- `agents/tool-use-patterns.prompt.md` - Memory, multi-agent, safety

**Agentic Systems (7 skills)**
- `agentic/agent-choreography-and-orchestration.prompt.md` - Orchestration patterns
- `agentic/distributed-consensus-for-agents.prompt.md` - Consensus mechanisms
- `agentic/agent-memory-systems.prompt.md` - Distributed memory
- `agentic/monitoring-and-observability.prompt.md` - Monitoring & tracing
- `agentic/failure-detection-and-recovery.prompt.md` - Failure handling
- `agentic/agent-communication-protocols.prompt.md` - Communication patterns
- `agentic/load-balancing-and-routing.prompt.md` - Load balancing

---

### Multimodal & Generation

**Multimodal (3 skills)**
- `multimodal/vision-language-loading.prompt.md` - CLIP, LLaVA, GPT-4V loading
- `multimodal/image-preprocessing.prompt.md` - Image normalization, resizing
- `multimodal/multimodal-serving.prompt.md` - Multimodal batching, serving

**Image Generation (1 skill)**
- `image-generation/diffusers-image-generation.prompt.md` - Stable Diffusion, adapters

**Video Generation (1 skill)**
- `video-generation/diffusers-video-generation.prompt.md` - Text-to-video, image-to-video

---

### Model Composition

**Model Merging (2 skills)**
- `model-merging/weight-merging.prompt.md` - Linear interpolation, SLERP
- `model-merging/adapter-composition.prompt.md` - LoRA composition, MoA

**Mixture of Experts (1 skill)**
- `moe/mixture-of-experts-loading.prompt.md` - MoE loading and guidance

---

### Evaluation & Safety

**Evaluation (2 skills)**
- `evaluation/llm-benchmarks.prompt.md` - MMLU, HELM, BIG-Bench, HellaSwag
- `evaluation/evaluation-metrics.prompt.md` - Semantic similarity, safety metrics

**Safety & Alignment (3 skills)**
- `safety-alignment/rlhf-and-dpo.prompt.md` - RLHF, DPO, reward modeling
- `safety-alignment/red-teaming.prompt.md` - Red-teaming, jailbreak testing
- `safety-alignment/alignment-eval.prompt.md` - AlpacaEval, MT-Bench, toxicity

**Prompt Engineering (2 skills)**
- `prompt-engineering/prompt-templates.prompt.md` - Role prompts, few-shot, CoT
- `prompt-engineering/prompt-optimization.prompt.md` - Optimization, caching

---

### Data Quality & Engineering

**Data Quality (4 skills)**
- `data-quality/data-quality-assessment.prompt.md` - Validation, profiling
- `data-quality/outlier-detection-handling.prompt.md` - Outlier detection methods
- `data-quality/class-imbalance-handling.prompt.md` - Balancing techniques
- `data-quality/missing-data-imputation.prompt.md` - Imputation methods

---

### Foundational Engineering

**Foundational (7 skills)**
- `foundational/advanced-python-patterns.prompt.md` - Design patterns
- `foundational/advanced-optimization-algorithms.prompt.md` - Optimization algorithms
- `foundational/error-handling-and-logging.prompt.md` - Error handling, logging
- `foundational/testing-and-validation.prompt.md` - Testing frameworks
- `foundational/performance-profiling.prompt.md` - Profiling techniques
- `foundational/dependency-management.prompt.md` - Package management
- `foundational/api-design-and-documentation.prompt.md` - API design

---

### General

**General Purpose (1 skill)**
- `general-python-development.prompt.md` - General Python engineering baseline

---

## Skills by Use Case

### 🚀 Deploy & Serve LLMs

1. Start with: `fast-inference/vllm-and-serving.prompt.md`
2. Optimize inference: `fast-inference/kv-cache-optimization.prompt.md`
3. Handle long context: `long-context/position-encodings.prompt.md`
4. For multimodal: `multimodal/vision-language-loading.prompt.md`
5. Monitor systems: `agentic/monitoring-and-observability.prompt.md`

### 📚 Train & Fine-tune Models

1. Start with: `foundational/advanced-optimization-algorithms.prompt.md`
2. Optimize memory: `training-optimization/gradient-accumulation-checkpointing.prompt.md`
3. Precision training: `training-optimization/mixed-precision-training.prompt.md`
4. Parameter-efficient: `fine-tuning/lora-advanced-techniques.prompt.md`
5. Distribute across GPUs: `training-optimization/distributed-training-optimization.prompt.md`

### 🤖 Build Agentic Systems

1. Start with: `agents/agent-frameworks.prompt.md`
2. Orchestrate: `agentic/agent-choreography-and-orchestration.prompt.md`
3. Ensure reliability: `agentic/failure-detection-and-recovery.prompt.md`
4. Monitor: `agentic/monitoring-and-observability.prompt.md`
5. Scale: `agentic/load-balancing-and-routing.prompt.md`

### 📊 Improve Data Quality

1. Assess: `data-quality/data-quality-assessment.prompt.md`
2. Clean: `data-quality/outlier-detection-handling.prompt.md`
3. Balance: `data-quality/class-imbalance-handling.prompt.md`
4. Impute: `data-quality/missing-data-imputation.prompt.md`

### 🔍 Build RAG Systems

1. Retrieval: `rag/retrieval-strategies.prompt.md`
2. Reranking: `rag/reranking-evaluation.prompt.md`
3. Prompt: `prompt-engineering/prompt-templates.prompt.md`
4. Evaluate: `evaluation/evaluation-metrics.prompt.md`

### 🎨 Generate & Process Multimodal Content

1. Vision-language: `multimodal/vision-language-loading.prompt.md`
2. Image preprocessing: `multimodal/image-preprocessing.prompt.md`
3. Image generation: `image-generation/diffusers-image-generation.prompt.md`
4. Video generation: `video-generation/diffusers-video-generation.prompt.md`

### 🛡️ Safety & Alignment

1. RLHF/DPO: `safety-alignment/rlhf-and-dpo.prompt.md`
2. Red-teaming: `safety-alignment/red-teaming.prompt.md`
3. Evaluation: `safety-alignment/alignment-eval.prompt.md`

### 📈 Evaluate Models

1. Benchmarks: `evaluation/llm-benchmarks.prompt.md`
2. Metrics: `evaluation/evaluation-metrics.prompt.md`
3. Alignment: `safety-alignment/alignment-eval.prompt.md`

---

## Getting Started

### Quick Start (5 minutes)

1. **Identify your use case** from [Skills by Use Case](#skills-by-use-case)
2. **Read the first skill** - Start with the recommended skill file
3. **Follow the examples** - Each skill has 5+ code examples
4. **Adapt to your needs** - Copy examples and customize for your project

### Documentation Structure

Each skill file includes:

- **Problem Statement** - When and why to use this skill
- **Theory & Fundamentals** - Mathematical formulations and concepts
- **Implementation Patterns** - Production-ready code examples
- **Framework Integration** - How to use with PyTorch, HuggingFace, etc.
- **Performance Considerations** - Memory, latency, throughput
- **Common Pitfalls** - What to watch out for
- **Research References** - 5-10 authoritative sources with URLs
- **Advanced Techniques** - Beyond the basics

### Recommended Learning Paths

**Beginner (1-2 weeks)**
- foundational/advanced-python-patterns.prompt.md
- foundational/error-handling-and-logging.prompt.md
- foundational/testing-and-validation.prompt.md
- fast-inference/vllm-and-serving.prompt.md

**Intermediate (2-4 weeks)**
- training-optimization/learning-rate-scheduling.prompt.md
- training-optimization/mixed-precision-training.prompt.md
- fine-tuning/lora-advanced-techniques.prompt.md
- agentic/agent-choreography-and-orchestration.prompt.md

**Advanced (4+ weeks)**
- training-optimization/distributed-training-optimization.prompt.md
- agentic/distributed-consensus-for-agents.prompt.md
- fast-inference/speculative-decoding.prompt.md
- evaluation/llm-benchmarks.prompt.md

---

## Usage Guidelines

1. **Choose the closest match** - Find the skill most relevant to your task
2. **Follow the checklist** - Each skill has step-by-step instructions
3. **Pin versions** - Always pin package versions in your project
4. **Validate locally** - Test with task-specific benchmarks before production
5. **Reference sources** - Use the research citations to deepen understanding

---

## Statistics

| Metric | Value |
|--------|-------|
| Total Skills | 73 |
| Total Categories | 27 |
| Total Lines | 63,459+ |
| Total Size | 2.3 MB |
| Code Examples | 600+ |
| Research Sources | 500+ |
| Math Formulations | 400+ |
| Frameworks Covered | 15+ |

---

## Contributing

To add a new skill:

1. Create a new `.prompt.md` file in the appropriate category
2. Follow the structure of existing skills
3. Include 5+ research sources
4. Add 5+ code examples
5. Include mathematical formulations where applicable
6. Update this README

---

## License & Attribution

Each skill includes author attribution: **Shuvam Banerji Seal**

Research-backed content with citations to original papers and implementations.

---

## Need Help?

- **Specific task?** Use [Skills by Use Case](#skills-by-use-case)
- **New to a topic?** Start with Beginner learning path
- **Looking for specific technique?** Use Ctrl+F to search
- **Want to learn more?** Check the research references in each skill

---

**Last Updated:** April 2026
**Total Skills:** 73 | **Lines of Code:** 63,459+ | **Research Sources:** 500+
