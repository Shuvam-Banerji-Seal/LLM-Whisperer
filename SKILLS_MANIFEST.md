# Skills Manifest

Complete inventory of all 73 LLM-Whisperer skills, organized by category with descriptions, line counts, and key focus areas.

**Generated:** April 6, 2026  
**Total Skills:** 73 | **Total Lines:** 63,459+ | **Total Size:** 2.3 MB

---

## Category Summary

| Category | Count | Total KB | Focus Area |
|----------|-------|----------|-----------|
| **Fast Inference** | 8 | 148 KB | Inference optimization, serving, parallelism |
| **Training Optimization** | 4 | 224 KB | Learning rates, mixed precision, distributed training |
| **Fine-tuning** | 4 | 244 KB | LoRA, adapters, prefix tuning |
| **Foundational** | 7 | 244 KB | Python patterns, testing, API design |
| **Agentic** | 7 | 156 KB | Agent orchestration, consensus, resilience |
| **Data Quality** | 4 | 112 KB | Data validation, outlier detection, imputation |
| **Long Context** | 3 | 32 KB | Position encodings, attention, serving |
| **Multimodal** | 3 | 40 KB | Vision-language, image generation, serving |
| **Safety Alignment** | 3 | 28 KB | RLHF, DPO, red-teaming, eval |
| **Evaluation** | 2 | 40 KB | Benchmarks, metrics, evaluation |
| **Model Merging** | 2 | 16 KB | Weight merging, adapter composition |
| **Quantization** | 2 | 112 KB | Model compression, quantization methods |
| **Agents** | 2 | 24 KB | Agent frameworks, tool use |
| **RAG** | 2 | 16 KB | Retrieval strategies, reranking |
| **Prompt Engineering** | 2 | 48 KB | Templates, optimization |
| **Image Generation** | 1 | 8.0 KB | Diffusion models, adapters |
| **Video Generation** | 1 | 108 KB | Text-to-video, image-to-video |
| **Diffusion** | 1 | 8.0 KB | Pipeline loading, optimization |
| **HuggingFace** | 1 | 112 KB | Token management, auth |
| **Transformers** | 1 | 8.0 KB | Model loading |
| **MoE** | 1 | 112 KB | Mixture of experts |
| **TurboQuant** | 1 | 8.0 KB | Quantization methods |
| **General** | 1 | 8.0 KB | Python development baseline |
| **Empty (for expansion)** | 11 | - | Reserved for future skills |
| **TOTAL** | **73** | **2.3 MB** | Complete LLM engineering toolkit |

---

## Skills Detailed Inventory

### Fast Inference (8 skills)

1. **vllm-and-serving.prompt.md** (14 KB, 410 lines)
   - Fast serving with vLLM, TGI, TensorRT-LLM, llama.cpp
   - Focus: Production serving, batching strategies
   - Key techniques: Continuous batching, page attention

2. **speculative-decoding.prompt.md** (24 KB, 791 lines)
   - Speculative decoding for 2-3x latency reduction
   - Focus: Draft models, acceptance sampling
   - Key techniques: Batch verification, dynamic draft length
   - Performance: 2-3x latency improvement

3. **kv-cache-optimization.prompt.md** (25 KB, 854 lines)
   - KV-cache management and optimization
   - Focus: Memory efficiency, throughput
   - Key techniques: Page attention, reuse patterns
   - Performance: 2-4x throughput improvement

4. **batch-serving-strategies.prompt.md** (25 KB, 850 lines)
   - Batching strategies for serving
   - Focus: Throughput maximization
   - Key techniques: Dynamic batching, continuous batching
   - Performance: 3-5x throughput

5. **tensor-parallelism.prompt.md** (10 KB, 367 lines)
   - Tensor parallelism patterns
   - Focus: Model parallelism across GPUs
   - Key techniques: Row/column sharding
   - Performance: 7-8x speedup

6. **pipeline-parallelism.prompt.md** (7.4 KB, 310 lines)
   - Pipeline parallelism for sequential stages
   - Focus: GPU utilization efficiency
   - Key techniques: Microbatching, gradient checkpointing
   - Performance: 90%+ efficiency

7. **model-distillation.prompt.md** (13 KB, 463 lines)
   - Knowledge distillation techniques
   - Focus: Model compression
   - Key techniques: Temperature scaling, attention transfer
   - Performance: 5-10x compression

8. **dynamic-shape-inference.prompt.md** (14 KB, 540 lines)
   - Dynamic shape optimization
   - Focus: Memory efficiency
   - Key techniques: Shape inference, memory reuse
   - Performance: 20-50% memory savings

---

### Training Optimization (4 skills)

1. **learning-rate-scheduling.prompt.md** (46 KB, ~1500 lines)
   - Comprehensive learning rate scheduling
   - Focus: Training stability and convergence
   - Key techniques: Cosine, warmup, LLRD, cyclic schedules
   - Algorithms: 8+ scheduling strategies

2. **gradient-accumulation-checkpointing.prompt.md** (45 KB, ~1400 lines)
   - Memory-efficient training techniques
   - Focus: Training large models with limited memory
   - Key techniques: Gradient accumulation, activation checkpointing
   - Performance: 70-80% memory reduction

3. **mixed-precision-training.prompt.md** (59 KB, ~1800 lines)
   - FP16/BF16/FP8 precision training
   - Focus: Speed and memory optimization
   - Key techniques: Loss scaling, dtype casting
   - Performance: 2x memory, 1.5x speed

4. **distributed-training-optimization.prompt.md** (74 KB, ~2300 lines)
   - Distributed training strategies
   - Focus: Multi-GPU and multi-node training
   - Key techniques: ZeRO, DDP, FSDP
   - Performance: 4-8x memory reduction, 85-95% scaling

---

### Fine-tuning (4 skills)

1. **llm-finetuning.prompt.md** (14 KB, ~400 lines)
   - Full fine-tuning and SFT playbooks
   - Focus: Task-specific adaptation
   - Key techniques: Full FT, SFT, supervised fine-tuning
   - Frameworks: HuggingFace, Huggingface transformers

2. **eval-and-ops.prompt.md** (14 KB, ~400 lines)
   - Evaluation, checkpointing, and operations
   - Focus: Training monitoring and management
   - Key techniques: Evaluation loops, checkpointing
   - Tools: Wandb, TensorBoard

3. **adapter-and-bottleneck-methods.prompt.md** (24 KB, ~750 lines)
   - Adapter and bottleneck architectures
   - Focus: Parameter-efficient fine-tuning
   - Key techniques: MAD-X, IA³, Compacter
   - Memory: 75%+ reduction vs full FT

4. **lora-advanced-techniques.prompt.md** (28 KB, ~850 lines)
   - Advanced LoRA techniques
   - Focus: Parameter-efficient fine-tuning
   - Key techniques: QLoRA, DoRA, LoftQ, rank optimization
   - Memory: 4x savings with QLoRA

---

### Foundational Engineering (7 skills)

1. **advanced-python-patterns.prompt.md** (21 KB, ~650 lines)
   - Advanced Python design patterns
   - Focus: Code quality and maintainability
   - Key techniques: Decorators, context managers, metaclasses
   - Examples: 10+ patterns with implementations

2. **advanced-optimization-algorithms.prompt.md** (64 KB, ~2000 lines)
   - Optimization algorithms for training
   - Focus: Convergence and stability
   - Key algorithms: AdamW, LION, Sophia, SAM, L-BFGS
   - Frameworks: PyTorch, TensorFlow

3. **error-handling-and-logging.prompt.md** (23 KB, ~700 lines)
   - Comprehensive error handling
   - Focus: Robustness and debugging
   - Key techniques: Exception handling, logging strategies
   - Tools: logging, structlog

4. **testing-and-validation.prompt.md** (26 KB, ~800 lines)
   - Testing frameworks and strategies
   - Focus: Code quality assurance
   - Key techniques: Unit tests, integration tests, fixtures
   - Tools: pytest, unittest, hypothesis

5. **performance-profiling.prompt.md** (19 KB, ~600 lines)
   - Performance profiling techniques
   - Focus: Bottleneck identification
   - Key techniques: Memory profiling, CPU profiling
   - Tools: cProfile, memory_profiler, py-spy

6. **dependency-management.prompt.md** (24 KB, ~700 lines)
   - Package and dependency management
   - Focus: Environment reproducibility
   - Key techniques: Virtual environments, requirements.txt
   - Tools: pip, conda, poetry

7. **api-design-and-documentation.prompt.md** (22 KB, ~650 lines)
   - REST API design and documentation
   - Focus: API quality and usability
   - Key techniques: OpenAPI, API design patterns
   - Tools: FastAPI, OpenAPI

---

### Agentic Systems (7 skills)

1. **agent-choreography-and-orchestration.prompt.md** (24 KB, ~650 lines)
   - Agent orchestration patterns
   - Focus: Multi-agent coordination
   - Key patterns: Saga patterns, choreography, state machines
   - Techniques: Compensation, rollback

2. **distributed-consensus-for-agents.prompt.md** (25 KB, ~775 lines)
   - Consensus mechanisms for agents
   - Focus: Distributed agreement
   - Key algorithms: Raft, PBFT, quorum voting
   - Applications: Leader election, log replication

3. **agent-memory-systems.prompt.md** (20 KB, ~605 lines)
   - Distributed memory systems
   - Focus: Consistency and coordination
   - Key techniques: Caching, consistency models
   - Systems: Redis, Memcached, custom solutions

4. **monitoring-and-observability.prompt.md** (20 KB, ~600 lines)
   - Monitoring and observability
   - Focus: System health and debugging
   - Key techniques: Distributed tracing, metrics collection
   - Tools: Jaeger, Prometheus, Datadog

5. **failure-detection-and-recovery.prompt.md** (21 KB, ~610 lines)
   - Failure detection and recovery
   - Focus: System resilience
   - Key patterns: Health checks, circuit breakers
   - Strategies: Graceful degradation, self-healing

6. **agent-communication-protocols.prompt.md** (18 KB, ~585 lines)
   - Communication protocols for agents
   - Focus: Reliable messaging
   - Key frameworks: gRPC, Kafka, RabbitMQ
   - Patterns: RPC, pub-sub, message queuing

7. **load-balancing-and-routing.prompt.md** (19 KB, ~596 lines)
   - Load balancing and routing
   - Focus: Distributing load efficiently
   - Key algorithms: Round-robin, least-connections, consistent hashing
   - Tools: HAProxy, Envoy, Nginx

---

### Data Quality (4 skills)

1. **data-quality-assessment.prompt.md** (15 KB, ~450 lines)
   - Data quality validation
   - Focus: Data profiling and assessment
   - Key techniques: Validation rules, quality metrics
   - Tools: Great Expectations, Pandas Profiling

2. **outlier-detection-handling.prompt.md** (16 KB, ~500 lines)
   - Outlier detection methods
   - Focus: Anomaly identification
   - Key algorithms: Statistical, Isolation Forest, Deep Learning (21+ methods)
   - Methods: Z-score, IQR, LOF

3. **class-imbalance-handling.prompt.md** (19 KB, ~600 lines)
   - Class imbalance solutions
   - Focus: Balanced training data
   - Key techniques: Oversampling, undersampling, cost-sensitive learning
   - Methods: SMOTE, stratification

4. **missing-data-imputation.prompt.md** (21 KB, ~650 lines)
   - Missing data imputation
   - Focus: Data completeness
   - Key techniques: KNN, regression, multiple imputation
   - Methods: Mean/median, KNN, iterative imputation

---

### Long Context (3 skills)

1. **position-encodings.prompt.md** (12 KB, ~370 lines)
   - Position encoding methods
   - Focus: Handling long sequences
   - Key encodings: RoPE, ALiBi, Linear scaling, YaRN
   - Techniques: Extrapolation, interpolation

2. **efficient-attention.prompt.md** (12 KB, ~380 lines)
   - Efficient attention mechanisms
   - Focus: Reducing computation for long sequences
   - Key techniques: FlashAttention-2, GQA, MQA
   - Patterns: Sparse attention, local attention

3. **long-context-serving.prompt.md** (8 KB, ~240 lines)
   - Serving long-context models
   - Focus: Efficient inference for 100K+ tokens
   - Key techniques: Chunking, streaming
   - Patterns: Sliding window, cached processing

---

### Multimodal (3 skills)

1. **vision-language-loading.prompt.md** (14 KB, ~420 lines)
   - Vision-language model loading
   - Focus: Multimodal inference
   - Key models: CLIP, LLaVA, GPT-4V, Qwen-VL
   - Patterns: Image-text pairs, batching

2. **image-preprocessing.prompt.md** (14 KB, ~410 lines)
   - Image preprocessing techniques
   - Focus: Image normalization and formatting
   - Key techniques: Resizing, normalization, augmentation
   - Tools: torchvision, PIL, OpenCV

3. **multimodal-serving.prompt.md** (12 KB, ~360 lines)
   - Multimodal model serving
   - Focus: Efficient batching and serving
   - Key techniques: Dynamic batching, tensor fusion
   - Tools: vLLM, TGI

---

### Safety & Alignment (3 skills)

1. **rlhf-and-dpo.prompt.md** (10 KB, ~310 lines)
   - RLHF and DPO training
   - Focus: Preference optimization
   - Key techniques: Reward modeling, Direct Preference Optimization
   - Methods: PPO, DPO

2. **red-teaming.prompt.md** (10 KB, ~300 lines)
   - Red-teaming methodologies
   - Focus: Adversarial testing
   - Key techniques: Jailbreak testing, prompt attacks
   - Methods: Systematic probing

3. **alignment-eval.prompt.md** (8 KB, ~240 lines)
   - Alignment evaluation
   - Focus: Safety benchmarking
   - Key benchmarks: AlpacaEval, MT-Bench
   - Metrics: Toxicity, alignment scores

---

### Evaluation (2 skills)

1. **llm-benchmarks.prompt.md** (21 KB, ~650 lines)
   - LLM benchmarking
   - Focus: Model evaluation
   - Key benchmarks: MMLU, HELM, BIG-Bench, HellaSwag
   - Tools: lm-evaluation-harness

2. **evaluation-metrics.prompt.md** (19 KB, ~600 lines)
   - Evaluation metrics
   - Focus: Quality measurement
   - Key metrics: BLEU, ROUGE, semantic similarity
   - Methods: Automatic metrics, human evaluation

---

### Model Merging (2 skills)

1. **weight-merging.prompt.md** (8 KB, ~240 lines)
   - Weight merging techniques
   - Focus: Model composition
   - Key techniques: Linear interpolation, SLERP
   - Methods: Task-specific merging

2. **adapter-composition.prompt.md** (8 KB, ~240 lines)
   - Adapter composition
   - Focus: Combining LoRA adapters
   - Key techniques: Stacking, parallel, sequential composition
   - Methods: MoA (Mixture of Adapters)

---

### Additional Skills (14 skills)

**Quantization (2 skills)**
- llm-quantization.prompt.md - Quantization methods and strategies
- turboquant-finetuning.prompt.md - TurboQuant methodology

**Agents (2 skills)**
- agent-frameworks.prompt.md - Agent framework patterns
- tool-use-patterns.prompt.md - Tool use and multi-agent patterns

**RAG (2 skills)**
- retrieval-strategies.prompt.md - Dense/sparse retrieval, BM25
- reranking-evaluation.prompt.md - Reranking and evaluation

**Prompt Engineering (2 skills)**
- prompt-templates.prompt.md - Prompt templates and few-shot
- prompt-optimization.prompt.md - Prompt optimization techniques

**Generation (2 skills)**
- image-generation/diffusers-image-generation.prompt.md - Image generation
- video-generation/diffusers-video-generation.prompt.md - Video generation

**Other (2 skills)**
- diffusion/loading-and-optimization.prompt.md - Diffusion pipeline
- transformers/loading-and-moe.prompt.md - Model loading
- moe/mixture-of-experts-loading.prompt.md - MoE loading
- huggingface/token-management.prompt.md - HF token management
- turboquant/turboquant-finetuning.prompt.md - TurboQuant methods
- general-python-development.prompt.md - General Python baseline

---

## Statistics Summary

| Metric | Value |
|--------|-------|
| **Total Skills** | 73 |
| **Total Categories** | 27 |
| **Total Lines** | 63,459+ |
| **Total Size** | 2.3 MB |
| **Average KB per skill** | 31.5 KB |
| **Average lines per skill** | 869 |
| **Code Examples** | 600+ |
| **Research Sources** | 500+ |
| **Math Formulations** | 400+ |
| **Frameworks Covered** | 15+ |
| **Algorithms Documented** | 100+ |
| **Techniques Described** | 300+ |

---

## Usage Tips

1. **Find by category** - Scroll to the category section
2. **Check line count** - Longer skills have more depth
3. **Review focus area** - Identify relevant techniques
4. **Check performance gains** - See what improvements are documented
5. **Read the skill file** - Follow examples and implementations

---

**Last Updated:** April 6, 2026  
**Author:** Shuvam Banerji Seal  
**Repository:** LLM-Whisperer Skills Library
