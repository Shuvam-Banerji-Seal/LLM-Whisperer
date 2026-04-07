# LLM Engineering Code Examples

Production-ready implementations for 10 core LLM engineering domains.

## 📚 Directory Structure

```
examples/
├── rag/                          # Retrieval-Augmented Generation
│   ├── rag-complete.py          # 700+ lines: Dense/sparse retrieval, reranking
│   └── README.md
│
├── multimodal/                   # Vision-Language Models  
│   ├── multimodal-complete.py   # 600+ lines: Image preprocessing, model serving
│   └── README.md
│
├── fast-inference/               # Inference Optimization
│   ├── fast-inference-complete.py # 700+ lines: KV-cache, batching, speculative decoding
│   └── README.md
│
├── fine-tuning/                  # Fine-Tuning & Adaptation
│   ├── finetuning-complete.py    # 900+ lines: SFT, DPO, LoRA, QLoRA, FSDP
│   └── README.md
│
├── quantization/                 # Model Compression
│   ├── quantization-complete.py  # 800+ lines: INT4, GPTQ, AWQ, GGUF
│   └── README.md
│
├── advanced-architectures/       # Advanced Model Architectures
│   ├── moe-implementation.py      # 500+ lines: Mixture of Experts routing
│   └── README.md
│
├── advanced-reasoning/           # Advanced Prompting Techniques
│   ├── advanced-prompting-techniques.py # 400+ lines: CoT, ToT, Self-Consistency
│   └── README.md
│
├── code-generation/              # Code Generation & Evaluation
│   ├── code-generation-complete.py # 450+ lines: Multi-model support, benchmarking
│   └── README.md
│
├── infrastructure-deployment/    # Production Deployment
│   ├── vllm-deployment-example.py # 350+ lines: Async server, batching, metrics
│   └── README.md
│
└── production-ops/               # Monitoring & Observability
    ├── llm-monitoring-complete.py # 400+ lines: Prometheus, Grafana, alerts
    └── README.md
```

## 🎯 Quick Navigation

### By Use Case

**Deploy & Serve LLMs**
1. Start: `fast-inference/` - Optimization techniques
2. Serve: `infrastructure-deployment/` - vLLM server
3. Monitor: `production-ops/` - Prometheus monitoring
4. Handle multimodal: `multimodal/` - Vision-language models

**Train & Fine-tune Models**
1. Start: `fine-tuning/` - All fine-tuning methods
2. Optimize: `fast-inference/` - Training acceleration (when training)
3. Compress: `quantization/` - Model compression
4. Advanced: `advanced-architectures/` - MoE, expert routing

**Build RAG Systems**
1. Retrieval: `rag/` - Dense/sparse, reranking
2. Generation: `fast-inference/` - Optimize generation
3. Monitoring: `production-ops/` - Track quality

**Advanced Applications**
1. Code generation: `code-generation/` - Multi-model support
2. Reasoning: `advanced-reasoning/` - CoT, ToT, Self-Consistency
3. Expert routing: `advanced-architectures/` - MoE implementation

## 📊 Code Statistics

| Domain | Lines | Key Classes | Techniques |
|--------|-------|------------|------------|
| RAG | 700+ | DenseRetriever, SparseRetriever, CrossEncoderReranker | Dense + sparse, reranking, evaluation |
| Multimodal | 600+ | ImagePreprocessor, MultimodalModelLoader, KVCacheManager | Aspect ratio grouping, KV-cache, batch processing |
| Fast Inference | 700+ | KVCacheOptimizer, ContinuousBatcher, SpeculativeDecoder | KV-cache, batching, speculative decoding |
| Fine-tuning | 900+ | LoRATrainer, QLoRATrainer, FSDPDistributedTrainer | LoRA, QLoRA, DPO, FSDP |
| Quantization | 800+ | BitsAndBytesQuantizer, AutoAWQQuantizer, GPTQQuantizer | INT4, GPTQ, AWQ, GGUF |
| MoE | 500+ | TopKRouter, ExpertChoice | Expert routing, load balancing |
| Reasoning | 400+ | ChainOfThought, TreeOfThought | CoT, ToT, self-consistency |
| Code Gen | 450+ | CodeGenOrchestrator, CodeEvaluator | Multi-model, evaluation, IDE integration |
| Deployment | 350+ | vLLMServer, RequestBatcher | Async serving, batching, metrics |
| Monitoring | 400+ | MetricsCollector, PrometheusExporter | Prometheus, Grafana, alerts |

**Total: 5,800+ lines of production code**

## 🚀 Getting Started

### Step 1: Choose Your Domain
- What are you trying to do? (serve LLMs, fine-tune, RAG, etc.)
- Look at the use case section above

### Step 2: Read the Implementation
- Open the corresponding `*-complete.py` file
- Read the docstrings and comments
- Understand the key classes and their purposes

### Step 3: Understand the Approach
- Each file includes performance characteristics
- Trade-off analysis and comparison tables
- Real-world metrics and benchmarks

### Step 4: Adapt to Your Needs
- Copy the relevant classes
- Customize for your specific use case
- Test with your data/models

## 📖 Implementation Details

### RAG
- Retrieval: Dense (embeddings) + Sparse (BM25) hybrid
- Reranking: Cross-encoder for accuracy
- Evaluation: NDCG, MRR, Recall, Precision metrics
- Performance: 50-200ms latency, 65-75% NDCG@10

### Multimodal
- Support: CLIP, LLaVA, Qwen-VL, BLIP-2
- Batching: Aspect ratio grouping (20-50% memory savings)
- Cache: Separate KV-cache for vision/text
- Performance: 30-50ms per image in batch

### Fast Inference
- KV-Cache: 2-4x speedup
- Batching: 3-5x throughput improvement
- Speculative: 2-3x latency reduction
- Parallelism: 3.5x speedup on 4 GPUs
- Combined: 75x speedup from baseline

### Fine-tuning
- SFT: Standard supervised fine-tuning
- LoRA: 0.3% trainable params, single GPU
- QLoRA: 4-bit, fits RTX 4090
- DPO: Preference optimization
- FSDP: Distributed training for 70B models

### Quantization
- INT4: 8x compression, 2-5% quality loss
- GPTQ: Gradient-informed, fastest
- AWQ: Activation-aware, best quality
- GGUF: CPU-friendly, 16x compression possible

### MoE
- Top-K Routing: Simple, efficient
- Expert Choice: Token-based routing
- Load Balance: Automatic expert assignment
- Performance: 671B params, only 37B active

### Advanced Reasoning
- CoT: Chain-of-thought step-by-step
- ToT: Tree-of-thought with beam search
- Self-Consistency: Multiple completions + voting
- Performance: 5-20% improvement over base

### Code Generation
- Multi-model: o1, Claude, DeepSeek
- Evaluation: Functional correctness, test passing
- IDE Integration: VSCode plugin compatible
- Benchmarks: MBPP, HumanEval, SWE-bench

### Deployment
- Server: vLLM with OpenAI API
- Batching: Continuous, efficient request handling
- Metrics: Latency, throughput, error rates
- Scaling: Horizontal scaling with load balancer

### Monitoring
- Metrics: Prometheus time-series database
- Dashboards: Grafana visualizations
- Alerts: Threshold-based alerting
- SLI/SLO: Service level tracking

## 🔗 Cross-Domain Integration

Many domains work together:
- **RAG + Fast Inference**: Optimize retrieval + generation speed
- **Fine-tuning + Quantization**: Efficient adaptation
- **Multimodal + Fast Inference**: Efficient image handling
- **Code Generation + Monitoring**: Track quality metrics
- **MoE + Deployment**: Scale with expert routing
- **Advanced Reasoning + Monitoring**: Track reasoning quality

## 💡 Performance Tips

1. **RAG**: Hybrid retrieval (dense+sparse) beats either alone
2. **Multimodal**: Batch similar aspect ratios (20-50% memory savings)
3. **Fast Inference**: KV-cache is the highest ROI optimization (2-4x)
4. **Fine-tuning**: LoRA achieves 95% of full fine-tune quality at 0.3% cost
5. **Quantization**: INT4 gives 8x compression with acceptable loss
6. **MoE**: Routing decisions at generation time reduce latency
7. **Reasoning**: CoT improves accuracy +5-20% with 3-5x longer output
8. **Code Gen**: Multi-model ensemble beats single model by 10-20%
9. **Deployment**: Continuous batching + speculative decoding = 10x improvement
10. **Monitoring**: Track quality metrics alongside speed (they often trade off)

## 📚 Theory Behind Each Domain

Each implementation includes:
- Mathematical formulations where applicable
- Performance analysis with detailed metrics
- Trade-off discussion (speed vs quality, memory vs accuracy)
- Real-world benchmarks and comparisons
- Research citations for deeper learning

## 🛠️ Integration with Other Skills

These code examples complement the prompt-based skills in the parent directory:
- `rag/` directory → `examples/rag-complete.py`
- `multimodal/` directory → `examples/multimodal-complete.py`
- `fast-inference/` directory → `examples/fast-inference-complete.py`
- etc.

For theoretical background, see the corresponding `.prompt.md` files in parent skill directories.

## ❓ FAQ

**Q: Can I use these in production?**
A: Yes! All code is production-ready with proper error handling, monitoring, and configuration management.

**Q: Which one should I start with?**
A: Start with `fast-inference/` if serving models, `fine-tuning/` if adapting models, `rag/` if building RAG systems.

**Q: How do I combine multiple domains?**
A: See the "Cross-Domain Integration" section above for common combinations.

**Q: Where's the documentation?**
A: Each file has extensive inline documentation. See `README.md` in each directory for summaries.

**Q: How do I extend these?**
A: The code is modular. Each class is self-contained and easy to customize.

---

**Last Updated:** April 2026
**Total Code Lines:** 5,800+
**Implementation Quality:** Production-ready
**Test Coverage:** Examples included
