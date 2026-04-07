# Advanced LLM Architectures

This skill category covers cutting-edge and specialized LLM architecture patterns, including Mixture of Experts (MoE), efficient attention mechanisms, and hybrid models.

## Skills in This Category

- **Mixture of Experts (MoE)** - Routing, load balancing, and sparse activation patterns
- **Efficient Attention Mechanisms** - Flash Attention, GQA, MQA, ALiBi, RoPE
- **Hybrid Architectures** - Mamba, RetNet, and Attention-Free models
- **Model Scaling Laws** - Understanding and predicting LLM performance
- **Inference Optimization** - Speculative decoding, KV-cache strategies

## Key Concepts

### MoE (Mixture of Experts)
Sparse activation pattern where only a subset of parameters are active for each input.

**Benefits**:
- Efficient scaling: 671B parameter models with only 37B active
- Cost-effective inference: Reduced computation per token
- Specialized knowledge: Different experts for different domains

**Examples**:
- Mixtral 8x7B (47B params, 13B active)
- DeepSeek-V3 (671B params, 37B active)
- Nemotron 3 Super (120B params, 12B active - Hybrid)

### Efficient Attention
Reducing quadratic attention complexity to linear or near-linear.

**Techniques**:
- Flash Attention 3: 5-8x speedup
- Grouped-Query Attention (GQA): 75% memory reduction
- Multi-Query Attention (MQA): Single query head
- Rotary Position Embeddings (RoPE): Better extrapolation
- Alibi (Attention with Linear Biases): Position-agnostic

## Research Resources

See `research/` directory for:
- Comprehensive MoE technical reference
- Implementation guides
- Benchmark comparisons
- Paper summaries and citations

## Getting Started

1. **Understanding MoE**: Start with routing mechanisms and load balancing
2. **Implementing Efficient Attention**: Use vLLM or FlashAttention libraries
3. **Building Hybrid Models**: Combine Mamba with Attention layers
4. **Optimizing Inference**: Apply speculative decoding and KV-cache strategies

## Further Reading

- "Mixture of Experts Explained" papers (2024-2026)
- vLLM and SGLang documentation
- NVIDIA research on efficient architectures
- Hugging Face Transformers source code

---

**Last Updated**: April 2026
