# Advanced Architectures: Mixture of Experts (MoE)

Expert routing and sparse computation for efficient scaling to 1T+ parameters.

## Overview

Mixture of Experts (MoE) enables efficient scaling by activating only a subset of parameters per input token. Key benefits:
- **Sparse Computation**: Only 15-25% of parameters active per token (vs 100% in dense models)
- **Massive Scale**: Train 1T+ parameter models with same FLOPs as 100B dense models
- **Expert Specialization**: Each expert learns different aspects (syntax, facts, reasoning)
- **Production Efficiency**: High throughput with modest latency increase

## Files Included

```
advanced-architectures/
├── moe-implementation.py    # Complete implementation (374 lines)
├── README.md                # This file
└── Examples:
    ├── Basic Sparse MoE
    ├── Expert Choice routing
    ├── Load-balanced MoE
    ├── Efficient inference
    └── Expert analysis
```

## Key Components

### 1. Basic Sparse MoE with Top-K Routing

Route each token to top-K experts:

```python
from moe_implementation import TopKRouter, Expert, SparseMoELayer

# Create router and experts
router = TopKRouter(
    hidden_size=4096,
    num_experts=8,
    top_k=2  # Select top 2 experts per token
)

experts = [Expert(hidden_size=4096) for _ in range(8)]

moe_layer = SparseMoELayer(
    hidden_size=4096,
    num_experts=8,
    expert_hidden_size=16384,
    top_k=2
)

# Forward pass
hidden_states = torch.randn(batch_size, seq_len, 4096)
output = moe_layer(hidden_states)
```

**Top-K Routing**:
- Routes each token to K experts independently
- Weights computed via softmax over top-K logits
- Simple and efficient
- Can cause load imbalance (some experts underutilized)

### 2. Expert Choice Routing
Experts select tokens instead of tokens selecting experts:

```python
from moe_implementation import ExpertChoiceRouter

# Expert choice: experts select their top-k tokens
router = ExpertChoiceRouter(
    hidden_size=4096,
    num_experts=8,
    tokens_per_expert=32  # Each expert handles 32 tokens
)

output = router(hidden_states)
```

**Advantages**:
- ✅ Guaranteed load balance (each expert handles same tokens)
- ✅ Better utilization than top-k
- ✅ Simpler training (no aux loss needed)
- ❌ Requires careful batching
- ❌ Less intuitive routing pattern

### 3. Load-Balanced MoE
Auxiliary loss ensures balanced expert utilization:

```python
from moe_implementation import LoadBalancedMoE

moe = LoadBalancedMoE(
    hidden_size=4096,
    num_experts=128,
    top_k=2,
    load_balance_weight=0.01  # Auxiliary loss coefficient
)

output, aux_loss = moe(hidden_states)

# Total loss = task_loss + aux_loss * load_balance_weight
total_loss = task_loss + aux_loss * 0.01
```

**Load Balancing Loss**:
- Encourages even token distribution across experts
- Computed as difference between expected and actual utilization
- Added to training loss during backprop

### 4. Efficient Inference with Batching

Batch inference with proper token arrangement:

```python
from moe_implementation import EfficientMoEInference

inference = EfficientMoEInference(
    model=moe_model,
    batch_size=128,
    padding_tokens_threshold=0.2
)

# Process batch with token reorganization
outputs = inference.batch_forward(
    hidden_states,
    expert_batch_size=64
)
```

**Inference Optimization**:
- Group tokens by expert assignments
- Minimize padding and reshaping
- Reduce memory bandwidth requirements

## MoE Characteristics

### Parameter Efficiency

| Model | Params | Active/Token | Effective |
|-------|--------|-------------|-----------|
| Dense 70B | 70B | 100% | 70B |
| MoE 140B (8 experts) | 140B | 12.5% | 17.5B |
| MoE 280B (16 experts) | 280B | 6.25% | 17.5B |
| MoE 1T (128 experts) | 1T | 0.78% | 7.8B |

### Computation & Memory

| Model | FLOPs/Token | Memory |
|-------|------------|--------|
| Dense 70B | 140B ops | 280 GB (FP32) |
| MoE-140B | 35B ops | 35 GB (FP32) |
| Inference Speedup | 4x | 8x |

### Expert Specialization Examples

Real MoE models develop specialization:
- **Expert 1**: Syntax, tokenization patterns
- **Expert 2**: Mathematical reasoning
- **Expert 3**: Code and programming
- **Expert 4**: General knowledge facts
- **Expert 5**: Instruction following
- **Expert 6**: Creative writing
- **Expert 7**: Reasoning chains
- **Expert 8**: Language translation

## Quick Start

```python
# Minimal MoE setup
from moe_implementation import SparseMoELayer

# Single MoE layer
moe = SparseMoELayer(
    hidden_size=4096,
    num_experts=8,
    expert_hidden_size=16384,
    top_k=2
)

# Forward pass
hidden_states = torch.randn(batch_size=32, seq_len=512, hidden_size=4096)
output = moe(hidden_states)

# Integration into larger model
class MoETransformer(nn.Module):
    def __init__(self, num_layers=24):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': Attention(...),
                'moe': SparseMoELayer(...)  # Replace FFN with MoE
            })
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states):
        for layer in self.layers:
            # Standard attention
            hidden_states = layer['self_attn'](hidden_states)
            # MoE instead of dense FFN
            hidden_states = layer['moe'](hidden_states)
        return hidden_states
```

## Training Considerations

### Gradient Flow
MoE training requires special care:
1. **Router Gradients**: Small router loss helps optimize routing
2. **Expert Loss**: Ensure all experts receive gradient signals
3. **Load Balancing**: Auxiliary loss prevents expert collapse

```python
# Training loop
for batch in dataloader:
    output, aux_loss = model(batch)
    
    # Task loss
    task_loss = compute_loss(output, targets)
    
    # Total loss includes auxiliary load balancing
    total_loss = task_loss + 0.01 * aux_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Communication Overhead
Multi-GPU MoE training has communication costs:
- **All-to-All Communication**: Route tokens between GPUs
- **Expert Parallelism**: Distribute experts across GPUs
- **Data Parallelism**: Replicate routing layer

**Solution**: Use expert parallelism for models >100B parameters

## Common Patterns

### Small MoE (Experiments)
- 8 experts, top-k=2
- Total params: 20-50B
- FLOPs per token: 4-8B
- Use for prototyping

### Production MoE (Balanced)
- 128 experts, top-k=2-4
- Total params: 140-300B
- FLOPs per token: 15-35B
- Well-studied (Grok-1, Mixtral models)

### Large-Scale MoE
- 1000+ experts, top-k=2
- Total params: 1T+
- FLOPs per token: 10-15B
- Research/frontier models

## Expert Load Analysis

Monitor expert utilization:

```python
# Analyze expert assignments
router_logits = model.get_router_logits(batch)
expert_indices, expert_weights = torch.topk(router_logits, k=2)

# Calculate load per expert
load_per_expert = torch.zeros(num_experts)
for indices in expert_indices:
    for exp_id in indices:
        load_per_expert[exp_id] += 1

print(f"Expert utilization: {load_per_expert / load_per_expert.sum()}")
print(f"Load variance: {load_per_expert.std()}")

# Target: uniform distribution (1/num_experts each)
```

## Troubleshooting

**Q: Some experts inactive?**
- Add auxiliary load balancing loss
- Increase top-k slightly
- Check router initialization

**Q: Training instability?**
- Reduce auxiliary loss weight (try 0.001)
- Use smaller learning rate
- Stabilize router logits

**Q: Inference slow despite sparse computation?**
- Check communication overhead (multi-GPU)
- Verify expert batching working
- Profile to identify bottleneck

## References

- **MoE Overview**: [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- **Expert Choice**: [Expert Choice Routing](https://arxiv.org/abs/2202.09368)
- **Load Balancing**: [Base Layers for Language Models](https://arxiv.org/abs/2308.02239)
- **Mixtral MoE**: [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- **GShard**: [Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)

## Integration with Other Skills

- **Fast Inference**: MoE with KV-cache for efficient serving
- **Quantization**: Quantize individual experts separately
- **Infrastructure**: Serve MoE with expert parallelism
- **Monitoring**: Track per-expert utilization and load balance

