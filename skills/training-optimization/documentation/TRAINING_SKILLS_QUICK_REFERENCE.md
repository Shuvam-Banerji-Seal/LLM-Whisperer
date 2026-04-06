# Quick Skills Reference Guide

## 10 Advanced LLM Training Skills - Quick Index

### Training Optimization (7 Skills)

| Skill | Focus | Key Techniques | Memory Impact | Speed Impact |
|-------|-------|----------------|--------------|--------------|
| **Advanced Optimizers** | Optimizer selection | AdamW, LION, Sophia, SAM | Neutral | ±0-5% |
| **Learning Rate Scheduling** | LR decay strategy | Cosine, Linear, Poly, LLRD | Neutral | -2-5% accuracy |
| **Gradient Accumulation + Checkpointing** | Memory efficiency | Gradient accum, activation ckpt | -70-80% | +5-35% time |
| **Mixed Precision** | Numeric precision | FP16/BF16/FP8, loss scaling | -50% | +50-100% speed |
| **Distributed Training** | Multi-GPU/TPU | ZeRO, DDP, FSDP, all-reduce | -75% (ZeRO-3) | Near-linear scaling |
| **Mixture of Experts** | Sparse computation | Routing, load balancing | -60% (sparse) | +100-200% speed |
| **Modular Routing** | Dynamic dispatch | Soft/hard routing, gating | Variable | Task-dependent |

### Fine-Tuning (3 Skills)

| Skill | Focus | Key Methods | Params Trained | Memory vs FT |
|-------|-------|-------------|-----------------|--------------|
| **Advanced LoRA** | Rank-based tuning | QLoRA, DoRA, LoftQ | 0.1-1% | -75% (QLoRA) |
| **Adapters** | Bottleneck modules | MAD-X, IA³, Compacter | 2-10% | -80% |
| **Prefix Tuning** | Soft prompting | Prefix, P-Tuning v2 | 0.5-2% | -85% |

---

## When to Use Each Skill

### Pre-training from Scratch
**Must use:**
1. Learning Rate Scheduling → cosine annealing + warmup
2. Advanced Optimizers → AdamW or LION
3. Mixed Precision → BF16
4. Distributed Training → ZeRO-3 for 70B+
5. Optional: MoE for parameter efficiency

### Fine-tuning (Memory-Constrained)
**Recommended stack:**
1. QLoRA (4x memory savings)
2. Mixed Precision (2x savings)
3. Gradient Accumulation (effective batch size)
4. Learning Rate Scheduling → LLRD for better accuracy

### Fine-tuning (Multi-Task)
**Recommended stack:**
1. Multiple Adapters (task-specific)
2. Learning Rate Scheduling → warmup + decay
3. Prefix Tuning (few-shot scenarios)
4. Mixed Precision

### Large Scale (70B+ Models)
**Required:**
1. Distributed Training → ZeRO-3 + FSDP
2. Mixed Precision → BF16
3. Gradient Accumulation + Checkpointing
4. Learning Rate Scaling (linear rule)
5. Advanced Optimizers → LION or Sophia

### Inference/Deployment
**Post-training:**
1. Merge LoRA weights with base model
2. Fuse adapters or use dynamic routing
3. Quantize for inference speed
4. Profile memory and latency

---

## Quick Configuration Templates

### QLoRA Fine-tuning
```python
from peft import LoraConfig
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```
**Result**: 4x memory savings, minimal accuracy loss

### Learning Rate Schedule
```python
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=500,  # ~5% of training
    num_training_steps=10000
)
```
**Result**: Better convergence than constant LR

### Mixed Precision Training
```python
from torch.amp import autocast, GradScaler
scaler = GradScaler()
with autocast(device_type='cuda'):
    loss = model(batch)
scaler.scale(loss).backward()
```
**Result**: 2x memory, 1.5x speedup, no quality loss with BF16

### Distributed Training (ZeRO-3)
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    }
}
```
**Result**: 8x memory reduction, train 70B on 8xA100

---

## Decision Tree: Choosing Techniques

```
┌─ Are you memory-constrained?
│  ├─ Yes → Use QLoRA + Mixed Precision + Gradient Accumulation
│  └─ No → Proceed to next
│
├─ Pre-training (from scratch)?
│  ├─ Yes → Use Distributed Training (ZeRO) + Advanced Optimizer
│  └─ No → Proceed to next
│
├─ Fine-tuning single task?
│  ├─ Yes → Use QLoRA or LoRA + LLRD scheduling
│  └─ No → Proceed to next
│
├─ Fine-tuning multi-task?
│  ├─ Yes → Use Multiple Adapters + Task routing
│  └─ No → Proceed to next
│
├─ Very large model (70B+)?
│  ├─ Yes → MUST use Distributed Training + ZeRO-3
│  └─ No → Done
│
└─ Want sparse computation?
   ├─ Yes → Use Mixture of Experts or Modular Routing
   └─ No → Done
```

---

## Performance Benchmarks Summary

### Memory Requirements (7B Model)
- Baseline: 120 GB
- Mixed Precision (BF16): 60 GB (-50%)
- With Checkpointing: 35 GB (-70%)
- With Gradient Accum: 25 GB (-80%)
- QLoRA: 10 GB (-92%)

### Training Speed (relative to FP32)
- BF16: 1.5-2.0x faster
- BF16 + Checkpointing: 1.2-1.5x faster (overhead)
- Distributed (4 GPUs): 3.5x faster (scaling efficiency 88%)
- Mixed Precision + ZeRO: 2.5x faster (combined)

### Model Accuracy (GLUE benchmark)
- Full fine-tuning: 100% baseline
- LoRA (r=8): 99.5-100.0% (-0.5% at worst)
- QLoRA: 99.0-99.5% (-0.5-1.0%)
- DoRA: 99.8-100.0% (+0.2% sometimes)
- Adapter: 99.0-99.5% (-0.5-1.0%)

---

## Common Issues & Solutions

| Issue | Skill | Solution |
|-------|-------|----------|
| OOM (Out of Memory) | Gradient Accum + Checkpointing | Reduce batch size, increase accumulation steps |
| Slow training | Mixed Precision | Enable BF16, check GPU utilization |
| Poor fine-tuning accuracy | Learning Rate Scheduling | Use LLRD, reduce LR by 2-5x |
| NaN/Inf gradients | Mixed Precision | Increase loss scale, add gradient clipping |
| Distributed training hangs | Distributed Training | Check synchronization, use DDP debug mode |
| Uneven expert utilization | MoE Training | Adjust auxiliary loss weight or use bias-based balancing |

---

## Research Papers to Read (By Skill)

### Advanced Optimizers
- [AdamW: A Method for Stochastic Optimization](https://arxiv.org/abs/1711.05101)
- [LION: Evolution-tinged Optimizers Are Surprisingly Strong for Training Vision](https://arxiv.org/abs/2302.06675)
- [Sophia: A Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342)
- [Sharpness Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)

### Learning Rate Scheduling
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Attention is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
- [Large Batch Optimization for Deep Learning (LAMB)](https://arxiv.org/abs/1904.00325)

### Memory Efficiency
- [Training Deep Nets with Sublinear Memory Usage](https://arxiv.org/abs/1604.06174) - Chen et al. 2016
- [DeepSpeed: System Optimizations Enable Training Giant Models](https://arxiv.org/abs/2201.11605)

### Parameter-Efficient Fine-tuning
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366)

### Mixture of Experts
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

---

## Integration Examples

### Pre-training Recipe (70B model)
```python
# Optimizer: LION
# Schedule: Cosine + warmup
# Precision: BF16
# Memory: Checkpointing + Gradient Accumulation
# Distributed: ZeRO-3 with FSDP
# Expected: 8x memory reduction, 85-90% scaling efficiency
```

### Fine-tuning Recipe (7B model on single GPU)
```python
# Method: QLoRA
# Optimizer: AdamW
# Schedule: Linear + LLRD
# Precision: BF16
# Memory: Gradient Accumulation
# Expected: 4x memory savings, 99.5% accuracy
```

### Multi-task Fine-tuning (10B model)
```python
# Method: Multiple Adapters + Routing
# Optimizer: AdamW
# Schedule: Warmup + Linear decay
# Precision: BF16
# Memory: Moderate (2-5% params per task)
# Expected: Effective multi-task learning with task-specific specialization
```

---

## Skill Files for Reference

```
/skills/foundational/
└── advanced-optimization-algorithms.prompt.md
    
/skills/training-optimization/
├── learning-rate-scheduling.prompt.md
├── gradient-accumulation-checkpointing.prompt.md
├── mixed-precision-training.prompt.md
├── distributed-training-optimization.prompt.md
├── mixture-of-experts-training.prompt.md
└── modular-training-routing.prompt.md

/skills/fine-tuning/
├── lora-advanced-techniques.prompt.md
├── adapter-and-bottleneck-methods.prompt.md
└── prefix-tuning-and-prompting.prompt.md
```

Each skill file contains:
- Detailed theory and mathematical formulations
- Complete working code examples
- Empirical benchmarks and results
- Integration patterns with other skills
- References to research papers
- Best practices and common pitfalls

---

**Status**: Ready for immediate use in LLM training and fine-tuning projects  
**Last Updated**: April 6, 2026
