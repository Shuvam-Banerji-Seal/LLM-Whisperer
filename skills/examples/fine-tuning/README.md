# Fine-Tuning Implementation

Complete guide to training-time optimization techniques for adapting LLMs to specific tasks.

## Overview

This implementation covers all major fine-tuning methods used in production:
- **Supervised Fine-Tuning (SFT)** - Standard instruction tuning
- **Direct Preference Optimization (DPO)** - Better than RLHF, no reference model
- **LoRA** - Parameter-efficient (0.1% trainable), single GPU
- **QLoRA** - 4-bit quantization + LoRA (single GPU with large models)
- **Adapters** - 2-5% trainable parameters, modular
- **Prefix Tuning** - Learnable prefix tokens
- **FSDP** - Fully Sharded Data Parallel for multi-GPU training

## Files Included

```
fine-tuning/
├── finetuning-complete.py    # Complete implementation (811 lines)
├── README.md                 # This file
└── Examples:
    ├── SFT on instruction data
    ├── DPO with preference pairs
    ├── LoRA with rank reduction
    ├── QLoRA on single GPU
    ├── Multi-GPU FSDP training
    └── Evaluation metrics
```

## Key Components

### 1. Supervised Fine-Tuning (SFT)
Standard instruction-following training on labeled data:

```python
from finetuning_complete import SupervisedFineTuning

trainer = SupervisedFineTuning(
    model_name="meta-llama/Llama-2-7b",
    dataset_path="instructions.jsonl",
    output_dir="./sft_model"
)

metrics = trainer.train(
    num_epochs=3,
    batch_size=32,
    learning_rate=1e-4,
    warmup_steps=100
)
```

**When to use**: Training on task-specific instructions
**Speedup**: Baseline (training on unoptimized model)
**Hardware**: Single GPU with 8GB+ VRAM
**Time**: 1-2 days for 7B model on 50K examples

### 2. Direct Preference Optimization (DPO)
Train model to prefer chosen over rejected responses without explicit RLHF:

```python
from finetuning_complete import DPOTrainer

trainer = DPOTrainer(
    model_name="meta-llama/Llama-2-7b",
    dataset_path="preferences.jsonl",  # {instruction, chosen, rejected}
    output_dir="./dpo_model"
)

metrics = trainer.train(
    num_epochs=2,
    batch_size=16,
    learning_rate=5e-5,
    beta=0.1  # Preference strength
)
```

**Key Differences from RLHF**:
- ✅ No reward model needed
- ✅ No reference model needed
- ✅ Single forward pass training
- ✅ Better alignment stability

**When to use**: When you have preference data (chosen vs rejected)
**Speedup**: 2-3x faster than RLHF (no reference model)
**Data Format**: `{instruction, chosen_response, rejected_response}`

### 3. LoRA (Low-Rank Adaptation)
Parameter-efficient fine-tuning with ~0.1% trainable parameters:

```python
from finetuning_complete import LoRATrainer

trainer = LoRATrainer(
    model_name="meta-llama/Llama-2-7b",
    dataset_path="instructions.jsonl",
    lora_config={
        "r": 8,  # Rank
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],  # Which layers
        "lora_dropout": 0.05
    },
    output_dir="./lora_weights"
)

metrics = trainer.train(
    num_epochs=3,
    batch_size=32,
    learning_rate=2e-4
)
```

**Advantages**:
- ✅ 99.9% less parameters to update
- ✅ Can fit 70B on single GPU
- ✅ Fast training (1/4 time of full FT)
- ✅ Supports multiple adapters per model

**Hardware**: Single GPU with 8GB VRAM (7B), 16GB (13B), 24GB (70B)
**Time**: 8 hours for 7B on 50K examples

### 4. QLoRA (Quantized LoRA)
4-bit quantization + LoRA for extreme memory efficiency:

```python
from finetuning_complete import QLoRATrainer

trainer = QLoRATrainer(
    model_name="meta-llama/Llama-2-70b",  # 70B on single GPU!
    dataset_path="instructions.jsonl",
    quantization_bits=4,
    lora_config={"r": 8, "lora_alpha": 16},
    output_dir="./qlora_weights"
)

metrics = trainer.train(
    num_epochs=1,
    batch_size=4,  # Smaller batch
    learning_rate=1e-4
)
```

**Memory Usage**:
| Model | Full FT | LoRA | QLoRA |
|-------|---------|------|-------|
| 7B | 28 GB | 9 GB | 6 GB |
| 13B | 52 GB | 16 GB | 10 GB |
| 70B | 280 GB | 90 GB | 48 GB |

**Trade-offs**:
- Slower training (4-bit ops less optimized)
- Slightly lower quality (quantization error)
- Can fine-tune 70B on consumer GPU!

### 5. Adapter Tuning
Bottleneck adapters with 2-5% trainable parameters:

```python
from finetuning_complete import AdapterTrainer

trainer = AdapterTrainer(
    model_name="meta-llama/Llama-2-7b",
    dataset_path="instructions.jsonl",
    adapter_config={
        "adapter_reduction_factor": 16,  # ~6% params
        "adapter_init": "bert"
    }
)

metrics = trainer.train(num_epochs=3, learning_rate=2e-4)
```

**Characteristics**:
- 2-5% trainable parameters (vs 0.1% for LoRA)
- Slightly better quality than LoRA
- Modular: Can stack multiple adapters
- Training speed: Similar to LoRA

### 6. Prefix Tuning
Learn a prefix of soft tokens prepended to input:

```python
from finetuning_complete import PrefixTuningTrainer

trainer = PrefixTuningTrainer(
    model_name="meta-llama/Llama-2-7b",
    dataset_path="instructions.jsonl",
    num_prefix_tokens=20,  # Number of learnable prefix tokens
    prefix_hidden_size=768
)

metrics = trainer.train(num_epochs=3, learning_rate=2e-4)
```

**Best for**:
- Few-shot in-context learning
- Task-specific prefix learning
- When fine-tuning full model is too expensive

### 7. FSDP (Fully Sharded Data Parallel)
Multi-GPU training with full gradient and parameter sharding:

```python
from finetuning_complete import FSDPDistributedTrainer

trainer = FSDPDistributedTrainer(
    model_name="meta-llama/Llama-2-70b",
    dataset_path="instructions.jsonl",
    num_gpus=8,
    sharding_strategy="FULL_SHARD",  # or SHARD_GRAD_OP, NO_SHARD
    output_dir="./fsdp_model"
)

metrics = trainer.train(
    num_epochs=1,
    batch_size=64,  # Per GPU
    learning_rate=2e-5
)
```

**Scaling**:
| GPUs | Memory per GPU | Training Time |
|------|----------------|---------------|
| 1 | 80 GB | 48 hours |
| 4 | 24 GB | 12 hours |
| 8 | 12 GB | 6 hours |

## Quick Comparison

| Method | Params | Memory | Speed | Quality | Ease |
|--------|--------|--------|-------|---------|------|
| SFT | 100% | High | Slow | Best | Easy |
| DPO | 100% | High | Slow | Great | Medium |
| LoRA | 0.1% | Low | Fast | Good | Easy |
| QLoRA | 0.1% | Very Low | Slow | Good | Medium |
| Adapter | 5% | Low | Fast | Good | Easy |
| Prefix | <1% | Very Low | Fast | OK | Hard |
| FSDP | 100% | Medium | Fast | Best | Hard |

## Quick Start

```python
# Simplest: LoRA fine-tuning
from finetuning_complete import LoRATrainer

trainer = LoRATrainer(
    model_name="meta-llama/Llama-2-7b",
    dataset_path="instructions.jsonl"
)

trainer.train(num_epochs=3, learning_rate=2e-4)
trainer.save_lora_weights("./checkpoints")

# Most efficient: QLoRA for large models
from finetuning_complete import QLoRATrainer

trainer = QLoRATrainer(
    model_name="meta-llama/Llama-2-70b",
    dataset_path="instructions.jsonl"
)

trainer.train(num_epochs=1, learning_rate=1e-4)
```

## Dataset Formats

### SFT/LoRA Format
```jsonl
{"instruction": "What is...", "output": "An answer..."}
{"instruction": "How do...", "output": "Here's how..."}
```

### DPO Format
```jsonl
{"instruction": "Question?", "chosen": "Good answer", "rejected": "Bad answer"}
```

## Evaluation Metrics

```python
from finetuning_complete import EvaluationMetrics

evaluator = EvaluationMetrics()

# Compute various metrics
results = evaluator.evaluate(
    predictions=model_outputs,
    references=ground_truth,
    metrics=["bleu", "rouge", "meteor", "exact_match"]
)
```

## Common Patterns

### Task-Specific Fine-Tuning (Recommended)
1. Start with SFT on task-specific instructions
2. Use LoRA for parameter efficiency
3. Evaluate on domain test set
4. Deploy with adapter weights

### Preference-Based Training (For Chat)
1. Collect human preference data
2. Use DPO for preference optimization
3. No extra reward model needed
4. Simpler than RLHF

### Large Model Deployment (Cost-Sensitive)
1. Use QLoRA for initial training
2. Merge LoRA weights into base model (optional)
3. Deploy with single adapter per customer
4. 100x cheaper than full model fine-tuning

## Troubleshooting

**Q: Training diverging?**
- Lower learning rate (try 5e-5 or 1e-4)
- Increase warmup steps
- Check for NaN in gradients

**Q: QLoRA too slow?**
- LoRA trains 2-3x faster
- Trade-off: Can fit larger models
- Use for 13B+ models only

**Q: LoRA not learning?**
- Increase rank (try r=16 or r=32)
- Check adapter layers (need sufficient coverage)
- Verify dataset quality

## References

- **LoRA**: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **QLoRA**: [QLoRA: Efficient Fine-tuning of LLMs](https://arxiv.org/abs/2305.14314)
- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **FSDP**: [Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
- **Adapters**: [Adapters for Transfer Learning](https://arxiv.org/abs/1902.00751)

## Integration with Other Skills

- **Quantization**: Combine with INT4 for extreme efficiency
- **RAG**: Fine-tune for domain-specific retrieval
- **Infrastructure**: Deploy with vLLM and adapters
- **Monitoring**: Track fine-tuning metrics in production

