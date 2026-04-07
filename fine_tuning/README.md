# Fine-Tuning

Production-grade fine-tuning recipes and workflows for LLMs, covering multiple tuning strategies and optimization techniques.

## Overview

This module provides structured templates and best practices for fine-tuning language models:
- Full fine-tuning baselines
- Parameter-efficient methods (LoRA, QLoRA, Adapters)
- RAG-augmented tuning
- Behavior and task-specific tuning
- Agentic capability tuning
- Multimodal fine-tuning
- Reward and preference model training

## Structure

```
fine_tuning/
├── README.md (this file)
├── base/              # Full fine-tune baselines
├── lora/              # LoRA recipes and variants
├── qlora/             # Quantized LoRA (4-bit training)
├── rag_tuning/        # Retrieval-augmented training
├── behavior_tuning/   # Style, persona, policy tuning
├── agentic_tuning/    # Tool use and planning
├── multimodal/        # Vision-language model tuning
├── reward_modeling/   # Preference and reward models
├── configs/           # Shared training configurations
└── templates/         # Starter scripts and templates
```

## Quick Start

### 1. Basic LoRA Fine-Tuning

```bash
cd fine_tuning/lora
bash examples/train_mistral.sh
```

### 2. QLoRA (4-bit Quantized)

```bash
cd fine_tuning/qlora
python scripts/train_qlora.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset alpaca \
  --output_dir ./checkpoints
```

### 3. RAG-Augmented Tuning

```bash
cd fine_tuning/rag_tuning
python train_rag.py --config configs/rag_tuning.yaml
```

## Core Methods

| Method | Size | Speed | Quality | Cost | Best For |
|--------|------|-------|---------|------|----------|
| **Full** | 100% | Slow | Highest | Highest | State-of-art, specialized domains |
| **LoRA** | 0.1-1% | Fast | High | Low | Most production use cases |
| **QLoRA** | 0.05-0.5% | Fast | High | Very Low | Edge/mobile deployment |
| **Adapters** | 1-3% | Fast | Good | Low | Quick experimentation |
| **RAG** | 0-1% | Very Fast | Good* | Minimal | Knowledge-heavy tasks |
| **DPO/IPO** | 1-5% | Medium | High+ | Medium | Behavior alignment |

*Quality depends on retrieval quality, not tuning alone
+Compared to SFT

## Recommended Workflows

### For General Instruction Following
```
base/ → lora/ → multimodal/ (if vision needed)
```

### For Domain-Specific Tasks
```
base/ → rag_tuning/ → lora/ (for fine-grained control)
```

### For Production Deployment
```
lora/ → qlora/ (for edge) OR
lora/ → multimodal/ (for vision) OR
lora/ → agentic_tuning/ (for tool use)
```

### For Preference Optimization
```
base/ (SFT) → reward_modeling/ (PRM) → behavior_tuning/ (DPO/IPO)
```

## Configuration

All methods use standard configs in `configs/`:

```yaml
# configs/lora_base.yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  quantization: null
  lora_r: 64
  lora_alpha: 16
  lora_dropout: 0.05

training:
  learning_rate: 5e-4
  batch_size: 16
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01
  
dataset:
  name: "alpaca"
  split: 0.9
  max_length: 512
```

## Key Features by Method

### Full Fine-Tuning (`base/`)
- All parameters trainable
- Maximum model capacity utilization
- Highest quality results
- Best for large models with sufficient compute

### LoRA (`lora/`)
- Low-rank adaptations to attention
- 99% parameter reduction
- 5-10x faster than full tuning
- Most recommended for practitioners
- Examples: Mistral, Llama, Qwen

### QLoRA (`qlora/`)
- 4-bit quantization + LoRA
- 96% parameter reduction
- Trains 7B on single GPU
- Minimal quality loss
- Examples: 70B models on consumer GPUs

### RAG Tuning (`rag_tuning/`)
- Combine retrieval with fine-tuning
- Train model to use context effectively
- Minimal data requirements
- Excellent for knowledge-intensive tasks

### Behavior Tuning (`behavior_tuning/`)
- Instruction following
- Output format control
- Persona/style adaptation
- Policy/safety constraints
- Uses preference data (DPO, IPO, Contrastive)

### Agentic Tuning (`agentic_tuning/`)
- Tool use patterns
- Planning capabilities
- Multi-step reasoning
- Action-observation loops
- Function calling

### Multimodal (`multimodal/`)
- Vision-language models
- Image-text instruction tuning
- Visual reasoning
- Image captioning, VQA
- Examples: LLaVA, InternVL

### Reward Modeling (`reward_modeling/`)
- Bradley-Terry preference models
- Process reward models (at each step)
- Outcome reward models
- Score prediction models
- Foundation for RLHF/DPO

## Performance Benchmarks

### Training Speed (per epoch, 10K examples)
| Method | Llama-7B | Llama-13B | Llama-70B |
|--------|----------|-----------|----------|
| Full | 2h (8x GPU) | 4h (8x GPU) | 16h (8x GPU) |
| LoRA | 15min | 30min | 90min |
| QLoRA | 20min | 45min | 120min |
| RAG | 10min | 20min | 60min |

### Memory Requirements
| Method | Llama-7B | Llama-13B | Llama-70B |
|--------|----------|-----------|----------|
| Full | 56GB | 104GB | 560GB |
| LoRA | 16GB | 24GB | 64GB |
| QLoRA | 8GB | 12GB | 32GB |

## Advanced Topics

### Hyperparameter Tuning
- LoRA rank: 8-64 (higher = more capacity but slower)
- Learning rate: 5e-4 to 2e-3 (inverse to LoRA rank)
- Epochs: 1-5 (depends on data size)
- Batch size: 8-64 (depends on memory)

### Mixed Precision Training
- Use bf16 for modern GPUs (A100, H100)
- Use fp16 for older GPUs (V100, RTX)
- 50-70% memory savings with minimal quality loss

### Gradient Checkpointing
- Trade computation for memory
- 50% memory savings, 30% speed reduction
- Recommended for large models on limited memory

### Distributed Training
- Multi-GPU: Data parallelism (DistributedDataParallel)
- Multi-node: Use HuggingFace Accelerate
- See `../inference/engines/` for vLLM integration

### Evaluation
- Use `../evaluation/` for task-specific benchmarks
- Compare before/after metrics
- Include human evaluation for critical tasks

## Recipes

### Recipe 1: Quick LoRA Experiment (30 min)
```bash
cd fine_tuning/lora/examples
python train.py --config quick_start.yaml
```

### Recipe 2: Production LoRA (with validation)
```bash
cd fine_tuning/lora
bash scripts/train_with_eval.sh --model mistral --dataset custom
```

### Recipe 3: QLoRA for 70B Model
```bash
cd fine_tuning/qlora
python train_qlora.py --model meta-llama/Llama-2-70b-hf --batch_size 2
```

### Recipe 4: DPO for Preference Alignment
```bash
cd fine_tuning/behavior_tuning
python train_dpo.py --base_model ./checkpoints/lora --preference_data preferences.json
```

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM error | Batch too large | Reduce batch_size or enable gradient_checkpointing |
| Poor convergence | LR too high | Try 1e-4 to 5e-4 |
| Quality regression | Too many epochs | Start with 1-2 epochs, evaluate early |
| Slow training | No gradient accumulation | Use gradient_accumulation_steps: 4 |
| Bad instruction following | Data quality | Review examples, ensure diversity |

## References

- **LoRA Paper**: Hu et al., 2021 - "LoRA: Low-Rank Adaptation of Large Language Models"
- **QLoRA Paper**: Dettmers et al., 2023 - "QLoRA: Efficient Finetuning of Quantized LLMs"
- **DPO Paper**: Rafailov et al., 2023 - "Direct Preference Optimization"
- **RAG Tuning**: Lewis et al., 2020 + adaptations
- See `../skills/fine-tuning/` for comprehensive guides

## Tools & Dependencies

```bash
# Core dependencies
pip install torch transformers peft bitsandbytes

# For specific methods
pip install trl  # For DPO, PPO
pip install axolotl  # For comprehensive training
pip install unsloth  # For fast QLoRA
```

## Contributing

When adding a new fine-tuning method:
1. Create subdirectory with clear name
2. Add `README.md` with method overview
3. Include working example script
4. Add configuration template
5. Include evaluation metrics
6. Document trade-offs (speed/quality/memory)

## Quick Reference

**Best for budget constraints**: QLoRA  
**Best for speed**: RAG tuning  
**Best for quality**: Full fine-tuning or LoRA  
**Best for edge deployment**: QLoRA  
**Best for production**: LoRA (balance of everything)  

## License

See LICENSE file in repository root.
