# LoRA Fine-tuning Module

## Overview

The `lora` module implements Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique that significantly reduces memory requirements and training time while maintaining model performance.

## Key Components

### Core Classes

- **`LoRALayer`**: Low-rank decomposition layer
  - Implements trainable A and B matrices with rank r
  - Proper weight initialization
  - Dropout support for regularization

- **`LoRALinear`**: Adapter for linear layers
  - Wraps original linear layers
  - Adds LoRA contribution: `output = original(x) + scaling * B(A(x))`
  - Maintains original layer parameters as frozen

- **`LoRAFinetuner`**: Main fine-tuning orchestrator
  - Extends `BaseFinetuner`
  - Automatically identifies and adapts target modules
  - Supports training, evaluation, and checkpoint management
  - Provides weight merging for inference optimization

### Configuration

- **`LoRAConfig`**: LoRA-specific configuration
  - `r`: Rank of low-rank decomposition (default: 8)
  - `lora_alpha`: Scaling factor (default: 16)
  - `target_modules`: Modules to apply LoRA (e.g., ["q_proj", "v_proj"])
  - `lora_dropout`: Dropout probability for LoRA layers
  - `bias`: Bias configuration ("none", "all", or "lora_only")

## How LoRA Works

LoRA adds trainable low-rank matrices to selected layers:

```
For a linear layer with weight W (d_in × d_out):
Output = xW + xBA  where A (d_in × r), B (r × d_out)

Parameters added: (d_in × r) + (r × d_out) << d_in × d_out
Typically: 1-2% of original parameters
```

## Usage

### Basic Setup

```python
from fine_tuning.lora import LoRAFinetuner, LoRAConfig

config = LoRAConfig(
    model_name="gpt2",
    output_dir="./output",
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

finetuner = LoRAFinetuner(config)
finetuner.setup_model()
finetuner.setup_optimizer()
finetuner.setup_scheduler()
```

### Training

```python
from torch.utils.data import DataLoader

# Assuming you have train_dataloader and eval_dataloader
results = finetuner.train(train_dataloader, eval_dataloader)

# Access results
print(f"Final training loss: {results['training_loss'][-1]}")
print(f"Final eval loss: {results['eval_loss'][-1]}")
```

### Weight Merging for Inference

```python
# Merge LoRA weights into base model
finetuner.merge_lora_weights()

# Now the model can be used for inference without LoRA layers
model = finetuner.get_model()
model.eval()
```

### Getting LoRA Weights

```python
# Get individual LoRA weights for inspection or export
lora_weights = finetuner.get_lora_weights()
for name, weight in lora_weights.items():
    print(f"{name}: {weight.shape}")
```

## Configuration Examples

### Small Models (GPT-2)

```python
config = LoRAConfig(
    model_name="gpt2",
    r=4,           # Smaller rank for small models
    lora_alpha=8,
    target_modules=["c_attn"],
)
```

### Medium Models (Mistral-7B)

```python
config = LoRAConfig(
    model_name="mistralai/Mistral-7B",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
```

### Large Models (Llama-70B)

```python
config = LoRAConfig(
    model_name="meta-llama/Llama-2-70b",
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
)
```

## Memory & Parameter Efficiency

### Example: GPT-2 (124M parameters)

| Approach | Parameters | Memory (GB) | Time |
|----------|-----------|-------------|------|
| Full Fine-tuning | 124M | ~12 | 100% |
| LoRA (r=8) | 1.6M | ~2 | ~30% |
| Reduction | 98.7% | ~83% | ~70% |

### Example: Llama-7B (7B parameters)

| Approach | Parameters | Memory (GB) | Time |
|----------|-----------|-------------|------|
| Full Fine-tuning | 7B | ~28 | 100% |
| LoRA (r=16) | 56M | ~4 | ~40% |
| Reduction | 99.2% | ~86% | ~60% |

## Advanced Features

### Target Module Selection

Choose modules strategically based on model architecture:

```python
# Transformers with attention
target_modules = ["q_proj", "v_proj"]  # Attention heads only

# All linear layers in attention
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Full fine-tuning equivalent
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
```

### Rank Selection

- Small rank (4-8): Maximum compression, ~1% of original params
- Medium rank (16-32): Balanced performance/efficiency
- Large rank (64+): Near full fine-tuning performance

### Inference Optimization

```python
# Option 1: Use model directly (with LoRA overhead)
output = model(**input)

# Option 2: Merge weights for faster inference
finetuner.merge_lora_weights()
output = model(**input)  # Faster, no LoRA overhead
```

## Integration with Other Modules

- **Base**: Uses `BaseFinetuner`, configuration, and utilities
- **QLoRA**: Builds on LoRA with quantization
- **Templates**: Provides configuration templates
- **Configs**: Centralized configuration management

## Best Practices

1. **Start with smaller rank and increase if needed**:
   ```python
   # Start conservative
   config.r = 8
   ```

2. **Use appropriate target modules**:
   ```python
   # Don't apply LoRA to embedding layers
   config.target_modules = ["q_proj", "v_proj"]
   ```

3. **Monitor training closely**:
   ```python
   finetuner.add_callback(CustomMonitoringCallback())
   ```

4. **Save checkpoints frequently**:
   ```python
   checkpoint = finetuner.save_checkpoint()
   ```

5. **Merge weights before deployment**:
   ```python
   finetuner.merge_lora_weights()
   model = finetuner.get_model()
   model.save_pretrained("./merged_model")
   ```

## Performance Tips

- Use smaller ranks (4-16) for memory-constrained environments
- Increase `lora_alpha` for stronger adaptation effects
- Apply LoRA to attention layers for best results
- Use mixed precision training for additional memory savings
- Consider QLoRA for even larger models

## Common Issues

### Issue: Out of Memory
- Reduce `batch_size`
- Reduce `r` (rank)
- Use mixed precision: `mixed_precision="fp16"`

### Issue: Poor Performance
- Increase `r` (rank)
- Increase `lora_alpha`
- Add more target modules
- Train longer with lower learning rate

### Issue: Slow Training
- Reduce model size
- Increase `batch_size`
- Use mixed precision
- Reduce logging frequency

## See Also

- [Base Module](../base/README.md)
- [QLoRA Module](../qlora/README.md)
- [Configuration Module](../configs/README.md)
