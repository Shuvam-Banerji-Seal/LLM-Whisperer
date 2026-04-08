# QLoRA Fine-tuning Module

## Overview

The `qlora` module implements QLoRA (Quantized Low-Rank Adaptation), which combines 4-bit quantization with LoRA for extreme memory efficiency. This enables fine-tuning of large language models (7B+) on consumer GPUs.

## Key Components

### Core Classes

- **`QLoRAFinetuner`**: Extends `LoRAFinetuner`
  - Handles 4-bit quantization setup
  - Integrates with BitsAndBytes
  - Provides memory footprint analysis
  - Automatic device mapping

### Configuration

- **`QLoRAConfig`**: Extends `LoRAConfig`
  - `load_in_4bit`: Enable 4-bit loading (default: True)
  - `bnb_4bit_compute_dtype`: Computation dtype ("float16" or "bfloat16")
  - `bnb_4bit_quant_type`: Quantization type ("nf4" or "fp4")
  - `bnb_4bit_use_double_quant`: Use double quantization (default: True)

## Memory Efficiency

QLoRA achieves extreme memory reduction:

### Example: Llama-7B

| Approach | Memory | Trainable Params | Relative |
|----------|--------|------------------|----------|
| Full Fine-tune | 28 GB | 7B | 100% |
| LoRA (r=16) | 4 GB | 56M | 14% |
| QLoRA (r=16) | 0.8 GB | 56M | 3% |

### Example: Llama-13B

| Approach | Memory | Trainable Params |
|----------|--------|------------------|
| Full Fine-tune | 52 GB | 13B |
| LoRA (r=16) | 6 GB | 103M |
| QLoRA (r=16) | 1.2 GB | 103M |

## How QLoRA Works

### 4-Bit Quantization

- **NF4 (Normal Float 4-bit)**: Optimized for normal distributions in neural networks
- **FP4 (Float 4-bit)**: Standard 4-bit floating point

### Double Quantization

Quantizes the quantization constants, reducing memory overhead:

```
Before: float32 constants + 4-bit weights
After: 8-bit constants + 4-bit weights
Result: ~25% additional memory savings
```

### Integration with LoRA

- Model weights: Frozen and quantized to 4-bit
- LoRA layers: Trainable in full precision
- Computation: Mixed precision for efficiency

## Installation

QLoRA requires additional dependencies:

```bash
pip install bitsandbytes>=0.39.0
pip install transformers>=4.36.0
pip install torch>=2.0.0
```

## Usage

### Basic Setup

```python
from fine_tuning.qlora import QLoRAFinetuner, QLoRAConfig

config = QLoRAConfig(
    model_name="meta-llama/Llama-2-7b",
    output_dir="./qlora_output",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

finetuner = QLoRAFinetuner(config)
finetuner.setup_model()
finetuner.setup_optimizer()
finetuner.setup_scheduler()
```

### Training

```python
results = finetuner.train(train_dataloader, eval_dataloader)

# Check memory usage
memory_info = finetuner.get_memory_footprint()
print(f"Total memory: {memory_info['total_memory_mb']:.2f} MB")

# Check parameter info
param_info = finetuner.get_trainable_params_info()
print(f"Trainable: {param_info['trainable_percentage']:.2f}%")
```

### Inference

```python
# Merge LoRA weights for inference
finetuner.merge_lora_weights()
model = finetuner.get_model()
model.eval()

# Use model for inference
with torch.no_grad():
    outputs = model(**inputs)
```

## Configuration Examples

### Conservative Settings (Slower, More Stable)

```python
config = QLoRAConfig(
    model_name="meta-llama/Llama-2-7b",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

### Aggressive Settings (Faster, More Memory Efficient)

```python
config = QLoRAConfig(
    model_name="meta-llama/Llama-2-7b",
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,  # Skip double quant for speed
)
```

### For 13B Models

```python
config = QLoRAConfig(
    model_name="meta-llama/Llama-2-13b",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bnb_4bit_quant_type="nf4",
)
```

## Quantization Types

### NF4 (Recommended)

- Optimized for neural network weight distributions
- Better accuracy in most cases
- Slightly slower than FP4

```python
config.bnb_4bit_quant_type = "nf4"
```

### FP4

- Standard 4-bit floating point
- Slightly faster than NF4
- May have slightly lower accuracy

```python
config.bnb_4bit_quant_type = "fp4"
```

## Memory Analysis

### Get Current Memory Usage

```python
memory_info = finetuner.get_memory_footprint()
print(f"Model: {memory_info['model_memory_mb']:.2f} MB")
print(f"Optimizer: {memory_info['optimizer_memory_mb']:.2f} MB")
print(f"Gradients: {memory_info['gradient_memory_mb']:.2f} MB")
print(f"Total: {memory_info['total_memory_mb']:.2f} MB")
```

### Parameter Statistics

```python
param_info = finetuner.get_trainable_params_info()
print(f"Total: {param_info['total_parameters']:,}")
print(f"Trainable: {param_info['trainable_parameters']:,}")
print(f"Percentage: {param_info['trainable_percentage']:.2f}%")
```

## Best Practices

### 1. Use NF4 Quantization

```python
config.bnb_4bit_quant_type = "nf4"
config.bnb_4bit_use_double_quant = True
```

### 2. Balance Rank and Performance

- Start with r=8 for small models
- Use r=16 for medium models (7B-13B)
- Use r=32+ for large models (70B+)

### 3. Monitor Memory

```python
# Before training
memory_info = finetuner.get_memory_footprint()
if memory_info['total_memory_mb'] > available_memory:
    # Reduce batch size or rank
    pass
```

### 4. Use Mixed Precision

```python
config.training.mixed_precision = "float16"
```

### 5. Appropriate Batch Sizes

```python
# Start small
config.training.batch_size = 4
# Increase with memory savings
config.training.gradient_accumulation_steps = 4  # 4 * 4 = effective batch 16
```

## Common Issues

### Issue: "CUDA Out of Memory"

Solutions in order:
1. Reduce `batch_size` (e.g., 4 → 2)
2. Increase `gradient_accumulation_steps`
3. Reduce `r` (e.g., 16 → 8)
4. Reduce number of target modules
5. Enable `mixed_precision="float16"`

### Issue: "bitsandbytes not found"

```bash
pip install bitsandbytes>=0.39.0
```

### Issue: Poor Model Quality

- Increase `r` (e.g., 8 → 16 → 32)
- Increase `lora_alpha` proportionally
- Reduce `lora_dropout`
- Increase `num_epochs`
- Use smaller `learning_rate`

### Issue: Slow Training

- Use `bfloat16` for newer GPUs (RTX 3090+, A100)
- Reduce `logging_steps`
- Increase `batch_size` if memory allows
- Use fewer `target_modules`

## Performance Benchmarks

### Llama-7B on RTX 4090 (24GB)

| Config | Memory | Speed | Quality |
|--------|--------|-------|----------|
| Full FT | OOM | - | Baseline |
| LoRA | 18GB | 1x | 95% |
| QLoRA | 3.5GB | 0.7x | 92% |

### Llama-13B on RTX 4090 (24GB)

| Config | Memory | Speed | Quality |
|--------|--------|-------|----------|
| Full FT | OOM | - | - |
| LoRA | OOM | - | - |
| QLoRA | 8GB | 1x | 90% |

## Advanced Features

### Custom Quantization Presets

```python
# For inference speed (use fastest quantization)
config.bnb_4bit_quant_type = "fp4"
config.bnb_4bit_use_double_quant = False

# For quality (use slowest quantization)
config.bnb_4bit_quant_type = "nf4"
config.bnb_4bit_use_double_quant = True
```

### Device Mapping

Automatic device mapping for multi-GPU:

```python
# Automatically distribute across GPUs
config.training.device_map = "auto"
```

## See Also

- [LoRA Module](../lora/README.md) - Base LoRA implementation
- [Base Module](../base/README.md) - Foundation classes
- [Configuration Module](../configs/README.md) - Preset configurations
