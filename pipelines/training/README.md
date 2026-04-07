# Training Pipeline README

Complete guide for training pipeline used in LLM-Whisperer.

## Quick Start

### 1. Full Fine-Tuning

```bash
python scripts/train.py --config configs/default_config.yaml
```

### 2. LoRA Fine-Tuning

```bash
python scripts/train.py --config configs/lora_config.yaml
```

### 3. QLoRA Fine-Tuning

```bash
python scripts/train.py --config configs/qlora_config.yaml
```

### 4. DPO Training

```bash
python scripts/train.py --config configs/dpo_config.yaml
```

## Architecture

### Core Modules

#### 1. **Orchestrator** (`src/orchestrator.py`)

Main training orchestration class that manages:
- Model loading and initialization
- Training setup (optimizer, scheduler)
- Checkpoint management
- Monitoring integration

```python
from pipelines.training.src.orchestrator import TrainingOrchestrator, TrainingConfig

config = TrainingConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    dataset_path="data/processed",
    output_dir="./outputs",
    training_method="lora",
    lora_rank=64
)

orchestrator = TrainingOrchestrator(config)
orchestrator.load_model()
orchestrator.setup_training()
```

#### 2. **Callbacks** (`src/callbacks.py`)

Extensible callback system for custom training logic:
- **LoggingCallback**: Console logging
- **WandbCallback**: Weights & Biases integration
- **MLflowCallback**: MLflow experiment tracking
- Custom callbacks: Extend `TrainingCallback` base class

```python
from pipelines.training.src.callbacks import (
    CallbackManager, LoggingCallback, WandbCallback
)

manager = CallbackManager()
manager.add_callback(LoggingCallback(log_interval=100))
manager.add_callback(WandbCallback())

manager.on_training_start({"model": "mistral-7b"})
manager.on_step_end(100, {"loss": 2.5})
manager.on_training_end({"final_loss": 1.8})
```

#### 3. **Checkpointing** (`src/checkpointing.py`)

Model checkpoint management:
- Automatic checkpoint saving
- Best checkpoint tracking
- Checkpoint loading and recovery
- Metadata tracking

```python
from pipelines.training.src.checkpointing import CheckpointManager

manager = CheckpointManager("checkpoints/", max_checkpoints=5)
manager.save_checkpoint(
    model, 
    step=1000,
    epoch=1,
    loss=2.3,
    eval_loss=2.1,
    is_best=True
)

# Load best checkpoint
model, metadata = manager.load_checkpoint(model)
```

#### 4. **Training Methods** (`src/methods.py`)

Support for multiple training approaches:
- **FullFineTune**: All parameters trainable
- **LoRA**: Low-rank adaptation (~0.1% trainable)
- **QLoRA**: Quantized LoRA
- **DPO**: Direct preference optimization

```python
from pipelines.training.src.methods import TrainingMethodFactory

# Apply LoRA
model = TrainingMethodFactory.apply_method(
    model,
    "lora",
    {"lora_rank": 64, "lora_alpha": 16}
)
```

## Configuration Options

### Model Configuration

```yaml
training:
  model_name: "mistralai/Mistral-7B-v0.1"
  dataset_path: "data/processed"
  output_dir: "./training_outputs"
  
  # Model parameters
  lora_rank: 64          # For LoRA/QLoRA
  lora_alpha: 16
  lora_dropout: 0.05
  quantization_enabled: false
  
  # Training method
  training_method: lora  # full_finetune, lora, qlora, dpo
```

### Training Hyperparameters

```yaml
training:
  # Core training
  num_epochs: 3
  batch_size: 16
  learning_rate: 5e-4
  warmup_steps: 100
  max_grad_norm: 1.0
  weight_decay: 0.01
  
  # Optimization
  mixed_precision: true
  gradient_checkpointing: true
  gradient_accumulation_steps: 1
```

### Monitoring Configuration

```yaml
training:
  # Checkpointing
  eval_steps: 500
  save_steps: 500
  log_steps: 100
  
  # Weights & Biases
  use_wandb: true
  wandb_project: "llm-training"
  
  # MLflow
  use_mlflow: false
  mlflow_experiment: null
```

## Training Methods Comparison

| Method | Trainable % | Memory | Speed | Quality |
|--------|------------|--------|-------|---------|
| Full Fine-Tune | 100% | High | Baseline | Best |
| LoRA | ~0.1% | ~50% | ~1.5x | Very Good |
| QLoRA | ~0.1% | ~25% | ~2x | Good |
| DPO | 100% | High | Slow | Best (alignment) |

## Examples

### Example 1: Basic LoRA Training

```yaml
# configs/my_lora.yaml
training:
  model_name: "mistralai/Mistral-7B-v0.1"
  dataset_path: "data/processed/alpaca"
  output_dir: "./outputs/alpaca-lora"
  training_method: lora
  lora_rank: 64
  batch_size: 16
  num_epochs: 3
  learning_rate: 5e-4
  use_wandb: true
  wandb_project: my-project
```

Run with:
```bash
python scripts/train.py --config configs/my_lora.yaml
```

### Example 2: QLoRA on Limited GPU

```yaml
# configs/memory_efficient.yaml
training:
  model_name: "mistralai/Mistral-7B-v0.1"
  dataset_path: "data/processed"
  output_dir: "./outputs/qlora"
  training_method: qlora
  quantization_enabled: true
  lora_rank: 32
  batch_size: 4
  gradient_accumulation_steps: 8
  mixed_precision: true
  gradient_checkpointing: true
```

### Example 3: DPO Training

```yaml
# configs/dpo_training.yaml
training:
  model_name: "mistralai/Mistral-7B-v0.1"
  dataset_path: "data/processed/dpo_pairs"
  output_dir: "./outputs/dpo"
  training_method: dpo
  num_epochs: 2
  batch_size: 8
  learning_rate: 5e-5  # Lower LR for DPO
```

## Best Practices

1. **Start with LoRA**: Quick iteration with minimal memory
2. **Monitor metrics**: Use W&B or MLflow for tracking
3. **Use gradient checkpointing**: Saves memory with minimal speed cost
4. **Checkpoint frequently**: Save best model for recovery
5. **Validate regularly**: Evaluate every 500-1000 steps
6. **Profile early**: Measure memory and speed with small data

## Common Issues

### Issue: Out of Memory
**Solution**: 
- Reduce batch_size
- Enable gradient_checkpointing
- Use LoRA/QLoRA instead of full fine-tune
- Reduce max_seq_length

### Issue: Training too slow
**Solution**:
- Increase batch_size
- Increase gradient_accumulation_steps
- Use mixed precision (mixed_precision: true)
- Use QLoRA instead of LoRA

### Issue: Poor convergence
**Solution**:
- Adjust learning_rate
- Increase warmup_steps
- Check data quality
- Verify gradient flow

## Integration with Data Pipeline

Use data pipeline outputs for training:

```python
from pipelines.data.src.splitting import DataSplitting

# Load preprocessed data from data pipeline
splitter = DataSplitting(...)
splitter.load_splits("data/processed")

train_data = splitter.splits['train']
val_data = splitter.splits['val']
```

## Contributing

When adding features:
1. Extend `TrainingCallback` for custom logic
2. Add new training method in `methods.py`
3. Update configuration examples
4. Document hyperparameter choices

## License

See LICENSE file in repository root.
