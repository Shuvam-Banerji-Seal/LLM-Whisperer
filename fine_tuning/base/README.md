# Base Fine-tuning Module

## Overview

The `base` module provides core abstractions, configurations, and utilities for fine-tuning large language models. It serves as the foundation for all other fine-tuning approaches in the LLM-Whisperer system.

## Key Components

### Core Classes

- **`BaseFinetuner`**: Abstract base class for all fine-tuning approaches
  - Manages training lifecycle
  - Handles device management and checkpointing
  - Supports callbacks for extensibility
  - Type-safe with comprehensive error handling

- **`FinetuningCallback`**: Abstract base class for fine-tuning callbacks
  - Extensible callback system
  - Hooks for training events (init, epoch start/end, step end, train end)
  - Enables monitoring and custom logic injection

- **`FinetuningMetrics`**: Container for fine-tuning metrics
  - Captures loss, accuracy, perplexity
  - Supports custom metrics
  - Timestamped for tracking

- **`FinetuningState`**: Manages training state
  - Tracks epochs, steps, and best metrics
  - Maintains metrics history
  - Provides summary statistics

### Configuration Classes

- **`BaseFinetuningConfig`**: Main configuration class
  - Model and output directory settings
  - Optimizer and scheduler configuration
  - Training hyperparameters
  - Validation and to_dict conversion

- **`OptimizerConfig`**: Optimizer configuration
  - Supports multiple optimizer types (Adam, AdamW, SGD, 8-bit variants)
  - Configurable learning rates, weight decay, and gradient norms
  - Built-in validation

- **`SchedulerConfig`**: Learning rate scheduler configuration
  - Multiple scheduler types (linear, cosine, polynomial, etc.)
  - Warmup and cycle configuration
  - Validation of step counts

- **`TrainingConfig`**: Training hyperparameters
  - Batch sizes, epochs, sequence length
  - Mixed precision settings
  - Checkpoint and logging strategies
  - Data loading configuration

### Utilities

- **`setup_logging()`**: Configure logging for training
- **`set_seed()`**: Set random seeds for reproducibility
- **`get_device()`**: Get appropriate device (CPU/CUDA)
- **`save_checkpoint()` / `load_checkpoint()`**: Checkpoint management
- **`count_parameters()` / `count_trainable_parameters()`**: Model analysis
- **`get_model_size_mb()`**: Memory analysis
- **`print_model_info()`**: Detailed model information
- **`cleanup_checkpoints()`**: Automatic checkpoint cleanup

## Usage

### Basic Setup

```python
from fine_tuning.base import (
    BaseFinetuningConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)

# Create configuration
config = BaseFinetuningConfig(
    model_name="gpt2",
    output_dir="./output",
    optimizer=OptimizerConfig(
        learning_rate=5e-5,
        weight_decay=0.01,
    ),
    training=TrainingConfig(
        num_epochs=3,
        batch_size=32,
        max_seq_length=512,
    ),
)
```

### Custom Fine-tuner Implementation

```python
from fine_tuning.base import BaseFinetuner

class MyFinetuner(BaseFinetuner):
    def setup_model(self):
        # Load and setup model
        pass

    def setup_optimizer(self):
        # Create optimizer
        pass

    def setup_scheduler(self):
        # Create scheduler
        pass

    def train(self, train_dataloader, eval_dataloader=None):
        # Implement training logic
        pass

    def evaluate(self, eval_dataloader):
        # Implement evaluation logic
        pass

# Use custom fine-tuner
finetuner = MyFinetuner(config)
finetuner.setup_model()
finetuner.setup_optimizer()
finetuner.setup_scheduler()
```

### Callbacks

```python
from fine_tuning.base import FinetuningCallback, FinetuningMetrics

class LoggingCallback(FinetuningCallback):
    def on_epoch_end(self, trainer, metrics: FinetuningMetrics):
        print(f"Epoch {metrics.epoch}: loss={metrics.loss}")

    # Implement other hooks...

finetuner.add_callback(LoggingCallback())
```

## Configuration Validation

All configuration classes include validation:

```python
# These will raise ValueError
config = OptimizerConfig(learning_rate=-1)  # Invalid learning rate
config = TrainingConfig(num_epochs=0)        # Invalid epoch count
config = BaseFinetuningConfig(model_name="") # Missing required field
```

## Error Handling

The module provides comprehensive error handling:

```python
try:
    checkpoint = load_checkpoint("path/to/checkpoint")
except FileNotFoundError:
    print("Checkpoint not found")

# Model must be setup before use
try:
    model = finetuner.get_model()  # Raises RuntimeError if not setup
except RuntimeError as e:
    print(f"Error: {e}")
```

## Type Safety

All code includes comprehensive type hints for IDE support and type checking:

```python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    config: Dict[str, Any],
    state: Dict[str, Any],
    checkpoint_path: str,
) -> None:
```

## Dependencies

- `torch`: PyTorch for model and optimization
- `numpy`: Numerical operations
- `dataclasses`: Configuration management

## Integration

The base module is designed to be extended by:

- **LoRA Fine-tuning**: Parameter-efficient fine-tuning
- **QLoRA Fine-tuning**: Quantized LoRA
- **Behavior Tuning**: Behavior-specific fine-tuning
- **Agentic Tuning**: Agent system fine-tuning
- **RAG Tuning**: RAG system fine-tuning
- **Reward Modeling**: Reward model training
- **Multimodal Tuning**: Multimodal model fine-tuning

## Best Practices

1. **Always set seed for reproducibility**:
   ```python
   from fine_tuning.base import set_seed
   set_seed(42)
   ```

2. **Use appropriate mixed precision**:
   ```python
   config.training.mixed_precision = "fp16"  # or "bf16"
   ```

3. **Monitor training with callbacks**:
   ```python
   finetuner.add_callback(CustomMonitoringCallback())
   ```

4. **Save checkpoints regularly**:
   ```python
   checkpoint_path = finetuner.save_checkpoint()
   ```

5. **Cleanup old checkpoints**:
   ```python
   from fine_tuning.base import cleanup_checkpoints
   cleanup_checkpoints("./output", keep_last_n=3)
   ```

## See Also

- [LoRA Module](../lora/README.md)
- [QLoRA Module](../qlora/README.md)
- [RAG Tuning Module](../rag_tuning/README.md)
