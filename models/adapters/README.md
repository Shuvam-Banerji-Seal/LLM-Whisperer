## Adapters Module - Model Adaptation Framework

The `adapters` module provides a framework for parameter-efficient model adaptation using various techniques like LoRA, Prefix Tuning, and custom adapters.

### Key Components

#### 1. **ModelAdapter** (Abstract)
Base class for all model adapters:
- `attach()` - Attach adapter to a model
- `detach()` - Detach adapter from model
- `enable()` - Enable adapter for use
- `disable()` - Disable adapter
- `get_trainable_params()` - Get trainable parameters
- `get_config()` - Get adapter configuration
- `get_info()` - Get adapter information

#### 2. **LoRAAdapter**
LoRA (Low-Rank Adaptation) implementation:
- Efficient rank-based adaptation
- Configurable rank and alpha parameters
- Per-layer LoRA addition
- Low memory footprint

#### 3. **PrefixTuningAdapter**
Prefix Tuning implementation:
- Prefix token prepending
- Configurable prefix length
- Minimal parameter overhead

#### 4. **AdapterRegistry**
Centralized adapter management:
- Register adapters
- Retrieve adapters by name
- Compose multiple adapters
- List available adapters
- Retrieve adapter information

#### 5. **AdapterConfig** (Dataclass)
Configuration for adapters:
- Adapter identification
- Type specification
- Base model reference
- Adapter-specific parameters
- Training settings
- Composability settings

#### 6. **AdapterType** (Enum)
Supported adapter types:
- PREFIX_TUNING
- LORA
- BOTTLENECK
- ADAPTER_LAYER
- HYPERNETWORK
- BITFIT
- COMPACTER
- CUSTOM

### Usage Example

```python
from models.adapters import (
    LoRAAdapter,
    AdapterConfig,
    AdapterRegistry,
    AdapterType,
)

# Create adapter configuration
config = AdapterConfig(
    name="my_lora",
    adapter_type=AdapterType.LORA,
    base_model_name="base_model",
    rank=8,
)

# Create LoRA adapter
adapter = LoRAAdapter(config, rank=8, alpha=16.0)

# Create registry and add adapter
registry = AdapterRegistry()
registry.add_adapter(adapter)

# Attach adapter to model
adapter.attach(base_model)
adapter.enable()

# Get trainable parameters
params = adapter.get_trainable_params()

# Compose adapters
registry.compose_adapters("task_composition", ["lora_1", "lora_2"])

# Get composed adapters
composed = registry.get_composed_adapters("task_composition")
```

### Adapter Lifecycle

```
Create Config → Create Adapter → Register → Attach → Enable → Use
                                                 ↓
                                           Disable → Detach
```

### Key Features

- **Parameter Efficiency**: Minimal additional parameters for fine-tuning
- **Composability**: Combine multiple adapters for complex tasks
- **Flexibility**: Support for multiple adaptation techniques
- **Management**: Centralized registry for adapter lifecycle management
- **Type Safety**: Type hints throughout for better development experience
