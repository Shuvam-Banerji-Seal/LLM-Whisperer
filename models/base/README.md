## Base Module - Model Classes and Interfaces

The `base` module provides the foundational classes and interfaces for all models in the LLM-Whisperer framework.

### Key Components

#### 1. **ModelInterface** (Abstract)
Defines the standard interface that all models must implement:
- `load()` - Load model into memory
- `unload()` - Unload model from memory
- `predict()` - Generate predictions
- `get_config()` - Retrieve model configuration
- `get_metadata()` - Retrieve model metadata
- `is_loaded()` - Check if model is loaded

#### 2. **BaseModel**
Concrete implementation providing:
- Lifecycle management (load/unload)
- Status tracking (UNINITIALIZED, LOADING, READY, RUNNING, ERROR, UNLOADED)
- Configuration and metadata management
- Common error handling and logging
- Protected methods for subclass customization (`_load_impl`, `_unload_impl`, `_predict_impl`)

#### 3. **ModelConfig** (Dataclass)
Configuration structure containing:
- Model identification (name, version, description)
- Model parameters (size, num_parameters, vocab_size)
- Inference settings (batch_size, precision, device)
- Framework information
- Custom metadata

#### 4. **ModelMetadata** (Dataclass)
Runtime metadata containing:
- Creation and modification timestamps
- Author and organization info
- Framework and type information
- Source URL and local path
- Performance metrics
- Custom metadata

#### 5. **ModelFactory**
Factory pattern implementation for creating models:
- Register custom model builders
- Create model instances by type
- List registered model types

### Usage Example

```python
from models.base import BaseModel, ModelConfig, ModelType

# Create configuration
config = ModelConfig(
    name="my_model",
    model_type=ModelType.LANGUAGE_MODEL,
    version="1.0.0",
    num_parameters=7_000_000_000,
)

# Create model instance
class MyModel(BaseModel):
    def _load_impl(self):
        # Framework-specific loading logic
        pass
    
    def _predict_impl(self, inputs, **kwargs):
        # Framework-specific prediction logic
        return results

model = MyModel(config)
model.load()
outputs = model.predict(inputs)
model.unload()
```

### Status Transitions

```
UNINITIALIZED → LOADING → READY ⟷ RUNNING → READY
                                ↘ ERROR ↙
```

### Extension Points

Subclasses can override:
- `_load_impl()` - Custom loading logic
- `_unload_impl()` - Custom unloading logic
- `_predict_impl()` - Custom prediction logic
