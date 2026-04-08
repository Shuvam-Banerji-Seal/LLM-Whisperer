## Merged Module - Model Merging and Combination

The `merged` module provides functionality for merging multiple models into a single combined model using various strategies and techniques.

### Key Components

#### 1. **ModelMerger** (Abstract)
Base class for all model mergers:
- `merge()` - Merge multiple models
- `add_model()` - Add models for merging
- `get_merge_info()` - Get merge information
- Strategy-specific implementations

#### 2. **LinearInterpolationMerger**
Merges models using linear interpolation:
- Configurable weights
- Smooth parameter blending
- Preserves model structure

#### 3. **WeightedAverageMerger**
Merges models using weighted averaging:
- Parameter-level averaging
- Configurable weights per model
- Weight normalization

#### 4. **MergedModel**
Represents a merged model:
- Source model tracking
- Merge strategy recording
- Weight preservation
- Model information retrieval

#### 5. **MergeConfig** (Dataclass)
Configuration for merging:
- Strategy specification
- Model list
- Weights and parameters
- Layer-wise settings
- Output configuration

#### 6. **MergeStrategy** (Enum)
Supported merge strategies:
- LINEAR_INTERPOLATION
- WEIGHTED_AVERAGE
- LAYER_WISE
- TASK_ARITHMETIC
- TIES_MERGING
- DARE_TIES
- SLERP
- CONSENSUS

#### 7. **MergerFactory**
Factory for creating mergers:
- Register custom mergers
- Create merger instances
- Manage merger implementations

### Usage Example

```python
from models.merged import (
    ModelMerger,
    MergedModel,
    MergeConfig,
    MergeStrategy,
    MergerFactory,
)

# Create merge configuration
config = MergeConfig(
    merge_strategy=MergeStrategy.WEIGHTED_AVERAGE,
    base_model_name="base_model",
    model_names=["model_1", "model_2", "model_3"],
    output_model_name="merged_model",
    weights=[0.5, 0.3, 0.2],
)

# Create merger
merger = MergerFactory.create_merger(config)

# Add models
merger.add_model("model_1", model1)
merger.add_model("model_2", model2)
merger.add_model("model_3", model3)

# Merge models
models_dict = {
    "model_1": model1,
    "model_2": model2,
    "model_3": model3,
}
result = merger.merge(models_dict)

# Create merged model representation
merged = MergedModel(
    name="merged_model",
    source_models=["model_1", "model_2", "model_3"],
    merge_strategy=MergeStrategy.WEIGHTED_AVERAGE,
    merged_data=result["merged_model"],
    weights=[0.5, 0.3, 0.2],
)

# Get information
info = merged.get_info()
```

### Merge Workflow

```
Models + Config → MergerFactory → Merger → Merge Process → MergedModel
```

### Supported Strategies

1. **Linear Interpolation**: Smooth blending between two models
2. **Weighted Average**: Parameter averaging with configurable weights
3. **Layer-wise**: Different merge strategies per layer
4. **Task Arithmetic**: Vector arithmetic in task space
5. **TIES Merging**: Trim, Intersect, and Exclude
6. **DARE TIES**: Distributed aware regularization
7. **SLERP**: Spherical linear interpolation
8. **Consensus**: Agreement-based merging

### Key Features

- **Multiple Strategies**: Support for various merging techniques
- **Flexible Weighting**: Per-model and per-layer weight configuration
- **History Tracking**: Record all merge operations
- **Extensibility**: Register custom mergers
- **Metadata Preservation**: Maintain merge information
- **Type Safety**: Type hints throughout
