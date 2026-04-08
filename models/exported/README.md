## Exported Module - Model Export and Loading

The `exported` module handles model serialization, export to various formats, and loading of exported models with metadata preservation.

### Key Components

#### 1. **ModelExporter** (Abstract)
Base class for all model exporters:
- `export()` - Export model to specific format
- Format-specific implementations (PyTorchExporter, ONNXExporter, etc.)
- Metadata and configuration preservation
- Optional quantization and compression

#### 2. **PyTorchExporter**
Exports models in PyTorch format:
- State dict serialization
- Model configuration saving
- Metadata preservation
- Support for optimizer state (optional)

#### 3. **ONNXExporter**
Exports models to ONNX format:
- ONNX model generation
- Cross-platform compatibility
- Inference optimization
- Model information metadata

#### 4. **ModelLoader**
Loads exported models from disk:
- Format detection and loading
- Metadata restoration
- Configuration loading
- Device and dtype specification
- Caching support

#### 5. **ExportConfig** (Dataclass)
Export configuration:
- Target format specification
- Output directory
- Version information
- Quantization settings
- Compression options
- Metadata inclusion flags

#### 6. **LoadConfig** (Dataclass)
Load configuration:
- Model path specification
- Format specification
- Device settings
- Loading options
- Memory optimization flags

#### 7. **ExportFormat** (Enum)
Supported export formats:
- PYTORCH
- ONNX
- TENSORFLOW
- TORCHSCRIPT
- SAFETENSORS
- HUGGINGFACE
- CUSTOM

#### 8. **ExporterFactory**
Factory for creating exporters:
- Register custom exporters
- Create exporter instances
- Manage exporter implementations

### Usage Example

```python
from pathlib import Path
from models.exported import (
    ModelExporter,
    ModelLoader,
    ExportConfig,
    LoadConfig,
    ExportFormat,
    ExporterFactory,
)

# Export a model to PyTorch format
export_config = ExportConfig(
    export_format=ExportFormat.PYTORCH,
    output_dir=Path("./models"),
    model_name="my_model",
    version="1.0.0",
    include_metadata=True,
    quantize=False,
)

exporter = ExporterFactory.create_exporter(export_config)
result = exporter.export(model)

# Load the exported model
load_config = LoadConfig(
    model_path=Path("./models"),
    format=ExportFormat.PYTORCH,
    model_name="my_model",
    device="cuda",
    load_metadata=True,
)

loader = ModelLoader(load_config)
loaded_model = loader.load()
```

### Export Workflow

```
Model → ExportConfig → ExporterFactory → Exporter → Files + Metadata
```

### Load Workflow

```
Files + Metadata → LoadConfig → ModelLoader → Model Instance
```

### Supported Features

- **Multiple Formats**: PyTorch, ONNX, TensorFlow, TorchScript, SafeTensors
- **Metadata Preservation**: Save and restore model metadata
- **Configuration Saving**: Persist model configuration
- **Quantization**: Optional model quantization
- **Compression**: File compression support
- **Device Support**: Load on specific device (CPU, CUDA, etc.)
- **Optimization**: Inference optimization options
- **Extensibility**: Register custom exporters and loaders
