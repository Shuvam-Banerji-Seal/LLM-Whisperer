## Registry Module - Model Registry and Discovery

The `registry` module provides centralized model discovery, registration, and metadata management for the LLM-Whisperer framework.

### Key Components

#### 1. **ModelRegistry**
Central registry for managing models:
- `register_model()` - Register a new model
- `unregister_model()` - Remove a model
- `get_model()` - Retrieve model metadata by ID
- `search()` - Search models with advanced queries
- `list_models()` - List all registered models
- `get_statistics()` - Get registry statistics
- `export_registry()` - Export registry to file
- `import_registry()` - Import registry from file

#### 2. **ModelMetadata**
Comprehensive model information:
- Model identification (ID, name, version)
- Classification (type, framework)
- Attribution (author, organization, license)
- Description and tags
- Parameters and source information
- Usage tracking (downloads, ratings)
- Timestamps (registered, updated, accessed)
- Custom metadata support

#### 3. **RegistryQuery**
Advanced model search:
- Filter by name, type, framework
- Filter by author, organization, license
- Filter by tags
- Filter by parameter range
- Custom metadata filters
- Pagination support

#### 4. **RegistryConfig** (Dataclass)
Registry configuration:
- Storage backend selection
- Cache settings
- Search settings
- Auto-save settings

#### 5. **RegistryBackend** (Enum)
Supported storage backends:
- MEMORY - In-memory storage
- JSON_FILE - JSON file storage
- DATABASE - Database backend
- REMOTE - Remote registry service

### Usage Example

```python
from models.registry import (
    ModelRegistry,
    ModelMetadata,
    RegistryQuery,
    RegistryConfig,
    RegistryBackend,
)
from pathlib import Path

# Create registry
config = RegistryConfig(
    backend=RegistryBackend.MEMORY,
    enable_cache=True,
)

registry = ModelRegistry(config)

# Register models
metadata = registry.register_model(
    model_id="llama2-7b",
    name="Llama 2 7B",
    version="1.0.0",
    model_type="language_model",
    framework="pytorch",
    author="Meta",
    organization="Meta AI",
    num_parameters=7_000_000_000,
    tags=["llama", "language", "7b"],
    description="Llama 2 7B parameter model",
)

# Search models
query = RegistryQuery(
    framework="pytorch",
    min_parameters=1_000_000_000,
    tags=["llama"],
)

results = registry.search(query)

# Get model
model = registry.get_model("llama2-7b")

# List all models
models = registry.list_models()

# Get statistics
stats = registry.get_statistics()

# Export/Import
registry.export_registry(Path("registry.json"))
registry.import_registry(Path("registry.json"))
```

### Search Capabilities

- **Name Search**: Partial matching on model name
- **Type Filter**: Filter by model type (language, embedding, etc.)
- **Framework Filter**: Filter by framework (pytorch, tensorflow, etc.)
- **Author/Organization**: Filter by creator/organization
- **License**: Filter by license type
- **Tags**: Multi-tag filtering
- **Parameters**: Range-based parameter filtering
- **Custom Metadata**: Advanced filtering on custom fields
- **Pagination**: Limit and offset support

### Metadata Tracking

Each registered model tracks:
- Registration timestamp
- Last update timestamp
- Last access timestamp
- Download count
- User rating
- Custom metadata fields

### Registry Statistics

- Total number of models
- Total parameters across all models
- Available frameworks
- Available model types
- Cache statistics

### Extensibility

- Custom metadata support
- Multiple backend implementations
- Advanced search index
- Cache layer for performance
- Import/export functionality
