# Dragonborn Module

Advanced tooling and specialized tools framework for LLM-Whisperer with plugin and extension support.

## Features

- **Plugin System**: Dynamic plugin architecture with lifecycle management
- **Extension Loading**: Load extensions from entry points
- **Plugin Manager**: Manage plugin registration, loading, and execution
- **Advanced Toolkit**: Unified interface combining plugins and extensions
- **Tool Plugins**: Create custom tools using the plugin framework
- **Flexible Configuration**: Dataclass-based configuration system

## Components

### AdvancedToolkit

Main container for plugins and extensions.

```python
from tools.dragonborn import AdvancedToolkit, ToolkitConfig

config = ToolkitConfig(
    name="my_toolkit",
    description="Advanced tools"
)

toolkit = AdvancedToolkit(config)
toolkit.load_all()
```

### Plugin

Base class for creating plugins with lifecycle management.

```python
from tools.dragonborn import Plugin, PluginConfig, PluginType

class MyPlugin(Plugin):
    def load(self) -> bool:
        # Load plugin resources
        return True
    
    def unload(self) -> bool:
        # Cleanup
        return True
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        # Execute plugin
        return {"result": "success"}

plugin = MyPlugin(PluginConfig(
    name="my_plugin",
    plugin_type=PluginType.TOOL
))
```

### ToolPlugin

Specialized plugin for tool functionality.

```python
from tools.dragonborn import ToolPlugin, PluginConfig

def my_tool(param1: str, param2: int):
    return f"Processed: {param1} x {param2}"

plugin = ToolPlugin(PluginConfig(name="tools"))
plugin.register_tool("process", my_tool)
```

### PluginManager

Manages plugin lifecycle and execution.

Features:
- Plugin registration
- Plugin loading/unloading
- Batch operations
- Execution and status monitoring

```python
manager = PluginManager()
manager.register_plugin(my_plugin)
manager.load_plugin("my_plugin")
result = manager.execute_plugin("my_plugin", param="value")
```

### ExtensionLoader

Dynamically loads extensions from entry points.

```python
from tools.dragonborn import ExtensionLoader, ExtensionConfig

loader = ExtensionLoader()
ext_config = ExtensionConfig(
    name="my_extension",
    entry_point="my_module:MyExtensionClass"
)
loader.load_extension(ext_config)
```

## Configuration

### ToolkitConfig

```python
@dataclass
class ToolkitConfig:
    name: str
    description: str = ""
    plugins: List[PluginConfig] = field(default_factory=list)
    extensions: List[ExtensionConfig] = field(default_factory=list)
    enable_plugin_loading: bool = True
    enable_auto_discovery: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### PluginConfig

```python
@dataclass
class PluginConfig:
    name: str
    version: str = "1.0.0"
    plugin_type: PluginType = PluginType.TOOL
    description: str = ""
    author: str = ""
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ExtensionConfig

```python
@dataclass
class ExtensionConfig:
    name: str
    entry_point: str = ""
    required: bool = False
    auto_load: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
```

## Plugin Types

### PluginType

- `TOOL`: Provides tool functionality
- `EXTENSION`: Extends existing functionality
- `MIDDLEWARE`: Intercepts and modifies processing
- `PROCESSOR`: Processes data

## Plugin States

### PluginState

- `UNLOADED`: Not loaded
- `LOADING`: Currently loading
- `LOADED`: Successfully loaded
- `INITIALIZED`: Initialized and ready
- `RUNNING`: Currently executing
- `FAILED`: Load/execution failed
- `UNLOADING`: Currently unloading

## Examples

### Basic Plugin

```python
from tools.dragonborn import (
    AdvancedToolkit, ToolkitConfig, 
    PluginConfig, PluginType, ToolPlugin
)

# Create plugin
plugin_config = PluginConfig(
    name="math_tools",
    plugin_type=PluginType.TOOL,
    description="Math utilities"
)

plugin = ToolPlugin(plugin_config)

# Register tools
plugin.register_tool("add", lambda a, b: a + b)
plugin.register_tool("multiply", lambda a, b: a * b)

# Create toolkit
toolkit_config = ToolkitConfig(
    name="advanced_toolkit",
    plugins=[plugin_config]
)

toolkit = AdvancedToolkit(toolkit_config)
toolkit.load_all()

# Execute plugin
result = toolkit.execute_plugin("math_tools", tool_name="add", a=5, b=3)
```

### Custom Plugin

```python
from tools.dragonborn import Plugin, PluginConfig, PluginType

class DataProcessorPlugin(Plugin):
    def load(self) -> bool:
        self.processors = {}
        return True
    
    def unload(self) -> bool:
        self.processors.clear()
        return True
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        operation = kwargs.get("operation")
        data = kwargs.get("data")
        
        if operation == "transform":
            result = self._transform_data(data)
            return {"status": "success", "result": result}
        
        return {"error": "Unknown operation"}
    
    def _transform_data(self, data):
        return [x * 2 for x in data]

config = PluginConfig(
    name="processor",
    plugin_type=PluginType.PROCESSOR
)

plugin = DataProcessorPlugin(config)
```

### Extension Loading

```python
from tools.dragonborn import ExtensionLoader, ExtensionConfig

loader = ExtensionLoader()

# Load from entry point
ext_config = ExtensionConfig(
    name="nlp_tools",
    entry_point="nlp_module:NLPExtension"
)

success = loader.load_extension(ext_config)

# Get extension instance
ext = loader.get_extension("nlp_tools")
```

### Batch Plugin Management

```python
from tools.dragonborn import PluginManager

manager = PluginManager()

# Register plugins
for plugin in [plugin1, plugin2, plugin3]:
    manager.register_plugin(plugin)

# Load all
manager.load_all_plugins()

# List status
for status in manager.list_plugins():
    print(f"{status['name']}: {status['state']}")
```

## Usage

### Create Toolkit

```python
toolkit_config = ToolkitConfig(
    name="my_tools",
    plugins=[plugin1_config, plugin2_config],
    extensions=[ext1_config, ext2_config]
)

toolkit = AdvancedToolkit(toolkit_config)
loaded = toolkit.load_all()
print(f"Loaded: {loaded}")
```

### Register and Execute

```python
# Register plugin
manager.register_plugin(my_plugin)

# Load plugin
manager.load_plugin("my_plugin")

# Execute
result = manager.execute_plugin("my_plugin", param="value")
print(result)
```

### Plugin Information

```python
# Get plugin info
info = toolkit.get_plugin("tool_name").get_info()

# List all plugins
plugins = toolkit.list_plugins()

# List extensions
extensions = toolkit.list_extensions()
```

## License

MIT
