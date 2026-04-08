"""Configuration dataclasses for dragonborn module."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class PluginType(Enum):
    """Types of plugins."""

    TOOL = "tool"
    EXTENSION = "extension"
    MIDDLEWARE = "middleware"
    PROCESSOR = "processor"


class PluginState(Enum):
    """Plugin lifecycle states."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FAILED = "failed"
    UNLOADING = "unloading"


@dataclass
class PluginConfig:
    """Configuration for a plugin.

    Attributes:
        name: Plugin name
        version: Plugin version
        plugin_type: Type of plugin
        description: Plugin description
        author: Plugin author
        enabled: Whether plugin is enabled
        dependencies: List of plugin dependencies
        settings: Plugin settings
        metadata: Additional metadata
    """

    name: str
    version: str = "1.0.0"
    plugin_type: PluginType = PluginType.TOOL
    description: str = ""
    author: str = ""
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtensionConfig:
    """Configuration for extensions.

    Attributes:
        name: Extension name
        entry_point: Module path to extension
        required: Whether extension is required
        auto_load: Whether to auto-load extension
        config: Extension-specific configuration
    """

    name: str
    entry_point: str = ""
    required: bool = False
    auto_load: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolkitConfig:
    """Configuration for advanced toolkit.

    Attributes:
        name: Toolkit name
        description: Toolkit description
        plugins: List of plugin configurations
        extensions: List of extension configurations
        enable_plugin_loading: Enable dynamic plugin loading
        enable_auto_discovery: Enable automatic plugin discovery
        metadata: Additional metadata
    """

    name: str
    description: str = ""
    plugins: List[PluginConfig] = field(default_factory=list)
    extensions: List[ExtensionConfig] = field(default_factory=list)
    enable_plugin_loading: bool = True
    enable_auto_discovery: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
