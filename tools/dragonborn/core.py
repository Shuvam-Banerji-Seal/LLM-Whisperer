"""Core advanced tooling and plugin framework."""

import logging
import importlib
import sys
from typing import Dict, Any, Optional, List, Callable, Type
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

from .config import (
    PluginConfig,
    ExtensionConfig,
    ToolkitConfig,
    PluginType,
    PluginState,
)

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Base class for plugins.

    Provides lifecycle management and extension capabilities.
    """

    def __init__(self, config: PluginConfig):
        """Initialize plugin.

        Args:
            config: Plugin configuration
        """
        self.config = config
        self.id = str(uuid.uuid4())
        self.state = PluginState.UNLOADED
        self.metadata: Dict[str, Any] = config.metadata.copy()
        self.settings: Dict[str, Any] = config.settings.copy()
        self.created_at = datetime.now()
        self.loaded_at: Optional[datetime] = None
        logger.debug(f"Plugin {config.name} initialized (id={self.id})")

    @abstractmethod
    def load(self) -> bool:
        """Load plugin.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def unload(self) -> bool:
        """Unload plugin.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute plugin functionality.

        Args:
            **kwargs: Plugin-specific arguments

        Returns:
            Execution result
        """
        pass

    def initialize(self) -> bool:
        """Initialize plugin after loading.

        Returns:
            True if successful
        """
        logger.info(f"Plugin {self.config.name}: initializing")
        self.state = PluginState.INITIALIZED
        return True

    def validate(self) -> bool:
        """Validate plugin integrity.

        Returns:
            True if valid
        """
        if not self.config.enabled:
            logger.warning(f"Plugin {self.config.name}: disabled")
            return False

        return True

    def get_info(self) -> Dict[str, Any]:
        """Get plugin information.

        Returns:
            Plugin info dictionary
        """
        return {
            "id": self.id,
            "name": self.config.name,
            "version": self.config.version,
            "type": self.config.plugin_type.value,
            "state": self.state.value,
            "description": self.config.description,
            "author": self.config.author,
            "enabled": self.config.enabled,
            "created_at": self.created_at.isoformat(),
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
        }


class ToolPlugin(Plugin):
    """Plugin that provides tool functionality."""

    def __init__(self, config: PluginConfig):
        """Initialize tool plugin.

        Args:
            config: Plugin configuration
        """
        super().__init__(config)
        self.tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable):
        """Register a tool.

        Args:
            name: Tool name
            func: Tool function
        """
        self.tools[name] = func
        logger.debug(f"Tool registered: {name}")

    def load(self) -> bool:
        """Load plugin."""
        if not self.validate():
            return False

        self.state = PluginState.LOADING
        logger.info(f"Plugin {self.config.name}: loading")

        try:
            self._load_tools()
            self.state = PluginState.LOADED
            self.loaded_at = datetime.now()
            logger.info(f"Plugin {self.config.name}: loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Plugin {self.config.name}: load failed - {str(e)}")
            self.state = PluginState.FAILED
            return False

    def unload(self) -> bool:
        """Unload plugin."""
        self.state = PluginState.UNLOADING
        logger.info(f"Plugin {self.config.name}: unloading")

        try:
            self.tools.clear()
            self.state = PluginState.UNLOADED
            logger.info(f"Plugin {self.config.name}: unloaded")
            return True
        except Exception as e:
            logger.error(f"Plugin {self.config.name}: unload failed - {str(e)}")
            return False

    def _load_tools(self):
        """Load tools (override in subclass)."""
        pass

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments

        Returns:
            Execution result
        """
        if tool_name not in self.tools:
            return {"error": f"Tool not found: {tool_name}"}

        try:
            result = self.tools[tool_name](**kwargs)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class ExtensionLoader:
    """Loads and manages dynamic extensions."""

    def __init__(self):
        """Initialize extension loader."""
        self.extensions: Dict[str, Any] = {}
        self.loaded_modules: Dict[str, Any] = {}
        logger.info("ExtensionLoader initialized")

    def load_extension(self, config: ExtensionConfig) -> bool:
        """Load an extension from entry point.

        Args:
            config: Extension configuration

        Returns:
            True if successful
        """
        logger.info(f"Loading extension: {config.name}")

        try:
            # Parse entry point
            module_path, class_name = self._parse_entry_point(config.entry_point)

            # Import module
            module = importlib.import_module(module_path)
            self.loaded_modules[config.name] = module

            # Get class from module
            if class_name and hasattr(module, class_name):
                extension_class = getattr(module, class_name)
                extension = extension_class(**config.config)
                self.extensions[config.name] = extension
                logger.info(f"Extension loaded: {config.name}")
                return True
            else:
                logger.error(f"Extension class not found: {class_name}")
                return False

        except ImportError as e:
            logger.error(f"Failed to import extension {config.name}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to load extension {config.name}: {str(e)}")
            return False

    def get_extension(self, name: str) -> Optional[Any]:
        """Get loaded extension.

        Args:
            name: Extension name

        Returns:
            Extension instance or None
        """
        return self.extensions.get(name)

    def unload_extension(self, name: str) -> bool:
        """Unload an extension.

        Args:
            name: Extension name

        Returns:
            True if successful
        """
        if name not in self.extensions:
            logger.warning(f"Extension not loaded: {name}")
            return False

        try:
            del self.extensions[name]
            logger.info(f"Extension unloaded: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload extension {name}: {str(e)}")
            return False

    def list_extensions(self) -> List[str]:
        """List all loaded extensions.

        Returns:
            List of extension names
        """
        return list(self.extensions.keys())

    def _parse_entry_point(self, entry_point: str) -> tuple:
        """Parse entry point string.

        Args:
            entry_point: Entry point in format "module.path:ClassName"

        Returns:
            Tuple of (module_path, class_name)
        """
        if ":" in entry_point:
            module_path, class_name = entry_point.split(":", 1)
        else:
            module_path = entry_point
            class_name = None

        return module_path, class_name


class PluginManager:
    """Manages plugin lifecycle and execution.

    Handles plugin loading, unloading, initialization, and execution.
    """

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_order: List[str] = []
        self.extension_loader = ExtensionLoader()
        logger.info("PluginManager initialized")

    def register_plugin(self, plugin: Plugin) -> bool:
        """Register a plugin.

        Args:
            plugin: Plugin to register

        Returns:
            True if successful
        """
        if plugin.config.name in self.plugins:
            logger.warning(f"Plugin already registered: {plugin.config.name}")
            return False

        self.plugins[plugin.config.name] = plugin
        self.plugin_order.append(plugin.config.name)
        logger.info(f"Plugin registered: {plugin.config.name}")
        return True

    def load_plugin(self, name: str) -> bool:
        """Load a registered plugin.

        Args:
            name: Plugin name

        Returns:
            True if successful
        """
        if name not in self.plugins:
            logger.error(f"Plugin not registered: {name}")
            return False

        plugin = self.plugins[name]

        try:
            if plugin.load():
                plugin.initialize()
                logger.info(f"Plugin loaded and initialized: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Plugin loading failed: {str(e)}")
            return False

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if successful
        """
        if name not in self.plugins:
            logger.error(f"Plugin not registered: {name}")
            return False

        plugin = self.plugins[name]

        try:
            if plugin.unload():
                logger.info(f"Plugin unloaded: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Plugin unloading failed: {str(e)}")
            return False

    def load_all_plugins(self) -> int:
        """Load all registered plugins.

        Returns:
            Number of successfully loaded plugins
        """
        loaded_count = 0
        for name in self.plugin_order:
            if self.load_plugin(name):
                loaded_count += 1

        logger.info(f"Loaded {loaded_count}/{len(self.plugin_order)} plugins")
        return loaded_count

    def execute_plugin(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a plugin.

        Args:
            name: Plugin name
            **kwargs: Plugin arguments

        Returns:
            Execution result
        """
        if name not in self.plugins:
            return {"error": f"Plugin not found: {name}"}

        plugin = self.plugins[name]

        if plugin.state != PluginState.INITIALIZED:
            return {"error": f"Plugin not initialized: {name}"}

        try:
            plugin.state = PluginState.RUNNING
            result = plugin.execute(**kwargs)
            plugin.state = PluginState.INITIALIZED
            return result
        except Exception as e:
            plugin.state = PluginState.FAILED
            logger.error(f"Plugin execution failed: {str(e)}")
            return {"error": str(e)}

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin instance.

        Args:
            name: Plugin name

        Returns:
            Plugin or None
        """
        return self.plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins.

        Returns:
            List of plugin info
        """
        return [plugin.get_info() for plugin in self.plugins.values()]

    def get_plugin_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get plugin status.

        Args:
            name: Plugin name

        Returns:
            Plugin info or None
        """
        plugin = self.get_plugin(name)
        return plugin.get_info() if plugin else None


class AdvancedToolkit:
    """Advanced toolkit combining plugins, extensions, and tools.

    Provides unified interface for plugin and extension management.
    """

    def __init__(self, config: ToolkitConfig):
        """Initialize advanced toolkit.

        Args:
            config: Toolkit configuration
        """
        self.config = config
        self.id = str(uuid.uuid4())
        self.plugin_manager = PluginManager()
        self.created_at = datetime.now()
        logger.info(f"AdvancedToolkit initialized: {config.name}")

        self._initialize_plugins()
        self._initialize_extensions()

    def _initialize_plugins(self):
        """Initialize plugins from configuration."""
        for plugin_config in self.config.plugins:
            plugin = ToolPlugin(plugin_config)
            self.plugin_manager.register_plugin(plugin)

    def _initialize_extensions(self):
        """Initialize extensions from configuration."""
        for ext_config in self.config.extensions:
            if ext_config.auto_load:
                self.plugin_manager.extension_loader.load_extension(ext_config)

    def load_all(self) -> Dict[str, int]:
        """Load all plugins and extensions.

        Returns:
            Dictionary with load statistics
        """
        logger.info(f"Loading toolkit: {self.config.name}")

        plugins_loaded = self.plugin_manager.load_all_plugins()
        extensions_loaded = len(self.plugin_manager.extension_loader.list_extensions())

        logger.info(
            f"Toolkit loaded: {plugins_loaded} plugins, {extensions_loaded} extensions"
        )

        return {
            "plugins_loaded": plugins_loaded,
            "extensions_loaded": extensions_loaded,
        }

    def load_plugin(self, name: str) -> bool:
        """Load a specific plugin.

        Args:
            name: Plugin name

        Returns:
            True if successful
        """
        return self.plugin_manager.load_plugin(name)

    def execute_plugin(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a plugin.

        Args:
            name: Plugin name
            **kwargs: Plugin arguments

        Returns:
            Execution result
        """
        return self.plugin_manager.execute_plugin(name, **kwargs)

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin instance.

        Args:
            name: Plugin name

        Returns:
            Plugin or None
        """
        return self.plugin_manager.get_plugin(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins.

        Returns:
            List of plugin info
        """
        return self.plugin_manager.list_plugins()

    def list_extensions(self) -> List[str]:
        """List all extensions.

        Returns:
            List of extension names
        """
        return self.plugin_manager.extension_loader.list_extensions()

    def get_info(self) -> Dict[str, Any]:
        """Get toolkit information.

        Returns:
            Toolkit info
        """
        return {
            "id": self.id,
            "name": self.config.name,
            "description": self.config.description,
            "plugins": len(self.config.plugins),
            "extensions": len(self.config.extensions),
            "created_at": self.created_at.isoformat(),
        }
