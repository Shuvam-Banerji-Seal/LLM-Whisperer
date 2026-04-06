"""
LangChain Tools Integration Module

This module provides comprehensive tool creation, management, and integration
patterns for LangChain agents. Includes tool definitions, registries, validation,
and dynamic loading capabilities.

Author: Shuvam Banerji Seal
Date: 2026-04-06

Source: https://python.langchain.com/api_reference
Documentation: https://python.langchain.com/docs/how_to/custom_tools
"""

import logging
import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categorization for organization and filtering."""

    UTILITY = "utility"
    DATA_RETRIEVAL = "data_retrieval"
    COMPUTATION = "computation"
    INTEGRATION = "integration"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    DATABASE = "database"
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    """Represents a tool parameter with type and description.

    Attributes:
        name: Parameter name
        type: Parameter type (str, int, float, bool, list, dict)
        description: Human-readable parameter description
        required: Whether parameter is required
        default: Default value if not required
    """

    name: str
    type: Union[str, Type]
    description: str
    required: bool = True
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter to dictionary format.

        Returns:
            Dictionary representation of parameter
        """
        type_str = self.type if isinstance(self.type, str) else self.type.__name__
        return {
            "name": self.name,
            "type": type_str,
            "description": self.description,
            "required": self.required,
            "default": self.default,
        }


@dataclass
class ToolSchema:
    """Schema definition for a tool with full metadata.

    Attributes:
        name: Unique tool identifier
        description: Tool description
        category: Tool categorization
        parameters: List of ToolParameter instances
        return_type: Return value type
        return_description: Description of return value
        version: Tool version
        author: Tool author
        tags: List of tags for categorization
    """

    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    return_type: str = "str"
    return_description: str = "Tool result"
    version: str = "1.0.0"
    author: str = "Unknown"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary format.

        Returns:
            Dictionary representation of schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "return_description": self.return_description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
        }

    def to_json(self) -> str:
        """Convert schema to JSON string.

        Returns:
            JSON representation of schema
        """
        return json.dumps(self.to_dict(), indent=2)


class BaseTool(ABC):
    """Abstract base class for all tools.

    This class defines the interface that all tools must implement.
    """

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result
        """
        pass

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get the tool's schema definition.

        Returns:
            ToolSchema instance
        """
        pass


class FunctionTool(BaseTool):
    """Wrapper for Python functions as tools.

    This class wraps Python functions and makes them usable as LangChain tools
    with automatic schema generation from function signatures.

    Example:
        >>> def add_numbers(a: int, b: int) -> int:
        ...     \"\"\"Add two numbers.\"\"\"
        ...     return a + b
        >>>
        >>> tool = FunctionTool(add_numbers, category=ToolCategory.COMPUTATION)
        >>> result = tool.execute(a=5, b=3)
        >>> print(result)  # Output: 8
    """

    def __init__(
        self,
        func: Callable,
        category: ToolCategory = ToolCategory.UTILITY,
        author: str = "Unknown",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Initialize function tool wrapper.

        Args:
            func: Python function to wrap
            category: Tool category
            author: Tool author name
            version: Tool version
            tags: List of tags for categorization

        Raises:
            TypeError: If func is not callable
        """
        if not callable(func):
            raise TypeError(f"Expected callable, got {type(func)}")

        self.func = func
        self.category = category
        self.author = author
        self.version = version
        self.tags = tags or []
        self._schema = self._generate_schema()

        logger.info(f"FunctionTool created for: {func.__name__}")

    def _generate_schema(self) -> ToolSchema:
        """Automatically generate schema from function signature.

        Returns:
            Generated ToolSchema instance
        """
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, str)
            param_type_str = (
                param_type.__name__
                if hasattr(param_type, "__name__")
                else str(param_type)
            )

            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type_str,
                    description=f"Parameter: {param_name}",
                    required=required,
                    default=default,
                )
            )

        return_type = type_hints.get("return", str)
        return_type_str = (
            return_type.__name__
            if hasattr(return_type, "__name__")
            else str(return_type)
        )

        return ToolSchema(
            name=self.func.__name__,
            description=self.func.__doc__ or f"Tool: {self.func.__name__}",
            category=self.category,
            parameters=parameters,
            return_type=return_type_str,
            return_description=f"Result from {self.func.__name__}",
            version=self.version,
            author=self.author,
            tags=self.tags,
        )

    def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped function.

        Args:
            **kwargs: Function parameters

        Returns:
            Function return value

        Raises:
            TypeError: If required parameters are missing
            Exception: If function execution fails
        """
        try:
            # Validate parameters
            sig = inspect.signature(self.func)
            bound_args = sig.bind_partial(**kwargs)

            logger.debug(f"Executing {self.func.__name__} with args: {kwargs}")
            result = self.func(**kwargs)
            logger.debug(f"Execution successful")
            return result
        except TypeError as e:
            logger.error(f"Invalid parameters for {self.func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing {self.func.__name__}: {e}")
            raise

    def get_schema(self) -> ToolSchema:
        """Get tool schema.

        Returns:
            ToolSchema instance
        """
        return self._schema


class ToolRegistry:
    """Central registry for managing tools.

    This class maintains a registry of available tools, enabling
    discovery, validation, and dynamic loading.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(tool)
        >>> retrieved_tool = registry.get_tool("tool_name")
        >>> all_tools = registry.get_all_tools()
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            cat: [] for cat in ToolCategory
        }
        logger.info("ToolRegistry initialized")

    def register(self, tool: BaseTool, override: bool = False) -> None:
        """Register a tool in the registry.

        Args:
            tool: BaseTool instance to register
            override: Allow overriding existing tool

        Raises:
            ValueError: If tool already exists and override=False
        """
        schema = tool.get_schema()
        tool_name = schema.name

        if tool_name in self._tools and not override:
            raise ValueError(
                f"Tool '{tool_name}' already registered. Set override=True to replace."
            )

        self._tools[tool_name] = tool
        self._categories[schema.category].append(tool_name)
        logger.info(f"Tool registered: {tool_name} (Category: {schema.category.value})")

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool from the registry.

        Args:
            tool_name: Name of tool to unregister

        Raises:
            KeyError: If tool not found
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' not found in registry")

        tool = self._tools.pop(tool_name)
        schema = tool.get_schema()
        self._categories[schema.category].remove(tool_name)
        logger.info(f"Tool unregistered: {tool_name}")

    def get_tool(self, tool_name: str) -> BaseTool:
        """Retrieve a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            BaseTool instance

        Raises:
            KeyError: If tool not found
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' not found in registry")
        return self._tools[tool_name]

    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a specific category.

        Args:
            category: ToolCategory to filter by

        Returns:
            List of tools in the category
        """
        tool_names = self._categories[category]
        return [self._tools[name] for name in tool_names]

    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools.

        Returns:
            Dictionary of all tools
        """
        return self._tools.copy()

    def search_tools(
        self, query: str, search_tags: bool = True, search_description: bool = True
    ) -> List[BaseTool]:
        """Search tools by name, tags, or description.

        Args:
            query: Search query string
            search_tags: Include tags in search
            search_description: Include descriptions in search

        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        matching_tools = []

        for tool_name, tool in self._tools.items():
            schema = tool.get_schema()

            # Check name
            if query_lower in tool_name.lower():
                matching_tools.append(tool)
                continue

            # Check tags
            if search_tags and any(query_lower in tag.lower() for tag in schema.tags):
                matching_tools.append(tool)
                continue

            # Check description
            if search_description and query_lower in schema.description.lower():
                matching_tools.append(tool)
                continue

        logger.debug(f"Found {len(matching_tools)} tools matching '{query}'")
        return matching_tools

    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for all registered tools.

        Returns:
            Dictionary of tool schemas
        """
        return {name: tool.get_schema().to_dict() for name, tool in self._tools.items()}

    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about the registry.

        Returns:
            Registry information dictionary
        """
        return {
            "total_tools": len(self._tools),
            "categories": {
                cat.value: len(tools) for cat, tools in self._categories.items()
            },
            "tools": list(self._tools.keys()),
        }


class ToolValidator:
    """Validates tool parameters and execution.

    This class ensures that tools are called with valid parameters
    and handles type checking and validation.
    """

    @staticmethod
    def validate_parameters(schema: ToolSchema, **kwargs: Any) -> Dict[str, Any]:
        """Validate parameters against schema.

        Args:
            schema: ToolSchema to validate against
            **kwargs: Parameters to validate

        Returns:
            Validated parameters dictionary

        Raises:
            ValueError: If validation fails
        """
        validated = {}
        provided_params = set(kwargs.keys())

        for param in schema.parameters:
            if param.name not in kwargs:
                if param.required:
                    raise ValueError(f"Required parameter missing: {param.name}")
                validated[param.name] = param.default
            else:
                value = kwargs[param.name]
                validated[param.name] = ToolValidator._validate_type(
                    param.name, value, param.type
                )

        # Check for extra parameters
        extra_params = provided_params - {p.name for p in schema.parameters}
        if extra_params:
            logger.warning(f"Extra parameters provided: {extra_params}")

        return validated

    @staticmethod
    def _validate_type(name: str, value: Any, expected_type: Union[str, Type]) -> Any:
        """Validate a single parameter's type.

        Args:
            name: Parameter name
            value: Parameter value
            expected_type: Expected type

        Returns:
            Validated value

        Raises:
            TypeError: If type validation fails
        """
        type_str = (
            expected_type if isinstance(expected_type, str) else expected_type.__name__
        )

        # Type coercion for basic types
        if type_str == "int":
            try:
                return int(value)
            except (ValueError, TypeError):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")
        elif type_str == "float":
            try:
                return float(value)
            except (ValueError, TypeError):
                raise TypeError(f"{name} must be float, got {type(value).__name__}")
        elif type_str == "str":
            return str(value)
        elif type_str == "bool":
            if isinstance(value, bool):
                return value
            raise TypeError(f"{name} must be bool, got {type(value).__name__}")

        return value


class DynamicToolLoader:
    """Load tools dynamically from modules or configurations.

    This class enables runtime loading of tools from Python modules
    or configuration files.

    Example:
        >>> loader = DynamicToolLoader()
        >>> loader.load_from_module("my_tools_module")
        >>> loader.load_from_file("tools_config.json")
    """

    def __init__(self, registry: ToolRegistry) -> None:
        """Initialize loader with target registry.

        Args:
            registry: ToolRegistry to load tools into
        """
        self.registry = registry
        logger.info("DynamicToolLoader initialized")

    def load_from_module(self, module_name: str) -> int:
        """Load tools from a Python module.

        Looks for BaseTool subclasses in the module.

        Args:
            module_name: Module name to load from

        Returns:
            Number of tools loaded

        Raises:
            ImportError: If module cannot be imported
        """
        try:
            module = importlib.import_module(module_name)
            logger.info(f"Loading tools from module: {module_name}")

            count = 0
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Check if it's a Tool class (not the base class itself)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseTool)
                    and attr is not BaseTool
                ):
                    try:
                        tool_instance = attr()
                        self.registry.register(tool_instance)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to instantiate {attr_name}: {e}")

            logger.info(f"Loaded {count} tools from {module_name}")
            return count
        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            raise

    def load_from_function(self, func: Callable, **kwargs: Any) -> None:
        """Load a Python function as a tool.

        Args:
            func: Function to load as tool
            **kwargs: Additional arguments for FunctionTool
        """
        tool = FunctionTool(func, **kwargs)
        self.registry.register(tool)
        logger.info(f"Loaded function as tool: {func.__name__}")


if __name__ == "__main__":
    print("LangChain Tools Integration Module")
    print("=" * 50)

    # Create a registry
    registry = ToolRegistry()

    # Example: Create and register a simple tool
    def calculate_sum(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    tool = FunctionTool(
        calculate_sum, category=ToolCategory.COMPUTATION, tags=["math", "arithmetic"]
    )

    registry.register(tool)
    print(f"Registered tool: {tool.get_schema().name}")
    print(f"Registry info: {registry.get_registry_info()}")
