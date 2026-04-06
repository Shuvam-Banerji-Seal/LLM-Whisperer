# Tool Integration Guide for LLM Agents

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Version:** 1.0

## Table of Contents

1. [Introduction](#introduction)
2. [Tool Definition and Specification](#tool-definition-and-specification)
3. [Building Custom Tools](#building-custom-tools)
4. [Tool Registration and Discovery](#tool-registration-and-discovery)
5. [Error Handling in Tools](#error-handling-in-tools)
6. [Tool Chaining and Composition](#tool-chaining-and-composition)
7. [AGNO Framework Integration](#agno-framework-integration)
8. [Langchain Framework Integration](#langchain-framework-integration)
9. [Best Practices](#best-practices)
10. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Introduction

Tools are fundamental components in agent-based systems. They enable Large Language Models (LLMs) to interact with external systems, APIs, databases, and perform computations beyond their training data. This guide covers comprehensive tool integration patterns for both AGNO and Langchain frameworks.

### Key Concepts

- **Tool**: A function or service that an agent can invoke with specific inputs
- **Tool Specification**: Defines the interface, parameters, and expected outputs
- **Tool Registry**: Central location where tools are registered and discovered
- **Tool Chaining**: Sequential or parallel execution of multiple tools
- **Tool Composition**: Combining tools to create more complex workflows

---

## Tool Definition and Specification

### Tool Specification Components

A well-defined tool must include:

1. **Name**: Unique identifier for the tool
2. **Description**: Clear explanation of what the tool does
3. **Input Schema**: Specification of expected input parameters
4. **Output Schema**: Expected output format and type
5. **Error Handling**: How errors are managed and reported
6. **Authentication**: Required credentials or API keys
7. **Rate Limits**: Usage constraints and throttling rules

### Basic Tool Specification Structure

```python
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ParameterType(Enum):
    """Enum for parameter types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"

@dataclass
class ToolParameter:
    """Represents a single parameter in a tool."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum_values: Optional[list] = None
    constraints: Optional[Dict[str, Any]] = None

@dataclass
class ToolSpecification:
    """Formal specification of a tool."""
    name: str
    description: str
    version: str
    parameters: list[ToolParameter]
    output_schema: Dict[str, Any]
    authentication_required: bool = False
    rate_limit_per_minute: Optional[int] = None
    timeout_seconds: int = 30
    tags: Optional[list[str]] = None

# Example Tool Specification
weather_tool_spec = ToolSpecification(
    name="get_weather",
    description="Retrieves current weather information for a given location",
    version="1.0.0",
    parameters=[
        ToolParameter(
            name="location",
            type=ParameterType.STRING,
            description="City name or coordinates (latitude,longitude)",
            required=True
        ),
        ToolParameter(
            name="units",
            type=ParameterType.STRING,
            description="Temperature units: 'celsius' or 'fahrenheit'",
            required=False,
            default="celsius",
            enum_values=["celsius", "fahrenheit"]
        )
    ],
    output_schema={
        "type": "object",
        "properties": {
            "temperature": {"type": "number"},
            "condition": {"type": "string"},
            "humidity": {"type": "number"},
            "wind_speed": {"type": "number"}
        }
    },
    authentication_required=True,
    rate_limit_per_minute=60,
    tags=["weather", "external-api"]
)
```

### Tool Input/Output Validation

```python
import json
from typing import Any
from jsonschema import validate, ValidationError

class ToolValidator:
    """Validates tool inputs and outputs against schemas."""
    
    @staticmethod
    def validate_input(parameters: Dict[str, Any], 
                       param_specs: list[ToolParameter]) -> tuple[bool, str]:
        """Validate tool input parameters."""
        try:
            # Check required parameters
            required_params = {p.name for p in param_specs if p.required}
            provided_params = set(parameters.keys())
            
            missing = required_params - provided_params
            if missing:
                return False, f"Missing required parameters: {missing}"
            
            # Validate types
            for param_name, param_value in parameters.items():
                param_spec = next((p for p in param_specs if p.name == param_name), None)
                if not param_spec:
                    return False, f"Unknown parameter: {param_name}"
                
                if not ToolValidator._validate_type(param_value, param_spec.type):
                    return False, f"Invalid type for {param_name}: expected {param_spec.type.value}"
            
            return True, ""
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def _validate_type(value: Any, param_type: ParameterType) -> bool:
        """Validate a single value against its type."""
        type_mapping = {
            ParameterType.STRING: str,
            ParameterType.INTEGER: int,
            ParameterType.FLOAT: float,
            ParameterType.BOOLEAN: bool,
            ParameterType.ARRAY: list,
            ParameterType.OBJECT: dict
        }
        
        expected_type = type_mapping.get(param_type)
        return isinstance(value, expected_type) if expected_type else False
```

---

## Building Custom Tools

### Generic Custom Tool Pattern

```python
from abc import ABC, abstractmethod
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.last_called: Optional[datetime] = None
        self.call_count = 0
        self.execution_times: list[float] = []
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass
    
    async def execute_async(self, **kwargs) -> Dict[str, Any]:
        """Asynchronous execution of the tool."""
        return self.execute(**kwargs)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata and statistics."""
        avg_exec_time = sum(self.execution_times) / len(self.execution_times) \
                       if self.execution_times else 0
        
        return {
            "name": self.name,
            "description": self.description,
            "call_count": self.call_count,
            "last_called": self.last_called.isoformat() if self.last_called else None,
            "average_execution_time": avg_exec_time
        }

# Example Implementation: Database Query Tool
class DatabaseQueryTool(BaseTool):
    """Tool for executing database queries."""
    
    def __init__(self, connection_string: str):
        super().__init__(
            name="database_query",
            description="Execute SQL queries against a database"
        )
        self.connection_string = connection_string
    
    def execute(self, query: str, max_rows: int = 100) -> Dict[str, Any]:
        """Execute a database query safely."""
        import time
        start_time = time.time()
        
        try:
            # In production, use proper database drivers
            logger.info(f"Executing query: {query[:100]}...")
            
            # Simulate execution
            result = {
                "rows": [],
                "row_count": 0,
                "status": "success"
            }
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self.last_called = datetime.now()
            self.call_count += 1
            
            return result
        
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }

# Example Implementation: API Call Tool
class APICallTool(BaseTool):
    """Tool for making HTTP API calls."""
    
    def __init__(self, timeout: int = 30):
        super().__init__(
            name="api_call",
            description="Make HTTP requests to external APIs"
        )
        self.timeout = timeout
    
    def execute(self, url: str, method: str = "GET", 
                headers: Optional[Dict] = None,
                body: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an HTTP request to an external API."""
        import time
        import requests
        
        start_time = time.time()
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers or {},
                json=body,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self.last_called = datetime.now()
            self.call_count += 1
            
            return {
                "status": "success",
                "status_code": response.status_code,
                "body": response.json() if response.headers.get('content-type') == 'application/json' else response.text,
                "headers": dict(response.headers),
                "execution_time_ms": execution_time * 1000
            }
        
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "url": url,
                "method": method
            }

# Example Implementation: File Processing Tool
class FileProcessingTool(BaseTool):
    """Tool for reading and processing files."""
    
    def __init__(self, allowed_extensions: Optional[list[str]] = None):
        super().__init__(
            name="file_processor",
            description="Read and process files with various formats"
        )
        self.allowed_extensions = allowed_extensions or ['.txt', '.json', '.csv', '.md']
    
    def execute(self, file_path: str, operation: str = "read") -> Dict[str, Any]:
        """Process a file based on the specified operation."""
        import time
        import os
        
        start_time = time.time()
        
        try:
            # Validate file extension
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in self.allowed_extensions:
                raise ValueError(f"File type {ext} not allowed")
            
            if operation == "read":
                with open(file_path, 'r') as f:
                    content = f.read()
                
                result = {
                    "status": "success",
                    "operation": operation,
                    "file_path": file_path,
                    "file_size": len(content),
                    "content": content[:1000],  # First 1000 chars
                    "truncated": len(content) > 1000
                }
            else:
                return {"status": "error", "error": f"Unknown operation: {operation}"}
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self.last_called = datetime.now()
            self.call_count += 1
            
            return result
        
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path,
                "operation": operation
            }
```

---

## Tool Registration and Discovery

### Tool Registry Pattern

```python
from typing import Optional, Callable
import json

class ToolRegistry:
    """Central registry for managing and discovering tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_specs: Dict[str, ToolSpecification] = {}
        self._aliases: Dict[str, str] = {}
        self._categories: Dict[str, list[str]] = {}
    
    def register(self, tool: BaseTool, 
                 spec: ToolSpecification,
                 category: Optional[str] = None,
                 aliases: Optional[list[str]] = None) -> None:
        """Register a tool with its specification."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        self._tool_specs[tool.name] = spec
        
        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = tool.name
        
        # Register category
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(tool.name)
        
        logging.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name or alias."""
        # Check direct name first
        if name in self._tools:
            return self._tools[name]
        
        # Check aliases
        if name in self._aliases:
            return self._tools[self._aliases[name]]
        
        return None
    
    def get_specification(self, name: str) -> Optional[ToolSpecification]:
        """Get the specification for a tool."""
        if name in self._tool_specs:
            return self._tool_specs[name]
        
        if name in self._aliases:
            return self._tool_specs[self._aliases[name]]
        
        return None
    
    def list_tools(self, category: Optional[str] = None) -> list[str]:
        """List available tools, optionally filtered by category."""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get complete information about a tool."""
        tool = self.get_tool(name)
        spec = self.get_specification(name)
        
        if not tool:
            return {"error": f"Tool '{name}' not found"}
        
        return {
            "name": tool.name,
            "description": tool.description,
            "specification": {
                "version": spec.version if spec else None,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type.value,
                        "description": p.description,
                        "required": p.required
                    }
                    for p in (spec.parameters if spec else [])
                ],
                "output_schema": spec.output_schema if spec else None
            },
            "metadata": tool.get_metadata()
        }
    
    def export_to_json(self, tool_name: str) -> str:
        """Export tool specification to JSON format."""
        info = self.get_tool_info(tool_name)
        return json.dumps(info, indent=2)

# Example Usage
registry = ToolRegistry()

# Register database tool
db_tool = DatabaseQueryTool("postgresql://localhost/mydb")
db_spec = ToolSpecification(
    name="database_query",
    description="Execute SQL queries against PostgreSQL database",
    version="1.0.0",
    parameters=[
        ToolParameter("query", ParameterType.STRING, "SQL query to execute", required=True),
        ToolParameter("max_rows", ParameterType.INTEGER, "Maximum rows to return", required=False, default=100)
    ],
    output_schema={
        "type": "object",
        "properties": {
            "rows": {"type": "array"},
            "row_count": {"type": "integer"},
            "status": {"type": "string"}
        }
    }
)
registry.register(db_tool, db_spec, category="database", aliases=["db_query", "sql"])

# Register API tool
api_tool = APICallTool(timeout=30)
api_spec = ToolSpecification(
    name="api_call",
    description="Make HTTP requests to external APIs",
    version="1.0.0",
    parameters=[
        ToolParameter("url", ParameterType.STRING, "API endpoint URL", required=True),
        ToolParameter("method", ParameterType.STRING, "HTTP method", required=False, default="GET"),
        ToolParameter("headers", ParameterType.OBJECT, "HTTP headers", required=False),
        ToolParameter("body", ParameterType.OBJECT, "Request body for POST/PUT", required=False)
    ],
    output_schema={
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "status_code": {"type": "integer"},
            "body": {},
            "execution_time_ms": {"type": "number"}
        }
    }
)
registry.register(api_tool, api_spec, category="external", aliases=["http", "rest"])
```

---

## Error Handling in Tools

### Comprehensive Error Handling Strategy

```python
from enum import Enum
from typing import Union
import traceback

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ToolError(Exception):
    """Base exception for tool-related errors."""
    
    def __init__(self, message: str, error_code: str, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        super().__init__(self.message)

class ValidationError(ToolError):
    """Raised when tool inputs fail validation."""
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, "VALIDATION_ERROR", ErrorSeverity.LOW, context)

class ExecutionError(ToolError):
    """Raised when tool execution fails."""
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, "EXECUTION_ERROR", ErrorSeverity.HIGH, context)

class TimeoutError(ToolError):
    """Raised when tool execution times out."""
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, "TIMEOUT_ERROR", ErrorSeverity.MEDIUM, context)

class RateLimitError(ToolError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: int, context: Optional[Dict] = None):
        super().__init__(message, "RATE_LIMIT_ERROR", ErrorSeverity.MEDIUM, context)
        self.retry_after = retry_after

class ErrorHandler:
    """Centralized error handling for tools."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_log: list[Dict[str, Any]] = []
    
    def handle_error(self, error: Exception, tool_name: str, 
                     operation: str) -> Dict[str, Any]:
        """Handle and log an error."""
        import time
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc() if isinstance(error, Exception) else None
        }
        
        if isinstance(error, ToolError):
            error_info.update({
                "error_code": error.error_code,
                "severity": error.severity.value,
                "context": error.context
            })
        
        self.error_log.append(error_info)
        logger.error(f"Tool error in {tool_name}.{operation}: {error_info}")
        
        return error_info
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried."""
        # Don't retry validation errors or rate limits (use retry_after)
        if isinstance(error, (ValidationError, RateLimitError)):
            return False
        
        # Retry execution and timeout errors up to max_retries
        if isinstance(error, (ExecutionError, TimeoutError)):
            return attempt < self.max_retries
        
        return False
    
    def get_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry using exponential backoff."""
        return self.backoff_factor ** attempt

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of logged errors."""
        by_type = {}
        by_severity = {}
        by_tool = {}
        
        for error in self.error_log:
            error_type = error['error_type']
            by_type[error_type] = by_type.get(error_type, 0) + 1
            
            if 'severity' in error:
                severity = error['severity']
                by_severity[severity] = by_severity.get(severity, 0) + 1
            
            tool = error['tool_name']
            by_tool[tool] = by_tool.get(tool, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_tool": by_tool
        }

# Resilient Tool Wrapper
class ResilientTool:
    """Wraps a tool with error handling and retry logic."""
    
    def __init__(self, tool: BaseTool, error_handler: ErrorHandler):
        self.tool = tool
        self.error_handler = error_handler
    
    def execute_with_retry(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with automatic retry on failure."""
        import time
        
        attempt = 0
        last_error = None
        
        while attempt < self.error_handler.max_retries:
            try:
                return self.tool.execute(**kwargs)
            except Exception as e:
                last_error = e
                self.error_handler.handle_error(e, self.tool.name, "execute")
                
                if not self.error_handler.should_retry(e, attempt):
                    break
                
                delay = self.error_handler.get_retry_delay(attempt)
                logger.warning(f"Retry attempt {attempt + 1} after {delay}s for {self.tool.name}")
                time.sleep(delay)
                attempt += 1
        
        # Return error result
        return {
            "status": "error",
            "error": str(last_error),
            "attempts": attempt,
            "tool": self.tool.name
        }
```

---

## Tool Chaining and Composition

### Tool Chaining Patterns

```python
from typing import List, Callable
from enum import Enum

class ChainType(Enum):
    """Types of tool chains."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"

class ToolChain:
    """Manages execution of chained tools."""
    
    def __init__(self, name: str, chain_type: ChainType = ChainType.SEQUENTIAL):
        self.name = name
        self.chain_type = chain_type
        self.steps: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
    
    def add_step(self, tool_name: str, 
                 input_mapping: Optional[Dict[str, str]] = None,
                 condition: Optional[Callable[[Dict], bool]] = None) -> 'ToolChain':
        """Add a tool to the chain."""
        step = {
            "tool_name": tool_name,
            "input_mapping": input_mapping or {},
            "condition": condition
        }
        self.steps.append(step)
        return self
    
    def execute(self, registry: ToolRegistry, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool chain."""
        self.context = initial_input.copy()
        results = []
        
        for i, step in enumerate(self.steps):
            # Check condition
            if step["condition"] and not step["condition"](self.context):
                logger.info(f"Skipping step {i} due to condition")
                continue
            
            # Get tool
            tool = registry.get_tool(step["tool_name"])
            if not tool:
                return {
                    "status": "error",
                    "error": f"Tool '{step['tool_name']}' not found",
                    "step": i
                }
            
            # Map inputs from context
            inputs = self._map_inputs(step["input_mapping"])
            
            # Execute tool
            result = tool.execute(**inputs)
            results.append({
                "step": i,
                "tool": step["tool_name"],
                "result": result
            })
            
            # Update context with result
            self.context.update(result)
        
        return {
            "status": "success",
            "chain": self.name,
            "steps_executed": len(results),
            "results": results,
            "final_context": self.context
        }
    
    def _map_inputs(self, mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map context values to tool inputs based on mapping."""
        inputs = {}
        for param_name, context_key in mapping.items():
            if context_key in self.context:
                inputs[param_name] = self.context[context_key]
        return inputs

# Example: Multi-step data processing chain
def build_data_pipeline(registry: ToolRegistry) -> ToolChain:
    """Build a chain to process data through multiple tools."""
    chain = ToolChain("data_pipeline", ChainType.SEQUENTIAL)
    
    # Step 1: Fetch data from API
    chain.add_step(
        "api_call",
        input_mapping={
            "url": "api_endpoint",
            "method": "http_method"
        }
    )
    
    # Step 2: Parse and validate response
    chain.add_step(
        "file_processor",
        input_mapping={
            "content": "body"  # Use response body as input
        }
    )
    
    # Step 3: Store in database (conditional)
    chain.add_step(
        "database_query",
        input_mapping={
            "query": "insert_query"
        },
        condition=lambda ctx: ctx.get("status") == "success"
    )
    
    return chain

class ToolComposition:
    """Combines multiple tools into a composite tool."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.sub_tools: List[BaseTool] = []
        self.execution_plan: Optional[Callable] = None
    
    def add_tool(self, tool: BaseTool) -> 'ToolComposition':
        """Add a tool to the composition."""
        self.sub_tools.append(tool)
        return self
    
    def set_execution_plan(self, plan: Callable[[List[BaseTool], Dict], Dict]) -> 'ToolComposition':
        """Set a custom execution plan function."""
        self.execution_plan = plan
        return self
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the composite tool."""
        if self.execution_plan:
            return self.execution_plan(self.sub_tools, kwargs)
        
        # Default: execute all tools sequentially
        results = []
        for tool in self.sub_tools:
            result = tool.execute(**kwargs)
            results.append(result)
        
        return {
            "composite_tool": self.name,
            "tools_executed": len(results),
            "results": results
        }

# Example Composite Tool: Data Validation and Storage
class DataValidationAndStorageTool(ToolComposition):
    """Composite tool that validates data and stores it."""
    
    def __init__(self):
        super().__init__(
            "data_validation_storage",
            "Validate and store data in database"
        )
        
        def execution_plan(tools, kwargs):
            # Validate data with first tool
            validation_result = tools[0].execute(**kwargs)
            
            if validation_result.get("status") != "success":
                return validation_result
            
            # Store validated data with second tool
            storage_result = tools[1].execute(**kwargs)
            
            return {
                "status": "success",
                "validation": validation_result,
                "storage": storage_result
            }
        
        self.set_execution_plan(execution_plan)
```

---

## AGNO Framework Integration

AGNO is a lightweight framework for building intelligent AI agents with advanced reasoning capabilities.

### Basic AGNO Agent with Tools

```python
"""
AGNO Integration Example
Reference: https://www.tinybird.co/blog/how-to-build-an-analytics-agent-with-agno-and-tinybird-step-by-step
"""

from typing import Optional, Any, Dict

class AGNOTool:
    """Wrapper to integrate custom tools with AGNO agents."""
    
    def __init__(self, name: str, description: str, 
                 callable_func: Callable, 
                 schema: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.callable_func = callable_func
        self.schema = schema or {}

class AGNOAgent:
    """AGNO Agent with tool integration."""
    
    def __init__(self, name: str, model: str = "gpt-4", tools: Optional[List[AGNOTool]] = None):
        self.name = name
        self.model = model
        self.tools = tools or []
        self.execution_history = []
    
    def add_tool(self, tool: AGNOTool) -> None:
        """Register a tool with the agent."""
        self.tools.append(tool)
        logger.info(f"Added tool '{tool.name}' to agent '{self.name}'")
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process user message and execute tools as needed."""
        response = {
            "message": message,
            "tool_calls": [],
            "final_response": ""
        }
        
        # Simulate tool invocation
        for tool in self.tools:
            if tool.name.lower() in message.lower():
                try:
                    tool_result = tool.callable_func()
                    response["tool_calls"].append({
                        "tool": tool.name,
                        "result": tool_result
                    })
                except Exception as e:
                    response["tool_calls"].append({
                        "tool": tool.name,
                        "error": str(e)
                    })
        
        self.execution_history.append(response)
        return response

# Example: Building an Analytics Agent with AGNO
def build_analytics_agent() -> AGNOAgent:
    """Build an analytics agent using AGNO."""
    
    agent = AGNOAgent("analytics_agent", model="gpt-4")
    
    # Define tools
    def query_database():
        """Query analytics database."""
        return {
            "data": [{"date": "2026-04-06", "visitors": 1250}],
            "status": "success"
        }
    
    def get_metrics():
        """Retrieve performance metrics."""
        return {
            "conversion_rate": 0.08,
            "bounce_rate": 0.35,
            "avg_session_duration": 245
        }
    
    def generate_report():
        """Generate analytics report."""
        return {
            "report_type": "monthly",
            "generated_at": "2026-04-06",
            "status": "ready"
        }
    
    # Add tools to agent
    agent.add_tool(AGNOTool(
        "query_database",
        "Query the analytics database for raw data",
        query_database,
        {"type": "object", "properties": {}}
    ))
    
    agent.add_tool(AGNOTool(
        "get_metrics",
        "Get computed analytics metrics",
        get_metrics,
        {"type": "object", "properties": {}}
    ))
    
    agent.add_tool(AGNOTool(
        "generate_report",
        "Generate a formatted analytics report",
        generate_report,
        {"type": "object", "properties": {"period": {"type": "string"}}}
    ))
    
    return agent

# Example Usage
def main_agno_example():
    agent = build_analytics_agent()
    result = agent.process_message("Can you get the metrics and generate a report?")
    print(json.dumps(result, indent=2))
```

---

## Langchain Framework Integration

LangChain provides robust abstractions for tool definition and agent execution.

### Langchain Tools Implementation

```python
"""
Langchain Integration Example
Reference: https://docs.langchain.com/oss/javascript/integrations/tools
"""

from langchain.tools import tool, BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, Type

# Define tool input schema using Pydantic
class WeatherInput(BaseModel):
    location: str = Field(description="The city to get the weather for")
    units: str = Field(default="celsius", description="Temperature units")

class DatabaseQueryInput(BaseModel):
    query: str = Field(description="SQL query to execute")
    max_rows: int = Field(default=100, description="Maximum rows to return")

# Example 1: Using @tool decorator
@tool
def get_weather(location: str, units: str = "celsius") -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city name or coordinates
        units: 'celsius' or 'fahrenheit'
    
    Returns:
        Weather information as a string
    """
    # In production, call actual weather API
    return f"Weather in {location}: 22°{units[0].upper()}, Partly Cloudy"

@tool
def calculate_distance(location1: str, location2: str) -> str:
    """Calculate distance between two locations.
    
    Args:
        location1: First location
        location2: Second location
    
    Returns:
        Distance in kilometers
    """
    # In production, use geolocation API
    return f"Distance between {location1} and {location2}: 250 km"

# Example 2: Creating custom tool class
class DatabaseQueryTool(BaseTool):
    """Custom tool for database queries."""
    
    name: str = "database_query"
    description: str = "Execute SQL queries against the database"
    args_schema: Type[DatabaseQueryInput] = DatabaseQueryInput
    
    def _run(self, query: str, max_rows: int = 100) -> str:
        """Execute the database query."""
        try:
            # In production, execute actual query
            return f"Query executed successfully. Returned {max_rows} rows."
        except Exception as e:
            return f"Query failed: {str(e)}"
    
    async def _arun(self, query: str, max_rows: int = 100) -> str:
        """Async version of the tool."""
        return self._run(query, max_rows)

# Example 3: Creating custom tool with complex logic
class DataProcessingTool(BaseTool):
    """Tool for processing data through a pipeline."""
    
    name: str = "data_processing"
    description: str = "Process data through multiple transformations"
    
    class DataProcessingInput(BaseModel):
        data: list = Field(description="Input data to process")
        operations: list = Field(description="List of operations to apply")
    
    args_schema: Type[BaseModel] = DataProcessingInput
    
    def _run(self, data: list, operations: list) -> str:
        """Process data through operations."""
        result = data
        for operation in operations:
            # Apply transformations
            if operation == "sort":
                result = sorted(result)
            elif operation == "unique":
                result = list(set(result))
            elif operation == "reverse":
                result = list(reversed(result))
        
        return f"Processed data: {result}"
    
    async def _arun(self, data: list, operations: list) -> str:
        return self._run(data, operations)

# Creating a Langchain Agent with Tools
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

def create_langchain_agent():
    """Create a Langchain agent with multiple tools."""
    
    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    # Create tools
    tools = [
        get_weather,
        calculate_distance,
        DatabaseQueryTool(),
        DataProcessingTool()
    ]
    
    # Initialize agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent

# Example: Using the agent
def run_langchain_agent_example():
    agent = create_langchain_agent()
    
    # Agent will automatically select and use appropriate tools
    response = agent.run(
        "What's the weather in London and Paris, and how far apart are they?"
    )
    print(response)
```

---

## Best Practices

### 1. Tool Design Principles

```python
# DO: Clear, single-responsibility tools
class SendEmailTool(BaseTool):
    """Send an email message."""
    name = "send_email"
    description = "Send an email to specified recipients"
    # Single, well-defined purpose

# DON'T: Overly complex, multi-purpose tools
class CommunicationTool(BaseTool):
    """Handle all communication."""
    # This does too many things: email, SMS, notifications, etc.
```

### 2. Error Messages

```python
# DO: Descriptive, actionable error messages
result = {
    "status": "error",
    "error_type": "VALIDATION_ERROR",
    "message": "Required parameter 'query' is missing",
    "suggestion": "Please provide a SQL query in the 'query' parameter",
    "example": "SELECT * FROM users WHERE id = 1"
}

# DON'T: Vague error messages
result = {
    "status": "error",
    "message": "Failed"  # Too vague, not actionable
}
```

### 3. Timeout Management

```python
# DO: Set appropriate timeouts
class APICallTool(BaseTool):
    timeout: int = 30  # Set reasonable timeout
    
    def execute(self, url: str) -> Dict:
        try:
            response = requests.get(url, timeout=self.timeout)
        except requests.Timeout:
            return {"status": "error", "error": "Request timed out"}

# DON'T: No timeout (can hang indefinitely)
response = requests.get(url)  # No timeout specified
```

### 4. Logging and Monitoring

```python
import logging

class MonitoredTool(BaseTool):
    """Tool with comprehensive logging."""
    
    def __init__(self, name: str):
        super().__init__(name, "")
        self.logger = logging.getLogger(name)
    
    def execute(self, **kwargs) -> Dict:
        self.logger.debug(f"Tool invoked with inputs: {kwargs}")
        
        start_time = time.time()
        try:
            result = self._execute(**kwargs)
            duration = time.time() - start_time
            self.logger.info(f"Tool executed successfully in {duration:.2f}s")
            return result
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
            raise
    
    def _execute(self, **kwargs) -> Dict:
        """Override this method in subclasses."""
        raise NotImplementedError
```

### 5. Input Validation

```python
# DO: Validate all inputs thoroughly
class ValidatedTool(BaseTool):
    def execute(self, **kwargs) -> Dict:
        # Validate required parameters
        required = ["query", "database"]
        missing = [p for p in required if p not in kwargs]
        if missing:
            return {"status": "error", "missing_parameters": missing}
        
        # Validate parameter types
        if not isinstance(kwargs["query"], str):
            return {"status": "error", "error": "query must be a string"}
        
        # Validate parameter values
        if len(kwargs["query"]) > 10000:
            return {"status": "error", "error": "query too long (max 10000 chars)"}
        
        # Execute validated tool
        return self._execute_safe(**kwargs)
```

---

## Common Pitfalls and Solutions

### 1. Tool Not Found Error

**Problem:** Agent tries to call a tool that hasn't been registered.

```python
# Problem
agent.execute("unknown_tool", args)  # Tool not found error

# Solution
registry = ToolRegistry()
tool = SomeTool()
spec = ToolSpecification(...)
registry.register(tool, spec)

# Verify tool exists
if registry.get_tool("unknown_tool") is None:
    print("Tool not registered!")
```

### 2. Parameter Type Mismatches

**Problem:** Tool receives parameters in wrong format.

```python
# Problem
tool.execute(number="123")  # String instead of int

# Solution
class TypeSafeWrapper(BaseTool):
    def execute(self, **kwargs) -> Dict:
        try:
            kwargs["number"] = int(kwargs.get("number", 0))
        except ValueError as e:
            return {"status": "error", "error": f"Invalid number format: {e}"}
        
        return self._execute(**kwargs)
```

### 3. Tool Timeout

**Problem:** Tool takes too long to execute, blocking agent.

```python
# Problem
def slow_tool():
    time.sleep(100)  # Too long

# Solution
import signal

class TimeoutTool(BaseTool):
    timeout_seconds: int = 30
    
    def execute(self, **kwargs) -> Dict:
        try:
            result = self._execute_with_timeout(**kwargs)
            return result
        except TimeoutError:
            return {"status": "error", "error": "Tool execution timed out"}
    
    def _execute_with_timeout(self, **kwargs):
        # Use threading or async to implement timeout
        pass
```

### 4. Circular Dependencies

**Problem:** Tools depend on each other in a cycle.

```python
# Problem
ToolA -> ToolB -> ToolC -> ToolA  # Circular

# Solution: Use a DAG (Directed Acyclic Graph) validator
class ChainValidator:
    @staticmethod
    def has_cycles(chain: ToolChain) -> bool:
        """Detect circular dependencies in tool chain."""
        # Implement DFS-based cycle detection
        pass
```

### 5. Resource Leaks

**Problem:** Tools don't clean up resources properly.

```python
# Problem
class BadDatabaseTool(BaseTool):
    def execute(self, query: str) -> Dict:
        conn = connect_to_database()
        # Forgot to close connection!
        return {"result": "data"}

# Solution: Use context managers
class GoodDatabaseTool(BaseTool):
    def execute(self, query: str) -> Dict:
        with connect_to_database() as conn:
            result = conn.execute(query)
            return {"result": result}
```

---

## References and Resources

- AGNO Framework: https://workos.com/blog/agno-the-agent-framework-for-python-teams
- LangChain Documentation: https://docs.langchain.com/
- Tinybird + AGNO Integration: https://www.tinybird.co/blog/how-to-build-an-analytics-agent-with-agno-and-tinybird-step-by-step
- LangChain Tools Guide: https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-tools-complete-guide-creating-using-custom-llm-tools-code-examples-2025

---

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026
