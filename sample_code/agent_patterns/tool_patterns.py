"""
Advanced Tool Patterns

Advanced patterns for building robust agent tools:
- Tool with retry logic
- Tool with caching
- Parallel tool execution
- Tool composition
- Dynamic tool loading

Author: Shuvam Banerji
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
import time
import logging
import threading
import concurrent.futures
from functools import wraps
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Base exception for tool errors."""
    pass


class ToolNotFoundError(ToolError):
    """Tool not found error."""
    pass


class ToolExecutionError(ToolError):
    """Tool execution error."""
    pass


class ToolTimeoutError(ToolError):
    """Tool timeout error."""
    pass


class RetryStrategy(Enum):
    """Retry strategies."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class ToolResult:
    """Result of tool execution."""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    attempts: int = 1
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        pass


class RetryableTool(Tool):
    """Tool with retry logic."""

    def __init__(
        self,
        base_tool: Tool,
        max_retries: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True
    ):
        """
        Initialize retryable tool.

        Args:
            base_tool: Underlying tool to wrap
            max_retries: Maximum retry attempts
            strategy: Retry strategy
            base_delay: Base delay between retries
            max_delay: Maximum delay
            jitter: Add randomness to delays
        """
        self.base_tool = base_tool
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    @property
    def name(self) -> str:
        return self.base_tool.name

    @property
    def description(self) -> str:
        return self.base_tool.description

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.base_tool.parameters

    def execute(self, **kwargs) -> Any:
        """Execute with retries."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = self.base_tool.execute(**kwargs)
                if attempt > 0:
                    logger.info(f"Retry {attempt} succeeded for {self.name}")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {self.name}: {e}")

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed for {self.name}")

        raise ToolExecutionError(f"Tool {self.name} failed after {self.max_retries + 1} attempts: {last_error}")

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            delay *= (0.5 + hashlib_md5(str(attempt)) % 100 / 100)

        return delay


def hashlib_md5(s: str) -> int:
    """Simple hash function."""
    import hashlib
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


class CachedTool(Tool):
    """Tool with result caching."""

    def __init__(self, base_tool: Tool, ttl_seconds: int = 3600, max_cache_size: int = 1000):
        """
        Initialize cached tool.

        Args:
            base_tool: Underlying tool to wrap
            ttl_seconds: Cache TTL in seconds
            max_cache_size: Maximum cache entries
        """
        self.base_tool = base_tool
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size

        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"cached_{self.base_tool.name}"

    @property
    def description(self) -> str:
        return f"Cached version of {self.base_tool.description}"

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.base_tool.parameters

    def execute(self, **kwargs) -> Any:
        """Execute with caching."""
        cache_key = self._make_cache_key(kwargs)

        with self._cache_lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]

                if time.time() - timestamp < self.ttl_seconds:
                    logger.info(f"Cache hit for {self.base_tool.name}")
                    return result

                del self._cache[cache_key]

        result = self.base_tool.execute(**kwargs)

        with self._cache_lock:
            if len(self._cache) >= self.max_cache_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[cache_key] = (result, time.time())

        return result

    def _make_cache_key(self, kwargs: Dict) -> str:
        """Create cache key from arguments."""
        sorted_items = sorted(kwargs.items())
        return str(sorted_items)

    def clear_cache(self) -> None:
        """Clear the cache."""
        with self._cache_lock:
            self._cache.clear()
            logger.info(f"Cache cleared for {self.base_tool.name}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            total_entries = len(self._cache)
            valid_entries = sum(
                1 for _, timestamp in self._cache.values()
                if time.time() - timestamp < self.ttl_seconds
            )

            return {
                "total_entries": total_entries,
                "valid_entries": valid_entries,
                "expired_entries": total_entries - valid_entries,
                "max_size": self.max_cache_size,
                "ttl_seconds": self.ttl_seconds
            }


class RateLimitedTool(Tool):
    """Tool with rate limiting."""

    def __init__(
        self,
        base_tool: Tool,
        max_calls: int = 100,
        window_seconds: int = 60
    ):
        """
        Initialize rate-limited tool.

        Args:
            base_tool: Underlying tool
            max_calls: Maximum calls per window
            window_seconds: Time window in seconds
        """
        self.base_tool = base_tool
        self.max_calls = max_calls
        self.window_seconds = window_seconds

        self._calls: List[float] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.base_tool.name

    @property
    def description(self) -> str:
        return f"Rate-limited {self.base_tool.description}"

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.base_tool.parameters

    def execute(self, **kwargs) -> Any:
        """Execute with rate limiting."""
        with self._lock:
            now = time.time()
            self._calls = [t for t in self._calls if now - t < self.window_seconds]

            if len(self._calls) >= self.max_calls:
                sleep_time = self.window_seconds - (now - self._calls[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached for {self.name}, sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    self._calls = [t for t in self._calls if time.time() - t < self.window_seconds]

            self._calls.append(time.time())

        return self.base_tool.execute(**kwargs)

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get rate limit status."""
        with self._lock:
            now = time.time()
            recent_calls = [t for t in self._calls if now - t < self.window_seconds]

            return {
                "calls_in_window": len(recent_calls),
                "max_calls": self.max_calls,
                "window_seconds": self.window_seconds,
                "remaining_calls": max(0, self.max_calls - len(recent_calls)),
                "reset_in_seconds": (
                    self.window_seconds - (now - self._calls[0])
                    if self._calls else 0
                )
            }


class ParallelToolExecutor:
    """Execute multiple tools in parallel."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum parallel workers
        """
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def execute_multiple(
        self,
        tools: List[Tuple[Tool, Dict]],
        timeout: Optional[float] = None
    ) -> List[ToolResult]:
        """
        Execute multiple tools in parallel.

        Args:
            tools: List of (tool, arguments) tuples
            timeout: Optional timeout

        Returns:
            List of ToolResults
        """
        futures = []

        for tool, args in tools:
            future = self.executor.submit(self._execute_single, tool, args)
            futures.append((tool.name, future))

        results = []

        for tool_name, future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except concurrent.futures.TimeoutError:
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error="Execution timeout"
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error=str(e)
                ))

        return results

    def _execute_single(self, tool: Tool, args: Dict) -> ToolResult:
        """Execute single tool and wrap result."""
        start_time = time.time()

        try:
            output = tool.execute(**args)
            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                tool_name=tool.name,
                success=True,
                output=output,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                tool_name=tool.name,
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=execution_time
            )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown executor."""
        self.executor.shutdown(wait=wait)


class ToolCompressor:
    """Compose multiple tools into one."""

    def __init__(self, tools: List[Tool], combine_fn: Callable):
        """
        Initialize tool compressor.

        Args:
            tools: Tools to compose
            combine_fn: Function to combine tool outputs
        """
        self.tools = tools
        self.combine_fn = combine_fn

    @property
    def name(self) -> str:
        return f"composed_{'_'.join(t.name for t in self.tools)}"

    @property
    def description(self) -> str:
        tool_descs = ", ".join(t.name for t in self.tools)
        return f"Composed tool combining: {tool_descs}"

    @property
    def parameters(self) -> Dict[str, Any]:
        combined = {"type": "object", "properties": {}}
        for tool in self.tools:
            if tool.parameters.get("properties"):
                combined["properties"].update(tool.parameters["properties"])
        return combined

    def execute(self, **kwargs) -> Any:
        """Execute all tools and combine results."""
        results = {}

        for tool in self.tools:
            tool_kwargs = {
                k: v for k, v in kwargs.items()
                if k in tool.parameters.get("properties", {})
            }

            if tool_kwargs:
                try:
                    results[tool.name] = tool.execute(**tool_kwargs)
                except Exception as e:
                    results[tool.name] = {"error": str(e)}

        return self.combine_fn(results)


class DynamicToolLoader:
    """Dynamically load tools from registry."""

    def __init__(self):
        self._registry: Dict[str, Tool] = {}
        self._loaders: Dict[str, Callable] = {}

    def register_loader(self, tool_type: str, loader: Callable) -> None:
        """
        Register a tool loader.

        Args:
            tool_type: Type identifier
            loader: Loader function
        """
        self._loaders[tool_type] = loader
        logger.info(f"Registered loader for tool type: {tool_type}")

    def load_tool(self, tool_type: str, config: Dict[str, Any]) -> Tool:
        """
        Load tool dynamically.

        Args:
            tool_type: Type identifier
            config: Tool configuration

        Returns:
            Loaded tool instance
        """
        if tool_type not in self._loaders:
            raise ToolNotFoundError(f"No loader registered for tool type: {tool_type}")

        tool = self._loaders[tool_type](config)
        self._registry[tool.name] = tool

        return tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._registry.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all loaded tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self._registry.values()
        ]


class MockSearchTool(Tool):
    """Mock search tool for demonstration."""

    def __init__(self):
        self.knowledge_base = {
            "python": "Python is a high-level programming language.",
            "machine learning": "ML enables computers to learn from data.",
            "ai": "AI stands for Artificial Intelligence.",
            "llm": "Large Language Models process text.",
            "rag": "RAG combines retrieval with generation."
        }

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search for information on a topic"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }

    def execute(self, query: str) -> str:
        """Execute search."""
        query_lower = query.lower()

        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return value

        return f"No information found for: {query}"


class MockCalculatorTool(Tool):
    """Mock calculator tool."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }

    def execute(self, operation: str, a: float, b: float) -> str:
        """Execute calculation."""
        if operation == "add":
            return f"{a} + {b} = {a + b}"
        elif operation == "subtract":
            return f"{a} - {b} = {a - b}"
        elif operation == "multiply":
            return f"{a} * {b} = {a * b}"
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return f"{a} / {b} = {a / b}"
        else:
            raise ValueError(f"Unknown operation: {operation}")


class MockWeatherTool(Tool):
    """Mock weather tool."""

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get weather information for a location"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }

    def execute(self, location: str) -> str:
        """Get weather."""
        return f"Weather in {location}: 72°F, Sunny"


def demo():
    """Demonstrate advanced tool patterns."""
    print("=" * 70)
    print("Advanced Tool Patterns Demo")
    print("=" * 70)

    print("\n--- Tool with Retry Logic ---")
    calculator = MockCalculatorTool()
    retry_calculator = RetryableTool(
        calculator,
        max_retries=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF
    )

    result = retry_calculator.execute(operation="add", a=5, b=3)
    print(f"Result: {result}")

    print("\n--- Tool with Caching ---")
    search = MockSearchTool()
    cached_search = CachedTool(search, ttl_seconds=3600)

    print(f"First call: {cached_search.execute(query='python')}")
    print(f"Second call (cached): {cached_search.execute(query='python')}")

    stats = cached_search.get_cache_stats()
    print(f"Cache stats: {stats}")

    print("\n--- Rate Limited Tool ---")
    weather = MockWeatherTool()
    rate_limited_weather = RateLimitedTool(weather, max_calls=5, window_seconds=60)

    for i in range(3):
        result = rate_limited_weather.execute(location="New York")
        print(f"Call {i + 1}: {result}")

    status = rate_limited_weather.get_rate_limit_status()
    print(f"Rate limit status: {status}")

    print("\n--- Parallel Tool Execution ---")
    executor = ParallelToolExecutor(max_workers=3)

    tools_and_args = [
        (MockCalculatorTool(), {"operation": "add", "a": 10, "b": 5}),
        (MockCalculatorTool(), {"operation": "multiply", "a": 3, "b": 7}),
        (MockCalculatorTool(), {"operation": "subtract", "a": 20, "b": 8}),
    ]

    results = executor.execute_multiple(tools_and_args)

    for result in results:
        print(f"{result.tool_name}: {result.output if result.success else result.error}")
        print(f"  Time: {result.execution_time_ms:.2f}ms, Success: {result.success}")

    executor.shutdown()

    print("\n--- Tool Composition ---")
    tools = [MockSearchTool(), MockCalculatorTool()]
    compressor = ToolCompressor(
        tools,
        combine_fn=lambda results: f"Combined {len(results)} tools"
    )

    result = compressor.execute(operation="add", a=5, b=3, query="python")
    print(f"Composed result: {result}")

    print("\n--- Dynamic Tool Loading ---")
    loader = DynamicToolLoader()

    def create_calculator(config: Dict) -> Tool:
        return MockCalculatorTool()

    loader.register_loader("calculator", create_calculator)

    tool = loader.load_tool("calculator", {})
    print(f"Loaded tool: {tool.name}")
    print(f"Tools available: {loader.list_tools()}")


if __name__ == "__main__":
    demo()