"""
Agent Patterns - Tool Use, Memory, Routing, Error Handling, and State Management

This module demonstrates reusable patterns and best practices for building robust agent systems.
Patterns include: tool integration, memory management, request routing, error recovery, and state persistence.

References:
- Building Production-Ready AI Agents: https://medium.com/data-science-collective/building-production-ready-ai-agents-with-agno-a-comprehensive-engineering-guide-22db32413fdd
- LangChain Patterns: https://docs.langchain.com/
- Agent Design Patterns: https://www.ai-agentsplus.com/blog/building-ai-agents-with-langchain-tutorial-2026/

Author: Shuvam Banerji Seal
"""

# Requirements:
# pip install pydantic>=2.0.0
# pip install typing-extensions

from typing import Dict, Any, Callable, List, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import logging


# ============================================================================
# PATTERN 1: Tool Use Patterns
# ============================================================================


class Tool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments."""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters."""
        pass


class MathTool(Tool):
    """Implements basic math operations."""

    @property
    def name(self) -> str:
        return "math_operation"

    @property
    def description(self) -> str:
        return "Perform basic mathematical operations: add, subtract, multiply, divide"

    def execute(self, operation: str, a: float, b: float) -> str:
        """Execute math operation."""
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return "Error: Division by zero"
                result = a / b
            else:
                return f"Unknown operation: {operation}"

            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        }


class SearchTool(Tool):
    """Simulated search tool."""

    def __init__(self):
        self.db = {
            "python": "Python is a high-level programming language",
            "langchain": "LangChain is a framework for LLM applications",
            "agent": "An agent is an AI system that can take actions",
        }

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search for information about a topic"

    def execute(self, query: str) -> str:
        """Execute search."""
        query_lower = query.lower()
        for key, value in self.db.items():
            if key in query_lower:
                return f"Found: {value}"
        return f"No results found for: {query}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        }


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if not tool:
            return f"Error: Unknown tool '{tool_name}'"
        return tool.execute(**kwargs)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [tool.get_schema() for tool in self.tools.values()]


# ============================================================================
# PATTERN 2: Memory Management Patterns
# ============================================================================


class Memory(ABC):
    """Abstract base class for memory systems."""

    @abstractmethod
    def add(self, message: str, role: str) -> None:
        """Add a message to memory."""
        pass

    @abstractmethod
    def get(self) -> List[Dict[str, str]]:
        """Get all messages."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear memory."""
        pass

    @abstractmethod
    def get_summary(self) -> str:
        """Get memory summary."""
        pass


class ShortTermMemory(Memory):
    """Short-term memory with fixed size."""

    def __init__(self, max_messages: int = 10):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages

    def add(self, message: str, role: str) -> None:
        """Add message and maintain max size."""
        self.messages.append({"role": role, "content": message})
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get(self) -> List[Dict[str, str]]:
        """Get recent messages."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear memory."""
        self.messages = []

    def get_summary(self) -> str:
        """Get summary of recent conversation."""
        if not self.messages:
            return "No messages in memory"
        return f"Last {len(self.messages)} messages: {len([m for m in self.messages if m['role'] == 'user'])} user, {len([m for m in self.messages if m['role'] == 'assistant'])} assistant"


class LongTermMemory(Memory):
    """Long-term memory with persistent storage."""

    def __init__(self, max_messages: int = 1000):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages
        self.summaries: List[str] = []

    def add(self, message: str, role: str) -> None:
        """Add message to long-term storage."""
        self.messages.append(
            {
                "role": role,
                "content": message,
                "timestamp": str(__import__("datetime").datetime.now()),
            }
        )

        # Periodically summarize old messages
        if len(self.messages) > 100 and len(self.messages) % 100 == 0:
            self._create_summary()

    def get(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear memory."""
        self.messages = []
        self.summaries = []

    def _create_summary(self) -> None:
        """Create summary of old messages."""
        if len(self.messages) > 20:
            old_messages = self.messages[:20]
            summary = f"Summary of {len(old_messages)} messages exchanged"
            self.summaries.append(summary)
            self.messages = self.messages[20:]

    def get_summary(self) -> str:
        """Get memory summary including historical."""
        return f"Long-term memory: {len(self.messages)} active messages, {len(self.summaries)} historical summaries"


class HybridMemory(Memory):
    """Combines short-term and long-term memory."""

    def __init__(self):
        self.short_term = ShortTermMemory(max_messages=5)
        self.long_term = LongTermMemory()

    def add(self, message: str, role: str) -> None:
        """Add to both memory systems."""
        self.short_term.add(message, role)
        self.long_term.add(message, role)

    def get(self) -> List[Dict[str, str]]:
        """Get recent messages."""
        return self.short_term.get()

    def get_full_history(self) -> List[Dict[str, str]]:
        """Get full history."""
        return self.long_term.get()

    def clear(self) -> None:
        """Clear both memory systems."""
        self.short_term.clear()
        self.long_term.clear()

    def get_summary(self) -> str:
        """Get hybrid memory summary."""
        return f"Short-term: {self.short_term.get_summary()} | Long-term: {self.long_term.get_summary()}"


# ============================================================================
# PATTERN 3: Request Routing Patterns
# ============================================================================


class Router:
    """Routes requests to appropriate handlers."""

    def __init__(self):
        self.routes: Dict[str, Callable] = {}

    def register_route(self, pattern: str, handler: Callable) -> None:
        """Register a route pattern and handler."""
        self.routes[pattern] = handler

    def route(self, request: str, context: Dict[str, Any]) -> str:
        """Route request to appropriate handler."""
        request_lower = request.lower()

        # Check patterns
        for pattern, handler in self.routes.items():
            if pattern in request_lower:
                return handler(request, context)

        return "Request did not match any known routes"


class SmartRouter(Router):
    """Router with priority and fuzzy matching."""

    def __init__(self):
        super().__init__()
        self.priority_routes: Dict[int, Dict[str, Callable]] = {}
        self.default_handler: Optional[Callable] = None

    def register_route(
        self, pattern: str, handler: Callable, priority: int = 0
    ) -> None:
        """Register route with priority."""
        if priority not in self.priority_routes:
            self.priority_routes[priority] = {}
        self.priority_routes[priority][pattern] = handler

    def set_default_handler(self, handler: Callable) -> None:
        """Set default handler for unmatched routes."""
        self.default_handler = handler

    def route(self, request: str, context: Dict[str, Any]) -> str:
        """Route with priority-based matching."""
        request_lower = request.lower()

        # Check by priority (highest first)
        for priority in sorted(self.priority_routes.keys(), reverse=True):
            for pattern, handler in self.priority_routes[priority].items():
                if pattern in request_lower:
                    return handler(request, context)

        # Use default if available
        if self.default_handler:
            return self.default_handler(request, context)

        return "No matching route found"


# ============================================================================
# PATTERN 4: Error Handling Patterns
# ============================================================================


class ErrorHandler:
    """Handles agent errors with recovery strategies."""

    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.max_retries = 3

    def handle_tool_error(self, tool_name: str, error: Exception) -> str:
        """Handle tool execution errors."""
        self.error_counts[tool_name] = self.error_counts.get(tool_name, 0) + 1

        retry_count = self.error_counts[tool_name]

        if retry_count >= self.max_retries:
            return f"❌ Tool '{tool_name}' failed {self.max_retries} times. Escalating to human review."

        return f"⚠️ Error in {tool_name}: {str(error)}. Retry {retry_count}/{self.max_retries}"

    def handle_validation_error(self, error: str) -> str:
        """Handle input validation errors."""
        return f"❌ Validation error: {error}"

    def handle_timeout(self, operation: str) -> str:
        """Handle operation timeout."""
        return f"⏱️ Operation '{operation}' timed out. Please try again."

    def reset_error_count(self, tool_name: str) -> None:
        """Reset error count for a tool."""
        self.error_counts[tool_name] = 0


class RetryStrategy:
    """Implements retry logic with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        import time

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                delay = self.base_delay * (2**attempt)
                print(f"Attempt {attempt + 1} failed. Retrying in {delay}s...")
                time.sleep(delay)


# ============================================================================
# PATTERN 5: State Management Patterns
# ============================================================================


@dataclass
class AgentState:
    """Represents agent execution state."""

    request_id: str
    input: str
    output: Optional[str] = None
    status: str = "pending"
    tools_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "request_id": self.request_id,
            "input": self.input,
            "output": self.output,
            "status": self.status,
            "tools_used": self.tools_used,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    def save(self, filepath: str) -> None:
        """Save state to file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class StateManager:
    """Manages agent state persistence and recovery."""

    def __init__(self):
        self.states: Dict[str, AgentState] = {}

    def save_state(self, state: AgentState) -> None:
        """Save state."""
        self.states[state.request_id] = state

    def get_state(self, request_id: str) -> Optional[AgentState]:
        """Get saved state."""
        return self.states.get(request_id)

    def update_state(self, request_id: str, **updates) -> None:
        """Update state fields."""
        if request_id in self.states:
            state = self.states[request_id]
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)

    def get_state_summary(self, request_id: str) -> str:
        """Get state summary."""
        state = self.get_state(request_id)
        if not state:
            return f"No state found for {request_id}"
        return f"Status: {state.status}, Tools: {state.tools_used}, Errors: {len(state.errors)}"


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Agent Patterns - Reusable Components and Best Practices")
    print("=" * 70)

    # PATTERN 1: Tool Use
    print("\n\n" + "=" * 70)
    print("PATTERN 1: Tool Use Patterns")
    print("=" * 70)

    registry = ToolRegistry()
    registry.register(MathTool())
    registry.register(SearchTool())

    print("\n📋 Available Tools:")
    for tool_schema in registry.list_tools():
        print(f"  - {tool_schema['name']}: {tool_schema['description']}")

    print("\n🔧 Executing Tools:")
    result1 = registry.execute_tool("math_operation", operation="add", a=5, b=3)
    print(f"  Math: {result1}")

    result2 = registry.execute_tool("search", query="python")
    print(f"  Search: {result2}")

    # PATTERN 2: Memory Management
    print("\n\n" + "=" * 70)
    print("PATTERN 2: Memory Management Patterns")
    print("=" * 70)

    memory_types = [
        ("Short-Term", ShortTermMemory()),
        ("Long-Term", LongTermMemory()),
        ("Hybrid", HybridMemory()),
    ]

    for name, memory in memory_types:
        memory.add("Hello, how can I help?", "user")
        memory.add("I can assist with various tasks", "assistant")
        print(f"\n  {name}: {memory.get_summary()}")

    # PATTERN 3: Request Routing
    print("\n\n" + "=" * 70)
    print("PATTERN 3: Request Routing Patterns")
    print("=" * 70)

    router = SmartRouter()
    router.register_route("hello", lambda r, c: "👋 Greeting handler", priority=2)
    router.register_route("help", lambda r, c: "🆘 Help handler", priority=2)
    router.register_route("search", lambda r, c: "🔍 Search handler", priority=1)
    router.set_default_handler(lambda r, c: "❓ Default handler")

    test_requests = ["hello", "help me", "search for python", "unknown command"]
    for request in test_requests:
        response = router.route(request, {})
        print(f"  Request: '{request}' → {response}")

    # PATTERN 4: Error Handling
    print("\n\n" + "=" * 70)
    print("PATTERN 4: Error Handling Patterns")
    print("=" * 70)

    handler = ErrorHandler()
    for i in range(4):
        result = handler.handle_tool_error(
            "search_tool", Exception("Connection timeout")
        )
        print(f"  Attempt {i + 1}: {result}")

    # PATTERN 5: State Management
    print("\n\n" + "=" * 70)
    print("PATTERN 5: State Management Patterns")
    print("=" * 70)

    manager = StateManager()

    state = AgentState(
        request_id="req_001",
        input="What is Python?",
        status="in_progress",
    )

    manager.save_state(state)
    manager.update_state(
        "req_001", output="Python is...", status="completed", tools_used=["search"]
    )

    summary = manager.get_state_summary("req_001")
    print(f"\n  State Summary: {summary}")

    print("\n" + "=" * 70)
    print("Pattern Applications")
    print("=" * 70)
    print("""
Tool Use Patterns:
  - ToolRegistry: Centralized tool management
  - Tool: Base class for extensible tools
  - MathTool, SearchTool: Concrete implementations

Memory Patterns:
  - ShortTermMemory: Recent conversation context
  - LongTermMemory: Historical information
  - HybridMemory: Combined short and long-term

Routing Patterns:
  - Router: Basic pattern matching
  - SmartRouter: Priority-based with defaults
  - Use for request type classification

Error Handling Patterns:
  - ErrorHandler: Error counting and escalation
  - RetryStrategy: Exponential backoff
  - Use for resilience and recovery

State Management Patterns:
  - AgentState: Immutable state representation
  - StateManager: Persistence and updates
  - Use for tracking request lifecycle
    """)
