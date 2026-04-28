"""
Minimal Agent Implementation

A minimal but complete agent system demonstrating:
- Tool definition and execution
- Simple memory
- Conversation loop

Author: Shuvam Banerji
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
import time
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


@dataclass
class Message:
    """A message in the conversation."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ToolDefinition:
    """Definition of a tool the agent can use."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable


@dataclass
class ToolResult:
    """Result of tool execution."""
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class AgentState:
    """Current state of the agent."""
    status: AgentStatus = AgentStatus.IDLE
    current_plan: Optional[str] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 10


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

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for tool parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool."""
        pass


class CalculatorTool(Tool):
    """Calculator tool for mathematical operations."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations. Supports: add, subtract, multiply, divide, power, sqrt"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt"],
                    "description": "Mathematical operation to perform"
                },
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand (not needed for sqrt)"}
            },
            "required": ["operation", "a"]
        }

    def execute(self, operation: str, a: float, b: Optional[float] = None) -> str:
        """Execute calculation."""
        try:
            if operation == "add":
                result = a + (b if b is not None else 0)
                return f"Result: {a} + {b} = {result}"
            elif operation == "subtract":
                result = a - (b if b is not None else 0)
                return f"Result: {a} - {b} = {result}"
            elif operation == "multiply":
                result = a * (b if b is not None else 1)
                return f"Result: {a} * {b} = {result}"
            elif operation == "divide":
                if b is None or b == 0:
                    return "Error: Division by zero"
                result = a / b
                return f"Result: {a} / {b} = {result}"
            elif operation == "power":
                result = a ** (b if b is not None else 2)
                return f"Result: {a} ^ {b} = {result}"
            elif operation == "sqrt":
                if a < 0:
                    return "Error: Cannot take square root of negative number"
                result = a ** 0.5
                return f"Result: sqrt({a}) = {result}"
            else:
                return f"Error: Unknown operation '{operation}'"
        except Exception as e:
            return f"Error: {str(e)}"


class SearchTool(Tool):
    """Search tool for information retrieval."""

    def __init__(self):
        self.knowledge_base = {
            "python": "Python is a high-level, interpreted programming language known for its clear syntax and readability.",
            "javascript": "JavaScript is a scripting language that enables interactive web pages and web applications.",
            "machine learning": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "neural network": "A neural network is a computing system inspired by biological neural networks in the brain.",
            "llm": "Large Language Models are AI models trained on vast text data to understand and generate human language.",
            "rag": "Retrieval-Augmented Generation combines information retrieval with generative AI for better answers.",
        }

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search for information on a topic. Returns relevant knowledge from the knowledge base."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }

    def execute(self, query: str) -> str:
        """Execute search."""
        query_lower = query.lower()

        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return f"Found information about '{key}': {value}"

        return f"No information found for query: '{query}'"


class WebFetchTool(Tool):
    """Tool to fetch content from URLs."""

    def __init__(self):
        self.cache: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch content from a URL. Returns the page title and summary."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"}
            },
            "required": ["url"]
        }

    def execute(self, url: str) -> str:
        """Fetch URL content."""
        if url in self.cache:
            return f"[Cached] {self.cache[url]}"

        if not url.startswith(("http://", "https://")):
            return f"Error: Invalid URL format. Must start with http:// or https://"

        content = f"Content from {url}: This is a mock response."
        self.cache[url] = content
        return f"Successfully fetched {url}: {content[:100]}..."


class ToolRegistry:
    """Registry for managing and executing tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        start_time = time.time()

        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool '{tool_name}' not found"
            )

        try:
            output = tool.execute(**kwargs)
            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                execution_time_ms=execution_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )


class SimpleMemory:
    """Simple memory system for agent conversations."""

    def __init__(self, max_messages: int = 50):
        self.messages: List[Message] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str) -> Message:
        """Add a message to memory."""
        message = Message(role=role, content=content)
        self.messages.append(message)

        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

        return message

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get recent messages."""
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()

    def get_conversation_history(self) -> str:
        """Get formatted conversation history."""
        history = []
        for msg in self.messages:
            role_name = "User" if msg.role == "user" else "Assistant"
            history.append(f"{role_name}: {msg.content}")
        return "\n".join(history)

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()

    def search(self, query: str) -> List[Message]:
        """Search messages containing query."""
        query_lower = query.lower()
        return [
            msg for msg in self.messages
            if query_lower in msg.content.lower()
        ]


class ResponseGenerator:
    """Simple response generator for the agent."""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name

    def generate(
        self,
        user_input: str,
        tool_results: List[ToolResult],
        conversation_history: str
    ) -> str:
        """
        Generate a response based on input and tool results.

        In a real implementation, this would use an LLM.
        """
        tool_summary = []
        for result in tool_results:
            if result.success:
                tool_summary.append(f"- {result.tool_name}: {result.output}")
            else:
                tool_summary.append(f"- {result.tool_name}: Error - {result.error}")

        if tool_summary:
            tools_str = "\n".join(tool_summary)
            response = (
                f"I used the following tools to help answer your question:\n\n"
                f"{tools_str}\n\n"
                f"Based on the tool results, here's my response to your question about '{user_input}':\n\n"
                f"This is a mock response generated by {self.model_name}. "
                f"In production, this would be generated by an LLM."
            )
        else:
            response = (
                f"I received your message: '{user_input}'\n\n"
                f"To help you better, I can use various tools. "
                f"Try asking me to calculate something, search for information, or fetch a URL."
            )

        return response


class MinimalAgent:
    """Minimal agent with tool execution and memory."""

    def __init__(self, name: str = "Assistant", max_iterations: int = 10):
        """
        Initialize the agent.

        Args:
            name: Agent name
            max_iterations: Maximum tool call iterations per query
        """
        self.name = name
        self.state = AgentState(max_iterations=max_iterations)
        self.memory = SimpleMemory()
        self.tool_registry = ToolRegistry()
        self.response_generator = ResponseGenerator()

        self._register_default_tools()
        logger.info(f"Agent '{name}' initialized")

    def _register_default_tools(self) -> None:
        """Register default tools."""
        self.tool_registry.register(CalculatorTool())
        self.tool_registry.register(SearchTool())
        self.tool_registry.register(WebFetchTool())

    def add_tool(self, tool: Tool) -> None:
        """Add a custom tool."""
        self.tool_registry.register(tool)

    def process(self, user_input: str) -> str:
        """
        Process user input and generate response.

        Args:
            user_input: User's message

        Returns:
            Agent's response
        """
        self.state.status = AgentStatus.THINKING
        self.state.iteration_count = 0
        self.state.tool_results.clear()

        self.memory.add_message("user", user_input)

        tool_results = self._execute_tools_if_needed(user_input)

        response = self.response_generator.generate(
            user_input=user_input,
            tool_results=tool_results,
            conversation_history=self.memory.get_conversation_history()
        )

        self.memory.add_message("assistant", response)
        self.state.status = AgentStatus.DONE

        return response

    def _execute_tools_if_needed(self, user_input: str) -> List[ToolResult]:
        """Determine if tools are needed and execute them."""
        user_lower = user_input.lower()
        tool_results = []

        if any(word in user_lower for word in ["calculate", "compute", "+", "-", "*", "/", "^"]):
            self.state.status = AgentStatus.ACTING
            result = self._handle_math_query(user_input)
            tool_results.append(result)

        elif any(word in user_lower for word in ["search", "find", "what is", "how does", "tell me about"]):
            self.state.status = AgentStatus.ACTING
            result = self._handle_search_query(user_input)
            tool_results.append(result)

        elif "fetch" in user_lower or "url" in user_lower:
            self.state.status = AgentStatus.ACTING
            result = self._handle_url_query(user_input)
            tool_results.append(result)

        return tool_results

    def _handle_math_query(self, query: str) -> ToolResult:
        """Handle mathematical query."""
        query_lower = query.lower()

        operations = {
            "add": "+", "subtract": "-", "multiply": "*", "divide": "/",
            "power": "^", "sqrt": "sqrt"
        }

        for op_name, op_symbol in operations.items():
            if op_name in query_lower:
                try:
                    numbers = [float(s) for s in query.split() if self._is_number(s)]
                    if len(numbers) >= 1:
                        a = numbers[0]
                        b = numbers[1] if len(numbers) > 1 else None
                        return self.tool_registry.execute("calculator", operation=op_name, a=a, b=b)
                except ValueError:
                    pass

        return ToolResult(
            tool_name="calculator",
            success=False,
            output="",
            error="Could not parse math expression"
        )

    def _handle_search_query(self, query: str) -> ToolResult:
        """Handle search query."""
        query_clean = query.lower()
        for phrase in ["search for", "find", "what is", "how does", "tell me about"]:
            query_clean = query_clean.replace(phrase, "").strip()

        return self.tool_registry.execute("search", query=query_clean)

    def _handle_url_query(self, query: str) -> ToolResult:
        """Handle URL fetch query."""
        import re
        urls = re.findall(r'https?://[^\s]+', query)

        if urls:
            return self.tool_registry.execute("web_fetch", url=urls[0])

        return ToolResult(
            tool_name="web_fetch",
            success=False,
            output="",
            error="No URL found in query"
        )

    @staticmethod
    def _is_number(s: str) -> bool:
        """Check if string is a number."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "name": self.name,
            "status": self.state.status.value,
            "iterations": self.state.iteration_count,
            "messages_in_memory": len(self.memory.messages),
            "tools_available": len(self.tool_registry.tools)
        }


def demo():
    """Demonstrate minimal agent."""
    print("=" * 70)
    print("Minimal Agent Implementation Demo")
    print("=" * 70)

    agent = MinimalAgent(name="HelperBot")

    print(f"\nAgent Status: {agent.get_status()}")
    print(f"Available Tools: {[t['name'] for t in agent.tool_registry.list_tools()]}")

    test_inputs = [
        "What is machine learning?",
        "Calculate 15 + 27",
        "What is the square root of 144?",
        "Search for information about Python programming",
    ]

    print("\n" + "-" * 70)
    print("Conversation:")
    print("-" * 70)

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        response = agent.process(user_input)
        print(f"{agent.name}: {response}")
        print(f"[Status: {agent.state.status.value}]")


if __name__ == "__main__":
    demo()