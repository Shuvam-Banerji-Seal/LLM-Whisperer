"""
AGNO Tool Integration and Function Calling

This module demonstrates how to integrate tools with AGNO agents,
enabling them to call external functions, APIs, and services.

Author: Shuvam Banerji Seal
Source: https://docs.agno.com/agents/tools
Source: https://docs.agno.com/mcp
Source: https://github.com/agno-agi/agno/tree/main/cookbook
"""

from typing import Optional, List, Dict, Any, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


class ToolCategory(Enum):
    """
    Categories of tools that AGNO agents can use.

    Reference: https://docs.agno.com/agents/tools
    """

    CODING = "coding"  # Code execution and analysis
    WEB_SEARCH = "web_search"  # Internet search
    DATABASE = "database"  # Database operations
    API = "api"  # REST/GraphQL APIs
    FILE_SYSTEM = "file_system"  # File operations
    EXTERNAL_SERVICE = "external"  # Third-party services
    MCP = "mcp"  # Model Context Protocol
    CUSTOM = "custom"  # User-defined tools


@dataclass
class ToolParameter:
    """Definition of a single tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameter to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "enum_values": self.enum_values,
        }


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool that AGNO agents can use.

    Tools enable agents to:
    - Execute code
    - Search information
    - Interact with databases
    - Call APIs
    - Perform system operations
    """

    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter]
    return_type: str = "string"
    error_handling: str = "raise"  # raise, return, log
    timeout_seconds: int = 30
    max_retries: int = 3
    requires_authentication: bool = False
    auth_type: Optional[str] = None  # api_key, oauth, basic

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool definition to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "requires_authentication": self.requires_authentication,
        }


class Tool(ABC):
    """
    Abstract base class for AGNO tools.

    All tools inherit from this to ensure consistent interface
    for AGNO agents to call them.
    """

    def __init__(self, definition: ToolDefinition):
        """Initialize tool with its definition."""
        self.definition = definition
        self.call_count = 0
        self.error_count = 0

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Execution result with status and output
        """
        pass

    def validate_parameters(self, **kwargs) -> bool:
        """Validate that provided parameters match definition."""
        required_params = {p.name for p in self.definition.parameters if p.required}
        provided_params = set(kwargs.keys())

        if not required_params.issubset(provided_params):
            missing = required_params - provided_params
            logger.error(f"Missing required parameters: {missing}")
            return False

        return True

    def call(self, **kwargs) -> Dict[str, Any]:
        """
        Call the tool with error handling and logging.

        AGNO Tool Calling Flow:
        1. Validate parameters
        2. Increment call counter
        3. Execute tool
        4. Handle errors
        5. Return result
        """
        logger.info(
            f"Tool call: {self.definition.name} with params {list(kwargs.keys())}"
        )

        # Validate parameters
        if not self.validate_parameters(**kwargs):
            return {
                "status": "error",
                "error": "Invalid parameters",
                "tool": self.definition.name,
            }

        self.call_count += 1

        try:
            result = self.execute(**kwargs)
            result["status"] = "success"
            result["tool"] = self.definition.name
            result["call_number"] = self.call_count
            return result
        except Exception as e:
            self.error_count += 1
            logger.error(f"Tool execution error: {e}")

            if self.definition.error_handling == "raise":
                raise

            return {
                "status": "error",
                "error": str(e),
                "tool": self.definition.name,
                "call_number": self.call_count,
            }


class WebSearchTool(Tool):
    """
    Web search tool for AGNO agents.

    Enables agents to search the internet and retrieve information.

    AGNO Pattern: Search tools are fundamental for:
    - Current event awareness
    - Real-time data gathering
    - Fact verification
    - Background research
    """

    def __init__(self):
        definition = ToolDefinition(
            name="web_search",
            description="Search the internet for information",
            category=ToolCategory.WEB_SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description="Search language",
                    required=False,
                    default="en",
                ),
            ],
            return_type="list",
            timeout_seconds=10,
        )
        super().__init__(definition)

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute web search."""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 10)

        # Simulated search results
        results = [
            {
                "title": f"Result {i + 1} for '{query}'",
                "url": f"https://example.com/result{i + 1}",
                "snippet": f"This is a search result about {query}",
                "rank": i + 1,
            }
            for i in range(min(max_results, 5))
        ]

        return {"query": query, "results": results, "result_count": len(results)}


class CodeExecutionTool(Tool):
    """
    Code execution tool for AGNO agents.

    Enables agents to write and execute code.

    AGNO Pattern: Code execution tools allow agents to:
    - Build and test applications
    - Analyze data with code
    - Solve computational problems
    - Demonstrate solutions

    Reference: https://docs.agno.com/agents/tools
    """

    def __init__(self):
        definition = ToolDefinition(
            name="code_execution",
            description="Execute Python code and return results",
            category=ToolCategory.CODING,
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
                    required=True,
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description="Programming language",
                    required=False,
                    default="python",
                    enum_values=["python", "javascript"],
                ),
            ],
            return_type="string",
            timeout_seconds=30,
        )
        super().__init__(definition)

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute code."""
        code = kwargs.get("code", "")
        language = kwargs.get("language", "python")

        logger.info(f"Executing {language} code")

        # In production, would use actual code execution
        # For safety, typically sandboxed environments
        # Reference: jupyter execution, docker containers, etc.

        return {
            "language": language,
            "code_snippet": code[:100] + "..." if len(code) > 100 else code,
            "execution_status": "completed",
            "output": "Code execution result (simulated)",
            "error": None,
        }


class DatabaseTool(Tool):
    """
    Database query tool for AGNO agents.

    Enables agents to query and manage databases.
    """

    def __init__(self, connection_string: str = ""):
        self.connection_string = connection_string

        definition = ToolDefinition(
            name="database_query",
            description="Execute database queries",
            category=ToolCategory.DATABASE,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="SQL query to execute",
                    required=True,
                ),
                ToolParameter(
                    name="database",
                    type="string",
                    description="Database name",
                    required=True,
                ),
            ],
            return_type="list",
            requires_authentication=True,
            auth_type="connection_string",
        )
        super().__init__(definition)

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute database query."""
        query = kwargs.get("query", "")
        database = kwargs.get("database", "")

        logger.info(f"Executing database query on {database}")

        return {
            "database": database,
            "query": query,
            "rows_returned": 0,
            "execution_time_ms": 45.2,
            "data": [],
        }


class APICallTool(Tool):
    """
    Generic API calling tool for AGNO agents.

    Enables agents to make HTTP requests to APIs.
    """

    def __init__(self):
        definition = ToolDefinition(
            name="api_call",
            description="Make HTTP requests to APIs",
            category=ToolCategory.API,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="API endpoint URL",
                    required=True,
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="HTTP method",
                    required=False,
                    default="GET",
                    enum_values=["GET", "POST", "PUT", "DELETE"],
                ),
                ToolParameter(
                    name="headers",
                    type="object",
                    description="HTTP headers",
                    required=False,
                ),
                ToolParameter(
                    name="data",
                    type="object",
                    description="Request body",
                    required=False,
                ),
            ],
            return_type="object",
            timeout_seconds=15,
        )
        super().__init__(definition)

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute API call."""
        url = kwargs.get("url", "")
        method = kwargs.get("method", "GET")

        logger.info(f"Making {method} request to {url}")

        return {
            "url": url,
            "method": method,
            "status_code": 200,
            "response": {"message": "API response (simulated)"},
            "response_time_ms": 234.5,
        }


class AGNOToolRegistry:
    """
    Registry for managing AGNO tools.

    The tool registry:
    - Stores available tools
    - Manages tool capabilities
    - Enables tool discovery
    - Handles tool versioning

    AGNO Pattern: Tool registry allows:
    - Central management of agent capabilities
    - Easy tool addition/removal
    - Tool capability checking
    - Runtime tool injection
    """

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {
            cat: [] for cat in ToolCategory
        }

    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance to register
        """
        name = tool.definition.name
        category = tool.definition.category

        self.tools[name] = tool
        if name not in self.categories[category]:
            self.categories[category].append(name)

        logger.info(f"Registered tool: {name} ({category.value})")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category."""
        return [self.tools[name] for name in self.categories[category]]

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [tool.definition.to_dict() for tool in self.tools.values()]

    def get_tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get definitions of all registered tools."""
        return {name: tool.definition.to_dict() for name, tool in self.tools.items()}


class AGNOFunctionCaller:
    """
    Manages function calling in AGNO agents.

    Function calling allows agents to:
    - Request specific tool invocations
    - Pass structured arguments
    - Handle tool responses
    - Manage tool execution flow

    AGNO Pattern: Function calling is the mechanism by which
    agents decide to use tools and how they interpret results.

    Reference: https://docs.agno.com/agents/tools
    """

    def __init__(self, tool_registry: AGNOToolRegistry):
        """Initialize function caller with tool registry."""
        self.tool_registry = tool_registry
        self.call_history: List[Dict[str, Any]] = []

    def call_function(
        self, tool_name: str, arguments: Dict[str, Any], agent_name: str = "Agent"
    ) -> Dict[str, Any]:
        """
        Call a function/tool with structured arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            agent_name: Name of the agent making the call

        Returns:
            Tool execution result

        AGNO Function Calling Flow:
        1. Validate tool exists
        2. Prepare arguments
        3. Call tool via registry
        4. Capture result
        5. Log call in history
        """
        logger.info(f"Agent {agent_name} calling tool: {tool_name}")

        tool = self.tool_registry.get_tool(tool_name)

        if not tool:
            error_result = {
                "status": "error",
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tool_registry.tools.keys()),
            }
            self.call_history.append(
                {
                    "agent": agent_name,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": error_result,
                }
            )
            return error_result

        # Call the tool
        result = tool.call(**arguments)

        # Log call in history
        self.call_history.append(
            {
                "agent": agent_name,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )

        return result

    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get complete function call history."""
        return self.call_history.copy()


def main():
    """
    Demonstration of AGNO tool integration and function calling.

    Reference Documentation:
    - https://docs.agno.com/agents/tools
    - https://docs.agno.com/mcp
    - https://github.com/agno-agi/agno/tree/main/cookbook
    """
    print("\n=== AGNO Tool Integration Demo ===\n")

    # Create tool registry
    registry = AGNOToolRegistry()

    # Register built-in tools
    print("1. Registering tools...")
    registry.register_tool(WebSearchTool())
    registry.register_tool(CodeExecutionTool())
    registry.register_tool(DatabaseTool())
    registry.register_tool(APICallTool())

    print(f"Registered {len(registry.tools)} tools\n")

    # List tools by category
    print("2. Tools by category:")
    for category in ToolCategory:
        tools = registry.get_tools_by_category(category)
        if tools:
            print(f"  {category.value}: {[t.definition.name for t in tools]}")

    # Create function caller
    print("\n3. Function calling examples...")
    caller = AGNOFunctionCaller(registry)

    # Call web search
    search_result = caller.call_function(
        tool_name="web_search",
        arguments={"query": "AGNO framework", "max_results": 5},
        agent_name="ResearchAgent",
    )
    print(f"\nWeb Search Result: {search_result['status']}")

    # Call code execution
    code_result = caller.call_function(
        tool_name="code_execution",
        arguments={"code": "print('Hello from AGNO agent')", "language": "python"},
        agent_name="DeveloperAgent",
    )
    print(f"Code Execution Result: {code_result['status']}")

    # Call API
    api_result = caller.call_function(
        tool_name="api_call",
        arguments={"url": "https://api.agno.com/info", "method": "GET"},
        agent_name="DataAgent",
    )
    print(f"API Call Result: {api_result['status']}")

    # Print call history
    print("\n4. Call History:")
    for call in caller.get_call_history():
        print(
            f"  - Agent: {call['agent']}, Tool: {call['tool']}, Status: {call['result']['status']}"
        )

    # Print tool definitions
    print("\n5. Tool Definitions (JSON):")
    print(json.dumps(registry.get_tool_definitions(), indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
