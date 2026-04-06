"""
LangChain Hello World Agent - Minimal Example

This is the simplest possible LangChain agent that demonstrates core concepts.
LangChain is a framework for developing applications powered by language models.

References:
- LangChain Official Docs: https://docs.langchain.com/
- Build Your First AI Agent with LangChain: https://langchain-tutorials.github.io/build-first-ai-agent-langchain-2026/
- LangChain Python Tutorial 2026: https://blog.jetbrains.com/pycharm/2026/02/langchain-tutorial-2026/
- Building AI Agents with LangChain Tutorial 2026: https://www.ai-agentsplus.com/blog/building-ai-agents-with-langchain-tutorial-2026

Author: Shuvam Banerji Seal
"""

# Requirements:
# pip install langchain>=0.1.0
# pip install openai>=1.0.0
# pip install langchain-community>=0.1.0

from typing import Optional, Any, Union
from functools import wraps


class SimpleLangChainAgent:
    """
    A minimal LangChain agent that demonstrates:
    - Tool creation and registration
    - Agent initialization with a language model
    - Tool execution and response generation
    """

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the LangChain agent.

        Args:
            model_name: The model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.tools = {}
        self.memory = []

        # In a real implementation:
        # from langchain_openai import ChatOpenAI
        # self.llm = ChatOpenAI(model=model_name, api_key=api_key)

    def tool(self, name: str, description: str):
        """
        Decorator to register a function as a tool.

        Args:
            name: The name of the tool
            description: Description of what the tool does
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result

            self.tools[name] = {
                "func": wrapper,
                "description": description,
                "original_func": func,
            }
            return wrapper

        return decorator

    def define_tools(self):
        """
        Define simple tools that the agent can use.
        """

        # Math tools
        @self.tool("add", "Adds two numbers together")
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        @self.tool("subtract", "Subtracts the second number from the first")
        def subtract(a: float, b: float) -> float:
            """Subtract two numbers."""
            return a - b

        @self.tool("multiply", "Multiplies two numbers together")
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b

        @self.tool("divide", "Divides the first number by the second")
        def divide(a: float, b: float) -> Union[float, str]:
            """Divide two numbers."""
            if b == 0:
                return "Error: Cannot divide by zero"
            return a / b

        @self.tool("get_agent_info", "Returns information about the agent")
        def get_info() -> str:
            """Get information about this agent."""
            return "I am a LangChain hello world agent that can perform math operations and provide information."

        return self.tools

    def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Execute a registered tool.

        Args:
            tool_name: The name of the tool to execute
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            The result of the tool execution
        """
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            result = self.tools[tool_name]["func"](*args, **kwargs)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def run(self, instruction: str) -> str:
        """
        Run the agent with a given instruction.

        Args:
            instruction: The task/question for the agent

        Returns:
            The agent's response
        """
        print(f"\n📤 User Input: {instruction}")

        # In a real implementation, LangChain would:
        # 1. Send the instruction to the LLM
        # 2. The LLM would decide which tools to use
        # 3. Execute the tools
        # 4. Generate a response based on tool results

        # For this hello world example, we'll simulate a response
        response = f"I received your instruction: '{instruction}'. "
        response += f"I have {len(self.tools)} tools available: {', '.join(self.tools.keys())}. "
        response += "In a full implementation, I would use my tools and language model to help you."

        # Store in memory
        self.memory.append({"input": instruction, "output": response})

        print(f"🤖 Agent Response: {response}")
        return response

    def get_memory(self):
        """Get the agent's conversation memory."""
        return self.memory


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LangChain Hello World Agent - Minimal Example")
    print("=" * 70)

    # Initialize the agent and define tools
    agent = SimpleLangChainAgent(model_name="gpt-4")
    agent.define_tools()

    print("\n📋 Available Tools:")
    for tool_name, tool_info in agent.tools.items():
        print(f"  - {tool_name}: {tool_info['description']}")

    # Example 1: Simple math request
    print("\n" + "=" * 70)
    print("Example 1: Simple Math Operation")
    print("=" * 70)
    result = agent.execute_tool("add", 5, 3)
    print(f"5 + 3 = {result}")

    agent.run("What is 5 plus 3?")

    # Example 2: Information request
    print("\n" + "=" * 70)
    print("Example 2: Information Request")
    print("=" * 70)
    agent.run("Tell me about yourself")

    # Example 3: Multiple operations
    print("\n" + "=" * 70)
    print("Example 3: Multiple Operations")
    print("=" * 70)
    result1 = agent.execute_tool("multiply", 10, 2)
    result2 = agent.execute_tool("add", result1, 5)
    print(f"(10 * 2) + 5 = {result2}")

    # Example 4: Error handling
    print("\n" + "=" * 70)
    print("Example 4: Error Handling")
    print("=" * 70)
    result = agent.execute_tool("divide", 10, 0)
    print(f"10 / 0 = {result}")

    print("\n" + "=" * 70)
    print("Full LangChain Implementation")
    print("=" * 70)
    print("""
To use the full LangChain framework:

1. Install LangChain: pip install langchain langchain-openai

2. Create an agent with tools:
   from langchain_openai import ChatOpenAI
   from langchain.agents import create_openai_functions_agent, AgentExecutor
   from langchain.tools import tool
   
   @tool
   def add(a: float, b: float) -> float:
       \"\"\"Adds two numbers\"\"\"
       return a + b
   
   llm = ChatOpenAI(model="gpt-4")
   tools = [add]
   agent = create_openai_functions_agent(llm, tools, prompt)
   executor = AgentExecutor(agent=agent, tools=tools)

3. Run the agent:
   executor.invoke({"input": "What is 5 plus 3?"})

Key Features of LangChain:
- Tool creation with @tool decorator
- Automatic tool schema generation
- Memory management (short-term and long-term)
- Chain composition
- Multiple agent types (ReAct, OpenAI Functions, etc.)
- Integration with various LLMs and tools
- Prompt templating
- Document loading and processing

See: https://docs.langchain.com/
    """)

    # Display conversation memory
    print("\n" + "=" * 70)
    print("Conversation Memory")
    print("=" * 70)
    for i, exchange in enumerate(agent.get_memory(), 1):
        print(f"\nExchange {i}:")
        print(f"  Input:  {exchange['input']}")
        print(f"  Output: {exchange['output'][:100]}...")
