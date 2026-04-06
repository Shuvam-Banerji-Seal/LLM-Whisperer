"""
AGNO Hello World Agent - Minimal Example

This is the simplest possible AGNO agent that demonstrates core concepts.
AGNO is an agentic AI framework for building production-ready AI agents with tools.

References:
- AGNO Framework: https://github.com/tobalo/ai-agent-hello-world
- Building Production-Ready Agents: https://medium.com/data-science-collective/building-production-ready-ai-agents-with-agno-a-comprehensive-engineering-guide-22db32413fdd
- Build AI Agents with Simple Code: https://medium.com/code-applied/build-ai-agents-tools-with-simple-code-5d6519c16e67

Author: Shuvam Banerji Seal
"""

# Requirements:
# pip install agno>=1.0.0
# pip install openai>=1.0.0  # or your preferred LLM provider

from typing import Optional


class SimpleAGNOAgent:
    """
    A minimal AGNO agent that demonstrates:
    - Agent initialization with a model
    - Simple tool definition
    - Agent execution with input
    """

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the AGNO agent.

        Args:
            model_name: The model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.model_name = model_name
        self.api_key = api_key

        # In a real implementation, you would initialize:
        # from agno import Agent
        # self.agent = Agent(model=model_name, api_key=api_key)

    def define_tools(self):
        """
        Define simple tools that the agent can use.
        Tools are functions that extend the agent's capabilities.
        """
        tools = {
            "add": self._add,
            "multiply": self._multiply,
            "get_info": self._get_info,
        }
        return tools

    @staticmethod
    def _add(a: float, b: float) -> float:
        """Simple addition tool."""
        return a + b

    @staticmethod
    def _multiply(a: float, b: float) -> float:
        """Simple multiplication tool."""
        return a * b

    @staticmethod
    def _get_info() -> str:
        """Get information about this agent."""
        return "I am a simple AGNO hello world agent that can perform basic math operations."

    def run(self, instruction: str) -> str:
        """
        Run the agent with a given instruction.

        Args:
            instruction: The task/question for the agent

        Returns:
            The agent's response
        """
        print(f"\n📤 User Input: {instruction}")

        # In a real implementation:
        # response = self.agent.run(instruction, tools=self.define_tools())
        # return response

        # For this hello world example, we'll simulate a response
        response = f"I received your instruction: '{instruction}'. In a full implementation, I would use my tools to help you."
        print(f"🤖 Agent Response: {response}")
        return response


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AGNO Hello World Agent - Minimal Example")
    print("=" * 70)

    # Initialize the agent
    agent = SimpleAGNOAgent(model_name="gpt-4")

    print("\n📋 Available Tools:")
    for tool_name in agent.define_tools():
        print(f"  - {tool_name}")

    # Example 1: Simple math request
    print("\n" + "=" * 70)
    print("Example 1: Simple Math Operation")
    print("=" * 70)
    agent.run("What is 5 plus 3?")

    # Example 2: Information request
    print("\n" + "=" * 70)
    print("Example 2: Information Request")
    print("=" * 70)
    agent.run("Tell me about yourself")

    # Example 3: Complex operation
    print("\n" + "=" * 70)
    print("Example 3: Complex Operation")
    print("=" * 70)
    agent.run("Calculate 10 multiplied by 2 plus 5")

    print("\n" + "=" * 70)
    print("Full AGNO Implementation")
    print("=" * 70)
    print("""
To use the full AGNO framework:

1. Install AGNO: pip install agno

2. Create an agent with models:
   from agno import Agent
   from agno.models import OpenAIChat
   
   agent = Agent(
       name="MathHelper",
       model=OpenAIChat(id="gpt-4"),
       tools=[add_tool, multiply_tool],
       show_tool_calls=True,
   )

3. Run the agent:
   agent.print_response("Calculate 5 + 3")

For production deployments, AGNO provides:
- Tool management and validation
- Multi-step reasoning
- State persistence
- Error handling and retries
- Monitoring and logging

See: https://github.com/tobalo/ai-agent-hello-world
    """)
