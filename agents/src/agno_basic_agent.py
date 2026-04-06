"""
AGNO Basic Agent Implementation

This module demonstrates the fundamental patterns for building agents with the AGNO framework.
It provides a simple yet complete example of agent initialization, configuration, and response handling.

Author: Shuvam Banerji Seal
Source: https://docs.agno.com/first-agent
Source: https://github.com/agno-agi/agno
"""

from typing import Optional, List, Dict, Any
import logging

# Configure logging for agent operations
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AGNOBasicAgent:
    """
    A basic AGNO agent demonstrating core functionality.

    This class shows how to:
    - Initialize an agent with a language model
    - Configure tools and capabilities
    - Handle streaming responses
    - Manage agent state and configuration

    AGNO Framework Overview:
        AGNO is a runtime for building agentic software with:
        - Stateful agent execution
        - Tool integration and function calling
        - Multi-agent orchestration
        - Production-ready deployment
        - Session and memory management

    Example:
        >>> agent = AGNOBasicAgent()
        >>> response = agent.run_query("What is AGNO?")
        >>> print(response)
    """

    def __init__(
        self,
        name: str = "BasicAgent",
        model_provider: str = "anthropic",
        model_id: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[str]] = None,
        enable_streaming: bool = True,
        add_history_to_context: bool = True,
        markdown_format: bool = True,
    ):
        """
        Initialize a basic AGNO agent.

        Args:
            name: Agent identifier and display name
            model_provider: LLM provider (anthropic, openai, etc.)
            model_id: Specific model identifier
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response tokens
            tools: List of tool names to enable
            enable_streaming: Enable streaming responses
            add_history_to_context: Include chat history in context
            markdown_format: Format responses as markdown

        Note:
            In real AGNO implementations, you would use:
            ```python
            from agno.agent import Agent
            from agno.models.anthropic import Claude

            agent = Agent(
                name=name,
                model=Claude(id=model_id),
                tools=[...],
                add_history_to_context=add_history_to_context,
                markdown=markdown_format,
            )
            ```
        """
        self.name = name
        self.model_provider = model_provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
        self.enable_streaming = enable_streaming
        self.add_history_to_context = add_history_to_context
        self.markdown_format = markdown_format

        # Initialize agent configuration
        self.config: Dict[str, Any] = {
            "name": name,
            "model": {
                "provider": model_provider,
                "id": model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "capabilities": {
                "streaming": enable_streaming,
                "history_context": add_history_to_context,
                "markdown": markdown_format,
                "tools": tools,
            },
        }

        logger.info(f"Initialized {self.name} agent with model {self.model_id}")

    def configure_model(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
    ) -> None:
        """
        Configure model parameters.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum token limit
            top_p: Nucleus sampling parameter

        AGNO Pattern:
            Model configuration is typically set during Agent initialization.
            These parameters control the behavior of the underlying LLM.
        """
        if temperature is not None:
            self.temperature = temperature
            self.config["model"]["temperature"] = temperature
            logger.info(f"Updated temperature to {temperature}")

        if max_tokens is not None:
            self.max_tokens = max_tokens
            self.config["model"]["max_tokens"] = max_tokens
            logger.info(f"Updated max_tokens to {max_tokens}")

        self.config["model"]["top_p"] = top_p
        logger.info(f"Set top_p to {top_p}")

    def add_tools(self, tool_names: List[str]) -> None:
        """
        Add tools to the agent's capabilities.

        Args:
            tool_names: List of tool identifiers to add

        AGNO Pattern:
            Tools enable agents to:
            - Call external APIs
            - Execute functions
            - Perform system operations
            - Access external knowledge bases

        Common AGNO tools:
            - coding: Execute and test code
            - websearch: Search the internet
            - database: Query databases
            - mcp: Model Context Protocol tools
        """
        self.tools.extend(tool_names)
        self.config["capabilities"]["tools"] = list(set(self.tools))
        logger.info(f"Added tools: {tool_names}")

    def run_query(self, query: str) -> str:
        """
        Execute a query with the agent.

        Args:
            query: User input/question for the agent

        Returns:
            Agent's response as a string

        AGNO Execution Flow:
            1. Parse user input
            2. Check if tools are needed
            3. Execute tool calls if necessary
            4. Synthesize response
            5. Stream or return result

        Example:
            >>> agent = AGNOBasicAgent()
            >>> response = agent.run_query("Build a todo app with tests")
            >>> print(response)
        """
        logger.info(f"Processing query: {query}")

        # Validate input
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        # In real AGNO usage:
        # response = agent.run(message=query)
        # return response.content

        # Demonstration response
        response = f"""
Agent: {self.name}
Model: {self.model_id}
Query: {query}

Response (simulated):
The agent would process this query using the {self.model_id} model.
If tools are configured ({len(self.tools)} tools available), 
the agent would call them as needed to provide accurate information.

Streaming: {"Enabled" if self.enable_streaming else "Disabled"}
History Context: {"Enabled" if self.add_history_to_context else "Disabled"}
"""
        return response

    def get_config(self) -> Dict[str, Any]:
        """
        Retrieve the current agent configuration.

        Returns:
            Dictionary containing all agent settings

        AGNO Pattern:
            Configuration management is important for:
            - Debugging agent behavior
            - Monitoring in production
            - A/B testing different configurations
            - Logging and auditing
        """
        return self.config.copy()

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get human-readable agent information.

        Returns:
            Dictionary with agent metadata
        """
        return {
            "name": self.name,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "tools_count": len(self.tools),
            "tools": self.tools,
            "capabilities": {
                "streaming": self.enable_streaming,
                "history": self.add_history_to_context,
                "markdown": self.markdown_format,
            },
        }

    def reset(self) -> None:
        """
        Reset agent state.

        AGNO Pattern:
            Resetting clears:
            - Message history
            - Session state
            - Temporary variables
            - But preserves configuration
        """
        logger.info(f"Resetting agent {self.name}")
        # In real AGNO: agent.reset_session()


def main():
    """
    Demonstration of AGNO Basic Agent usage.

    This function shows typical workflows for:
    - Creating an agent
    - Configuring tools
    - Processing queries
    - Managing agent state

    Reference Documentation:
        - https://docs.agno.com/first-agent
        - https://docs.agno.com/agents/overview
        - https://github.com/agno-agi/agno/tree/main/cookbook
    """
    # Initialize agent
    agent = AGNOBasicAgent(
        name="DemoAgent",
        model_provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=2048,
        tools=["websearch", "coding"],
        enable_streaming=True,
        add_history_to_context=True,
    )

    print("\n=== AGNO Basic Agent Demo ===\n")
    print("Agent Configuration:")
    print(agent.get_agent_info())

    # Add more tools
    print("\nAdding additional tools...")
    agent.add_tools(["database", "mcp"])

    # Execute queries
    print("\nExecuting sample query...")
    response = agent.run_query("What is the current state of AI agents in 2026?")
    print(response)

    # Get configuration
    print("\nAgent Full Configuration:")
    import json

    print(json.dumps(agent.get_config(), indent=2))


if __name__ == "__main__":
    main()
