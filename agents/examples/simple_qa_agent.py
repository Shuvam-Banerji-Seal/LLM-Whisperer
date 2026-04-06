"""
AGNO Simple Q&A Agent Example

A practical example of building a basic question-answering agent
using the AGNO framework with streaming responses.

Author: Shuvam Banerji Seal
Source: https://docs.agno.com/first-agent
Source: https://docs.agno.com/agents/overview
"""

import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleQAAgent:
    """
    A basic Q&A agent that answers user questions.

    This agent demonstrates:
    - Simple agent initialization
    - Streaming response handling
    - Multi-turn conversation
    - Error handling

    Example:
        >>> agent = SimpleQAAgent()
        >>> agent.ask("What is machine learning?")
        >>> agent.ask("How does it differ from deep learning?")

    Production AGNO Code:
    ```python
    from agno.agent import Agent
    from agno.models.anthropic import Claude

    agent = Agent(
        name="QA Assistant",
        model=Claude(id="claude-3-5-sonnet-20241022"),
        instructions="You are a helpful Q&A assistant.",
        add_history_to_context=True,
        markdown=True,
    )

    agent.print_response("What is AGNO?", stream=True)
    ```
    """

    def __init__(
        self,
        name: str = "QAAgent",
        model: str = "claude-3-5-sonnet-20241022",
        instructions: Optional[str] = None,
    ):
        """
        Initialize the Q&A agent.

        Args:
            name: Agent identifier
            model: Model to use (default: Claude)
            instructions: System instructions for the agent
        """
        self.name = name
        self.model = model
        self.instructions = (
            instructions
            or "You are a helpful Q&A assistant. Answer questions accurately and concisely."
        )
        self.conversation_history = []

        logger.info(f"Initialized {name} with model {model}")

    def ask(self, question: str, stream: bool = True) -> str:
        """
        Ask the agent a question.

        Args:
            question: The question to ask
            stream: Whether to stream the response

        Returns:
            The agent's response

        AGNO Streaming Pattern:
        Agents can stream responses for real-time feedback to users.
        This is particularly useful for long-form content.
        """
        logger.info(f"Processing question: {question[:100]}")

        # Add question to history
        self.conversation_history.append({"role": "user", "content": question})

        # Simulate agent response
        # In real AGNO: response = agent.run(message=question)
        response = self._generate_response(question)

        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def _generate_response(self, question: str) -> str:
        """
        Generate a response to the question.

        In production, this would call the actual LLM via AGNO.
        """
        # Simulated responses based on question
        if "agno" in question.lower():
            return (
                "AGNO is a runtime for building agentic software. "
                "It provides agents, teams, and workflows for building AI systems at scale. "
                "AGNO includes stateful execution, tool integration, and production deployment."
            )
        elif "machine learning" in question.lower():
            return (
                "Machine learning is a field of AI where systems learn patterns from data "
                "without being explicitly programmed. It includes supervised learning, "
                "unsupervised learning, and reinforcement learning approaches."
            )
        elif "python" in question.lower():
            return (
                "Python is a popular programming language known for its simplicity and readability. "
                "It's widely used in data science, machine learning, and web development. "
                "Python has extensive libraries like NumPy, Pandas, and TensorFlow."
            )
        else:
            return (
                "This is a simulated Q&A response. "
                "In production, the AGNO agent would use an LLM like Claude to generate responses."
            )

    def get_conversation_history(self) -> list:
        """Get the conversation history."""
        return self.conversation_history.copy()

    def reset_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Cleared conversation history")

    def print_conversation(self) -> None:
        """Print the entire conversation."""
        print(f"\n=== Conversation with {self.name} ===\n")

        for message in self.conversation_history:
            role = message["role"].upper()
            content = message["content"]

            # Format long content
            if len(content) > 200:
                content = content[:200] + "..."

            print(f"{role}: {content}\n")


def main():
    """
    Example usage of the SimpleQAAgent.

    This demonstrates:
    - Creating an agent
    - Asking multiple questions
    - Managing conversation history
    - Viewing the conversation

    Reference Documentation:
    - https://docs.agno.com/first-agent
    - https://docs.agno.com/agents/overview
    - https://docs.agno.com/cookbook
    """
    print("\n=== AGNO Simple Q&A Agent Example ===\n")

    # Create the agent
    agent = SimpleQAAgent(
        name="HelpfulAssistant",
        model="claude-3-5-sonnet-20241022",
        instructions="You are a helpful assistant that answers questions about AGNO, AI, and programming.",
    )

    # Ask questions
    print("1. Asking about AGNO...")
    response1 = agent.ask("What is AGNO?")
    print(f"Response: {response1}\n")

    print("2. Asking about Machine Learning...")
    response2 = agent.ask("Tell me about machine learning")
    print(f"Response: {response2}\n")

    print("3. Follow-up question...")
    response3 = agent.ask("Can you explain deep learning?")
    print(f"Response: {response3}\n")

    # Show conversation history
    print("4. Conversation History:")
    agent.print_conversation()

    # Show conversation count
    history = agent.get_conversation_history()
    print(f"\n5. Summary:")
    print(f"   Total messages: {len(history)}")
    print(f"   User questions: {sum(1 for m in history if m['role'] == 'user')}")
    print(f"   Agent responses: {sum(1 for m in history if m['role'] == 'assistant')}")


if __name__ == "__main__":
    main()
