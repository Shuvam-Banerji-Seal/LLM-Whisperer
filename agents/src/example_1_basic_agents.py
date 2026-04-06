"""
Example 1: Basic Agent Setup and Conversation

This example demonstrates:
- Creating agent configurations
- Initializing agents with different LLM providers
- Managing conversation history
- Using the AgentFactory for specialized agents

Source: https://python.langchain.com/docs/modules/agents/
"""

from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_agent_basics import BasicAgent, AgentConfig, LLMProvider, AgentFactory


def example_1_basic_agent():
    """Create and use a basic agent."""
    print("=" * 60)
    print("Example 1: Basic Agent Setup")
    print("=" * 60)

    # Create agent configuration
    config = AgentConfig(
        model="gpt-4",
        provider=LLMProvider.OPENAI,
        temperature=0.7,
        max_tokens=2048,
        system_prompt="You are a helpful assistant specialized in answering questions.",
    )

    print(f"\nAgent Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Provider: {config.provider.value}")
    print(f"  Temperature: {config.temperature}")
    print(f"  System Prompt: {config.system_prompt[:50]}...")

    # Initialize agent
    try:
        agent = BasicAgent(config)
        print(f"\n✓ Agent initialized successfully")

        # Simulate conversation
        print("\n--- Simulated Conversation ---")
        user_input = "What is machine learning?"
        print(f"User: {user_input}")

        # Note: This would require actual API key in production
        print("Assistant: [Response would appear here with valid API key]")

        # Show conversation would be tracked
        history = agent.get_conversation_history()
        print(f"\nConversation history: {len(history)} messages tracked")

    except ImportError as e:
        print(f"⚠ LLM provider not installed: {e}")
        print("  Install with: pip install langchain-openai")


def example_2_agent_factory():
    """Demonstrate AgentFactory for specialized agents."""
    print("\n" + "=" * 60)
    print("Example 2: Using AgentFactory")
    print("=" * 60)

    print("\nAvailable pre-configured agents:")

    agents_info = [
        ("Research Agent", "Optimized for detailed, accurate research"),
        ("Creative Agent", "Optimized for creative writing and ideation"),
        ("Code Agent", "Optimized for programming tasks"),
    ]

    for agent_name, description in agents_info:
        print(f"  • {agent_name}: {description}")

    print("\nExample - Creating a Code Agent:")
    print("```python")
    print("code_agent = AgentFactory.create_code_agent()")
    print("response = code_agent.invoke('Write a Python function to reverse a string')")
    print("```")

    print("\nKey differences between agent types:")
    print("  • Research: temperature=0.3 (more focused, less creative)")
    print("  • Creativity: temperature=1.2 (more diverse, creative)")
    print("  • Code: temperature=0.2 (more precise, logical)")


def example_3_conversation_tracking():
    """Demonstrate conversation history tracking."""
    print("\n" + "=" * 60)
    print("Example 3: Conversation History Tracking")
    print("=" * 60)

    config = AgentConfig(
        model="gpt-4",
        provider=LLMProvider.OPENAI,
        system_prompt="You are a helpful assistant.",
    )

    # Note: Can't actually invoke without API key, but show the concept
    print("\nConversation Tracking Example:")
    print("""
The agent automatically tracks all messages:

1. User message is added
2. Agent processes message
3. Assistant response is added
4. Conversation history is maintained

Example output:
[
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI stands for Artificial Intelligence..."},
    {"role": "user", "content": "Tell me more about machine learning"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."}
]

Access with: agent.get_conversation_history()
Clear with: agent.clear_history()
    """)


def example_4_llm_providers():
    """Show different LLM provider configuration."""
    print("\n" + "=" * 60)
    print("Example 4: LLM Provider Configurations")
    print("=" * 60)

    providers = [
        {
            "name": "OpenAI",
            "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "install": "pip install langchain-openai",
        },
        {
            "name": "Anthropic",
            "models": ["claude-3-opus", "claude-3-sonnet-20240229", "claude-3-haiku"],
            "install": "pip install langchain-anthropic",
        },
        {
            "name": "Google",
            "models": ["gemini-pro", "gemini-1.5-pro"],
            "install": "pip install langchain-google-genai",
        },
        {
            "name": "Local (Ollama)",
            "models": ["llama2", "mistral", "neural-chat"],
            "install": "pip install langchain-ollama",
        },
    ]

    for provider in providers:
        print(f"\n{provider['name']}:")
        print(f"  Install: {provider['install']}")
        print(f"  Models: {', '.join(provider['models'][:2])}...")


def example_5_best_practices():
    """Show best practices for agent configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Best Practices")
    print("=" * 60)

    print("""
1. TEMPERATURE SETTINGS:
   - Factual/Research: 0.3-0.5 (focused, deterministic)
   - General Tasks: 0.7 (balanced)
   - Creative: 1.2+ (diverse, varied)

2. SYSTEM PROMPTS:
   - Clear role definition
   - Specific instructions
   - Output format guidance
   
   Example:
   "You are an expert Python programmer. Write clean, efficient code
    with comprehensive docstrings. Follow PEP 8 conventions."

3. MAX TOKENS:
   - Short responses: 256-512
   - Medium responses: 1024-2048
   - Long-form: 4096+
   
   Rule: Use minimum needed to reduce costs

4. ERROR HANDLING:
   - Wrap API calls in try-except
   - Log errors for debugging
   - Provide meaningful error messages

5. CONVERSATION MANAGEMENT:
   - Clear history periodically for long sessions
   - Use summary memory for context efficiency
   - Monitor total tokens used

6. API KEY MANAGEMENT:
   - Use environment variables
   - Never hardcode API keys
   - Rotate keys periodically
   
   Example:
   import os
   api_key = os.getenv("OPENAI_API_KEY")
""")


if __name__ == "__main__":
    print("\n")
    print("█" * 60)
    print("█  LangChain Agents Framework - Examples")
    print("█" * 60)

    example_1_basic_agent()
    example_2_agent_factory()
    example_3_conversation_tracking()
    example_4_llm_providers()
    example_5_best_practices()

    print("\n" + "=" * 60)
    print("For production use:")
    print("  1. Set API keys as environment variables")
    print("  2. Implement proper error handling")
    print("  3. Use LangSmith for tracing (set LANGSMITH_TRACING=true)")
    print("  4. Monitor token usage and costs")
    print("=" * 60 + "\n")
