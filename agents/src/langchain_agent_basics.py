"""
LangChain Agent Basics Module

This module provides foundational implementations for creating and managing LangChain agents.
Includes agent initialization, LLM provider integration, and basic execution patterns.

Author: Shuvam Banerji Seal
Date: 2026-04-06

Source: https://python.langchain.com/api_reference
Documentation: https://python.langchain.com/docs/modules/agents/
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers for agent initialization."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass
class AgentConfig:
    """Configuration dataclass for agent initialization.

    Attributes:
        model: Model identifier (e.g., "gpt-4", "claude-3-sonnet")
        provider: LLM provider type
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in response
        system_prompt: System instruction for the agent
        streaming: Whether to enable streaming responses
        api_key: API key for the provider (optional)
        base_url: Custom base URL for API (optional)
    """

    model: str
    provider: LLMProvider
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = "You are a helpful assistant."
    streaming: bool = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLMInitializer:
    """Initialize Language Models from various providers.

    This class handles the creation of LLM instances with proper
    configuration and error handling.

    Example:
        >>> config = AgentConfig(
        ...     model="claude-3-sonnet-20240229",
        ...     provider=LLMProvider.ANTHROPIC
        ... )
        >>> initializer = LLMInitializer(config)
        >>> llm = initializer.initialize()
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize LLM initializer with configuration.

        Args:
            config: AgentConfig instance with model parameters

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        logger.info(f"LLMInitializer created for {config.provider.value} provider")

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if self.config.temperature < 0.0 or self.config.temperature > 2.0:
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got {self.config.temperature}"
            )

        if self.config.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be positive, got {self.config.max_tokens}"
            )

        if not self.config.model:
            raise ValueError("model parameter cannot be empty")

        logger.debug("Configuration validation passed")

    def initialize(self) -> Any:
        """Initialize and return LLM instance based on provider.

        Returns:
            Initialized LLM instance from langchain

        Raises:
            ImportError: If required provider module is not installed
            ValueError: If provider is not supported

        Example:
            >>> from langchain_anthropic import ChatAnthropic
            >>> llm = initializer.initialize()
            >>> response = llm.invoke("What is machine learning?")
        """
        try:
            if self.config.provider == LLMProvider.OPENAI:
                return self._initialize_openai()
            elif self.config.provider == LLMProvider.ANTHROPIC:
                return self._initialize_anthropic()
            elif self.config.provider == LLMProvider.GOOGLE:
                return self._initialize_google()
            elif self.config.provider == LLMProvider.LOCAL:
                return self._initialize_local()
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
        except ImportError as e:
            logger.error(f"Failed to import LLM provider: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def _initialize_openai(self) -> Any:
        """Initialize OpenAI ChatGPT model.

        Requires: pip install langchain-openai

        Returns:
            ChatOpenAI instance
        """
        try:
            from langchain_openai import ChatOpenAI

            logger.info(f"Initializing OpenAI model: {self.config.model}")

            llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key,
                streaming=self.config.streaming,
            )
            logger.debug("OpenAI LLM initialized successfully")
            return llm
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. "
                "Install with: pip install langchain-openai"
            )

    def _initialize_anthropic(self) -> Any:
        """Initialize Anthropic Claude model.

        Requires: pip install langchain-anthropic

        Returns:
            ChatAnthropic instance
        """
        try:
            from langchain_anthropic import ChatAnthropic

            logger.info(f"Initializing Anthropic model: {self.config.model}")

            llm = ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key,
                streaming=self.config.streaming,
            )
            logger.debug("Anthropic LLM initialized successfully")
            return llm
        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed. "
                "Install with: pip install langchain-anthropic"
            )

    def _initialize_google(self) -> Any:
        """Initialize Google Gemini model.

        Requires: pip install langchain-google-genai

        Returns:
            ChatGoogleGenerativeAI instance
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            logger.info(f"Initializing Google model: {self.config.model}")

            llm = ChatGoogleGenerativeAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                google_api_key=self.config.api_key,
                streaming=self.config.streaming,
            )
            logger.debug("Google LLM initialized successfully")
            return llm
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. "
                "Install with: pip install langchain-google-genai"
            )

    def _initialize_local(self) -> Any:
        """Initialize local/self-hosted model.

        Requires: pip install langchain-ollama or similar

        Returns:
            Local LLM instance
        """
        try:
            from langchain_ollama import OllamaLLM

            logger.info(f"Initializing local model: {self.config.model}")

            llm = OllamaLLM(
                model=self.config.model,
                base_url=self.config.base_url or "http://localhost:11434",
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens,
            )
            logger.debug("Local LLM initialized successfully")
            return llm
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. "
                "Install with: pip install langchain-ollama"
            )


class BasicAgent:
    """Basic LangChain agent wrapper with common functionality.

    This class provides a simplified interface for creating and managing
    basic LangChain agents with standard operations.

    Example:
        >>> config = AgentConfig(
        ...     model="gpt-4",
        ...     provider=LLMProvider.OPENAI
        ... )
        >>> agent = BasicAgent(config)
        >>> response = agent.invoke("Explain quantum computing")
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize basic agent with configuration.

        Args:
            config: AgentConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.initializer = LLMInitializer(config)
        self.llm = self.initializer.initialize()
        self.conversation_history: List[Dict[str, str]] = []
        logger.info(f"BasicAgent initialized with model: {config.model}")

    def invoke(self, user_input: str, **kwargs: Any) -> str:
        """Send a message to the agent and get response.

        Args:
            user_input: User's input message
            **kwargs: Additional arguments to pass to LLM

        Returns:
            Agent's response as string

        Raises:
            RuntimeError: If LLM invocation fails

        Example:
            >>> response = agent.invoke("What is the capital of France?")
            >>> print(response)
        """
        try:
            logger.info(f"Invoking agent with input: {user_input[:100]}...")

            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Invoke LLM
            response = self.llm.invoke(user_input, **kwargs)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Add response to history
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )

            logger.debug(f"Agent response: {response_text[:100]}...")
            return response_text
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            raise RuntimeError(f"Failed to invoke agent: {e}") from e

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history.

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        return self.conversation_history.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt.

        Args:
            new_prompt: New system prompt string
        """
        self.config.system_prompt = new_prompt
        logger.info(f"System prompt updated")

    def get_config(self) -> AgentConfig:
        """Get current agent configuration.

        Returns:
            Current AgentConfig instance
        """
        return self.config


class AgentFactory:
    """Factory for creating pre-configured agents for common use cases.

    This class provides convenient methods to create agents optimized
    for specific tasks without manual configuration.

    Example:
        >>> agent = AgentFactory.create_research_agent()
        >>> response = agent.invoke("Research quantum computing")
    """

    @staticmethod
    def create_openai_agent(
        model: str = "gpt-4",
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> BasicAgent:
        """Create an OpenAI-based agent.

        Args:
            model: OpenAI model identifier
            system_prompt: Custom system prompt
            api_key: OpenAI API key

        Returns:
            Configured BasicAgent instance
        """
        config = AgentConfig(
            model=model,
            provider=LLMProvider.OPENAI,
            system_prompt=system_prompt or "You are a helpful assistant.",
            api_key=api_key,
        )
        return BasicAgent(config)

    @staticmethod
    def create_anthropic_agent(
        model: str = "claude-3-sonnet-20240229",
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> BasicAgent:
        """Create an Anthropic Claude-based agent.

        Args:
            model: Anthropic model identifier
            system_prompt: Custom system prompt
            api_key: Anthropic API key

        Returns:
            Configured BasicAgent instance
        """
        config = AgentConfig(
            model=model,
            provider=LLMProvider.ANTHROPIC,
            system_prompt=system_prompt or "You are a helpful assistant.",
            api_key=api_key,
        )
        return BasicAgent(config)

    @staticmethod
    def create_research_agent() -> BasicAgent:
        """Create an agent optimized for research tasks.

        Returns:
            BasicAgent configured for research
        """
        config = AgentConfig(
            model="gpt-4",
            provider=LLMProvider.OPENAI,
            temperature=0.3,
            system_prompt=(
                "You are a research assistant. Provide detailed, "
                "well-sourced information. Be accurate and thorough."
            ),
        )
        return BasicAgent(config)

    @staticmethod
    def create_creative_agent() -> BasicAgent:
        """Create an agent optimized for creative tasks.

        Returns:
            BasicAgent configured for creative work
        """
        config = AgentConfig(
            model="gpt-4",
            provider=LLMProvider.OPENAI,
            temperature=1.2,
            system_prompt=(
                "You are a creative assistant. Generate original ideas, "
                "stories, and content. Be imaginative and innovative."
            ),
        )
        return BasicAgent(config)

    @staticmethod
    def create_code_agent() -> BasicAgent:
        """Create an agent optimized for code generation.

        Returns:
            BasicAgent configured for coding
        """
        config = AgentConfig(
            model="gpt-4",
            provider=LLMProvider.OPENAI,
            temperature=0.2,
            system_prompt=(
                "You are an expert programmer. Provide clean, efficient, "
                "well-documented code. Follow best practices and standards."
            ),
        )
        return BasicAgent(config)


if __name__ == "__main__":
    # Example usage
    print("LangChain Agent Basics Module")
    print("=" * 50)

    # Create a basic agent configuration
    config = AgentConfig(
        model="gpt-4",
        provider=LLMProvider.OPENAI,
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
    )

    print(f"Configuration created: {config}")
    print(f"Provider: {config.provider.value}")
    print(f"Temperature: {config.temperature}")
