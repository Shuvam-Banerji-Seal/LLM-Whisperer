"""
LangChain Memory Systems Module

This module provides comprehensive memory implementations for LangChain agents.
Includes various memory strategies: buffer, summary, entity, vector, and custom backends.

Author: Shuvam Banerji Seal
Date: 2026-04-06

Source: https://python.langchain.com/api_reference
Documentation: https://blog.langchain.com/how-we-built-agent-builders-memory-system/
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message role enumeration for conversation tracking."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """Represents a single message in conversation.

    Attributes:
        role: Message role (user, assistant, system)
        content: Message content text
        timestamp: When message was created
        metadata: Additional message metadata
    """

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary.

        Returns:
            Dictionary representation of message
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_langchain_format(self) -> Dict[str, str]:
        """Convert to LangChain message format.

        Returns:
            LangChain-compatible message format
        """
        return {"role": self.role.value, "content": self.content}


class BaseMemory(ABC):
    """Abstract base class for memory implementations.

    All memory systems must implement these core methods.
    """

    @abstractmethod
    def add_message(
        self, role: MessageRole, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """Add a message to memory.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        pass

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """Get all messages from memory.

        Returns:
            List of Message objects
        """
        pass

    @abstractmethod
    def get_context(self) -> str:
        """Get memory context for LLM.

        Returns:
            Formatted context string
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from memory."""
        pass


class ConversationBufferMemory(BaseMemory):
    """Simple buffer that stores all conversation messages.

    This is the most straightforward memory implementation, storing
    all messages without any reduction or summarization.

    Example:
        >>> memory = ConversationBufferMemory()
        >>> memory.add_message(MessageRole.USER, "Hello!")
        >>> memory.add_message(MessageRole.ASSISTANT, "Hi there!")
        >>> context = memory.get_context()
    """

    def __init__(self, max_messages: Optional[int] = None) -> None:
        """Initialize buffer memory.

        Args:
            max_messages: Maximum messages to retain (None = unlimited)
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages
        logger.info(f"ConversationBufferMemory initialized (max={max_messages})")

    def add_message(
        self, role: MessageRole, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """Add message to buffer.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        message = Message(role, content, metadata=metadata or {})
        self.messages.append(message)

        # Enforce max messages limit
        if self.max_messages and len(self.messages) > self.max_messages:
            removed = self.messages.pop(0)
            logger.debug(f"Removed oldest message: {removed.role.value}")

        logger.debug(f"Message added: {role.value}")

    def get_messages(self) -> List[Message]:
        """Get all buffered messages.

        Returns:
            List of all messages
        """
        return self.messages.copy()

    def get_context(self) -> str:
        """Format messages as context string.

        Returns:
            Formatted conversation context
        """
        context_lines = []
        for msg in self.messages:
            context_lines.append(f"{msg.role.value.upper()}: {msg.content}")
        return "\n".join(context_lines)

    def clear(self) -> None:
        """Clear all messages from buffer."""
        count = len(self.messages)
        self.messages.clear()
        logger.info(f"Cleared {count} messages from buffer")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        total_tokens = sum(len(msg.content.split()) for msg in self.messages)
        user_msgs = sum(1 for msg in self.messages if msg.role == MessageRole.USER)
        assistant_msgs = sum(
            1 for msg in self.messages if msg.role == MessageRole.ASSISTANT
        )

        return {
            "total_messages": len(self.messages),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "total_tokens_approx": total_tokens,
            "max_messages": self.max_messages,
        }


class ConversationSummaryMemory(BaseMemory):
    """Memory that periodically summarizes conversation history.

    This memory type keeps recent messages and summarizes older ones
    to maintain context while reducing token usage.

    Example:
        >>> memory = ConversationSummaryMemory(keep_recent=10)
        >>> memory.add_message(MessageRole.USER, "What is AI?")
        >>> memory.add_message(MessageRole.ASSISTANT, "AI is...")
        >>> summary = memory.get_context()  # Includes recent + summary
    """

    def __init__(
        self, keep_recent: int = 5, summarizer: Optional[callable] = None
    ) -> None:
        """Initialize summary memory.

        Args:
            keep_recent: Number of recent messages to keep
            summarizer: Optional function to summarize messages
        """
        self.messages: List[Message] = []
        self.keep_recent = keep_recent
        self.summarizer = summarizer or self._default_summarizer
        self.summary: Optional[str] = None
        logger.info(
            f"ConversationSummaryMemory initialized (keep_recent={keep_recent})"
        )

    def _default_summarizer(self, messages: List[Message]) -> str:
        """Default summarization strategy.

        Args:
            messages: Messages to summarize

        Returns:
            Summary string
        """
        if not messages:
            return ""

        # Create a simple summary from first and last messages
        summary_parts = []
        if messages:
            summary_parts.append(f"Initial topic: {messages[0].content[:100]}")
        if len(messages) > 1:
            summary_parts.append(f"Recent progress: {messages[-1].content[:100]}")

        return ". ".join(summary_parts)

    def add_message(
        self, role: MessageRole, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """Add message and trigger summarization if needed.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        message = Message(role, content, metadata=metadata or {})
        self.messages.append(message)

        # Check if summarization is needed
        if len(self.messages) > self.keep_recent:
            self._summarize()

        logger.debug(f"Message added: {role.value}")

    def _summarize(self) -> None:
        """Summarize older messages, keep recent ones."""
        if len(self.messages) <= self.keep_recent:
            return

        # Separate into old and recent
        to_summarize = self.messages[: -self.keep_recent]
        recent = self.messages[-self.keep_recent :]

        # Create summary
        self.summary = self.summarizer(to_summarize)
        self.messages = recent

        logger.info(f"Summary created, kept {len(recent)} recent messages")

    def get_messages(self) -> List[Message]:
        """Get recent messages.

        Returns:
            List of recent messages
        """
        return self.messages.copy()

    def get_context(self) -> str:
        """Get context including summary and recent messages.

        Returns:
            Formatted context with summary and recent messages
        """
        context_lines = []

        if self.summary:
            context_lines.append(f"[SUMMARY]\n{self.summary}\n")

        context_lines.append("[RECENT MESSAGES]")
        for msg in self.messages:
            context_lines.append(f"{msg.role.value.upper()}: {msg.content}")

        return "\n".join(context_lines)

    def clear(self) -> None:
        """Clear all messages and summary."""
        self.messages.clear()
        self.summary = None
        logger.info("Summary memory cleared")


class EntityMemory(BaseMemory):
    """Memory that tracks entities and their relationships.

    This memory type extracts and tracks entities mentioned in conversation,
    along with their attributes and relationships.

    Example:
        >>> memory = EntityMemory()
        >>> memory.add_message(MessageRole.USER, "Alice is 30 years old")
        >>> memory.add_entity("Alice", {"age": "30"})
        >>> entities = memory.get_entities()
    """

    def __init__(self) -> None:
        """Initialize entity memory."""
        self.messages: List[Message] = []
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Tuple[str, str, str]] = []
        logger.info("EntityMemory initialized")

    def add_message(
        self, role: MessageRole, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """Add message to memory.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        message = Message(role, content, metadata=metadata or {})
        self.messages.append(message)
        logger.debug(f"Message added: {role.value}")

    def add_entity(
        self, name: str, attributes: Dict[str, Any], description: Optional[str] = None
    ) -> None:
        """Add or update an entity.

        Args:
            name: Entity name
            attributes: Entity attributes
            description: Entity description
        """
        if name not in self.entities:
            self.entities[name] = {}

        self.entities[name].update(attributes)
        if description:
            self.entities[name]["description"] = description

        logger.debug(f"Entity added/updated: {name}")

    def add_relationship(self, entity1: str, relationship: str, entity2: str) -> None:
        """Add a relationship between entities.

        Args:
            entity1: First entity
            relationship: Relationship description
            entity2: Second entity
        """
        self.relationships.append((entity1, relationship, entity2))
        logger.debug(f"Relationship added: {entity1} {relationship} {entity2}")

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Get entity details.

        Args:
            name: Entity name

        Returns:
            Entity attributes or None if not found
        """
        return self.entities.get(name)

    def get_entities(self) -> Dict[str, Dict[str, Any]]:
        """Get all entities.

        Returns:
            Dictionary of all entities and their attributes
        """
        return self.entities.copy()

    def get_relationships(self) -> List[Tuple[str, str, str]]:
        """Get all entity relationships.

        Returns:
            List of (entity1, relationship, entity2) tuples
        """
        return self.relationships.copy()

    def get_messages(self) -> List[Message]:
        """Get all messages.

        Returns:
            List of messages
        """
        return self.messages.copy()

    def get_context(self) -> str:
        """Get context with entities and relationships.

        Returns:
            Formatted context
        """
        context_lines = ["[ENTITIES]"]

        for entity_name, attrs in self.entities.items():
            context_lines.append(f"- {entity_name}: {attrs}")

        if self.relationships:
            context_lines.append("\n[RELATIONSHIPS]")
            for entity1, rel, entity2 in self.relationships:
                context_lines.append(f"- {entity1} {rel} {entity2}")

        return "\n".join(context_lines)

    def clear(self) -> None:
        """Clear all entities and relationships."""
        self.messages.clear()
        self.entities.clear()
        self.relationships.clear()
        logger.info("Entity memory cleared")


class VectorMemory(BaseMemory):
    """Memory using vector embeddings for semantic search.

    This memory type stores messages as embeddings and retrieves
    semantically similar messages based on queries.

    Example:
        >>> memory = VectorMemory()
        >>> memory.add_message(MessageRole.USER, "Tell me about Python")
        >>> similar = memory.search("programming language")
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        """Initialize vector memory.

        Args:
            embedding_dim: Embedding dimension
        """
        self.messages: List[Message] = []
        self.embeddings: List[List[float]] = []
        self.embedding_dim = embedding_dim
        logger.info(f"VectorMemory initialized (dim={embedding_dim})")

    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add message with optional embedding.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
            embedding: Optional pre-computed embedding
        """
        message = Message(role, content, metadata=metadata or {})
        self.messages.append(message)

        if embedding is None:
            # Simple hash-based "embedding" for demo
            embedding = self._compute_embedding(content)

        self.embeddings.append(embedding)
        logger.debug(f"Message added with embedding: {role.value}")

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute simple embedding from text.

        In production, use proper embedding models like Sentence Transformers.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Simple hash-based embedding for demonstration
        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        # Generate pseudo-random embedding
        embedding = []
        for i in range(self.embedding_dim):
            embedding.append((hash_int >> i) % 1000 / 1000.0)

        return embedding

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Message, float]]:
        """Search for semantically similar messages.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (message, similarity_score) tuples
        """
        if not self.messages:
            return []

        query_embedding = self._compute_embedding(query)

        # Compute similarities (simple dot product)
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = sum(a * b for a, b in zip(query_embedding, embedding))
            similarities.append((self.messages[i], similarity))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_messages(self) -> List[Message]:
        """Get all messages.

        Returns:
            List of messages
        """
        return self.messages.copy()

    def get_context(self) -> str:
        """Get context from all messages.

        Returns:
            Formatted context
        """
        context_lines = []
        for msg in self.messages:
            context_lines.append(f"{msg.role.value.upper()}: {msg.content}")
        return "\n".join(context_lines)

    def clear(self) -> None:
        """Clear all messages and embeddings."""
        self.messages.clear()
        self.embeddings.clear()
        logger.info("Vector memory cleared")


class CustomMemoryBackend(BaseMemory):
    """Template for custom memory implementations.

    Extend this class to create custom memory backends with
    specialized storage or retrieval logic.

    Example:
        >>> class DatabaseMemory(CustomMemoryBackend):
        ...     def __init__(self, db_connection):
        ...         self.db = db_connection
        ...
        ...     def add_message(self, role, content, metadata=None):
        ...         # Store in database
        ...         pass
    """

    def __init__(self, name: str = "CustomBackend") -> None:
        """Initialize custom memory backend.

        Args:
            name: Name of the custom backend
        """
        self.name = name
        self.messages: List[Message] = []
        logger.info(f"CustomMemoryBackend initialized: {name}")

    def add_message(
        self, role: MessageRole, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """Add message to custom backend.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        message = Message(role, content, metadata=metadata or {})
        self.messages.append(message)
        logger.debug(f"Message added to {self.name}: {role.value}")

    def get_messages(self) -> List[Message]:
        """Get messages from custom backend.

        Returns:
            List of messages
        """
        return self.messages.copy()

    def get_context(self) -> str:
        """Get context from custom backend.

        Returns:
            Formatted context
        """
        return "\n".join(
            f"{msg.role.value.upper()}: {msg.content}" for msg in self.messages
        )

    def clear(self) -> None:
        """Clear custom backend."""
        self.messages.clear()


class MemoryFactory:
    """Factory for creating memory instances with preset configurations.

    Example:
        >>> memory = MemoryFactory.create_buffer_memory(max_messages=100)
        >>> memory = MemoryFactory.create_summary_memory()
    """

    @staticmethod
    def create_buffer_memory(
        max_messages: Optional[int] = None,
    ) -> ConversationBufferMemory:
        """Create buffer memory with optional size limit.

        Args:
            max_messages: Maximum messages to store

        Returns:
            ConversationBufferMemory instance
        """
        return ConversationBufferMemory(max_messages=max_messages)

    @staticmethod
    def create_summary_memory(keep_recent: int = 5) -> ConversationSummaryMemory:
        """Create summary memory with recent message window.

        Args:
            keep_recent: Number of recent messages to keep

        Returns:
            ConversationSummaryMemory instance
        """
        return ConversationSummaryMemory(keep_recent=keep_recent)

    @staticmethod
    def create_entity_memory() -> EntityMemory:
        """Create entity tracking memory.

        Returns:
            EntityMemory instance
        """
        return EntityMemory()

    @staticmethod
    def create_vector_memory(embedding_dim: int = 384) -> VectorMemory:
        """Create vector memory with embeddings.

        Args:
            embedding_dim: Dimension of embeddings

        Returns:
            VectorMemory instance
        """
        return VectorMemory(embedding_dim=embedding_dim)


if __name__ == "__main__":
    print("LangChain Memory Systems Module")
    print("=" * 50)

    # Example usage
    memory = ConversationBufferMemory(max_messages=10)

    memory.add_message(MessageRole.USER, "Hello, how are you?")
    memory.add_message(MessageRole.ASSISTANT, "I'm doing well, thank you for asking!")

    print("\nMemory Stats:")
    print(memory.get_stats())

    print("\nContext:")
    print(memory.get_context())
