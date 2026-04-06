"""
AGNO Memory Management and Session State

This module demonstrates memory and state management patterns in AGNO,
including session handling, context management, and persistence.

Author: Shuvam Banerji Seal
Source: https://docs.agno.com/agents/memory
Source: https://docs.agno.com/agent-os/sessions
Source: https://github.com/agno-agi/agno/tree/main/cookbook
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from abc import ABC, abstractmethod
from enum import Enum

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """
    Types of memory in AGNO agents.

    AGNO agents maintain different memory types to handle
    different aspects of state and history.
    """

    SHORT_TERM = "short_term"  # Current conversation/session
    LONG_TERM = "long_term"  # Persistent, cross-session
    WORKING = "working"  # Temporary, in-execution
    EPISODIC = "episodic"  # Event-based memories
    SEMANTIC = "semantic"  # Facts and knowledge


@dataclass
class Message:
    """Single message in conversation history."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
        }


@dataclass
class Context:
    """
    Execution context for an agent.

    Context contains all state and information needed
    for an agent to execute a task.
    """

    session_id: str
    user_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class MemoryStore(ABC):
    """
    Abstract base class for memory storage in AGNO.

    Memory stores handle persistence of agent state,
    conversation history, and learned information.
    """

    @abstractmethod
    def store(self, key: str, value: Any) -> None:
        """Store a value in memory."""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from memory."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        pass


class InMemoryStore(MemoryStore):
    """
    Simple in-memory storage for development and testing.

    AGNO Pattern: In-memory stores are suitable for:
    - Development and testing
    - Stateless operation
    - Low-persistence requirements
    - High-speed access needs

    For production, use database-backed stores like SqliteDb.
    """

    def __init__(self):
        """Initialize in-memory store."""
        self.data: Dict[str, Any] = {}
        logger.info("Initialized in-memory store")

    def store(self, key: str, value: Any) -> None:
        """Store value in memory."""
        self.data[key] = value
        logger.debug(f"Stored key: {key}")

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value from memory."""
        return self.data.get(key)

    def delete(self, key: str) -> None:
        """Delete value from memory."""
        if key in self.data:
            del self.data[key]
            logger.debug(f"Deleted key: {key}")

    def clear(self) -> None:
        """Clear all memory."""
        self.data.clear()
        logger.info("Cleared all memory")


class DatabaseStore(MemoryStore):
    """
    Database-backed memory storage for production.

    AGNO Pattern: Database stores enable:
    - Persistent state across sessions
    - Multi-instance agent deployment
    - State recovery and resilience
    - Audit trails and compliance

    AGNO provides built-in support for:
    - SqliteDb: Local SQLite database
    - PostgreSQL: Enterprise database
    - Custom backends via AgentDB interface

    Reference: https://docs.agno.com/agent-os/database
    """

    def __init__(self, db_file: str = "agno.db"):
        """
        Initialize database store.

        Args:
            db_file: Path to database file

        In real AGNO usage:
        ```python
        from agno.db.sqlite import SqliteDb
        db = SqliteDb(db_file="agno.db")
        agent = Agent(db=db, ...)
        ```
        """
        self.db_file = db_file
        # In production, this would be actual database connection
        self.in_memory_cache: Dict[str, Any] = {}
        logger.info(f"Initialized database store: {db_file}")

    def store(self, key: str, value: Any) -> None:
        """Store value in database."""
        self.in_memory_cache[key] = value
        logger.debug(f"Stored in database: {key}")
        # In production: INSERT/UPDATE in actual database

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value from database."""
        # In production: SELECT from actual database
        return self.in_memory_cache.get(key)

    def delete(self, key: str) -> None:
        """Delete value from database."""
        if key in self.in_memory_cache:
            del self.in_memory_cache[key]
        # In production: DELETE from actual database

    def clear(self) -> None:
        """Clear all data from database."""
        self.in_memory_cache.clear()
        # In production: DELETE all from tables


class ConversationHistory:
    """
    Manages conversation history for AGNO agents.

    Conversation history is essential for:
    - Context awareness
    - Coherent multi-turn interactions
    - Learning from past exchanges
    - Providing audit trails

    AGNO Pattern:
    Agents can control history through add_history_to_context parameter.
    """

    def __init__(self, max_messages: int = 100):
        """
        Initialize conversation history.

        Args:
            max_messages: Maximum messages to keep in memory
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages
        logger.info(f"Initialized conversation history (max: {max_messages})")

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add a message to the history.

        Args:
            role: Message source ("user", "assistant", "system")
            content: Message text
            metadata: Additional message metadata
            tool_calls: Function calls made in this message
            tool_results: Results from function calls
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            tool_calls=tool_calls,
            tool_results=tool_results,
        )

        self.messages.append(message)

        # Keep history size under control
        if len(self.messages) > self.max_messages:
            # Remove oldest messages
            self.messages = self.messages[-self.max_messages :]

        logger.debug(f"Added message from {role}")

    def get_context(self, num_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation context for agent input.

        Args:
            num_messages: Number of recent messages to include

        Returns:
            List of message dictionaries

        AGNO Pattern:
        Agents use this to understand conversation context.
        """
        messages = self.messages

        if num_messages:
            messages = messages[-num_messages:]

        return [m.to_dict() for m in messages]

    def get_summary(self) -> str:
        """
        Get a summary of the conversation.

        AGNO Pattern:
        For long conversations, agents may use summaries
        to stay within token limits while retaining context.
        """
        if not self.messages:
            return "No conversation history"

        summary_parts = []
        for msg in self.messages[-5:]:  # Last 5 messages
            summary_parts.append(f"{msg.role}: {msg.content[:100]}")

        return "\n".join(summary_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        logger.info("Cleared conversation history")

    def get_size(self) -> int:
        """Get number of messages in history."""
        return len(self.messages)


class SessionManager:
    """
    Manages user sessions in AGNO applications.

    Sessions are fundamental to AGNO's stateful execution model.

    AGNO Session Features:
    - Per-user isolation
    - State persistence
    - Session timeout handling
    - Multi-agent state coordination

    Reference: https://docs.agno.com/agent-os/sessions
    """

    def __init__(self, session_timeout_hours: int = 24):
        """
        Initialize session manager.

        Args:
            session_timeout_hours: How long sessions persist
        """
        self.sessions: Dict[str, Context] = {}
        self.histories: Dict[str, ConversationHistory] = {}
        self.session_timeout = timedelta(hours=session_timeout_hours)
        logger.info(f"Initialized session manager (timeout: {session_timeout_hours}h)")

    def create_session(
        self, session_id: str, user_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Context:
        """
        Create a new session.

        Args:
            session_id: Unique session identifier
            user_id: Associated user ID
            metadata: Session metadata

        Returns:
            Created context
        """
        context = Context(
            session_id=session_id, user_id=user_id, metadata=metadata or {}
        )

        self.sessions[session_id] = context
        self.histories[session_id] = ConversationHistory()

        logger.info(f"Created session {session_id} for user {user_id}")
        return context

    def get_session(self, session_id: str) -> Optional[Context]:
        """Get session context."""
        return self.sessions.get(session_id)

    def get_history(self, session_id: str) -> Optional[ConversationHistory]:
        """Get conversation history for a session."""
        return self.histories.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.histories:
            del self.histories[session_id]
        logger.info(f"Deleted session {session_id}")

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions removed
        """
        now = datetime.now()
        expired = [
            sid
            for sid, ctx in self.sessions.items()
            if now - ctx.last_updated > self.session_timeout
        ]

        for sid in expired:
            self.delete_session(sid)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)


class ContextManager:
    """
    Manages execution context for AGNO agents.

    Context includes variables, metadata, and state
    needed during agent execution.

    AGNO Pattern:
    Context flows through the agent execution pipeline,
    allowing different components to access and modify state.
    """

    def __init__(self):
        """Initialize context manager."""
        self.active_contexts: Dict[str, Context] = {}

    def set_variable(self, session_id: str, key: str, value: Any) -> None:
        """
        Set a context variable.

        Args:
            session_id: Session identifier
            key: Variable name
            value: Variable value
        """
        if session_id not in self.active_contexts:
            self.active_contexts[session_id] = Context(
                session_id=session_id, user_id="unknown"
            )

        self.active_contexts[session_id].variables[key] = value
        self.active_contexts[session_id].last_updated = datetime.now()
        logger.debug(f"Set variable {key} in session {session_id}")

    def get_variable(self, session_id: str, key: str) -> Optional[Any]:
        """Get a context variable."""
        ctx = self.active_contexts.get(session_id)
        if ctx:
            return ctx.variables.get(key)
        return None

    def get_context(self, session_id: str) -> Optional[Context]:
        """Get full context for a session."""
        return self.active_contexts.get(session_id)


def main():
    """
    Demonstration of AGNO memory and session management.

    Reference Documentation:
    - https://docs.agno.com/agents/memory
    - https://docs.agno.com/agent-os/sessions
    - https://docs.agno.com/agent-os/database
    """
    print("\n=== AGNO Memory Management Demo ===\n")

    # 1. Session Manager
    print("1. Session Management...")
    session_mgr = SessionManager(session_timeout_hours=1)

    ctx1 = session_mgr.create_session("sess_001", "user_123")
    ctx2 = session_mgr.create_session("sess_002", "user_456")

    print(f"Active sessions: {session_mgr.get_session_count()}")

    # 2. Conversation History
    print("\n2. Conversation History...")
    history = session_mgr.get_history("sess_001")

    history.add_message("user", "What is AGNO?")
    history.add_message(
        "assistant",
        "AGNO is a framework for building agentic software.",
        metadata={"model": "claude-3-5-sonnet"},
    )
    history.add_message("user", "How does it handle memory?")
    history.add_message(
        "assistant",
        "AGNO provides stateful execution with session and memory management.",
    )

    print(f"Messages in history: {history.get_size()}")
    print(
        f"Context for agent:\n{json.dumps(history.get_context(num_messages=2), indent=2)}"
    )

    # 3. Context Variables
    print("\n3. Context Management...")
    ctx_mgr = ContextManager()

    ctx_mgr.set_variable("sess_001", "user_name", "Alice")
    ctx_mgr.set_variable("sess_001", "conversation_topic", "AI agents")
    ctx_mgr.set_variable("sess_001", "language", "en")

    print(f"User name: {ctx_mgr.get_variable('sess_001', 'user_name')}")
    print(f"Topic: {ctx_mgr.get_variable('sess_001', 'conversation_topic')}")

    # 4. Memory Stores
    print("\n4. Memory Storage...")

    # In-memory store
    mem_store = InMemoryStore()
    mem_store.store("key1", {"data": "value1"})
    mem_store.store("key2", {"data": "value2"})
    print(f"In-memory store - key1: {mem_store.retrieve('key1')}")

    # Database store
    db_store = DatabaseStore("agno.db")
    db_store.store("agent_state", {"status": "running"})
    print(f"Database store - agent_state: {db_store.retrieve('agent_state')}")

    # 5. Summary
    print("\n5. Session Summary...")
    print(
        json.dumps(
            {
                "sessions_created": 2,
                "active_sessions": session_mgr.get_session_count(),
                "history_size": history.get_size(),
                "context_variables": ctx_mgr.get_context("sess_001").variables
                if ctx_mgr.get_context("sess_001")
                else {},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
