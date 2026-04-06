"""
Conversational Agent with Persistent Memory Management
Author: Shuvam Banerji Seal

This module implements a conversational AI agent with dual memory systems:
- Short-term memory: Recent conversation buffer (configurable window size)
- Long-term memory: Vector-based semantic memory store for historical context

The agent maintains conversation continuity across sessions and can recall
relevant past interactions to provide contextually informed responses.

Source: https://medium.com/@saurabhzodex/building-memory-augmented-ai-agents-with-langchain-part-1-2c21cc8050da
Source: https://python.langchain.com/api_reference
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.language_model import BaseLanguageModel
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import (
    AgentExecutor,
    Tool,
    create_tool_calling_agent,
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Vector-based semantic memory store for persistent conversation history.

    Uses Chroma for efficient similarity search and metadata-based filtering.
    Stores interactions with timestamps for temporal querying.

    Attributes:
        vectorstore: Chroma vector store instance
        retriever: Configured retriever for similarity search
        persist_dir: Directory path for persistent storage
    """

    def __init__(
        self,
        persist_dir: str = "./data/memory_store",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        collection_name: str = "conversation_history",
    ) -> None:
        """
        Initialize long-term memory with vector database backend.

        Args:
            persist_dir: Directory for persistent storage
            embedding_model: HuggingFace embedding model name
            collection_name: Chroma collection name
        """
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Initialize Chroma vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )

        # Configure retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant memories
        )

        logger.info(f"LongTermMemory initialized: {persist_dir}")

    def get_relevant_memories(self, query: str) -> List[Document]:
        """
        Retrieve semantically similar memories from the vector store.

        Args:
            query: User query or conversation input

        Returns:
            List of relevant Document objects with metadata
        """
        try:
            memories = self.retriever.invoke(query)
            logger.debug(f"Retrieved {len(memories)} relevant memories")
            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a new interaction in long-term memory.

        Args:
            text: Conversation text to store
            metadata: Optional metadata (timestamp added automatically)

        Example:
            >>> memory.add_memory(
            ...     "User: What is Python? Agent: Python is...",
            ...     metadata={"session_id": "session_123"}
            ... )
        """
        try:
            if metadata is None:
                metadata = {}

            # Add timestamp automatically
            metadata["timestamp"] = datetime.now().isoformat()

            self.vectorstore.add_texts(texts=[text], metadatas=[metadata])
            logger.debug(f"Memory stored: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error storing memory: {e}")

    def format_memory_context(self, memories: List[Document]) -> str:
        """
        Format retrieved memories into a readable context string.

        Args:
            memories: List of Document objects

        Returns:
            Formatted string with memory context
        """
        if not memories:
            return "No relevant past interactions found."

        formatted = "Past interactions:\n"
        for i, memory in enumerate(memories, 1):
            timestamp = memory.metadata.get("timestamp", "Unknown")
            formatted += f"\n[Memory {i}] ({timestamp})\n{memory.page_content}\n"

        return formatted


class ConversationalAgent:
    """
    Conversational AI agent with dual memory system.

    Combines short-term conversation buffer with long-term semantic memory
    to maintain context across sessions and provide informed responses.

    Attributes:
        llm: Language model instance
        memory: Short-term conversation buffer
        long_term_memory: Long-term semantic memory store
        agent_executor: Agent execution engine
        session_id: Unique session identifier
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        memory_window_size: int = 3,
        llm_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize conversational agent with memory systems.

        Args:
            model: LLM model name
            temperature: LLM temperature parameter (0.0-1.0)
            memory_window_size: Number of recent messages to keep in short-term memory
            llm_api_key: API key for the LLM (uses env var if not provided)

        Example:
            >>> agent = ConversationalAgent(
            ...     model="gpt-3.5-turbo",
            ...     temperature=0.7,
            ...     memory_window_size=5
            ... )
        """
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize LLM
        api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

        # Initialize short-term memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=memory_window_size,
            return_messages=True,
        )

        # Initialize long-term memory
        self.long_term_memory = LongTermMemory(
            persist_dir=f"./data/memory_store/{self.session_id}"
        )

        # Setup basic tools for the agent
        self.tools = self._setup_tools()

        # Initialize agent executor
        self.agent_executor = self._create_agent_executor()

        logger.info(f"ConversationalAgent initialized (Session: {self.session_id})")

    def _setup_tools(self) -> List[Tool]:
        """
        Configure tools available to the agent.

        Returns:
            List of Tool objects
        """
        tools = [
            Tool(
                name="recall_memory",
                func=self._recall_memory_tool,
                description="Retrieve relevant past conversations from memory",
            ),
            Tool(
                name="store_memory",
                func=self._store_memory_tool,
                description="Store current interaction in long-term memory",
            ),
            Tool(
                name="get_context",
                func=self._get_context_tool,
                description="Get formatted conversation context",
            ),
        ]
        return tools

    def _recall_memory_tool(self, query: str) -> str:
        """Tool to recall relevant memories."""
        memories = self.long_term_memory.get_relevant_memories(query)
        return self.long_term_memory.format_memory_context(memories)

    def _store_memory_tool(self, text: str) -> str:
        """Tool to store interaction in long-term memory."""
        self.long_term_memory.add_memory(text, metadata={"session_id": self.session_id})
        return f"Memory stored successfully"

    def _get_context_tool(self, _: str) -> str:
        """Tool to get current conversation context."""
        history = self.memory.load_memory_variables({})
        return json.dumps(history, indent=2)

    def _create_agent_executor(self) -> AgentExecutor:
        """
        Create and configure the agent executor.

        Returns:
            Configured AgentExecutor instance
        """
        # Create system prompt
        system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful conversational AI assistant with memory capabilities.
You can:
- Recall relevant past interactions
- Store important information for future reference
- Maintain context across conversations
- Provide thoughtful and contextually-aware responses

Always leverage your memory system to provide better responses.""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create the agent
        agent = create_tool_calling_agent(self.llm, self.tools, system_prompt)

        # Create executor
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        return executor

    def chat(self, user_input: str, use_long_term_memory: bool = True) -> str:
        """
        Process user input and generate response using memory systems.

        Args:
            user_input: User message
            use_long_term_memory: Whether to retrieve long-term context

        Returns:
            Agent response string

        Example:
            >>> agent = ConversationalAgent()
            >>> response = agent.chat("What did we discuss earlier?")
            >>> print(response)
        """
        try:
            # Retrieve long-term context if enabled
            long_term_context = ""
            if use_long_term_memory:
                memories = self.long_term_memory.get_relevant_memories(user_input)
                long_term_context = self.long_term_memory.format_memory_context(
                    memories
                )

            # Prepare enhanced input
            enhanced_input = user_input
            if long_term_context:
                enhanced_input = (
                    f"{long_term_context}\n\nCurrent question: {user_input}"
                )

            # Get response from agent
            result = self.agent_executor.invoke({"input": enhanced_input})
            response = result.get("output", "No response generated")

            # Store interaction in long-term memory
            self.long_term_memory.add_memory(
                f"User: {user_input}\nAssistant: {response}",
                metadata={"session_id": self.session_id, "type": "conversation"},
            )

            logger.info(f"Response generated (Session: {self.session_id})")
            return response

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I encountered an error processing your request: {str(e)}"

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session.

        Returns:
            Dictionary with session statistics

        Example:
            >>> summary = agent.get_session_summary()
            >>> print(f"Session ID: {summary['session_id']}")
        """
        history = self.memory.load_memory_variables({})
        chat_messages = history.get("chat_history", [])

        return {
            "session_id": self.session_id,
            "messages_in_short_term": len(chat_messages),
            "timestamp": datetime.now().isoformat(),
        }

    def clear_short_term_memory(self) -> None:
        """Clear the short-term conversation buffer."""
        self.memory.clear()
        logger.info("Short-term memory cleared")

    def export_session(self, filepath: str) -> None:
        """
        Export session conversation to a JSON file.

        Args:
            filepath: Path to save the session export
        """
        try:
            history = self.memory.load_memory_variables({})
            session_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "conversation": [
                    {"role": msg.type, "content": msg.content}
                    for msg in history.get("chat_history", [])
                ],
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(session_data, f, indent=2)

            logger.info(f"Session exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting session: {e}")


# ============================================================================
# Usage Examples
# ============================================================================


def example_basic_conversation() -> None:
    """
    Basic conversation example demonstrating memory capabilities.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Conversation with Memory")
    print("=" * 70)

    # Load environment variables
    load_dotenv()

    # Initialize agent
    agent = ConversationalAgent(
        model="gpt-3.5-turbo", temperature=0.7, memory_window_size=3
    )

    # Example conversation turns
    conversations = [
        "Hi, my name is Alice and I'm interested in machine learning.",
        "What are the key concepts in deep learning?",
        "Can you summarize what we discussed so far?",
        "How does that relate to my earlier interest?",
    ]

    for user_input in conversations:
        print(f"\nUser: {user_input}")
        response = agent.chat(user_input, use_long_term_memory=True)
        print(f"Assistant: {response}")

    # Print session summary
    summary = agent.get_session_summary()
    print(f"\nSession Summary: {json.dumps(summary, indent=2)}")


def example_memory_persistence() -> None:
    """
    Example showing memory persistence across sessions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Memory Persistence Across Sessions")
    print("=" * 70)

    load_dotenv()

    # First session
    print("\n--- Session 1 ---")
    agent1 = ConversationalAgent()
    response1 = agent1.chat("I work as a data scientist in healthcare.")
    print(f"User: I work as a data scientist in healthcare.")
    print(f"Assistant: {response1}")

    # In a real scenario, you would create a new agent instance but point to
    # the same memory store to demonstrate persistence
    print("\n--- Session 2 (Same Memory Store) ---")
    agent2 = ConversationalAgent()
    response2 = agent2.chat("What was my professional background?")
    print(f"User: What was my professional background?")
    print(f"Assistant: {response2}")


def example_session_export() -> None:
    """
    Example showing session export functionality.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Session Export")
    print("=" * 70)

    load_dotenv()

    agent = ConversationalAgent()

    # Have a brief conversation
    agent.chat("Hello! I'm learning about LangChain agents.")
    agent.chat("Can you explain tool integration?")

    # Export session
    export_path = "./exported_sessions/session_example.json"
    agent.export_session(export_path)
    print(f"\nSession exported to: {export_path}")


if __name__ == "__main__":
    """
    Main entry point demonstrating agent capabilities.
    
    Required environment variables:
    - OPENAI_API_KEY: OpenAI API key for LLM access
    
    Setup Instructions:
    1. Install dependencies: pip install langchain langchain-openai langchain-huggingface chromadb python-dotenv
    2. Set up .env file with OPENAI_API_KEY
    3. Run: python chat_agent_with_memory.py
    """

    print("Conversational Agent with Memory - Example Usage")
    print("=" * 70)

    # Uncomment examples to run
    # example_basic_conversation()
    # example_memory_persistence()
    # example_session_export()

    print("\nTo run examples, uncomment them in the __main__ section")
    print("Examples available:")
    print("  - example_basic_conversation()")
    print("  - example_memory_persistence()")
    print("  - example_session_export()")
