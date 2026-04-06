"""
LangChain & LangGraph Agents Framework - Python Package

This package provides high-quality, production-ready implementations
for building intelligent agents with LangChain and LangGraph.

Modules:
    - langchain_agent_basics: Core agent initialization and management
    - langchain_tools_integration: Tool creation, registry, and validation
    - langchain_memory_systems: Multiple memory implementations
    - langgraph_workflow: Core workflow building blocks and state graphs
    - langgraph_state_management: State management, validation, and checkpointing
    - langgraph_subgraphs: Composable reusable workflow patterns

Author: Shuvam Banerji Seal
Version: 1.0.0

Usage:
    from agents.src import (
        BasicAgent, AgentConfig, LLMProvider,
        ToolRegistry, FunctionTool, ToolCategory,
        ConversationBufferMemory, EntityMemory,
        WorkflowBuilder, StateManager, LinearSubgraph
    )
"""

from .langchain_agent_basics import (
    BasicAgent,
    AgentConfig,
    LLMProvider,
    LLMInitializer,
    AgentFactory,
)

from .langchain_tools_integration import (
    BaseTool,
    FunctionTool,
    ToolRegistry,
    ToolSchema,
    ToolParameter,
    ToolCategory,
    ToolValidator,
    DynamicToolLoader,
)

from .langchain_memory_systems import (
    BaseMemory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
    EntityMemory,
    VectorMemory,
    CustomMemoryBackend,
    MemoryFactory,
    Message,
    MessageRole,
)

from .langgraph_workflow import (
    WorkflowState,
    Node,
    ProcessorNode,
    ConditionalRouter,
    WorkflowBuilder,
    CompiledWorkflow,
    ErrorHandlingStrategy,
    RetryStrategy,
    FallbackStrategy,
    create_simple_workflow,
    NodeType,
)

from .langgraph_state_management import (
    StateSchema,
    StateManager,
    CheckpointManager,
    MultiAgentStateCoordinator,
    StateValidator,
    SchemaValidator,
    RangeValidator,
    StateCheckpoint,
    StateChange,
    StateChangeType,
)

from .langgraph_subgraphs import (
    BaseSubgraph,
    LinearSubgraph,
    ConditionalSubgraph,
    StateTranslator,
    ComposableSubgraph,
    SubgraphRegistry,
    SubgraphMetadata,
    SubgraphType,
    register_subgraph,
    get_subgraph,
    list_registered_subgraphs,
)

__version__ = "1.0.0"
__author__ = "Shuvam Banerji Seal"

__all__ = [
    # Agent basics
    "BasicAgent",
    "AgentConfig",
    "LLMProvider",
    "LLMInitializer",
    "AgentFactory",
    # Tools
    "BaseTool",
    "FunctionTool",
    "ToolRegistry",
    "ToolSchema",
    "ToolParameter",
    "ToolCategory",
    "ToolValidator",
    "DynamicToolLoader",
    # Memory
    "BaseMemory",
    "ConversationBufferMemory",
    "ConversationSummaryMemory",
    "EntityMemory",
    "VectorMemory",
    "CustomMemoryBackend",
    "MemoryFactory",
    "Message",
    "MessageRole",
    # LangGraph Workflow
    "WorkflowState",
    "Node",
    "ProcessorNode",
    "ConditionalRouter",
    "WorkflowBuilder",
    "CompiledWorkflow",
    "ErrorHandlingStrategy",
    "RetryStrategy",
    "FallbackStrategy",
    "create_simple_workflow",
    "NodeType",
    # LangGraph State Management
    "StateSchema",
    "StateManager",
    "CheckpointManager",
    "MultiAgentStateCoordinator",
    "StateValidator",
    "SchemaValidator",
    "RangeValidator",
    "StateCheckpoint",
    "StateChange",
    "StateChangeType",
    # LangGraph Subgraphs
    "BaseSubgraph",
    "LinearSubgraph",
    "ConditionalSubgraph",
    "StateTranslator",
    "ComposableSubgraph",
    "SubgraphRegistry",
    "SubgraphMetadata",
    "SubgraphType",
    "register_subgraph",
    "get_subgraph",
    "list_registered_subgraphs",
]
