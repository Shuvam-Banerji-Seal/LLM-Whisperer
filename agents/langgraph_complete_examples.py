"""
LangGraph Complete Implementation Guide and Examples

This module demonstrates comprehensive usage of the LangGraph modules
for building stateful, composable AI workflows.

Author: Shuvam Banerji Seal
Source: https://tutorialq.com/ai/frameworks/langgraph-stateful-workflows
Source: https://medium.com/@khanbasil2002/building-intelligent-workflows-with-langgraph-from-simple-graphs-to-complex-ai-agents-9ed74c22a6ad
Source: https://wiki.tapnex.tech/articles/en/technology/langgraph-the-complete-guide-to-building-stateful-multi-agent-ai-workflows-2025
Source: https://machinelearningplus.com/gen-ai/langgraph-subgraphs-composing-reusable-workflows/

Examples included:
    1. Basic Workflow Creation
    2. State Management and Validation
    3. Checkpoint Management
    4. Conditional Routing
    5. Linear Subgraphs
    6. Multi-Agent State Coordination
    7. Complex Workflow Composition
"""

import logging
from typing import Any, Dict, TypedDict, Annotated
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage

# Import our modules
from src.langgraph_workflow import (
    WorkflowBuilder,
    WorkflowState,
    ConditionalRouter,
    NodeType,
    ProcessorNode,
    create_simple_workflow,
)

from src.langgraph_state_management import (
    StateSchema,
    StateManager,
    CheckpointManager,
    MultiAgentStateCoordinator,
    SchemaValidator,
    RangeValidator,
)

from src.langgraph_subgraphs import (
    LinearSubgraph,
    ConditionalSubgraph,
    StateTranslator,
    ComposableSubgraph,
    SubgraphType,
    register_subgraph,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Basic Workflow Creation
# ============================================================================


def example_1_basic_workflow():
    """
    Demonstrates creating a simple linear workflow with three nodes.

    This example shows:
    - Creating a WorkflowBuilder
    - Adding nodes with callable functions
    - Setting entry point
    - Adding edges
    - Compiling and executing the workflow
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Workflow Creation")
    print("=" * 80)

    # Define node functions
    def input_node(state: Dict[str, Any]) -> dict:
        """Extract and prepare input."""
        logger.info("Input node: Processing initial message")
        return {"messages": state.get("messages", [])}

    def processor_node(state: Dict[str, Any]) -> dict:
        """Process the message."""
        logger.info("Processor node: Transforming message")
        if state.get("messages"):
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "content"):
                processed = f"Processed: {last_msg.content.upper()}"
                return {"messages": [AIMessage(content=processed)]}
        return state

    def output_node(state: Dict[str, Any]) -> dict:
        """Finalize output."""
        logger.info("Output node: Finalizing results")
        return state

    # Build workflow
    builder = WorkflowBuilder(WorkflowState)
    builder.add_node("input", input_node)
    builder.add_node("processor", processor_node)
    builder.add_node("output", output_node)
    builder.set_entry_point("input")
    builder.add_edge("input", "processor")
    builder.add_edge("processor", "output")
    builder.add_edge("output", "END")

    workflow = builder.build()

    # Execute workflow
    initial_state = {
        "messages": [HumanMessage(content="hello world")],
        "state_id": "example-1",
        "metadata": {"source": "example_1"},
        "error_count": 0,
        "execution_log": [],
    }

    result = workflow.invoke(initial_state)

    print(f"\nResult messages: {result['messages']}")
    print(f"Execution log: {result['execution_log']}")
    print("✓ Example 1 completed successfully")


# ============================================================================
# EXAMPLE 2: State Management and Validation
# ============================================================================


class UserProcessingState(TypedDict):
    """State for user processing workflow."""

    user_id: int
    user_name: str
    confidence_score: float
    messages: list


def example_2_state_management():
    """
    Demonstrates state schema definition, validation, and management.

    This example shows:
    - Creating a StateSchema with multiple field types
    - Adding validators
    - Creating and managing state with StateManager
    - Updating state with validation
    - Tracking state history
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: State Management and Validation")
    print("=" * 80)

    # Create schema
    schema = StateSchema("UserProcessingState")
    schema.add_field(
        "user_id", int, required=True, description="Unique user identifier"
    )
    schema.add_field("user_name", str, required=True, description="User's name")
    schema.add_field(
        "confidence_score",
        float,
        required=False,
        default=0.0,
        description="ML model confidence",
    )
    schema.add_field(
        "messages",
        list,
        required=False,
        default=[],
        description="Communication history",
    )

    # Add validators
    range_validator = RangeValidator({"confidence_score": (0.0, 1.0)})
    schema.add_validator(range_validator)

    # Create state manager
    manager = StateManager(schema)

    # Initialize state
    initial_state = {
        "user_id": 123,
        "user_name": "Alice",
        "confidence_score": 0.5,
        "messages": [],
    }

    manager.set_state(initial_state, source="initialization")
    logger.info("State initialized")

    # Update state
    manager.update_state({"confidence_score": 0.95}, source="model_update")
    logger.info("State updated")

    # Create checkpoint
    checkpoint = manager.create_checkpoint(
        "checkpoint_after_processing",
        "processor_node",
        metadata={"reason": "midpoint_save"},
    )
    logger.info(f"Checkpoint created: {checkpoint.checkpoint_id}")

    # Show history
    print(f"\nCurrent state: {manager.get_state()}")
    print(f"State history length: {len(manager.get_state_history())}")
    print(f"Changes recorded: {len(manager.get_change_history())}")
    print("✓ Example 2 completed successfully")


# ============================================================================
# EXAMPLE 3: Checkpoint Management
# ============================================================================


def example_3_checkpoint_management():
    """
    Demonstrates different checkpoint backends (memory and SQLite).

    This example shows:
    - Creating memory-based checkpoints
    - Creating persistent SQLite checkpoints
    - Using checkpoints with workflow execution
    - Restoring from checkpoints
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Checkpoint Management")
    print("=" * 80)

    # In-memory checkpointing
    print("\n--- In-Memory Checkpoints ---")
    memory_manager = CheckpointManager(backend="memory")
    checkpointer = memory_manager.get_checkpointer()
    config = memory_manager.create_config("user-session-123")

    logger.info(f"Memory checkpointer created: {checkpointer}")
    logger.info(f"Config for thread_id 'user-session-123': {config}")

    # SQLite checkpointing (would be used in production)
    print("\n--- SQLite Checkpoints (Production) ---")
    sqlite_manager = CheckpointManager(
        backend="sqlite", db_path="/tmp/workflow_checkpoints.db"
    )

    try:
        sqlite_checkpointer = sqlite_manager.get_checkpointer()
        logger.info(f"SQLite checkpointer configured: {sqlite_manager.db_path}")

        # Create multiple thread configs
        for i in range(3):
            thread_config = sqlite_manager.create_config(f"workflow-thread-{i}")
            logger.info(f"Thread config {i}: {thread_config}")
    except Exception as e:
        logger.info(f"SQLite example (might need dependencies): {e}")

    print("✓ Example 3 completed successfully")


# ============================================================================
# EXAMPLE 4: Conditional Routing
# ============================================================================


def example_4_conditional_routing():
    """
    Demonstrates conditional edge routing based on state.

    This example shows:
    - Creating a ConditionalRouter
    - Defining multiple routing conditions
    - Setting default routes
    - Using router in a workflow
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Conditional Routing")
    print("=" * 80)

    # Create router
    router = ConditionalRouter()

    # Add routing conditions
    router.add_route("urgent", lambda state: state.get("priority") == "high")
    router.add_route("normal", lambda state: state.get("priority") == "normal")
    router.add_route("low", lambda state: state.get("priority") == "low")
    router.add_default_route("normal")

    # Test routing logic
    test_states = [
        {"priority": "high", "task": "critical_bug"},
        {"priority": "normal", "task": "feature"},
        {"priority": "low", "task": "refactor"},
        {"priority": "unknown", "task": "misc"},
    ]

    print("\nRouting decisions:")
    for state in test_states:
        next_route = router.route(state)
        print(f"  State {state} -> Route: {next_route}")

    print("✓ Example 4 completed successfully")


# ============================================================================
# EXAMPLE 5: Linear Subgraphs
# ============================================================================


class TextProcessingState(TypedDict):
    """State for text processing subgraph."""

    input_text: str
    cleaned_text: str
    tokens: list


def example_5_linear_subgraphs():
    """
    Demonstrates creating and using linear subgraphs.

    This example shows:
    - Creating a LinearSubgraph
    - Adding sequential nodes
    - Building and compiling subgraph
    - Setting metadata
    - Registering subgraph for reuse
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Linear Subgraphs")
    print("=" * 80)

    # Define processing functions
    def clean_text(state: Dict[str, Any]) -> dict:
        """Clean and normalize text."""
        text = state.get("input_text", "").strip().lower()
        return {"cleaned_text": text}

    def tokenize(state: Dict[str, Any]) -> dict:
        """Tokenize text into words."""
        tokens = state.get("cleaned_text", "").split()
        return {"tokens": tokens}

    def analyze_tokens(state: Dict[str, Any]) -> dict:
        """Analyze token statistics."""
        tokens = state.get("tokens", [])
        logger.info(f"Analyzed {len(tokens)} tokens")
        return state

    # Create linear subgraph
    subgraph = LinearSubgraph("text_processor", TextProcessingState)
    subgraph.add_node("clean", clean_text)
    subgraph.add_node("tokenize", tokenize)
    subgraph.add_node("analyze", analyze_tokens)

    # Add metadata
    subgraph.set_metadata(
        description="Processes text through cleaning and tokenization",
        author="Shuvam Banerji Seal",
        tags=["text-processing", "nlp", "utility"],
    )

    # Build and compile
    subgraph.build()
    compiled = subgraph.compile()

    # Test the subgraph
    test_input = {
        "input_text": "  Hello World! This is a Test.  ",
        "cleaned_text": "",
        "tokens": [],
    }

    result = compiled.invoke(test_input)

    print(f"\nInput: '{test_input['input_text']}'")
    print(f"Cleaned: '{result['cleaned_text']}'")
    print(f"Tokens: {result['tokens']}")

    # Register for reuse
    register_subgraph("text_processor_v1", subgraph)
    logger.info("Registered subgraph: text_processor_v1")

    print("✓ Example 5 completed successfully")


# ============================================================================
# EXAMPLE 6: Multi-Agent State Coordination
# ============================================================================


class ResearchAgentState(TypedDict):
    """State for research agent."""

    topic: str
    research_findings: str


class WritingAgentState(TypedDict):
    """State for writing agent."""

    topic: str
    draft: str


def example_6_multi_agent_coordination():
    """
    Demonstrates multi-agent state coordination.

    This example shows:
    - Creating a MultiAgentStateCoordinator
    - Registering multiple agents
    - Merging states from different agents
    - Validating multi-agent states
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Multi-Agent State Coordination")
    print("=" * 80)

    # Create coordinator
    coordinator = MultiAgentStateCoordinator()

    # Create schemas for each agent
    research_schema = StateSchema("ResearchAgentState")
    research_schema.add_field("topic", str, required=True)
    research_schema.add_field("research_findings", str, required=True)

    writing_schema = StateSchema("WritingAgentState")
    writing_schema.add_field("topic", str, required=True)
    writing_schema.add_field("draft", str, required=True)

    # Register agents
    coordinator.register_agent("research_agent", "research", research_schema)
    coordinator.register_agent("writing_agent", "writing", writing_schema)

    # Simulate agent results
    agent_results = {
        "research_agent": {
            "topic": "AI Agents",
            "research_findings": "AI agents are increasingly important in 2026",
        },
        "writing_agent": {
            "topic": "AI Agents",
            "draft": "An introduction to modern AI agents...",
        },
    }

    # Merge states
    merged = coordinator.merge_states(agent_results)

    # Validate all agents
    is_valid, validation_results = coordinator.validate_all_agents(agent_results)

    print(f"\nMerged state keys: {list(merged.keys())}")
    print(f"All agents valid: {is_valid}")
    print(f"Validation results: {validation_results}")

    print("✓ Example 6 completed successfully")


# ============================================================================
# EXAMPLE 7: Complex Workflow Composition
# ============================================================================


def example_7_complex_composition():
    """
    Demonstrates complex workflow composition with subgraphs and state translation.

    This example shows:
    - Creating a StateTranslator
    - Composing subgraphs as nodes in parent workflows
    - Complex state mappings
    - Full end-to-end workflow execution
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Complex Workflow Composition")
    print("=" * 80)

    # Define parent state
    class ComplexWorkflowState(TypedDict):
        """State for complex parent workflow."""

        user_input: str
        processed_output: str
        final_result: str
        messages: list
        state_id: str
        metadata: dict
        error_count: int
        execution_log: list

    # Create subgraph
    def transform_func(state: Dict[str, Any]) -> dict:
        """Transform text."""
        return {"processed_output": state.get("user_input", "").upper()}

    subgraph = LinearSubgraph("transformer", TextProcessingState)
    subgraph.add_node("transform", transform_func)
    subgraph.build()
    compiled_sub = subgraph.compile()

    # Create state translator
    translator = StateTranslator()
    translator.map_input("user_input", "input_text")
    translator.map_output("processed_output", "processed_output")

    # Create composable subgraph
    composable = ComposableSubgraph("text_transformer", compiled_sub, translator)

    # Define parent workflow nodes
    def init_node(state: Dict[str, Any]) -> dict:
        """Initialize workflow."""
        return {"messages": [HumanMessage(content="Starting workflow")]}

    def finalize_node(state: Dict[str, Any]) -> dict:
        """Finalize results."""
        return {"final_result": state.get("processed_output", "")}

    # Build parent workflow
    builder = WorkflowBuilder(ComplexWorkflowState)
    builder.add_node("init", init_node)
    builder.add_node("transform", composable)
    builder.add_node("finalize", finalize_node)
    builder.set_entry_point("init")
    builder.add_edge("init", "transform")
    builder.add_edge("transform", "finalize")
    builder.add_edge("finalize", "END")

    workflow = builder.build()

    # Execute
    initial_state = {
        "user_input": "test message",
        "processed_output": "",
        "final_result": "",
        "messages": [],
        "state_id": "complex-001",
        "metadata": {},
        "error_count": 0,
        "execution_log": [],
    }

    result = workflow.invoke(initial_state)

    print(f"\nUser input: {result['user_input']}")
    print(f"Final result: {result['final_result']}")
    print(f"Execution log entries: {len(result['execution_log'])}")

    print("✓ Example 7 completed successfully")


# ============================================================================
# EXAMPLE 8: Using Factory Functions
# ============================================================================


def example_8_factory_functions():
    """
    Demonstrates using factory functions for quick workflow creation.

    This example shows:
    - Using create_simple_workflow for rapid prototyping
    - Processing functions
    - Quick execution and testing
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Using Factory Functions")
    print("=" * 80)

    # Define a simple processing function
    def reverse_and_uppercase(text: str) -> str:
        """Reverse text and make uppercase."""
        return text[::-1].upper()

    # Create simple workflow
    workflow = create_simple_workflow("test", reverse_and_uppercase)

    # Execute
    result = workflow.invoke(
        {
            "messages": [HumanMessage(content="hello")],
            "state_id": "factory-001",
            "metadata": {},
            "error_count": 0,
            "execution_log": [],
        }
    )

    print(f"\nInput: 'hello'")
    print(f"Output: {result['messages'][-1].content}")
    print(f"Execution log: {result['execution_log']}")

    print("✓ Example 8 completed successfully")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "LangGraph Complete Implementation Examples".center(78) + "║")
    print("║" + "Author: Shuvam Banerji Seal".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        example_1_basic_workflow()
        example_2_state_management()
        example_3_checkpoint_management()
        example_4_conditional_routing()
        example_5_linear_subgraphs()
        example_6_multi_agent_coordination()
        example_7_complex_composition()
        example_8_factory_functions()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  ✓ LangGraph enables stateful, deterministic AI workflows")
        print("  ✓ State management provides validation and checkpointing")
        print("  ✓ Subgraphs enable reusable, composable workflow components")
        print("  ✓ Multi-agent coordination simplifies complex systems")
        print("  ✓ Proper structure scales to production use cases")
        print("\nNext Steps:")
        print("  1. Review the src/ modules for detailed implementations")
        print("  2. Adapt examples for your specific use cases")
        print("  3. Implement persistence and monitoring")
        print("  4. Deploy to production with proper error handling")
        print()

    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
