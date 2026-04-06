"""
LangGraph Workflow Foundations Module

This module provides foundational classes and utilities for building stateful
workflow graphs using LangGraph. It includes utilities for state graph creation,
node definition, conditional edge routing, graph compilation, and error handling.

Author: Shuvam Banerji Seal
Source: https://tutorialq.com/ai/frameworks/langgraph-stateful-workflows
Source: https://medium.com/@khanbasil2002/building-intelligent-workflows-with-langgraph-from-simple-graphs-to-complex-ai-agents-9ed74c22a6ad
"""

from typing import Annotated, Callable, Dict, Any, Literal, Optional, TypedDict, Union
from typing_extensions import TypedDict as ExtTypeDict
from abc import ABC, abstractmethod
from enum import Enum
import logging
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NodeType(Enum):
    """Enumeration of node types in the workflow."""

    PROCESSOR = "processor"
    DECISION = "decision"
    TOOL = "tool"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"


class WorkflowState(TypedDict):
    """
    Base state schema for LangGraph workflows.

    Attributes:
        messages: List of chat messages with automatic message appending.
        state_id: Unique identifier for the workflow execution.
        metadata: Dictionary for storing arbitrary workflow metadata.
        error_count: Number of errors encountered during execution.
        execution_log: List of execution events for debugging.
    """

    messages: Annotated[list, add_messages]
    state_id: str
    metadata: Dict[str, Any]
    error_count: int
    execution_log: list[str]


class Node(ABC):
    """
    Abstract base class for workflow nodes.

    Defines the interface that all node implementations must follow.
    Each node is responsible for a specific task in the workflow.
    """

    def __init__(self, name: str, node_type: NodeType):
        """
        Initialize a workflow node.

        Args:
            name: Unique identifier for the node.
            node_type: Type of node (processor, decision, tool, etc).
        """
        self.name = name
        self.node_type = node_type
        self.created_at = datetime.now()

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node's logic.

        Args:
            state: Current workflow state.

        Returns:
            Dictionary containing updates to merge into the workflow state.
        """
        pass

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the node callable."""
        try:
            result = self.execute(state)
            # Log execution
            execution_log = state.get("execution_log", [])
            execution_log.append(
                f"[{datetime.now().isoformat()}] Node '{self.name}' executed successfully"
            )
            return {**result, "execution_log": execution_log}
        except Exception as e:
            logger.error(f"Error in node '{self.name}': {str(e)}")
            error_count = state.get("error_count", 0) + 1
            execution_log = state.get("execution_log", [])
            execution_log.append(
                f"[{datetime.now().isoformat()}] Node '{self.name}' failed: {str(e)}"
            )
            return {
                "error_count": error_count,
                "execution_log": execution_log,
                "messages": [AIMessage(content=f"Error in {self.name}: {str(e)}")],
            }


class ProcessorNode(Node):
    """
    Processor node that executes a callable function.

    This is the most common node type for implementing business logic.
    """

    def __init__(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Initialize a processor node.

        Args:
            name: Unique node identifier.
            func: Callable that takes state and returns updated state.
        """
        super().__init__(name, NodeType.PROCESSOR)
        self.func = func

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the processor function."""
        return self.func(state)


class ConditionalRouter:
    """
    Handles conditional routing in workflows.

    Provides a pattern for deciding which node to execute next based on
    current state. Useful for implementing decision trees and branching logic.

    Example:
        router = ConditionalRouter()
        router.add_route("high_value", lambda s: s.get("priority") == "high")
        router.add_route("normal", lambda s: s.get("priority") == "normal")
        router.add_default_route("low")

        next_node = router.route(current_state)
    """

    def __init__(self):
        """Initialize the conditional router."""
        self.routes: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self.default_route: Optional[str] = None

    def add_route(
        self, route_name: str, condition: Callable[[Dict[str, Any]], bool]
    ) -> "ConditionalRouter":
        """
        Add a conditional route.

        Args:
            route_name: Name of the route destination.
            condition: Callable that returns True if this route should be taken.

        Returns:
            Self for method chaining.
        """
        self.routes[route_name] = condition
        return self

    def add_default_route(self, route_name: str) -> "ConditionalRouter":
        """
        Set the default route if no conditions match.

        Args:
            route_name: Name of the default route destination.

        Returns:
            Self for method chaining.
        """
        self.default_route = route_name
        return self

    def route(self, state: Dict[str, Any]) -> str:
        """
        Determine which route to take based on current state.

        Args:
            state: Current workflow state.

        Returns:
            Name of the route to take.

        Raises:
            ValueError: If no matching route found and no default route set.
        """
        for route_name, condition in self.routes.items():
            if condition(state):
                logger.debug(f"Routing to: {route_name}")
                return route_name

        if self.default_route:
            logger.debug(f"Using default route: {self.default_route}")
            return self.default_route

        raise ValueError("No matching route found and no default route set")


class WorkflowBuilder:
    """
    Builder pattern for constructing LangGraph workflows.

    Provides a fluent API for building complex workflows with nodes,
    edges, conditional routing, and proper error handling.

    Example:
        builder = WorkflowBuilder(initial_state_schema)
        builder.add_node("process", ProcessorNode("process", func))
        builder.add_node("decide", decision_node)
        builder.add_edge(START, "process")
        builder.add_conditional_edge("process", router)
        builder.add_edge("decide", END)

        workflow = builder.build()
    """

    def __init__(self, state_schema: type = WorkflowState):
        """
        Initialize the workflow builder.

        Args:
            state_schema: TypedDict class defining the state structure.
        """
        self.state_schema = state_schema
        self.graph = StateGraph(state_schema)
        self.nodes: Dict[str, Node] = {}
        self.compiled = False

    def add_node(self, name: str, node: Union[Node, Callable]) -> "WorkflowBuilder":
        """
        Add a node to the workflow.

        Args:
            name: Unique node identifier.
            node: Node instance or callable function.

        Returns:
            Self for method chaining.
        """
        if isinstance(node, Node):
            self.nodes[name] = node
            self.graph.add_node(name, node)
        else:
            # Wrap callable in ProcessorNode
            self.graph.add_node(name, node)

        logger.info(f"Added node: {name}")
        return self

    def add_edge(self, source: str, destination: str) -> "WorkflowBuilder":
        """
        Add a fixed edge between two nodes.

        Args:
            source: Source node name (or START).
            destination: Destination node name (or END).

        Returns:
            Self for method chaining.
        """
        self.graph.add_edge(source, destination)
        logger.info(f"Added edge: {source} -> {destination}")
        return self

    def add_conditional_edge(
        self,
        source: str,
        router: ConditionalRouter,
        default_destination: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """
        Add a conditional edge with routing logic.

        Args:
            source: Source node name.
            router: ConditionalRouter instance.
            default_destination: Optional default destination if routing fails.

        Returns:
            Self for method chaining.
        """

        def routing_func(state: Dict[str, Any]) -> str:
            try:
                return router.route(state)
            except ValueError as e:
                if default_destination:
                    logger.warning(
                        f"Routing failed: {str(e)}, using default: {default_destination}"
                    )
                    return default_destination
                raise

        # Get all possible routes for the edge mapping
        route_mapping = {name: name for name in router.routes.keys()}
        if router.default_route:
            route_mapping[router.default_route] = router.default_route

        self.graph.add_conditional_edges(source, routing_func, route_mapping)
        logger.info(f"Added conditional edge from: {source}")
        return self

    def set_entry_point(self, node_name: str) -> "WorkflowBuilder":
        """
        Set the workflow entry point.

        Args:
            node_name: Name of the starting node.

        Returns:
            Self for method chaining.
        """
        self.graph.set_entry_point(node_name)
        logger.info(f"Set entry point: {node_name}")
        return self

    def build(self) -> "CompiledWorkflow":
        """
        Compile and return the workflow.

        Returns:
            CompiledWorkflow instance ready for execution.
        """
        compiled_graph = self.graph.compile()
        self.compiled = True
        logger.info("Workflow compiled successfully")
        return CompiledWorkflow(compiled_graph, self.state_schema)


class CompiledWorkflow:
    """
    Represents a compiled, executable LangGraph workflow.

    Provides methods for invoking the workflow and managing execution.
    """

    def __init__(self, graph: Any, state_schema: type):
        """
        Initialize a compiled workflow.

        Args:
            graph: Compiled LangGraph StateGraph.
            state_schema: State schema TypedDict.
        """
        self.graph = graph
        self.state_schema = state_schema
        self.execution_history: list[Dict[str, Any]] = []

    def invoke(
        self, initial_state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the workflow with given initial state.

        Args:
            initial_state: Starting state for the workflow.
            config: Optional configuration dict (e.g., thread_id for checkpointing).

        Returns:
            Final state after workflow execution.
        """
        logger.info(f"Starting workflow execution with config: {config}")

        # Ensure required fields exist
        if "messages" not in initial_state:
            initial_state["messages"] = []
        if "state_id" not in initial_state:
            initial_state["state_id"] = datetime.now().isoformat()
        if "metadata" not in initial_state:
            initial_state["metadata"] = {}
        if "error_count" not in initial_state:
            initial_state["error_count"] = 0
        if "execution_log" not in initial_state:
            initial_state["execution_log"] = []

        try:
            result = self.graph.invoke(initial_state, config=config)
            self.execution_history.append(result)
            logger.info(
                f"Workflow completed successfully. Execution log: {result.get('execution_log', [])}"
            )
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise

    def stream(
        self, initial_state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ):
        """
        Stream workflow execution updates.

        Args:
            initial_state: Starting state for the workflow.
            config: Optional configuration dict.

        Yields:
            Updates from each node execution.
        """
        logger.info("Starting streaming workflow execution")
        for update in self.graph.stream(initial_state, config=config):
            yield update

    def get_graph_mermaid(self) -> str:
        """
        Get Mermaid diagram representation of the workflow.

        Returns:
            Mermaid diagram string.
        """
        return self.graph.get_graph().draw_mermaid()

    def get_execution_history(self) -> list[Dict[str, Any]]:
        """
        Get history of all workflow executions.

        Returns:
            List of execution results.
        """
        return self.execution_history


class ErrorHandlingStrategy(ABC):
    """Abstract base for error handling strategies."""

    @abstractmethod
    def handle(self, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an error and return state updates.

        Args:
            error: The exception that occurred.
            state: Current workflow state.

        Returns:
            State updates to apply.
        """
        pass


class RetryStrategy(ErrorHandlingStrategy):
    """
    Retry strategy that logs errors and increments error count.

    Useful for transient failures that might succeed on retry.
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize retry strategy.

        Args:
            max_retries: Maximum number of retry attempts.
        """
        self.max_retries = max_retries

    def handle(self, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error by logging and incrementing counter."""
        error_count = state.get("error_count", 0) + 1

        if error_count >= self.max_retries:
            raise error

        return {"error_count": error_count}


class FallbackStrategy(ErrorHandlingStrategy):
    """
    Fallback strategy that returns a default response on error.

    Useful for graceful degradation.
    """

    def __init__(self, fallback_response: str):
        """
        Initialize fallback strategy.

        Args:
            fallback_response: Default message to return on error.
        """
        self.fallback_response = fallback_response

    def handle(self, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error by returning fallback response."""
        return {
            "messages": [AIMessage(content=self.fallback_response)],
            "error_count": state.get("error_count", 0) + 1,
        }


def create_simple_workflow(
    initial_message: str, processing_func: Callable[[str], str]
) -> CompiledWorkflow:
    """
    Factory function to create a simple linear workflow.

    Creates a workflow with an input node, processor node, and output node.

    Args:
        initial_message: Starting message for the workflow.
        processing_func: Function to process the message.

    Returns:
        CompiledWorkflow ready for execution.

    Example:
        def my_processor(message: str) -> str:
            return message.upper()

        workflow = create_simple_workflow("hello", my_processor)
        result = workflow.invoke({"messages": [HumanMessage(content="hello")]})
    """

    def input_node(state: WorkflowState) -> dict:
        """Input processing node."""
        return {"messages": state.get("messages", [])}

    def processor_node(state: WorkflowState) -> dict:
        """Processing node."""
        if state.get("messages"):
            last_message = state["messages"][-1]
            if hasattr(last_message, "content"):
                processed = processing_func(last_message.content)
                return {"messages": [AIMessage(content=processed)]}
        return state

    def output_node(state: WorkflowState) -> dict:
        """Output node."""
        return state

    builder = WorkflowBuilder(WorkflowState)
    builder.add_node("input", input_node)
    builder.add_node("processor", processor_node)
    builder.add_node("output", output_node)
    builder.set_entry_point("input")
    builder.add_edge("input", "processor")
    builder.add_edge("processor", "output")
    builder.add_edge("output", END)

    return builder.build()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create a simple workflow
    workflow = create_simple_workflow("test", lambda x: f"Processed: {x.upper()}")

    # Execute workflow
    result = workflow.invoke(
        {
            "messages": [HumanMessage(content="hello world")],
            "state_id": "test-001",
            "metadata": {"source": "example"},
            "error_count": 0,
            "execution_log": [],
        }
    )

    print(f"Final messages: {result['messages']}")
    print(f"Execution log: {result['execution_log']}")
