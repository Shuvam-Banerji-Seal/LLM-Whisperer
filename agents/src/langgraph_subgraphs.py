"""
LangGraph Subgraphs Module

This module provides patterns and utilities for building composable, reusable
workflow subgraphs. Subgraphs allow decomposition of complex workflows into
smaller, testable, and reusable components.

Author: Shuvam Banerji Seal
Source: https://machinelearningplus.com/gen-ai/langgraph-subgraphs-composing-reusable-workflows/
"""

from typing import Any, Callable, Dict, List, Optional, Type, TypedDict, Union, Tuple
from typing_extensions import TypedDict as ExtTypeDict
from abc import ABC, abstractmethod
from enum import Enum
import logging
from datetime import datetime
from dataclasses import dataclass, field
from copy import deepcopy

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SubgraphType(Enum):
    """Types of subgraph patterns."""

    LINEAR = "linear"  # Sequential nodes
    CONDITIONAL = "conditional"  # Decision trees
    PARALLEL = "parallel"  # Multiple paths
    LOOP = "loop"  # Cyclic patterns
    SUPERVISED = "supervised"  # Supervisor coordination
    HIERARCHICAL = "hierarchical"  # Multi-level composition


@dataclass
class SubgraphMetadata:
    """Metadata describing a subgraph."""

    name: str
    description: str
    subgraph_type: SubgraphType
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    author: str = "Unknown"
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.subgraph_type.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "author": self.author,
            "dependencies": self.dependencies,
        }


class BaseSubgraph(ABC):
    """
    Abstract base class for all subgraphs.

    Defines the interface that all subgraph implementations must follow.
    """

    def __init__(
        self,
        name: str,
        state_schema: Type[TypedDict],
        subgraph_type: SubgraphType = SubgraphType.LINEAR,
    ):
        """
        Initialize base subgraph.

        Args:
            name: Unique identifier for the subgraph.
            state_schema: TypedDict defining state structure.
            subgraph_type: Type of subgraph pattern.
        """
        self.name = name
        self.state_schema = state_schema
        self.subgraph_type = subgraph_type
        self.metadata = SubgraphMetadata(
            name=name, description="", subgraph_type=subgraph_type
        )
        self._graph: Optional[StateGraph] = None
        self._compiled = None

    @abstractmethod
    def build(self) -> "BaseSubgraph":
        """Build the subgraph structure. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def compile(self):
        """Compile the subgraph into an executable form."""
        pass

    def set_metadata(
        self, description: str = "", author: str = "", tags: Optional[List[str]] = None
    ) -> "BaseSubgraph":
        """
        Set metadata for the subgraph.

        Args:
            description: Description of what the subgraph does.
            author: Author of the subgraph.
            tags: List of tags/keywords.

        Returns:
            Self for method chaining.
        """
        if description:
            self.metadata.description = description
        if author:
            self.metadata.author = author
        if tags:
            self.metadata.tags = tags
        return self

    def add_dependency(self, dependency: str) -> "BaseSubgraph":
        """
        Add a dependency to this subgraph.

        Args:
            dependency: Name of a required dependency.

        Returns:
            Self for method chaining.
        """
        if dependency not in self.metadata.dependencies:
            self.metadata.dependencies.append(dependency)
        return self


class LinearSubgraph(BaseSubgraph):
    """
    Linear subgraph with nodes connected in sequence.

    Useful for pipeline-style workflows where each step depends on
    the output of the previous step.

    Example:
        subgraph = LinearSubgraph("processing", MyState)
        subgraph.add_node("validate", validate_func)
        subgraph.add_node("transform", transform_func)
        subgraph.add_node("store", store_func)
        compiled = subgraph.build().compile()
    """

    def __init__(self, name: str, state_schema: Type[TypedDict]):
        """Initialize linear subgraph."""
        super().__init__(name, state_schema, SubgraphType.LINEAR)
        self.nodes: List[Tuple[str, Callable]] = []

    def add_node(self, node_name: str, node_func: Callable) -> "LinearSubgraph":
        """
        Add a node to the linear sequence.

        Args:
            node_name: Unique name for the node.
            node_func: Callable that processes state.

        Returns:
            Self for method chaining.
        """
        self.nodes.append((node_name, node_func))
        logger.info(f"Added node '{node_name}' to linear subgraph '{self.name}'")
        return self

    def build(self) -> "LinearSubgraph":
        """Build the linear graph structure."""
        self._graph = StateGraph(self.state_schema)

        if not self.nodes:
            raise ValueError("LinearSubgraph must have at least one node")

        # Add all nodes
        for node_name, node_func in self.nodes:
            self._graph.add_node(node_name, node_func)

        # Connect entry point
        self._graph.add_edge(START, self.nodes[0][0])

        # Connect nodes in sequence
        for i in range(len(self.nodes) - 1):
            current_node = self.nodes[i][0]
            next_node = self.nodes[i + 1][0]
            self._graph.add_edge(current_node, next_node)

        # Connect last node to exit
        self._graph.add_edge(self.nodes[-1][0], END)

        logger.info(f"Built linear subgraph '{self.name}' with {len(self.nodes)} nodes")
        return self

    def compile(self):
        """Compile the linear subgraph."""
        if self._graph is None:
            self.build()
        self._compiled = self._graph.compile()
        return self._compiled


class ConditionalSubgraph(BaseSubgraph):
    """
    Subgraph with conditional branching logic.

    Implements decision trees where different paths are taken
    based on state conditions.

    Example:
        subgraph = ConditionalSubgraph("router", MyState)
        subgraph.add_node("classifier", classify_func)
        subgraph.add_branch(
            "high_priority",
            lambda s: s.get("priority") == "high",
            "process_urgent"
        )
        subgraph.add_branch(
            "normal",
            lambda s: s.get("priority") == "normal",
            "process_normal"
        )
    """

    def __init__(self, name: str, state_schema: Type[TypedDict]):
        """Initialize conditional subgraph."""
        super().__init__(name, state_schema, SubgraphType.CONDITIONAL)
        self.nodes: Dict[str, Callable] = {}
        self.router_node: Optional[str] = None
        self.branches: Dict[
            str, Tuple[Callable, str]
        ] = {}  # condition -> (func, destination)
        self.default_branch: Optional[str] = None

    def add_node(self, node_name: str, node_func: Callable) -> "ConditionalSubgraph":
        """Add a node to the subgraph."""
        self.nodes[node_name] = node_func
        return self

    def set_router(self, node_name: str, node_func: Callable) -> "ConditionalSubgraph":
        """
        Set the routing node that determines which branch to take.

        Args:
            node_name: Name of the router node.
            node_func: Callable that examines state and returns branch decision.

        Returns:
            Self for method chaining.
        """
        self.router_node = node_name
        self.nodes[node_name] = node_func
        return self

    def add_branch(
        self,
        condition_name: str,
        condition: Callable[[Dict[str, Any]], bool],
        destination: str,
    ) -> "ConditionalSubgraph":
        """
        Add a conditional branch.

        Args:
            condition_name: Name for this branch condition.
            condition: Callable that evaluates to True/False.
            destination: Node to route to if condition is True.

        Returns:
            Self for method chaining.
        """
        self.branches[condition_name] = (condition, destination)
        return self

    def set_default_branch(self, destination: str) -> "ConditionalSubgraph":
        """
        Set default branch if no conditions match.

        Args:
            destination: Default node to route to.

        Returns:
            Self for method chaining.
        """
        self.default_branch = destination
        return self

    def build(self) -> "ConditionalSubgraph":
        """Build the conditional subgraph."""
        self._graph = StateGraph(self.state_schema)

        if not self.router_node:
            raise ValueError("ConditionalSubgraph requires a router node")

        # Add all nodes
        for node_name, node_func in self.nodes.items():
            self._graph.add_node(node_name, node_func)

        # Entry point to router
        self._graph.add_edge(START, self.router_node)

        # Create routing function
        def routing_func(state: Dict[str, Any]) -> str:
            for condition_name, (condition, destination) in self.branches.items():
                if condition(state):
                    logger.debug(f"Routing via condition: {condition_name}")
                    return destination
            if self.default_branch:
                logger.debug(f"Using default branch: {self.default_branch}")
                return self.default_branch
            raise ValueError("No matching condition and no default branch")

        # Add conditional edges from router
        destinations = {dest for _, (_, dest) in self.branches.items()}
        if self.default_branch:
            destinations.add(self.default_branch)

        self._graph.add_conditional_edges(
            self.router_node, routing_func, {dest: dest for dest in destinations}
        )

        # Connect destination nodes to exit
        for dest in destinations:
            self._graph.add_edge(dest, END)

        logger.info(
            f"Built conditional subgraph '{self.name}' with {len(self.branches)} branches"
        )
        return self

    def compile(self):
        """Compile the conditional subgraph."""
        if self._graph is None:
            self.build()
        self._compiled = self._graph.compile()
        return self._compiled


class StateTranslator:
    """
    Translates state between parent and subgraph schemas.

    Used when a subgraph has a different state schema than its parent.
    Handles mapping between different state representations.

    Example:
        translator = StateTranslator()
        translator.map_input("parent_key", "subgraph_key")
        translator.map_output("subgraph_key", "parent_key")

        sub_input = translator.translate_input(parent_state)
        parent_update = translator.translate_output(subgraph_result)
    """

    def __init__(self):
        """Initialize state translator."""
        self.input_mapping: Dict[str, str] = {}  # parent_key -> subgraph_key
        self.output_mapping: Dict[str, str] = {}  # subgraph_key -> parent_key
        self.transformers: Dict[str, Callable] = {}  # key -> transformer function

    def map_input(self, parent_key: str, subgraph_key: str) -> "StateTranslator":
        """
        Map a parent state key to subgraph state key for input.

        Args:
            parent_key: Key in parent state.
            subgraph_key: Key in subgraph state.

        Returns:
            Self for method chaining.
        """
        self.input_mapping[parent_key] = subgraph_key
        return self

    def map_output(self, subgraph_key: str, parent_key: str) -> "StateTranslator":
        """
        Map a subgraph state key to parent state key for output.

        Args:
            subgraph_key: Key in subgraph state.
            parent_key: Key in parent state.

        Returns:
            Self for method chaining.
        """
        self.output_mapping[subgraph_key] = parent_key
        return self

    def add_transformer(
        self, key: str, transformer: Callable[[Any], Any]
    ) -> "StateTranslator":
        """
        Add a transformation function for a key.

        Args:
            key: State key to transform.
            transformer: Function that transforms the value.

        Returns:
            Self for method chaining.
        """
        self.transformers[key] = transformer
        return self

    def translate_input(self, parent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate parent state to subgraph state.

        Args:
            parent_state: State from parent graph.

        Returns:
            Translated state for subgraph.
        """
        subgraph_state = {}

        for parent_key, subgraph_key in self.input_mapping.items():
            if parent_key in parent_state:
                value = parent_state[parent_key]

                # Apply transformer if available
                if parent_key in self.transformers:
                    value = self.transformers[parent_key](value)

                subgraph_state[subgraph_key] = value

        return subgraph_state

    def translate_output(self, subgraph_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate subgraph state to parent state updates.

        Args:
            subgraph_state: Final state from subgraph.

        Returns:
            Updates to apply to parent state.
        """
        parent_updates = {}

        for subgraph_key, parent_key in self.output_mapping.items():
            if subgraph_key in subgraph_state:
                value = subgraph_state[subgraph_key]

                # Apply transformer if available
                if subgraph_key in self.transformers:
                    value = self.transformers[subgraph_key](value)

                parent_updates[parent_key] = value

        return parent_updates


class ComposableSubgraph:
    """
    Wrapper for using a subgraph as a node in a parent graph.

    Handles state translation and integration with parent graphs.

    Example:
        subgraph_compiled = subgraph.build().compile()
        translator = StateTranslator()
        translator.map_input("parent_text", "text")
        translator.map_output("result", "processed_result")

        composable = ComposableSubgraph(
            "processor",
            subgraph_compiled,
            translator
        )

        # Use in parent graph
        parent_builder.add_node("process", composable)
    """

    def __init__(
        self,
        name: str,
        compiled_subgraph: Any,
        translator: Optional[StateTranslator] = None,
    ):
        """
        Initialize composable subgraph.

        Args:
            name: Name for this subgraph usage.
            compiled_subgraph: Compiled subgraph instance.
            translator: Optional StateTranslator for schema mapping.
        """
        self.name = name
        self.compiled_subgraph = compiled_subgraph
        self.translator = translator or StateTranslator()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the subgraph as a callable node.

        Args:
            state: Parent graph state.

        Returns:
            Updates to merge into parent state.
        """
        try:
            # Translate input
            subgraph_input = self.translator.translate_input(state)

            # Execute subgraph
            subgraph_result = self.compiled_subgraph.invoke(subgraph_input)

            # Translate output
            parent_updates = self.translator.translate_output(subgraph_result)

            logger.info(f"ComposableSubgraph '{self.name}' executed successfully")
            return parent_updates

        except Exception as e:
            logger.error(f"Error in ComposableSubgraph '{self.name}': {str(e)}")
            raise


class SubgraphRegistry:
    """
    Registry for managing reusable subgraph templates.

    Allows registration, retrieval, and management of subgraph
    definitions for reuse across projects.

    Example:
        registry = SubgraphRegistry()

        # Register a subgraph template
        registry.register("text_processor", text_processing_subgraph)

        # Retrieve and use
        subgraph = registry.get("text_processor")
        compiled = subgraph.build().compile()
    """

    def __init__(self):
        """Initialize subgraph registry."""
        self.subgraphs: Dict[str, BaseSubgraph] = {}
        self.metadata: Dict[str, SubgraphMetadata] = {}

    def register(
        self, key: str, subgraph: BaseSubgraph, overwrite: bool = False
    ) -> None:
        """
        Register a subgraph template.

        Args:
            key: Unique identifier for the subgraph.
            subgraph: BaseSubgraph instance to register.
            overwrite: If True, overwrite existing registration.

        Raises:
            ValueError: If key already exists and overwrite is False.
        """
        if key in self.subgraphs and not overwrite:
            raise ValueError(
                f"Subgraph '{key}' already registered. Set overwrite=True to replace."
            )

        self.subgraphs[key] = subgraph
        self.metadata[key] = subgraph.metadata
        logger.info(f"Registered subgraph: {key}")

    def get(self, key: str) -> Optional[BaseSubgraph]:
        """
        Retrieve a registered subgraph.

        Args:
            key: Identifier of the subgraph to retrieve.

        Returns:
            BaseSubgraph instance or None if not found.
        """
        subgraph = self.subgraphs.get(key)
        if subgraph is None:
            logger.warning(f"Subgraph '{key}' not found in registry")
        return subgraph

    def list_subgraphs(self) -> List[str]:
        """
        List all registered subgraphs.

        Returns:
            List of registered subgraph keys.
        """
        return list(self.subgraphs.keys())

    def get_metadata(self, key: str) -> Optional[SubgraphMetadata]:
        """Get metadata for a registered subgraph."""
        return self.metadata.get(key)

    def get_by_type(self, subgraph_type: SubgraphType) -> List[str]:
        """
        Get all subgraphs of a specific type.

        Args:
            subgraph_type: SubgraphType to filter by.

        Returns:
            List of matching subgraph keys.
        """
        return [
            key
            for key, sg in self.subgraphs.items()
            if sg.subgraph_type == subgraph_type
        ]

    def get_by_tag(self, tag: str) -> List[str]:
        """
        Get all subgraphs with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of matching subgraph keys.
        """
        return [key for key, metadata in self.metadata.items() if tag in metadata.tags]


# Global registry instance
_global_registry = SubgraphRegistry()


def register_subgraph(
    key: str, subgraph: BaseSubgraph, overwrite: bool = False
) -> None:
    """
    Register a subgraph in the global registry.

    Args:
        key: Unique identifier.
        subgraph: BaseSubgraph instance.
        overwrite: If True, overwrite existing.
    """
    _global_registry.register(key, subgraph, overwrite)


def get_subgraph(key: str) -> Optional[BaseSubgraph]:
    """
    Retrieve a subgraph from the global registry.

    Args:
        key: Subgraph identifier.

    Returns:
        BaseSubgraph instance or None.
    """
    return _global_registry.get(key)


def list_registered_subgraphs() -> List[str]:
    """Get list of all registered subgraphs."""
    return _global_registry.list_subgraphs()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Define a simple state
    class SimpleState(TypedDict):
        input_text: str
        output_text: str

    # Create a linear subgraph
    def transform_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"output_text": state.get("input_text", "").upper()}

    subgraph = LinearSubgraph("transformer", SimpleState)
    subgraph.add_node("transform", transform_node)
    subgraph.set_metadata(
        description="Transforms text to uppercase",
        author="Shuvam Banerji Seal",
        tags=["text", "transformation"],
    )

    # Build and compile
    subgraph.build()
    compiled = subgraph.compile()

    # Register it
    register_subgraph("text_transformer", subgraph)

    print(f"Registered subgraphs: {list_registered_subgraphs()}")
    print(f"Subgraph metadata: {get_subgraph('text_transformer').metadata.to_dict()}")
