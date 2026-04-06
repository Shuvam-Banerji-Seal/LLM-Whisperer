"""
LangGraph State Management Module

This module provides comprehensive state management utilities for LangGraph workflows,
including state schema definition, state persistence, multi-agent coordination,
state validation, and checkpoint management.

Author: Shuvam Banerji Seal
Source: https://wiki.tapnex.tech/articles/en/technology/langgraph-the-complete-guide-to-building-stateful-multi-agent-ai-workflows-2025
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
    Union,
    Annotated,
    Generic,
    TypeVar,
)
from typing_extensions import TypedDict as ExtTypeDict
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import json
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass, field, asdict
from copy import deepcopy

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T = TypeVar("T")


class StateChangeType(Enum):
    """Types of state changes that can occur."""

    CREATED = "created"
    UPDATED = "updated"
    MERGED = "merged"
    VALIDATED = "validated"
    CHECKPOINT = "checkpoint"


@dataclass
class StateCheckpoint:
    """
    Represents a saved checkpoint of workflow state.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        timestamp: When the checkpoint was created.
        state_data: The actual state data.
        node_name: Which node created this checkpoint.
        metadata: Additional context about the checkpoint.
    """

    checkpoint_id: str
    timestamp: datetime
    state_data: Dict[str, Any]
    node_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp.isoformat(),
            "state_data": self.state_data,
            "node_name": self.node_name,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StateCheckpoint":
        """Create checkpoint from dictionary."""
        return StateCheckpoint(
            checkpoint_id=data["checkpoint_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state_data=data["state_data"],
            node_name=data["node_name"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class StateChange:
    """
    Records a change to the workflow state.

    Attributes:
        change_type: Type of change (created, updated, merged, etc).
        timestamp: When the change occurred.
        affected_keys: Which state keys were affected.
        old_values: Previous values of affected keys.
        new_values: New values of affected keys.
        change_source: Which node/component made the change.
    """

    change_type: StateChangeType
    timestamp: datetime
    affected_keys: List[str]
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    change_source: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert change record to dictionary."""
        return {
            "change_type": self.change_type.value,
            "timestamp": self.timestamp.isoformat(),
            "affected_keys": self.affected_keys,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "change_source": self.change_source,
        }


class StateValidator(ABC):
    """Abstract base class for state validation."""

    @abstractmethod
    def validate(self, state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate the state.

        Args:
            state: State to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        pass


class SchemaValidator(StateValidator):
    """
    Validates state against a schema definition.

    Ensures required keys are present and have correct types.
    """

    def __init__(
        self,
        required_keys: Dict[str, type],
        optional_keys: Optional[Dict[str, type]] = None,
    ):
        """
        Initialize schema validator.

        Args:
            required_keys: Dict of key -> expected type for required keys.
            optional_keys: Dict of key -> expected type for optional keys.
        """
        self.required_keys = required_keys
        self.optional_keys = optional_keys or {}

    def validate(self, state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate state against schema."""
        # Check required keys
        for key, expected_type in self.required_keys.items():
            if key not in state:
                return False, f"Missing required key: {key}"
            if not isinstance(state[key], expected_type):
                return (
                    False,
                    f"Key '{key}' has wrong type. Expected {expected_type}, got {type(state[key])}",
                )

        # Check optional keys if present
        for key, expected_type in self.optional_keys.items():
            if key in state and not isinstance(state[key], expected_type):
                return (
                    False,
                    f"Key '{key}' has wrong type. Expected {expected_type}, got {type(state[key])}",
                )

        return True, None


class RangeValidator(StateValidator):
    """Validates numeric state values are within acceptable ranges."""

    def __init__(self, constraints: Dict[str, tuple[float, float]]):
        """
        Initialize range validator.

        Args:
            constraints: Dict of key -> (min_val, max_val) ranges.
        """
        self.constraints = constraints

    def validate(self, state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate values are within ranges."""
        for key, (min_val, max_val) in self.constraints.items():
            if key in state:
                value = state[key]
                if not isinstance(value, (int, float)):
                    return False, f"Key '{key}' is not numeric"
                if not (min_val <= value <= max_val):
                    return (
                        False,
                        f"Key '{key}' value {value} not in range [{min_val}, {max_val}]",
                    )

        return True, None


class StateSchema:
    """
    Defines and manages the schema for workflow state.

    Provides type safety and validation for state operations.

    Example:
        schema = StateSchema()
        schema.add_field("user_id", str, required=True)
        schema.add_field("messages", list, required=True)
        schema.add_field("confidence", float, required=False, default=0.0)
    """

    def __init__(self, name: str = "WorkflowState"):
        """
        Initialize state schema.

        Args:
            name: Name of the state schema.
        """
        self.name = name
        self.fields: Dict[str, Dict[str, Any]] = {}
        self.validators: List[StateValidator] = []

    def add_field(
        self,
        field_name: str,
        field_type: type,
        required: bool = False,
        default: Any = None,
        description: str = "",
    ) -> "StateSchema":
        """
        Add a field to the schema.

        Args:
            field_name: Name of the field.
            field_type: Expected type of the field.
            required: Whether field is required.
            default: Default value if not provided.
            description: Human-readable description.

        Returns:
            Self for method chaining.
        """
        self.fields[field_name] = {
            "type": field_type,
            "required": required,
            "default": default,
            "description": description,
        }
        logger.info(f"Added field '{field_name}' to schema '{self.name}'")
        return self

    def add_validator(self, validator: StateValidator) -> "StateSchema":
        """
        Add a validator to the schema.

        Args:
            validator: StateValidator instance.

        Returns:
            Self for method chaining.
        """
        self.validators.append(validator)
        return self

    def validate(self, state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate state against schema and all validators.

        Args:
            state: State to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate required fields
        for field_name, field_info in self.fields.items():
            if field_info["required"] and field_name not in state:
                return False, f"Required field '{field_name}' missing"

            if field_name in state:
                expected_type = field_info["type"]
                if not isinstance(state[field_name], expected_type):
                    return False, (
                        f"Field '{field_name}' has wrong type. "
                        f"Expected {expected_type}, got {type(state[field_name])}"
                    )

        # Run custom validators
        for validator in self.validators:
            is_valid, error_msg = validator.validate(state)
            if not is_valid:
                return False, error_msg

        return True, None

    def get_initial_state(self) -> Dict[str, Any]:
        """
        Create initial state with default values.

        Returns:
            Dictionary with all fields set to their defaults.
        """
        initial = {}
        for field_name, field_info in self.fields.items():
            initial[field_name] = field_info.get("default")
        return initial

    def to_typed_dict(self) -> type:
        """
        Generate a TypedDict class from this schema.

        Returns:
            A TypedDict class matching this schema.
        """
        annotations = {name: info["type"] for name, info in self.fields.items()}
        return type(self.name, (dict,), {"__annotations__": annotations})


class StateManager:
    """
    Manages workflow state with history, validation, and checkpointing.

    Provides centralized state management with features like:
    - State validation
    - Change tracking
    - Checkpoint management
    - State retrieval

    Example:
        manager = StateManager(schema)
        manager.set_state(initial_state)
        manager.update_state({"user_id": 123})
        checkpoint = manager.create_checkpoint("after_init", "init_node")
    """

    def __init__(self, schema: StateSchema):
        """
        Initialize state manager.

        Args:
            schema: StateSchema defining valid state structure.
        """
        self.schema = schema
        self.current_state: Dict[str, Any] = {}
        self.state_history: List[Dict[str, Any]] = []
        self.change_history: List[StateChange] = []
        self.checkpoints: Dict[str, StateCheckpoint] = {}

    def set_state(self, state: Dict[str, Any], source: str = "initialization") -> bool:
        """
        Set the current state with validation.

        Args:
            state: New state dictionary.
            source: Source/context of state change.

        Returns:
            True if state was set successfully.

        Raises:
            ValueError: If state fails validation.
        """
        is_valid, error_msg = self.schema.validate(state)
        if not is_valid:
            raise ValueError(f"State validation failed: {error_msg}")

        old_state = deepcopy(self.current_state)
        self.current_state = deepcopy(state)

        # Record change
        change = StateChange(
            change_type=StateChangeType.CREATED
            if not old_state
            else StateChangeType.UPDATED,
            timestamp=datetime.now(),
            affected_keys=list(state.keys()),
            old_values=old_state,
            new_values=state,
            change_source=source,
        )
        self.change_history.append(change)
        self.state_history.append(deepcopy(state))

        logger.info(f"State set from {source}")
        return True

    def update_state(
        self, updates: Dict[str, Any], source: str = "update", merge: bool = True
    ) -> bool:
        """
        Update the current state.

        Args:
            updates: Dictionary of key-value pairs to update.
            source: Source of the update.
            merge: If True, merge with existing state; if False, replace.

        Returns:
            True if update was successful.

        Raises:
            ValueError: If resulting state fails validation.
        """
        old_state = deepcopy(self.current_state)

        if merge:
            new_state = {**self.current_state, **updates}
        else:
            new_state = updates

        is_valid, error_msg = self.schema.validate(new_state)
        if not is_valid:
            raise ValueError(f"State update failed validation: {error_msg}")

        self.current_state = new_state

        # Record change
        change = StateChange(
            change_type=StateChangeType.MERGED if merge else StateChangeType.UPDATED,
            timestamp=datetime.now(),
            affected_keys=list(updates.keys()),
            old_values={k: old_state.get(k) for k in updates.keys()},
            new_values=updates,
            change_source=source,
        )
        self.change_history.append(change)
        self.state_history.append(deepcopy(self.current_state))

        logger.info(f"State updated from {source}")
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return deepcopy(self.current_state)

    def create_checkpoint(
        self,
        checkpoint_id: str,
        node_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateCheckpoint:
        """
        Create a checkpoint of current state.

        Args:
            checkpoint_id: Unique identifier for checkpoint.
            node_name: Name of node creating checkpoint.
            metadata: Optional additional metadata.

        Returns:
            StateCheckpoint instance.
        """
        checkpoint = StateCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            state_data=deepcopy(self.current_state),
            node_name=node_name,
            metadata=metadata or {},
        )
        self.checkpoints[checkpoint_id] = checkpoint
        logger.info(f"Checkpoint created: {checkpoint_id}")
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore.

        Returns:
            True if restore was successful.

        Raises:
            KeyError: If checkpoint doesn't exist.
        """
        if checkpoint_id not in self.checkpoints:
            raise KeyError(f"Checkpoint '{checkpoint_id}' not found")

        checkpoint = self.checkpoints[checkpoint_id]
        self.set_state(
            checkpoint.state_data, source=f"checkpoint_restore:{checkpoint_id}"
        )
        logger.info(f"State restored from checkpoint: {checkpoint_id}")
        return True

    def get_change_history(self) -> List[StateChange]:
        """Get all state changes."""
        return deepcopy(self.change_history)

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get history of all states."""
        return deepcopy(self.state_history)

    def export_checkpoints(self, file_path: Union[str, Path]) -> None:
        """
        Export all checkpoints to a file.

        Args:
            file_path: Path to save checkpoints.
        """
        file_path = Path(file_path)
        checkpoints_data = {cid: cp.to_dict() for cid, cp in self.checkpoints.items()}

        with open(file_path, "w") as f:
            json.dump(checkpoints_data, f, indent=2)

        logger.info(f"Checkpoints exported to {file_path}")

    def import_checkpoints(self, file_path: Union[str, Path]) -> None:
        """
        Import checkpoints from a file.

        Args:
            file_path: Path to load checkpoints from.
        """
        file_path = Path(file_path)

        with open(file_path, "r") as f:
            checkpoints_data = json.load(f)

        for cid, data in checkpoints_data.items():
            self.checkpoints[cid] = StateCheckpoint.from_dict(data)

        logger.info(f"Checkpoints imported from {file_path}")


class CheckpointManager:
    """
    Manages workflow checkpoints using different storage backends.

    Supports both in-memory and persistent (SQLite) checkpointing.

    Example:
        manager = CheckpointManager(backend="sqlite", db_path="workflow.db")
        checkpointer = manager.get_checkpointer()

        # Use with graph
        agent = graph.compile(checkpointer=checkpointer)
    """

    def __init__(self, backend: str = "memory", db_path: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            backend: "memory" or "sqlite".
            db_path: Path to SQLite database (required for sqlite backend).
        """
        self.backend = backend
        self.db_path = db_path
        self._checkpointer = None

    def get_checkpointer(self):
        """
        Get the checkpointer instance.

        Returns:
            MemorySaver or SqliteSaver instance.
        """
        if self._checkpointer is not None:
            return self._checkpointer

        if self.backend == "memory":
            self._checkpointer = MemorySaver()
            logger.info("Using in-memory checkpointer")
        elif self.backend == "sqlite":
            if not self.db_path:
                raise ValueError("db_path required for sqlite backend")
            self._checkpointer = SqliteSaver.from_conn_string(self.db_path)
            logger.info(f"Using SQLite checkpointer: {self.db_path}")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return self._checkpointer

    def create_config(self, thread_id: str) -> Dict[str, Any]:
        """
        Create a configuration dict for a workflow execution thread.

        Args:
            thread_id: Unique identifier for the execution thread.

        Returns:
            Configuration dict for use with graph.invoke().
        """
        return {"configurable": {"thread_id": thread_id}}


class MultiAgentStateCoordinator:
    """
    Coordinates state across multiple agents in a workflow.

    Handles state merging, conflict resolution, and synchronization.

    Example:
        coordinator = MultiAgentStateCoordinator()
        coordinator.register_agent("research_agent", research_state_schema)
        coordinator.register_agent("writing_agent", writing_state_schema)

        merged_state = coordinator.merge_states({
            "research_agent": research_result,
            "writing_agent": writing_result
        })
    """

    def __init__(self):
        """Initialize multi-agent state coordinator."""
        self.agents: Dict[str, str] = {}  # agent_name -> state_key mapping
        self.state_schemas: Dict[str, StateSchema] = {}
        self.merge_strategies: Dict[str, Callable] = {}

    def register_agent(
        self,
        agent_name: str,
        state_key: str,
        schema: StateSchema,
        merge_strategy: Optional[Callable] = None,
    ) -> "MultiAgentStateCoordinator":
        """
        Register an agent and its state schema.

        Args:
            agent_name: Name of the agent.
            state_key: Key in the shared state where agent state lives.
            schema: StateSchema for this agent's state.
            merge_strategy: Optional custom merge function.

        Returns:
            Self for method chaining.
        """
        self.agents[agent_name] = state_key
        self.state_schemas[agent_name] = schema
        if merge_strategy:
            self.merge_strategies[agent_name] = merge_strategy
        return self

    def merge_states(self, agent_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge states from multiple agents.

        Args:
            agent_states: Dict of agent_name -> agent_state.

        Returns:
            Merged state dictionary.
        """
        merged = {}

        for agent_name, agent_state in agent_states.items():
            if agent_name not in self.agents:
                logger.warning(f"Unknown agent: {agent_name}")
                continue

            state_key = self.agents[agent_name]

            # Apply merge strategy if available
            if agent_name in self.merge_strategies:
                merged[state_key] = self.merge_strategies[agent_name](agent_state)
            else:
                # Default: simple assignment
                merged[state_key] = agent_state

        return merged

    def validate_all_agents(
        self, agent_states: Dict[str, Dict[str, Any]]
    ) -> tuple[bool, Dict[str, Optional[str]]]:
        """
        Validate all agent states.

        Args:
            agent_states: Dict of agent_name -> agent_state.

        Returns:
            Tuple of (all_valid, validation_results).
        """
        results = {}
        all_valid = True

        for agent_name, agent_state in agent_states.items():
            if agent_name not in self.state_schemas:
                results[agent_name] = f"No schema for agent: {agent_name}"
                all_valid = False
                continue

            schema = self.state_schemas[agent_name]
            is_valid, error_msg = schema.validate(agent_state)
            results[agent_name] = error_msg if not is_valid else None
            if not is_valid:
                all_valid = False

        return all_valid, results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create and use state manager
    schema = StateSchema("TestState")
    schema.add_field("user_id", int, required=True)
    schema.add_field("messages", list, required=True, default=[])
    schema.add_field("confidence", float, required=False, default=0.5)

    manager = StateManager(schema)
    manager.set_state({"user_id": 123, "messages": [], "confidence": 0.5})
    manager.update_state({"confidence": 0.9})

    checkpoint = manager.create_checkpoint("checkpoint_1", "test_node")
    print(f"Checkpoint created: {checkpoint.checkpoint_id}")
    print(f"Current state: {manager.get_state()}")
