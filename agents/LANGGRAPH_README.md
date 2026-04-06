# LangGraph Stateful Multi-Agent Workflows

High-quality, production-ready LangGraph implementations for building stateful, composable AI workflows with advanced patterns for multi-agent systems.

**Author**: Shuvam Banerji Seal  
**Version**: 1.0.0  
**Last Updated**: April 2026

## Overview

This directory contains comprehensive implementations of LangGraph workflows, organized into three core modules:

### 1. **langgraph_workflow.py** - Workflow Foundations
- StateGraph basics and management
- Node definition and composition patterns
- Conditional edge routing
- Graph compilation and execution
- Built-in error handling strategies

### 2. **langgraph_state_management.py** - State Management
- State schema definition with validation
- State persistence and retrieval
- Multi-agent state coordination
- Checkpoint management (memory and SQLite)
- State change tracking and history

### 3. **langgraph_subgraphs.py** - Composable Workflows
- Linear subgraph patterns
- Conditional subgraph patterns
- State translation between schemas
- Composable subgraph wrappers
- Subgraph registry for reusable components

## Key Features

✅ **Type-Safe State Management**
- TypedDict-based state schemas
- Automatic validation
- Type hints throughout

✅ **Production-Ready Error Handling**
- Custom error strategies (Retry, Fallback)
- Comprehensive logging
- Graceful degradation

✅ **Checkpoint & Persistence**
- In-memory checkpoints for development
- SQLite persistence for production
- State history tracking

✅ **Composable Architecture**
- Build complex workflows from simple components
- Reusable subgraph patterns
- Flexible state translation

✅ **Multi-Agent Coordination**
- Parallel agent execution
- State merging strategies
- Conflict resolution

## Installation

```bash
# Install dependencies
pip install langgraph langchain langchain-openai

# Clone and setup
cd /path/to/LLM-Whisperer
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Quick Start

### Basic Workflow

```python
from agents.src import WorkflowBuilder, WorkflowState
from langchain_core.messages import HumanMessage, AIMessage

# Define node functions
def process_message(state):
    msg = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Processed: {msg}")]}

# Build workflow
builder = WorkflowBuilder(WorkflowState)
builder.add_node("process", process_message)
builder.set_entry_point("process")
builder.add_edge("process", "END")

# Compile and execute
workflow = builder.build()
result = workflow.invoke({
    "messages": [HumanMessage(content="hello")],
    "state_id": "test-001",
    "metadata": {},
    "error_count": 0,
    "execution_log": []
})
```

### State Management with Validation

```python
from agents.src import StateSchema, StateManager, SchemaValidator

# Define state schema
schema = StateSchema("UserState")
schema.add_field("user_id", int, required=True)
schema.add_field("confidence", float, default=0.5)

# Add validation
schema.add_validator(SchemaValidator(
    required_keys={"user_id": int},
    optional_keys={"confidence": float}
))

# Manage state
manager = StateManager(schema)
manager.set_state({"user_id": 123, "confidence": 0.9})

# Create checkpoint
checkpoint = manager.create_checkpoint("ckpt_1", "node_name")
```

### Linear Subgraphs

```python
from agents.src import LinearSubgraph
from typing import TypedDict

class TextState(TypedDict):
    input: str
    output: str

# Create subgraph
subgraph = LinearSubgraph("processor", TextState)
subgraph.add_node("clean", lambda s: {"output": s["input"].strip()})
subgraph.add_node("transform", lambda s: {"output": s["output"].upper()})

# Build and use
compiled = subgraph.build().compile()
result = compiled.invoke({"input": "  test  ", "output": ""})
```

## Documentation

### Module Structure

```
agents/
├── src/
│   ├── __init__.py                          # Package exports
│   ├── langgraph_workflow.py               # Core workflow classes
│   ├── langgraph_state_management.py       # State handling
│   ├── langgraph_subgraphs.py              # Composable patterns
│   └── ... (other agent frameworks)
├── langgraph_complete_examples.py          # 8 comprehensive examples
└── LANGGRAPH_README.md                     # This file
```

### Core Classes

#### WorkflowBuilder
Fluent API for constructing workflows:
```python
builder = WorkflowBuilder(StateSchema)
builder.add_node(name, callable_or_node)
builder.add_edge(source, destination)
builder.add_conditional_edge(source, router, destinations)
builder.set_entry_point(node_name)
workflow = builder.build()
```

#### StateManager
Centralized state management with validation:
```python
manager = StateManager(schema)
manager.set_state(state, source="init")
manager.update_state(updates, merge=True)
manager.create_checkpoint(id, node, metadata)
manager.restore_checkpoint(id)
```

#### CheckpointManager
Multi-backend checkpoint support:
```python
# In-memory
memory_mgr = CheckpointManager(backend="memory")

# SQLite (production)
sqlite_mgr = CheckpointManager(
    backend="sqlite",
    db_path="workflow.db"
)
checkpointer = sqlite_mgr.get_checkpointer()
```

#### LinearSubgraph
Sequential node execution:
```python
subgraph = LinearSubgraph(name, state_schema)
subgraph.add_node(name, func)
subgraph.add_node(name2, func2)
subgraph.build()
compiled = subgraph.compile()
```

#### ConditionalSubgraph
Decision-tree routing:
```python
subgraph = ConditionalSubgraph(name, state_schema)
subgraph.set_router("decide", decision_func)
subgraph.add_branch("path1", condition1, "node1")
subgraph.add_branch("path2", condition2, "node2")
subgraph.set_default_branch("default_node")
subgraph.build()
```

#### MultiAgentStateCoordinator
Multi-agent orchestration:
```python
coordinator = MultiAgentStateCoordinator()
coordinator.register_agent("agent1", "state_key1", schema1)
coordinator.register_agent("agent2", "state_key2", schema2)

merged = coordinator.merge_states(agent_results)
is_valid, errors = coordinator.validate_all_agents(results)
```

## Examples

Run the comprehensive examples:

```bash
python agents/langgraph_complete_examples.py
```

This demonstrates:
1. **Basic Workflow Creation** - Simple linear workflows
2. **State Management** - Schema definition and validation
3. **Checkpoint Management** - Persistence strategies
4. **Conditional Routing** - Decision-based routing
5. **Linear Subgraphs** - Reusable components
6. **Multi-Agent Coordination** - State merging
7. **Complex Composition** - Nested subgraphs
8. **Factory Functions** - Rapid prototyping

## Architecture Patterns

### Linear Workflow
```
START → Node1 → Node2 → Node3 → END
```

### Conditional Workflow
```
     ┌→ HighPriority
START → Router ┼→ NormalPriority → END
     └→ LowPriority
```

### Multi-Agent Workflow
```
START → Coordinator ──┬→ ResearchAgent
                      ├→ WritingAgent
                      └→ ReviewAgent
                      
        ↓
        StateAggregator → END
```

### Hierarchical Subgraph Composition
```
Parent Graph
├─ SubGraph 1
│  ├─ Node A
│  └─ Node B
├─ SubGraph 2
│  ├─ Node C (contains SubGraph 3)
│  │  ├─ Node D
│  │  └─ Node E
│  └─ Node F
└─ SubGraph 3
```

## Best Practices

### State Management
- ✓ Use TypedDict for state schemas
- ✓ Define required vs optional fields
- ✓ Add validators early
- ✓ Use checkpoints at critical points
- ✓ Keep state size minimal

### Workflow Design
- ✓ Keep nodes focused and single-responsibility
- ✓ Use subgraphs for reusable components
- ✓ Handle errors gracefully
- ✓ Add comprehensive logging
- ✓ Document node inputs/outputs

### Multi-Agent Systems
- ✓ Isolate agent state when possible
- ✓ Use state translator for schema differences
- ✓ Validate before merging
- ✓ Plan for conflicts
- ✓ Monitor coordination overhead

### Production Deployment
- ✓ Use SQLite checkpointing
- ✓ Implement proper error handling
- ✓ Add monitoring and logging
- ✓ Use connection pooling
- ✓ Test checkpoint recovery

## API Reference

### WorkflowBuilder Methods
- `add_node(name, callable)` - Add a workflow node
- `add_edge(source, destination)` - Connect nodes
- `add_conditional_edge(source, router, destinations)` - Conditional routing
- `set_entry_point(node_name)` - Define workflow start
- `build()` - Compile workflow

### StateManager Methods
- `set_state(state, source)` - Initialize state
- `update_state(updates, source, merge)` - Update state
- `get_state()` - Retrieve current state
- `create_checkpoint(id, node, metadata)` - Save checkpoint
- `restore_checkpoint(id)` - Restore from checkpoint
- `get_change_history()` - Get state changes
- `get_state_history()` - Get state timeline

### Subgraph Methods
- `add_node(name, func)` - Add subgraph node
- `build()` - Structure subgraph
- `compile()` - Create executable
- `set_metadata(description, author, tags)` - Add metadata

## Performance Considerations

- **State Size**: Keep state objects minimal to reduce serialization overhead
- **Checkpoint Frequency**: Balance between recovery capability and I/O cost
- **Subgraph Depth**: Limit nesting to avoid execution complexity
- **Validator Count**: Minimize validators for performance-critical paths
- **Memory vs SQLite**: Use SQLite only when persistence is needed

## Troubleshooting

### Common Issues

**Issue**: State validation fails
- Solution: Check all required fields are present and have correct types

**Issue**: Checkpoint not found
- Solution: Ensure checkpoint_id is correct and backend is configured

**Issue**: Routing fails with "No matching route"
- Solution: Add a default_route or ensure conditions cover all cases

**Issue**: Subgraph state mismatch
- Solution: Use StateTranslator to map between parent and child schemas

## References

- [LangGraph Stateful Workflows](https://tutorialq.com/ai/frameworks/langgraph-stateful-workflows)
- [Building Intelligent Workflows with LangGraph](https://medium.com/@khanbasil2002/building-intelligent-workflows-with-langgraph-from-simple-graphs-to-complex-ai-agents-9ed74c22a6ad)
- [Complete Guide to Stateful Multi-Agent Workflows](https://wiki.tapnex.tech/articles/en/technology/langgraph-the-complete-guide-to-building-stateful-multi-agent-ai-workflows-2025)
- [LangGraph Subgraphs: Composing Reusable Workflows](https://machinelearningplus.com/gen-ai/langgraph-subgraphs-composing-reusable-workflows/)

## Contributing

When adding new features:
1. Follow PEP 8 style guide
2. Add comprehensive docstrings with examples
3. Include type hints for all parameters
4. Add unit tests
5. Update this README

## License

This implementation is part of the LLM-Whisperer project.

## Support

For issues or questions:
1. Check the examples in `langgraph_complete_examples.py`
2. Review the docstrings in each module
3. Consult the reference documentation
4. Open an issue in the repository

---

**Happy workflow building!** 🚀
