# Agents Layer - LLM-Whisperer

This folder is the execution and governance layer for multi-agent workflows in LLM Whisperer. It provides production-ready agent implementations using leading frameworks including **LangChain**, **LangGraph**, and **AGNO**.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start Guide](#quick-start-guide)
4. [Module Documentation](#module-documentation)
5. [Features](#features)
6. [API Reference](#api-reference)
7. [Resources](#resources)
8. [Contributing](#contributing)

---

## Overview

### What is LangChain?

**LangChain** is a framework for developing applications powered by language models. It enables you to:

- **Build chains** of LLM calls with data sources, APIs, and other tools
- **Create agents** that can autonomously choose actions and tools
- **Integrate retrieval** systems for RAG (Retrieval-Augmented Generation)
- **Manage prompts** and templates with sophisticated control

**When to use LangChain:**
- Building chains of LLM calls with structured data flow
- Creating higher-level abstractions for agents (use LangChain agents before moving to LangGraph)
- Rapid prototyping and development
- When you need pre-built integrations with popular services

**Key strengths:**
- Rich ecosystem of integrations (200+ integrations)
- High-level abstractions for common patterns
- Great for rapid development
- Built-in memory and state management

### What is LangGraph?

**LangGraph** is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents. It focuses entirely on **agent orchestration**.

**When to use LangGraph:**
- Building sophisticated, production-grade agents
- When you need full control over workflow logic
- Complex state management and durable execution
- Human-in-the-loop workflows
- Long-running agents that need to persist through failures

**Key strengths:**
- Durable execution (persist and resume state)
- Human-in-the-loop capabilities with interrupts
- Comprehensive memory systems
- Full streaming support
- Production-ready deployment
- Deep visibility with LangSmith integration

### What is AGNO?

**AGNO** is a next-generation agent framework for Python teams, designed for building self-learning, reliable agents. It emphasizes:

- **Multi-agent systems** orchestration
- **Self-learning capabilities** with feedback loops
- **Enterprise-ready** with security and privacy
- **Team-friendly** with easy configuration and deployment

**When to use AGNO:**
- Building enterprise multi-agent systems
- When agents need to learn and improve over time
- Complex orchestration across many agents
- Teams building production AI systems

### Key Differences

| Feature | LangChain | LangGraph | AGNO |
|---------|-----------|-----------|------|
| **Level** | High-level abstractions | Low-level orchestration | Mid-level framework |
| **Focus** | Chains, agents, RAG | Orchestration, state | Multi-agent systems |
| **Learning curve** | Moderate | Steep | Moderate |
| **Deployment** | Development → LangSmith | LangSmith (native) | Built-in deployment |
| **State management** | Basic | Comprehensive | Advanced |
| **Best for** | Rapid prototyping | Production agents | Enterprise systems |

### Real-World Use Cases

**LangChain:**
- Q&A chatbots over documentation
- RAG systems for document analysis
- Workflow automation with API chains
- Rapid MVP development

**LangGraph:**
- Autonomous research agents
- Code generation and debugging systems
- Multi-step customer service workflows
- Agents with human approval gates

**AGNO:**
- Complex enterprise workflows
- Multi-team AI operations
- Self-improving agent systems
- Large-scale AI deployment platforms

---

## Architecture

### Overall System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Applications                         │
└────────────┬────────────────────────────────────┬───────────┘
             │                                    │
    ┌────────▼─────────┐              ┌──────────▼────────────┐
    │   LangChain      │              │    LangGraph         │
    │   High-level     │              │   Low-level          │
    │   abstractions   │              │   orchestration      │
    └────────┬─────────┘              └──────────┬───────────┘
             │                                   │
    ┌────────▼───────────────────────────────────▼─────────┐
    │              LangSmith Integration                   │
    │  (Observability, Debugging, Deployment)             │
    └──────────────────────────────────────────────────────┘
             │
    ┌────────▼──────────┐
    │  Language Models  │
    │  Tools & APIs     │
    │  Memory Systems   │
    └───────────────────┘
```

### Component Breakdown

#### `src/` - Core Implementation
- **models/**: Language model integrations
- **agents/**: Agent implementations and orchestration
- **tools/**: Tool definitions and executors
- **memory/**: Memory systems (short-term, long-term)
- **state/**: State management and schemas

#### `examples/` - Implementation Templates
- **langchain_examples/**: High-level chain and agent patterns
- **langgraph_examples/**: Advanced orchestration patterns
- **agno_examples/**: Multi-agent system examples
- **integrations/**: Tool and API integration examples

#### `configs/` - Configuration Files
- **schemas/**: JSON schema definitions for validation
- **profiles/**: Runtime configuration profiles
- **agents/**: Agent-specific configurations

#### `prompts/` - Prompt Engineering
- **roles/**: System prompts for different agent roles
- **tasks/**: Task-specific instructions
- **shared/**: Shared prompt templates

#### `workflows/` - Orchestration Definitions
- **langgraph/**: Graph-based workflow definitions
- **shared/**: Reusable workflow components

#### `evaluation/` - Quality Assurance
- **cases/**: Benchmark test cases
- **judges/**: Evaluation rubrics
- **reports/**: Evaluation results

### Data Flow Architecture

```
Input → [Framework Layer] → [LLM Model] → [Tool Selection]
                                              ↓
                        [Tool Execution] → [State Update]
                                              ↓
                        [Memory Store] → [Evaluation]
                                              ↓
                                           Output
```

### Module Dependencies

```
Applications
    ↓
LangChain / LangGraph / AGNO
    ↓
langchain-core / langgraph-core
    ↓
LLM Providers (OpenAI, Anthropic, etc.)
Tools & Integrations
Memory Backends
State Management
```

---

## Quick Start Guide

### Installation Instructions

```bash
# Install LangChain
pip install langchain langchain-core

# Install LangGraph for advanced orchestration
pip install langgraph

# Install AGNO for multi-agent systems
pip install agno

# Install integrations (choose what you need)
pip install langchain-openai langchain-anthropic

# For local development
pip install -e .
```

### Basic Setup

```python
# src/setup.py - Framework initialization
from langchain_core.language_model import BaseLLM
from langchain_openai import ChatOpenAI
import os

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

# Verify setup
print(f"LLM Model: {llm.model_name}")
print("Setup complete!")
```

### First Agent Creation (Minimal Example)

#### LangChain Agent
```python
# examples/langchain_examples/minimal_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

tools = [calculate]
llm = ChatOpenAI(model="gpt-4")

agent = create_tool_calling_agent(llm, tools, prompt=...)
executor = AgentExecutor(agent=agent, tools=tools)

# Run the agent
result = executor.invoke({
    "input": "What is 42 * 3?"
})
print(result["output"])
```

#### LangGraph Agent
```python
# examples/langgraph_examples/minimal_graph.py
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

def agent_node(state: MessagesState):
    """Process messages and generate response."""
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
graph = graph.compile()

# Run graph
result = graph.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})
print(result["messages"][-1].content)
```

### Running Examples

```bash
# Run individual example
python examples/langchain_examples/minimal_agent.py

# Run with LangSmith tracing
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_key_here
python examples/langgraph_examples/minimal_graph.py

# Run evaluation suite
python evaluation/scripts/run_benchmarks.py
```

---

## Module Documentation

### Core Modules

#### `src/models/` - Language Model Integration

**Module**: `src/models/llm_factory.py`

Provides factory for creating and configuring language models.

```python
from src.models.llm_factory import create_llm

# Create model with configuration
llm = create_llm(
    provider="openai",
    model="gpt-4",
    temperature=0.7,
    max_tokens=2000
)
```

**Supported Models:**
- OpenAI: GPT-4, GPT-3.5-Turbo
- Anthropic: Claude 3 (Opus, Sonnet, Haiku)
- Local: Ollama, Hugging Face
- API-based: Azure OpenAI, Cohere, Groq

#### `src/agents/` - Agent Implementations

**Module**: `src/agents/base_agent.py:42`

Abstract base class for all agents.

```python
from src.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, llm, tools):
        super().__init__(llm, tools)
    
    async def run(self, input_text: str) -> str:
        """Execute agent logic."""
        pass
```

**Available agent types:**
- `ReActAgent`: Reason+Act pattern
- `PlanExecuteAgent`: Planning then execution
- `SelfCritiqueAgent`: With self-evaluation
- `MultiStepAgent`: Complex orchestration

#### `src/tools/` - Tool Management

**Module**: `src/tools/tool_registry.py:89`

Central registry for tool definitions.

```python
from src.tools.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register_tool(
    name="web_search",
    description="Search the internet",
    func=search_function,
    input_schema={...}
)
```

#### `src/memory/` - Memory Systems

**Module**: `src/memory/memory_manager.py:156`

Manages both short-term and long-term memory.

```python
from src.memory.memory_manager import MemoryManager

memory = MemoryManager(
    short_term_type="message_buffer",
    long_term_type="vector_store",
    max_messages=100
)

# Add to conversation memory
memory.add_message("user", "What's the capital of France?")
memory.add_message("assistant", "The capital of France is Paris.")

# Retrieve relevant context
context = memory.retrieve_context(query="capitals", top_k=5)
```

#### `src/state/` - State Management

**Module**: `src/state/state_schema.py:203`

Defines state schemas for workflows.

```python
from src.state.state_schema import AgentState
from typing import TypedDict

class CustomState(TypedDict):
    messages: list
    current_step: int
    artifacts: dict
    metadata: dict
```

### Example Implementations

#### LangChain Chain Example
**File**: `examples/langchain_examples/document_qa.py`

Building a document question-answering system:
- Load documents with document loaders
- Create retriever from documents
- Build QA chain with prompt templates
- Execute with streaming support

#### LangGraph Workflow Example
**File**: `examples/langgraph_examples/multi_step_workflow.py`

Multi-step workflow with conditional routing:
- Define state schema
- Create nodes for each step
- Add conditional routing logic
- Implement human-in-the-loop interrupts

#### AGNO Multi-Agent Example
**File**: `examples/agno_examples/research_team.py`

Research team with multiple specialist agents:
- Define individual agent roles
- Set up inter-agent communication
- Implement coordination logic
- Monitor and log interactions

### Configuration Guide

#### Agent Configuration (`configs/agents/agent.yaml`)

```yaml
name: "research_agent"
version: "1.0.0"
description: "Autonomous research agent"

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

tools:
  - name: "web_search"
    enabled: true
  - name: "file_reader"
    enabled: true

memory:
  type: "message_buffer"
  max_messages: 100
  persistence: true

behavior:
  max_iterations: 10
  timeout_seconds: 300
  error_handling: "retry"
```

#### Model Configuration (`configs/models/model_config.yaml`)

```yaml
models:
  gpt4:
    provider: "openai"
    model_name: "gpt-4"
    temperature: 0.7
  claude3:
    provider: "anthropic"
    model_name: "claude-3-opus"
    temperature: 0.5
```

### Best Practices for Each Module

1. **Models**: Always validate API keys before initialization
2. **Agents**: Use tool descriptions for better action selection
3. **Tools**: Implement proper error handling and validation
4. **Memory**: Periodically clean up old messages to manage tokens
5. **State**: Use TypedDict for type safety in workflow state

---

## Features

### Memory Systems

#### Short-term Memory
- **Message Buffer**: Maintains conversation history
- **Sliding Window**: Keeps recent N messages
- **Token-aware**: Respects token limits

```python
from src.memory.short_term import MessageBuffer

memory = MessageBuffer(max_messages=50)
memory.add_message("user", "Hello")
memory.add_message("assistant", "Hi there!")
history = memory.get_history()
```

#### Long-term Memory
- **Vector Store**: Semantic search across past interactions
- **SQL Database**: Structured fact storage
- **Hybrid**: Combined vector + structured approach

```python
from src.memory.long_term import VectorStore

vector_store = VectorStore(
    embeddings_model="text-embedding-3-small",
    similarity_threshold=0.7
)
vector_store.add_memory("User prefers formal language", metadata={"user_id": "123"})
similar = vector_store.search("language preference", top_k=5)
```

### Tool Integration

- **Tool Definition**: Using `@tool` decorator or `Tool` class
- **Input Validation**: Schema-based validation
- **Error Handling**: Graceful degradation
- **Chaining**: Composable tool pipelines

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information about a query."""
    # Implementation here
    return results

@tool
def read_file(path: str) -> str:
    """Read contents of a file."""
    with open(path) as f:
        return f.read()

# Compose tools
tools = [search_web, read_file]
```

### State Management

- **Schema Definition**: Type-safe state schemas
- **State Persistence**: Durable state across sessions
- **Conditional Routing**: Route based on state values
- **State Mutations**: Safe state updates in workflows

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    plan: str
    current_action: str

graph = StateGraph(AgentState)
# Add nodes and edges...
```

### Workflow Composition

- **Graph-based**: Define workflows as directed acyclic graphs
- **Conditional Edges**: Route based on conditions
- **Subgraphs**: Compose larger workflows from smaller ones
- **Human-in-the-loop**: Pause for human approval

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)
graph.add_node("verify", verify_node)

graph.add_edge(START, "plan")
graph.add_conditional_edges(
    "plan",
    should_execute,
    {True: "execute", False: END}
)
graph.add_edge("execute", "verify")
graph.add_edge("verify", END)

compiled = graph.compile()
```

### Error Handling

- **Try-except blocks** with meaningful messages
- **Fallback strategies**: Alternative tools or actions
- **Exponential backoff**: For retry logic
- **Timeout management**: Prevent infinite loops

```python
from src.errors import ToolExecutionError, InvalidStateError

try:
    result = agent.run(input_text)
except ToolExecutionError as e:
    print(f"Tool failed: {e.tool_name} - {e.message}")
    # Implement fallback
except InvalidStateError as e:
    print(f"Invalid state transition: {e}")
```

---

## API Reference

### Key Classes

#### `BaseAgent`
**Module**: `src/agents/base_agent.py`

Abstract base for all agent implementations.

**Methods:**
- `__init__(llm, tools, memory=None)`: Initialize agent
- `async run(input_text: str) -> str`: Execute agent
- `async step() -> tuple`: Single step execution
- `reset()`: Clear state and memory

#### `StateGraph`
**Module**: `langgraph.graph`

Build stateful agent workflows.

**Methods:**
- `add_node(name: str, node: Callable)`: Add computation node
- `add_edge(from: str, to: str)`: Direct edge
- `add_conditional_edges(from: str, condition: Callable, mapping: dict)`: Conditional routing
- `compile() -> CompiledGraph`: Create executable workflow

#### `MemoryManager`
**Module**: `src/memory.memory_manager`

Manage multi-level memory systems.

**Methods:**
- `add_message(role: str, content: str)`: Add to memory
- `retrieve_context(query: str, top_k: int) -> list`: Semantic search
- `clear_memory(type: str = "all")`: Clear memory
- `get_summary() -> str`: Extract summary

### Important Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.7 | LLM randomness (0-1) |
| `max_tokens` | int | 2000 | Maximum output length |
| `top_p` | float | 0.9 | Nucleus sampling parameter |
| `max_iterations` | int | 10 | Agent loop limit |
| `timeout_seconds` | int | 300 | Execution timeout |

### Return Types

**Agent Execution:**
```python
{
    "output": "Final response",
    "messages": [...],  # Full conversation
    "steps": 5,         # Number of steps taken
    "success": True,    # Execution success
    "metadata": {...}   # Additional data
}
```

**Tool Execution:**
```python
ToolResult(
    content="Tool output",
    is_error=False,
    tool_call_id="tool_123"
)
```

---

## Resources

### Official Documentation

**LangChain Documentation:**
- Main docs: https://docs.langchain.com/oss/python/langchain/overview
- API Reference: https://python.langchain.com/api_reference
- New v1.0 reference: https://reference.langchain.com/python

**LangGraph Documentation:**
- Overview: https://docs.langchain.com/oss/python/langgraph/overview
- Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- Durable Execution: https://docs.langchain.com/oss/python/langgraph/durable-execution
- Human-in-the-loop: https://docs.langchain.com/oss/python/langgraph/interrupts

**AGNO Documentation:**
- Official site: https://www.agno.com/
- Framework guide: https://workos.com/blog/agno-the-agent-framework-for-python-teams
- GitHub: https://github.com/agno-ai/agno

### Relevant Guides and Articles

**Framework Comparisons & Tutorials:**
- LangChain vs LangGraph: https://blog.jetbrains.com/pycharm/2026/02/langchain-tutorial-2026/
- LangGraph 2026 Guide: https://www.linkedin.com/pulse/complete-guide-langgraph-2026-edition-learnbay-esb7c
- Building Agents with AGNO: https://www.linkedin.com/pulse/building-ai-agents-using-agno-next-generation-intelligent-pqlqc
- AGNO Framework Deep Dive: https://medium.com/@devipriyakaruppiah/agentic-framework-deep-dive-series-part-2-agno-c45da579b7c0
- AGNO on GeeksforGeeks: https://www.geeksforgeeks.org/artificial-intelligence/building-ai-agents-using-agno/

**Memory and State Management:**
- Blog post on memory systems: https://blog.langchain.com/how-we-built-agent-builders-memory-system/
- LangGraph stateful workflows: https://tutorialq.com/ai/frameworks/langgraph-stateful-workflows

### Engineering and Best Practices

**Production Deployment:**
- Building Effective Agents (Anthropic): https://www.anthropic.com/engineering/building-effective-agents
- Google SRE Book: https://sre.google/sre-book/table-of-contents/

**Safety and Governance:**
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Top 10 for LLM Applications: https://owasp.org/www-project-top-10-for-large-language-model-applications/

**Standards and Protocols:**
- Model Context Protocol: https://modelcontextprotocol.io/specification
- JSON Schema: https://json-schema.org/draft/2020-12
- RFC 8259 (JSON): https://www.rfc-editor.org/rfc/rfc8259
- RFC 9457 (HTTP Problem Details): https://www.rfc-editor.org/rfc/rfc9457

**Research Papers:**
- ReAct (Reasoning + Acting): https://arxiv.org/abs/2210.03629
- RAG (Retrieval-Augmented Generation): https://arxiv.org/abs/2005.11401
- QLoRA: https://arxiv.org/abs/2305.14314
- HELM Paper: https://arxiv.org/abs/2211.09110

### Community and Support

- **LangChain GitHub**: https://github.com/langchain-ai/langchain
- **LangGraph GitHub**: https://github.com/langchain-ai/langgraph
- **LangChain Discord**: https://discord.gg/langchain
- **LangSmith**: https://smith.langchain.com/ (Observability platform)

---

## Contributing

### How to Add New Modules

1. **Create module directory** in `src/`:
   ```bash
   mkdir -p src/new_module
   touch src/new_module/__init__.py
   ```

2. **Define module interface** in `src/new_module/base.py`:
   ```python
   from abc import ABC, abstractmethod
   
   class BaseModule(ABC):
       @abstractmethod
       def execute(self, **kwargs):
           pass
   ```

3. **Implement concrete classes**:
   ```python
   from .base import BaseModule
   
   class ConcreteModule(BaseModule):
       def execute(self, **kwargs):
           # Implementation
           pass
   ```

4. **Add tests** in `tests/test_new_module.py`

5. **Document in README** or create `src/new_module/README.md`

### Testing Guidelines

**Unit Tests:**
```python
# tests/test_agents.py
import pytest
from src.agents.base_agent import BaseAgent

def test_agent_initialization():
    agent = BaseAgent(llm=mock_llm, tools=[])
    assert agent is not None

@pytest.mark.asyncio
async def test_agent_execution():
    result = await agent.run("test input")
    assert result is not None
    assert result["success"] is True
```

**Integration Tests:**
```python
# tests/integration/test_workflows.py
def test_complete_workflow():
    graph = build_workflow()
    result = graph.invoke({"input": "test"})
    assert "output" in result
```

**Run tests:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_agents.py -v
```

### Code Style Requirements

**Python Style (PEP 8):**
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints for all functions

```python
def create_agent(llm: BaseLLM, tools: List[Tool]) -> BaseAgent:
    """Create a new agent instance.
    
    Args:
        llm: Language model instance
        tools: List of available tools
        
    Returns:
        Initialized agent ready for execution
    """
    return BaseAgent(llm=llm, tools=tools)
```

**Docstring Format:**
- Use Google-style docstrings
- Include Args, Returns, Raises sections
- Provide usage examples for public APIs

**Naming Conventions:**
- Classes: PascalCase (e.g., `ReActAgent`)
- Functions: snake_case (e.g., `create_agent`)
- Constants: UPPER_CASE (e.g., `MAX_ITERATIONS`)
- Private: Prefix with underscore (e.g., `_internal_method`)

**Configuration:**
- Use YAML for static configs
- Use environment variables for secrets
- Validate all inputs at boundaries

---

## Structure Overview

This design follows a practical separation used in production systems:

- **prompts/**: Human intent and behavior control
- **configs/**: System constraints and defaults
- **workflows/**: Reusable process orchestration
- **evaluation/**: Quality and safety gates
- **src/**: Core implementations
- **examples/**: Sample code and templates

Keeping these independent prevents quality logic from being mixed into execution logic.

## Folder Map

```text
agents/
├── prompts/
│   ├── roles/           # Agent role definitions
│   ├── tasks/           # Task-specific instructions
│   └── shared/          # Shared prompt templates
├── configs/
│   ├── schemas/         # JSON schema definitions
│   ├── profiles/        # Runtime configurations
│   └── agents/          # Agent-specific configs
├── src/
│   ├── models/          # LLM integrations
│   ├── agents/          # Agent implementations
│   ├── tools/           # Tool definitions
│   ├── memory/          # Memory systems
│   └── state/           # State management
├── examples/
│   ├── langchain_examples/      # LangChain patterns
│   ├── langgraph_examples/      # LangGraph workflows
│   ├── agno_examples/           # AGNO multi-agent examples
│   └── integrations/            # Tool integrations
├── workflows/
│   ├── langgraph/       # Graph definitions
│   └── shared/          # Reusable components
└── evaluation/
    ├── cases/           # Benchmark test cases
    ├── judges/          # Evaluation rubrics
    ├── reports/         # Results and metrics
    └── scripts/         # Evaluation scripts
```

## Operating Principles

1. **Start simple**: Begin with deterministic workflows; add autonomy only when needed.
2. **Explicit over implicit**: Prefer explicit routing and gates over hidden behavior.
3. **Safety first**: Keep safety checks as first-class workflow steps.
4. **Deterministic gates**: Use deterministic checks for release gates; use LLM-as-judge for support.
5. **Source-backed**: Require source-backed claims for research-heavy tasks.

---

## AGNO Framework Implementation

This repository now includes comprehensive AGNO framework implementations, examples, and configurations.

### AGNO Directory Structure

```
agents/
├── src/
│   ├── agno_basic_agent.py           # Basic agent initialization
│   ├── agno_multi_agent_workflow.py  # Team orchestration patterns
│   ├── agno_tool_integration.py      # Tool usage and function calling
│   ├── agno_memory_management.py     # Session and state management
│   └── agno_reasoning_agent.py       # Advanced reasoning patterns
├── examples/
│   ├── simple_qa_agent.py            # Q&A agent example
│   ├── research_agent.py             # Research/synthesis agent
│   └── code_analysis_agent.py        # Code analysis agent
└── configs/
    ├── agno_runtime.yaml             # Runtime configuration
    ├── agno_models.yaml              # Model setup
    └── agno_tools.yaml               # Tool definitions
```

### Quick Start with AGNO

1. **Install dependencies:**
   ```bash
   pip install agno[all] anthropic
   ```

2. **Set API keys:**
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   ```

3. **Run examples:**
   ```bash
   python examples/simple_qa_agent.py
   python examples/research_agent.py
   python examples/code_analysis_agent.py
   ```

### Key AGNO Files

- **Reference Guide:** See `AGNO_FRAMEWORK_GUIDE.md` (created below)
- **Core API Examples:** See `src/agno_*.py` files
- **Practical Examples:** See `examples/` directory
- **Configuration Reference:** See `configs/agno_*.yaml` files

### AGNO Resources

- **Official Docs:** https://docs.agno.com
- **GitHub:** https://github.com/agno-agi/agno
- **Discord:** https://www.agno.com/discord

---

**Last Updated**: April 6, 2026  
**Version**: 2.1.0  
**Status**: Production-Ready
**New**: AGNO Framework Integration (v2.1.0)
