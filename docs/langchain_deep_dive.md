# LangChain and LangGraph Deep Dive: Complete Framework Guide

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Framework Versions:** LangChain 0.1+, LangGraph 0.1+

---

## Table of Contents

1. [Introduction](#introduction)
2. [Official Documentation Links](#official-documentation-links)
3. [LangChain Overview](#langchain-overview)
4. [LangGraph Overview](#langgraph-overview)
5. [Core Concepts and Architecture](#core-concepts-and-architecture)
6. [Installation and Setup](#installation-and-setup)
7. [Basic Code Examples](#basic-code-examples)
8. [Advanced Patterns](#advanced-patterns)
9. [State Machine Architecture in LangGraph](#state-machine-architecture-in-langgraph)
10. [Performance Considerations](#performance-considerations)
11. [Common Issues and Solutions](#common-issues-and-solutions)
12. [Comparison: LangChain vs LangGraph](#comparison-langchain-vs-langgraph)
13. [Production Deployment](#production-deployment)
14. [Resources and References](#resources-and-references)

---

## Introduction

LangChain and LangGraph are two complementary frameworks for building AI agents and autonomous applications. **LangChain** provides a higher-level abstraction with prebuilt agent architectures, while **LangGraph** offers low-level orchestration with fine-grained control over agent workflows through state machines.

This guide covers both frameworks in depth, with practical examples, architectural patterns, and production deployment strategies.

---

## Official Documentation Links

### Primary Resources

- **LangChain Official Documentation:** https://python.langchain.com/docs/
- **LangChain Concepts Guide:** https://python.langchain.com/docs/concepts
- **LangChain How-To Guides:** https://python.langchain.com/docs/how_to
- **LangGraph Official Documentation:** https://docs.langchain.com/oss/python/langgraph/overview
- **LangGraph Graph API Reference:** https://docs.langchain.com/oss/python/langgraph/graph-api
- **LangGraph Workflows & Agents:** https://docs.langchain.com/oss/python/langgraph/workflows-agents
- **LangSmith (Observability):** https://docs.langchain.com/langsmith

### GitHub Repositories

- **LangChain Main Repository:** https://github.com/langchain-ai/langchain
- **LangChain Community Integrations:** https://github.com/langchain-ai/langchain-community
- **LangChain Academy (Tutorials):** https://github.com/langchain-ai/langchain-academy
- **LangChain Documentation Repository:** https://github.com/langchain-ai/docs
- **LangChain JavaScript Version:** https://github.com/langchain-ai/langchainjs

---

## LangChain Overview

### What is LangChain?

LangChain is an open-source framework that simplifies building applications with Large Language Models (LLMs). It provides:

- **Standard model interfaces** for seamlessly switching between different LLM providers (OpenAI, Anthropic, Google, etc.)
- **Prebuilt agent architectures** that allow you to build agents in under 10 lines of code
- **Easy integration** with tools, memory systems, and retrieval-augmented generation (RAG)
- **Built on LangGraph** for durable execution, streaming, human-in-the-loop support, and persistence

### Key Components

1. **Models:** Standardized interfaces for LLMs (chat models, text completion)
2. **Tools:** Functions that agents can call to perform actions
3. **Agents:** Autonomous systems that decide which tools to use and when
4. **Memory:** Systems for maintaining conversation history and context
5. **Chains:** Sequences of calls to LLMs and other tools
6. **RAG (Retrieval-Augmented Generation):** Combining LLMs with external knowledge

### When to Use LangChain

✅ **Use LangChain if you:**
- Want to quickly build agents and autonomous applications
- Need prebuilt agent architectures with good defaults
- Want standard interfaces across different LLM providers
- Need integration with many third-party services
- Are building typical use cases (Q&A, chatbots, content generation)

---

## LangGraph Overview

### What is LangGraph?

LangGraph is a low-level orchestration framework and runtime specifically designed for building, managing, and deploying long-running, stateful agents. It focuses on:

- **Agent orchestration** through state machines and graph-based execution
- **Durable execution** with checkpointing and resumability
- **Streaming support** for real-time output
- **Human-in-the-loop** capabilities with interrupts
- **Complex workflow management** with branching, loops, and parallelization
- **Production-ready runtime** for deploying sophisticated agent systems

### Key Concepts

1. **Graphs:** Define agent workflows as directed graphs with nodes and edges
2. **State:** Shared data structure representing the current snapshot of your application
3. **Nodes:** Functions that perform computations or side-effects
4. **Edges:** Functions that determine which node to execute next
5. **Message Passing:** Pregel-based algorithm for distributed graph computation
6. **Checkpointing:** Persistence layer for resuming from failures

### When to Use LangGraph

✅ **Use LangGraph if you:**
- Need fine-grained control over agent behavior
- Have complex workflows with branching, loops, and state management
- Require durable execution that persists through failures
- Need human-in-the-loop capabilities with interrupts
- Are building enterprise-grade agent systems
- Need to combine deterministic and agentic workflows
- Want to avoid vendor lock-in with custom orchestration

---

## Core Concepts and Architecture

### LangChain Agent Architecture

```
User Input
    ↓
[Agent Executor]
    ↓
[LLM with Tool Binding]
    ↓
[Tool Selection Decision]
    ├→ Execute Tool(s)
    │    ↓
    │ [Tool Results]
    │    ↓
    └→ Loop back to LLM (if more tools needed)
    ├→ Generate Response (if done)
    ↓
Output to User
```

**Components:**
- **Agent:** Orchestrates the interaction between LLM and tools
- **Tools:** Callable functions that the agent can use
- **Memory:** Maintains conversation history
- **Callbacks:** Hooks for logging, tracing, and monitoring

### LangGraph State Machine Architecture

```
[START] Node
    ↓
[Node A] → processes state → outputs updates
    ↓
[Conditional Router] → decides next node based on state
    ├→ [Node B] → processes state
    ├→ [Node C] → processes state
    └→ [Node D] → processes state
    ↓
[END] Node
```

**Key Differences:**
1. **Explicit state management** - all data is tracked in a TypedDict
2. **Flexible routing** - conditional edges make complex decisions
3. **Message passing** - nodes communicate through state updates
4. **Parallelization** - multiple nodes can execute simultaneously
5. **Persistence** - state can be checkpointed and resumed

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip or uv package manager
- API keys for LLM providers (OpenAI, Anthropic, etc.)

### Basic Installation

```bash
# Core LangChain
pip install langchain langchain-core

# With specific LLM providers
pip install langchain-openai langchain-anthropic langchain-google-genai

# LangGraph
pip install langgraph

# For visualization and debugging
pip install langsmith

# Optional: for Jupyter notebooks
pip install jupyter ipython
```

### Installation with UV (Recommended)

```bash
# Using UV for faster dependency resolution
uv add langchain langchain-core langgraph langsmith

# Or with specific providers
uv add langchain langchain-core langchain-anthropic langgraph langsmith
```

### Environment Configuration

```bash
# .env file or export commands
export LANGCHAIN_API_KEY="your_api_key"
export LANGCHAIN_TRACING_V2=true
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Python Setup Script

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify installations
try:
    import langchain
    import langgraph
    import langsmith
    print("✓ All dependencies installed successfully")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")

# Set up LangSmith tracing (optional)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
```

---

## Basic Code Examples

### 1. Simple LangChain Agent

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool

# Define tools
@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's 72°F and sunny in {city}!"

@tool
def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    import datetime
    return f"Current time: {datetime.datetime.now()}"

# Create model
model = ChatAnthropic(model="claude-sonnet-4-6")

# Create agent
agent = create_agent(
    model=model,
    tools=[get_weather, get_time],
    system_prompt="You are a helpful assistant with access to weather and time tools."
)

# Run agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

print(result)
```

### 2. Simple LangGraph Workflow

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_anthropic import ChatAnthropic
from langchain.messages import HumanMessage

# Initialize model
model = ChatAnthropic(model="claude-sonnet-4-6")

# Define node functions
def greeting_node(state: MessagesState):
    """Add a greeting to the messages"""
    return {"messages": [{"role": "system", "content": "Hello! How can I help you?"}]}

def process_node(state: MessagesState):
    """Process the user message with LLM"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("greeting", greeting_node)
builder.add_node("process", process_node)

# Add edges
builder.add_edge(START, "greeting")
builder.add_edge("greeting", "process")
builder.add_edge("process", END)

# Compile and run
graph = builder.compile()
result = graph.invoke({"messages": [HumanMessage(content="Hello!")]})
print(result)
```

### 3. Agent with Tools in LangGraph

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_anthropic import ChatAnthropic
from langchain.messages import HumanMessage, ToolMessage, SystemMessage
from langchain.tools import tool

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Tools mapping
tools = [calculator]
tools_by_name = {tool.name: tool for tool in tools}

# Initialize model with tools
model = ChatAnthropic(model="claude-sonnet-4-6")
model_with_tools = model.bind_tools(tools)

# Define nodes
def llm_node(state: MessagesState):
    """LLM decides whether to call tools"""
    response = model_with_tools.invoke(
        [SystemMessage(content="You are a helpful math assistant.")] + state["messages"]
    )
    return {"messages": [response]}

def tool_node(state: MessagesState):
    """Execute the tool call"""
    results = []
    last_message = state["messages"][-1]
    
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": results}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide whether to continue with tools or end"""
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("llm", llm_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm")

# Compile and run
graph = builder.compile()
messages = graph.invoke({"messages": [HumanMessage(content="What is 42 * 7?")]})
for msg in messages["messages"]:
    print(f"{msg.type}: {msg.content}")
```

---

## Advanced Patterns

### 1. Prompt Chaining in LangGraph

Prompt chaining breaks complex tasks into sequential steps where each LLM call processes the output of the previous one.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    initial_response: str
    refined_response: str
    final_response: str

# Initialize model
model = ChatAnthropic(model="claude-sonnet-4-6")

def generate_response(state: State):
    """First LLM call - generate initial response"""
    response = model.invoke(f"Explain {state['topic']} in simple terms")
    return {"initial_response": response.content}

def refine_response(state: State):
    """Second LLM call - refine for clarity"""
    response = model.invoke(
        f"Make this clearer and more concise: {state['initial_response']}"
    )
    return {"refined_response": response.content}

def finalize_response(state: State):
    """Third LLM call - add examples"""
    response = model.invoke(
        f"Add practical examples to this explanation: {state['refined_response']}"
    )
    return {"final_response": response.content}

# Build workflow
builder = StateGraph(State)
builder.add_node("generate", generate_response)
builder.add_node("refine", refine_response)
builder.add_node("finalize", finalize_response)

builder.add_edge(START, "generate")
builder.add_edge("generate", "refine")
builder.add_edge("refine", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()
result = graph.invoke({"topic": "Machine Learning"})
print(result["final_response"])
```

### 2. Parallelization Pattern

Execute multiple independent tasks simultaneously to improve performance.

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: Annotated[list[str], operator.add]
    story: Annotated[list[str], operator.add]
    poem: Annotated[list[str], operator.add]

model = ChatAnthropic(model="claude-sonnet-4-6")

def generate_joke(state: State):
    """Generate a joke about the topic"""
    response = model.invoke(f"Write a funny joke about {state['topic']}")
    return {"joke": [response.content]}

def generate_story(state: State):
    """Generate a story about the topic"""
    response = model.invoke(f"Write a short story about {state['topic']}")
    return {"story": [response.content]}

def generate_poem(state: State):
    """Generate a poem about the topic"""
    response = model.invoke(f"Write a poem about {state['topic']}")
    return {"poem": [response.content]}

# Build workflow - all three nodes run in parallel
builder = StateGraph(State)
builder.add_node("joke", generate_joke)
builder.add_node("story", generate_story)
builder.add_node("poem", generate_poem)

# All three start from START, execute in parallel
builder.add_edge(START, "joke")
builder.add_edge(START, "story")
builder.add_edge(START, "poem")

builder.add_edge("joke", END)
builder.add_edge("story", END)
builder.add_edge("poem", END)

graph = builder.compile()
result = graph.invoke({"topic": "Space"})
print("Joke:", result["joke"])
print("Story:", result["story"])
print("Poem:", result["poem"])
```

### 3. Router Pattern

Route inputs to different specialized processors based on content analysis.

```python
from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class Route(BaseModel):
    """Router decision schema"""
    topic: Literal["math", "science", "history", "other"]
    explanation: str

class State(TypedDict):
    query: str
    route: str
    response: str

model = ChatAnthropic(model="claude-sonnet-4-6")
router = model.with_structured_output(Route)

def route_query(state: State):
    """Route the query to appropriate handler"""
    route_decision = router.invoke(
        f"Classify this query and explain why: {state['query']}"
    )
    return {"route": route_decision.topic}

def handle_math(state: State):
    """Handle math queries"""
    response = model.invoke(f"Solve this math problem: {state['query']}")
    return {"response": f"[MATH] {response.content}"}

def handle_science(state: State):
    """Handle science queries"""
    response = model.invoke(f"Explain this science concept: {state['query']}")
    return {"response": f"[SCIENCE] {response.content}"}

def handle_history(state: State):
    """Handle history queries"""
    response = model.invoke(f"Explain this historical event: {state['query']}")
    return {"response": f"[HISTORY] {response.content}"}

def handle_other(state: State):
    """Handle other queries"""
    response = model.invoke(state["query"])
    return {"response": f"[OTHER] {response.content}"}

def route_decision(state: State) -> Literal["math_handler", "science_handler", "history_handler", "other_handler"]:
    """Conditional routing"""
    if state["route"] == "math":
        return "math_handler"
    elif state["route"] == "science":
        return "science_handler"
    elif state["route"] == "history":
        return "history_handler"
    return "other_handler"

# Build workflow
builder = StateGraph(State)
builder.add_node("router", route_query)
builder.add_node("math_handler", handle_math)
builder.add_node("science_handler", handle_science)
builder.add_node("history_handler", handle_history)
builder.add_node("other_handler", handle_other)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_decision)
builder.add_edge("math_handler", END)
builder.add_edge("science_handler", END)
builder.add_edge("history_handler", END)
builder.add_edge("other_handler", END)

graph = builder.compile()
result = graph.invoke({"query": "What is quantum entanglement?"})
print(result["response"])
```

### 4. Orchestrator-Worker Pattern (Map-Reduce)

Orchestrator breaks down tasks and distributes them to workers in parallel.

```python
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.types import Send
import operator

class Section(BaseModel):
    """Report section schema"""
    title: str
    description: str

class State(TypedDict):
    """Main orchestrator state"""
    topic: str
    sections: List[Section]
    completed_sections: Annotated[list[str], operator.add]
    final_report: str

class WorkerState(TypedDict):
    """Worker state"""
    section: Section
    completed_sections: Annotated[list[str], operator.add]

model = ChatAnthropic(model="claude-sonnet-4-6")
planner = model.with_structured_output(
    {"sections": List[Section]}
)

def orchestrator(state: State):
    """Plan the report structure"""
    response = model.invoke(
        f"Create a 3-section outline for a report on: {state['topic']}"
    )
    # Parse response and create sections
    sections = [
        Section(title="Introduction", description="Overview"),
        Section(title="Main Content", description="Detailed analysis"),
        Section(title="Conclusion", description="Summary and takeaways")
    ]
    return {"sections": sections}

def worker(state: WorkerState):
    """Worker writes a section"""
    section_text = model.invoke(
        f"Write the '{state['section'].title}' section: {state['section'].description}"
    )
    return {"completed_sections": [section_text.content]}

def assign_workers(state: State):
    """Assign a worker for each section"""
    return [Send("worker", {"section": s}) for s in state["sections"]]

def synthesizer(state: State):
    """Combine all sections into final report"""
    final = "\n\n---\n\n".join(state["completed_sections"])
    return {"final_report": final}

# Build workflow
builder = StateGraph(State)
builder.add_node("orchestrator", orchestrator)
builder.add_node("worker", worker)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
builder.add_edge("worker", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()
result = graph.invoke({"topic": "Artificial Intelligence"})
print(result["final_report"])
```

### 5. Human-in-the-Loop with Interrupts

Pause execution for human review and decision-making.

```python
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

class State(TypedDict):
    user_input: str
    ai_response: str
    human_approved: bool
    final_response: str

model = ChatAnthropic(model="claude-sonnet-4-6")

def generate_response(state: State):
    """Generate response"""
    response = model.invoke(state["user_input"])
    return {"ai_response": response.content}

def human_review(state: State):
    """Pause for human review"""
    # This will pause graph execution until human provides feedback
    human_decision = interrupt(
        f"AI Response:\n{state['ai_response']}\n\nApprove? (yes/no)"
    )
    
    if human_decision.lower() == "yes":
        return Command(
            update={"human_approved": True, "final_response": state["ai_response"]},
            goto=END
        )
    else:
        return Command(
            update={"human_approved": False},
            goto="regenerate"
        )

def regenerate_response(state: State):
    """Regenerate with feedback"""
    feedback = interrupt("What should be improved?")
    improved = model.invoke(
        f"Original: {state['ai_response']}\n\nFeedback: {feedback}\n\nPlease improve:"
    )
    return {"ai_response": improved.content}

# Build workflow
builder = StateGraph(State)
builder.add_node("generate", generate_response)
builder.add_node("review", human_review)
builder.add_node("regenerate", regenerate_response)

builder.add_edge(START, "generate")
builder.add_edge("generate", "review")
builder.add_edge("regenerate", "review")

graph = builder.compile()

# First invocation
config = {"configurable": {"thread_id": "user_1"}}
result = graph.invoke({"user_input": "Write a poem"}, config)

# Resume after human input
result = graph.invoke(
    Command(resume="yes"),
    config
)
print(result["final_response"])
```

---

## State Machine Architecture in LangGraph

### Understanding LangGraph's Pregel Algorithm

LangGraph uses Google's **Pregel** algorithm for distributed graph computation with a message-passing model:

```
Super-Step 1:
├─ All START nodes are ACTIVE
├─ All other nodes are INACTIVE
└─ START nodes send messages to their destinations

Super-Step 2:
├─ Nodes that received messages become ACTIVE
├─ Previously active nodes become INACTIVE (no more messages)
├─ Active nodes execute their functions
└─ Output messages sent to destination nodes

Super-Step N:
├─ Repeat until no node has messages to send
└─ Graph execution terminates when all nodes INACTIVE
```

### State Definition with Reducers

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add, itemgetter
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """State with different reducer strategies"""
    
    # Default reducer - overwrites with latest value
    topic: str
    
    # Custom reducer - appends to list
    messages: Annotated[list, add]
    
    # Message reducer - handles message IDs and updates
    chat_messages: Annotated[list, add_messages]
    
    # Custom function reducer
    scores: Annotated[list[float], lambda x, y: x + y if isinstance(y, list) else x + [y]]
```

### State Channel Management

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class InputState(TypedDict):
    """External input schema"""
    user_query: str

class PrivateState(TypedDict):
    """Internal state not exposed"""
    embeddings: list[float]
    search_results: list[str]

class OutputState(TypedDict):
    """External output schema"""
    answer: str

class OverallState(TypedDict):
    """Complete internal state"""
    user_query: str
    embeddings: list[float]
    search_results: list[str]
    answer: str

# Define nodes with different state types
def node1(state: InputState) -> OverallState:
    # Input: user_query
    # Can write to any OverallState key
    return {"embeddings": [0.1, 0.2, 0.3]}

def node2(state: OverallState) -> PrivateState:
    # Can read from OverallState
    # Can write to PrivateState
    return {"search_results": ["doc1", "doc2"]}

def node3(state: PrivateState) -> OutputState:
    # Can read from PrivateState
    # Can write to OutputState
    return {"answer": "Final answer"}

# Build with explicit schemas
builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState
)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)
```

### Command-Based Routing

```python
from langgraph.types import Command
from typing import Literal
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int
    should_continue: bool

def process_node(state: State) -> Command[Literal["increment_node", "final_node"]]:
    """Use Command to combine state updates and routing"""
    
    new_counter = state["counter"] + 1
    
    # Return Command with both update and routing
    if new_counter < 5:
        return Command(
            update={"counter": new_counter},
            goto="increment_node"
        )
    else:
        return Command(
            update={"counter": new_counter},
            goto="final_node"
        )

# This combines conditional logic and state updates in one step
```

### Subgraphs for Modularity

```python
from langgraph.graph import StateGraph

# Define subgraph
def create_subgraph():
    """Create a reusable subgraph component"""
    
    class SubState(TypedDict):
        input_data: str
        processed_data: str
    
    def process(state: SubState):
        return {"processed_data": state["input_data"].upper()}
    
    builder = StateGraph(SubState)
    builder.add_node("process", process)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    
    return builder.compile()

# Use subgraph in main graph
main_builder = StateGraph(State)
# Add subgraph as a node
subgraph = create_subgraph()
main_builder.add_node("subgraph_runner", subgraph)
```

---

## Performance Considerations

### 1. Optimization Strategies

```python
# ✓ Good: Use streaming for large outputs
for chunk in graph.stream({"messages": [...]}, stream_mode="updates"):
    print(chunk)

# ✗ Avoid: Loading entire output before processing
result = graph.invoke({"messages": [...]})  # Blocks until complete

# ✓ Good: Batch multiple requests
from concurrent.futures import ThreadPoolExecutor

results = []
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(graph.invoke, input_i) for input_i in inputs]
    results = [f.result() for f in futures]

# ✓ Good: Use async for non-blocking I/O
async def process_async(input_data):
    result = await graph.ainvoke(input_data)
    return result
```

### 2. Memory Management

```python
# Use checkpointing for long-running agents
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Resume from checkpoint
config = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke({"messages": [...]}, config)

# Resume later
result = graph.invoke({"messages": [...]}, config)  # Continues from last checkpoint
```

### 3. Token Usage Optimization

```python
# Monitor token usage
import os
os.environ["LANGSMITH_TRACING"] = "true"

# Use LangSmith for detailed metrics
from langsmith import Client

client = Client()
runs = client.list_runs(project_name="my_agent", limit=10)
for run in runs:
    print(f"Tokens: {run.cumulative_cost}")
```

### 4. Recursion Limit Configuration

```python
# Set appropriate recursion limits
graph.invoke(
    {"messages": [...]},
    config={"recursion_limit": 50}  # Prevent infinite loops
)

# Monitor recursion usage
def monitor_recursion(state: State, config: RunnableConfig) -> dict:
    current_step = config["metadata"]["langgraph_step"]
    print(f"Currently on step: {current_step}")
    return state
```

---

## Common Issues and Solutions

### Issue 1: Agent Goes into Infinite Loop

**Problem:** Agent keeps calling tools indefinitely.

**Solution:**
```python
# Set recursion limit
config = {"recursion_limit": 10}
result = graph.invoke({"messages": [...]}, config)

# Or use RemainingSteps managed value
from langgraph.managed import RemainingSteps

def check_steps(state: State) -> Literal["continue", END]:
    if state["remaining_steps"] <= 1:
        return END
    return "continue"
```

### Issue 2: State Updates Not Persisting

**Problem:** State changes in one node don't appear in the next.

**Solution:**
```python
# Ensure you return state updates as dictionary
def node_func(state: State) -> State:
    # ✓ Correct - return as dict
    return {"key": "new_value"}
    
    # ✗ Wrong - don't return modified object
    # state["key"] = "new_value"  
    # return state
```

### Issue 3: Tool Calls Not Being Recognized

**Problem:** LLM makes tool calls but they're not being executed.

**Solution:**
```python
# Ensure model is bound with tools
model_with_tools = model.bind_tools(tools)

# Check tool_calls attribute exists
if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
    # Process tool calls
    pass

# Verify tool signatures match what LLM expects
@tool
def my_tool(param: str) -> str:  # Type hints are required
    """Docstring is required for tool description"""
    return f"Result: {param}"
```

### Issue 4: Memory/Checkpointing Not Working

**Problem:** Graph doesn't resume from previous state.

**Solution:**
```python
# Use persistent checkpointer
from langgraph.checkpoint.postgres import PostgresSaver

# Connection string for PostgreSQL
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/langgraph"
)

graph = builder.compile(checkpointer=checkpointer)

# Always provide same thread_id to resume
config = {"configurable": {"thread_id": "conversation_123"}}
result = graph.invoke(input1, config)
result = graph.invoke(input2, config)  # Continues from previous state
```

### Issue 5: Streaming Mode Issues

**Problem:** Streaming doesn't produce expected output.

**Solution:**
```python
# Use correct stream mode
# "values" - full state snapshots
for chunk in graph.stream(input_data, stream_mode="values"):
    print(chunk)

# "updates" - only node updates
for chunk in graph.stream(input_data, stream_mode="updates"):
    print(chunk)

# "messages" - only message-related updates (if using MessagesState)
for chunk in graph.stream(input_data, stream_mode="messages"):
    print(chunk)
```

---

## Comparison: LangChain vs LangGraph

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| **Abstraction Level** | High-level | Low-level |
| **Learning Curve** | Easier | Steeper |
| **Setup Time** | Minutes | Hours |
| **Prebuilt Agents** | Yes | No |
| **Customization** | Limited | Extensive |
| **State Management** | Implicit | Explicit |
| **Control Over Routing** | Limited | Full |
| **Persistence** | Yes | Yes |
| **Streaming** | Partial | Full |
| **Human-in-Loop** | Limited | Advanced |
| **Multi-Agent** | Basic | Advanced |
| **Production Ready** | Yes | Yes |
| **Documentation** | Extensive | Growing |

### Decision Tree

```
Do you want to:
├─ Quickly build a simple agent?
│  └─ Use LangChain ✓
├─ Need fine-grained control over agent behavior?
│  └─ Use LangGraph ✓
├─ Have complex multi-agent systems?
│  └─ Use LangGraph ✓
├─ Want minimal setup time?
│  └─ Use LangChain ✓
├─ Need human-in-the-loop with interrupts?
│  └─ Use LangGraph ✓
└─ Building enterprise-grade systems?
   └─ Use LangGraph ✓
```

---

## Production Deployment

### 1. Docker Containerization

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**
```
langchain==0.1.0
langgraph==0.1.0
langchain-anthropic==0.1.0
langsmith==0.1.0
uvicorn==0.24.0
fastapi==0.104.0
pydantic==2.5.0
```

### 2. API Server with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END, MessagesState

app = FastAPI()

# Initialize graph
graph = create_agent_graph()

class QueryRequest(BaseModel):
    thread_id: str
    message: str

@app.post("/query")
async def query(request: QueryRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config
        )
        return {"response": result.get("response", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 3. Monitoring and Observability

```python
import logging
from langsmith import Client
from langsmith.run_trees import RunTree

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangSmith integration
client = Client()

# Trace execution
def trace_agent(graph, input_data, run_name: str):
    with RunTree(
        name=run_name,
        run_type="chain",
        client=client
    ) as run:
        result = graph.invoke(input_data)
        run.end()
    return result
```

### 4. Environment Configuration

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    
    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "sqlite:///./langgraph.db"
    )
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    
    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-6")
```

---

## Resources and References

### Official Documentation

- **LangChain Documentation:** https://python.langchain.com/docs/
- **LangGraph Documentation:** https://docs.langchain.com/oss/python/langgraph/overview
- **LangSmith Documentation:** https://docs.langchain.com/langsmith
- **LangChain API Reference:** https://reference.langchain.com/python

### Tutorials and Guides

- **LangChain Academy:** https://academy.langchain.com
- **LangGraph Quickstart:** https://docs.langchain.com/oss/python/langgraph/quick_start
- **Building Agents:** https://python.langchain.com/docs/how_to/tool_use_parallel
- **Workflow Patterns:** https://docs.langchain.com/oss/python/langgraph/workflows-agents

### Advanced Articles

- **LangChain in 2026: Core Concepts (2026):** https://medium.com/@vapbooksfeedback/tech-54-langchain-in-2026-the-5-concepts-that-handle-90-of-real-use-cases-19a96f654ba2
- **Clean State Architecture in LangGraph (Mar 2026):** https://medium.com/@ladvishal1985/everything-ive-learned-about-clean-state-architecture-in-langgraph-6d1352b0c00c
- **LangGraph Deep Dive: State Machines (Mar 2026):** https://blog.premai.io/langgraph-deep-dive-state-machines-tools-and-human-in-the-loop/
- **ReAct Agent Pattern in LangGraph (Mar 2026):** https://www.abstractalgorithms.dev/langgraph-react-agent-pattern
- **From LangChain to LangGraph (Mar 2026):** https://www.abstractalgorithms.dev/from-langchain-to-langgraph-when-agents-need-state-machines

### Community Resources

- **GitHub - LangChain Main:** https://github.com/langchain-ai/langchain
- **GitHub - LangChain Community:** https://github.com/langchain-ai/langchain-community
- **GitHub - LangChain Academy:** https://github.com/langchain-ai/langchain-academy
- **Discord Community:** https://discord.gg/langchain

### Related Tools

- **OpenAI API Documentation:** https://platform.openai.com/docs
- **Anthropic Claude API:** https://docs.anthropic.com/
- **Google Generative AI:** https://cloud.google.com/generative-ai
- **LangSmith (Observability):** https://smith.langchain.com/

### Performance & Scaling

- **Pregel: A System for Large-Scale Graph Processing:** https://research.google/pubs/pub37252/
- **Apache Beam (Inspiration):** https://beam.apache.org/
- **NetworkX Documentation:** https://networkx.org/documentation/latest/

---

## Best Practices Summary

### Code Organization

1. **Separate concerns:** Keep models, tools, and graph logic separate
2. **Modular graphs:** Use subgraphs for reusable components
3. **Type hints:** Always use TypedDict for state definition
4. **Error handling:** Wrap tool calls in try-except blocks

### State Management

1. **Explicit state:** Define all state keys upfront
2. **Reducers:** Use appropriate reducers for list accumulation
3. **Message handling:** Use `add_messages` for chat messages
4. **State scope:** Keep internal and external schemas separate

### Performance

1. **Streaming:** Use streaming for long-running operations
2. **Parallelization:** Run independent tasks simultaneously
3. **Caching:** Cache expensive computations
4. **Limits:** Set recursion limits to prevent infinite loops

### Production

1. **Checkpointing:** Enable persistence for fault tolerance
2. **Monitoring:** Use LangSmith for tracing and debugging
3. **Logging:** Log all agent decisions and tool calls
4. **Testing:** Test state transitions and edge cases

---

**Author:** Shuvam Banerji Seal  
**Date:** April 2026  
**Version:** 1.0  
**License:** MIT

For issues, questions, or contributions, visit the official LangChain GitHub repositories.
