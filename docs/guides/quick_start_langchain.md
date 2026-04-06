# Quick Start Guide: LangChain Framework

**Author:** Shuvam Banerji Seal

## Overview

LangChain is an open-source framework for building LLM-powered agents and applications. It provides a standardized interface for working with different language models and includes pre-built agent architectures.

**Official Documentation:** https://python.langchain.com/docs/

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv (recommended)
- An API key for your model provider (Anthropic, OpenAI, Google, etc.)

### Step 1: Set Up Virtual Environment

**Using uv (recommended):**
```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Using venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Install LangChain

Basic installation:
```bash
pip install -U langchain
```

With Anthropic (Claude):
```bash
pip install -U langchain 'langchain[anthropic]'
```

With OpenAI (GPT):
```bash
pip install -U langchain 'langchain[openai]'
```

Full installation with all integrations:
```bash
pip install -U langchain langsmith langgraph
```

### Step 3: Configure API Keys

**For Anthropic:**
```bash
export ANTHROPIC_API_KEY=sk-your-key-here
```

**For OpenAI:**
```bash
export OPENAI_API_KEY=sk-your-key-here
```

**Optional: Enable LangSmith Tracing**
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your-api-key
```

## Your First Agent (5 Minutes)

### Basic Agent Creation

Create a file `first_agent.py`:

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

# Define a simple tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's sunny in {city}!"

# Create the model
model = ChatAnthropic(model="claude-sonnet-4")

# Create the agent
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant.",
)

# Run the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

print(result)
```

Run it:
```bash
python first_agent.py
```

### Agent with Multiple Tools

Create `multi_tool_agent.py`:

```python
from langchain.agents import create_agent, tool
from langchain_anthropic import ChatAnthropic

# Define tools using the @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's sunny in {city}!"

@tool
def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a timezone."""
    from datetime import datetime
    return str(datetime.now())

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for '{query}': Example result 1, Example result 2"

# Create the agent
agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4"),
    tools=[get_weather, get_time, search_web],
    system_prompt="You are a helpful assistant with access to weather, time, and web search.",
)

# Run the agent
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What's the weather in NYC and what time is it?"}
    ]
})

print(result["messages"][-1].content)
```

## Basic LangGraph State Machine Setup

LangGraph is a more advanced framework for building stateful agents with control flow.

### Simple LangGraph Agent

Create `langgraph_agent.py`:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_anthropic import ChatAnthropic

# Define the model
model = ChatAnthropic(model="claude-sonnet-4")

# Define node functions
def agent_node(state: MessagesState):
    """Process messages with the LLM."""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Create the graph
graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("agent", agent_node)

# Add edges
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

# Compile the graph
app = graph.compile()

# Run the agent
result = app.invoke({
    "messages": [{"role": "user", "content": "Hello! What's 2+2?"}]
})

print(result["messages"][-1].content)
```

### LangGraph with Tool Use

Create `langgraph_tools.py`:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool

# Define tools
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}, 72°F"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [get_weather, calculate]

# Create model with tool binding
model = ChatAnthropic(model="claude-sonnet-4")
model_with_tools = model.bind_tools(tools)

# Define agent node
def agent_node(state: MessagesState):
    """Run the LLM."""
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Define tool node
tool_node = ToolNode(tools)

# Create graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add conditional logic
def should_use_tools(state: MessagesState):
    """Check if tool use is needed."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tools)
graph.add_edge("tools", "agent")

# Compile and run
app = graph.compile()

result = app.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Paris and 15*8?"}]
})

print(result["messages"][-1].content)
```

## Running Your First Agent

### Development with Agent.invoke()

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4"),
    tools=[],
    system_prompt="You are helpful.",
)

# Single invocation
response = agent.invoke({"messages": [{"role": "user", "content": "Hi!"}]})
print(response)
```

### Streaming Responses

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4"),
    tools=[],
)

# Stream the response
for chunk in agent.stream({"messages": [{"role": "user", "content": "Tell me a story"}]}):
    print(chunk, end="")
```

### Async Execution

```python
import asyncio
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

async def run_agent():
    agent = create_agent(
        model=ChatAnthropic(model="claude-sonnet-4"),
        tools=[],
    )
    
    # Async invocation
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Hello!"}]
    })
    return response

result = asyncio.run(run_agent())
print(result)
```

## Common Patterns

### 1. Stateful Conversation

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4"),
    tools=[],
    system_prompt="You are a helpful chatbot. Remember context from the conversation.",
)

messages = []

# First turn
messages.append({"role": "user", "content": "My name is Alice"})
response = agent.invoke({"messages": messages})
messages.append({"role": "assistant", "content": response["messages"][-1].content})

# Second turn - context preserved
messages.append({"role": "user", "content": "What's my name?"})
response = agent.invoke({"messages": messages})
print(response["messages"][-1].content)  # Should remember Alice
```

### 2. Tool as Pydantic Model (Structured Output)

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from typing import Optional

class WeatherInfo(BaseModel):
    """Weather information."""
    city: str
    temperature: int
    condition: str
    humidity: Optional[int] = None

@tool(args_schema=WeatherInfo)
def get_weather(city: str, temperature: int, condition: str, humidity: int = 50) -> str:
    """Get weather with structured output."""
    return f"Weather in {city}: {condition}, {temperature}°F, {humidity}% humidity"

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4"),
    tools=[get_weather],
)
```

### 3. Agent with Memory Store

```python
from langchain.agents import create_agent, tool
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory

# Create memory store
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

@tool
def remember(information: str) -> str:
    """Store information for later."""
    memory.save_context(
        {"input": "information"},
        {"output": "stored"}
    )
    return f"Remembered: {information}"

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4"),
    tools=[remember],
    system_prompt="You can remember important information.",
)
```

### 4. Error Handling and Retries

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def run_agent_with_retry():
    agent = create_agent(
        model=ChatAnthropic(model="claude-sonnet-4"),
        tools=[],
    )
    
    return agent.invoke({
        "messages": [{"role": "user", "content": "Hello"}]
    })

result = run_agent_with_retry()
print(result)
```

### 5. Custom Tool Implementation

```python
from langchain.tools import BaseTool
from typing import Optional

class CustomSearchTool(BaseTool):
    name = "custom_search"
    description = "Search for information using our custom backend"
    
    def _run(self, query: str) -> str:
        # Your custom search logic
        return f"Results for {query}"
    
    async def _arun(self, query: str) -> str:
        # Async version
        return f"Results for {query}"

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4"),
    tools=[CustomSearchTool()],
)
```

## Debugging and Monitoring

### Enable LangSmith Tracing

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your-api-key
```

Then use your agent normally - all calls will be traced to https://smith.langchain.com

### Debug Verbose Output

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4"),
    tools=[],
)

# Enable verbose output
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    verbose=True,  # Shows all agent steps
)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'langchain'"

**Solution:**
```bash
source .venv/bin/activate
pip install -U langchain langchain-anthropic
```

### Issue: "API key not configured"

**Solution:**
```bash
export ANTHROPIC_API_KEY=sk-your-key
# or
export OPENAI_API_KEY=sk-your-key
```

### Issue: Tool not being called

**Solution:** Ensure tool has proper type hints and docstring:
```python
@tool
def my_tool(param: str) -> str:
    """Clear description of what the tool does."""
    return result
```

### Issue: Slow response times

**Solution:** Enable streaming for better UX:
```python
for chunk in agent.stream({"messages": messages}):
    print(chunk)
```

### Issue: Out of memory with long conversations

**Solution:** Implement sliding window memory:
```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=ChatAnthropic(model="claude-sonnet-4"),
    max_token_limit=1000,
)
```

## Next Steps

1. **Explore Agents:** https://python.langchain.com/docs/how_to/create_agent
2. **Learn LangGraph:** https://docs.langchain.com/oss/python/langgraph/overview
3. **Deep Agents (Advanced):** https://python.langchain.com/docs/deepagents/overview
4. **Tools Integration:** https://python.langchain.com/docs/integrations/tools/
5. **Use LangSmith:** https://smith.langchain.com
6. **Deployment:** https://python.langchain.com/docs/deploy/

## Useful Resources

- **Official Docs:** https://python.langchain.com/docs/
- **API Reference:** https://api.python.langchain.com/
- **GitHub:** https://github.com/langchain-ai/langchain
- **Examples:** https://github.com/langchain-ai/langchain/tree/master/examples
- **Community Discord:** https://discord.gg/6adMQxSpJS
- **LangSmith:** https://smith.langchain.com
- **LangGraph Docs:** https://docs.langchain.com/oss/python/langgraph/overview

## Summary

You've learned:
- How to install and configure LangChain
- Creating basic agents with tools
- Using LangGraph for state machines
- Streaming responses
- Common agent patterns
- Debugging with LangSmith
- Error handling and retries

Start simple with LangChain agents, then graduate to LangGraph for more complex workflows!
