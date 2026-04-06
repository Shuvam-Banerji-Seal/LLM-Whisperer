# Quick Start Guide: Agno Framework

**Author:** Shuvam Banerji Seal

## Overview

Agno is a framework for building, deploying, and managing agentic software at scale. It provides a simple but powerful API for creating agents with tools, memory, knowledge bases, and guardrails.

**Official Documentation:** https://docs.agno.com/

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv (recommended)
- An API key for your chosen model provider (Anthropic, OpenAI, Google, etc.)

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

### Step 2: Install Agno

Basic installation:
```bash
pip install -U agno
```

For Anthropic models (Claude):
```bash
pip install -U 'agno[anthropic]'
```

For OpenAI models (GPT):
```bash
pip install -U 'agno[openai]'
```

For full feature support (AgentOS with streaming, API, UI):
```bash
pip install -U 'agno[os]' anthropic
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

**For Google:**
```bash
export GOOGLE_API_KEY=your-key-here
```

## Your First Agent (5 Minutes)

### Basic Agent

Create a file `hello_agent.py`:

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

# Create an agent with Claude
agent = Agent(
    name="Hello Agent",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="You are a helpful assistant. Be concise and friendly.",
    markdown=True,
)

# Run the agent
agent.print_response("Hello! What's your name?", stream=True)
```

Run it:
```bash
python hello_agent.py
```

### Agent with Tools

Create `news_agent.py`:

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    name="News Agent",
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    instructions="Fetch and summarize top trending stories",
    markdown=True,
)

agent.print_response("What are the top 5 trending stories?", stream=True)
```

Run it:
```bash
python news_agent.py
```

### Agent with Memory (Database)

Create `memory_agent.py`:

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.anthropic import Claude

agent = Agent(
    name="Memory Agent",
    model=Claude(id="claude-sonnet-4-5"),
    db=SqliteDb(db_file="agent_memory.db"),
    add_history_to_context=True,
    num_history_runs=3,  # Include last 3 conversations
    markdown=True,
)

# First run
agent.print_response("I like Python programming")

# Second run - agent remembers context from first run
agent.print_response("What did I tell you earlier?", stream=True)
```

## Running Your First Agent with AgentOS

AgentOS provides streaming, authentication, session isolation, and API endpoints automatically.

### Setup AgentOS

Create `agent_os_app.py`:

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.anthropic import Claude
from agno.os import AgentOS
from agno.tools.hackernews import HackerNewsTools

# Define your agents
main_agent = Agent(
    name="Main Agent",
    model=Claude(id="claude-sonnet-4-5"),
    db=SqliteDb(db_file="agno.db"),
    tools=[HackerNewsTools()],
    add_history_to_context=True,
    markdown=True,
)

research_agent = Agent(
    name="Research Agent",
    model=Claude(id="claude-sonnet-4-5"),
    db=SqliteDb(db_file="agno.db"),
    instructions="You are a research specialist. Provide detailed analysis.",
    markdown=True,
)

# Create AgentOS
agent_os = AgentOS(
    agents=[main_agent, research_agent],
    tracing=True,
)

app = agent_os.get_app()
```

### Run AgentOS

```bash
# Install FastAPI
pip install -U fastapi uvicorn

# Run the server
fastapi dev agent_os_app.py
```

Access the API:
- **Base URL:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **OpenAPI Schema:** http://localhost:8000/openapi.json

## Common First Steps

### 1. Add Multiple Tools

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    name="Multi-Tool Agent",
    model=Claude(id="claude-sonnet-4-5"),
    tools=[
        HackerNewsTools(),
        YFinanceTools(),
        DuckDuckGoTools(),
    ],
    markdown=True,
)

agent.print_response("What's trending in tech news and how's the stock market?")
```

### 2. Add Knowledge Base

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.knowledge import KnowledgeBase

knowledge = KnowledgeBase(
    sources=[
        "path/to/documents",
        "path/to/pdfs",
    ],
    vector_store="sqlite",
)

agent = Agent(
    name="Knowledge Agent",
    model=Claude(id="claude-sonnet-4-5"),
    knowledge=knowledge,
    instructions="Answer questions based on the knowledge base",
    markdown=True,
)

agent.print_response("Tell me about the company policies")
```

### 3. Use Callable Tools (Dynamic Tool Selection)

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

def get_tools(run_context):
    role = (run_context.session_state or {}).get("role", "general")
    if role == "finance":
        return [YFinanceTools()]
    return [DuckDuckGoTools()]

agent = Agent(
    name="Role-Based Agent",
    model=Claude(id="claude-sonnet-4-5"),
    tools=get_tools,
    markdown=True,
)

# Finance role uses different tools
agent.print_response("AAPL stock price?", session_state={"role": "finance"})

# General role uses different tools
agent.print_response("Latest news?", session_state={"role": "general"})
```

### 4. Stream Responses Programmatically

```python
from agno.agent import Agent, RunEvent
from agno.models.anthropic import Claude

agent = Agent(
    name="Streaming Agent",
    model=Claude(id="claude-sonnet-4-5"),
    markdown=True,
)

# Stream response
stream = agent.run("Explain quantum computing", stream=True)
for chunk in stream:
    if chunk.event == RunEvent.run_content:
        print(chunk.content, end="", flush=True)
```

### 5. Add Guardrails

```python
from agno.agent import Agent
from agno.guardrails import Guardrails
from agno.models.anthropic import Claude

guardrails = Guardrails(
    blacklist_keywords=["forbidden_topic"],
    allowed_domains=["example.com"],
)

agent = Agent(
    name="Safe Agent",
    model=Claude(id="claude-sonnet-4-5"),
    guardrails=guardrails,
    markdown=True,
)
```

## Agent Configuration Reference

Here are commonly used Agent parameters:

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.db.sqlite import SqliteDb

agent = Agent(
    # Basic configuration
    name="My Agent",
    model=Claude(id="claude-sonnet-4-5"),
    
    # Instructions and behavior
    instructions="You are a helpful assistant",
    description="Describe what this agent does",
    
    # Tools and knowledge
    tools=[],  # List of tools
    knowledge=None,  # Knowledge base
    
    # Database and memory
    db=SqliteDb(db_file="agent.db"),  # Persistent storage
    add_history_to_context=True,  # Include past runs
    num_history_runs=3,  # How many past runs to include
    
    # Display options
    markdown=True,  # Format output as markdown
    
    # Timestamps
    add_datetime_to_context=True,  # Add current time to context
    
    # Guardrails
    guardrails=None,
    
    # Model settings
    max_tokens=4096,
)
```

## Troubleshooting

### Issue: ImportError for Agno

**Solution:** Make sure you're in the virtual environment and Agno is installed:
```bash
source .venv/bin/activate
pip install -U agno
```

### Issue: "API key not found"

**Solution:** Set your API key as an environment variable:
```bash
export ANTHROPIC_API_KEY=sk-your-key
# or for OpenAI
export OPENAI_API_KEY=sk-your-key
```

### Issue: Agent not using tools

**Solution:** Ensure tools are properly passed and the model supports tool use:
```python
agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),  # Use a model that supports tools
    tools=[YourTool()],  # Pass tools in a list
)
```

### Issue: Slow performance with tools

**Solution:** Agno caches tool results by default during development. For production:
```python
agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[YourTool()],
    cache_response=False,  # Disable caching for real-time data
)
```

### Issue: Memory not persisting

**Solution:** Make sure to use a database backend:
```python
agent = Agent(
    db=SqliteDb(db_file="agent.db"),  # Add this line
    add_history_to_context=True,
)
```

### Issue: AgentOS not starting

**Solution:** Install required dependencies and ensure port 8000 is available:
```bash
pip install -U fastapi uvicorn
# Or use a different port
fastapi dev agent_os_app.py --port 8001
```

## Next Steps

1. **Explore More Tools:** Check available tools at https://docs.agno.com/tools/overview
2. **Add Knowledge:** Learn about knowledge bases at https://docs.agno.com/knowledge/overview
3. **Deploy to Production:** See deployment guides at https://docs.agno.com/deploy/introduction
4. **Use AgentOS UI:** Connect to https://os.agno.com for visual management
5. **Multi-Agent Systems:** Learn to orchestrate multiple agents at https://docs.agno.com/multi-agent/overview
6. **Browse Examples:** 2000+ code examples at https://docs.agno.com/examples/introduction

## Useful Resources

- **Official Documentation:** https://docs.agno.com/
- **GitHub Repository:** https://github.com/agno-ai/agno
- **API Reference:** https://docs.agno.com/api/agent
- **Examples:** https://docs.agno.com/examples/introduction
- **Community:** Discord and GitHub Discussions

## Summary

You've learned how to:
- Install Agno and set up a development environment
- Create a basic agent with Claude
- Add tools and memory to agents
- Run agents in development and production (AgentOS)
- Configure dynamic tool selection
- Handle streaming responses
- Add guardrails for safe agent behavior

Start with the basic examples and gradually add features as needed!
