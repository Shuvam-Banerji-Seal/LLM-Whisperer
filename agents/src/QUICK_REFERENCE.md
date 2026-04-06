# LangChain Agents Framework - Quick Reference Guide

## Installation

```bash
pip install langchain langchain-openai langchain-anthropic
```

## 1-Minute Quick Start

### Create & Use Agent
```python
from langchain_agents import AgentFactory

agent = AgentFactory.create_research_agent()
response = agent.invoke("What is machine learning?")
print(response)
```

### Create Tool
```python
from langchain_agents import FunctionTool, ToolRegistry

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

tool = FunctionTool(add)
registry = ToolRegistry()
registry.register(tool)
result = tool.execute(a=5, b=3)  # Returns: 8
```

### Use Memory
```python
from langchain_agents import ConversationBufferMemory, MessageRole

memory = ConversationBufferMemory()
memory.add_message(MessageRole.USER, "Hello!")
memory.add_message(MessageRole.ASSISTANT, "Hi there!")
context = memory.get_context()
```

---

## Core Classes

### Agents
| Class | Purpose |
|-------|---------|
| `AgentConfig` | Configure agent parameters |
| `BasicAgent` | Main agent for interactions |
| `LLMInitializer` | Initialize LLMs from providers |
| `AgentFactory` | Create pre-configured agents |

### Tools
| Class | Purpose |
|-------|---------|
| `FunctionTool` | Wrap Python functions |
| `ToolRegistry` | Manage tool collection |
| `ToolValidator` | Validate parameters |
| `ToolSchema` | Tool metadata |

### Memory
| Class | Purpose |
|-------|---------|
| `ConversationBufferMemory` | Store all messages |
| `ConversationSummaryMemory` | Summarize + keep recent |
| `EntityMemory` | Track entities & relationships |
| `VectorMemory` | Semantic similarity search |
| `MemoryFactory` | Create memory instances |

---

## Agent Configuration

```python
from langchain_agents import AgentConfig, LLMProvider

config = AgentConfig(
    model="gpt-4",
    provider=LLMProvider.OPENAI,
    temperature=0.7,  # 0.3-0.5 (focused), 0.7 (balanced), 1.2+ (creative)
    max_tokens=2048,
    system_prompt="You are a helpful assistant."
)

agent = BasicAgent(config)
response = agent.invoke("Your question here")
```

---

## Supported LLM Providers

| Provider | Models | Install |
|----------|--------|---------|
| **OpenAI** | gpt-4, gpt-3.5-turbo | `pip install langchain-openai` |
| **Anthropic** | claude-3-opus, claude-3-sonnet | `pip install langchain-anthropic` |
| **Google** | gemini-pro | `pip install langchain-google-genai` |
| **Local** | llama2, mistral | `pip install langchain-ollama` |

---

## Tool Creation

### From Function
```python
def search(query: str) -> list:
    """Search for documents."""
    return ["doc1", "doc2"]

tool = FunctionTool(
    search,
    category=ToolCategory.DATA_RETRIEVAL,
    tags=["search", "documents"]
)
```

### Custom Tool Class
```python
from langchain_agents import BaseTool, ToolSchema, ToolCategory

class CustomTool(BaseTool):
    def execute(self, **kwargs):
        return "result"
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="my_tool",
            description="Does something",
            category=ToolCategory.UTILITY
        )
```

---

## Memory Types Comparison

| Memory | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Buffer** | Short chats | Simple, fast | Token heavy |
| **Summary** | Long chats | Token efficient | Loses detail |
| **Entity** | Relationships | Structured data | Limited context |
| **Vector** | Semantic search | Selective retrieval | Requires embeddings |

---

## Temperature Settings

```
0.0 ─────────────── 0.5 ─────────────── 0.7 ─────────────── 1.5 ─────────────── 2.0
└─ Deterministic      └─ Analytical      └─ Balanced         └─ Creative        └─ Random
     (Math)            (Research)        (General)          (Writing)          (Too random)
```

---

## Common Patterns

### Pattern: Multi-turn Conversation
```python
memory = ConversationBufferMemory()

while True:
    user_input = input("> ")
    memory.add_message(MessageRole.USER, user_input)
    
    response = agent.invoke(memory.get_context())
    memory.add_message(MessageRole.ASSISTANT, response)
    
    print(f"Agent: {response}")
```

### Pattern: Tool-Enabled Agent
```python
registry = ToolRegistry()
registry.register(tool1)
registry.register(tool2)

# Pass tools to agent
tools_info = str(registry.get_registry_info())
response = agent.invoke(f"Available tools: {tools_info}\nTask: {task}")
```

### Pattern: Entity Tracking
```python
memory = EntityMemory()
memory.add_entity("Alice", {"role": "Engineer", "company": "TechCorp"})
memory.add_relationship("Alice", "works at", "TechCorp")

# Query later
alice = memory.get_entity("Alice")
relationships = memory.get_relationships()
```

---

## Error Handling

```python
from langchain_agents import BasicAgent, AgentConfig, LLMProvider

try:
    config = AgentConfig(
        model="gpt-4",
        provider=LLMProvider.OPENAI
    )
    agent = BasicAgent(config)
    response = agent.invoke("Your question")
except ImportError as e:
    print(f"LLM provider not installed: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Runtime error: {e}")
```

---

## Best Practices

### ✓ DO
- Use type hints for functions
- Provide clear docstrings
- Validate parameters
- Log errors
- Use environment variables for API keys
- Clear memory periodically for long sessions
- Monitor token usage

### ✗ DON'T
- Hardcode API keys
- Ignore errors
- Create too-broad tools
- Use unlimited memory
- Skip parameter validation
- Mix concerns in tools
- Suppress exceptions

---

## Configuration Examples

### Research Agent
```python
research_agent = AgentFactory.create_research_agent()
# temperature=0.3 (focused, accurate)
# Optimized for fact-finding
```

### Creative Agent
```python
creative_agent = AgentFactory.create_creative_agent()
# temperature=1.2 (creative, diverse)
# Optimized for content creation
```

### Code Agent
```python
code_agent = AgentFactory.create_code_agent()
# temperature=0.2 (precise, logical)
# Optimized for programming
```

---

## API Key Setup

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="AIzaSy..."
```

In code:
```python
import os
api_key = os.getenv("OPENAI_API_KEY")
```

---

## Debugging

### Enable Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Agent Configuration
```python
config = agent.get_config()
print(f"Model: {config.model}")
print(f"Provider: {config.provider.value}")
print(f"Temperature: {config.temperature}")
```

### Check Memory Stats
```python
stats = memory.get_stats()
print(f"Messages: {stats['total_messages']}")
print(f"Tokens: {stats['total_tokens_approx']}")
```

### Check Registry Info
```python
info = registry.get_registry_info()
print(f"Total tools: {info['total_tools']}")
print(f"By category: {info['categories']}")
```

---

## Resources

- **Full Guide**: `USAGE_GUIDE.md`
- **Examples**: `example_*.py` files
- **API Reference**: `README.md`
- **LangChain Docs**: https://python.langchain.com
- **Source Code**: See module docstrings

---

## Module Organization

```
langchain_agents/
├── langchain_agent_basics     # Agents & LLM setup
├── langchain_tools_integration # Tools & registry
├── langchain_memory_systems    # Memory implementations
└── __init__.py                 # Clean API exports
```

---

**Quick Tip**: Start with `AgentFactory` for pre-configured agents, then customize with `AgentConfig` as needed.

