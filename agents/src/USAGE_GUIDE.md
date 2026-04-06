# LangChain Agents Framework - Complete Usage Guide

## Overview

This framework provides production-ready implementations for building intelligent agents with LangChain. It includes:

1. **Agent Basics** - Agent initialization, LLM integration, and execution
2. **Tools Integration** - Tool creation, management, and validation
3. **Memory Systems** - Multiple memory backends for context management

## Installation

```bash
# Core requirements
pip install langchain

# For OpenAI support
pip install langchain-openai

# For Anthropic support
pip install langchain-anthropic

# For Google Gemini support
pip install langchain-google-genai

# For local models (Ollama)
pip install langchain-ollama

# Optional: For vector embeddings
pip install sentence-transformers
```

## Quick Start

### 1. Creating a Basic Agent

```python
from langchain_agents import AgentConfig, LLMProvider, BasicAgent

# Create configuration
config = AgentConfig(
    model="gpt-4",
    provider=LLMProvider.OPENAI,
    temperature=0.7,
    system_prompt="You are a helpful assistant."
)

# Initialize agent
agent = BasicAgent(config)

# Invoke agent
response = agent.invoke("What is machine learning?")
print(response)
```

### 2. Using the Agent Factory

```python
from langchain_agents import AgentFactory

# Create specialized agents
research_agent = AgentFactory.create_research_agent()
creative_agent = AgentFactory.create_creative_agent()
code_agent = AgentFactory.create_code_agent()

# Use them
response = research_agent.invoke("Explain quantum entanglement")
```

### 3. Creating and Using Tools

```python
from langchain_agents import (
    FunctionTool, ToolRegistry, ToolCategory
)

# Define a function
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Create tool
tool = FunctionTool(
    add_numbers,
    category=ToolCategory.COMPUTATION,
    tags=["math", "arithmetic"]
)

# Create registry and register tool
registry = ToolRegistry()
registry.register(tool)

# Execute tool
result = tool.execute(a=5, b=3)  # Returns: 8

# Query registry
tools = registry.search_tools("math")
all_tools = registry.get_all_tools()
```

### 4. Memory Management

```python
from langchain_agents import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    EntityMemory,
    MessageRole,
    MemoryFactory
)

# Buffer memory (stores all messages)
memory = ConversationBufferMemory(max_messages=100)
memory.add_message(MessageRole.USER, "Hello!")
memory.add_message(MessageRole.ASSISTANT, "Hi there!")
context = memory.get_context()

# Summary memory (keeps recent, summarizes old)
summary_memory = ConversationSummaryMemory(keep_recent=5)
summary_memory.add_message(MessageRole.USER, "Tell me about AI")
summary_memory.add_message(MessageRole.ASSISTANT, "AI is...")

# Entity memory (tracks entities and relationships)
entity_memory = EntityMemory()
entity_memory.add_entity("Alice", {"age": 30, "role": "Engineer"})
entity_memory.add_relationship("Alice", "works at", "TechCorp")

# Using factory
memory = MemoryFactory.create_buffer_memory(max_messages=50)
```

## Advanced Usage

### Custom Tool Creation

```python
from langchain_agents import BaseTool, ToolSchema, ToolCategory

class CustomTool(BaseTool):
    def execute(self, **kwargs):
        # Your implementation
        return "result"
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="my_tool",
            description="Description of my tool",
            category=ToolCategory.UTILITY
        )
```

### Dynamic Tool Loading

```python
from langchain_agents import DynamicToolLoader, ToolRegistry

registry = ToolRegistry()
loader = DynamicToolLoader(registry)

# Load from module
loader.load_from_module("my_tools")

# Load function
def my_function():
    """A simple function."""
    return "result"

loader.load_from_function(my_function)
```

### Tool Validation

```python
from langchain_agents import ToolValidator, ToolSchema

schema = tool.get_schema()
validated_params = ToolValidator.validate_parameters(
    schema,
    param1="value1",
    param2=123
)
```

## API Reference

### Agent Basics

#### `AgentConfig`
Configuration dataclass for agents.
- `model`: Model identifier
- `provider`: LLM provider (OPENAI, ANTHROPIC, GOOGLE, LOCAL)
- `temperature`: Sampling temperature (0.0-2.0)
- `max_tokens`: Maximum response tokens
- `system_prompt`: System instruction
- `streaming`: Enable streaming responses

#### `BasicAgent`
Main agent class for interactions.
- `invoke(input, **kwargs)`: Send message and get response
- `get_conversation_history()`: Get chat history
- `clear_history()`: Clear conversation history
- `update_system_prompt(prompt)`: Update system instruction
- `get_config()`: Get current configuration

#### `AgentFactory`
Factory for pre-configured agents.
- `create_openai_agent()`: Create OpenAI agent
- `create_anthropic_agent()`: Create Anthropic agent
- `create_research_agent()`: Optimized for research
- `create_creative_agent()`: Optimized for creativity
- `create_code_agent()`: Optimized for coding

### Tools Integration

#### `ToolRegistry`
Manages collection of tools.
- `register(tool, override=False)`: Add tool to registry
- `get_tool(name)`: Retrieve tool by name
- `get_tools_by_category(category)`: Get tools by category
- `search_tools(query)`: Search tools
- `get_all_tools()`: Get all registered tools

#### `FunctionTool`
Wraps Python functions as tools.
- `execute(**kwargs)`: Execute function
- `get_schema()`: Get tool schema

#### `ToolValidator`
Validates tool parameters.
- `validate_parameters(schema, **kwargs)`: Validate inputs

### Memory Systems

#### `ConversationBufferMemory`
Simple buffer storing all messages.
- `add_message(role, content)`: Add message
- `get_messages()`: Get all messages
- `get_context()`: Format context for LLM
- `clear()`: Clear memory
- `get_stats()`: Get memory statistics

#### `ConversationSummaryMemory`
Summarizes old messages, keeps recent.
- `add_message(role, content)`: Add message
- `get_messages()`: Get recent messages
- `get_context()`: Get summary + recent

#### `EntityMemory`
Tracks entities and relationships.
- `add_entity(name, attributes)`: Add entity
- `add_relationship(entity1, rel, entity2)`: Add relationship
- `get_entities()`: Get all entities
- `get_relationships()`: Get all relationships

#### `VectorMemory`
Uses embeddings for semantic search.
- `add_message(role, content, embedding)`: Add with embedding
- `search(query, top_k)`: Find similar messages

## Examples

### Example 1: Research Assistant

```python
from langchain_agents import AgentFactory, ConversationBufferMemory, MessageRole

# Create agent and memory
agent = AgentFactory.create_research_agent()
memory = ConversationBufferMemory()

# Conversation loop
user_input = "Explain photosynthesis"
memory.add_message(MessageRole.USER, user_input)

response = agent.invoke(user_input)
memory.add_message(MessageRole.ASSISTANT, response)

print(f"Agent: {response}")

# Continue conversation with context
follow_up = "What about the light-dependent reactions?"
memory.add_message(MessageRole.USER, follow_up)
response = agent.invoke(f"{memory.get_context()}\n{follow_up}")
memory.add_message(MessageRole.ASSISTANT, response)
```

### Example 2: Tool-Enabled Agent

```python
from langchain_agents import (
    FunctionTool, ToolRegistry, ToolCategory, BasicAgent, AgentConfig, LLMProvider
)

# Create tools
def calculate_area(length: float, width: float) -> float:
    """Calculate rectangle area."""
    return length * width

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"

# Register tools
registry = ToolRegistry()
registry.register(FunctionTool(calculate_area, category=ToolCategory.COMPUTATION))
registry.register(FunctionTool(get_weather, category=ToolCategory.INTEGRATION))

# Create agent with tools
agent = BasicAgent(AgentConfig(model="gpt-4", provider=LLMProvider.OPENAI))

# Agent can now reason about available tools
response = agent.invoke(f"Available tools: {registry.get_registry_info()}\nCalculate area of 10x20 rectangle")
```

### Example 3: Entity Tracking

```python
from langchain_agents import EntityMemory, MessageRole

memory = EntityMemory()

# Simulate conversation
memory.add_message(MessageRole.USER, "Meet Alice, she's a software engineer at TechCorp")
memory.add_entity("Alice", {
    "role": "Software Engineer",
    "company": "TechCorp",
    "experience": 5
})
memory.add_relationship("Alice", "works at", "TechCorp")

memory.add_message(MessageRole.USER, "Bob works with Alice")
memory.add_entity("Bob", {"role": "Manager"})
memory.add_relationship("Bob", "manages", "Alice")

# Query entities
alice_info = memory.get_entity("Alice")
relationships = memory.get_relationships()

print(f"Alice: {alice_info}")
print(f"Relationships: {relationships}")
```

## Sources & Documentation

- **LangChain Documentation**: https://python.langchain.com/api_reference
- **LangChain Agents**: https://python.langchain.com/docs/modules/agents/
- **LangChain Tools**: https://python.langchain.com/docs/how_to/custom_tools
- **Memory Systems**: https://blog.langchain.com/how-we-built-agent-builders-memory-system/
- **OpenAI API**: https://platform.openai.com/docs/api-reference
- **Anthropic API**: https://docs.anthropic.com

## Best Practices

1. **Temperature Settings**:
   - Research/Analysis: 0.3-0.5 (low = more focused)
   - General Tasks: 0.7 (balanced)
   - Creative Tasks: 1.2+ (high = more diverse)

2. **Memory Management**:
   - Use buffer memory for short conversations
   - Use summary memory for long conversations
   - Use entity memory when tracking relationships

3. **Tool Design**:
   - Keep tools focused and single-purpose
   - Provide clear descriptions and parameters
   - Handle errors gracefully

4. **Performance**:
   - Cache tool schemas
   - Use streaming for long responses
   - Monitor token usage

## Contributing

To extend this framework:

1. Create custom implementations by extending base classes
2. Add new memory backends extending `BaseMemory`
3. Create custom tools extending `BaseTool`
4. Follow PEP 8 style guidelines
5. Include comprehensive docstrings

## License

Part of LLM-Whisperer project
Author: Shuvam Banerji Seal
