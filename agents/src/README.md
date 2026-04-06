# LangChain Agents Framework - Production-Ready Modules

## Overview

This directory contains high-quality, production-ready Python modules for building intelligent agents with LangChain. Created by **Shuvam Banerji Seal** on April 6, 2026.

## Contents

### Core Modules

#### 1. **langchain_agent_basics.py** (565 lines)
Foundational agent implementations with full LLM provider support.

**Key Classes:**
- `LLMProvider`: Enum for supported providers (OpenAI, Anthropic, Google, Local)
- `AgentConfig`: Configuration dataclass with all agent parameters
- `LLMInitializer`: Initialize LLMs from various providers with error handling
- `BasicAgent`: Main agent class with conversation management
- `AgentFactory`: Pre-configured agents for specialized tasks

**Features:**
- Multi-provider LLM support with automatic initialization
- Conversation history tracking
- Type hints throughout
- Comprehensive error handling and logging
- Pre-configured agents: Research, Creative, Code

**Reference:** https://python.langchain.com/api_reference

---

#### 2. **langchain_tools_integration.py** (750+ lines)
Complete tool system for creating, managing, and validating tools.

**Key Classes:**
- `ToolCategory`: Enum for organizing tools
- `ToolParameter`: Represents tool parameters with validation
- `ToolSchema`: Complete tool metadata and documentation
- `BaseTool`: Abstract base for all tools
- `FunctionTool`: Wraps Python functions as tools
- `ToolRegistry`: Central registry for tool management
- `ToolValidator`: Parameter validation and type checking
- `DynamicToolLoader`: Runtime tool loading from modules

**Features:**
- Automatic schema generation from function signatures
- Tool registry with search and categorization
- Parameter validation with type coercion
- Dynamic tool loading at runtime
- Comprehensive tool documentation
- Tool versioning and tagging

**Reference:** https://python.langchain.com/docs/how_to/custom_tools

---

#### 3. **langchain_memory_systems.py** (850+ lines)
Multiple memory implementations for context management.

**Key Classes:**
- `Message`: Individual message representation
- `MessageRole`: Enum for message roles
- `BaseMemory`: Abstract base for memory systems
- `ConversationBufferMemory`: Simple buffer memory
- `ConversationSummaryMemory`: Summarizes old messages
- `EntityMemory`: Tracks entities and relationships
- `VectorMemory`: Semantic search with embeddings
- `CustomMemoryBackend`: Template for custom implementations
- `MemoryFactory`: Factory for creating memory instances

**Features:**
- 5 different memory implementations
- Conversation history tracking
- Entity and relationship management
- Semantic similarity search
- Memory statistics and analysis
- Easy memory type switching via factory

**Reference:** https://blog.langchain.com/how-we-built-agent-builders-memory-system/

---

### Supporting Files

#### **__init__.py**
Package initialization with clean API exports:
```python
from langchain_agents import (
    BasicAgent, AgentConfig, LLMProvider,
    FunctionTool, ToolRegistry, ToolCategory,
    ConversationBufferMemory, EntityMemory
)
```

#### **USAGE_GUIDE.md**
Comprehensive documentation including:
- Installation instructions
- Quick start examples
- Complete API reference
- Advanced usage patterns
- Best practices
- Sources and references

#### **example_1_basic_agents.py** (350+ lines)
Demonstrates:
- Creating agent configurations
- Initializing agents with different LLM providers
- Using AgentFactory for specialized agents
- Conversation history tracking
- Best practices for temperature, system prompts, and error handling

#### **example_2_tools_integration.py** (400+ lines)
Demonstrates:
- Creating tools from Python functions
- Managing tools with registry
- Parameter validation and schemas
- Dynamic tool loading
- Tool discovery and search
- Best practices for tool design

#### **example_3_memory_systems.py** (450+ lines)
Demonstrates:
- All memory implementations
- Conversation flow management
- Entity tracking and relationships
- Semantic similarity search
- Memory factory usage
- Best practices for memory selection

---

## Quick Start

### Installation
```bash
pip install langchain langchain-openai langchain-anthropic
```

### Basic Usage
```python
from langchain_agents import BasicAgent, AgentConfig, LLMProvider

# Create and configure agent
config = AgentConfig(
    model="gpt-4",
    provider=LLMProvider.OPENAI,
    temperature=0.7
)

# Initialize and use
agent = BasicAgent(config)
response = agent.invoke("What is machine learning?")
print(response)
```

### With Tools
```python
from langchain_agents import FunctionTool, ToolRegistry, ToolCategory

def calculate_sum(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

tool = FunctionTool(calculate_sum, category=ToolCategory.COMPUTATION)
registry = ToolRegistry()
registry.register(tool)

result = tool.execute(a=10, b=5)
print(result)  # Output: 15
```

### With Memory
```python
from langchain_agents import ConversationBufferMemory, MessageRole

memory = ConversationBufferMemory()
memory.add_message(MessageRole.USER, "Hello!")
memory.add_message(MessageRole.ASSISTANT, "Hi there!")

context = memory.get_context()
stats = memory.get_stats()
```

---

## Architecture

### Design Patterns Used
- **Factory Pattern**: AgentFactory, MemoryFactory
- **Registry Pattern**: ToolRegistry
- **Strategy Pattern**: Multiple memory implementations
- **Template Method**: BaseMemory, BaseTool
- **Builder Pattern**: AgentConfig

### Code Quality
- ✓ Type hints throughout
- ✓ Comprehensive docstrings with examples
- ✓ PEP 8 compliant
- ✓ Proper error handling
- ✓ Logging for debugging
- ✓ Production-ready code

### Testing Coverage
- Each module is independently testable
- Configuration validation
- Parameter validation
- Error handling
- Schema generation

---

## File Statistics

| File | Lines | Purpose |
|------|-------|---------|
| langchain_agent_basics.py | 565 | Agent initialization and execution |
| langchain_tools_integration.py | 750+ | Tool management and validation |
| langchain_memory_systems.py | 850+ | Memory implementations |
| __init__.py | 55 | Package exports |
| USAGE_GUIDE.md | 300+ | Complete documentation |
| example_1_basic_agents.py | 350+ | Agent examples |
| example_2_tools_integration.py | 400+ | Tools examples |
| example_3_memory_systems.py | 450+ | Memory examples |
| **Total** | **4,000+** | Complete framework |

---

## Features Summary

### Agents
- ✓ Multi-provider LLM support
- ✓ Conversation history tracking
- ✓ System prompt customization
- ✓ Temperature and token control
- ✓ Streaming support
- ✓ Error handling and logging
- ✓ Pre-configured specialized agents

### Tools
- ✓ Automatic schema generation
- ✓ Parameter validation and coercion
- ✓ Tool registry with search
- ✓ Dynamic tool loading
- ✓ Tool versioning and tagging
- ✓ Tool categorization
- ✓ Custom tool support

### Memory
- ✓ Buffer memory (complete history)
- ✓ Summary memory (compression)
- ✓ Entity memory (relationship tracking)
- ✓ Vector memory (semantic search)
- ✓ Custom backend template
- ✓ Memory statistics
- ✓ Easy type switching

---

## Documentation References

### Official LangChain Documentation
- **Agents API**: https://python.langchain.com/api_reference
- **Modules**: https://python.langchain.com/docs/modules/
- **Tools**: https://python.langchain.com/docs/how_to/custom_tools
- **Memory**: https://blog.langchain.com/how-we-built-agent-builders-memory-system/

### Provider Documentation
- **OpenAI**: https://platform.openai.com/docs/api-reference
- **Anthropic**: https://docs.anthropic.com
- **Google Gemini**: https://ai.google.dev
- **LangChain Integrations**: https://python.langchain.com/docs/integrations/providers

---

## Usage Patterns

### Pattern 1: Simple Agent
```python
agent = AgentFactory.create_research_agent()
response = agent.invoke("Explain quantum computing")
```

### Pattern 2: Agent with Tools
```python
registry = ToolRegistry()
registry.register(my_tool)
agent = BasicAgent(config)
response = agent.invoke(f"Tools: {registry.get_registry_info()}\nTask: {task}")
```

### Pattern 3: Multi-turn Conversation
```python
memory = ConversationBufferMemory()
while True:
    user_input = input("> ")
    memory.add_message(MessageRole.USER, user_input)
    response = agent.invoke(memory.get_context())
    memory.add_message(MessageRole.ASSISTANT, response)
```

### Pattern 4: Entity Tracking
```python
memory = EntityMemory()
memory.add_entity("Person", {"role": "Engineer", "company": "TechCorp"})
memory.add_relationship("Person", "works at", "TechCorp")
context = memory.get_context()
```

---

## Best Practices

1. **Temperature Settings**
   - Research/Analysis: 0.3-0.5
   - General: 0.7
   - Creative: 1.2+

2. **Memory Selection**
   - Short conversations: BufferMemory
   - Long conversations: SummaryMemory
   - Relationship tracking: EntityMemory
   - Semantic search: VectorMemory

3. **Tool Design**
   - Single responsibility
   - Clear descriptions
   - Type hints
   - Error handling

4. **Error Handling**
   - Validate inputs
   - Log exceptions
   - Provide fallbacks
   - Don't suppress errors

5. **Performance**
   - Monitor token usage
   - Use streaming for long responses
   - Cache tool schemas
   - Batch operations

---

## Contributing

To extend this framework:

1. Create subclasses of BaseTool or BaseMemory
2. Follow existing code patterns
3. Add comprehensive docstrings
4. Include usage examples
5. Add type hints
6. Write unit tests

---

## Author

**Shuvam Banerji Seal**
- Date: April 6, 2026
- Project: LLM-Whisperer
- Location: /agents/src/

---

## Version

**Framework Version**: 1.0.0

**Python Compatibility**: 3.8+

**Dependencies**:
- langchain >= 0.1.0
- langchain-openai >= 0.1.0 (optional)
- langchain-anthropic >= 0.1.0 (optional)
- langchain-google-genai >= 0.1.0 (optional)
- langchain-ollama >= 0.1.0 (optional)

---

## License

Part of the LLM-Whisperer project.

---

## Support

For questions or issues:
1. Check USAGE_GUIDE.md
2. Review example files
3. Check source documentation links
4. Refer to docstrings in modules

---

**End of README**
