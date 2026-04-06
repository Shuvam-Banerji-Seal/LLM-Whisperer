# LangChain Agent Examples

High-quality, production-ready implementations of LangChain agents for various use cases.

## Overview

This directory contains three complete agent implementations demonstrating different LangChain patterns and capabilities:

### 1. **Chat Agent with Memory** (`chat_agent_with_memory.py`)

A conversational AI agent with dual memory systems (short-term and long-term) for context-aware responses.

**Features:**
- Dual memory architecture: conversation buffer + vector database
- Semantic memory retrieval using embeddings
- Session management and persistence
- Memory export functionality
- Type hints and comprehensive documentation

**Key Classes:**
- `LongTermMemory`: Vector-based semantic memory store
- `ConversationalAgent`: Main agent with memory systems

**Use Cases:**
- Customer support chatbots
- Virtual assistants with context awareness
- Multi-turn conversation systems
- Knowledge retention across sessions

**Source Reference:** [Building Memory-Augmented AI Agents with LangChain](https://medium.com/@saurabhzodex/building-memory-augmented-ai-agents-with-langchain-part-1-2c21cc8050da)

### 2. **Search & Research Agent** (`search_research_agent.py`)

An intelligent research agent for web search, information synthesis, and source tracking.

**Features:**
- Web search tool integration (Tavily)
- Deep research with follow-up queries
- Source tracking and citation management
- Result synthesis using LLM
- Research history and export (Markdown/JSON)
- Structured data classes for results

**Key Classes:**
- `SearchResult`: Structured search result representation
- `SynthesizedAnswer`: Research findings with citations
- `ResearchTools`: Collection of research utilities
- `ResearchAgent`: Main research execution agent

**Use Cases:**
- Market research automation
- Academic research assistance
- Fact-checking and verification
- Competitive intelligence gathering
- Information synthesis for reports

**Source References:**
- [Web Search APIs & LangChain Integration](https://www.firecrawl.dev/glossary/web-search-apis/web-search-apis-langchain-ai-frameworks-integration)
- [LangChain Google Search Agent Tutorial 2026](https://www.searchcans.com/blog/langchain-google-search-agent-tutorial/)

### 3. **Autonomous Workflow Agent** (`autonomous_workflow_agent.py`)

An agent for executing complex multi-step workflows with error handling and task orchestration.

**Features:**
- Task dependency management
- Autonomous decision-making and planning
- Error recovery with exponential backoff retries
- Task prioritization and scheduling
- Progress tracking and reporting
- Workflow visualization and export
- Comprehensive execution statistics

**Key Classes:**
- `Task`: Executable task representation
- `ExecutionResult`: Task execution outcome
- `WorkflowTools`: Workflow execution utilities
- `AutonomousWorkflowAgent`: Main workflow orchestration agent

**Use Cases:**
- Data pipeline orchestration
- Multi-step ETL workflows
- Complex business process automation
- Parallel task execution with dependencies
- Error recovery in distributed systems

**Source References:**
- [Production-Ready LangChain Error Handling Patterns](https://langchain-tutorials.github.io/production-ready-langchain-error-handling-patterns/)
- [Error Handling in LangGraph](https://callsphere.tech/blog/langgraph-error-handling-retry-nodes-fallback-paths-recovery)

---

## Installation

### Prerequisites

- Python 3.10+
- LangChain 0.3+
- OpenAI API key (for LLM access)
- Optional: Tavily API key (for research agent)

### Setup

1. **Install dependencies:**

```bash
pip install langchain langchain-openai langchain-huggingface langchain-community chromadb python-dotenv tavily-python
```

2. **Create `.env` file:**

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=your-tavily-key-here  # Optional, for research agent
```

3. **Verify installation:**

```bash
python -c "import langchain; print(f'LangChain version: {langchain.__version__}')"
```

---

## Usage Examples

### Chat Agent with Memory

```python
from chat_agent_with_memory import ConversationalAgent

# Initialize agent
agent = ConversationalAgent(
    model="gpt-3.5-turbo",
    temperature=0.7,
    memory_window_size=5
)

# Have a conversation
response = agent.chat("Hi, my name is Alice")
print(response)

response = agent.chat("What did I tell you earlier?")
print(response)

# Export session
agent.export_session("./session_export.json")
```

### Research Agent

```python
from search_research_agent import ResearchAgent

# Initialize agent
agent = ResearchAgent(
    model="gpt-3.5-turbo",
    temperature=0.3
)

# Conduct research
answer = agent.research(
    "What are the latest advances in quantum computing?",
    use_deep_research=True
)

# Print formatted answer with citations
print(answer.to_markdown())

# Export findings
agent.export_research("./research_findings.md")
agent.export_research("./research_findings.json")
```

### Autonomous Workflow Agent

```python
from autonomous_workflow_agent import AutonomousWorkflowAgent, Task, TaskPriority

# Initialize agent
agent = AutonomousWorkflowAgent()

# Define tasks
tasks = [
    Task(
        task_id="fetch_data",
        name="Fetch Data",
        description="Retrieve data from source",
        priority=TaskPriority.HIGH
    ),
    Task(
        task_id="process_data",
        name="Process Data",
        description="Clean and transform data",
        dependencies=["fetch_data"]
    ),
    Task(
        task_id="analyze",
        name="Analyze Results",
        description="Perform analysis",
        dependencies=["process_data"]
    ),
]

# Execute workflow
results = agent.execute_workflow(tasks)

# Get execution report
report = agent.get_workflow_report()
print(report)

# Export report
agent.export_workflow_report("./workflow_report.json")
```

---

## Architecture & Design

### Memory Architecture (Chat Agent)

```
User Input
    ↓
Long-Term Memory Retrieval → Semantic Search in Vector Store
    ↓
Short-Term Memory → Recent Conversation Buffer (k=3)
    ↓
LLM Processing with Enhanced Context
    ↓
Response Generation
    ↓
Store in Long-Term Memory for Future Recall
```

### Research Workflow (Research Agent)

```
User Query
    ↓
Web Search (Tavily API)
    ↓
Deep Research (Multi-query) or Direct Search
    ↓
Structure & Format Results
    ↓
LLM Synthesis & Integration
    ↓
Citation Management & Formatting
    ↓
Export (Markdown/JSON)
```

### Task Execution (Workflow Agent)

```
Task Definition
    ↓
Dependency Analysis
    ↓
Task Scheduling & Prioritization
    ↓
Parallel/Sequential Execution
    ↓
Error Detection & Retry Logic
    ↓
Progress Tracking
    ↓
Report Generation
```

---

## API Reference

### Chat Agent with Memory

#### `ConversationalAgent`

```python
agent = ConversationalAgent(
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    memory_window_size: int = 3,
    llm_api_key: Optional[str] = None
)

# Methods
agent.chat(user_input: str, use_long_term_memory: bool = True) -> str
agent.get_session_summary() -> Dict[str, Any]
agent.clear_short_term_memory() -> None
agent.export_session(filepath: str) -> None
```

#### `LongTermMemory`

```python
memory = LongTermMemory(
    persist_dir: str = "./data/memory_store",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    collection_name: str = "conversation_history"
)

# Methods
memory.get_relevant_memories(query: str) -> List[Document]
memory.add_memory(text: str, metadata: Optional[Dict] = None) -> None
memory.format_memory_context(memories: List[Document]) -> str
```

### Research Agent

#### `ResearchAgent`

```python
agent = ResearchAgent(
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    llm_api_key: Optional[str] = None,
    search_api_key: Optional[str] = None
)

# Methods
agent.research(question: str, use_deep_research: bool = False) -> SynthesizedAnswer
agent.get_research_history() -> List[Dict[str, Any]]
agent.export_research(filepath: str) -> None
```

#### `SearchResult`

```python
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.8
    timestamp: str = None
```

### Autonomous Workflow Agent

#### `AutonomousWorkflowAgent`

```python
agent = AutonomousWorkflowAgent(
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    llm_api_key: Optional[str] = None
)

# Methods
agent.execute_workflow(tasks: List[Task], parallel_execution: bool = False) -> Dict[str, ExecutionResult]
agent.get_workflow_report() -> Dict[str, Any]
agent.export_workflow_report(filepath: str) -> None
```

#### `Task`

```python
@dataclass
class Task:
    task_id: str
    name: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    max_retries: int = 3
    timeout_seconds: int = 300
    subtasks: List[Task] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
```

---

## Configuration & Customization

### Custom Embeddings Model (Chat Agent)

```python
agent = ConversationalAgent()
agent.long_term_memory = LongTermMemory(
    embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2"
)
```

### Custom Search Provider (Research Agent)

```python
# Extend ResearchTools class for custom search
class CustomResearchTools(ResearchTools):
    def web_search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        # Custom search implementation
        pass
```

### Custom Tool Integration

All agents support adding custom tools:

```python
custom_tool = Tool(
    name="my_tool",
    func=my_function,
    description="My custom tool"
)
agent.tools.append(custom_tool)
```

---

## Error Handling

### Chat Agent

```python
try:
    response = agent.chat(user_input)
except Exception as e:
    logger.error(f"Chat error: {e}")
    # Graceful fallback
```

### Research Agent

- Automatic retry on API failures
- Empty result handling
- JSON parsing error recovery

### Workflow Agent

- Exponential backoff retry mechanism
- Dependency failure detection
- Task timeout handling
- Comprehensive error logging

---

## Logging & Debugging

All modules use Python's `logging` module:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### Log Levels

- `DEBUG`: Detailed execution steps
- `INFO`: Major operations and milestones
- `WARNING`: Potential issues or skipped operations
- `ERROR`: Failed operations and exceptions

---

## Performance Considerations

### Chat Agent
- Vector store size impacts retrieval speed
- Embedding generation is CPU/GPU bound
- Configure `memory_window_size` based on context needs

### Research Agent
- Web search API calls are network-dependent
- Deep research performs multiple sequential queries
- Result caching reduces redundant searches

### Workflow Agent
- Task dependencies create sequential bottlenecks
- Parallel task execution reduces overall time
- Retry logic with backoff prevents API rate limiting

---

## Testing

Run individual example sections:

```bash
# Test chat agent
python chat_agent_with_memory.py
# Uncomment example functions in __main__

# Test research agent
python search_research_agent.py

# Test workflow agent
python autonomous_workflow_agent.py
```

---

## Production Deployment

### Best Practices

1. **Environment Variables**: Always use `.env` files for secrets
2. **Logging**: Configure structured logging for monitoring
3. **Error Handling**: Implement comprehensive error handling
4. **Rate Limiting**: Respect API rate limits
5. **Monitoring**: Track agent performance metrics

### Scaling Considerations

- Use persistent vector stores (Chroma, Pinecone)
- Implement caching layers for repeated searches
- Configure connection pooling for APIs
- Use async/await for I/O operations

---

## Contributing

To extend these examples:

1. Create feature branches from examples
2. Add comprehensive docstrings
3. Include type hints throughout
4. Add usage examples
5. Update this README with new features

---

## References

### Documentation
- [LangChain Official Documentation](https://python.langchain.com/)
- [LangChain API Reference](https://python.langchain.com/api_reference)
- [LangChain Community Tools](https://python.langchain.com/docs/integrations/tools/)

### Articles & Tutorials
- [Building Memory-Augmented AI Agents](https://medium.com/@saurabhzodex/building-memory-augmented-ai-agents-with-langchain-part-1-2c21cc8050da)
- [LangChain Agents Tutorial 2026](https://blog.jetbrains.com/pycharm/2026/02/langchain-tutorial-2026/)
- [Web Search APIs Integration](https://www.firecrawl.dev/glossary/web-search-apis/web-search-apis-langchain-ai-frameworks-integration)
- [Error Handling Patterns](https://langchain-tutorials.github.io/production-ready-langchain-error-handling-patterns/)
- [LangGraph Error Handling](https://callsphere.tech/blog/langgraph-error-handling-retry-nodes-fallback-paths-recovery)

### Related Projects
- [LangChain Repository](https://github.com/langchain-ai/langchain)
- [LangGraph Repository](https://github.com/langchain-ai/langgraph)

---

## License

These examples are provided as-is for educational and production use.

## Author

**Shuvam Banerji Seal**

For questions or improvements, please refer to the OpenCode documentation or repository issues.

---

## Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Create `.env` file with API keys
- [ ] Run examples with `python <agent_file>.py`
- [ ] Review docstrings and type hints
- [ ] Customize for your use case
- [ ] Deploy with proper logging and monitoring

---

## Changelog

### Version 1.0 (Initial Release)
- Added Chat Agent with Memory
- Added Research Agent with web search
- Added Autonomous Workflow Agent
- Comprehensive documentation and examples
