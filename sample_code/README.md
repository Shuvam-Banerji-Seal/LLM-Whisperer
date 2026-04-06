# LLM-Whisperer Sample Code Repository

High-quality Python sample code, notebooks, and templates for building AI agent systems with AGNO, LangChain, and LangGraph frameworks.

**Author:** Shuvam Banerji Seal  
**Repository:** LLM-Whisperer - Advanced Domain-Specific LLM Engineering Skills

---

## 📁 Repository Structure

```
sample_code/
├── agent_frameworks/          # Complete agent framework implementations
│   ├── minimal/              # Simplest possible examples ("hello world")
│   ├── end_to_end/          # Full production-ready applications
│   └── reference_apps/       # Specialized agent examples
├── agent_patterns/           # Reusable patterns and best practices
└── integration_examples/     # External system integrations
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd LLM-Whisperer

# Install dependencies
pip install -r requirements.txt
```

### Running Examples

```bash
# Run minimal AGNO example
python sample_code/agent_frameworks/minimal/agno_hello_world.py

# Run LangChain research assistant
python sample_code/agent_frameworks/end_to_end/langchain_research_assistant.py

# Run LangGraph workflow
python sample_code/agent_frameworks/end_to_end/langgraph_workflow_orchestrator.py

# Run agent patterns demonstration
python sample_code/agent_patterns/agent_patterns.py

# Run integration examples
python sample_code/integration_examples/integration_examples.py
```

---

## 📚 Framework Documentation

### AGNO Framework
- **Official Docs:** [AGNO on GitHub](https://github.com/tobalo/ai-agent-hello-world)
- **References:**
  - [Building Production-Ready AI Agents](https://medium.com/data-science-collective/building-production-ready-ai-agents-with-agno-a-comprehensive-engineering-guide-22db32413fdd)
  - [Build AI Agents with Simple Code](https://medium.com/code-applied/build-ai-agents-tools-with-simple-code-5d6519c16e67)

### LangChain Framework
- **Official Docs:** [LangChain Documentation](https://docs.langchain.com/)
- **References:**
  - [LangChain Python Tutorial 2026](https://blog.jetbrains.com/pycharm/2026/02/langchain-tutorial-2026/)
  - [Building AI Agents with LangChain](https://www.ai-agentsplus.com/blog/building-ai-agents-with-langchain-tutorial-2026/)

### LangGraph Framework
- **Official Docs:** [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- **References:**
  - [LangGraph Tutorial 2026: Stateful AI Agents](https://growai.in/langgraph-tutorial-stateful-ai-agents-2026/)
  - [Building Your First Stateful Agent](https://abstractalgorithms.dev/langgraph-101-building-your-first-stateful-agent)
  - [LangGraph Installation & Setup](https://machinelearningplus.com/gen-ai/langgraph-installation-setup-first-graph/)

---

## 📂 Detailed Contents

### 1. Agent Frameworks

#### Minimal Examples (`agent_frameworks/minimal/`)

**Purpose:** Simplest possible implementations to understand core concepts.

##### `agno_hello_world.py`
- **What it does:** Creates the most basic AGNO agent
- **Key concepts:** Tool definition, agent initialization, simple execution
- **Dependencies:** agno, openai
- **Run:** `python sample_code/agent_frameworks/minimal/agno_hello_world.py`

##### `langchain_hello_world.py`
- **What it does:** Simplest LangChain agent with math tools
- **Key concepts:** Tool decorator, tool execution, memory management
- **Dependencies:** langchain, langchain-community, openai
- **Run:** `python sample_code/agent_frameworks/minimal/langchain_hello_world.py`

##### `langgraph_hello_world.py`
- **What it does:** Basic state graph with linear node execution
- **Key concepts:** StateGraph, nodes, edges, state management
- **Dependencies:** langgraph, langchain
- **Run:** `python sample_code/agent_frameworks/minimal/langgraph_hello_world.py`

#### End-to-End Applications (`agent_frameworks/end_to_end/`)

**Purpose:** Production-ready applications demonstrating complete workflows.

##### `agno_customer_support_agent.py`
- **What it does:** Full customer support system with ticket creation and escalation
- **Key patterns:**
  - Multi-turn conversation management
  - Knowledge base search
  - Ticket creation and tracking
  - Context awareness
  - Escalation logic
- **Features:**
  - Customer lookup
  - Support ticket system
  - Knowledge base integration
  - Status tracking
- **Run:** `python sample_code/agent_frameworks/end_to_end/agno_customer_support_agent.py`

##### `langchain_research_assistant.py`
- **What it does:** Multi-source research and information gathering system
- **Key patterns:**
  - Document retrieval and ranking
  - Semantic search
  - Multi-source synthesis
  - Citation tracking
  - Confidence scoring
- **Features:**
  - Document management
  - Relevance-based retrieval
  - Citation generation
  - Response synthesis
- **Run:** `python sample_code/agent_frameworks/end_to_end/langchain_research_assistant.py`

##### `langgraph_workflow_orchestrator.py`
- **What it does:** Complex multi-step approval workflow
- **Key patterns:**
  - Complex state management
  - Conditional routing
  - Risk assessment
  - Human-in-the-loop approval
  - Execution path tracking
- **Features:**
  - Request validation
  - Risk scoring
  - Auto-approval for low-risk items
  - Human review for high-risk items
  - Result synthesis
- **Run:** `python sample_code/agent_frameworks/end_to_end/langgraph_workflow_orchestrator.py`

#### Reference Applications (`agent_frameworks/reference_apps/`)

**Purpose:** Specialized agents for specific domains.

##### `agno_code_reviewer.py`
- **What it does:** Automated code analysis and review agent
- **Key features:**
  - Security vulnerability detection
  - Performance issue identification
  - Code quality metrics
  - Automated report generation
  - Issue prioritization
- **Patterns demonstrated:**
  - Pattern-based analysis
  - Quality scoring
  - Issue categorization
  - Severity assessment
- **Run:** `python sample_code/agent_frameworks/reference_apps/agno_code_reviewer.py`

##### `langchain_doc_qa_system.py`
- **What it does:** Document Q&A system with semantic search
- **Key features:**
  - Document ingestion and chunking
  - Semantic search
  - Answer generation with citations
  - Session memory management
  - Confidence scoring
- **Patterns demonstrated:**
  - Vector store simulation
  - Text chunking strategies
  - Citation generation
  - Multi-document support
- **Run:** `python sample_code/agent_frameworks/reference_apps/langchain_doc_qa_system.py`

##### `langgraph_data_analysis_pipeline.py`
- **What it does:** Multi-step data analysis workflow
- **Key features:**
  - Data validation and cleaning
  - Statistical analysis
  - Outlier detection
  - Result synthesis
  - Execution logging
- **Patterns demonstrated:**
  - Multi-step pipeline execution
  - Conditional routing
  - Error handling
  - Progress tracking
- **Run:** `python sample_code/agent_frameworks/reference_apps/langgraph_data_analysis_pipeline.py`

### 2. Agent Patterns (`agent_patterns/`)

**Purpose:** Reusable patterns for building robust agent systems.

#### `agent_patterns.py`

Demonstrates five core patterns:

##### Pattern 1: Tool Use
- `Tool` - Abstract base class
- `ToolRegistry` - Centralized tool management
- `MathTool`, `SearchTool` - Concrete implementations
- **Use case:** Extend agents with custom capabilities

##### Pattern 2: Memory Management
- `Memory` - Abstract memory interface
- `ShortTermMemory` - Recent conversation context (fixed size)
- `LongTermMemory` - Historical storage with summarization
- `HybridMemory` - Combined short and long-term
- **Use case:** Maintain conversation context and history

##### Pattern 3: Request Routing
- `Router` - Basic pattern matching
- `SmartRouter` - Priority-based with defaults
- **Use case:** Route requests to appropriate handlers

##### Pattern 4: Error Handling
- `ErrorHandler` - Error tracking and escalation
- `RetryStrategy` - Exponential backoff
- **Use case:** Resilience and automatic recovery

##### Pattern 5: State Management
- `AgentState` - Immutable state representation
- `StateManager` - Persistence and updates
- **Use case:** Track request lifecycle and recovery

**Run:** `python sample_code/agent_patterns/agent_patterns.py`

### 3. Integration Examples (`integration_examples/`)

**Purpose:** Integrating agents with external systems.

#### `integration_examples.py`

##### Integration 1: Database Connection
- `DatabaseAdapter` - SQLite/PostgreSQL operations
- CRUD operations through agent tools
- User management example
- **Use case:** Persistent data storage

##### Integration 2: External APIs
- `APIClient` - REST API wrapper
- Weather, news, stock data retrieval
- Error handling and retries
- **Use case:** Real-world data integration

##### Integration 3: RAG System
- `RAGSystem` - Document retrieval and generation
- Semantic search
- Answer synthesis from documents
- **Use case:** Knowledge-augmented generation

##### Integration 4: Fine-tuned Models
- `FineTunedModelAdapter` - Custom model serving
- Sentiment classification example
- Batch prediction support
- **Use case:** Domain-specific model integration

##### Integration 5: Async Workflows
- `AsyncAgent` - Async/await patterns
- Concurrent API calls
- Parallel database operations
- **Use case:** Performance optimization

**Run:** `python sample_code/integration_examples/integration_examples.py`

---

## 🏗️ Architecture Patterns

### Agent Architecture
```
┌─────────────────────────────────────────┐
│         User Input/Request              │
└────────────────┬────────────────────────┘
                 │
┌─────────────────▼────────────────────────┐
│      Request Routing & Validation        │
└────────────────┬────────────────────────┘
                 │
┌─────────────────▼────────────────────────┐
│     Agent Decision & Planning            │
└────────────────┬────────────────────────┘
                 │
┌─────────────────▼────────────────────────┐
│    Tool Selection & Execution            │
└────────────────┬────────────────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
  ┌───▼────┐          ┌────▼────┐
  │Database │          │API Call │
  └────┬────┘          └────┬────┘
      │                     │
      └──────────┬──────────┘
                 │
┌─────────────────▼────────────────────────┐
│     Response Generation & Memory        │
└────────────────┬────────────────────────┘
                 │
┌─────────────────▼────────────────────────┐
│         Return to User                   │
└─────────────────────────────────────────┘
```

### State Management
```
┌──────────────────────────────────────────┐
│         Initial State                    │
└────────────────┬─────────────────────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
  ┌──▼──┐    ┌──▼──┐    ┌──▼──┐
  │Node1│    │Node2│    │Node3│
  └──┬──┘    └──┬──┘    └──┬──┘
     │          │          │
     └────┬─────┴────┬─────┘
          │          │
       ┌──▼─────────▼──┐
       │ Error Handler │
       └──┬────────────┘
          │
     ┌────▼─────┐
     │Final State│
     └───────────┘
```

---

## 🛠️ Implementation Guide

### For Beginners

1. **Start with minimal examples** to understand framework basics
   ```bash
   python sample_code/agent_frameworks/minimal/agno_hello_world.py
   python sample_code/agent_frameworks/minimal/langchain_hello_world.py
   python sample_code/agent_frameworks/minimal/langgraph_hello_world.py
   ```

2. **Study agent patterns** to learn reusable components
   ```bash
   python sample_code/agent_patterns/agent_patterns.py
   ```

3. **Explore end-to-end examples** for complete applications
   ```bash
   python sample_code/agent_frameworks/end_to_end/agno_customer_support_agent.py
   ```

### For Experienced Developers

1. **Review reference applications** for specialized domains
2. **Examine integration examples** for system integration
3. **Implement custom patterns** using provided templates
4. **Scale to production** using deployment guidelines

### Custom Agent Development

```python
from sample_code.agent_patterns import ToolRegistry, HybridMemory, SmartRouter

# 1. Set up tools
registry = ToolRegistry()
registry.register(CustomTool())

# 2. Configure memory
memory = HybridMemory()

# 3. Define routing
router = SmartRouter()
router.register_route("pattern", handler)

# 4. Build agent
class CustomAgent:
    def __init__(self):
        self.tools = registry
        self.memory = memory
        self.router = router
    
    def run(self, request):
        # Implement your agent logic
        pass
```

---

## 📊 Framework Comparison

| Feature | AGNO | LangChain | LangGraph |
|---------|------|-----------|-----------|
| **Learning Curve** | Moderate | Steep | Moderate |
| **Tool Integration** | Excellent | Excellent | Good |
| **State Management** | Basic | Good | Excellent |
| **Memory Options** | Limited | Multiple | Flexible |
| **Async Support** | Yes | Yes | Yes |
| **Production Ready** | Yes | Yes | Yes |
| **Community** | Growing | Large | Growing |
| **Documentation** | Good | Excellent | Good |

---

## 🔑 Key Concepts

### Tools
- **Definition:** Functions that agents can invoke to accomplish tasks
- **Types:** System tools (search, math), API tools (weather, news), database tools
- **Pattern:** Tool registry for centralized management

### Memory
- **Short-term:** Recent conversation context (fixed size)
- **Long-term:** Historical information with summarization
- **Hybrid:** Combination of both for comprehensive context

### State
- **Purpose:** Track agent execution progress and results
- **Management:** State machines for predictable flows
- **Persistence:** Save and restore state for recovery

### Routing
- **Purpose:** Direct requests to appropriate handlers
- **Patterns:** Pattern matching, priority-based, ML-based
- **Implementation:** Configurable routing rules

### Error Handling
- **Strategies:** Retry with backoff, fallback handlers, escalation
- **Monitoring:** Track failures for improvement
- **Recovery:** Automatic and manual recovery mechanisms

---

## 🚀 Production Deployment

### Checklist

- [ ] Environment configuration (API keys, database URLs)
- [ ] Error handling and logging
- [ ] Rate limiting and throttling
- [ ] Monitoring and alerting
- [ ] Authentication and authorization
- [ ] Input validation and sanitization
- [ ] Output formatting and validation
- [ ] Performance optimization
- [ ] Database connection pooling
- [ ] API retry strategies
- [ ] State persistence
- [ ] Audit logging

### Deployment Options

```bash
# Docker deployment
docker build -t agent-service .
docker run -p 8000:8000 agent-service

# Kubernetes deployment
kubectl apply -f deployment.yaml

# Serverless deployment
serverless deploy
```

---

## 📈 Performance Optimization

### Caching
- Cache API responses
- Use embeddings caching for RAG
- Memoize expensive computations

### Parallelization
- Use async/await for I/O operations
- Batch process requests
- Run tools concurrently

### Indexing
- Index database queries
- Use vector indices for RAG
- Maintain search indices

---

## 🔒 Security Considerations

1. **Input Validation:** Validate all user inputs
2. **API Keys:** Use environment variables, never hardcode
3. **Rate Limiting:** Implement rate limiting for APIs
4. **Access Control:** Restrict tool access based on permissions
5. **Audit Logging:** Log all agent actions
6. **Data Privacy:** Encrypt sensitive data
7. **Injection Prevention:** Escape user inputs in prompts

---

## 📚 Additional Resources

### Learning Paths

**Python Fundamentals**
- Python official docs: https://docs.python.org/3/
- Real Python tutorials: https://realpython.com/

**AI/ML Fundamentals**
- Fast.ai: https://www.fast.ai/
- Deeplearning.AI: https://www.deeplearning.ai/

**LLM Engineering**
- LLM Papers: https://arxiv.org/
- Hugging Face NLP Course: https://huggingface.co/course

### Tools & Libraries

**Development**
- VS Code: https://code.visualstudio.com/
- Jupyter: https://jupyter.org/
- Poetry: https://python-poetry.org/

**Testing**
- pytest: https://pytest.org/
- hypothesis: https://hypothesis.readthedocs.io/

**Production**
- FastAPI: https://fastapi.tiangolo.com/
- Uvicorn: https://www.uvicorn.org/
- PostgreSQL: https://www.postgresql.org/

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear documentation
4. Add tests for new functionality
5. Submit a pull request

**Code Style:** Follow PEP 8, use type hints, include docstrings

---

## 📝 License

This sample code repository is part of the LLM-Whisperer project.

---

## 👥 Author

**Shuvam Banerji Seal**

- LLM Engineering Expert
- AI Systems Architect
- Building production-ready agent systems

---

## 📞 Support

- **Issues:** Report bugs and request features on GitHub
- **Discussions:** Ask questions in GitHub Discussions
- **Documentation:** See inline code documentation and references

---

## 🌟 Acknowledgments

Thanks to:
- AGNO framework team
- LangChain community
- LangGraph developers
- All contributors and users

---

## 📅 Version History

### v1.0.0 (Current)
- Initial release with AGNO, LangChain, and LangGraph examples
- Complete agent patterns implementation
- Integration examples for databases, APIs, RAG
- Production-ready reference applications

---

**Last Updated:** April 2026  
**Status:** Actively Maintained
