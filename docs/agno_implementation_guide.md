# AGNO Framework Implementation Guide

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026

---

## Quick Reference and Implementation Patterns

This document provides ready-to-use code patterns and implementations for AGNO.

---

## 1. Basic Agent Setup

### Minimal Agent

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

# Simplest possible agent
agent = Agent(model=Claude(id="claude-sonnet-4-5"))
agent.print_response("Hello AGNO!")
```

### Agent with LLM Choice

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAI
from agno.models.google import Gemini

# Different LLM options
agents = {
    "claude": Agent(model=Claude(id="claude-sonnet-4-5")),
    "openai": Agent(model=OpenAI(id="gpt-4")),
    "gemini": Agent(model=Gemini(id="gemini-2.0-flash")),
}

# Use as needed
agent = agents["claude"]
response = agent.run("What is 2+2?")
```

### Agent with Configuration

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

agent = Agent(
    name="Advanced Agent",
    model=Claude(id="claude-sonnet-4-5"),
    
    # Behavior
    instructions="You are a helpful assistant. Always provide accurate information.",
    instructions_extend="Only use provided tools.",
    
    # Response format
    markdown=True,
    structured_output=False,
    show_tool_calls=True,
    
    # Limits
    max_iterations=10,
    timeout=30,
    
    # Logging
    show_prompt=False,
    debug_mode=False
)
```

---

## 2. Working with Tools

### Using Built-in Tools

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.coding import CodingTools

agent = Agent(
    name="Developer Assistant",
    model=Claude(id="claude-sonnet-4-5"),
    tools=[
        CodingTools(),
        DuckDuckGoTools(),
    ]
)

# Agent can now:
# 1. Execute Python code
# 2. Search the web
response = agent.run("Search for Python best practices and write example code")
```

### Creating Custom Tools

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import Toolkit
from typing import Any

class DataAnalysisTool(Toolkit):
    """Custom tool for data analysis"""
    
    def calculate_statistics(self, data: list[float]) -> dict:
        """Calculate statistics for data"""
        import statistics
        return {
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "stdev": statistics.stdev(data) if len(data) > 1 else 0,
            "min": min(data),
            "max": max(data)
        }
    
    def find_outliers(self, data: list[float], threshold: float = 2.0) -> list[float]:
        """Find outliers using standard deviation"""
        import statistics
        mean = statistics.mean(data)
        stdev = statistics.stdev(data)
        return [x for x in data if abs(x - mean) > threshold * stdev]

# Use custom tool
agent = Agent(
    name="Data Analyst",
    model=Claude(id="claude-sonnet-4-5"),
    tools=[DataAnalysisTool()]
)

agent.print_response("Analyze this data: [1, 2, 3, 4, 5, 100]")
```

### Tool with API Integration

```python
from agno.tools import Toolkit
import requests

class WeatherTool(Toolkit):
    """Tool for weather data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.weatherapi.com/v1"
    
    def get_current_weather(self, city: str) -> dict:
        """Get current weather for a city"""
        url = f"{self.base_url}/current.json"
        params = {"key": self.api_key, "q": city}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "city": data["location"]["name"],
                "temperature": data["current"]["temp_c"],
                "condition": data["current"]["condition"]["text"],
                "humidity": data["current"]["humidity"]
            }
        return {"error": "City not found"}
    
    def get_forecast(self, city: str, days: int = 3) -> list[dict]:
        """Get weather forecast"""
        url = f"{self.base_url}/forecast.json"
        params = {"key": self.api_key, "q": city, "days": days}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()["forecast"]["forecastday"]
        return []

# Use in agent
import os
agent = Agent(
    name="Weather Agent",
    model=Claude(id="claude-sonnet-4-5"),
    tools=[WeatherTool(api_key=os.getenv("WEATHER_API_KEY"))]
)
```

---

## 3. Memory and State Management

### Conversation History

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

# With conversation history
agent = Agent(
    name="Memory Agent",
    model=Claude(id="claude-sonnet-4-5"),
    add_history_to_context=True,
    num_history_runs=5,  # Remember last 5 conversations
)

# First turn
agent.run("My favorite color is blue")

# Second turn - agent remembers
response = agent.run("What's my favorite color?")
# Output: "Your favorite color is blue"
```

### Persistent Memory (SQLite)

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.db.sqlite import SqliteDb
from pathlib import Path

# Create persistent agent
agent = Agent(
    name="Memory Agent",
    model=Claude(id="claude-sonnet-4-5"),
    db=SqliteDb(db_file="agent_memory.db"),
    add_history_to_context=True,
)

# Conversation 1 (Session 1)
session_1_response = agent.run("I like Python programming", session_id="user_123")

# ... days later ...

# Conversation 2 (Session 2) - still remembers
session_2_response = agent.run(
    "What did I say I like?", 
    session_id="user_123"
)
# Output: "You said you like Python programming"
```

### Knowledge Base Integration

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.knowledge import FileKnowledgeBase, PDFKnowledgeBase

# File-based knowledge
agent = Agent(
    name="FAQ Agent",
    model=Claude(id="claude-sonnet-4-5"),
    knowledge=FileKnowledgeBase(
        path="docs/",  # Directory with markdown/txt files
        vector_db="chroma",  # Or pinecone, weaviate
    ),
    instructions="Answer questions using the provided knowledge base"
)

# PDF-based knowledge
pdf_agent = Agent(
    name="Document Agent",
    model=Claude(id="claude-sonnet-4-5"),
    knowledge=PDFKnowledgeBase(
        path="pdfs/",
        chunk_size=500,
        overlap=50
    )
)
```

### Custom Context

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

# Agent with custom context
agent = Agent(
    name="Contextual Agent",
    model=Claude(id="claude-sonnet-4-5"),
    context={
        "user_name": "John Doe",
        "user_role": "Manager",
        "company": "Acme Corp",
        "today": "Monday, April 7, 2026"
    },
    instructions="""You are an assistant for {user_name} from {company}. 
    Today is {today}. {user_name} is a {user_role}."""
)
```

---

## 4. Team-Based Systems

### Simple Team

```python
from agno.team import Team
from agno.agent import Agent
from agno.models.anthropic import Claude

# Create agents
researcher = Agent(
    name="Researcher",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="Research thoroughly and provide detailed information"
)

writer = Agent(
    name="Writer",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="Write clear, engaging content based on research"
)

# Create team
team = Team(
    name="Content Team",
    model=Claude(id="claude-sonnet-4-5"),
    members=[researcher, writer]
)

# Use team
response = team.run("Write an article about machine learning")
```

### Hierarchical Team

```python
from agno.team import Team
from agno.agent import Agent
from agno.models.anthropic import Claude

# Create hierarchy
ceo = Agent(name="CEO", role="leader")

engineering_lead = Agent(
    name="Engineering Lead",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="Manage engineering decisions"
)

product_lead = Agent(
    name="Product Lead",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="Manage product strategy"
)

# Leadership team
leadership_team = Team(
    name="Leadership",
    model=Claude(id="claude-sonnet-4-5"),
    members=[engineering_lead, product_lead],
    moderator=ceo
)

# Can also create sub-teams
frontend_dev = Agent(name="Frontend Dev")
backend_dev = Agent(name="Backend Dev")

engineering_team = Team(
    name="Engineering",
    members=[frontend_dev, backend_dev],
    lead=engineering_lead
)
```

### Debate Team

```python
from agno.team import Team
from agno.agent import Agent
from agno.models.anthropic import Claude

# Create debaters
pro_advocate = Agent(
    name="Pro Advocate",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="Present arguments in favor of the proposal"
)

critic = Agent(
    name="Critic",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="Present counterarguments and critique"
)

moderator = Agent(
    name="Moderator",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="Facilitate discussion and summarize points"
)

# Debate team
debate_team = Team(
    name="Debate Panel",
    model=Claude(id="claude-sonnet-4-5"),
    members=[pro_advocate, critic, moderator],
    mode="debate"
)

response = debate_team.run("Should we implement a 4-day work week?")
```

---

## 5. Workflows and Orchestration

### Linear Workflow

```python
from agno.workflow import Workflow
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.duckduckgo import DuckDuckGoTools

# Create workflow steps
researcher = Agent(
    name="Researcher",
    model=Claude(id="claude-sonnet-4-5"),
    tools=[DuckDuckGoTools()]
)

analyzer = Agent(
    name="Analyzer",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="Analyze the research findings"
)

writer = Agent(
    name="Writer",
    model=Claude(id="claude-sonnet-4-5"),
    instructions="Write a comprehensive report"
)

# Linear workflow
workflow = Workflow(
    name="Research Report",
    steps=[researcher, analyzer, writer]
)

response = workflow.run("Research recent AI developments")
```

### Conditional Workflow

```python
from agno.workflow import Workflow, Step, Branch, Decision
from agno.agent import Agent
from agno.models.anthropic import Claude

# Decision function
def quality_check(output: str) -> bool:
    """Check if quality threshold met"""
    quality_score = len(output) > 500
    return quality_score

# Create workflow with branching
workflow = Workflow(
    name="Conditional Workflow",
    steps=[
        # Step 1: Generate content
        Step(agent=generator, name="generate"),
        
        # Step 2: Quality check
        Decision(
            condition=quality_check,
            true_steps=[
                Step(agent=approver, name="approve"),
                Step(agent=publisher, name="publish")
            ],
            false_steps=[
                Step(agent=reviewer, name="review"),
                Step(agent=generator, name="regenerate")
            ]
        )
    ]
)
```

### Parallel Workflow

```python
from agno.workflow import Workflow, Parallel
from agno.agent import Agent
from agno.models.anthropic import Claude

# Create parallel workflow
workflow = Workflow(
    name="Parallel Processing",
    steps=[
        # First sequential step
        Step(agent=data_loader, name="load"),
        
        # Parallel processing
        Parallel(
            steps=[
                Step(agent=cleaner, name="clean"),
                Step(agent=validator, name="validate"),
                Step(agent=transformer, name="transform")
            ]
        ),
        
        # Final step
        Step(agent=aggregator, name="aggregate")
    ]
)
```

---

## 6. Production Deployment with AgentOS

### Basic AgentOS Setup

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS

# Create agent with persistence
agent = Agent(
    name="Chatbot",
    model=Claude(id="claude-sonnet-4-5"),
    db=SqliteDb(db_file="agent.db"),
    add_history_to_context=True,
    markdown=True
)

# Create AgentOS
agent_os = AgentOS(
    agents=[agent],
    tracing=True,
    debug=False
)

# Get FastAPI app
app = agent_os.get_app()

# Run with: uvicorn script:app --reload
```

### Multi-Agent AgentOS

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS

# Create multiple agents
support_agent = Agent(
    name="Support Agent",
    model=Claude(id="claude-sonnet-4-5"),
    db=SqliteDb(db_file="support.db"),
    role="support"
)

sales_agent = Agent(
    name="Sales Agent",
    model=Claude(id="claude-sonnet-4-5"),
    db=SqliteDb(db_file="sales.db"),
    role="sales"
)

# Deploy all agents
agent_os = AgentOS(
    agents=[support_agent, sales_agent],
    tracing=True
)

app = agent_os.get_app()

# Usage:
# POST /chat?agent_id=support_agent
# POST /chat?agent_id=sales_agent
```

### AgentOS with Authentication

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.os import AgentOS
from agno.auth import APIKeyAuth, OAuthAuth

# Create authentication
auth = APIKeyAuth(
    api_key_header="X-API-Key",
    required=True
)

# Create AgentOS with auth
agent_os = AgentOS(
    agents=[agent],
    auth=auth,
    tracing=True
)

app = agent_os.get_app()

# Client usage:
# curl -H "X-API-Key: your-key" http://localhost:8000/chat
```

---

## 7. Advanced Features

### Structured Output

```python
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.anthropic import Claude

class Article(BaseModel):
    title: str
    content: str
    tags: list[str]
    word_count: int

# Agent with structured output
agent = Agent(
    name="Article Writer",
    model=Claude(id="claude-sonnet-4-5"),
    response_model=Article
)

# Get structured response
article = agent.run("Write an article about Python")
print(f"Title: {article.title}")
print(f"Tags: {article.tags}")
```

### Guardrails

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.guardrails import Guardrail

class SafetyGuardrail(Guardrail):
    """Ensure response safety"""
    
    def validate(self, response: str) -> tuple[bool, str]:
        unsafe_terms = ["malware", "exploit", "hack"]
        for term in unsafe_terms:
            if term in response.lower():
                return False, f"Unsafe content: {term}"
        return True, "Safe"

# Agent with guardrails
agent = Agent(
    name="Safe Agent",
    model=Claude(id="claude-sonnet-4-5"),
    guardrails=[SafetyGuardrail()]
)
```

### Approval Workflows

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.approvals import ApprovalRequired

# Agent with approval requirements
agent = Agent(
    name="Admin Agent",
    model=Claude(id="claude-sonnet-4-5"),
    approval_required=True,
    approvals=[
        ApprovalRequired(
            action="delete_data",
            role="admin",
            timeout=3600,
            notify=True
        ),
        ApprovalRequired(
            action="modify_config",
            role="manager",
            timeout=1800
        )
    ]
)
```

### Cost Control

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.cost_control import CostLimit

agent = Agent(
    name="Cost-Controlled Agent",
    model=Claude(id="claude-sonnet-4-5"),
    
    # Token limits
    max_tokens=4096,
    
    # Cost limit
    cost_limit=CostLimit(
        per_request=0.50,  # Max $0.50 per request
        per_day=100.00,    # Max $100 per day
        per_month=3000.00  # Max $3000 per month
    )
)
```

---

## 8. Monitoring and Observability

### Metrics Collection

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.metrics import MetricsCollector

# Create metrics collector
metrics = MetricsCollector(backend="sqlite")

agent = Agent(
    name="Monitored Agent",
    model=Claude(id="claude-sonnet-4-5"),
    metrics_collector=metrics
)

# Track metrics
response = agent.run("Hello")

# Get metrics
print(f"Total requests: {metrics.total_requests}")
print(f"Avg response time: {metrics.avg_response_time}ms")
print(f"Total tokens: {metrics.total_tokens}")
print(f"Error rate: {metrics.error_rate}%")
```

### Logging

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

agent = Agent(
    name="Logged Agent",
    model=Claude(id="claude-sonnet-4-5"),
    log_level="debug",
    show_prompt=True,
    show_tool_calls=True
)

# Logs will show all operations
response = agent.run("Test")
```

### Tracing

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tracing import LangfuseTracing

# Setup tracing
tracer = LangfuseTracing(api_key="your-key")

agent = Agent(
    name="Traced Agent",
    model=Claude(id="claude-sonnet-4-5"),
    tracer=tracer
)

# All operations traced
response = agent.run("Test")

# View traces at https://cloud.langfuse.com
```

---

## 9. Error Handling and Resilience

### Error Handling

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
import logging

logger = logging.getLogger(__name__)

try:
    agent = Agent(
        name="Agent",
        model=Claude(id="claude-sonnet-4-5"),
        timeout=30
    )
    response = agent.run("Query")
    
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    response = "I couldn't process that request."
    
except TimeoutError as e:
    logger.error(f"Request timeout: {e}")
    response = "Request took too long. Please try again."
    
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    response = "An error occurred. Please contact support."
```

### Retry Logic

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def run_agent(agent: Agent, query: str) -> str:
    return agent.run(query)

# Usage
agent = Agent(model=Claude(id="claude-sonnet-4-5"))
response = run_agent(agent, "Test query")
```

### Fallback Mechanisms

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAI

# Primary and fallback agents
primary_agent = Agent(
    name="Primary",
    model=Claude(id="claude-sonnet-4-5")
)

fallback_agent = Agent(
    name="Fallback",
    model=OpenAI(id="gpt-4")
)

def get_response(query: str) -> str:
    try:
        return primary_agent.run(query)
    except Exception as e:
        logger.warning(f"Primary failed: {e}, using fallback")
        return fallback_agent.run(query)
```

---

## 10. Testing

### Unit Testing Agents

```python
import pytest
from agno.agent import Agent
from agno.models.anthropic import Claude

@pytest.fixture
def agent():
    return Agent(
        name="Test Agent",
        model=Claude(id="claude-sonnet-4-5")
    )

def test_agent_response(agent):
    """Test basic agent response"""
    response = agent.run("2 + 2 = ?")
    assert "4" in response
    assert isinstance(response, str)

def test_agent_with_tools(agent):
    """Test agent with tools"""
    agent.tools = [TestTool()]
    response = agent.run("Use test tool")
    assert "test" in response.lower()

def test_agent_memory(agent):
    """Test agent memory"""
    agent.add_history_to_context = True
    agent.run("My name is John")
    response = agent.run("What's my name?")
    assert "John" in response
```

### Integration Testing

```python
import pytest
from agno.team import Team
from agno.agent import Agent

@pytest.fixture
def team():
    agent1 = Agent(name="Agent1")
    agent2 = Agent(name="Agent2")
    return Team(name="Test Team", members=[agent1, agent2])

@pytest.mark.integration
def test_team_collaboration(team):
    """Test team collaboration"""
    response = team.run("Collaborate on a task")
    assert response is not None
    assert len(response) > 0

@pytest.mark.integration
def test_agentos_deployment():
    """Test AgentOS deployment"""
    from agno.os import AgentOS
    
    agent = Agent(name="Test Agent")
    agent_os = AgentOS(agents=[agent])
    app = agent_os.get_app()
    
    assert app is not None
    assert hasattr(app, "get")
```

---

## 11. Performance Optimization

### Token Optimization

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

agent = Agent(
    name="Optimized Agent",
    model=Claude(id="claude-sonnet-4-5"),
    
    # Reduce context
    add_history_to_context=True,
    num_history_runs=3,  # Only last 3
    use_history_summary=True,  # Summarize old history
    
    # Optimize instructions
    instructions="Be concise.",  # Short instructions
    
    # Cache responses
    response_cache_ttl=3600
)
```

### Parallel Execution

```python
from agno.team import Team
from agno.agent import Agent
from concurrent.futures import ThreadPoolExecutor

# Create agents
agents = [
    Agent(name=f"Agent {i}")
    for i in range(5)
]

# Parallel execution
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(agent.run, "Task")
        for agent in agents
    ]
    results = [f.result() for f in futures]
```

### Caching

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.cache import ResponseCache

# Setup cache
cache = ResponseCache(
    backend="redis",  # Or "sqlite"
    ttl=3600,
    max_size=1000
)

agent = Agent(
    name="Cached Agent",
    model=Claude(id="claude-sonnet-4-5"),
    response_cache=cache
)

# First call - slow
agent.run("What is Python?")

# Second call - fast (from cache)
agent.run("What is Python?")
```

---

## 12. Deployment Checklist

### Pre-deployment

- [ ] Test agents thoroughly
- [ ] Set up logging and monitoring
- [ ] Configure error handling
- [ ] Set resource limits
- [ ] Set up authentication
- [ ] Configure rate limiting
- [ ] Enable tracing
- [ ] Setup backups for databases

### Deployment

- [ ] Create environment variables
- [ ] Setup database (SQLite/PostgreSQL)
- [ ] Configure LLM API keys
- [ ] Deploy AgentOS
- [ ] Test endpoints
- [ ] Setup monitoring alerts
- [ ] Enable CORS if needed
- [ ] Configure SSL/TLS

### Post-deployment

- [ ] Monitor performance metrics
- [ ] Watch error logs
- [ ] Track token usage and costs
- [ ] Collect user feedback
- [ ] Plan for scaling
- [ ] Regular backup checks

---

## Quick Commands

```bash
# Install AGNO
pip install agno[os]

# Run local agent
python agent_script.py

# Deploy with uvicorn
uvicorn agent_script:app --reload

# Connect to AgentOS UI
# Visit https://os.agno.com and add local endpoint

# View FastAPI docs
# http://localhost:8000/docs

# View RedDoc
# http://localhost:8000/redoc

# Test endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

---

**Happy building with AGNO!**

For more examples and tutorials, visit: https://docs.agno.com/
