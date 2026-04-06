# AI Agent Design Patterns: A Comprehensive Guide

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Version:** 1.0

## Table of Contents

1. [Introduction](#introduction)
2. [The Five Levels of Agentic Software](#the-five-levels-of-agentic-software)
3. [Single-Agent Patterns](#single-agent-patterns)
4. [Multi-Agent Coordination Patterns](#multi-agent-coordination-patterns)
5. [Hierarchical Agent Patterns](#hierarchical-agent-patterns)
6. [Tool-Using Agent Patterns](#tool-using-agent-patterns)
7. [Planning Agent Patterns](#planning-agent-patterns)
8. [Retrieval-Augmented Agent Patterns](#retrieval-augmented-agent-patterns)
9. [Pattern Selection Guide](#pattern-selection-guide)
10. [Trade-offs Analysis](#trade-offs-analysis)

---

## Introduction

AI agent design patterns are architectural blueprints for building autonomous systems that can perceive their environment, reason about tasks, and take actions. As of 2026, agentic AI is foundational to 40% of enterprise applications (Gartner), and the patterns you choose will determine whether your agents remain reliable novelties or become production powerhouses.

This guide synthesizes the latest architectural patterns from leading frameworks: **Agno**, **LangChain/LangGraph**, **CrewAI**, and **AutoGen**. The patterns scale from stateless single agents to fully autonomous, self-learning multi-agent systems.

---

## The Five Levels of Agentic Software

The Five Levels of Agentic Software is a progressive framework developed by Ashpreet Bedi (Agno CEO) that outlines how agent capability and complexity should evolve step by step. Complexity only makes sense to pay for when simpler approaches hit their ceiling.

### Level 1: Stateless AI Agent

**What it is:** An LLM with a basic toolset (read files, write files, execute code).

**When to use:** Isolated, self-contained tasks.

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.coding import CodingTools
from pathlib import Path

WORKSPACE = Path(__file__).parent.joinpath("workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

agent = Agent(
    name="CodeGenerator",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=(
        "You are a coding agent. Write clean, well-documented code. "
        "Always save your work to files and test by running them."
    ),
    tools=[CodingTools(base_dir=WORKSPACE, all=True)],
    markdown=True,
)

agent.print_response(
    "Write a Fibonacci function, save it to fib.py, and run it to verify",
    stream=True,
)
```

**Trade-offs:**
- ✅ Simple, fast to implement, easy to debug
- ❌ No memory between sessions, limited context, all information must be in prompt

### Level 2: Agents with Session Storage and Knowledge Bases

**What it is:** Agents that remember previous interactions and can retrieve domain knowledge.

**When to use:** When agents serve the same users repeatedly or must follow team conventions.

```python
from agno.db.sqlite import SqliteDb
from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.chroma import ChromaDb, SearchType

db = SqliteDb(db_file=str(WORKSPACE / "agents.db"))

knowledge = Knowledge(
    vector_db=ChromaDb(
        collection="coding-standards",
        path=str(WORKSPACE / "chromadb"),
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

agent = Agent(
    name="CodeGenerator",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[CodingTools(base_dir=WORKSPACE, all=True)],
    knowledge=knowledge,
    search_knowledge=True,
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    markdown=True,
)

# Load coding standards into knowledge base
knowledge.insert(text_content="""
## Project Conventions
- Use type hints on all function signatures
- Write docstrings in Google style
- Prefer list comprehensions over map/filter
- Maximum line length: 88 characters
""")
```

**Trade-offs:**
- ✅ Agents follow standards they weren't trained on, maintain conversation continuity
- ❌ Slight latency increase from knowledge retrieval

### Level 3: Agents with Agentic Memory and Learning

**What it is:** Agents that extract facts from conversations and improve from experience without fine-tuning.

**When to use:** Personal coding assistants, team tools where continuous improvement matters.

```python
from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.tools.reasoning import ReasoningTools

learned_knowledge = Knowledge(
    vector_db=ChromaDb(
        collection="coding-learnings",
        path=str(WORKSPACE / "chromadb"),
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

agent = Agent(
    name="CodeGenerator",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[
        CodingTools(base_dir=WORKSPACE, all=True),
        ReasoningTools(),
    ],
    knowledge=docs_knowledge,
    search_knowledge=True,
    learning=LearningMachine(
        knowledge=learned_knowledge,
        learned_knowledge=LearnedKnowledgeConfig(
            mode=LearningMode.AGENTIC,
        ),
    ),
    enable_agentic_memory=True,
    db=db,
    markdown=True,
)

# Session 1: User teaches a preference
agent.print_response(
    "I prefer functional programming style — no classes, "
    "use pure functions and immutable data. Write a data pipeline.",
    session_id="session_1",
)

# Session 2: New task — agent applies the preference automatically
agent.print_response(
    "Write a log parser that extracts error counts by category.",
    session_id="session_2",
)
```

**Trade-offs:**
- ✅ Continuous improvement without retraining, user profiles build over time
- ❌ Requires infrastructure for learning mechanisms

### Level 4: Multi-Agent Teams

**What it is:** Specialized agents coordinated by a team leader agent.

**When to use:** Need multiple perspectives (code review), naturally decomposing tasks.

```python
from agno.team.team import Team

coder = Agent(
    name="Coder",
    role="Write code based on requirements",
    tools=[CodingTools(base_dir=WORKSPACE, all=True)],
)

reviewer = Agent(
    name="Reviewer",
    role="Review code for quality, bugs, and best practices",
    tools=[CodingTools(base_dir=WORKSPACE,
                       enable_write_file=False,
                       enable_edit_file=False,
                       enable_run_shell=False)],
)

tester = Agent(
    name="Tester",
    role="Write and run tests for the code",
    tools=[CodingTools(base_dir=WORKSPACE, all=True)],
)

coding_team = Team(
    name="Coding Team",
    members=[coder, reviewer, tester],
    show_members_responses=True,
    markdown=True,
)

# The team coordinates itself for complex tasks
coding_team.print_response(
    "Implement a binary search algorithm with comprehensive tests",
)
```

**Trade-offs:**
- ✅ Multiple perspectives, specialized roles, can handle complex tasks
- ❌ Coordination overhead, debugging harder, unpredictable

### Level 5: Production Deployment with AgentOS

**What it is:** Full runtime with observability, persistence, API exposure.

**When to use:** Multiple users, uptime requirements, production reliability needed.

```python
from agno.db.postgres import PostgresDb
from agno.vectordb.pgvector import PgVector, SearchType
from agno.os import AgentOS

db_url = "postgresql+psycopg://ai:ai@localhost:5432/ai"
db = PostgresDb(db_url=db_url)

knowledge = Knowledge(
    vector_db=PgVector(
        db_url=db_url,
        table_name="coding_knowledge",
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

agent_os = AgentOS(
    id="ProductionCodeAgent",
    agents=[coding_agent],
    teams=[coding_team],
    config=config_path,
    tracing=True,
)
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="run:app", reload=True)
```

---

## Single-Agent Patterns

Single-agent patterns are the foundation. They're sufficient for more tasks than teams typically realize.

### Pattern 1: Tool-Calling Agent

The most basic pattern: an LLM with tools for perception and action.

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub

@tool
def search_documentation(query: str) -> str:
    """Search through project documentation."""
    # Implementation
    return f"Documentation results for: {query}"

@tool
def execute_code(code: str) -> str:
    """Execute Python code safely."""
    # Implementation with sandbox
    return "Code executed successfully"

@tool
def read_file(filepath: str) -> str:
    """Read file contents."""
    with open(filepath, 'r') as f:
        return f.read()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_documentation, execute_code, read_file]

# Use hub prompt or create custom
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({
    "input": "Read the README file and explain the project architecture"
})
```

**When to use:**
- Simple, well-scoped tasks
- Tool count < 10
- Single execution path
- Limited reasoning steps

**Reference:** https://python.langchain.com/docs/modules/agents/agent_types/openai_tools

### Pattern 2: ReAct (Reasoning + Acting)

The agent alternates between reasoning about what to do and executing actions.

```python
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI

# The ReAct prompt is designed for step-by-step reasoning
prompt = hub.pull("hwchase17/react-chat")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

result = executor.invoke({
    "input": "Debug why the tests are failing"
})

# Output example:
# Thought: I need to understand what tests are failing
# Action: run_tests
# Observation: Tests failed with error X
# Thought: I need to read the test file to understand what's wrong
# Action: read_file test_file.py
# Observation: [test contents]
# Thought: Now I can see the issue...
```

**When to use:**
- Multi-step reasoning required
- Need explainability
- User expects to see thinking process

**Reference:** https://react-lm.github.io/

---

## Multi-Agent Coordination Patterns

### Pattern 1: Hierarchical (Supervisor) Architecture

A supervisor agent routes tasks to specialist agents.

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Literal
import operator

class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str

llm = ChatOpenAI(model="gpt-4o")

# Define specialist agents
researcher = create_react_agent(
    llm,
    tools=[search_tool, scrape_tool],
    state_modifier="You are a research specialist. Find accurate information."
)

analyst = create_react_agent(
    llm,
    tools=[data_analysis_tool],
    state_modifier="You are a data analyst. Extract insights."
)

writer = create_react_agent(
    llm,
    tools=[write_tool],
    state_modifier="You are a technical writer. Create clear reports."
)

# Supervisor routes tasks
def supervisor_router(state: SupervisorState) -> Literal["researcher", "analyst", "writer", "__end__"]:
    """Route to appropriate agent based on task."""
    messages = state["messages"]
    
    response = llm.invoke([
        {"role": "system", "content": """You are a supervisor managing a team.
        Analyze the task and route to:
        - researcher: for information gathering
        - analyst: for data analysis
        - writer: for documentation
        Return __end__ when complete."""},
        {"role": "user", "content": messages[-1].content}
    ])
    
    routing_decision = response.content.lower()
    if "research" in routing_decision:
        return "researcher"
    elif "analyz" in routing_decision:
        return "analyst"
    elif "write" in routing_decision or "report" in routing_decision:
        return "writer"
    return "__end__"

# Graph composition
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_router)
graph.add_node("researcher", researcher)
graph.add_node("analyst", analyst)
graph.add_node("writer", writer)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", supervisor_router)
graph.add_edge("researcher", "supervisor")
graph.add_edge("analyst", "supervisor")
graph.add_edge("writer", "supervisor")

app = graph.compile()

result = app.invoke({"messages": [{"role": "user", "content": "Research AI trends and write a report"}]})
```

**Advantages:**
- Central control and coordination
- Dynamic task routing
- Easy to understand flow

**Disadvantages:**
- Supervisor becomes bottleneck
- Single point of failure
- Supervisor must make routing decisions correctly

### Pattern 2: Sequential Pipeline

Agents pass work in a fixed sequence, each building on previous output.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

class PipelineState(TypedDict):
    messages: Annotated[list, operator.add]
    research_output: str
    analysis_output: str
    report: str

def research_node(state: PipelineState):
    """Stage 1: Information gathering."""
    result = researcher.invoke({"messages": state["messages"]})
    return {"research_output": result["messages"][-1].content}

def analysis_node(state: PipelineState):
    """Stage 2: Data analysis."""
    analysis_prompt = f"Analyze this research:\n{state['research_output']}"
    result = analyst.invoke({"messages": [{"role": "user", "content": analysis_prompt}]})
    return {"analysis_output": result["messages"][-1].content}

def report_node(state: PipelineState):
    """Stage 3: Report writing."""
    report_prompt = f"""Create a report:
    
Research: {state['research_output']}
Analysis: {state['analysis_output']}"""
    result = writer.invoke({"messages": [{"role": "user", "content": report_prompt}]})
    return {"report": result["messages"][-1].content}

# Pipeline graph
pipeline = StateGraph(PipelineState)
pipeline.add_node("research", research_node)
pipeline.add_node("analysis", analysis_node)
pipeline.add_node("report", report_node)

pipeline.add_edge(START, "research")
pipeline.add_edge("research", "analysis")
pipeline.add_edge("analysis", "report")
pipeline.add_edge("report", END)

app = pipeline.compile()
```

**Advantages:**
- Predictable execution
- Easy to test each stage
- Output of previous stage clearly defined

**Disadvantages:**
- Inflexible: can't change order based on conditions
- Sequential execution (no parallelism)
- Later stages can't influence earlier ones

### Pattern 3: Fan-Out / Fan-In (Parallel Execution)

Multiple agents work on independent tasks in parallel, then results are combined.

```python
from langgraph.graph import StateGraph, START, END
import asyncio

class ParallelState(TypedDict):
    messages: Annotated[list, operator.add]
    research_result: str
    analysis_result: str
    technical_review: str

async def run_parallel_agents(state: ParallelState):
    """Run independent agents in parallel."""
    
    async def research_task():
        return await researcher.ainvoke({"messages": state["messages"]})
    
    async def analysis_task():
        return await analyst.ainvoke({"messages": state["messages"]})
    
    async def review_task():
        return await reviewer.ainvoke({"messages": state["messages"]})
    
    # Execute all in parallel
    results = await asyncio.gather(research_task(), analysis_task(), review_task())
    
    return {
        "research_result": results[0]["messages"][-1].content,
        "analysis_result": results[1]["messages"][-1].content,
        "technical_review": results[2]["messages"][-1].content,
    }

def synthesis_node(state: ParallelState):
    """Combine parallel results."""
    synthesis_prompt = f"""Synthesize these parallel results into a coherent output:
    
Research: {state['research_result']}
Analysis: {state['analysis_result']}
Review: {state['technical_review']}"""
    
    result = writer.invoke({"messages": [{"role": "user", "content": synthesis_prompt}]})
    return {"messages": [result]}

parallel_graph = StateGraph(ParallelState)
parallel_graph.add_node("research", researcher)
parallel_graph.add_node("analysis", analyst)
parallel_graph.add_node("review", reviewer)
parallel_graph.add_node("synthesis", synthesis_node)

# Fan-out: START branches to multiple agents
parallel_graph.add_edge(START, "research")
parallel_graph.add_edge(START, "analysis")
parallel_graph.add_edge(START, "review")

# Fan-in: all agents feed to synthesis
parallel_graph.add_edge("research", "synthesis")
parallel_graph.add_edge("analysis", "synthesis")
parallel_graph.add_edge("review", "synthesis")
parallel_graph.add_edge("synthesis", END)

app = parallel_graph.compile()
```

**Advantages:**
- Faster execution (parallel vs sequential)
- Better utilization of resources
- Combines multiple perspectives

**Disadvantages:**
- Need to handle partial failures
- Results may conflict
- Synthesis requires careful design

---

## Hierarchical Agent Patterns

Hierarchical patterns work best when clear decision-making authority and task decomposition exist.

### Pattern 1: Manager-Worker

A manager agent decomposes problems into sub-tasks for worker agents.

```python
from typing import TypedDict, Literal

class ManagerWorkerState(TypedDict):
    original_task: str
    subtasks: list[dict]  # [{"id": 1, "task": "...", "worker": "..."}, ...]
    subtask_results: dict  # {"subtask_id": "result"}
    final_output: str

def manager_node(state: ManagerWorkerState):
    """Decompose task into subtasks."""
    decomposition_prompt = f"""Break down this task into 3-5 specific subtasks:
    
Task: {state['original_task']}

For each subtask, specify:
1. What needs to be done
2. Which worker should handle it (researcher, analyst, or implementer)

Return as JSON with 'subtasks' array."""
    
    response = llm.invoke([{"role": "user", "content": decomposition_prompt}])
    # Parse response to extract subtasks
    subtasks = parse_json(response.content)["subtasks"]
    
    return {
        "subtasks": [
            {
                "id": i,
                "task": st["description"],
                "worker": st["assigned_worker"],
                "status": "pending"
            }
            for i, st in enumerate(subtasks)
        ]
    }

def execute_subtask(state: ManagerWorkerState, subtask_id: int):
    """Execute a single subtask."""
    subtask = next(st for st in state["subtasks"] if st["id"] == subtask_id)
    worker = get_worker_agent(subtask["worker"])
    
    result = worker.invoke({
        "messages": [{"role": "user", "content": subtask["task"]}]
    })
    
    return {
        "subtask_results": {
            **state["subtask_results"],
            f"subtask_{subtask_id}": result["messages"][-1].content
        }
    }

def synthesize_results(state: ManagerWorkerState):
    """Combine all subtask results."""
    all_results = "\n".join([
        f"Subtask {i}: {state['subtask_results'].get(f'subtask_{i}', '')}"
        for i in range(len(state['subtasks']))
    ])
    
    synthesis_prompt = f"""Synthesize these subtask results into a cohesive response:
    
{all_results}"""
    
    response = writer.invoke({
        "messages": [{"role": "user", "content": synthesis_prompt}]
    })
    
    return {"final_output": response["messages"][-1].content}
```

**When to use:**
- Complex problems with clear sub-components
- Different sub-tasks benefit from different expertise
- Need to parallelize work

### Pattern 2: Hierarchical Teams with Specialized Sub-Teams

Teams can be nested, with higher-level team managing lower-level teams.

```python
from agno.team.team import Team

# Level 1: Specialist teams
code_review_team = Team(
    name="Code Review Team",
    members=[
        Agent(name="StyleChecker", role="Check code style"),
        Agent(name="SecurityReviewer", role="Check security issues"),
        Agent(name="TestCoverageAnalyzer", role="Analyze test coverage"),
    ]
)

documentation_team = Team(
    name="Documentation Team",
    members=[
        Agent(name="DocWriter", role="Write documentation"),
        Agent(name="ExampleCreator", role="Create code examples"),
        Agent(name="Diagrammer", role="Create diagrams"),
    ]
)

# Level 2: Orchestrating team manages sub-teams
orchestrator = Agent(
    name="Orchestrator",
    role="Manage code review and documentation teams"
)

# Orchestrator can delegate to either team
```

---

## Tool-Using Agent Patterns

Agents interact with external systems through tools. Tool design heavily influences agent quality.

### Pattern 1: Tool Calling with Structured Output

Tools return structured data that agents can reliably parse.

```python
from pydantic import BaseModel, Field
from typing import Optional

class ToolResult(BaseModel):
    """Structured tool result."""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

class FileReadResult(BaseModel):
    content: str
    line_count: int
    encoding: str

@tool
def read_file_structured(filepath: str) -> FileReadResult:
    """Read file and return structured result."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return FileReadResult(
            content=content,
            line_count=len(content.split('\n')),
            encoding='utf-8'
        )
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

# Agents can reliably parse FileReadResult
```

### Pattern 2: Tool Composition and Chaining

Build complex operations by chaining simpler tools.

```python
@tool
def extract_functions(code: str) -> list[str]:
    """Extract function definitions from code."""
    import ast
    tree = ast.parse(code)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

@tool
def analyze_function(code: str, function_name: str) -> dict:
    """Analyze a specific function."""
    # Extract the function from code and analyze it
    pass

# Pipeline: read_file -> extract_functions -> analyze_function
def analyze_code_file(filepath: str):
    """Multi-step tool composition."""
    code = read_file_structured(filepath).content
    functions = extract_functions(code)
    analyses = [analyze_function(code, func) for func in functions]
    return analyses
```

### Pattern 3: Parallel Tool Execution

Execute independent tool calls in parallel.

```python
import asyncio
from typing import Coroutine

async def parallel_tool_execution(tasks: list[Coroutine]):
    """Execute multiple tool calls in parallel."""
    results = await asyncio.gather(*tasks)
    return results

# Usage in agent:
async def multi_file_analysis(filepaths: list[str]):
    tasks = [read_file_structured(fp) for fp in filepaths]
    results = await parallel_tool_execution(tasks)
    return results
```

---

## Planning Agent Patterns

Planning agents reason about multi-step sequences before execution.

### Pattern 1: Plan-Then-Execute

Agent creates a plan, then executes it step by step.

```python
from pydantic import BaseModel

class PlanStep(BaseModel):
    step_number: int
    description: str
    tool: str
    tool_input: dict
    expected_output: str
    depends_on: list[int] = []

class Plan(BaseModel):
    steps: list[PlanStep]
    dependencies: dict[int, list[int]]

def planning_agent(task: str) -> Plan:
    """Create a plan for the task."""
    planning_prompt = f"""Create a detailed plan to accomplish:
    
Task: {task}

For each step, specify:
1. Description of what to do
2. Which tool to use
3. Input for the tool
4. Expected output
5. Dependencies on other steps

Return as JSON matching the Plan schema."""
    
    response = llm.with_structured_output(Plan).invoke([
        {"role": "user", "content": planning_prompt}
    ])
    return response

def execute_plan(plan: Plan) -> dict:
    """Execute plan step by step."""
    results = {}
    
    for step in plan.steps:
        # Wait for dependencies
        while not all(dep in results for dep in step.depends_on):
            await asyncio.sleep(0.1)
        
        # Execute step
        tool = get_tool(step.tool)
        result = tool(**step.tool_input)
        results[step.step_number] = result
    
    return results

# Usage
plan = planning_agent("Build and test a new feature")
results = execute_plan(plan)
```

### Pattern 2: Adaptive Planning with Re-planning

Agent can adjust plan based on execution results.

```python
def adaptive_planner(task: str, previous_results: dict = None) -> Plan:
    """Create or adjust plan based on new information."""
    context = ""
    if previous_results:
        context = "Previous execution results:\n" + str(previous_results)
    
    prompt = f"""Create/adjust a plan:
    
Task: {task}
{context}

If there are previous results, explain how you're adjusting the plan based on them."""
    
    response = llm.with_structured_output(Plan).invoke([
        {"role": "user", "content": prompt}
    ])
    return response

def execute_with_replanning(task: str, max_iterations: int = 3):
    """Execute with adaptive replanning."""
    results = {}
    
    for iteration in range(max_iterations):
        plan = adaptive_planner(task, results)
        
        try:
            new_results = execute_plan(plan)
            results.update(new_results)
            
            # Check if done
            if is_task_complete(results):
                return results
        except Exception as e:
            # Continue to next iteration with more context
            results[f"error_iteration_{iteration}"] = str(e)
            continue
    
    return results
```

---

## Retrieval-Augmented Agent Patterns

RAG agents retrieve relevant context before reasoning.

### Pattern 1: Retrieve-Then-Read (RTR)

Retrieve relevant documents, then read and extract information.

```python
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseReranker

class RetrieveReadAgent:
    def __init__(self, vectorstore: Chroma, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Compress results to most relevant
        compressor = LLMListwiseReranker.from_llm(
            llm=llm,
            top_n=3
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
    
    def answer_question(self, question: str) -> str:
        """Retrieve and read."""
        # Step 1: Retrieve relevant documents
        docs = self.compression_retriever.get_relevant_documents(question)
        
        # Step 2: Read and extract
        extraction_prompt = f"""Based on these documents, answer the question:

Question: {question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

Provide a comprehensive answer."""
        
        response = self.llm.invoke([
            {"role": "user", "content": extraction_prompt}
        ])
        return response.content
```

### Pattern 2: Agentic RAG with Iterative Retrieval

Agent decides when to retrieve and can refine queries.

```python
class IterativeRetrievalAgent:
    def __init__(self, vectorstore: Chroma, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retrieval_history = []
    
    def decide_retrieval(self, query: str, context: str) -> bool:
        """Decide if more retrieval is needed."""
        decision_prompt = f"""Based on the query and current context, decide if you need to retrieve more information.

Query: {query}
Current context: {context}

Respond with YES or NO."""
        
        response = self.llm.invoke([
            {"role": "user", "content": decision_prompt}
        ])
        return "YES" in response.content.upper()
    
    def refine_query(self, original_query: str, previous_results: str) -> str:
        """Refine retrieval query based on previous results."""
        refinement_prompt = f"""The previous retrieval didn't fully answer the query.

Original query: {original_query}
Previous results: {previous_results}

Suggest a refined query to retrieve more relevant information."""
        
        response = self.llm.invoke([
            {"role": "user", "content": refinement_prompt}
        ])
        return response.content
    
    def answer_with_iterations(self, query: str, max_iterations: int = 3) -> str:
        """Answer with iterative retrieval."""
        context = ""
        refined_query = query
        
        for iteration in range(max_iterations):
            # Retrieve
            docs = self.vectorstore.as_retriever().get_relevant_documents(refined_query)
            context += "\n".join([doc.page_content for doc in docs])
            
            # Check if we need more retrieval
            if not self.decide_retrieval(query, context):
                break
            
            # Refine for next iteration
            refined_query = self.refine_query(query, context)
        
        # Generate final answer
        final_prompt = f"""Using this context, answer the question thoroughly:

Question: {query}
Context: {context}"""
        
        response = self.llm.invoke([
            {"role": "user", "content": final_prompt}
        ])
        return response.content
```

### Pattern 3: Multi-Source RAG

Retrieve from multiple knowledge sources and reconcile results.

```python
class MultiSourceRAGAgent:
    def __init__(self, sources: dict[str, Chroma], llm: ChatOpenAI):
        """
        sources: {"documentation": chroma_db, "code": chroma_db, "issues": chroma_db}
        """
        self.sources = sources
        self.llm = llm
    
    async def retrieve_from_all_sources(self, query: str) -> dict[str, list]:
        """Retrieve from all sources in parallel."""
        results = {}
        
        async def retrieve_source(name: str, vectorstore: Chroma):
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            return name, retriever.get_relevant_documents(query)
        
        tasks = [
            retrieve_source(name, vectorstore)
            for name, vectorstore in self.sources.items()
        ]
        
        source_results = await asyncio.gather(*tasks)
        for name, docs in source_results:
            results[name] = docs
        
        return results
    
    def reconcile_results(self, query: str, source_results: dict[str, list]) -> str:
        """Reconcile and synthesize results from multiple sources."""
        reconciliation_prompt = f"""Question: {query}

Results from different sources:
"""
        for source_name, docs in source_results.items():
            reconciliation_prompt += f"\n{source_name.upper()}:\n"
            reconciliation_prompt += "\n".join([
                f"- {doc.page_content[:200]}..."
                for doc in docs
            ])
        
        reconciliation_prompt += """

Synthesize these results into a coherent answer. Note any conflicts or agreements."""
        
        response = self.llm.invoke([
            {"role": "user", "content": reconciliation_prompt}
        ])
        return response.content
```

---

## Pattern Selection Guide

### Decision Matrix

| Pattern | Task Complexity | Time Sensitivity | Reasoning Needs | Parallelizable | Learning Required |
|---------|-----------------|------------------|-----------------|-----------------|-----------------|
| Stateless Single Agent | Low | High | Low | No | No |
| ReAct Agent | Medium | Medium | High | No | No |
| Hierarchical (Supervisor) | High | Medium | High | Yes | No |
| Sequential Pipeline | High | Low | Medium | No | No |
| Parallel (Fan-out) | High | High | Medium | Yes | No |
| Multi-Agent Team | Very High | Low | Very High | Yes | Yes |
| Agentic RAG | High | Medium | High | Yes | Yes |

### Quick Selection Flow

```
Start here: What is your task?

├─ Is it a single, focused task?
│  └─ YES → Use Single-Agent or ReAct Pattern
│     └─ Need to show reasoning? → ReAct
│     └─ Simple execution → Single-Agent with Tools
│
├─ Multiple independent parallel tasks?
│  └─ YES → Use Fan-Out / Fan-In (Parallel) Pattern
│
├─ Clear sequential steps?
│  └─ YES → Use Sequential Pipeline Pattern
│
├─ Need dynamic task routing based on input?
│  └─ YES → Use Hierarchical (Supervisor) Pattern
│
├─ Need expertise from multiple specialists?
│  └─ YES → Use Multi-Agent Team Pattern
│
└─ Need to retrieve and reason over documents?
   └─ YES → Use Retrieval-Augmented Agent Pattern
      └─ Iterative refinement needed? → Agentic RAG
      └─ Static retrieval OK? → Retrieve-Then-Read
```

---

## Trade-offs Analysis

### Complexity vs Reliability

```
┌─────────────────────────────────────────────┐
│                                             │
│  Multi-Agent Teams (Level 4-5)              │
│  - Higher capability                        │
│  - More complex debugging                   │
│  - Unpredictable failures                   │
│                                             │
│  Hierarchical Agent (Level 3)               │
│  - Good balance                             │
│  - More predictable                         │
│  - Supervisor bottleneck                    │
│                                             │
│  Pipeline Pattern (Level 2)                 │
│  - Clear execution path                     │
│  - Limited flexibility                      │
│  - Easy to debug                            │
│                                             │
│  Single Agent (Level 1)                     │
│  - Simple and fast                          │
│  - Limited capability                       │
│  - Easy to understand                       │
│                                             │
└─────────────────────────────────────────────┘
   Complexity ─────────────────────→
```

### Speed vs Accuracy

- **Fast**: Single agents, stateless (Level 1)
- **Balanced**: Agents with context (Level 2)
- **Accurate**: Multi-agent with verification (Level 4-5)

### Cost vs Capability

- **Cost-effective**: Single agents with tools (Level 1)
- **Moderate cost**: Agents with memory + RAG (Level 2-3)
- **Higher cost**: Multi-agent systems (Level 4-5)

### Context Window Overflow Solutions

1. **Level 1 (Stateless)**: Accept limitations
2. **Level 2 (Session Storage)**: Checkpoints + sliding windows
3. **Level 3 (Agentic Memory)**: Extract patterns over time
4. **Level 4 (Multi-Agent)**: Distribute context across agents
5. **Level 5 (Production)**: PostgreSQL + distributed storage

### Debugging Complexity

```
Single Agent:
  - All state visible in one place
  - Straightforward execution
  - Easy to add logging

Supervisor Pattern:
  - Routing decisions can be opaque
  - Need visibility into which agent executed
  - State split across supervisor + agents

Multi-Agent Teams:
  - Coordination failures hard to trace
  - Need comprehensive logging/tracing
  - May need observability tools (LangSmith, OTEL)
```

### Production Readiness

| Level | Data Consistency | Error Recovery | Observability | Scalability |
|-------|------------------|-----------------|-----|----------|
| 1 | None | Manual | Logs | Limited |
| 2 | Per-session | Conversation history | Logs + DB | Moderate |
| 3 | Knowledge base | Learning recovery | Logs + metrics | Good |
| 4 | Multi-agent state | Team fallbacks | Advanced | Good |
| 5 | PostgreSQL | Full checkpointing | OTEL + LangSmith | Excellent |

---

## Summary

The most successful agent architectures share common principles:

1. **Start simple** - Begin with single agents, add complexity only when needed
2. **Make state explicit** - Don't hide state in prompts or context windows
3. **Verify assumptions** - Add verification steps before acting on reasoning
4. **Learn from experience** - Build learning mechanisms early
5. **Observe everything** - Implement comprehensive logging and tracing
6. **Plan for failure** - Circuit breakers, retries, fallbacks are non-negotiable

The pattern you choose should match your problem, not your ambitions. A well-designed single agent beats a poorly-coordinated multi-agent team every time.

---

## References

1. **Agno Framework**: https://www.agno.com/blog/the-5-levels-of-agentic-software-a-progressive-framework-for-building-reliable-ai-agents
2. **LangChain Documentation**: https://python.langchain.com/
3. **LangGraph**: https://langchain-ai.github.io/langgraph/
4. **CrewAI**: https://docs.crewai.com/
5. **AutoGen**: https://microsoft.github.io/autogen/
6. **Production LangChain Agents**: https://dev.to/akisharan/building-production-ready-langchain-agents-architectural-patterns-that-work-54af
7. **Multi-Agent Orchestration Patterns**: https://www.youngju.dev/blog/ai-platform/2026-03-14-ai-agent-multi-agent-orchestration-patterns.en
8. **Agentic AI Design Patterns 2026**: https://www.sitepoint.com/the-definitive-guide-to-agentic-design-patterns-in-2026/

