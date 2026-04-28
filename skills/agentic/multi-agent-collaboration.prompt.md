# Multi-Agent Collaboration — Agentic Skill Prompt

Designing and implementing systems where multiple AI agents work together to solve complex tasks.

---

## 1. Identity and Mission

Implement multi-agent systems where specialized agents collaborate, communicate, and coordinate to solve problems beyond any single agent's capability. This includes task decomposition across agents, inter-agent communication protocols, shared memory and state management, and mechanisms for resolving conflicts and synthesizing outputs.

---

## 2. Theory & Fundamentals

### 2.1 Multi-Agent Architecture Patterns

**Hierarchical:** Master agent delegates to specialized workers
```
Master Agent
    ├── Researcher Agent
    ├── Coder Agent
    └── Reviewer Agent
```

**Peer-to-Peer:** Agents of equal capability collaborate
```
Agent A ↔ Agent B ↔ Agent C
```

**Debate:** Multiple agents propose, critique, and refine
```
Agent 1 (Propose) → Agent 2 (Critique) → Agent 3 (Mediate)
```

**Market:** Agents as producers/consumers of information
```
Information Market: Agents bid for tasks, provide services
```

### 2.2 Communication Patterns

**Message Types:**
- `REQUEST`: One agent asks another to perform task
- `RESPONSE`: Result of a request
- `QUERY`: Question without expectation of immediate action
- `NOTIFY`: Information broadcast
- `PROPOSE`: Suggestion for group consideration

**Message Passing Semantics:**
- Fire-and-forget: Send without waiting
- Send-and-wait: Block until response
- Publish-subscribe: Topic-based broadcasting

### 2.3 Coordination Mechanisms

**Shared State:** Agents coordinate through common data store
** Blackboard:** Shared knowledge repository
** Contract Net:** Task announcement and bidding
** Token Passing:** Sequential access control

---

## 3. Implementation Patterns

### Pattern 1: Basic Multi-Agent Orchestration

```python
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time

class MessageType(Enum):
    """Types of messages between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    QUERY = "query"
    NOTIFY = "notify"
    PROPOSE = "propose"

@dataclass
class Message:
    """A message passed between agents."""
    id: str
    sender: str
    receivers: List[str]
    message_type: MessageType
    content: Dict
    reply_to: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    role: str
    capabilities: List[str]
    llm: Any
    system_prompt: str = ""

class Agent(ABC):
    """
    Base class for an agent in the multi-agent system.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.running = False
        self.shared_state: Optional[Dict] = None

    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process an incoming message and potentially return a response.
        """
        pass

    async def run(self):
        """Main agent loop."""
        self.running = True

        while self.running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                self.inbox.append(message)

                # Process message
                response = await self.process_message(message)

                if response:
                    await self.send_message(response)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Agent {self.config.name} error: {e}")

    async def send_message(self, message: Message):
        """Send a message through the mediator."""
        if self.mediator:
            await self.mediator.route_message(message)

    def receive_message(self, message: Message):
        """Receive a message (called by mediator)."""
        self.message_queue.put_nowait(message)

    def set_mediator(self, mediator: Any):
        """Set the message mediator."""
        self.mediator = mediator

    def set_shared_state(self, state: Dict):
        """Set shared state reference."""
        self.shared_state = state

    async def stop(self):
        """Stop the agent."""
        self.running = False


class AgentMediator:
    """
    Mediator that routes messages between agents.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.message_log: List[Message] = []

    def register_agent(self, agent: Agent):
        """Register an agent with the mediator."""
        self.agents[agent.config.name] = agent
        agent.set_mediator(self)

    def unregister_agent(self, name: str):
        """Unregister an agent."""
        if name in self.agents:
            del self.agents[name]

    async def route_message(self, message: Message):
        """Route a message to its recipients."""
        self.message_log.append(message)

        for receiver in message.receivers:
            if receiver in self.agents:
                self.agents[receiver].receive_message(message)

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def get_messages(
        self,
        agent_name: Optional[str] = None,
        message_type: Optional[MessageType] = None,
    ) -> List[Message]:
        """Get messages from log."""
        messages = self.message_log

        if agent_name:
            messages = [
                m for m in messages
                if agent_name in m.receivers or m.sender == agent_name
            ]

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        return messages


class MultiAgentSystem:
    """
    Orchestrates multiple agents working together.
    """

    def __init__(self):
        self.mediator = AgentMediator()
        self.agents: Dict[str, Agent] = {}
        self.shared_state: Dict[str, Any] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    def add_agent(self, agent: Agent):
        """Add an agent to the system."""
        self.agents[agent.config.name] = agent
        self.mediator.register_agent(agent)
        agent.set_shared_state(self.shared_state)

    async def start_agent(self, name: str):
        """Start an agent."""
        if name in self.agents and name not in self.tasks:
            self.tasks[name] = asyncio.create_task(self.agents[name].run())

    async def start_all(self):
        """Start all agents."""
        for name in self.agents:
            await self.start_agent(name)

    async def stop_agent(self, name: str):
        """Stop an agent."""
        if name in self.tasks:
            self.tasks[name].cancel()
            del self.tasks[name]

        if name in self.agents:
            await self.agents[name].stop()

    async def stop_all(self):
        """Stop all agents."""
        for name in list(self.tasks.keys()):
            await self.stop_agent(name)

    async def broadcast(
        self,
        sender: str,
        content: Dict,
        message_type: MessageType = MessageType.NOTIFY,
    ):
        """Broadcast a message to all agents."""
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender,
            receivers=list(self.agents.keys()),
            message_type=message_type,
            content=content,
        )
        await self.mediator.route_message(message)

    async def send_message(
        self,
        sender: str,
        receiver: str,
        content: Dict,
        message_type: MessageType = MessageType.REQUEST,
        wait_for_response: bool = True,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """Send a message and optionally wait for response."""
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender,
            receivers=[receiver],
            message_type=message_type,
            content=content,
        )

        await self.mediator.route_message(message)

        if wait_for_response:
            # Wait for response
            try:
                while True:
                    await asyncio.sleep(0.1)
                    messages = self.mediator.get_messages(
                        agent_name=sender,
                        message_type=MessageType.RESPONSE,
                    )

                    # Find response to our message
                    for msg in reversed(messages):
                        if msg.content.get("in_reply_to") == message.id:
                            return msg

            except asyncio.TimeoutError:
                return None

        return None


# Example specialized agents

class ResearcherAgent(Agent):
    """Agent specialized in research and information gathering."""

    async def process_message(self, message: Message) -> Optional[Message]:
        if message.message_type == MessageType.REQUEST:
            query = message.content.get("query", "")

            # Simulate research
            results = await self._research(query)

            return Message(
                id=str(uuid.uuid4()),
                sender=self.config.name,
                receivers=[message.sender],
                message_type=MessageType.RESPONSE,
                content={
                    "results": results,
                    "in_reply_to": message.id,
                },
            )

    async def _research(self, query: str) -> List[str]:
        """Perform research."""
        await asyncio.sleep(0.5)  # Simulate work
        return [
            f"Found information about {query}",
            f"Another relevant finding about {query}",
        ]


class CoderAgent(Agent):
    """Agent specialized in code generation."""

    async def process_message(self, message: Message) -> Optional[Message]:
        if message.message_type == MessageType.REQUEST:
            task = message.content.get("task", "")

            code = await self._generate_code(task)

            return Message(
                id=str(uuid.uuid4()),
                sender=self.config.name,
                receivers=[message.sender],
                message_type=MessageType.RESPONSE,
                content={
                    "code": code,
                    "in_reply_to": message.id,
                },
            )

    async def _generate_code(self, task: str) -> str:
        """Generate code."""
        await asyncio.sleep(0.5)  # Simulate work
        return f"# Code to {task}\nprint('Hello')"


class ReviewerAgent(Agent):
    """Agent specialized in reviewing and critique."""

    async def process_message(self, message: Message) -> Optional[Message]:
        if message.message_type == MessageType.REQUEST:
            artifact = message.content.get("artifact", "")
            artifact_type = message.content.get("type", "code")

            review = await self._review(artifact, artifact_type)

            return Message(
                id=str(uuid.uuid4()),
                sender=self.config.name,
                receivers=[message.sender],
                message_type=MessageType.RESPONSE,
                content={
                    "review": review,
                    "in_reply_to": message.id,
                },
            )

    async def _review(self, artifact: str, artifact_type: str) -> Dict:
        """Review an artifact."""
        await asyncio.sleep(0.3)
        return {
            "quality": "good",
            "issues": [],
            "suggestions": ["Consider adding tests"],
        }


# Example usage

async def example_multi_agent():
    """Example multi-agent collaboration."""
    system = MultiAgentSystem()

    # Create agents
    researcher = ResearcherAgent(AgentConfig(
        name="researcher",
        role="research",
        capabilities=["search", "summarize", "analyze"],
        llm=None,
    ))

    coder = CoderAgent(AgentConfig(
        name="coder",
        role="coding",
        capabilities=["write_code", "refactor", "debug"],
        llm=None,
    ))

    reviewer = ReviewerAgent(AgentConfig(
        name="reviewer",
        role="review",
        capabilities=["review_code", "review_docs", "quality_check"],
        llm=None,
    ))

    # Add to system
    system.add_agent(researcher)
    system.add_agent(coder)
    system.add_agent(reviewer)

    # Start all agents
    await system.start_all()

    # Orchestrate a task
    print("Starting multi-agent task...")

    # 1. Request research
    research_response = await system.send_message(
        sender="orchestrator",
        receiver="researcher",
        content={"query": "best practices for API design"},
    )

    print(f"Research results: {research_response.content if research_response else 'None'}")

    # 2. Request code generation
    coding_response = await system.send_message(
        sender="orchestrator",
        receiver="coder",
        content={"task": "create a REST API endpoint"},
    )

    print(f"Generated code: {coding_response.content if coding_response else 'None'}")

    # 3. Request review
    if coding_response:
        review_response = await system.send_message(
            sender="orchestrator",
            receiver="reviewer",
            content={
                "artifact": coding_response.content.get("code", ""),
                "type": "code",
            },
        )
        print(f"Review: {review_response.content if review_response else 'None'}")

    # Stop all
    await system.stop_all()
```

### Pattern 2: Hierarchical Task Delegation

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    """A task in the system."""
    id: str
    description: str
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

class MasterOrchestrator:
    """
    Master orchestrator that decomposes and delegates tasks.
    """

    def __init__(
        self,
        agent_registry: Dict[str, Agent],
        shared_state: Dict,
    ):
        self.agent_registry = agent_registry
        self.shared_state = shared_state
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()

    def create_task(
        self,
        description: str,
        assigned_to: Optional[str] = None,
        dependencies: List[str] = None,
    ) -> str:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            description=description,
            assigned_to=assigned_to,
            dependencies=dependencies or [],
        )
        self.tasks[task_id] = task
        return task_id

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies met)."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Check dependencies
            deps_met = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if deps_met:
                ready.append(task)

        return ready

    async def execute_task(self, task: Task) -> Any:
        """Execute a single task."""
        task.status = TaskStatus.IN_PROGRESS

        if task.assigned_to and task.assigned_to in self.agent_registry:
            # Delegate to agent
            agent = self.agent_registry[task.assigned_to]

            message = Message(
                id=str(uuid.uuid4()),
                sender="orchestrator",
                receivers=[task.assigned_to],
                message_type=MessageType.REQUEST,
                content={
                    "task_id": task.id,
                    "description": task.description,
                    "shared_state": self.shared_state,
                },
            )

            agent.receive_message(message)

            # Wait for completion (simplified)
            await asyncio.sleep(1.0)

            # Get result from shared state
            task.result = self.shared_state.get(f"task_result_{task.id}")
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()

        else:
            # Execute directly
            task.result = f"Executed: {task.description}"
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()

        return task.result

    async def run_workflow(self, workflow: List[Dict]) -> Dict:
        """
        Run a workflow of tasks.

        workflow = [
            {"description": "Research X", "agent": "researcher"},
            {"description": "Implement Y", "agent": "coder", "depends_on": [0]},
            {"description": "Review Z", "agent": "reviewer", "depends_on": [1]},
        ]
        """
        # Create tasks
        task_ids = []
        for step in workflow:
            task_id = self.create_task(
                description=step["description"],
                assigned_to=step.get("agent"),
                dependencies=[task_ids[i] for i in step.get("depends_on", [])],
            )
            task_ids.append(task_id)

        # Execute tasks
        completed = []
        while len(completed) < len(task_ids):
            ready_tasks = self.get_ready_tasks()

            if not ready_tasks:
                await asyncio.sleep(0.1)
                continue

            # Execute ready tasks in parallel
            results = await asyncio.gather(
                *[self.execute_task(task) for task in ready_tasks],
                return_exceptions=True,
            )

            completed.extend([t for t in ready_tasks if t.status == TaskStatus.COMPLETED])

        return {
            "tasks": {tid: {"status": self.tasks[tid].status.value, "result": self.tasks[tid].result}
                      for tid in task_ids},
            "shared_state": self.shared_state,
        }


class DynamicTaskDecomposer:
    """
    Dynamically decompose complex tasks using LLM.
    """

    def __init__(self, llm: Any):
        self.llm = llm

    async def decompose(self, task_description: str) -> List[Dict]:
        """Decompose a complex task into subtasks."""
        prompt = f"""Decompose this task into smaller subtasks that can be executed by specialized agents.

Task: {task_description}

Available agent types:
- researcher: For information gathering and analysis
- coder: For code generation and implementation
- reviewer: For reviewing and quality assurance
- writer: For documentation and communication

Respond in this format (JSON):
{{
  "subtasks": [
    {{
      "description": "subtask description",
      "agent": "agent type",
      "depends_on": [index of dependent subtask, or -1 for none]
    }}
  ]
}}"""

        response = await self.llm.generate(prompt)
        # Parse JSON response
        import json
        try:
            data = json.loads(response)
            return data.get("subtasks", [])
        except:
            return [{"description": task_description, "agent": "coder", "depends_on": []}]
```

### Pattern 3: Agent Debate and Consensus

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class DebateRound:
    """A single round of debate."""
    round_number: int
    agent_name: str
    position: str
    arguments: List[str]
    counter_arguments: List[str] = field(default_factory=list)

class DebateAgent:
    """Agent that participates in debates."""

    def __init__(self, name: str, llm: Any, position: str):
        self.name = name
        self.llm = llm
        self.position = position
        self.arguments: List[str] = []

    async def generate_arguments(self, topic: str, previous_arguments: List[str] = None) -> List[str]:
        """Generate arguments for the debate."""
        context = ""
        if previous_arguments:
            context = "Previous arguments:\n" + "\n".join(previous_arguments)

        prompt = f"""You are debating: {topic}
Your position: {self.position}

{context}

Generate 2-3 arguments supporting your position.
Be concise and persuasive."""

        response = await self.llm.generate(prompt)
        self.arguments = [line.strip() for line in response.split("\n") if line.strip()]
        return self.arguments

    async def respond_to_arguments(
        self,
        topic: str,
        opposing_arguments: List[str],
    ) -> List[str]:
        """Respond to opposing arguments."""
        opposing = "\n".join(opposing_arguments)

        prompt = f"""You are debating: {topic}
Your position: {self.position}

Opposing arguments:
{opposing}

Provide counter-arguments to the opposing view. Be direct and specific."""

        response = await self.llm.generate(prompt)
        return [line.strip() for line in response.split("\n") if line.strip()]


class DebateModerator:
    """Moderates debates between agents."""

    def __init__(self, llm: Any):
        self.llm = llm

    async def moderate(
        self,
        topic: str,
        agents: List[DebateAgent],
        num_rounds: int = 3,
    ) -> Dict:
        """Moderate a debate between agents."""
        debate_log: List[DebateRound] = []
        all_arguments: Dict[str, List[str]] = {agent.name: [] for agent in agents}

        for round_num in range(num_rounds):
            print(f"\n=== Round {round_num + 1} ===")

            for agent in agents:
                # Get other agents' arguments
                other_arguments = []
                for other_agent in agents:
                    if other_agent.name != agent.name:
                        other_arguments.extend(all_arguments[other_agent.name])

                # Generate or respond
                if round_num == 0:
                    args = await agent.generate_arguments(topic)
                else:
                    args = await agent.respond_to_arguments(
                        topic,
                        other_arguments,
                    )

                all_arguments[agent.name].extend(args)

                # Log round
                debate_log.append(DebateRound(
                    round_number=round_num,
                    agent_name=agent.name,
                    position=agent.position,
                    arguments=args,
                ))

                print(f"{agent.name}: {args}")

        # Generate final verdict
        verdict = await self._generate_verdict(topic, all_arguments)

        return {
            "topic": topic,
            "debate_log": debate_log,
            "final_arguments": all_arguments,
            "verdict": verdict,
        }

    async def _generate_verdict(self, topic: str, arguments: Dict[str, List[str]]) -> Dict:
        """Generate a verdict based on the debate."""
        args_str = "\n".join([
            f"{name}:\n" + "\n".join(args)
            for name, args in arguments.items()
        ])

        prompt = f"""Based on this debate on: {topic}

Arguments:
{args_str}

Provide a summary of the key points and a balanced conclusion."""

        summary = await self.llm.generate(prompt)

        return {
            "summary": summary,
            "consensus_reached": len(arguments) <= 2,
        }


class ConsensusBuilder:
    """Build consensus among multiple agents."""

    def __init__(self, llm: Any):
        self.llm = llm

    async def build_consensus(
        self,
        question: str,
        agents: List[DebateAgent],
    ) -> Dict:
        """
        Build consensus through structured discussion.
        """
        # Round 1: Initial positions
        initial_positions = {}
        for agent in agents:
            args = await agent.generate_arguments(question)
            initial_positions[agent.name] = args

        # Round 2: Identify common ground and conflicts
        common_ground, conflicts = await self._identify_overlap(initial_positions)

        # Round 3: Address conflicts
        if conflicts:
            await self._resolve_conflicts(question, agents, conflicts)

        # Generate consensus position
        consensus = await self._synthesize_position(question, agents)

        return {
            "initial_positions": initial_positions,
            "common_ground": common_ground,
            "conflicts": conflicts,
            "consensus": consensus,
        }

    async def _identify_overlap(
        self,
        positions: Dict[str, List[str]],
    ) -> tuple[List[str], List[str]]:
        """Identify common ground and conflicts."""
        # Simple keyword-based overlap detection
        all_args = []
        for args in positions.values():
            all_args.extend(args)

        common = []  # Arguments mentioned by multiple agents
        unique = []  # Arguments mentioned by only one agent

        # This is simplified - real implementation would use embeddings
        return common, unique

    async def _resolve_conflicts(
        self,
        question: str,
        agents: List[DebateAgent],
        conflicts: List[str],
    ):
        """Resolve conflicts between agents."""
        pass

    async def _synthesize_position(
        self,
        question: str,
        agents: List[DebateAgent],
    ) -> str:
        """Synthesize a consensus position."""
        positions_text = "\n".join([
            f"{agent.name} ({agent.position}): " + ", ".join(agent.arguments)
            for agent in agents
        ])

        prompt = f"""Given the following positions on '{question}', synthesize a consensus position.

{positions_text}

Provide a balanced synthesis that addresses all viewpoints:"""

        return await self.llm.generate(prompt)
```

### Pattern 4: Blackboard Pattern for Shared Knowledge

```python
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

class KnowledgeType(Enum):
    """Types of knowledge on the blackboard."""
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    PLAN = "plan"
    QUESTION = "question"
    RESULT = "result"

@dataclass
class KnowledgeEntry:
    """An entry on the blackboard."""
    id: str
    knowledge_type: KnowledgeType
    content: Any
    source_agent: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class Blackboard:
    """
    Shared knowledge repository (blackboard) for agent coordination.
    """

    def __init__(self):
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.subscribers: Dict[str, Callable] = {}
        self.tags_index: Dict[str, List[str]] = {}  # tag -> entry IDs
        self.type_index: Dict[KnowledgeType, List[str]] = {}

    def add_entry(
        self,
        knowledge_type: KnowledgeType,
        content: Any,
        source_agent: str,
        tags: List[str] = None,
        confidence: float = 1.0,
        metadata: Dict = None,
    ) -> str:
        """Add a new entry to the blackboard."""
        entry_id = str(uuid.uuid4())

        entry = KnowledgeEntry(
            id=entry_id,
            knowledge_type=knowledge_type,
            content=content,
            source_agent=source_agent,
            tags=tags or [],
            confidence=confidence,
            metadata=metadata or {},
        )

        self.entries[entry_id] = entry

        # Update indexes
        if entry.knowledge_type not in self.type_index:
            self.type_index[entry.knowledge_type] = []
        self.type_index[entry.knowledge_type].append(entry_id)

        for tag in entry.tags:
            if tag not in self.tags_index:
                self.tags_index[tag] = []
            self.tags_index[tag].append(entry_id)

        # Notify subscribers
        self._notify_subscribers(entry)

        return entry_id

    def get_entries(
        self,
        knowledge_type: Optional[KnowledgeType] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[KnowledgeEntry]:
        """Query entries by type, tags, or confidence."""
        results = list(self.entries.values())

        if knowledge_type:
            results = [e for e in results if e.knowledge_type == knowledge_type]

        if tags:
            results = [e for e in results if any(tag in e.tags for tag in tags)]

        if min_confidence > 0:
            results = [e for e in results if e.confidence >= min_confidence]

        return results

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a specific entry."""
        return self.entries.get(entry_id)

    def update_entry(self, entry_id: str, **kwargs):
        """Update an existing entry."""
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            for key, value in kwargs.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)

    def subscribe(
        self,
        agent_name: str,
        callback: Callable[[KnowledgeEntry], None],
    ):
        """Subscribe to blackboard updates."""
        self.subscribers[agent_name] = callback

    def _notify_subscribers(self, entry: KnowledgeEntry):
        """Notify subscribers of new entry."""
        for callback in self.subscribers.values():
            callback(entry)

    def get_agent_contributions(self, agent_name: str) -> List[KnowledgeEntry]:
        """Get all entries from a specific agent."""
        return [
            e for e in self.entries.values()
            if e.source_agent == agent_name
        ]

    def clear(self):
        """Clear the blackboard."""
        self.entries.clear()
        self.tags_index.clear()
        self.type_index.clear()


class BlackboardAgent(Agent):
    """Agent that uses blackboard for coordination."""

    def __init__(
        self,
        config: AgentConfig,
        blackboard: Blackboard,
    ):
        super().__init__(config)
        self.blackboard = blackboard
        self.blackboard.subscribe(config.name, self._on_blackboard_update)

    def _on_blackboard_update(self, entry: KnowledgeEntry):
        """React to blackboard updates."""
        # Can be overridden by subclasses
        pass

    def post_to_blackboard(
        self,
        knowledge_type: KnowledgeType,
        content: Any,
        tags: List[str] = None,
        confidence: float = 1.0,
    ) -> str:
        """Post knowledge to blackboard."""
        return self.blackboard.add_entry(
            knowledge_type=knowledge_type,
            content=content,
            source_agent=self.config.name,
            tags=tags,
            confidence=confidence,
        )

    def query_blackboard(
        self,
        knowledge_type: Optional[KnowledgeType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[KnowledgeEntry]:
        """Query the blackboard."""
        return self.blackboard.get_entries(
            knowledge_type=knowledge_type,
            tags=tags,
        )
```

### Pattern 5: Contract Net Protocol

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

class ContractNetState(Enum):
    """States in the contract net protocol."""
    ANNOUNCING = "announcing"
    BIDDING = "bidding"
    AWARDED = "awarded"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TaskAnnouncement:
    """A task announcement in the contract net."""
    task_id: str
    description: str
    requirements: Dict[str, Any]
    deadline: Optional[float] = None
    bid_deadline: Optional[float] = None

@dataclass
class Bid:
    """A bid for a task."""
    task_id: str
    agent_name: str
    capability_description: str
    estimated_completion_time: float
    cost: float
    confidence: float

class ContractNetInitiator:
    """Agent that announces tasks and awards contracts."""

    def __init__(self, name: str, mediator: AgentMediator):
        self.name = name
        self.mediator = mediator
        self.active_announcements: Dict[str, TaskAnnouncement] = {}
        self.received_bids: Dict[str, List[Bid]] = {}

    async def announce_task(
        self,
        description: str,
        requirements: Dict[str, Any],
        recipients: List[str],
        deadline: Optional[float] = None,
        bid_deadline: Optional[float] = None,
    ) -> str:
        """Announce a task to potential contractors."""
        task_id = str(uuid.uuid4())

        announcement = TaskAnnouncement(
            task_id=task_id,
            description=description,
            requirements=requirements,
            deadline=deadline,
            bid_deadline=bid_deadline,
        )

        self.active_announcements[task_id] = announcement
        self.received_bids[task_id] = []

        # Send announcement
        message = Message(
            id=str(uuid.uuid4()),
            sender=self.name,
            receivers=recipients,
            message_type=MessageType.PROPOSE,
            content={
                "type": "task_announcement",
                "task_id": task_id,
                "description": description,
                "requirements": requirements,
                "bid_deadline": bid_deadline,
            },
        )

        await self.mediator.route_message(message)
        return task_id

    async def collect_bids(self, task_id: str, timeout: float = 30.0) -> List[Bid]:
        """Collect bids for a task announcement."""
        start = time.time()

        while time.time() - start < timeout:
            bids = self.received_bids.get(task_id, [])
            if len(bids) >= 3:  # Or other threshold
                return bids
            await asyncio.sleep(0.5)

        return self.received_bids.get(task_id, [])

    async def award_contract(
        self,
        task_id: str,
        winning_agent: str,
        contractors: List[str],
    ):
        """Award the contract to a winning bidder."""
        # Notify winner
        award_message = Message(
            id=str(uuid.uuid4()),
            sender=self.name,
            receivers=[winning_agent],
            message_type=MessageType.REQUEST,
            content={
                "type": "contract_awarded",
                "task_id": task_id,
            },
        )
        await self.mediator.route_message(award_message)

        # Notify losers
        for agent in contractors:
            if agent != winning_agent:
                reject_message = Message(
                    id=str(uuid.uuid4()),
                    sender=self.name,
                    receivers=[agent],
                    message_type=MessageType.NOTIFY,
                    content={
                        "type": "contract_rejected",
                        "task_id": task_id,
                    },
                )
                await self.mediator.route_message(reject_message)


class ContractNetContractor:
    """Agent that bids on and executes tasks."""

    def __init__(
        self,
        name: str,
        capabilities: Dict[str, Any],
        mediator: AgentMediator,
    ):
        self.name = name
        self.capabilities = capabilities
        self.mediator = mediator
        self.active_contracts: Dict[str, Any] = {}

    async def receive_announcement(self, message: Message):
        """Receive a task announcement."""
        content = message.content

        if content.get("type") == "task_announcement":
            task_id = content["task_id"]
            requirements = content["requirements"]

            # Evaluate if we can fulfill
            if self._can_fulfill(requirements):
                # Submit bid
                bid = Bid(
                    task_id=task_id,
                    agent_name=self.name,
                    capability_description="Can complete task",
                    estimated_completion_time=5.0,
                    cost=1.0,
                    confidence=0.9,
                )

                bid_message = Message(
                    id=str(uuid.uuid4()),
                    sender=self.name,
                    receivers=[message.sender],
                    message_type=MessageType.RESPONSE,
                    content={
                        "type": "bid",
                        "task_id": task_id,
                        "agent_name": self.name,
                        "estimated_time": bid.estimated_completion_time,
                        "cost": bid.cost,
                        "confidence": bid.confidence,
                    },
                )

                await self.mediator.route_message(bid_message)

    async def receive_award(self, message: Message):
        """Receive contract award."""
        content = message.content

        if content.get("type") == "contract_awarded":
            task_id = content["task_id"]
            self.active_contracts[task_id] = {
                "status": ContractNetState.EXECUTING,
            }

            # Execute task (simplified)
            await self._execute_task(task_id)

    async def _can_fulfill(self, requirements: Dict[str, Any]) -> bool:
        """Check if we can fulfill requirements."""
        required_skills = requirements.get("required_skills", [])
        return all(skill in self.capabilities for skill in required_skills)

    async def _execute_task(self, task_id: str):
        """Execute the contracted task."""
        await asyncio.sleep(1.0)  # Simulate work
        self.active_contracts[task_id]["status"] = ContractNetState.COMPLETED
```

---

## 4. Framework Integration

### LangChain Agent Integration

```python
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Multi-agent system as LangChain tool
class MultiAgentTool:
    def __init__(self, multi_agent_system: MultiAgentSystem):
        self.system = multi_agent_system

    def run(self, task: str):
        # Use multi-agent system to complete task
        result = asyncio.run(
            self.system.execute_workflow([{"description": task, "agent": "researcher"}])
        )
        return result

# Usage in LangChain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = OpenAIFunctionsAgent.from_llm_and_tools(
    llm=llm,
    tools=[...],
    prompt=prompt,
)
```

---

## 5. Performance Considerations

### Multi-Agent System Benchmarks

| Pattern | Latency | Scalability | Complexity |
|---------|---------|-------------|------------|
| Hierarchical | Medium | Good | Low |
| Peer-to-Peer | Low | Limited | Medium |
| Debate | High | Limited | High |
| Blackboard | Medium | Good | Medium |
| Contract Net | High | Good | High |

### Optimization Tips

1. **Message Batching**: Batch messages to reduce network overhead
2. **Caching Shared State**: Cache frequently accessed blackboard entries
3. **Agent Pooling**: Reuse agent instances for similar tasks
4. **Async Communication**: Use async message passing throughout
5. **Selective Forwarding**: Only send messages to relevant agents

---

## 6. Common Pitfalls

1. **Deadlock**: Agents waiting on each other indefinitely
2. **Starvation**: Some agents never get to act
3. **Message Flood**: Too many messages overwhelming agents
4. **State Inconsistency**: Agents disagreeing on shared state
5. **Cascading Failures**: One agent failure affecting others
6. **Coordination Overhead**: Too much coordination reducing parallelism

---

## 7. Research References

1. https://arxiv.org/abs/2308.12532 — "Multi-Agent Collaboration for RAG"

2. https://arxiv.org/abs/2304.08178 — "Task Decomposition in Multi-Agent Systems"

3. https://arxiv.org/abs/2308.08967 — "AgentVerse: Multi-Agent Collaboration"

4. https://arxiv.org/abs/2307.02106 — "ChatDev: Multi-Agent Communication"

5. https://arxiv.org/abs/2306.04618 — "Multi-Agent Planning with LLM"

6. https://arxiv.org/abs/2309.02738 — "Cooperative Multi-Agent Systems"

7. https://arxiv.org/abs/2308.05928 — "Role-Playing Multi-Agent Collaboration"

---

## 8. Uncertainty and Limitations

**Not Covered:** Agent security and sandboxing, agent resource allocation, distributed multi-agent systems across machines.

**Production Considerations:** Multi-agent systems add complexity. Start with simple hierarchies before attempting complex peer-to-peer or debate patterns. Implement proper timeout and circuit breaker patterns.

(End of file - total 1520 lines)