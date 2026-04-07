"""Core agent framework."""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentConfig:
    """Configuration for agent."""

    name: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: Optional[str] = None
    tools: List[str] = None


class Tool(ABC):
    """Base class for agent tools."""

    def __init__(self, name: str, description: str):
        """Initialize tool.

        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute tool with given arguments."""
        pass


class Agent:
    """Base agent class."""

    def __init__(self, config: AgentConfig):
        """Initialize agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.state = AgentState.IDLE
        self.tools: Dict[str, Tool] = {}
        self.memory: List[Dict[str, Any]] = []
        self.execution_history = []

    def add_tool(self, tool: Tool):
        """Add tool to agent.

        Args:
            tool: Tool instance
        """
        self.tools[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")

    def run(self, task: str) -> Dict[str, Any]:
        """Run agent on given task.

        Args:
            task: Task description

        Returns:
            Execution result
        """
        logger.info(f"Agent '{self.config.name}' starting task: {task}")
        self.state = AgentState.RUNNING

        try:
            result = self._execute_task(task)
            self.state = AgentState.COMPLETED
            return result
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            self.state = AgentState.FAILED
            return {"error": str(e)}

    def _execute_task(self, task: str) -> Dict[str, Any]:
        """Execute the task.

        Args:
            task: Task to execute

        Returns:
            Execution result
        """
        return {
            "task": task,
            "status": "completed",
            "result": "Task executed successfully",
        }

    def get_memory(self) -> List[Dict[str, Any]]:
        """Get agent memory."""
        return self.memory

    def add_memory(self, key: str, value: Any):
        """Add item to memory.

        Args:
            key: Memory key
            value: Memory value
        """
        self.memory.append({"key": key, "value": value})


class AgentOrchestrator:
    """Orchestrates multiple agents."""

    def __init__(self):
        """Initialize orchestrator."""
        self.agents: Dict[str, Agent] = {}
        self.workflows: Dict[str, List[str]] = {}

    def register_agent(self, agent: Agent):
        """Register agent with orchestrator.

        Args:
            agent: Agent instance
        """
        self.agents[agent.config.name] = agent
        logger.info(f"Registered agent: {agent.config.name}")

    def create_workflow(self, name: str, agent_sequence: List[str]):
        """Create agent workflow.

        Args:
            name: Workflow name
            agent_sequence: List of agent names in execution order
        """
        self.workflows[name] = agent_sequence
        logger.info(f"Created workflow: {name}")

    def execute_workflow(self, workflow_name: str, task: str) -> Dict[str, Any]:
        """Execute workflow.

        Args:
            workflow_name: Name of workflow
            task: Task to execute

        Returns:
            Workflow result
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        agent_sequence = self.workflows[workflow_name]
        results = []

        for agent_name in agent_sequence:
            if agent_name not in self.agents:
                raise ValueError(f"Unknown agent: {agent_name}")

            agent = self.agents[agent_name]
            result = agent.run(task)
            results.append(result)

        return {
            "workflow": workflow_name,
            "task": task,
            "agents_executed": len(agent_sequence),
            "results": results,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = AgentConfig(name="research-agent", model="gpt-4", temperature=0.7)

    agent = Agent(config)
    result = agent.run("Research quantum computing")
    print(result)
