"""
AGNO Multi-Agent Workflow and Team Orchestration

This module demonstrates advanced patterns for coordinating multiple agents,
managing teams, and implementing complex workflows using the AGNO framework.

Author: Shuvam Banerji Seal
Source: https://docs.agno.com/teams/overview
Source: https://docs.agno.com/workflows/overview
Source: https://deepwiki.com/agno-agi/agno/2.2-team-orchestration
Source: https://www.agno.com/blog/one-agent-is-all-you-need-until-it-isnt
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """
    AGNO Team Execution Modes

    Reference: https://www.agno.com/changelog/orchestrate-multi-agent-teams-with-four-built-in-execution-modes
    """

    SEQUENTIAL = "sequential"  # Agents run one after another
    PARALLEL = "parallel"  # Agents run simultaneously
    HIERARCHICAL = "hierarchical"  # One agent manages others
    DYNAMIC = "dynamic"  # Runtime decision on execution


@dataclass
class AgentRole:
    """
    Definition of an agent's role in a team.

    This captures the responsibilities, capabilities, and
    specialization of individual team members.
    """

    name: str
    description: str
    specializations: List[str]
    tools: List[str] = field(default_factory=list)
    can_delegate: bool = False
    requires_approval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize role to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "specializations": self.specializations,
            "tools": self.tools,
            "can_delegate": self.can_delegate,
            "requires_approval": self.requires_approval,
        }


@dataclass
class WorkflowStep:
    """
    Single step in a workflow execution.

    Each step represents a discrete task performed by
    one or more agents in a coordinated workflow.
    """

    name: str
    description: str
    agent_name: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    error_handler: Optional[Callable] = None
    timeout_seconds: int = 300
    retry_count: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Serialize step to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "agent_name": self.agent_name,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
        }


class AGNOTeam:
    """
    AGNO Team - Coordinated multi-agent system

    Teams enable orchestration of multiple specialized agents
    working together to solve complex problems.

    Key Capabilities:
    - Agent coordination and communication
    - Task delegation and distribution
    - Execution mode selection
    - State management across agents
    - Result aggregation and synthesis

    Example:
        >>> # Create specialized agents
        >>> researcher = AgentRole(
        ...     name="Researcher",
        ...     description="Gathers and synthesizes information",
        ...     specializations=["research", "synthesis"],
        ...     tools=["websearch", "database"]
        ... )
        >>> writer = AgentRole(
        ...     name="Writer",
        ...     description="Produces polished content",
        ...     specializations=["writing", "editing"],
        ...     tools=["text_formatter"]
        ... )
        >>>
        >>> # Create team
        >>> team = AGNOTeam(
        ...     name="ContentTeam",
        ...     members=[researcher, writer],
        ...     execution_mode=ExecutionMode.SEQUENTIAL
        ... )
        >>>
        >>> # Execute team task
        >>> result = team.execute("Create a blog post about AI agents")
    """

    def __init__(
        self,
        name: str,
        members: List[AgentRole],
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        model_provider: str = "anthropic",
        model_id: str = "claude-3-5-sonnet-20241022",
        enable_feedback_loops: bool = True,
        max_iterations: int = 5,
        success_criteria: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a multi-agent team.

        Args:
            name: Team identifier
            members: List of agent roles in the team
            execution_mode: How agents coordinate execution
            model_provider: LLM provider for all agents
            model_id: Model to use for all agents
            enable_feedback_loops: Allow agents to refine results
            max_iterations: Maximum refinement iterations
            success_criteria: Conditions for team success

        AGNO Concepts:
        - Sequential: Agents run one after another, output of one
          feeds into the next. Good for linear workflows.
        - Parallel: Multiple agents work simultaneously on independent
          tasks, results combined. Good for parallel processing.
        - Hierarchical: One agent supervises/coordinates others.
          Good for complex decision trees.
        - Dynamic: Runtime selection of execution strategy based
          on task characteristics.
        """
        self.name = name
        self.members = {member.name: member for member in members}
        self.execution_mode = execution_mode
        self.model_provider = model_provider
        self.model_id = model_id
        self.enable_feedback_loops = enable_feedback_loops
        self.max_iterations = max_iterations
        self.success_criteria = success_criteria or {}

        # Team state tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.agent_outputs: Dict[str, Any] = {}
        self.team_context: Dict[str, Any] = {}

        logger.info(
            f"Created team '{name}' with {len(self.members)} members "
            f"using {execution_mode.value} execution"
        )

    def add_member(self, role: AgentRole) -> None:
        """
        Add an agent to the team.

        Args:
            role: Agent role to add
        """
        self.members[role.name] = role
        logger.info(f"Added {role.name} to team {self.name}")

    def remove_member(self, member_name: str) -> None:
        """
        Remove an agent from the team.

        Args:
            member_name: Name of agent to remove
        """
        if member_name in self.members:
            del self.members[member_name]
            logger.info(f"Removed {member_name} from team {self.name}")

    def get_team_composition(self) -> Dict[str, Any]:
        """Get information about team members."""
        return {
            "name": self.name,
            "member_count": len(self.members),
            "members": [m.to_dict() for m in self.members.values()],
            "execution_mode": self.execution_mode.value,
            "model": f"{self.model_provider}/{self.model_id}",
        }

    def execute_sequential(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task with agents running sequentially.

        Output from one agent becomes input to the next.
        This pattern is ideal for linear workflows like:
        - Research → Analysis → Writing → Editing
        - Data Collection → Processing → Visualization
        """
        logger.info(f"Executing task sequentially: {task}")
        results = {"task": task, "steps": []}
        current_input = {"user_request": task, **context}

        for agent_name in self.members.keys():
            step_result = {
                "agent": agent_name,
                "input": current_input.copy(),
                "output": f"Processed by {agent_name}",
                "status": "completed",
            }
            results["steps"].append(step_result)
            current_input = {"previous_result": step_result["output"]}
            self.agent_outputs[agent_name] = step_result["output"]

        return results

    def execute_parallel(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task with agents running in parallel.

        Multiple agents work on the same or different aspects
        simultaneously. Useful for:
        - Parallel research on different topics
        - Simultaneous code review and documentation
        - Multi-perspective analysis
        """
        logger.info(f"Executing task in parallel: {task}")
        results = {"task": task, "parallel_steps": []}

        for agent_name in self.members.keys():
            step_result = {
                "agent": agent_name,
                "input": {"user_request": task, **context},
                "output": f"Parallel work by {agent_name}",
                "status": "completed",
            }
            results["parallel_steps"].append(step_result)
            self.agent_outputs[agent_name] = step_result["output"]

        return results

    def execute_hierarchical(
        self, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute task with hierarchical agent coordination.

        One agent (coordinator/manager) directs others.
        Useful for:
        - Complex problem decomposition
        - Resource allocation
        - Approval workflows
        """
        logger.info(f"Executing task hierarchically: {task}")

        members_list = list(self.members.keys())
        coordinator = members_list[0] if members_list else None
        subordinates = members_list[1:] if len(members_list) > 1 else []

        results = {"task": task, "coordinator": coordinator, "coordination_steps": []}

        # Coordinator decomposes task
        decomposition = {
            "agent": coordinator,
            "action": "decompose_task",
            "subtasks": [f"Subtask for {sub}" for sub in subordinates],
        }
        results["coordination_steps"].append(decomposition)

        # Subordinates execute
        for agent_name in subordinates:
            execution = {
                "agent": agent_name,
                "action": "execute_subtask",
                "status": "completed",
            }
            results["coordination_steps"].append(execution)
            self.agent_outputs[agent_name] = f"Completed by {agent_name}"

        return results

    def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task using the team.

        Args:
            task: Task description for the team
            context: Additional context/variables

        Returns:
            Execution result with outputs from all agents

        AGNO Team Execution:
        1. Prepare context and validate task
        2. Select execution strategy
        3. Coordinate agent execution
        4. Gather and synthesize results
        5. Apply feedback loops if enabled
        """
        if not context:
            context = {}

        logger.info(f"Team execution started: {task}")
        self.team_context = context.copy()

        # Execute based on mode
        if self.execution_mode == ExecutionMode.SEQUENTIAL:
            result = self.execute_sequential(task, context)
        elif self.execution_mode == ExecutionMode.PARALLEL:
            result = self.execute_parallel(task, context)
        elif self.execution_mode == ExecutionMode.HIERARCHICAL:
            result = self.execute_hierarchical(task, context)
        else:
            # Dynamic selection
            result = self.execute_sequential(task, context)

        # Apply feedback loops for refinement
        if self.enable_feedback_loops:
            result = self._apply_feedback_loops(result)

        self.execution_history.append(result)
        return result

    def _apply_feedback_loops(self, initial_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine results through feedback loops.

        AGNO Pattern: Agentic systems often benefit from
        iterative refinement where agents review and improve
        each other's work.
        """
        logger.info("Applying feedback loops for refinement")
        result = initial_result.copy()
        result["refinement_iterations"] = 0

        for iteration in range(self.max_iterations):
            # Check success criteria
            if self._check_success(result):
                logger.info(f"Success criteria met after {iteration} iterations")
                break

            result["refinement_iterations"] = iteration + 1

        return result

    def _check_success(self, result: Dict[str, Any]) -> bool:
        """Check if result meets success criteria."""
        if not self.success_criteria:
            return True

        # In production, would evaluate actual criteria
        # For now, basic success check
        return len(self.agent_outputs) == len(self.members)


class AGNOWorkflow:
    """
    AGNO Workflow - Structured multi-step task execution

    Workflows enable complex, multi-step processes where:
    - Each step is explicitly defined
    - Dependencies between steps are tracked
    - Execution can be monitored and debugged
    - State flows through the pipeline

    Difference from Teams:
    - Teams: Flexible, agent-focused coordination
    - Workflows: Structured, step-focused execution

    Example Use Cases:
    - Data processing pipelines
    - Content creation workflows
    - Code generation and testing pipelines
    - Multi-stage analysis workflows

    Reference: https://docs.agno.com/workflows/overview
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize a workflow.

        Args:
            name: Workflow identifier
            description: What the workflow does
        """
        self.name = name
        self.description = description
        self.steps: List[WorkflowStep] = []
        self.step_results: Dict[str, Any] = {}
        self.execution_log: List[Dict[str, Any]] = []

        logger.info(f"Created workflow: {name}")

    def add_step(self, step: WorkflowStep) -> None:
        """
        Add a step to the workflow.

        Args:
            step: WorkflowStep to add
        """
        self.steps.append(step)
        logger.info(f"Added step '{step.name}' to workflow {self.name}")

    def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow.

        Args:
            initial_input: Initial data for the workflow

        Returns:
            Final output after all steps

        AGNO Workflow Execution:
        1. Validate all steps and dependencies
        2. Execute steps in order (respecting dependencies)
        3. Track inputs/outputs for each step
        4. Handle errors with configured handlers
        5. Return final result
        """
        logger.info(f"Starting workflow execution: {self.name}")
        current_input = initial_input.copy()

        for step in self.steps:
            # Check dependencies are met
            if not self._dependencies_met(step):
                logger.error(f"Dependencies not met for step {step.name}")
                raise ValueError(f"Dependencies not satisfied for {step.name}")

            # Execute step
            try:
                output = self._execute_step(step, current_input)
                self.step_results[step.name] = output
                current_input = output

                self.execution_log.append(
                    {"step": step.name, "status": "success", "output": output}
                )
            except Exception as e:
                logger.error(f"Step {step.name} failed: {e}")

                if step.error_handler:
                    output = step.error_handler(e, current_input)
                    self.step_results[step.name] = output
                else:
                    raise

        return self.step_results

    def _dependencies_met(self, step: WorkflowStep) -> bool:
        """Check if all dependencies for a step are satisfied."""
        for dep in step.dependencies:
            if dep not in self.step_results:
                return False
        return True

    def _execute_step(
        self, step: WorkflowStep, step_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        logger.info(f"Executing step: {step.name}")

        # Simulated step execution
        output = {
            "step": step.name,
            "input": step_input,
            "processed_by": step.agent_name,
            "timestamp": str(__import__("datetime").datetime.now()),
        }

        return output


def main():
    """
    Demonstration of AGNO multi-agent coordination.

    Reference Documentation:
    - https://docs.agno.com/teams/overview
    - https://docs.agno.com/workflows/overview
    - https://www.agno.com/blog/one-agent-is-all-you-need-until-it-isnt
    - https://medium.com/@juanc.olamendy/agno-workflow-building-intelligent-multi-agent-pipelines-for-automated-content-creation-55798e42fc5c
    """
    print("\n=== AGNO Multi-Agent Coordination Demo ===\n")

    # Example 1: Content Creation Team
    print("1. Creating Content Team...")
    researcher = AgentRole(
        name="Researcher",
        description="Gathers and synthesizes information",
        specializations=["research", "information_synthesis"],
        tools=["websearch", "database", "arxiv"],
        can_delegate=False,
    )

    writer = AgentRole(
        name="Writer",
        description="Produces polished, well-structured content",
        specializations=["writing", "editing"],
        tools=["text_formatter", "spell_check"],
        can_delegate=False,
    )

    editor = AgentRole(
        name="Editor",
        description="Reviews and refines content",
        specializations=["editing", "quality_assurance"],
        tools=["text_analyzer", "grammar_check"],
        can_delegate=False,
    )

    team = AGNOTeam(
        name="ContentCreationTeam",
        members=[researcher, writer, editor],
        execution_mode=ExecutionMode.SEQUENTIAL,
        enable_feedback_loops=True,
    )

    # Execute team task
    result = team.execute(
        task="Create a comprehensive blog post about AGNO framework",
        context={"target_audience": "AI engineers", "post_length": "2000 words"},
    )

    print("\nTeam Composition:")
    print(json.dumps(team.get_team_composition(), indent=2))

    print("\nExecution Result:")
    print(json.dumps(result, indent=2))

    # Example 2: Workflow
    print("\n2. Creating Data Processing Workflow...")
    workflow = AGNOWorkflow(
        name="DataProcessingWorkflow", description="End-to-end data processing pipeline"
    )

    workflow.add_step(
        WorkflowStep(
            name="data_collection",
            description="Collect raw data",
            agent_name="DataCollector",
        )
    )

    workflow.add_step(
        WorkflowStep(
            name="data_validation",
            description="Validate and clean data",
            agent_name="DataValidator",
            dependencies=["data_collection"],
        )
    )

    workflow.add_step(
        WorkflowStep(
            name="data_analysis",
            description="Analyze processed data",
            agent_name="DataAnalyst",
            dependencies=["data_validation"],
        )
    )

    workflow_result = workflow.execute({"source": "database", "table": "users"})

    print("\nWorkflow Execution Log:")
    print(json.dumps(workflow.execution_log, indent=2))


if __name__ == "__main__":
    main()
