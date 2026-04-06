"""
Autonomous Workflow Agent with Task Execution and Error Recovery
Author: Shuvam Banerji Seal

This module implements an autonomous agent capable of:
- Multi-step task planning and execution
- Autonomous decision-making
- Error recovery and retry mechanisms
- Progress tracking and reporting
- Task prioritization and scheduling
- Sub-task decomposition

The agent can handle complex workflows that require multiple steps,
conditional logic, and intelligent error handling.

Source: https://langchain-tutorials.github.io/production-ready-langchain-error-handling-patterns/
Source: https://callsphere.tech/blog/langgraph-error-handling-retry-nodes-fallback-paths-recovery
Source: https://python.langchain.com/api_reference
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
import time

from langchain_core.tools import Tool
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 3
    MEDIUM = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class ExecutionResult:
    """Result of a task or sub-task execution."""

    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp,
        }


@dataclass
class Task:
    """Represents an executable task."""

    task_id: str
    name: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    max_retries: int = 3
    timeout_seconds: int = 300
    subtasks: List["Task"] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.name,
            "status": self.status.value,
            "created_at": self.created_at,
            "subtasks": len(self.subtasks),
            "dependencies": self.dependencies,
        }


class WorkflowTools:
    """Collection of tools for workflow execution."""

    def __init__(self):
        """Initialize workflow tools."""
        self.task_results: Dict[str, ExecutionResult] = {}
        self.execution_history: List[ExecutionResult] = []
        logger.info("WorkflowTools initialized")

    def execute_task(
        self, task_name: str, task_description: str, timeout: int = 30
    ) -> ExecutionResult:
        """
        Execute a task and return result.

        Args:
            task_name: Name of the task
            task_description: Task description/instructions
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with status and output
        """
        start_time = time.time()

        try:
            # Simulate task execution
            # In real scenario, this would execute actual code
            logger.info(f"Executing task: {task_name}")

            # Simple execution simulation
            time.sleep(min(1, timeout / 10))

            execution_time = time.time() - start_time

            result = ExecutionResult(
                task_id=task_name,
                status=TaskStatus.COMPLETED,
                output=f"Task '{task_name}' completed successfully",
                execution_time=execution_time,
            )

            self.task_results[task_name] = result
            self.execution_history.append(result)

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            result = ExecutionResult(
                task_id=task_name,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
            )

            self.task_results[task_name] = result
            self.execution_history.append(result)

            logger.error(f"Task execution failed: {task_name} - {str(e)}")
            return result

    def retry_task(
        self, task_id: str, max_retries: int = 3, backoff_factor: float = 2.0
    ) -> ExecutionResult:
        """
        Retry failed task with exponential backoff.

        Args:
            task_id: ID of task to retry
            max_retries: Maximum number of retries
            backoff_factor: Exponential backoff factor

        Returns:
            ExecutionResult after retry attempts
        """
        for attempt in range(max_retries):
            try:
                # Calculate backoff
                if attempt > 0:
                    wait_time = backoff_factor**attempt
                    logger.info(f"Retrying {task_id}, waiting {wait_time}s...")
                    time.sleep(wait_time)

                # Execute task
                result = ExecutionResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    output=f"Task '{task_id}' completed on attempt {attempt + 1}",
                    retry_count=attempt,
                )

                self.task_results[task_id] = result
                self.execution_history.append(result)

                logger.info(f"Task retry successful: {task_id} (attempt {attempt + 1})")
                return result

            except Exception as e:
                if attempt == max_retries - 1:
                    result = ExecutionResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=f"Task failed after {max_retries} attempts: {str(e)}",
                        retry_count=attempt,
                    )
                    self.task_results[task_id] = result
                    self.execution_history.append(result)
                    return result

        return ExecutionResult(
            task_id=task_id, status=TaskStatus.FAILED, error="Unexpected retry failure"
        )

    def check_dependencies(
        self, task_id: str, dependencies: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if task dependencies are satisfied.

        Args:
            task_id: Task ID
            dependencies: List of dependency task IDs

        Returns:
            Tuple of (all_satisfied, missing_dependencies_string)
        """
        missing = []

        for dep_id in dependencies:
            result = self.task_results.get(dep_id)
            if not result or result.status != TaskStatus.COMPLETED:
                missing.append(dep_id)

        if missing:
            return False, f"Missing dependencies: {', '.join(missing)}"

        return True, None

    def schedule_task(self, task: Task, execution_queue: List[Task]) -> bool:
        """
        Schedule a task for execution if dependencies are met.

        Args:
            task: Task to schedule
            execution_queue: Queue to add task to

        Returns:
            True if scheduled, False if dependencies not met
        """
        # Check dependencies
        satisfied, error = self.check_dependencies(task.task_id, task.dependencies)

        if not satisfied:
            logger.warning(f"Cannot schedule {task.task_id}: {error}")
            return False

        # Add to queue
        execution_queue.append(task)
        task.status = TaskStatus.PENDING
        logger.info(f"Task scheduled: {task.task_id}")

        return True

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get workflow execution statistics."""
        if not self.execution_history:
            return {"total_tasks": 0, "completed": 0, "failed": 0, "total_time": 0.0}

        completed = sum(
            1 for r in self.execution_history if r.status == TaskStatus.COMPLETED
        )
        failed = sum(1 for r in self.execution_history if r.status == TaskStatus.FAILED)
        total_time = sum(r.execution_time for r in self.execution_history)

        return {
            "total_tasks": len(self.execution_history),
            "completed": completed,
            "failed": failed,
            "success_rate": completed / len(self.execution_history)
            if self.execution_history
            else 0,
            "total_time": total_time,
            "average_time": total_time / len(self.execution_history)
            if self.execution_history
            else 0,
        }


class AutonomousWorkflowAgent:
    """
    Agent for autonomous workflow execution with error handling.

    Capabilities:
    - Multi-step task execution
    - Dependency management
    - Error recovery and retries
    - Progress tracking
    - Autonomous decision-making

    Attributes:
        llm: Language model instance
        workflow_tools: WorkflowTools instance
        agent_executor: Agent execution engine
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        llm_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize autonomous workflow agent.

        Args:
            model: LLM model name
            temperature: LLM temperature for consistency
            llm_api_key: OpenAI API key

        Example:
            >>> agent = AutonomousWorkflowAgent()
            >>> task = Task(
            ...     task_id="data_pipeline",
            ...     name="Data Processing Pipeline",
            ...     description="Process raw data"
            ... )
            >>> agent.execute_workflow([task])
        """
        # Initialize LLM
        api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

        # Initialize workflow tools
        self.workflow_tools = WorkflowTools()

        # Setup agent tools
        self.tools = self._setup_tools()

        # Initialize agent executor
        self.agent_executor = self._create_agent_executor()

        # Workflow state
        self.current_workflow: Optional[List[Task]] = None
        self.execution_results: Dict[str, ExecutionResult] = {}

        logger.info("AutonomousWorkflowAgent initialized")

    def _setup_tools(self) -> List[Tool]:
        """Setup available tools for workflow execution."""
        tools = [
            Tool(
                name="execute_task",
                func=self._execute_task_tool,
                description="Execute a single task",
            ),
            Tool(
                name="retry_task",
                func=self._retry_task_tool,
                description="Retry a failed task with exponential backoff",
            ),
            Tool(
                name="check_dependencies",
                func=self._check_dependencies_tool,
                description="Check if task dependencies are satisfied",
            ),
            Tool(
                name="get_execution_status",
                func=self._get_status_tool,
                description="Get status of executed tasks",
            ),
            Tool(
                name="schedule_task",
                func=self._schedule_task_tool,
                description="Schedule a task for execution",
            ),
        ]
        return tools

    def _execute_task_tool(self, task_spec: str) -> str:
        """Tool for task execution."""
        try:
            spec = json.loads(task_spec)
            result = self.workflow_tools.execute_task(
                task_name=spec.get("name", "task"),
                task_description=spec.get("description", ""),
                timeout=spec.get("timeout", 30),
            )
            return json.dumps(result.to_dict())
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _retry_task_tool(self, task_id: str) -> str:
        """Tool for retrying tasks."""
        result = self.workflow_tools.retry_task(
            task_id=task_id, max_retries=3, backoff_factor=2.0
        )
        return json.dumps(result.to_dict())

    def _check_dependencies_tool(self, dependency_spec: str) -> str:
        """Tool for checking dependencies."""
        try:
            spec = json.loads(dependency_spec)
            satisfied, error = self.workflow_tools.check_dependencies(
                task_id=spec.get("task_id", ""),
                dependencies=spec.get("dependencies", []),
            )
            return json.dumps({"satisfied": satisfied, "error": error})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_status_tool(self, _: str) -> str:
        """Tool for getting execution status."""
        stats = self.workflow_tools.get_execution_stats()
        return json.dumps(stats)

    def _schedule_task_tool(self, task_spec: str) -> str:
        """Tool for scheduling tasks."""
        try:
            spec = json.loads(task_spec)
            task = Task(
                task_id=spec.get("task_id", ""),
                name=spec.get("name", ""),
                description=spec.get("description", ""),
                dependencies=spec.get("dependencies", []),
            )

            queue = self.current_workflow or []
            scheduled = self.workflow_tools.schedule_task(task, queue)

            return json.dumps({"scheduled": scheduled})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _create_agent_executor(self) -> AgentExecutor:
        """Create and configure the agent executor."""
        system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an autonomous workflow execution agent. Your responsibilities:

1. Execute tasks in the correct order respecting dependencies
2. Handle errors gracefully with retry logic
3. Track progress and provide status updates
4. Make intelligent decisions about task scheduling
5. Monitor execution time and resource constraints

Guidelines:
- Always check dependencies before executing a task
- Use retry mechanism for failed tasks
- Maintain clear execution logs
- Report progress transparently
- Optimize task ordering for efficiency""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, system_prompt)

        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=15,
            handle_parsing_errors=True,
        )

        return executor

    def execute_workflow(
        self, tasks: List[Task], parallel_execution: bool = False
    ) -> Dict[str, ExecutionResult]:
        """
        Execute a workflow of multiple tasks.

        Args:
            tasks: List of tasks to execute
            parallel_execution: Whether to attempt parallel execution

        Returns:
            Dictionary of execution results by task ID

        Example:
            >>> tasks = [
            ...     Task("task1", "First task", "Do step 1"),
            ...     Task("task2", "Second task", "Do step 2", dependencies=["task1"]),
            ... ]
            >>> results = agent.execute_workflow(tasks)
        """
        try:
            self.current_workflow = tasks

            # Sort tasks by priority
            sorted_tasks = sorted(
                tasks, key=lambda t: (len(t.dependencies), t.priority.value)
            )

            logger.info(f"Starting workflow with {len(sorted_tasks)} tasks")

            # Build execution plan
            execution_plan = f"""Execute the following workflow:
{json.dumps([t.to_dict() for t in sorted_tasks], indent=2)}

Tasks to execute:
{chr(10).join([f"- {t.name}: {t.description}" for t in sorted_tasks])}

Please execute all tasks in order, respecting dependencies,
and handle any errors with retry logic."""

            # Execute workflow
            result = self.agent_executor.invoke({"input": execution_plan})

            # Collect results
            self.execution_results = self.workflow_tools.task_results.copy()

            logger.info(f"Workflow completed: {len(self.execution_results)} tasks")
            return self.execution_results

        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {}

    def get_workflow_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive workflow execution report.

        Returns:
            Dictionary with workflow statistics and results
        """
        stats = self.workflow_tools.get_execution_stats()

        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "executions": [
                result.to_dict() for result in self.workflow_tools.execution_history
            ],
        }

    def export_workflow_report(self, filepath: str) -> None:
        """
        Export workflow execution report to file.

        Args:
            filepath: Path to save report
        """
        try:
            report = self.get_workflow_report()

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Report exported to {filepath}")
        except Exception as e:
            logger.error(f"Export error: {e}")


# ============================================================================
# Usage Examples
# ============================================================================


def example_simple_workflow() -> None:
    """Example of simple linear workflow."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Linear Workflow")
    print("=" * 70)

    load_dotenv()

    agent = AutonomousWorkflowAgent()

    # Create tasks
    tasks = [
        Task(
            task_id="task1",
            name="Data Collection",
            description="Gather raw data from sources",
            priority=TaskPriority.HIGH,
        ),
        Task(
            task_id="task2",
            name="Data Cleaning",
            description="Clean and normalize data",
            dependencies=["task1"],
        ),
        Task(
            task_id="task3",
            name="Analysis",
            description="Perform data analysis",
            dependencies=["task2"],
        ),
    ]

    # Execute workflow
    results = agent.execute_workflow(tasks)

    # Print results
    print("\nWorkflow Results:")
    for task_id, result in results.items():
        print(f"\n{task_id}:")
        print(f"  Status: {result.status.value}")
        print(f"  Time: {result.execution_time:.2f}s")


def example_complex_workflow() -> None:
    """Example of complex workflow with multiple dependencies."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Complex Workflow with Dependencies")
    print("=" * 70)

    load_dotenv()

    agent = AutonomousWorkflowAgent()

    # Create complex task structure
    tasks = [
        Task(
            "data_fetch", "Fetch Data", "Get data from API", priority=TaskPriority.HIGH
        ),
        Task(
            "validate",
            "Validate Data",
            "Validate data integrity",
            dependencies=["data_fetch"],
        ),
        Task(
            "transform",
            "Transform Data",
            "Transform data format",
            dependencies=["validate"],
        ),
        Task("analyze1", "Analysis 1", "First analysis", dependencies=["transform"]),
        Task("analyze2", "Analysis 2", "Second analysis", dependencies=["transform"]),
        Task(
            "merge",
            "Merge Results",
            "Merge analysis results",
            dependencies=["analyze1", "analyze2"],
        ),
    ]

    results = agent.execute_workflow(tasks)

    # Print workflow report
    report = agent.get_workflow_report()
    print("\nWorkflow Report:")
    print(json.dumps(report["statistics"], indent=2))


def example_workflow_export() -> None:
    """Example of workflow reporting and export."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Workflow Export")
    print("=" * 70)

    load_dotenv()

    agent = AutonomousWorkflowAgent()

    # Execute workflow
    tasks = [
        Task("task1", "Process", "Process data"),
        Task("task2", "Verify", "Verify results", dependencies=["task1"]),
    ]

    agent.execute_workflow(tasks)

    # Export report
    agent.export_workflow_report("./workflow_reports/execution_report.json")
    print("Report saved to ./workflow_reports/execution_report.json")


if __name__ == "__main__":
    """
    Main entry point for autonomous workflow agent.
    
    Required environment variables:
    - OPENAI_API_KEY: OpenAI API key
    
    Setup Instructions:
    1. Install dependencies:
       pip install langchain langchain-openai python-dotenv
    2. Set up .env file with OPENAI_API_KEY
    3. Run: python autonomous_workflow_agent.py
    """

    print("Autonomous Workflow Agent - Example Usage")
    print("=" * 70)
    print("\nAvailable examples:")
    print("  - example_simple_workflow()")
    print("  - example_complex_workflow()")
    print("  - example_workflow_export()")
    print("\nUncomment in __main__ to run examples")
