"""Core automation framework for workflows and scheduling."""

import logging
import time
import threading
import asyncio
from typing import Dict, Any, Optional, List, Callable, Coroutine
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid

from .config import WorkflowConfig, TaskConfig, ScheduleConfig, ScheduleType

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class WorkflowStatus(Enum):
    """Workflow execution status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    """Represents a single task in a workflow.

    Attributes:
        config: Task configuration
        status: Current task status
        result: Task execution result
        started_at: Task start timestamp
        completed_at: Task completion timestamp
    """

    def __init__(self, config: TaskConfig):
        """Initialize task.

        Args:
            config: Task configuration
        """
        self.config = config
        self.id = str(uuid.uuid4())
        self.status = TaskStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.retry_count = 0
        self.handler: Optional[Callable] = None

    def set_handler(self, handler: Callable):
        """Set task handler function.

        Args:
            handler: Callable to execute task
        """
        self.handler = handler
        logger.debug(f"Task {self.config.name}: handler set")

    async def execute(self) -> Dict[str, Any]:
        """Execute task.

        Returns:
            Execution result dictionary
        """
        if not self.config.enabled:
            logger.info(f"Task {self.config.name}: skipped (disabled)")
            self.status = TaskStatus.SKIPPED
            return {"status": "skipped"}

        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

        try:
            logger.info(f"Task {self.config.name}: starting execution")

            if self.handler:
                if asyncio.iscoroutinefunction(self.handler):
                    self.result = await self.handler(**self.config.parameters)
                else:
                    self.result = self.handler(**self.config.parameters)
            else:
                self.result = self._execute_default()

            self.status = TaskStatus.COMPLETED
            self.completed_at = datetime.now()
            logger.info(
                f"Task {self.config.name}: completed in "
                f"{(self.completed_at - self.started_at).total_seconds():.2f}s"
            )
            return {"status": "completed", "result": self.result}

        except Exception as e:
            logger.error(f"Task {self.config.name}: execution failed - {str(e)}")
            self.error = str(e)
            self.completed_at = datetime.now()

            if self.retry_count < self.config.schedule.max_retries:
                self.retry_count += 1
                self.status = TaskStatus.RETRYING
                return {
                    "status": "retrying",
                    "retry_count": self.retry_count,
                    "error": str(e),
                }
            else:
                self.status = TaskStatus.FAILED
                return {"status": "failed", "error": str(e)}

    def _execute_default(self) -> Dict[str, Any]:
        """Execute task with default behavior.

        Returns:
            Execution result
        """
        logger.debug(
            f"Task {self.config.name}: executing with type {self.config.task_type}"
        )

        if self.config.task_type == "sleep":
            duration = self.config.parameters.get("duration", 1)
            time.sleep(duration)
            return {"slept_for": duration}

        return {"status": "executed", "handler": "default"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.config.name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }


class Workflow:
    """Represents a workflow composed of multiple tasks.

    Attributes:
        config: Workflow configuration
        tasks: Dictionary of task ID to Task
        status: Current workflow status
    """

    def __init__(self, config: WorkflowConfig):
        """Initialize workflow.

        Args:
            config: Workflow configuration
        """
        self.config = config
        self.id = str(uuid.uuid4())
        self.tasks: Dict[str, Task] = {}
        self.status = WorkflowStatus.IDLE
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Dict[str, Any] = {}

        self._initialize_tasks()

    def _initialize_tasks(self):
        """Initialize tasks from configuration."""
        for task_config in self.config.tasks:
            task = Task(task_config)
            self.tasks[task.id] = task
            logger.debug(f"Workflow {self.config.name}: added task {task_config.name}")

    def add_task(self, task: Task):
        """Add task to workflow.

        Args:
            task: Task to add
        """
        self.tasks[task.id] = task
        self.config.tasks.append(task.config)
        logger.info(f"Workflow {self.config.name}: added task {task.config.name}")

    def get_task(self, name: str) -> Optional[Task]:
        """Get task by name.

        Args:
            name: Task name

        Returns:
            Task or None
        """
        for task in self.tasks.values():
            if task.config.name == name:
                return task
        return None

    async def execute(self) -> Dict[str, Any]:
        """Execute workflow.

        Returns:
            Workflow execution result
        """
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.now()
        logger.info(f"Workflow {self.config.name}: starting execution")

        try:
            if self.config.parallel_execution:
                await self._execute_parallel()
            else:
                await self._execute_sequential()

            self.status = WorkflowStatus.COMPLETED
            self.completed_at = datetime.now()

            logger.info(
                f"Workflow {self.config.name}: completed in "
                f"{(self.completed_at - self.started_at).total_seconds():.2f}s"
            )

        except Exception as e:
            logger.error(f"Workflow {self.config.name}: execution failed - {str(e)}")
            self.status = WorkflowStatus.FAILED
            self.completed_at = datetime.now()

        return self.to_dict()

    async def _execute_sequential(self):
        """Execute tasks sequentially."""
        logger.debug(f"Workflow {self.config.name}: executing tasks sequentially")

        for task in self.tasks.values():
            # Check dependencies
            if not self._check_dependencies(task):
                task.status = TaskStatus.SKIPPED
                logger.warning(
                    f"Task {task.config.name}: skipped (dependencies not met)"
                )
                continue

            result = await task.execute()

            if result["status"] == "failed" and self.config.on_failure == "stop":
                self.status = WorkflowStatus.FAILED
                logger.error(
                    f"Workflow {self.config.name}: stopping due to task failure"
                )
                break

            self.result[task.config.name] = result

    async def _execute_parallel(self):
        """Execute tasks in parallel where possible."""
        logger.debug(f"Workflow {self.config.name}: executing tasks in parallel")

        # Create task groups by dependency
        executing = set()

        for task in self.tasks.values():
            if not self._check_dependencies(task):
                task.status = TaskStatus.SKIPPED
                continue

            executing.add(asyncio.create_task(task.execute()))

        results = await asyncio.gather(*executing, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Workflow {self.config.name}: parallel task failed - {result}"
                )
                if self.config.on_failure == "stop":
                    self.status = WorkflowStatus.FAILED
                    break

    def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are met.

        Args:
            task: Task to check

        Returns:
            True if dependencies are met
        """
        for dep_name in task.config.dependencies:
            dep_task = self.get_task(dep_name)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.config.name,
            "status": self.status.value,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "results": self.result,
        }


class WorkflowEngine:
    """Main workflow execution engine.

    Manages workflow creation, scheduling, and execution.
    """

    def __init__(self):
        """Initialize workflow engine."""
        self.workflows: Dict[str, Workflow] = {}
        self.scheduler = TaskScheduler()
        logger.info("WorkflowEngine initialized")

    def create_workflow(self, config: WorkflowConfig) -> Workflow:
        """Create a new workflow.

        Args:
            config: Workflow configuration

        Returns:
            Created workflow
        """
        workflow = Workflow(config)
        self.workflows[workflow.id] = workflow
        logger.info(f"Workflow created: {config.name}")
        return workflow

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow or None
        """
        return self.workflows.get(workflow_id)

    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute workflow.

        Args:
            workflow_id: Workflow ID to execute

        Returns:
            Execution result
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return {"error": f"Workflow not found: {workflow_id}"}

        return await workflow.execute()

    def register_task_handler(
        self, workflow_id: str, task_name: str, handler: Callable
    ):
        """Register handler for a task.

        Args:
            workflow_id: Workflow ID
            task_name: Task name
            handler: Handler function
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return

        task = workflow.get_task(task_name)
        if not task:
            logger.error(f"Task not found: {task_name}")
            return

        task.set_handler(handler)
        logger.info(f"Handler registered for task: {task_name}")

    def schedule_workflow(
        self, workflow_id: str, schedule_config: ScheduleConfig
    ) -> str:
        """Schedule workflow for execution.

        Args:
            workflow_id: Workflow ID
            schedule_config: Schedule configuration

        Returns:
            Job ID
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return ""

        job_id = self.scheduler.schedule_job(
            job_name=workflow.config.name,
            job_func=lambda: asyncio.run(self.execute_workflow(workflow_id)),
            schedule_config=schedule_config,
        )

        logger.info(f"Workflow scheduled: {workflow.config.name} (job_id={job_id})")
        return job_id

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows.

        Returns:
            List of workflow information
        """
        return [
            {"id": w.id, "name": w.config.name, "status": w.status.value}
            for w in self.workflows.values()
        ]


class TaskScheduler:
    """Scheduler for periodic task execution."""

    def __init__(self):
        """Initialize task scheduler."""
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.running = False
        logger.info("TaskScheduler initialized")

    def schedule_job(
        self, job_name: str, job_func: Callable, schedule_config: ScheduleConfig
    ) -> str:
        """Schedule a job.

        Args:
            job_name: Job name
            job_func: Function to execute
            schedule_config: Schedule configuration

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "name": job_name,
            "func": job_func,
            "schedule": schedule_config,
            "last_run": None,
            "next_run": datetime.now(),
            "run_count": 0,
        }

        logger.info(f"Job scheduled: {job_name} (id={job_id})")
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job data or None
        """
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs.

        Returns:
            List of jobs
        """
        return [
            {
                "id": job_id,
                "name": job["name"],
                "next_run": job["next_run"].isoformat(),
                "run_count": job["run_count"],
            }
            for job_id, job in self.jobs.items()
        ]

    def start(self):
        """Start scheduler."""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True
        logger.info("TaskScheduler started")

        scheduler_thread = threading.Thread(target=self._run, daemon=True)
        scheduler_thread.start()

    def stop(self):
        """Stop scheduler."""
        self.running = False
        logger.info("TaskScheduler stopped")

    def _run(self):
        """Scheduler main loop."""
        while self.running:
            now = datetime.now()

            for job_id, job in self.jobs.items():
                if job["next_run"] <= now:
                    logger.info(f"Executing job: {job['name']}")
                    try:
                        job["func"]()
                        job["last_run"] = now
                        job["run_count"] += 1
                        job["next_run"] = self._calculate_next_run(now, job["schedule"])
                    except Exception as e:
                        logger.error(f"Job {job['name']} failed: {str(e)}")

            time.sleep(1)

    def _calculate_next_run(
        self, current_time: datetime, schedule: ScheduleConfig
    ) -> datetime:
        """Calculate next run time for a job.

        Args:
            current_time: Current time
            schedule: Schedule configuration

        Returns:
            Next run time
        """
        if schedule.schedule_type == ScheduleType.ONCE:
            return current_time + timedelta(days=365 * 100)

        elif schedule.schedule_type == ScheduleType.INTERVAL:
            if schedule.interval_seconds:
                return current_time + timedelta(seconds=schedule.interval_seconds)

        return current_time + timedelta(hours=1)
