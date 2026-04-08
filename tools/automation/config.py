"""Configuration dataclasses for automation module."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class ScheduleType(Enum):
    """Types of schedule for tasks."""

    ONCE = "once"
    RECURRING = "recurring"
    CRON = "cron"
    INTERVAL = "interval"


@dataclass
class ScheduleConfig:
    """Configuration for task scheduling.

    Attributes:
        schedule_type: Type of schedule (once, recurring, cron, interval)
        interval_seconds: Interval in seconds for recurring tasks
        cron_expression: Cron expression for cron-based schedules
        max_retries: Maximum number of retries on failure
        retry_delay_seconds: Delay between retries in seconds
        timeout_seconds: Task execution timeout in seconds
    """

    schedule_type: ScheduleType = ScheduleType.ONCE
    interval_seconds: Optional[int] = None
    cron_expression: Optional[str] = None
    max_retries: int = 3
    retry_delay_seconds: int = 5
    timeout_seconds: int = 3600


@dataclass
class TaskConfig:
    """Configuration for individual tasks.

    Attributes:
        name: Task name
        description: Task description
        task_type: Type of task (python, shell, http, etc.)
        handler: Handler function or command
        dependencies: List of task names this task depends on
        parameters: Task-specific parameters
        schedule: Schedule configuration
        enabled: Whether task is enabled
        metadata: Additional metadata
    """

    name: str
    description: str = ""
    task_type: str = "python"
    handler: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    """Configuration for workflows.

    Attributes:
        name: Workflow name
        description: Workflow description
        tasks: List of task configurations
        timeout_seconds: Total workflow timeout in seconds
        parallel_execution: Whether tasks can execute in parallel
        on_failure: Action on task failure (stop, continue, retry)
        metadata: Additional metadata
    """

    name: str
    description: str = ""
    tasks: List[TaskConfig] = field(default_factory=list)
    timeout_seconds: int = 86400
    parallel_execution: bool = False
    on_failure: str = "stop"  # stop, continue, retry
    metadata: Dict[str, Any] = field(default_factory=dict)
