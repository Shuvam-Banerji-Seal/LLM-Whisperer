"""Automation module for LLM-Whisperer.

Provides automated workflows, task scheduling, and workflow orchestration.
"""

from .core import (
    WorkflowEngine,
    TaskScheduler,
    Workflow,
    Task,
    WorkflowStatus,
    TaskStatus,
)
from .config import WorkflowConfig, TaskConfig, ScheduleConfig

__all__ = [
    "WorkflowEngine",
    "TaskScheduler",
    "Workflow",
    "Task",
    "WorkflowStatus",
    "TaskStatus",
    "WorkflowConfig",
    "TaskConfig",
    "ScheduleConfig",
]
