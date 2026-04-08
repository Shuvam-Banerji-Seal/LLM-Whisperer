# Automation Module

Provides automated workflows, task scheduling, and workflow orchestration for LLM-Whisperer.

## Features

- **Workflow Engine**: Orchestrate complex multi-task workflows
- **Task Scheduling**: Schedule tasks to run at specific times or intervals
- **Dependency Management**: Define and manage task dependencies
- **Parallel Execution**: Execute independent tasks in parallel
- **Error Handling**: Configurable retry and failure handling
- **Async Support**: Full async/await support for task execution

## Components

### WorkflowEngine

Main orchestration engine that manages workflow creation, execution, and scheduling.

```python
from tools.automation import WorkflowEngine, WorkflowConfig, TaskConfig

engine = WorkflowEngine()

# Create workflow configuration
workflow_config = WorkflowConfig(
    name="data_pipeline",
    description="ETL workflow",
    parallel_execution=False,
    on_failure="stop"
)

# Create workflow
workflow = engine.create_workflow(workflow_config)

# Execute
result = await engine.execute_workflow(workflow.id)
```

### Workflow

Container for multiple tasks with dependency management and execution orchestration.

Features:
- Sequential or parallel task execution
- Dependency resolution
- Status tracking
- Error handling and recovery

### Task

Individual unit of work in a workflow.

Features:
- Async/sync handler execution
- Retry mechanism with exponential backoff
- Status tracking
- Timeout support
- Parameter passing

### TaskScheduler

Manages periodic task execution with flexible scheduling options.

Supports:
- One-time execution
- Recurring execution with intervals
- Cron-based scheduling
- Job listing and monitoring

## Configuration

### WorkflowConfig

```python
@dataclass
class WorkflowConfig:
    name: str
    description: str = ""
    tasks: List[TaskConfig] = field(default_factory=list)
    timeout_seconds: int = 86400
    parallel_execution: bool = False
    on_failure: str = "stop"  # stop, continue, retry
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### TaskConfig

```python
@dataclass
class TaskConfig:
    name: str
    description: str = ""
    task_type: str = "python"
    handler: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ScheduleConfig

```python
@dataclass
class ScheduleConfig:
    schedule_type: ScheduleType = ScheduleType.ONCE
    interval_seconds: Optional[int] = None
    cron_expression: Optional[str] = None
    max_retries: int = 3
    retry_delay_seconds: int = 5
    timeout_seconds: int = 3600
```

## Examples

### Basic Workflow

```python
import asyncio
from tools.automation import WorkflowEngine, WorkflowConfig, TaskConfig

async def main():
    engine = WorkflowEngine()
    
    # Create tasks
    task1_config = TaskConfig(
        name="task1",
        description="First task",
        task_type="sleep",
        parameters={"duration": 1}
    )
    
    task2_config = TaskConfig(
        name="task2",
        description="Second task",
        dependencies=["task1"],
        task_type="sleep",
        parameters={"duration": 1}
    )
    
    # Create workflow
    workflow_config = WorkflowConfig(
        name="example_workflow",
        tasks=[task1_config, task2_config]
    )
    
    workflow = engine.create_workflow(workflow_config)
    result = await engine.execute_workflow(workflow.id)
    print(result)

asyncio.run(main())
```

### Custom Task Handlers

```python
async def custom_handler(param1: str, param2: int):
    return {"processed": True, "param1": param1, "param2": param2}

# Register handler
engine.register_task_handler(workflow.id, "task1", custom_handler)
```

### Task Scheduling

```python
from tools.automation import ScheduleConfig, ScheduleType

schedule_config = ScheduleConfig(
    schedule_type=ScheduleType.INTERVAL,
    interval_seconds=3600
)

job_id = engine.schedule_workflow(workflow.id, schedule_config)

# List scheduled jobs
jobs = engine.scheduler.list_jobs()
```

## Status Values

### TaskStatus
- `pending`: Waiting to execute
- `running`: Currently executing
- `completed`: Successfully completed
- `failed`: Execution failed
- `skipped`: Skipped execution
- `retrying`: Retrying after failure

### WorkflowStatus
- `idle`: Not started
- `running`: Currently executing
- `paused`: Execution paused
- `completed`: Successfully completed
- `failed`: Execution failed

## Error Handling

Configure error handling behavior with the `on_failure` option:

- `stop`: Stop workflow on first failure
- `continue`: Continue to next task on failure
- `retry`: Retry failed task up to max_retries

Each task can be configured with:
- Maximum retry attempts
- Retry delay
- Execution timeout

## License

MIT
