"""
Planning Patterns for Agents

Advanced planning patterns for agent systems:
- Hierarchical planning
- Replanning strategies
- Plan execution and monitoring
- Plan repair

Author: Shuvam Banerji
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
import time
import logging
import heapq
from enum import Enum
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """Plan execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"
    PAUSED = "paused"


class PlanStepStatus(Enum):
    """Individual step status."""
    PENDING = "pending"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a plan."""
    step_id: str
    description: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration_sec: float = 1.0
    actual_duration_sec: float = 0.0

    def __lt__(self, other: "PlanStep") -> bool:
        return self.step_id < other.step_id


@dataclass
class Plan:
    """Complete plan with steps."""
    plan_id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    current_step_index: int = 0

    def add_step(self, step: PlanStep) -> None:
        """Add step to plan."""
        self.steps.append(step)

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute."""
        ready = []
        completed_ids = {
            s.step_id for s in self.steps
            if s.status == PlanStepStatus.COMPLETED
        }

        for step in self.steps:
            if step.status != PlanStepStatus.PENDING:
                continue

            deps_satisfied = all(dep in completed_ids for dep in step.dependencies)
            if deps_satisfied:
                ready.append(step)

        return ready

    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(s.status == PlanStepStatus.COMPLETED for s in self.steps)

    def get_execution_order(self) -> List[str]:
        """Get steps in topological order."""
        in_degree = {s.step_id: 0 for s in self.steps}
        graph = {s.step_id: [] for s in self.steps}

        for step in self.steps:
            for dep in step.dependencies:
                if dep in graph:
                    graph[dep].append(step.step_id)
                    in_degree[step.step_id] += 1

        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result


class Planner(ABC):
    """Abstract base class for planners."""

    @abstractmethod
    def create_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Create a plan to achieve goal."""
        pass


class HierarchicalPlanner(Planner):
    """Hierarchical planner that decomposes goals into subgoals."""

    def __init__(self):
        self.decomposition_rules: Dict[str, Callable] = {}

    def register_decomposition(
        self,
        action_pattern: str,
        decomposition_fn: Callable
    ) -> None:
        """Register decomposition rule."""
        self.decomposition_rules[action_pattern] = decomposition_fn

    def create_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Create hierarchical plan."""
        plan = Plan(plan_id=str(uuid.uuid4())[:8], goal=goal)

        steps = self._decompose_goal(goal, context)

        for step_data in steps:
            step = PlanStep(
                step_id=step_data["id"],
                description=step_data["description"],
                action=step_data["action"],
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", [])
            )
            plan.add_step(step)

        return plan

    def _decompose_goal(self, goal: str, context: Dict) -> List[Dict]:
        """Decompose goal into steps."""
        goal_lower = goal.lower()

        if "research" in goal_lower or "find information" in goal_lower:
            return self._decompose_research_goal(goal, context)
        elif "build" in goal_lower or "create" in goal_lower:
            return self._decompose_build_goal(goal, context)
        elif "analyze" in goal_lower:
            return self._decompose_analysis_goal(goal, context)
        else:
            return [{"id": "step_1", "description": goal, "action": "execute", "parameters": {}}]

    def _decompose_research_goal(self, goal: str, context: Dict) -> List[Dict]:
        """Decompose research goals."""
        return [
            {"id": "step_1", "description": "Gather initial information", "action": "search", "parameters": {"query": goal}},
            {"id": "step_2", "description": "Validate findings", "action": "verify", "dependencies": ["step_1"]},
            {"id": "step_3", "description": "Compile results", "action": "compile", "dependencies": ["step_2"]}
        ]

    def _decompose_build_goal(self, goal: str, context: Dict) -> List[Dict]:
        """Decompose build goals."""
        return [
            {"id": "step_1", "description": "Plan architecture", "action": "plan_architecture"},
            {"id": "step_2", "description": "Create components", "action": "create_components", "dependencies": ["step_1"]},
            {"id": "step_3", "description": "Integrate parts", "action": "integrate", "dependencies": ["step_2"]},
            {"id": "step_4", "description": "Test and validate", "action": "test", "dependencies": ["step_3"]}
        ]

    def _decompose_analysis_goal(self, goal: str, context: Dict) -> List[Dict]:
        """Decompose analysis goals."""
        return [
            {"id": "step_1", "description": "Collect data", "action": "collect_data"},
            {"id": "step_2", "description": "Process data", "action": "process_data", "dependencies": ["step_1"]},
            {"id": "step_3", "description": "Generate insights", "action": "generate_insights", "dependencies": ["step_2"]}
        ]


class SimplePlanner(Planner):
    """Simple linear planner."""

    def create_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Create simple plan with single step."""
        plan = Plan(plan_id=str(uuid.uuid4())[:8], goal=goal)

        step = PlanStep(
            step_id="main_step",
            description=f"Achieve: {goal}",
            action="execute",
            parameters={"goal": goal}
        )
        plan.add_step(step)

        return plan


class Replanner(ABC):
    """Abstract base class for replanning strategies."""

    @abstractmethod
    def should_replan(
        self,
        plan: Plan,
        step: PlanStep,
        error: Exception
    ) -> bool:
        """Determine if replanning is needed."""
        pass

    @abstractmethod
    def create_replacement_plan(
        self,
        original_plan: Plan,
        failed_step: PlanStep,
        context: Dict[str, Any]
    ) -> Plan:
        """Create replacement plan."""
        pass


class ErrorThresholdReplanner(Replanner):
    """Replan when error threshold is exceeded."""

    def __init__(self, max_retries_per_step: int = 3):
        self.max_retries_per_step = max_retries_per_step

    def should_replan(
        self,
        plan: Plan,
        step: PlanStep,
        error: Exception
    ) -> bool:
        """Check if should replan."""
        return step.retry_count >= self.max_retries_per_step

    def create_replacement_plan(
        self,
        original_plan: Plan,
        failed_step: PlanStep,
        context: Dict[str, Any]
    ) -> Plan:
        """Create replacement plan with alternative approach."""
        new_plan = Plan(
            plan_id=str(uuid.uuid4())[:8],
            goal=original_plan.goal,
            metadata={"original_plan_id": original_plan.plan_id, "replanned_from": failed_step.step_id}
        )

        for s in original_plan.steps:
            if s.step_id == failed_step.step_id:
                new_step = PlanStep(
                    step_id=f"{s.step_id}_alt",
                    description=f"{s.description} (alternative)",
                    action=s.action,
                    parameters={**s.parameters, "alternative": True},
                    dependencies=s.dependencies
                )
                new_plan.add_step(new_step)
            elif s.status == PlanStepStatus.COMPLETED:
                new_plan.add_step(PlanStep(
                    step_id=s.step_id,
                    description=s.description,
                    action=s.action,
                    status=PlanStepStatus.COMPLETED,
                    result=s.result
                ))

        return new_plan


class ConditionReplanner(Replanner):
    """Replan based on condition evaluation."""

    def __init__(self, conditions: Optional[Dict[str, Callable]] = None):
        self.conditions = conditions or {}

    def should_replan(
        self,
        plan: Plan,
        step: PlanStep,
        error: Exception
    ) -> bool:
        """Check conditions for replanning."""
        for name, condition_fn in self.conditions.items():
            try:
                if condition_fn(plan, step, error):
                    logger.info(f"Replan condition met: {name}")
                    return True
            except Exception as e:
                logger.warning(f"Condition check failed: {e}")

        return False

    def create_replacement_plan(
        self,
        original_plan: Plan,
        failed_step: PlanStep,
        context: Dict[str, Any]
    ) -> Plan:
        """Create modified replacement plan."""
        new_plan = Plan(
            plan_id=str(uuid.uuid4())[:8],
            goal=original_plan.goal
        )

        for s in original_plan.steps:
            if s.step_id == failed_step.step_id:
                new_plan.add_step(PlanStep(
                    step_id=f"{s.step_id}_modified",
                    description=f"{s.description} (modified approach)",
                    action=s.action,
                    parameters={**s.parameters, "modified": True}
                ))
            elif s.status != PlanStepStatus.COMPLETED:
                new_plan.add_step(PlanStep(
                    step_id=s.step_id,
                    description=s.description,
                    action=s.action,
                    parameters=s.parameters,
                    dependencies=s.dependencies
                ))

        return new_plan


class PlanExecutor:
    """Execute plans with monitoring."""

    def __init__(
        self,
        planner: Planner,
        replanner: Optional[Replanner] = None,
        execution_callback: Optional[Callable] = None
    ):
        """
        Initialize plan executor.

        Args:
            planner: Planner to create plans
            replanner: Optional replanner for error recovery
            execution_callback: Optional callback for step execution
        """
        self.planner = planner
        self.replanner = replanner
        self.execution_callback = execution_callback

        self.current_plan: Optional[Plan] = None
        self.execution_history: List[Dict] = []

    def execute_goal(self, goal: str, context: Dict[str, Any]) -> Plan:
        """
        Execute goal with planning.

        Args:
            goal: Goal to achieve
            context: Execution context

        Returns:
            Completed or failed plan
        """
        logger.info(f"Executing goal: {goal}")

        plan = self.planner.create_plan(goal, context)
        self.current_plan = plan
        plan.status = PlanStatus.IN_PROGRESS
        plan.started_at = time.time()

        while not plan.is_complete() and plan.status == PlanStatus.IN_PROGRESS:
            ready_steps = plan.get_ready_steps()

            if not ready_steps:
                if not plan.is_complete():
                    logger.warning("No ready steps but plan not complete")
                    plan.status = PlanStatus.FAILED
                    break
                continue

            for step in ready_steps:
                if plan.current_step_index >= len(plan.steps):
                    break

                success = self._execute_step(step, context)

                if not success and self.replanner:
                    should_replan = self.replanner.should_replan(
                        plan, step, Exception("Step failed")
                    )

                    if should_replan:
                        plan.status = PlanStatus.REPLANNING
                        new_plan = self.replanner.create_replacement_plan(plan, step, context)
                        self.current_plan = new_plan
                        plan = new_plan
                        plan.status = PlanStatus.IN_PROGRESS

                plan.current_step_index += 1

        plan.status = PlanStatus.COMPLETED if plan.is_complete() else PlanStatus.FAILED
        plan.completed_at = time.time()

        self.execution_history.append({
            "plan_id": plan.plan_id,
            "goal": goal,
            "status": plan.status.value,
            "duration_sec": plan.completed_at - plan.started_at if plan.completed_at and plan.started_at else 0
        })

        return plan

    def _execute_step(self, step: PlanStep, context: Dict[str, Any]) -> bool:
        """Execute single step."""
        logger.info(f"Executing step: {step.description}")

        step.status = PlanStepStatus.EXECUTING
        start_time = time.time()

        try:
            if self.execution_callback:
                result = self.execution_callback(step, context)
            else:
                result = self._default_step_execution(step, context)

            step.result = result
            step.status = PlanStepStatus.COMPLETED
            step.actual_duration_sec = time.time() - start_time

            logger.info(f"Step {step.step_id} completed in {step.actual_duration_sec:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Step {step.step_id} failed: {e}")
            step.status = PlanStepStatus.FAILED
            step.error = str(e)
            step.retry_count += 1
            step.actual_duration_sec = time.time() - start_time

            return False

    def _default_step_execution(self, step: PlanStep, context: Dict) -> Any:
        """Default step execution."""
        time.sleep(0.1)
        return f"Executed {step.action} with params {step.parameters}"


class PlanMonitor:
    """Monitor plan execution."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def track_step_duration(self, step_id: str, duration_sec: float) -> None:
        """Track step duration."""
        if step_id not in self.metrics:
            self.metrics[step_id] = []
        self.metrics[step_id].append(duration_sec)

    def get_execution_stats(self, plan: Plan) -> Dict[str, Any]:
        """Get execution statistics."""
        completed_steps = [s for s in plan.steps if s.status == PlanStepStatus.COMPLETED]
        failed_steps = [s for s in plan.steps if s.status == PlanStepStatus.FAILED]

        total_duration = sum(s.actual_duration_sec for s in plan.steps)
        avg_step_duration = (
            total_duration / len(completed_steps) if completed_steps else 0
        )

        return {
            "total_steps": len(plan.steps),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "total_duration_sec": total_duration,
            "avg_step_duration_sec": avg_step_duration,
            "success_rate": len(completed_steps) / len(plan.steps) if plan.steps else 0
        }


class PlanRepair:
    """Repair failed plans."""

    def repair_plan(
        self,
        failed_plan: Plan,
        failed_step: PlanStep
    ) -> Plan:
        """
        Repair plan by modifying failed step.

        Args:
            failed_plan: Original plan
            failed_step: Failed step

        Returns:
            Repaired plan
        """
        new_plan = Plan(
            plan_id=str(uuid.uuid4())[:8],
            goal=failed_plan.goal,
            metadata={"repaired_from": failed_plan.plan_id}
        )

        for step in failed_plan.steps:
            if step.step_id == failed_step.step_id:
                repaired_step = self._repair_step(step)
                new_plan.add_step(repaired_step)
            else:
                new_plan.add_step(PlanStep(
                    step_id=step.step_id,
                    description=step.description,
                    action=step.action,
                    parameters=step.parameters,
                    dependencies=step.dependencies,
                    status=step.status,
                    result=step.result
                ))

        return new_plan

    def _repair_step(self, step: PlanStep) -> PlanStep:
        """Repair individual step."""
        return PlanStep(
            step_id=f"{step.step_id}_repaired",
            description=f"{step.description} (repaired)",
            action=step.action,
            parameters={**step.parameters, "retry_safe": True},
            dependencies=step.dependencies,
            max_retries=step.max_retries + 2
        )


class MockActionExecutor:
    """Mock action executor for demonstration."""

    def __init__(self):
        self.action_handlers: Dict[str, Callable] = {
            "search": self._search_action,
            "verify": self._verify_action,
            "compile": self._compile_action,
            "execute": self._generic_action,
        }

    def execute(self, action: str, parameters: Dict) -> Any:
        """Execute action."""
        handler = self.action_handlers.get(action, self._generic_action)
        return handler(parameters)

    def _search_action(self, params: Dict) -> str:
        """Mock search."""
        return f"Found information for query: {params.get('query', 'unknown')}"

    def _verify_action(self, params: Dict) -> str:
        """Mock verify."""
        return "Information verified successfully"

    def _compile_action(self, params: Dict) -> str:
        """Mock compile."""
        return "Results compiled"

    def _generic_action(self, params: Dict) -> str:
        """Generic action."""
        return f"Executed action with params: {params}"


def demo():
    """Demonstrate planning patterns."""
    print("=" * 70)
    print("Planning Patterns Demo")
    print("=" * 70)

    print("\n--- Hierarchical Planning ---")
    planner = HierarchicalPlanner()

    research_plan = planner.create_plan("Research the topic of machine learning", {})
    print(f"Created plan with {len(research_plan.steps)} steps:")
    for step in research_plan.steps:
        print(f"  - {step.step_id}: {step.description}")

    execution_order = research_plan.get_execution_order()
    print(f"Execution order: {execution_order}")

    print("\n--- Plan Execution ---")
    executor = PlanExecutor(
        planner=planner,
        replanner=ErrorThresholdReplanner(max_retries_per_step=2)
    )

    result_plan = executor.execute_goal(
        "Research machine learning basics",
        {"context": "learning"}
    )

    print(f"Plan status: {result_plan.status.value}")
    print(f"Completed steps: {sum(1 for s in result_plan.steps if s.status == PlanStepStatus.COMPLETED)}")

    print("\n--- Plan Monitoring ---")
    monitor = PlanMonitor()
    stats = monitor.get_execution_stats(result_plan)
    print(f"Execution stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n--- Plan Repair ---")
    repair = PlanRepair()

    failed_step = PlanStep(
        step_id="step_1",
        description="Original step",
        action="search"
    )

    repaired_plan = repair.repair_plan(result_plan, failed_step)
    print(f"Repaired plan with {len(repaired_plan.steps)} steps")

    print("\n--- Replanning Strategy ---")
    replanner = ConditionReplanner(conditions={
        "high_failure_rate": lambda p, s, e: s.retry_count > 2
    })

    print("Replanner configured with custom conditions")


if __name__ == "__main__":
    demo()