# Autonomous Planning — Agentic Skill Prompt

Building agents that autonomously plan and execute complex multi-step tasks.

---

## 1. Identity and Mission

Implement autonomous planning agents that can decompose high-level goals into executable plans, adapt plans based on execution feedback, handle unexpected obstacles, and complete complex tasks without continuous human intervention. This includes planning algorithms, plan representation, execution monitoring, and plan repair mechanisms.

---

## 2. Theory & Fundamentals

### 2.1 Planning Problem Formulation

**Classical Planning:**
```
State: S = {predicate values}
Action: A = (preconditions, effects)
Goal: G = {target predicates}
Plan: π = [a1, a2, ..., an] such that:
      apply(a1, s0) → s1
      apply(a2, s1) → s2
      ...
      apply(an, sn) ⊨ G
```

### 2.2 Planning Paradigms

**Hierarchical Task Network (HTN):**
```
Task: high-level goal
  → Decompose into subtasks
      → Decompose further
          → Primitive actions
```

**Goal Reasoning:**
```
Goal: G
  → Identify subgoals
  → Order subgoals
  → Handle conflicts
```

**Replanning:**
```
Plan → Execute → Monitor → Detect failure → Replan → ...
```

### 2.3 Plan Representation

**Action Sequence:** Simple ordered list of actions
**Conditional Plans:** Include branches based on observations
**Hierarchical Plans:** Task decomposition tree
**Partial Plans:** Partially specified plans that are refined

### 2.4 Execution Monitoring

**Plan Condition Monitoring:** Check preconditions before actions
**Execution Monitoring:** Check if actions have expected effects
**Plan Repair:** Modify plan when deviations detected

---

## 3. Implementation Patterns

### Pattern 1: Classical Planning Engine

```python
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq

class Predicate:
    """A predicate in planning."""
    def __init__(self, name: str, args: Tuple[str, ...]):
        self.name = name
        self.args = args

    def __eq__(self, other):
        return self.name == other.name and self.args == other.args

    def __hash__(self):
        return hash((self.name, self.args))

    def __repr__(self):
        return f"{self.name}({', '.join(self.args)})"

    def __str__(self):
        return self.__repr__()


@dataclass
class Action:
    """An action in planning."""
    name: str
    parameters: Tuple[str, ...]
    preconditions: Set[Predicate] = field(default_factory=set)
    effects_add: Set[Predicate] = field(default_factory=set)
    effects_delete: Set[Predicate] = field(default_factory=set)

    def __repr__(self):
        return f"{self.name}({', '.join(self.parameters)})"


class PlanningState:
    """A state in the planning problem."""
    def __init__(self, predicates: Set[Predicate] = None):
        self.predicates = predicates or set()

    def copy(self):
        return PlanningState(set(self.predicates))

    def add(self, pred: Predicate):
        self.predicates.add(pred)

    def remove(self, pred: Predicate):
        self.predicates.discard(pred)

    def has(self, pred: Predicate) -> bool:
        return pred in self.predicates

    def satisfies(self, preconditions: Set[Predicate]) -> bool:
        return preconditions.issubset(self.predicates)

    def apply_action(self, action: Action) -> 'PlanningState':
        """Apply action effects to state."""
        new_state = self.copy()
        for pred in action.effects_add:
            new_state.add(pred)
        for pred in action.effects_delete:
            new_state.remove(pred)
        return new_state

    def __hash__(self):
        return hash(frozenset(self.predicates))

    def __eq__(self, other):
        return self.predicates == other.predicates


class ClassicalPlanner:
    """
    Classical planning using A* search with heuristics.
    """

    def __init__(self, actions: List[Action]):
        self.actions = actions
        self.action_index = {a.name: a for a in actions}

    def plan(
        self,
        initial_state: PlanningState,
        goal: Set[Predicate],
        max_expansions: int = 10000,
    ) -> Optional[List[Action]]:
        """
        Find a plan from initial state to goal.
        Uses A* search with relaxed plan heuristic.
        """
        if goal.issubset(initial_state.predicates):
            return []

        # Priority queue: (f, g, state, plan)
        open_set = [(0, 0, initial_state, [])]
        closed = set()

        expansions = 0

        while open_set and expansions < max_expansions:
            f, g, state, plan = heapq.heappop(open_set)

            # Check if goal reached
            if goal.issubset(state.predicates):
                return plan

            # Skip if already visited
            if hash(state) in closed:
                continue
            closed.add(hash(state))

            # Expand
            for action in self.actions:
                # Check preconditions
                if state.satisfies(action.preconditions):
                    # Apply action
                    new_state = state.apply_action(action)
                    new_g = g + 1

                    # Heuristic: relaxed plan length
                    h = self._relaxed_plan_heuristic(new_state, goal, action)

                    heapq.heappush(open_set, (new_g + h, new_g, new_state, plan + [action]))

            expansions += 1

        return None

    def _relaxed_plan_heuristic(
        self,
        state: PlanningState,
        goal: Set[Predicate],
        last_action: Action,
    ) -> int:
        """
        Compute heuristic: number of remaining goals in relaxed space.
        Relaxed = ignore delete effects.
        """
        remaining = goal - state.predicates
        if not remaining:
            return 0

        # Simple count heuristic
        return len(remaining)

    def _get_applicable_actions(self, state: PlanningState) -> List[Action]:
        """Get actions applicable in state."""
        return [a for a in self.actions if state.satisfies(a.preconditions)]


class HTNPlanner:
    """
    Hierarchical Task Network (HTN) planner.
    """

    def __init__(self, methods: Dict[str, List], operators: Dict[str, Action]):
        self.methods = methods  # task_name -> [Method]
        self.operators = operators  # operator_name -> Action

    def plan(
        self,
        initial_state: PlanningState,
        tasks: List[str],
    ) -> Optional[List[Action]]:
        """Find plan for tasks."""
        return self._decompose(initial_state, tasks)

    def _decompose(
        self,
        state: PlanningState,
        tasks: List[str],
    ) -> Optional[List[Action]]:
        """Recursively decompose tasks."""
        if not tasks:
            return []

        task = tasks[0]
        subtasks = tasks[1:]

        # Check if primitive
        if task in self.operators:
            action = self.operators[task]

            if not state.satisfies(action.preconditions):
                return None  # Cannot execute

            new_state = state.apply_action(action)
            rest_plan = self._decompose(new_state, subtasks)

            if rest_plan is None:
                return None

            return [action] + rest_plan

        # Non-primitive - try methods
        if task not in self.methods:
            return None  # No method available

        for method in self.methods[task]:
            # Check method preconditions
            if not state.satisfies(method.preconditions):
                continue

            # Decompose
            new_state = state.apply_action(method.effect) if method.effect else state
            plan = self._decompose(new_state, method.subtasks + subtasks)

            if plan is not None:
                return plan

        return None


@dataclass
class Method:
    """An HTN method."""
    name: str
    task: str
    preconditions: Set[Predicate] = field(default_factory=set)
    subtasks: List[str] = field(default_factory=list)
    effect: Optional[Action] = None  # Optional state-changing effect
```

### Pattern 2: LLM-Based Planning Agent

```python
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import asyncio

@dataclass
class PlanStep:
    """A single step in a plan."""
    step_id: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    status: str = "pending"  # pending, executing, completed, failed

@dataclass
class Plan:
    """A complete plan."""
    plan_id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    current_step: int = 0
    status: str = "planning"  # planning, executing, completed, failed
    execution_history: List[Dict] = field(default_factory=list)

class LLMPlanner:
    """
    Use LLM to generate plans for complex tasks.
    """

    def __init__(self, llm: Any, tools: List[Dict]):
        self.llm = llm
        self.tools = tools
        self.tool_descriptions = self._format_tools(tools)

    def _format_tools(self, tools: List[Dict]) -> str:
        """Format tools for prompt."""
        lines = []
        for tool in tools:
            lines.append(f"- {tool['name']}: {tool['description']}")
            if tool.get("parameters"):
                lines.append(f"  Parameters: {json.dumps(tool['parameters'])}")
        return "\n".join(lines)

    async def generate_plan(
        self,
        goal: str,
        context: str = "",
        max_steps: int = 10,
    ) -> Plan:
        """Generate a plan for the given goal."""
        prompt = f"""Given a goal and available tools, generate a step-by-step plan.

Goal: {goal}

Context:
{context}

Available Tools:
{self.tool_descriptions}

Generate a plan with at most {max_steps} steps.
For each step, specify:
1. The action to take
2. The parameters
3. What preconditions must be true
4. The expected outcome

Respond in JSON format:
{{
  "steps": [
    {{
      "action": "tool_name",
      "parameters": {{}},
      "preconditions": [],
      "expected_outcome": ""
    }}
  ]
}}"""

        response = await self.llm.generate(prompt)
        plan_data = json.loads(response)

        plan = Plan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            steps=[
                PlanStep(
                    step_id=f"step_{i}",
                    action=s["action"],
                    parameters=s.get("parameters", {}),
                    preconditions=s.get("preconditions", []),
                    expected_outcome=s.get("expected_outcome", ""),
                )
                for i, s in enumerate(plan_data.get("steps", []))
            ],
        )

        return plan

    async def refine_plan(
        self,
        plan: Plan,
        execution_feedback: str,
    ) -> Plan:
        """Refine plan based on execution feedback."""
        prompt = f"""Given a plan and feedback from execution, suggest improvements.

Original Goal: {plan.goal}

Current Plan:
{self._format_plan(plan)}

Execution Feedback:
{execution_feedback}

Should we:
1. Continue with current plan
2. Modify remaining steps
3. Abandon and create new plan

Provide an updated plan if needed:"""

        response = await self.llm.generate(prompt)

        try:
            data = json.loads(response)
            if "steps" in data:
                plan.steps = [
                    PlanStep(
                        step_id=f"step_{i}",
                        action=s["action"],
                        parameters=s.get("parameters", {}),
                        preconditions=s.get("preconditions", []),
                        expected_outcome=s.get("expected_outcome", ""),
                    )
                    for i, s in enumerate(data["steps"])
                ]
        except:
            pass  # Keep original plan

        return plan

    def _format_plan(self, plan: Plan) -> str:
        """Format plan for prompt."""
        lines = []
        for i, step in enumerate(plan.steps):
            lines.append(f"Step {i+1}: {step.action}")
            if step.parameters:
                lines.append(f"  Parameters: {json.dumps(step.parameters)}")
        return "\n".join(lines)


class AutonomousPlanningAgent:
    """
    Agent that plans and executes autonomously.
    """

    def __init__(
        self,
        planner: LLMPlanner,
        executor: Any,
        max_replan_count: int = 3,
    ):
        self.planner = planner
        self.executor = executor
        self.max_replan_count = max_replan_count

    async def run(self, goal: str, context: str = "") -> Dict:
        """Run the agent to accomplish the goal."""
        # Generate initial plan
        plan = await self.planner.generate_plan(goal, context)

        replan_count = 0
        execution_history = []

        while plan.status != "completed" and plan.status != "failed":
            # Execute current step
            if plan.current_step < len(plan.steps):
                step = plan.steps[plan.current_step]
                step.status = "executing"

                # Execute
                result = await self.executor.execute(
                    step.action,
                    step.parameters,
                )

                step.status = "completed" if result["success"] else "failed"
                execution_history.append({
                    "step": step.step_id,
                    "action": step.action,
                    "result": result,
                })

                # Check if we need to replan
                if not result["success"] and replan_count < self.max_replan_count:
                    # Get feedback and replan
                    feedback = self._generate_feedback(step, result)
                    plan = await self.planner.refine_plan(plan, feedback)
                    replan_count += 1
                else:
                    plan.current_step += 1

            else:
                # All steps completed
                plan.status = "completed"

        return {
            "goal": goal,
            "status": plan.status,
            "plan": plan,
            "execution_history": execution_history,
            "replan_count": replan_count,
        }

    def _generate_feedback(self, step: PlanStep, result: Dict) -> str:
        """Generate feedback for replanning."""
        return f"Step '{step.action}' with params {step.parameters} failed. Result: {result}"
```

### Pattern 3: Replanning Agent

```python
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class ExecutionResult:
    """Result of action execution."""
    action: str
    success: bool
    observation: Any
    expected: Any
    deviation: float = 0.0

class ReplanningAgent:
    """
    Agent with continuous planning and replanning capabilities.
    """

    def __init__(
        self,
        planner: LLMPlanner,
        executor: Any,
        monitor: Any,
        replan_threshold: float = 0.5,
    ):
        self.planner = planner
        self.executor = executor
        self.monitor = monitor
        self.replan_threshold = replan_threshold

    async def run_with_replanning(
        self,
        goal: str,
        context: str = "",
    ) -> Dict:
        """Run agent with automatic replanning."""
        # Initial planning
        plan = await self.planner.generate_plan(goal, context)
        initial_plan = plan

        all_history = []
        total_cost = 0.0

        while not self._is_goal_achieved(goal, all_history):
            # Execute next step
            if plan.current_step >= len(plan.steps):
                # Need more steps
                plan = await self.planner.refine_plan(plan, "Need more steps")
                continue

            step = plan.steps[plan.current_step]

            # Execute with monitoring
            start_time = time.time()
            result = await self.executor.execute(step.action, step.parameters)
            duration = time.time() - start_time
            total_cost += duration

            # Monitor for deviations
            deviation = self.monitor.check_deviation(step, result)

            # Record
            exec_result = ExecutionResult(
                action=step.action,
                success=result["success"],
                observation=result.get("observation"),
                expected=step.expected_outcome,
                deviation=deviation,
            )
            all_history.append(exec_result)

            # Decide: continue, repair, or replan
            if not result["success"] or deviation > self.replan_threshold:
                if deviation > self.replan_threshold:
                    # Recover and repair
                    repair_result = await self._repair(step, result, all_history)
                    if not repair_result:
                        # Full replan
                        plan = await self.planner.generate_plan(goal, context)
                else:
                    # Action failed - try alternative
                    plan = await self._find_alternative(plan, step, result)
            else:
                # Success - move to next step
                plan.current_step += 1

        return {
            "goal": goal,
            "status": "completed",
            "history": all_history,
            "total_cost": total_cost,
            "plan": plan,
        }

    async def _repair(
        self,
        failed_step: PlanStep,
        result: Dict,
        history: List[ExecutionResult],
    ) -> bool:
        """Attempt to repair from failed step."""
        # Try compensating actions
        compensation = self._get_compensation(failed_step.action)
        if compensation:
            await self.executor.execute(compensation["action"], compensation["parameters"])
            return True
        return False

    async def _find_alternative(
        self,
        plan: Plan,
        failed_step: PlanStep,
        result: Dict,
    ) -> Plan:
        """Find alternative way to achieve failed step."""
        feedback = f"Step {failed_step.action} failed. Find alternative approach."
        return await self.planner.refine_plan(plan, feedback)

    def _get_compensation(self, failed_action: str) -> Optional[Dict]:
        """Get compensating action for a failed action."""
        compensations = {
            "write_file": {"action": "delete_file", "parameters": {}},
            "send_email": {"action": "recall_email", "parameters": {}},
            "update_db": {"action": "rollback_db", "parameters": {}},
        }
        return compensations.get(failed_action)

    def _is_goal_achieved(self, goal: str, history: List[ExecutionResult]) -> bool:
        """Check if goal has been achieved."""
        # Check if all steps completed successfully
        return all(h.success for h in history)


class ExecutionMonitor:
    """
    Monitor action execution for deviations.
    """

    def __init__(self, tolerance: float = 0.2):
        self.tolerance = tolerance

    def check_deviation(self, step: PlanStep, result: Dict) -> float:
        """
        Check if result deviates from expected.
        Returns deviation score 0.0 (perfect) to 1.0 (completely wrong).
        """
        expected = step.expected_outcome.lower()
        observed = str(result.get("observation", "")).lower()

        if not expected or not observed:
            return 0.0

        # Simple word overlap deviation
        expected_words = set(expected.split())
        observed_words = set(observed.split())

        if not expected_words:
            return 0.0

        overlap = len(expected_words & observed_words)
        deviation = 1.0 - (overlap / len(expected_words))

        return deviation
```

### Pattern 4: Goal Reasoning Agent

```python
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import heapq

@dataclass
class Goal:
    """A goal in the system."""
    goal_id: str
    description: str
    priority: float = 1.0
    preconditions: Set[str] = field(default_factory=set)
    subgoals: List["Goal"] = field(default_factory=list)
    achieved: bool = False

class GoalReasoner:
    """
    Decompose and reason about goals.
    """

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    async def decompose_goal(
        self,
        goal: str,
        context: str = "",
    ) -> Goal:
        """Decompose a high-level goal into subgoals."""
        if self.llm:
            return await self._llm_decompose(goal, context)
        else:
            return self._rule_based_decompose(goal)

    async def _llm_decompose(self, goal: str, context: str) -> Goal:
        """Use LLM to decompose goal."""
        prompt = f"""Decompose this goal into ordered subgoals.

Goal: {goal}

Context: {context}

Each subgoal should be:
1. Achievable in 1-3 actions
2. Have clear completion criteria
3. Be ordered correctly

Respond in JSON:
{{
  "goal_id": "main",
  "description": "{goal}",
  "priority": 1.0,
  "subgoals": [
    {{
      "goal_id": "sub1",
      "description": "",
      "priority": 1.0,
      "preconditions": [],
      "subgoals": []
    }}
  ]
}}"""

        response = await self.llm.generate(prompt)
        data = json.loads(response)

        return self._parse_goal(data)

    def _rule_based_decompose(self, goal: str) -> Goal:
        """Simple rule-based decomposition."""
        # Very simplified - just create atomic goal
        return Goal(
            goal_id="main",
            description=goal,
            priority=1.0,
        )

    def _parse_goal(self, data: Dict) -> Goal:
        """Parse goal from JSON data."""
        subgoals = [
            Goal(
                goal_id=g["goal_id"],
                description=g["description"],
                priority=g.get("priority", 1.0),
                preconditions=set(g.get("preconditions", [])),
            )
            for g in data.get("subgoals", [])
        ]

        return Goal(
            goal_id=data["goal_id"],
            description=data["description"],
            priority=data.get("priority", 1.0),
            preconditions=set(data.get("preconditions", [])),
            subgoals=subgoals,
        )

    def prioritize_goals(self, goals: List[Goal]) -> List[Goal]:
        """Prioritize goals based on dependencies and priority."""
        # Topological sort with priority
        in_degree = {g.goal_id: 0 for g in goals}
        adj_list = {g.goal_id: [] for g in goals}

        # Build dependency graph
        for g in goals:
            for pre in g.preconditions:
                for other in goals:
                    if other.description == pre:
                        adj_list[other.goal_id].append(g.goal_id)
                        in_degree[g.goal_id] += 1

        # Priority queue
        pq = [(g.priority, g.goal_id) for g in goals if in_degree[g.goal_id] == 0]
        heapq.heapify(pq)

        ordered = []
        while pq:
            priority, gid = heapq.heappop(pq)
            for g in goals:
                if g.goal_id == gid:
                    ordered.append(g)
                    for neighbor in adj_list[gid]:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            for other in goals:
                                if other.goal_id == neighbor:
                                    heapq.heappush(pq, (other.priority, neighbor))
                    break

        return ordered

    def detect_conflicts(self, goals: List[Goal]) -> List[Tuple[Goal, Goal]]:
        """Detect conflicting goals."""
        conflicts = []
        goal_descs = [(g.goal_id, g.description.lower()) for g in goals]

        for i, (id1, desc1) in enumerate(goal_descs):
            for id2, desc2 in goal_descs[i+1:]:
                if self._are_conflicting(desc1, desc2):
                    g1 = next(g for g in goals if g.goal_id == id1)
                    g2 = next(g for g in goals if g.goal_id == id2)
                    conflicts.append((g1, g2))

        return conflicts

    def _are_conflicting(self, desc1: str, desc2: str) -> bool:
        """Check if two goal descriptions conflict."""
        # Simple keyword-based conflict detection
        conflict_pairs = [
            ("start", "stop"),
            ("create", "delete"),
            ("increase", "decrease"),
        ]

        words1 = set(desc1.split())
        words2 = set(desc2.split())

        for w1, w2 in conflict_pairs:
            if (w1 in words1 and w2 in words2) or (w2 in words1 and w1 in words2):
                return True

        return False


class HierarchicalPlanningAgent:
    """
    Agent with hierarchical goal reasoning and planning.
    """

    def __init__(
        self,
        goal_reasoner: GoalReasoner,
        planner: LLMPlanner,
        executor: Any,
    ):
        self.goal_reasoner = goal_reasoner
        self.planner = planner
        self.executor = executor
        self.goal_stack: List[Goal] = []

    async def run(self, high_level_goal: str) -> Dict:
        """Run agent with hierarchical planning."""
        # Decompose goal
        goal_tree = await self.goal_reasoner.decompose_goal(high_level_goal)

        # Prioritize
        ordered_goals = self.goal_reasoner.prioritize_goals([goal_tree])
        ordered_goals.extend(self._flatten_subgoals(goal_tree))

        # Check for conflicts
        conflicts = self.goal_reasoner.detect_conflicts(ordered_goals)
        if conflicts:
            # Resolve conflicts
            ordered_goals = self._resolve_conflicts(ordered_goals, conflicts)

        # Execute
        execution_history = []
        for goal in ordered_goals:
            if goal.achieved:
                continue

            # Generate plan for goal
            plan = await self.planner.generate_plan(goal.description)

            # Execute plan
            for step in plan.steps:
                result = await self.executor.execute(step.action, step.parameters)
                execution_history.append({
                    "goal": goal.description,
                    "step": step.action,
                    "result": result,
                })

                if not result["success"]:
                    break  # Abort goal on failure

            goal.achieved = True

        return {
            "goal": high_level_goal,
            "goals_achieved": sum(1 for g in ordered_goals if g.achieved),
            "total_goals": len(ordered_goals),
            "history": execution_history,
        }

    def _flatten_subgoals(self, goal: Goal) -> List[Goal]:
        """Flatten goal tree to list."""
        result = []
        for subgoal in goal.subgoals:
            result.append(subgoal)
            result.extend(self._flatten_subgoals(subgoal))
        return result

    def _resolve_conflicts(
        self,
        goals: List[Goal],
        conflicts: List[Tuple[Goal, Goal]],
    ) -> List[Goal]:
        """Resolve goal conflicts."""
        # Simple resolution: keep higher priority goal
        conflict_ids = set()
        for g1, g2 in conflicts:
            if g1.priority > g2.priority:
                conflict_ids.add(g2.goal_id)
            else:
                conflict_ids.add(g1.goal_id)

        return [g for g in goals if g.goal_id not in conflict_ids]
```

### Pattern 5: Plan Execution with State Tracking

```python
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import time

@dataclass
class WorldState:
    """Track current state of the world."""
    facts: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, float] = field(default_factory=dict)
    entities: Dict[str, Dict] = field(default_factory=dict)

    def add_fact(self, key: str, value: Any):
        self.facts[key] = value

    def get_fact(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def has_fact(self, key: str) -> bool:
        return key in self.facts

    def remove_fact(self, key: str):
        self.facts.pop(key, None)

    def use_resource(self, resource: str, amount: float) -> bool:
        if self.resources.get(resource, 0) >= amount:
            self.resources[resource] -= amount
            return True
        return False

    def add_resource(self, resource: str, amount: float):
        self.resources[resource] = self.resources.get(resource, 0) + amount


class StatefulExecutionEngine:
    """
    Execute actions with full state tracking.
    """

    def __init__(self):
        self.world_state = WorldState()
        self.action_registry: Dict[str, callable] = {}

    def register_action(
        self,
        name: str,
        action_fn: callable,
        preconditions: List[str],
        effects: Dict[str, Any],
        resource_cost: Dict[str, float] = None,
    ):
        """Register an action with its metadata."""
        self.action_registry[name] = {
            "fn": action_fn,
            "preconditions": preconditions,
            "effects": effects,
            "resource_cost": resource_cost or {},
        }

    async def execute(
        self,
        action_name: str,
        parameters: Dict[str, Any],
    ) -> Dict:
        """Execute an action with state tracking."""
        if action_name not in self.action_registry:
            return {"success": False, "error": f"Unknown action: {action_name}"}

        action_spec = self.action_registry[action_name]

        # Check preconditions
        for precond in action_spec["preconditions"]:
            if not self._check_precondition(precond):
                return {
                    "success": False,
                    "error": f"Precondition not met: {precond}",
                }

        # Check and use resources
        for resource, cost in action_spec["resource_cost"].items():
            if not self.world_state.use_resource(resource, cost):
                return {
                    "success": False,
                    "error": f"Insufficient {resource}",
                }

        # Execute action
        try:
            result = await action_spec["fn"](self.world_state, **parameters)

            # Apply effects
            for key, value in action_spec["effects"].items():
                if value is None:
                    self.world_state.remove_fact(key)
                else:
                    self.world_state.add_fact(key, value)

            return {
                "success": True,
                "result": result,
                "state": self.world_state.facts.copy(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _check_precondition(self, precond: str) -> bool:
        """Check if a precondition is satisfied."""
        # Parse and check precondition
        # Simplified: check if fact exists
        return self.world_state.has_fact(precond)


class PlanValidator:
    """
    Validate plans before execution.
    """

    def __init__(self, execution_engine: StatefulExecutionEngine):
        self.engine = execution_engine

    def validate_plan(self, plan: Plan) -> Dict:
        """Validate that a plan can be executed."""
        errors = []
        warnings = []

        # Simulate execution
        test_state = WorldState()
        original_state = self.engine.world_state

        # Temporarily use test state
        self.engine.world_state = test_state

        for i, step in enumerate(plan.steps):
            # Check if action exists
            if step.action not in self.engine.action_registry:
                errors.append(f"Step {i}: Unknown action {step.action}")
                continue

            action_spec = self.engine.action_registry[step.action]

            # Check preconditions
            for precond in action_spec["preconditions"]:
                if not self._check_fact(precond):
                    errors.append(f"Step {i}: Precondition '{precond}' not met")

            # Check resources
            for resource, cost in action_spec["resource_cost"].items():
                if test_state.resources.get(resource, 0) < cost:
                    warnings.append(
                        f"Step {i}: May run low on {resource} (have {test_state.resources.get(resource, 0)}, need {cost})"
                    )

        # Restore original state
        self.engine.world_state = original_state

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _check_fact(self, fact_key: str) -> bool:
        """Check if a fact is in the test state."""
        return self.engine.world_state.has_fact(fact_key)


class ContingentPlanner:
    """
    Plan with contingencies for possible failures.
    """

    def __init__(self, planner: LLMPlanner):
        self.planner = planner

    async def plan_with_contingencies(
        self,
        goal: str,
        possible_observations: Dict[str, str],
    ) -> Plan:
        """
        Create a plan that handles multiple possible observations.

        possible_observations: Maps situation descriptions to expected observations
        """
        prompt = f"""Create a plan for '{goal}' that handles the following contingencies:

{json.dumps(possible_observations, indent=2)}

For each step, specify:
1. What to do
2. What to check after
3. What to do if check fails

Format as a plan with branches:"""

        response = await self.planner.llm.generate(prompt)
        data = json.loads(response)

        # Parse into plan with contingencies
        plan = Plan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            steps=[PlanStep(step_id=f"step_{i}", **s) for i, s in enumerate(data.get("steps", []))],
        )

        return plan
```

### Pattern 6: PDDL-Based Classical Planning

```python
from typing import List, Dict, Any, Optional
import re

class PDDLParser:
    """
    Parse Planning Domain Definition Language (PDDL).
    """

    def __init__(self):
        self.domain_name = ""
        self.requirements: List[str] = []
        self.types: Dict[str, str] = {}
        self.predicates: Dict[str, List] = {}
        self.actions: Dict[str, Action] = {}

    def parse_domain(self, domain_text: str) -> Dict:
        """Parse PDDL domain definition."""
        # Extract domain name
        name_match = re.search(r'\(:domain\s+(\w+)', domain_text)
        if name_match:
            self.domain_name = name_match.group(1)

        # Parse requirements
        req_match = re.search(r':requirements\s+([^\)]+)', domain_text)
        if req_match:
            self.requirements = req_match.group(1).split()

        # Parse types
        types_match = re.search(r':types\s+([^\)]+)', domain_text)
        if types_match:
            self._parse_types(types_match.group(1))

        # Parse predicates
        pred_match = re.search(r':predicates\s+([^\)]+\))', domain_text, re.DOTALL)
        if pred_match:
            self._parse_predicates(pred_match.group(1))

        # Parse actions
        action_matches = re.finditer(r'\(:action\s+(\w+)', domain_text)
        for match in action_matches:
            action_text = self._extract_action(domain_text, match.start())
            if action_text:
                action = self._parse_action(action_text)
                if action:
                    self.actions[action.name] = action

        return {
            "domain": self.domain_name,
            "types": self.types,
            "predicates": self.predicates,
            "actions": {k: {"name": v.name, "parameters": v.parameters} for k, v in self.actions.items()},
        }

    def _parse_types(self, types_text: str) -> Dict[str, str]:
        """Parse type definitions."""
        # Simple parsing - returns empty for now
        return {}

    def _parse_predicates(self, pred_text: str) -> Dict[str, List]:
        """Parse predicate definitions."""
        predicates = {}
        pred_lines = pred_text.strip().split('\n')

        for line in pred_lines:
            line = line.strip()
            if not line or line.startswith(':'):
                continue

            # Simple predicate parsing
            match = re.match(r'(\w+)\s+-\s+(\w+)', line)
            if match:
                pred_name = match.group(1)
                pred_type = match.group(2)
                predicates[pred_name] = [pred_type]

        return predicates

    def _parse_action(self, action_text: str) -> Optional[Action]:
        """Parse an action definition."""
        name_match = re.search(r':action\s+(\w+)', action_text)
        if not name_match:
            return None

        name = name_match.group(1)

        # Parse parameters
        params_match = re.search(r':parameters\s+\?\w+', action_text)
        params = []
        if params_match:
            params = re.findall(r'\?(\w+)', params_match.group(0))

        # Parse preconditions
        precond_match = re.search(r':precondition\s+\(([^)]+)\)', action_text)
        preconditions = set()
        if precond_match:
            precond_str = precond_match.group(1)
            preconditions = self._parse_predicates_list(precond_str)

        # Parse effects
        effect_match = re.search(r':effect\s+\(([^)]+)\)', action_text)
        effects_add = set()
        effects_delete = set()
        if effect_match:
            effect_str = effect_match.group(1)
            # Parse effects (simplified)
            effects_add = self._parse_predicates_list(effect_str)

        return Action(
            name=name,
            parameters=tuple(params),
            preconditions=preconditions,
            effects_add=effects_add,
            effects_delete=effects_delete,
        )

    def _parse_predicates_list(self, pred_str: str) -> set:
        """Parse a list of predicates."""
        predicates = set()
        # Very simplified - just extract predicate names
        words = pred_str.split()
        for word in words:
            if word and not word.startswith('?'):
                predicates.add(Predicate(word, ()))
        return predicates

    def _extract_action(self, text: str, start: int) -> str:
        """Extract action definition from text."""
        # Find matching closing paren
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        return text[start:end]


class PDDLExecutor:
    """
    Execute plans in PDDL-defined domains.
    """

    def __init__(
        self,
        domain_text: str,
        problem_text: str,
    ):
        self.parser = PDDLParser()
        self.domain = self.parser.parse_domain(domain_text)
        self.planner = ClassicalPlanner(list(self.parser.actions.values()))
        self.state = PlanningState()

        # Parse problem
        self._parse_problem(problem_text)

    def _parse_problem(self, problem_text: str):
        """Parse PDDL problem."""
        # Set initial state
        init_match = re.search(r':init\s+\(([^)]+)\)', problem_text, re.DOTALL)
        if init_match:
            init_str = init_match.group(1)
            for pred_str in init_str.split(')'):
                if pred_str.strip():
                    # Add to initial state
                    pass

        # Set goal
        goal_match = re.search(r':goal\s+\(([^)]+)\)', problem_text, re.DOTALL)
        self.goal = set()
        if goal_match:
            goal_str = goal_match.group(1)
            # Parse goal predicates
            pass

    def execute_plan(self, plan: List[Action]) -> Dict:
        """Execute a plan."""
        history = []

        for action in plan:
            # Check preconditions
            if not self.state.satisfies(action.preconditions):
                return {
                    "success": False,
                    "failed_at": action.name,
                    "history": history,
                }

            # Apply effects
            self.state = self.state.apply_action(action)
            history.append(action.name)

        # Check goal
        if self.goal.issubset(self.state.predicates):
            return {
                "success": True,
                "history": history,
                "final_state": self.state,
            }
        else:
            return {
                "success": False,
                "failed_at": "goal_check",
                "history": history,
            }
```

---

## 4. Framework Integration

### LangChain Integration

```python
from langchain.agents import Agent
from langchain.prompts import PromptTemplate

# Planning agent in LangChain
planner_prompt = PromptTemplate(
    template="""Given the goal: {goal}
Available tools: {tools}

Create a step-by-step plan:""",
    input_variables=["goal", "tools"],
)

# Use with LangChain agent
# agent = load_agent("./planner_agent")
```

### Rossum Integration

```python
# Rossum is a planning framework
from rossum import PlanningSystem

planning_system = PlanningSystem()
planning_system.load_domain("domain.pddl")
plan = planning_system.solve("problem.pddl")
```

---

## 5. Performance Considerations

### Planning Algorithm Comparison

| Algorithm | Completeness | Optimality | Speed |
|-----------|--------------|------------|-------|
| A* | Complete | Optimal | Medium |
| BFS | Complete | Not optimal | Fast |
| HTN | Depends | Heuristic | Fast |
| LLM Planning | Approximate | N/A | Variable |

### Optimization Tips

1. **Caching**: Cache plans for similar goals
2. **Plan Libraries**: Store successful plans for reuse
3. **Anytime Planning**: Start with fast approximate, refine over time
4. **Parallel Execution**: Execute independent steps in parallel
5. **Partial Plans**: Start execution before full plan is complete

---

## 6. Common Pitfalls

1. **Planning Horizon**: Planning too far ahead when context changes
2. **State Estimation**: Incorrect assumptions about world state
3. **Goal Conflicts**: Subgoals that contradict each other
4. **Infinite Loops**: Cyclic plans that never complete
5. **Fragility**: Plans that break on minor deviations
6. **Over-planning**: Too much planning time vs. execution time

---

## 7. Research References

1. https://arxiv.org/abs/2305.10601 — "Tree of Thoughts: Deliberate Problem Solving"

2. https://arxiv.org/abs/2308.03688 — "Tool Learning with Foundation Models"

3. https://arxiv.org/abs/2209.11345 — "Reasoning and Planning in Language Models"

4. https://arxiv.org/abs/2304.10128 — "Self-Discover: LLM Self-Improvement"

5. https://arxiv.org/abs/2308.00352 — "Self-RAG: Learning to Retrieve, Generate, and Critique"

6. https://ai.science/ns rest/ — Classical planning references

7. https://arxiv.org/abs/2201.07207 — "Foundation Models for Planning"

---

## 8. Uncertainty and Limitations

**Not Covered:** PDDL parsing in depth, partial-order planning, multi-agent planning, temporal planning.

**Production Considerations:** Start with simple plan representations. Use hierarchical planning to manage complexity. Always implement execution monitoring and graceful failure handling.

(End of file - total 1480 lines)