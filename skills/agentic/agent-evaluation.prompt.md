# Agent Evaluation — Agentic Skill Prompt

Evaluating agent performance, behavior, and capabilities through systematic benchmarking.

---

## 1. Identity and Mission

Implement comprehensive agent evaluation frameworks that measure agent performance across multiple dimensions including task completion, tool use accuracy, reasoning quality, efficiency, robustness, and safety. Effective evaluation enables agent improvement through benchmarking, regression testing, and comparative analysis.

---

## 2. Theory & Fundamentals

### 2.1 Evaluation Dimensions

**Task Performance:**
- Success rate
- Correctness of output
- Completeness of task

**Behavioral Quality:**
- Tool use accuracy
- Reasoning coherence
- Error recovery
- Consistency

**Efficiency:**
- Steps to completion
- Time to complete
- Resource usage

**Robustness:**
- Handling edge cases
- Graceful degradation
- Adversarial robustness

**Safety:**
- Harmful action avoidance
- Privacy preservation
- Alignment with instructions

### 2.2 Evaluation Methods

**Gold Standard Comparison:** Compare against known correct outputs
**LLM-as-Judge:** Use LLM to evaluate responses
**Human Evaluation:** Expert human assessment
**Trajectory Matching:** Compare to reference trajectories
**Automated Metrics:** Programmatic success criteria

### 2.3 Benchmark Design

**Static Benchmarks:** Fixed test cases
**Dynamic Benchmarks:** Procedurally generated
**Interactive Benchmarks:** Environment-based evaluation
**Long-horizon Benchmarks:** Multi-step task evaluation

---

## 3. Implementation Patterns

### Pattern 1: Comprehensive Agent Benchmark

```python
import json
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    SUCCESS_RATE = "success_rate"
    TASK_COMPLETION = "task_completion"
    TOOL_ACCURACY = "tool_accuracy"
    REASONING_QUALITY = "reasoning_quality"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    SAFETY = "safety"

@dataclass
class TestCase:
    """A test case for agent evaluation."""
    case_id: str
    task: str
    task_type: str
    expected_outcome: Any
    evaluation_criteria: Dict[str, Callable] = field(default_factory=dict)
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"

@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""
    case_id: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    trajectory: List[Dict] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Results of a full benchmark."""
    benchmark_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    results: List[EvaluationResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    category_breakdown: Dict[str, Dict] = field(default_factory=dict)
    execution_time: float = 0.0

class AgentBenchmark:
    """
    Comprehensive benchmark for agent evaluation.
    """

    def __init__(
        self,
        name: str,
        test_cases: List[TestCase],
        evaluator: Any,
    ):
        self.name = name
        self.test_cases = test_cases
        self.evaluator = evaluator

    async def run(
        self,
        agent: Any,
        max_concurrent: int = 1,
    ) -> BenchmarkResult:
        """Run benchmark on agent."""
        start_time = time.time()
        results = []

        # Run test cases
        for i in range(0, len(self.test_cases), max_concurrent):
            batch = self.test_cases[i:i + max_concurrent]
            batch_results = await asyncio.gather(*[
                self._evaluate_case(agent, case)
                for case in batch
            ])
            results.extend(batch_results)

        execution_time = time.time() - start_time

        # Compute aggregate metrics
        aggregate = self._compute_aggregates(results)
        category_breakdown = self._compute_category_breakdown(results)

        return BenchmarkResult(
            benchmark_name=self.name,
            total_cases=len(results),
            passed_cases=sum(1 for r in results if r.success),
            failed_cases=sum(1 for r in results if not r.success),
            results=results,
            aggregate_metrics=aggregate,
            category_breakdown=category_breakdown,
            execution_time=execution_time,
        )

    async def _evaluate_case(
        self,
        agent: Any,
        case: TestCase,
    ) -> EvaluationResult:
        """Evaluate a single test case."""
        start_time = time.time()

        try:
            # Run agent on task
            trajectory = []
            final_result = await agent.run(case.task)

            # Extract trajectory
            for step in final_result.get("history", []):
                trajectory.append({
                    "action": step.get("action"),
                    "observation": step.get("observation"),
                })

            # Evaluate
            metrics = await self.evaluator.evaluate(
                case=case,
                final_result=final_result,
                trajectory=trajectory,
            )

            success = metrics.get("overall_success", False)

            return EvaluationResult(
                case_id=case.case_id,
                success=success,
                metrics=metrics,
                trajectory=trajectory,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return EvaluationResult(
                case_id=case.case_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _compute_aggregates(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Compute aggregate metrics."""
        total = len(results)
        if total == 0:
            return {}

        aggregates = {
            "success_rate": sum(1 for r in results if r.success) / total,
            "avg_execution_time": sum(r.execution_time for r in results) / total,
        }

        # Average all named metrics
        all_metric_names = set()
        for r in results:
            all_metric_names.update(r.metrics.keys())

        for metric_name in all_metric_names:
            values = [r.metrics.get(metric_name, 0) for r in results]
            aggregates[f"avg_{metric_name}"] = sum(values) / len(values)

        return aggregates

    def _compute_category_breakdown(self, results: List[EvaluationResult]) -> Dict[str, Dict]:
        """Break down results by category."""
        case_map = {c.case_id: c for c in self.test_cases}
        breakdown = {}

        for result in results:
            case = case_map.get(result.case_id)
            if not case:
                continue

            category = case.category
            if category not in breakdown:
                breakdown[category] = {"total": 0, "passed": 0}

            breakdown[category]["total"] += 1
            if result.success:
                breakdown[category]["passed"] += 1

        # Compute pass rates
        for category in breakdown:
            total = breakdown[category]["total"]
            passed = breakdown[category]["passed"]
            breakdown[category]["pass_rate"] = passed / total if total > 0 else 0

        return breakdown


class MultiDimensionEvaluator:
    """
    Evaluate agent across multiple dimensions.
    """

    def __init__(self, llm: Any = None):
        self.llm = llm

    async def evaluate(
        self,
        case: TestCase,
        final_result: Dict,
        trajectory: List[Dict],
    ) -> Dict[str, float]:
        """Evaluate final result and trajectory."""
        metrics = {}

        # Task completion
        metrics["task_completion"] = self._evaluate_task_completion(
            case, final_result
        )

        # Tool accuracy
        metrics["tool_accuracy"] = self._evaluate_tool_accuracy(trajectory)

        # Reasoning quality
        metrics["reasoning_quality"] = self._evaluate_reasoning_quality(
            trajectory, case
        )

        # Efficiency
        metrics["efficiency"] = self._evaluate_efficiency(trajectory, case)

        # Overall success
        metrics["overall_success"] = (
            metrics["task_completion"] >= 0.7 and
            metrics["tool_accuracy"] >= 0.8
        )

        return metrics

    def _evaluate_task_completion(
        self,
        case: TestCase,
        final_result: Dict,
    ) -> float:
        """Evaluate if task was completed correctly."""
        if callable(case.expected_outcome):
            return 1.0 if case.expected_outcome(final_result) else 0.0

        # String comparison
        if isinstance(case.expected_outcome, str):
            response = final_result.get("response", "")
            # Simple containment check
            return 1.0 if case.expected_outcome.lower() in response.lower() else 0.0

        return 0.5  # Unknown

    def _evaluate_tool_accuracy(
        self,
        trajectory: List[Dict],
    ) -> float:
        """Evaluate tool use accuracy."""
        if not trajectory:
            return 0.0

        correct_calls = 0
        total_calls = 0

        for step in trajectory:
            action = step.get("action", "")
            if action and action != "none":
                total_calls += 1
                # Check if call was successful (simplified)
                if step.get("observation") and not step.get("error"):
                    correct_calls += 1

        if total_calls == 0:
            return 1.0  # No tool calls needed

        return correct_calls / total_calls

    def _evaluate_reasoning_quality(
        self,
        trajectory: List[Dict],
        case: TestCase,
    ) -> float:
        """Evaluate quality of reasoning."""
        # Simple heuristic: check for thought/process indicators
        if not trajectory:
            return 0.0

        reasoning_indicators = ["think", "because", "therefore", "thus", "since"]
        has_reasoning = 0
        total_steps = len(trajectory)

        for step in trajectory:
            action = str(step.get("action", "")).lower()
            if any(ind in action for ind in reasoning_indicators):
                has_reasoning += 1

        return has_reasoning / total_steps if total_steps > 0 else 0.0

    def _evaluate_efficiency(
        self,
        trajectory: List[Dict],
        case: TestCase,
    ) -> float:
        """Evaluate efficiency."""
        # Fewer steps is better, but not too few
        num_steps = len(trajectory)

        if num_steps == 0:
            return 0.0
        elif num_steps <= 3:
            return 1.0
        elif num_steps <= 10:
            return 0.8
        else:
            return max(0.1, 1.0 - (num_steps - 10) * 0.05)
```

### Pattern 2: Task-Specific Test Case Generation

```python
from typing import List, Dict, Any, Optional, Callable
import json
import random

class TestCaseGenerator:
    """
    Generate test cases for agent evaluation.
    """

    def __init__(
        self,
        llm: Any,
        domain_templates: Dict[str, List],
    ):
        self.llm = llm
        self.domain_templates = domain_templates

    async def generate_test_cases(
        self,
        domain: str,
        num_cases: int,
        difficulty_distribution: Dict[str, float] = None,
    ) -> List[TestCase]:
        """Generate test cases for a domain."""
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

        templates = self.domain_templates.get(domain, [])

        if templates:
            return self._generate_from_templates(
                templates, num_cases, difficulty_distribution
            )
        else:
            return await self._generate_llm_cases(
                domain, num_cases, difficulty_distribution
            )

    def _generate_from_templates(
        self,
        templates: List[Dict],
        num_cases: int,
        difficulty_dist: Dict[str, float],
    ) -> List[TestCase]:
        """Generate cases from templates."""
        cases = []

        for i in range(num_cases):
            # Sample template
            template = random.choice(templates)

            # Generate case from template
            case = self._instantiate_template(template, i, difficulty_dist)
            cases.append(case)

        return cases

    def _instantiate_template(
        self,
        template: Dict,
        index: int,
        difficulty_dist: Dict[str, float],
    ) -> TestCase:
        """Instantiate a template with random values."""
        difficulty = random.choices(
            list(difficulty_dist.keys()),
            weights=list(difficulty_dist.values()),
        )[0]

        # Replace variables in template
        task = template["task_template"]
        for var, values in template.get("variables", {}).items():
            task = task.replace(f"{{{var}}}", random.choice(values))

        return TestCase(
            case_id=f"case_{index}",
            task=task,
            task_type=template.get("type", "general"),
            expected_outcome=template.get("expected_outcome"),
            evaluation_criteria=template.get("criteria", {}),
            difficulty=difficulty,
            category=template.get("category", "general"),
        )

    async def _generate_llm_cases(
        self,
        domain: str,
        num_cases: int,
        difficulty_dist: Dict[str, float],
    ) -> List[TestCase]:
        """Use LLM to generate test cases."""
        prompt = f"""Generate {num_cases} test cases for evaluating an AI agent in the domain of {domain}.

For each test case, provide:
1. A specific task description
2. The expected outcome or success criteria
3. Difficulty level (easy/medium/hard)
4. Category

Generate diverse cases covering different aspects of {domain}.

Respond in JSON format:
{{
  "test_cases": [
    {{
      "task": "task description",
      "expected_outcome": "what counts as success",
      "difficulty": "easy/medium/hard",
      "category": "category name"
    }}
  ]
}}"""

        response = await self.llm.generate(prompt)
        data = json.loads(response)

        cases = []
        for i, tc in enumerate(data.get("test_cases", [])):
            cases.append(TestCase(
                case_id=f"case_{i}",
                task=tc["task"],
                task_type=tc.get("type", "general"),
                expected_outcome=tc.get("expected_outcome"),
                difficulty=tc.get("difficulty", "medium"),
                category=tc.get("category", "general"),
            ))

        return cases


class EdgeCaseGenerator:
    """
    Generate edge cases and adversarial test cases.
    """

    def __init__(self, llm: Any):
        self.llm = llm

    async def generate_edge_cases(
        self,
        base_tasks: List[str],
        edge_case_types: List[str] = None,
    ) -> List[TestCase]:
        """Generate edge cases from base tasks."""
        if edge_case_types is None:
            edge_case_types = [
                "ambiguous_input",
                "missing_information",
                "conflicting_instructions",
                "edge_values",
                "adversarial",
            ]

        edge_cases = []

        for base_task in base_tasks:
            for edge_type in edge_case_types:
                case = await self._generate_edge_case(base_task, edge_type)
                if case:
                    edge_cases.append(case)

        return edge_cases

    async def _generate_edge_case(
        self,
        base_task: str,
        edge_type: str,
    ) -> Optional[TestCase]:
        """Generate a specific type of edge case."""
        transformations = {
            "ambiguous_input": f"{base_task} [Add subtle ambiguity]",
            "missing_information": f"{base_task} [Remove key info]",
            "conflicting_instructions": f"{base_task} [Add conflicting constraint]",
            "edge_values": f"{base_task} [Test with extreme values]",
            "adversarial": f"{base_task} [Add adversarial elements]",
        }

        if edge_type not in transformations:
            return None

        return TestCase(
            case_id=f"edge_{edge_type}_{hash(base_task)}",
            task=transformations[edge_type],
            task_type="edge_case",
            expected_outcome=self._get_edge_case_expected(edge_type),
            difficulty="hard",
            category=f"edge_{edge_type}",
        )

    def _get_edge_case_expected(self, edge_type: str) -> Callable:
        """Get expected outcome function for edge case type."""
        def expected(result: Dict) -> bool:
            # Edge cases expect graceful handling
            if edge_type == "ambiguous_input":
                return "clarify" in str(result.get("response", "")).lower()
            elif edge_type == "missing_information":
                return "cannot" in str(result.get("response", "")).lower()
            else:
                return True

        return expected
```

### Pattern 3: LLM-as-Judge Evaluation

```python
from typing import Dict, Any, List, Optional
import json

class LLMJudge:
    """
    Use LLM to evaluate agent outputs.
    """

    def __init__(
        self,
        judge_llm: Any,
        evaluation_prompt_template: str = None,
    ):
        self.judge_llm = judge_llm
        self.prompt_template = evaluation_prompt_template or self._default_template()

    def _default_template(self) -> str:
        """Default evaluation prompt template."""
        return """You are evaluating an AI agent's response.

Task: {task}

Agent Response:
{response}

Evaluation Criteria:
{criteria}

Score the response on the following scale:
- Overall Quality: 1-5
- Task Completion: 1-5
- Reasoning Quality: 1-5
- Safety: 1-5

Respond in JSON format:
{{
  "overall_quality": score,
  "task_completion": score,
  "reasoning_quality": score,
  "safety": score,
  "feedback": "brief explanation"
}}"""

    async def evaluate(
        self,
        task: str,
        response: str,
        criteria: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Evaluate response using LLM judge."""
        prompt = self.prompt_template.format(
            task=task,
            response=response,
            criteria=json.dumps(criteria or {}, indent=2),
        )

        judge_response = await self.judge_llm.generate(prompt)

        try:
            scores = json.loads(judge_response)
            return {
                "overall_quality": scores.get("overall_quality", 3),
                "task_completion": scores.get("task_completion", 3),
                "reasoning_quality": scores.get("reasoning_quality", 3),
                "safety": scores.get("safety", 3),
                "feedback": scores.get("feedback", ""),
            }
        except:
            return {
                "overall_quality": 3,
                "feedback": judge_response,
            }

    async def compare_responses(
        self,
        task: str,
        response_a: str,
        response_b: str,
    ) -> Dict[str, Any]:
        """Compare two responses and determine which is better."""
        prompt = f"""Compare these two AI agent responses for the task: {task}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider:
1. Task completion
2. Accuracy
3. Clarity
4. Safety

Respond in JSON:
{{
  "winner": "A" or "B",
  "confidence": 0.0-1.0,
  "reasoning": "why"
}}"""

        result = await self.judge_llm.generate(prompt)

        try:
            return json.loads(result)
        except:
            return {"winner": "A", "reasoning": result}


class PreferencePairEvaluator:
    """
    Evaluate agent using preference pairs.
    """

    def __init__(self, llm: Any):
        self.llm = llm

    async def evaluate_preference(
        self,
        trajectory_a: List[Dict],
        trajectory_b: List[Dict],
        task: str,
    ) -> Dict[str, Any]:
        """Evaluate which trajectory is better."""
        response_a = self._format_trajectory(trajectory_a)
        response_b = self._format_trajectory(trajectory_b)

        prompt = f"""Compare two agent trajectories for task: {task}

Trajectory A:
{response_a}

Trajectory B:
{response_b}

Which trajectory is better? Consider:
- Correctness of actions
- Efficiency (fewer steps preferred)
- Error recovery
- Final outcome

Respond:
{{
  "preference": "A" or "B" or "tie",
  "scores": {{"A": score, "B": score}},
  "rationale": "explanation"
}}"""

        result = await self.llm.generate(prompt)

        try:
            return json.loads(result)
        except:
            return {"preference": "tie", "rationale": result}

    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """Format trajectory for comparison."""
        lines = []
        for i, step in enumerate(trajectory):
            lines.append(f"Step {i+1}: {step.get('action', '')}")
            if step.get("observation"):
                lines.append(f"  Result: {step.get('observation')}")
        return "\n".join(lines)
```

### Pattern 4: Regression Testing for Agents

```python
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RegressionTest:
    """A regression test case."""
    test_id: str
    task: str
    expected_behavior: str
    min_success_rate: float = 0.9

@dataclass
class RegressionResult:
    """Result of regression test."""
    test_id: str
    passed: bool
    success_rate: float
    baseline_rate: Optional[float] = None
    regression_detected: bool = False

class AgentRegressionSuite:
    """
    Run regression tests to detect agent degradation.
    """

    def __init__(
        self,
        baseline_results: Optional[Dict] = None,
        regression_threshold: float = 0.1,
    ):
        self.baseline_results = baseline_results or {}
        self.regression_threshold = regression_threshold
        self.current_results: Dict[str, RegressionResult] = {}

    def add_baseline(
        self,
        test_id: str,
        success_rate: float,
        metrics: Dict[str, float],
    ):
        """Add baseline result for a test."""
        self.baseline_results[test_id] = {
            "success_rate": success_rate,
            "metrics": metrics,
            "timestamp": time.time(),
        }

    def run_regression_test(
        self,
        test: RegressionTest,
        agent: Any,
    ) -> RegressionResult:
        """Run a single regression test."""
        # Run agent on test multiple times
        num_runs = 10
        successes = 0

        for _ in range(num_runs):
            result = agent.run(test.task)
            if self._check_success(result, test.expected_behavior):
                successes += 1

        success_rate = successes / num_runs

        # Compare to baseline
        baseline = self.baseline_results.get(test.test_id)
        regression_detected = False

        if baseline:
            regression_detected = (
                success_rate < baseline["success_rate"] - self.regression_threshold
            )

        result = RegressionResult(
            test_id=test.test_id,
            passed=success_rate >= test.min_success_rate and not regression_detected,
            success_rate=success_rate,
            baseline_rate=baseline["success_rate"] if baseline else None,
            regression_detected=regression_detected,
        )

        self.current_results[test.test_id] = result
        return result

    def run_suite(
        self,
        tests: List[RegressionTest],
        agent: Any,
    ) -> Dict[str, Any]:
        """Run full regression suite."""
        results = []

        for test in tests:
            result = self.run_regression_test(test, agent)
            results.append(result)

        # Compute summary
        regressions = [r for r in results if r.regression_detected]
        failed = [r for r in results if not r.passed]

        return {
            "total_tests": len(results),
            "passed": len(results) - len(failed),
            "failed": len(failed),
            "regressions_detected": len(regressions),
            "regression_tests": [r.test_id for r in regressions],
            "failed_tests": [r.test_id for r in failed],
            "results": {
                r.test_id: {
                    "success_rate": r.success_rate,
                    "baseline": r.baseline_rate,
                    "passed": r.passed,
                }
                for r in results
            },
        }

    def _check_success(self, result: Dict, expected: str) -> bool:
        """Check if result matches expected behavior."""
        response = str(result.get("response", "")).lower()
        expected_lower = expected.lower()

        return expected_lower in response

    def save_baseline(self, path: str):
        """Save current results as baseline."""
        baseline_data = {
            test_id: {
                "success_rate": r.success_rate,
                "baseline_rate": r.baseline_rate,
            }
            for test_id, r in self.current_results.items()
        }

        with open(path, 'w') as f:
            json.dump(baseline_data, f, indent=2)

    def load_baseline(self, path: str):
        """Load baseline from file."""
        with open(path, 'r') as f:
            self.baseline_results = json.load(f)
```

### Pattern 5: Interactive Environment Evaluation

```python
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import asyncio

@dataclass
class EnvironmentStep:
    """A step in an interactive environment."""
    observation: Any
    action: Any
    reward: float
    done: bool
    info: Dict = {}

class InteractiveEnvironment:
    """
    Interactive environment for agent evaluation.
    """

    def __init__(
        self,
        env_id: str,
        reset_fn: Callable,
        step_fn: Callable,
        max_steps: int = 100,
    ):
        self.env_id = env_id
        self.reset_fn = reset_fn
        self.step_fn = step_fn
        self.max_steps = max_steps

    async def reset(self) -> Any:
        """Reset environment."""
        return await self.reset_fn()

    async def step(self, action: Any) -> EnvironmentStep:
        """Take a step in environment."""
        return await self.step_fn(action)

    async def run_episode(
        self,
        agent: Any,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a full episode with agent."""
        max_steps = max_steps or self.max_steps

        obs = await self.reset()
        total_reward = 0
        steps = []

        for step_num in range(max_steps):
            # Agent chooses action
            action = await agent.act(obs)

            # Environment steps
            step_result = await self.step(action)

            # Record
            steps.append({
                "step": step_num,
                "observation": obs,
                "action": action,
                "reward": step_result.reward,
                "done": step_result.done,
            })

            total_reward += step_result.reward
            obs = step_result.observation

            if step_result.done:
                break

        return {
            "episode_length": len(steps),
            "total_reward": total_reward,
            "steps": steps,
            "success": step_result.done,
        }


class MultiTaskEnvironmentEvaluator:
    """
    Evaluate agent across multiple interactive tasks.
    """

    def __init__(
        self,
        environments: Dict[str, InteractiveEnvironment],
    ):
        self.environments = environments
        self.results: Dict[str, List[Dict]] = {}

    async def evaluate_on_environment(
        self,
        env_name: str,
        agent: Any,
        num_episodes: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate agent on a specific environment."""
        if env_name not in self.environments:
            return {"error": f"Unknown environment: {env_name}"}

        env = self.environments[env_name]
        episodes = []

        for episode_num in range(num_episodes):
            result = await env.run_episode(agent)
            episodes.append(result)

        # Compute statistics
        successes = [e["success"] for e in episodes]
        rewards = [e["total_reward"] for e in episodes]
        lengths = [e["episode_length"] for e in episodes]

        return {
            "environment": env_name,
            "num_episodes": num_episodes,
            "success_rate": sum(successes) / len(successes),
            "avg_reward": sum(rewards) / len(rewards),
            "avg_episode_length": sum(lengths) / len(lengths),
            "episodes": episodes,
        }

    async def evaluate_all(
        self,
        agent: Any,
        num_episodes: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate agent on all environments."""
        all_results = {}

        for env_name in self.environments:
            result = await self.evaluate_on_environment(
                env_name, agent, num_episodes
            )
            all_results[env_name] = result

        # Compute overall statistics
        all_success_rates = [
            r["success_rate"] for r in all_results.values() if "success_rate" in r
        ]

        return {
            "environments": all_results,
            "overall_success_rate": (
                sum(all_success_rates) / len(all_success_rates)
                if all_success_rates else 0
            ),
            "num_environments": len(self.environments),
        }
```

### Pattern 6: Benchmark Comparison and Reporting

```python
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""
    report_id: str
    timestamp: str
    agent_name: str
    benchmark_name: str
    benchmark_result: BenchmarkResult
    comparison_with_baseline: Optional[Dict] = None
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

class BenchmarkComparator:
    """
    Compare agent performance across benchmarks or versions.
    """

    def __init__(self):
        self.benchmark_history: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        """Add benchmark result to history."""
        self.benchmark_history.append(result)

    def compare_versions(
        self,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """Compare two agent versions."""
        # Find results for each version
        results_a = [
            r for r in self.benchmark_history
            if r.benchmark_name == version_a
        ]
        results_b = [
            r for r in self.benchmark_history
            if r.benchmark_name == version_b
        ]

        if not results_a or not results_b:
            return {"error": "Insufficient data for comparison"}

        # Average results
        avg_a = self._average_results(results_a)
        avg_b = self._average_results(results_b)

        # Compute deltas
        deltas = {}
        for metric in avg_a:
            if metric in avg_b:
                delta = avg_b[metric] - avg_a[metric]
                deltas[metric] = {
                    "change": delta,
                    "percent_change": (delta / avg_a[metric] * 100) if avg_a[metric] != 0 else 0,
                    "improved": delta > 0,
                }

        return {
            "version_a": version_a,
            "version_b": version_b,
            "metrics_a": avg_a,
            "metrics_b": avg_b,
            "deltas": deltas,
        }

    def _average_results(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """Average metrics across results."""
        if not results:
            return {}

        avg = {}
        for metric_name in results[0].aggregate_metrics:
            values = [r.aggregate_metrics.get(metric_name, 0) for r in results]
            avg[metric_name] = sum(values) / len(values)

        return avg


class BenchmarkReporter:
    """
    Generate comprehensive benchmark reports.
    """

    def __init__(self):
        self.formatters = {
            "json": self._format_json,
            "html": self._format_html,
            "markdown": self._format_markdown,
        }

    def generate_report(
        self,
        result: BenchmarkResult,
        agent_name: str,
        format: str = "markdown",
    ) -> str:
        """Generate a benchmark report."""
        formatter = self.formatters.get(format, self._format_markdown)
        return formatter(result, agent_name)

    def _format_json(
        self,
        result: BenchmarkResult,
        agent_name: str,
    ) -> str:
        """Format as JSON."""
        data = {
            "report_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "benchmark_name": result.benchmark_name,
            "summary": {
                "total_cases": result.total_cases,
                "passed": result.passed_cases,
                "failed": result.failed_cases,
                "pass_rate": result.passed_cases / result.total_cases if result.total_cases > 0 else 0,
            },
            "aggregate_metrics": result.aggregate_metrics,
            "category_breakdown": result.category_breakdown,
        }

        return json.dumps(data, indent=2)

    def _format_markdown(
        self,
        result: BenchmarkResult,
        agent_name: str,
    ) -> str:
        """Format as Markdown."""
        lines = [
            f"# {result.benchmark_name} Benchmark Report",
            f"",
            f"**Agent:** {agent_name}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Duration:** {result.execution_time:.2f}s",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|---------|-------|",
            f"| Total Cases | {result.total_cases} |",
            f"| Passed | {result.passed_cases} |",
            f"| Failed | {result.failed_cases} |",
            f"| Pass Rate | {result.passed_cases/result.total_cases*100:.1f}% |",
            f"",
            f"## Aggregate Metrics",
            f"",
        ]

        for metric, value in result.aggregate_metrics.items():
            lines.append(f"- **{metric}:** {value:.3f}")

        lines.extend([
            f"",
            f"## Category Breakdown",
            f"",
            f"| Category | Pass Rate |",
            f"|----------|-----------|",
        ])

        for category, data in result.category_breakdown.items():
            rate = data.get("pass_rate", 0) * 100
            lines.append(f"| {category} | {rate:.1f}% |")

        return "\n".join(lines)

    def _format_html(
        self,
        result: BenchmarkResult,
        agent_name: str,
    ) -> str:
        """Format as HTML."""
        pass_rate = result.passed_cases / result.total_cases * 100 if result.total_cases > 0 else 0

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{result.benchmark_name} Benchmark</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f0f0f0; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .category {{ margin: 10px 0; padding: 10px; background: #e8e8e8; }}
    </style>
</head>
<body>
    <h1>{result.benchmark_name}</h1>
    <p>Agent: {agent_name}</p>

    <div class="metric">
        <div>Pass Rate</div>
        <div class="metric-value">{pass_rate:.1f}%</div>
    </div>

    <h2>Aggregate Metrics</h2>
    <ul>
"""

        for metric, value in result.aggregate_metrics.items():
            html += f"        <li>{metric}: {value:.3f}</li>\n"

        html += """
    </ul>
</body>
</html>
"""
        return html
```

---

## 4. Framework Integration

### LangChain Testing

```python
from langchain.evaluation import load_evaluator
from langchain.evaluation.criteria import Criteria

# Criteria evaluation
evaluator = load_evaluator("criteria", criteria=Criteria.CONCISENESS)
result = evaluator.evaluate_strings(
    prediction="short answer",
    reference="reference answer",
    input="question",
)
```

### HF Evaluate Integration

```python
from datasets import load_metric

# Load standard metrics
accuracy = load_metric("accuracy")
precision = load_metric("precision")
recall = load_metric("recall")

# Compute metrics
results = accuracy.compute(predictions=preds, references=refs)
```

---

## 5. Performance Considerations

### Evaluation Speed Optimization

| Method | Time per Case | Accuracy |
|--------|--------------|----------|
| Gold Standard | Fast | High |
| LLM-as-Judge | Slow | Variable |
| Automated Metrics | Very Fast | Medium |
| Human Evaluation | Very Slow | Highest |

### Cost Optimization

1. **Sample Size**: Balance statistical significance with cost
2. **Parallel Execution**: Run multiple evaluations concurrently
3. **Caching**: Cache evaluation results for regression testing
4. **Tiered Evaluation**: Fast filter then detailed evaluation

---

## 6. Common Pitfalls

1. **Overfitting to Benchmarks**: Agent memorizes test cases
2. **Metric Gaming**: Optimizing for metric not true objective
3. **Benchmark Contamination**: Training data includes test cases
4. **Insufficient Diversity**: Narrow test case distribution
5. **Evaluation Bias**: LLM judge has consistent biases
6. **Temporal Drift**: Benchmarks become outdated over time

---

## 7. Research References

1. https://arxiv.org/abs/2308.03688 — "Tool Learning with Foundation Models"

2. https://arxiv.org/abs/2304.06702 — "Task Decomposition for Agent Planning"

3. https://github.com/AGI-Edgerunners/AgentBench — Agent Benchmark

4. https://github.com/ShannonAI/BET — Benchmarking Tool-use Agents

5. https://arxiv.org/abs/2306.08637 — "WebArena: Web Agent Benchmark"

6. https://arxiv.org/abs/2308.10848 — "Iterative Retrieval-Augmented Language Models"

7. https://github.com/GAIR/digbench — Directory of agent benchmarks

---

## 8. Uncertainty and Limitations

**Not Covered:** Human preference studies, A/B testing infrastructure, production monitoring.

**Production Considerations:** Run evaluation continuously in CI/CD. Track metrics over time to detect regressions. Use multiple evaluation methods for comprehensive coverage.

(End of file - total 1420 lines)