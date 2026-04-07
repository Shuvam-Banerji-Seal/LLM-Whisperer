"""
LLM Regression Test Framework

A pytest-based framework for running regression tests on LLM systems,
comparing current performance against baseline scores.

Based on:
- DeepEval framework patterns
- Confident AI regression testing guide
"""

import json
import pytest
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import statistics

from golden_dataset_builder import GoldenDataset, TestCase, Severity, CaseType


class MetricType(Enum):
    """Types of evaluation metrics."""

    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LLM_JUDGE = "llm_judge"
    CUSTOM = "custom"


@dataclass
class MetricResult:
    """Result of evaluating a single metric on a test case."""

    metric_name: str
    score: float  # 0.0 to 1.0
    passed: bool
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestCaseResult:
    """Result of evaluating a single test case."""

    case_id: str
    title: str
    severity: Severity
    case_type: CaseType
    actual_output: str
    metric_results: List[MetricResult]
    passed: bool
    execution_time_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["case_type"] = self.case_type.value
        d["metric_results"] = [mr.to_dict() for mr in self.metric_results]
        return d


@dataclass
class RegressionTestResult:
    """Aggregate results of a regression test run."""

    dataset_name: str
    dataset_version: str
    run_id: str
    timestamp: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    error_cases: int
    case_results: List[TestCaseResult]
    aggregate_scores: Dict[str, float]
    baseline_comparison: Dict[str, Dict[str, float]]
    execution_time_ms: float

    @property
    def pass_rate(self) -> float:
        """Calculate overall pass rate."""
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "error_cases": self.error_cases,
            "pass_rate": self.pass_rate,
            "aggregate_scores": self.aggregate_scores,
            "baseline_comparison": self.baseline_comparison,
            "execution_time_ms": self.execution_time_ms,
            "case_results": [cr.to_dict() for cr in self.case_results],
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


class BaseMetric:
    """Base class for evaluation metrics."""

    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold

    def evaluate(self, test_case: TestCase, actual_output: str) -> MetricResult:
        """Evaluate the metric for a test case."""
        raise NotImplementedError


class ExactMatchMetric(BaseMetric):
    """Check if output exactly matches expected."""

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = False):
        super().__init__("exact_match", threshold)
        self.case_sensitive = case_sensitive

    def evaluate(self, test_case: TestCase, actual_output: str) -> MetricResult:
        if not test_case.expected_output:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"reason": "No expected output defined"},
            )

        expected = test_case.expected_output
        actual = actual_output

        if not self.case_sensitive:
            expected = expected.lower()
            actual = actual.lower()

        score = 1.0 if expected == actual else 0.0
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details={"expected": test_case.expected_output, "actual": actual_output},
        )


class ContainsMetric(BaseMetric):
    """Check if output contains required patterns."""

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = False):
        super().__init__("contains", threshold)
        self.case_sensitive = case_sensitive

    def evaluate(self, test_case: TestCase, actual_output: str) -> MetricResult:
        if not test_case.must_have:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"reason": "No must_have patterns defined"},
            )

        output = actual_output if self.case_sensitive else actual_output.lower()
        patterns = (
            test_case.must_have
            if self.case_sensitive
            else [p.lower() for p in test_case.must_have]
        )

        found = [p for p in patterns if p in output]
        missing = [p for p in patterns if p not in output]
        score = len(found) / len(patterns) if patterns else 1.0

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details={"found": found, "missing": missing},
        )


class NotContainsMetric(BaseMetric):
    """Check if output does not contain forbidden patterns."""

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = False):
        super().__init__("not_contains", threshold)
        self.case_sensitive = case_sensitive

    def evaluate(self, test_case: TestCase, actual_output: str) -> MetricResult:
        if not test_case.must_not_have:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"reason": "No must_not_have patterns defined"},
            )

        output = actual_output if self.case_sensitive else actual_output.lower()
        patterns = (
            test_case.must_not_have
            if self.case_sensitive
            else [p.lower() for p in test_case.must_not_have]
        )

        violations = [p for p in patterns if p in output]
        clean = [p for p in patterns if p not in output]
        score = len(clean) / len(patterns) if patterns else 1.0

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details={"violations": violations, "clean": clean},
        )


class LLMJudgeMetric(BaseMetric):
    """
    Use an LLM to judge the quality of outputs.

    This is a placeholder - implement with your preferred LLM provider.
    """

    def __init__(
        self,
        name: str,
        criteria: str,
        threshold: float = 0.7,
        judge_fn: Optional[Callable[[str, str, str], float]] = None,
    ):
        super().__init__(name, threshold)
        self.criteria = criteria
        self.judge_fn = judge_fn

    def evaluate(self, test_case: TestCase, actual_output: str) -> MetricResult:
        if not self.judge_fn:
            # Return placeholder score
            return MetricResult(
                metric_name=self.name,
                score=0.5,
                passed=True,
                threshold=self.threshold,
                details={"reason": "No judge function provided - using placeholder"},
            )

        # Get input text
        input_text = " ".join(
            [msg.get("content", "") for msg in test_case.input_messages]
        )

        # Call judge function
        score = self.judge_fn(input_text, actual_output, self.criteria)

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details={"criteria": self.criteria},
        )


class RegressionTestSuite:
    """
    Regression test suite for LLM systems.

    Usage:
        suite = RegressionTestSuite(
            dataset=GoldenDataset.load("golden.json"),
            llm_fn=my_llm_function,
            metrics=[ExactMatchMetric(), ContainsMetric()]
        )
        results = suite.run()
    """

    def __init__(
        self,
        dataset: GoldenDataset,
        llm_fn: Callable[[List[Dict[str, str]]], str],
        metrics: Optional[List[BaseMetric]] = None,
        baseline_scores: Optional[Dict[str, float]] = None,
    ):
        self.dataset = dataset
        self.llm_fn = llm_fn
        self.metrics = metrics or [
            ExactMatchMetric(),
            ContainsMetric(),
            NotContainsMetric(),
        ]
        self.baseline_scores = baseline_scores or dataset.baseline_scores

    def _run_single_case(self, test_case: TestCase) -> TestCaseResult:
        """Run a single test case."""
        import time

        start_time = time.time()

        try:
            # Get LLM output
            actual_output = self.llm_fn(test_case.input_messages)

            # Evaluate all metrics
            metric_results = [
                metric.evaluate(test_case, actual_output) for metric in self.metrics
            ]

            # Determine if case passed (all metrics must pass)
            passed = all(mr.passed for mr in metric_results)

            execution_time = (time.time() - start_time) * 1000

            return TestCaseResult(
                case_id=test_case.case_id,
                title=test_case.title,
                severity=test_case.severity,
                case_type=test_case.case_type,
                actual_output=actual_output,
                metric_results=metric_results,
                passed=passed,
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return TestCaseResult(
                case_id=test_case.case_id,
                title=test_case.title,
                severity=test_case.severity,
                case_type=test_case.case_type,
                actual_output="",
                metric_results=[],
                passed=False,
                execution_time_ms=execution_time,
                error=str(e),
            )

    def run(self, run_id: Optional[str] = None) -> RegressionTestResult:
        """Run all test cases and return aggregate results."""
        import time
        import uuid

        start_time = time.time()
        run_id = run_id or str(uuid.uuid4())[:8]

        # Run all cases
        case_results = [self._run_single_case(tc) for tc in self.dataset.test_cases]

        # Calculate aggregates
        passed = sum(1 for cr in case_results if cr.passed)
        failed = sum(1 for cr in case_results if not cr.passed and not cr.error)
        errors = sum(1 for cr in case_results if cr.error)

        # Calculate aggregate scores per metric
        aggregate_scores = {}
        for metric in self.metrics:
            scores = [
                mr.score
                for cr in case_results
                for mr in cr.metric_results
                if mr.metric_name == metric.name
            ]
            if scores:
                aggregate_scores[metric.name] = statistics.mean(scores)

        # Compare to baseline
        baseline_comparison = {}
        for metric_name, current_score in aggregate_scores.items():
            if metric_name in self.baseline_scores:
                baseline = self.baseline_scores[metric_name]
                delta = current_score - baseline
                baseline_comparison[metric_name] = {
                    "current": current_score,
                    "baseline": baseline,
                    "delta": delta,
                    "delta_pct": (delta / baseline * 100) if baseline > 0 else 0,
                }

        total_time = (time.time() - start_time) * 1000

        return RegressionTestResult(
            dataset_name=self.dataset.name,
            dataset_version=self.dataset.version,
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            total_cases=len(case_results),
            passed_cases=passed,
            failed_cases=failed,
            error_cases=errors,
            case_results=case_results,
            aggregate_scores=aggregate_scores,
            baseline_comparison=baseline_comparison,
            execution_time_ms=total_time,
        )


# Pytest integration
def create_pytest_suite(
    dataset_path: str,
    llm_fn: Callable[[List[Dict[str, str]]], str],
    metrics: Optional[List[BaseMetric]] = None,
):
    """
    Create pytest test functions for the golden dataset.

    Usage:
        # In conftest.py or test file
        test_cases = create_pytest_suite(
            "golden.json",
            my_llm_fn,
            [ContainsMetric(), NotContainsMetric()]
        )
    """
    dataset = GoldenDataset.load(dataset_path)
    metrics = metrics or [ContainsMetric(), NotContainsMetric()]

    @pytest.fixture
    def golden_dataset():
        return dataset

    @pytest.mark.parametrize("test_case", dataset.test_cases, ids=lambda tc: tc.case_id)
    def test_golden_case(test_case):
        """Test a single case from the golden dataset."""
        actual_output = llm_fn(test_case.input_messages)

        for metric in metrics:
            result = metric.evaluate(test_case, actual_output)
            assert result.passed, f"{metric.name} failed: {result.details}"

    return {"golden_dataset": golden_dataset, "test_golden_case": test_golden_case}


# Example usage
if __name__ == "__main__":
    from golden_dataset_builder import create_sample_golden_dataset

    # Create sample dataset
    dataset = create_sample_golden_dataset()

    # Mock LLM function
    def mock_llm(messages: List[Dict[str, str]]) -> str:
        """Mock LLM for demonstration."""
        last_message = messages[-1]["content"] if messages else ""
        if "capital of France" in last_message:
            return "The capital of France is Paris."
        elif "factorial" in last_message:
            return "This is a recursive factorial function that multiplies n by factorial(n-1)."
        elif "Einstein" in last_message:
            return "Einstein won the Nobel Prize in Physics in 1921, not in Chemistry."
        elif "cost" in last_message.lower():
            return "I'd need more context. Could you please clarify what specific product or service you're asking about?"
        else:
            return "I can help with that question."

    # Run regression tests
    suite = RegressionTestSuite(
        dataset=dataset,
        llm_fn=mock_llm,
        metrics=[ExactMatchMetric(), ContainsMetric(), NotContainsMetric()],
    )

    results = suite.run()
    print(f"Regression Test Results:")
    print(f"  Pass Rate: {results.pass_rate:.1%}")
    print(f"  Passed: {results.passed_cases}/{results.total_cases}")
    print(f"  Failed: {results.failed_cases}")
    print(f"  Errors: {results.error_cases}")
    print(f"  Execution Time: {results.execution_time_ms:.0f}ms")
    print(f"\nAggregate Scores:")
    for metric, score in results.aggregate_scores.items():
        print(f"  {metric}: {score:.2%}")
