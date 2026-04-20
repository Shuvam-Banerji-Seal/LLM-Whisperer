"""
LLM Regression Test Framework

A pytest-based framework for running regression tests on LLM systems,
comparing current performance against baseline scores.

Based on:
- DeepEval framework patterns
- Confident AI regression testing guide
"""

import json
import os
import re
import pytest
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import statistics

from golden_dataset_builder import GoldenDataset, TestCase, Severity, CaseType


# ============== LLM Provider Configuration ==============

class LLMProvider(Enum):
    """Supported LLM providers for judge evaluation."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    CUSTOM = "custom"


@dataclass
class LLMJudgeConfig:
    """Configuration for LLM-as-judge evaluation."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 500
    timeout: int = 30
    # Custom prompt template (optional)
    prompt_template: Optional[str] = None
    # Expected response format
    response_format: str = "numeric"  # "numeric", "json", or "text"
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


class LLMClient:
    """
    Generic LLM client for LLM-as-judge evaluation.
    Supports multiple providers: OpenAI, Anthropic, Azure OpenAI, and custom.
    """

    # Default evaluation prompt template
    DEFAULT_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of AI-generated responses.

## Input Query:
{input_text}

## AI Response:
{actual_output}

## Evaluation Criteria:
{criteria}

## Instructions:
Evaluate the AI response based on the criteria above. Provide:
1. A score between 0.0 and 1.0 where:
   - 1.0 = Excellent, fully meets all criteria
   - 0.7-0.9 = Good, minor issues
   - 0.4-0.6 = Fair, significant issues
   - 0.0-0.3 = Poor, fails to meet criteria
2. A brief explanation of your scoring (1-2 sentences)

Respond in this exact format:
Score: <number between 0.0 and 1.0>
Explanation: <your explanation>"""

    # JSON response format template
    JSON_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of AI-generated responses.

## Input Query:
{input_text}

## AI Response:
{actual_output}

## Evaluation Criteria:
{criteria}

## Instructions:
Evaluate the AI response based on the criteria above.

Respond with ONLY a JSON object in this exact format (no markdown, no backticks):
{{"score": <number between 0.0 and 1.0>, "explanation": "<your brief explanation>"}}"""

    def __init__(self, config: LLMJudgeConfig):
        self.config = config
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.config.provider == LLMProvider.OPENAI:
            self._init_openai()
        elif self.config.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic()
        elif self.config.provider == LLMProvider.AZURE_OPENAI:
            self._init_azure_openai()
        elif self.config.provider == LLMProvider.CUSTOM:
            self._client = None  # Custom handler expected

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.config.api_base,
            )
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY"),
            )
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

    def _init_azure_openai(self):
        """Initialize Azure OpenAI client."""
        try:
            import openai
            self._client = openai.AzureOpenAI(
                api_key=self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=self.config.api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-02-01",
            )
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

    def _build_prompt(
        self, input_text: str, actual_output: str, criteria: str
    ) -> str:
        """Build the evaluation prompt."""
        if self.config.prompt_template:
            template = self.config.prompt_template
        elif self.config.response_format == "json":
            template = self.JSON_PROMPT_TEMPLATE
        else:
            template = self.DEFAULT_PROMPT_TEMPLATE

        return template.format(
            input_text=input_text,
            actual_output=actual_output,
            criteria=criteria,
        )

    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        """Parse the LLM response to extract score and explanation."""
        if self.config.response_format == "json":
            return self._parse_json_response(response_text)
        else:
            return self._parse_numeric_response(response_text)

    def _parse_json_response(self, response_text: str) -> Tuple[float, str]:
        """Parse JSON formatted response."""
        try:
            # Clean up the response
            text = response_text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            data = json.loads(text)
            score = float(data.get("score", 0.0))
            explanation = data.get("explanation", "")
            return max(0.0, min(1.0, score)), explanation
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to numeric parsing
            return self._parse_numeric_response(response_text)

    def _parse_numeric_response(self, response_text: str) -> Tuple[float, str]:
        """Parse numeric score from text response."""
        # Look for Score: X.XX pattern
        score_match = re.search(
            r"[Ss]core[:\s]+([0-9]*\.?[0-9]+)", response_text
        )
        if score_match:
            score = float(score_match.group(1))
        else:
            # Try to find any number between 0 and 1
            numbers = re.findall(r"\b(0?\.\d+|1\.0|0|1)\b", response_text)
            if numbers:
                score = float(numbers[0])
            else:
                score = 0.5  # Default fallback

        # Extract explanation
        explanation_match = re.search(
            r"[Ee]xplanation[:\s]+(.+?)(?:\n|$)", response_text, re.DOTALL
        )
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            # Use remaining text as explanation
            lines = response_text.split("\n")
            explanation = lines[-1].strip() if lines else ""

        return max(0.0, min(1.0, score)), explanation

    def evaluate(
        self, input_text: str, actual_output: str, criteria: str
    ) -> Tuple[float, str]:
        """
        Evaluate the output using the configured LLM.

        Returns:
            Tuple of (score, explanation)
        """
        import time

        prompt = self._build_prompt(input_text, actual_output, criteria)
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response_text = self._call_llm(prompt)
                score, explanation = self._parse_response(response_text)
                return score, explanation
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        # All retries failed
        raise RuntimeError(
            f"LLM evaluation failed after {self.config.max_retries} attempts: {last_error}"
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM based on provider."""
        if self.config.provider == LLMProvider.OPENAI:
            return self._call_openai(prompt)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(prompt)
        elif self.config.provider == LLMProvider.AZURE_OPENAI:
            return self._call_azure_openai(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_azure_openai(self, prompt: str) -> str:
        """Call Azure OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )
        return response.choices[0].message.content


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

    Supports multiple LLM providers (OpenAI, Anthropic, Azure OpenAI)
    with configurable evaluation criteria and prompt engineering.
    """

    def __init__(
        self,
        name: str,
        criteria: str,
        threshold: float = 0.7,
        judge_config: Optional[LLMJudgeConfig] = None,
        judge_fn: Optional[Callable[[str, str, str], Union[float, Tuple[float, str]]]] = None,
    ):
        """
        Initialize LLM Judge metric.

        Args:
            name: Name of the metric
            criteria: Description of what to evaluate
            threshold: Minimum score to pass (0.0 to 1.0)
            judge_config: LLM configuration for automated evaluation.
                          If provided, uses LLMClient for evaluation.
            judge_fn: Optional custom judge function.
                     Takes (input_text, actual_output, criteria) and returns
                     either a float score or a tuple of (score, explanation).
        """
        super().__init__(name, threshold)
        self.criteria = criteria
        self.judge_config = judge_config
        self.judge_fn = judge_fn
        self._llm_client: Optional[LLMClient] = None

        # Initialize LLM client if config provided
        if judge_config:
            self._llm_client = LLMClient(judge_config)

    def evaluate(self, test_case: TestCase, actual_output: str) -> MetricResult:
        # Get input text
        input_text = " ".join(
            [msg.get("content", "") for msg in test_case.input_messages]
        )

        try:
            if self.judge_fn:
                # Use custom judge function
                result = self.judge_fn(input_text, actual_output, self.criteria)
                if isinstance(result, tuple):
                    score, explanation = result
                else:
                    score, explanation = result, ""

                return MetricResult(
                    metric_name=self.name,
                    score=score,
                    passed=score >= self.threshold,
                    threshold=self.threshold,
                    details={
                        "criteria": self.criteria,
                        "explanation": explanation,
                        "judge_type": "custom_function",
                    },
                )

            elif self._llm_client:
                # Use LLM client for evaluation
                score, explanation = self._llm_client.evaluate(
                    input_text, actual_output, self.criteria
                )

                return MetricResult(
                    metric_name=self.name,
                    score=score,
                    passed=score >= self.threshold,
                    threshold=self.threshold,
                    details={
                        "criteria": self.criteria,
                        "explanation": explanation,
                        "provider": self.judge_config.provider.value,
                        "model": self.judge_config.model,
                        "judge_type": "llm_client",
                    },
                )

            else:
                # No judge configured - raise error with helpful message
                raise ValueError(
                    "LLMJudgeMetric requires either judge_config or judge_fn. "
                    "Please provide one of:\n"
                    "  1. judge_config: LLMJudgeConfig with provider settings\n"
                    "  2. judge_fn: Custom evaluation function\n\n"
                    "Example:\n"
                    "  config = LLMJudgeConfig(\n"
                    "      provider=LLMProvider.OPENAI,\n"
                    "      model='gpt-4',\n"
                    "      api_key='your-api-key'\n"
                    "  )\n"
                    "  metric = LLMJudgeMetric(\n"
                    "      name='quality',\n"
                    "      criteria='Evaluate response quality',\n"
                    "      judge_config=config\n"
                    "  )"
                )

        except Exception as e:
            # Return failed result with error details
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                details={
                    "criteria": self.criteria,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
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


# ============== Mock LLM and Test Implementations ==============

class MockLLMClient:
    """
    A proper mock LLM client for testing the LLM judge functionality.
    Simulates LLM responses without requiring actual API calls.
    """

    def __init__(self, response_map: Optional[Dict[str, str]] = None):
        """
        Initialize mock client with optional response mappings.

        Args:
            response_map: Dict mapping input patterns to responses
        """
        self.response_map = response_map or {}
        self.call_history: List[Dict[str, Any]] = []

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response based on the last message content."""
        last_message = messages[-1]["content"] if messages else ""

        # Check response map first
        for pattern, response in self.response_map.items():
            if pattern.lower() in last_message.lower():
                self.call_history.append({
                    "input": last_message,
                    "output": response,
                    "matched_pattern": pattern,
                })
                return response

        # Default responses based on query type
        if "capital" in last_message.lower():
            return "The capital of France is Paris."
        elif "factorial" in last_message.lower():
            return "This is a recursive factorial function that multiplies n by factorial(n-1)."
        elif "einstein" in last_message.lower():
            return "Einstein won the Nobel Prize in Physics in 1921, not in Chemistry."
        elif "cost" in last_message.lower() or "price" in last_message.lower():
            return "I'd need more context. Could you please clarify what specific product or service you're asking about?"
        elif "hello" in last_message.lower() or "hi" in last_message.lower():
            return "Hello! How can I help you today?"
        else:
            return "I can help with that question."


def create_mock_judge_fn(
    score_map: Optional[Dict[str, float]] = None
) -> Callable[[str, str, str], Tuple[float, str]]:
    """
    Create a mock judge function for testing LLM judge metrics.

    Args:
        score_map: Optional dict mapping criteria patterns to scores

    Returns:
        Judge function that returns (score, explanation)
    """
    default_scores = {
        "accurate": 0.9,
        "complete": 0.85,
        "concise": 0.8,
        "helpful": 0.9,
        "safe": 0.95,
    }
    scores = {**default_scores, **(score_map or {})}

    def mock_judge(input_text: str, actual_output: str, criteria: str) -> Tuple[float, str]:
        """Mock judge that evaluates based on criteria keywords."""
        criteria_lower = criteria.lower()

        # Check for exact criteria matches
        for key, score in scores.items():
            if key in criteria_lower:
                explanation = f"Mock judge evaluated '{key}' criteria and assigned score {score}"
                return score, explanation

        # Default evaluation based on output length
        if len(actual_output) < 10:
            score = 0.3
            explanation = "Output too short, assigned low score"
        elif len(actual_output) > 500:
            score = 0.6
            explanation = "Output verbose but may contain useful information"
        else:
            score = 0.75
            explanation = "Output has reasonable length and content"

        return score, explanation

    return mock_judge


def create_simulated_llm_judge(
    provider: LLMProvider = LLMProvider.OPENAI,
    model: str = "gpt-4",
) -> LLMJudgeConfig:
    """
    Create an LLM judge configuration for testing.

    This is a helper function to create a judge config. In production,
    you would use real API keys from environment variables.
    """
    return LLMJudgeConfig(
        provider=provider,
        model=model,
        api_key=os.getenv("OPENAI_API_KEY") if provider == LLMProvider.OPENAI else os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.0,
        max_tokens=500,
        response_format="text",
        max_retries=3,
    )


# Example usage
if __name__ == "__main__":
    from golden_dataset_builder import create_sample_golden_dataset

    print("=" * 70)
    print("LLM Regression Test Framework - Example Usage")
    print("=" * 70)

    # Create sample dataset
    dataset = create_sample_golden_dataset()

    # Create mock LLM
    mock_llm_client = MockLLMClient({
        "translate": "Translation: Hello, how are you?",
        "code": "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n```",
    })

    def mock_llm(messages: List[Dict[str, str]]) -> str:
        """Mock LLM for demonstration - proper test implementation."""
        return mock_llm_client.generate(messages)

    print("\n1. Basic Metrics Test (Exact Match, Contains, Not Contains)")
    print("-" * 70)

    # Run basic regression tests
    suite = RegressionTestSuite(
        dataset=dataset,
        llm_fn=mock_llm,
        metrics=[ExactMatchMetric(), ContainsMetric(), NotContainsMetric()],
    )

    results = suite.run()
    print(f"Regression Test Results:")
    print(f" Pass Rate: {results.pass_rate:.1%}")
    print(f" Passed: {results.passed_cases}/{results.total_cases}")
    print(f" Failed: {results.failed_cases}")
    print(f" Errors: {results.error_cases}")
    print(f" Execution Time: {results.execution_time_ms:.0f}ms")
    print(f"\nAggregate Scores:")
    for metric, score in results.aggregate_scores.items():
        print(f" {metric}: {score:.2%}")

    print("\n2. LLM Judge Metric Test (with Mock Judge Function)")
    print("-" * 70)

    # Create LLM judge with mock function
    mock_judge = create_mock_judge_fn({
        "accuracy": 0.92,
        "completeness": 0.88,
        "safety": 0.98,
    })

    llm_judge_metric = LLMJudgeMetric(
        name="quality_score",
        criteria="Evaluate the accuracy, completeness, and safety of the response",
        threshold=0.7,
        judge_fn=mock_judge,
    )

    # Test with a sample case
    test_input = [{"role": "user", "content": "What is the capital of France?"}]
    test_output = mock_llm(test_input)

    # Create a minimal test case for demonstration
    from golden_dataset_builder import TestCase, Severity, CaseType
    sample_case = TestCase(
        case_id="demo-001",
        title="Demo Quality Evaluation",
        input_messages=test_input,
        expected_output="Paris",
        severity=Severity.MEDIUM,
        case_type=CaseType.FACTUAL,
    )

    judge_result = llm_judge_metric.evaluate(sample_case, test_output)
    print(f"LLM Judge Result:")
    print(f" Metric: {judge_result.metric_name}")
    print(f" Score: {judge_result.score:.2f}")
    print(f" Passed: {judge_result.passed}")
    print(f" Threshold: {judge_result.threshold}")
    print(f" Details: {judge_result.details}")

    print("\n3. Demonstrating LLMJudgeConfig Usage")
    print("-" * 70)
    print("Example configuration for different providers:")

    # OpenAI config example
    openai_config = LLMJudgeConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        temperature=0.0,
        max_tokens=500,
        response_format="text",
    )
    print(f"\nOpenAI Config:")
    print(f"  Provider: {openai_config.provider.value}")
    print(f"  Model: {openai_config.model}")
    print(f"  Temperature: {openai_config.temperature}")

    # Anthropic config example
    anthropic_config = LLMJudgeConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-3-sonnet-20240229",
        temperature=0.0,
        max_tokens=500,
        response_format="json",
    )
    print(f"\nAnthropic Config:")
    print(f"  Provider: {anthropic_config.provider.value}")
    print(f"  Model: {anthropic_config.model}")
    print(f"  Response Format: {anthropic_config.response_format}")

    print("\n4. Error Handling Demonstration")
    print("-" * 70)

    # Test without judge_fn or judge_config - should return error
    incomplete_judge = LLMJudgeMetric(
        name="incomplete_judge",
        criteria="Some criteria",
        threshold=0.7,
        # No judge_fn or judge_config provided
    )

    error_result = incomplete_judge.evaluate(sample_case, test_output)
    print(f"Incomplete Judge Result:")
    print(f" Score: {error_result.score}")
    print(f" Passed: {error_result.passed}")
    print(f" Error in details: {'error' in error_result.details}")
    if 'error' in error_result.details:
        print(f" Error Message: {error_result.details['error'][:100]}...")

    print("\n" + "=" * 70)
    print("Example usage complete!")
    print("=" * 70)
