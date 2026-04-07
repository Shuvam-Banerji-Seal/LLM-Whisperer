"""
Golden Dataset Builder for LLM Regression Testing

This module provides utilities for creating, managing, and versioning golden datasets
from production logs for LLM regression testing.

Based on best practices from:
- OptyxStack Golden Dataset Guide (2026)
- DeepEval Framework
- Industry MLOps patterns
"""

import json
import hashlib
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import re


class Severity(Enum):
    """Severity levels for test cases."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CaseType(Enum):
    """Types of test cases in golden dataset."""

    COMMON_PATH = "common_path"  # Frequent production patterns
    KNOWN_FAILURE = "known_failure"  # Historical failures
    EDGE_CASE = "edge_case"  # High-severity edge cases
    HIGH_VALUE = "high_value"  # Premium/enterprise cohort
    FORMAT_CHECK = "format_check"  # Schema/format validation


@dataclass
class TestCase:
    """
    A single test case in the golden dataset.

    Attributes:
        case_id: Unique identifier for the test case
        title: Short descriptive title
        input_messages: Input to the LLM system
        context: Supporting context (retrieved docs, tool outputs, etc.)
        expected_output: Reference output or rubric criteria
        severity: Impact level if this case fails
        case_type: Category of the test case
        cohort_tags: Labels for stratification (intent, locale, tier, etc.)
        must_have: Required behaviors in the output
        must_not_have: Forbidden patterns in the output
        failure_taxonomy: Classification of potential failures
        metadata: Additional metadata
    """

    case_id: str
    title: str
    input_messages: List[Dict[str, str]]
    expected_output: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    severity: Severity = Severity.MEDIUM
    case_type: CaseType = CaseType.COMMON_PATH
    cohort_tags: List[str] = field(default_factory=list)
    must_have: List[str] = field(default_factory=list)
    must_not_have: List[str] = field(default_factory=list)
    failure_taxonomy: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["severity"] = self.severity.value
        d["case_type"] = self.case_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create from dictionary."""
        data = data.copy()
        data["severity"] = Severity(data.get("severity", "medium"))
        data["case_type"] = CaseType(data.get("case_type", "common_path"))
        return cls(**data)


@dataclass
class GoldenDataset:
    """
    A versioned golden dataset for LLM regression testing.

    Attributes:
        name: Dataset name
        version: Semantic version
        description: Purpose and scope
        test_cases: List of test cases
        created_at: Creation timestamp
        baseline_scores: Historical baseline scores
        metadata: Additional metadata
    """

    name: str
    version: str
    description: str
    test_cases: List[TestCase]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    baseline_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute dataset hash for integrity checking."""
        self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of test cases."""
        content = json.dumps([tc.to_dict() for tc in self.test_cases], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save(self, path: str) -> None:
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "checksum": self.checksum,
            "baseline_scores": self.baseline_scores,
            "metadata": self.metadata,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "GoldenDataset":
        """Load dataset from JSON file."""
        data = json.loads(Path(path).read_text())
        test_cases = [TestCase.from_dict(tc) for tc in data["test_cases"]]
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            test_cases=test_cases,
            created_at=data["created_at"],
            baseline_scores=data.get("baseline_scores", {}),
            metadata=data.get("metadata", {}),
        )

    def get_cases_by_type(self, case_type: CaseType) -> List[TestCase]:
        """Filter cases by type."""
        return [tc for tc in self.test_cases if tc.case_type == case_type]

    def get_cases_by_severity(self, severity: Severity) -> List[TestCase]:
        """Filter cases by severity."""
        return [tc for tc in self.test_cases if tc.severity == severity]

    def get_cases_by_cohort(self, cohort_tag: str) -> List[TestCase]:
        """Filter cases by cohort tag."""
        return [tc for tc in self.test_cases if cohort_tag in tc.cohort_tags]

    def statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "total_cases": len(self.test_cases),
            "by_type": {ct.value: len(self.get_cases_by_type(ct)) for ct in CaseType},
            "by_severity": {
                s.value: len(self.get_cases_by_severity(s)) for s in Severity
            },
            "unique_cohorts": list(
                set(tag for tc in self.test_cases for tag in tc.cohort_tags)
            ),
        }


class GoldenDatasetBuilder:
    """
    Builder for creating golden datasets from production logs.

    Follows the sampling strategy:
    - 35-45% common happy path cases
    - 20-30% known failure cases
    - 15-20% high-severity edge cases
    - 10-20% high-value cohort cases
    """

    DEFAULT_SAMPLING_RATIOS = {
        CaseType.COMMON_PATH: 0.40,
        CaseType.KNOWN_FAILURE: 0.25,
        CaseType.EDGE_CASE: 0.20,
        CaseType.HIGH_VALUE: 0.10,
        CaseType.FORMAT_CHECK: 0.05,
    }

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        sampling_ratios: Optional[Dict[CaseType, float]] = None,
    ):
        self.name = name
        self.version = version
        self.description = description
        self.sampling_ratios = sampling_ratios or self.DEFAULT_SAMPLING_RATIOS
        self.test_cases: List[TestCase] = []
        self._case_counter = 0

    def _generate_case_id(self) -> str:
        """Generate unique case ID."""
        self._case_counter += 1
        return f"{self.name}_{self.version}_{self._case_counter:04d}"

    def add_case(
        self,
        title: str,
        input_messages: List[Dict[str, str]],
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: Severity = Severity.MEDIUM,
        case_type: CaseType = CaseType.COMMON_PATH,
        cohort_tags: Optional[List[str]] = None,
        must_have: Optional[List[str]] = None,
        must_not_have: Optional[List[str]] = None,
        failure_taxonomy: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GoldenDatasetBuilder":
        """Add a test case to the dataset."""
        case = TestCase(
            case_id=self._generate_case_id(),
            title=title,
            input_messages=input_messages,
            expected_output=expected_output,
            context=context,
            severity=severity,
            case_type=case_type,
            cohort_tags=cohort_tags or [],
            must_have=must_have or [],
            must_not_have=must_not_have or [],
            failure_taxonomy=failure_taxonomy or [],
            metadata=metadata or {},
        )
        self.test_cases.append(case)
        return self

    def add_from_log(
        self,
        log_entry: Dict[str, Any],
        classifier: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> "GoldenDatasetBuilder":
        """
        Add a test case from a production log entry.

        Args:
            log_entry: Raw log entry with input/output
            classifier: Function to classify the log entry
        """
        classified = classifier(log_entry)
        return self.add_case(**classified)

    def sample_from_logs(
        self,
        logs: List[Dict[str, Any]],
        classifier: Callable[[Dict[str, Any]], Dict[str, Any]],
        target_size: int = 50,
    ) -> "GoldenDatasetBuilder":
        """
        Sample logs according to the sampling ratios.

        Args:
            logs: List of production log entries
            classifier: Function to classify each log entry
            target_size: Target number of test cases
        """
        # Classify all logs
        classified = [classifier(log) for log in logs]

        # Group by case type
        by_type: Dict[CaseType, List[Dict]] = {ct: [] for ct in CaseType}
        for entry in classified:
            case_type = entry.get("case_type", CaseType.COMMON_PATH)
            if isinstance(case_type, str):
                case_type = CaseType(case_type)
            by_type[case_type].append(entry)

        # Sample according to ratios
        for case_type, ratio in self.sampling_ratios.items():
            count = int(target_size * ratio)
            available = by_type[case_type]
            selected = random.sample(available, min(count, len(available)))
            for entry in selected:
                self.add_case(**entry)

        return self

    def build(self) -> GoldenDataset:
        """Build the golden dataset."""
        return GoldenDataset(
            name=self.name,
            version=self.version,
            description=self.description,
            test_cases=self.test_cases,
        )


class PIIRedactor:
    """
    Redact PII from log entries before adding to golden dataset.
    """

    # Common PII patterns
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        self.patterns = {**self.PATTERNS, **(custom_patterns or {})}

    def redact(self, text: str) -> str:
        """Redact PII from text."""
        result = text
        for name, pattern in self.patterns.items():
            result = re.sub(pattern, f"[REDACTED_{name.upper()}]", result)
        return result

    def redact_log(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from a log entry."""

        def redact_value(value: Any) -> Any:
            if isinstance(value, str):
                return self.redact(value)
            elif isinstance(value, dict):
                return {k: redact_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [redact_value(item) for item in value]
            return value

        return redact_value(log_entry)


# Example usage and sample golden dataset
def create_sample_golden_dataset() -> GoldenDataset:
    """Create a sample golden dataset for demonstration."""

    builder = GoldenDatasetBuilder(
        name="llm_qa_golden",
        version="1.0.0",
        description="Golden dataset for QA LLM regression testing",
    )

    # Common path cases (40%)
    builder.add_case(
        title="Simple factual question",
        input_messages=[{"role": "user", "content": "What is the capital of France?"}],
        expected_output="Paris",
        severity=Severity.LOW,
        case_type=CaseType.COMMON_PATH,
        cohort_tags=["factual", "geography"],
        must_have=["Paris"],
        must_not_have=["London", "Berlin"],
    )

    builder.add_case(
        title="Code explanation request",
        input_messages=[
            {
                "role": "user",
                "content": "Explain what this Python code does: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            }
        ],
        severity=Severity.MEDIUM,
        case_type=CaseType.COMMON_PATH,
        cohort_tags=["code", "explanation"],
        must_have=["factorial", "recursive"],
    )

    # Known failure cases (25%)
    builder.add_case(
        title="Hallucination trap - non-existent fact",
        input_messages=[
            {
                "role": "user",
                "content": "When did Albert Einstein win the Nobel Prize in Chemistry?",
            }
        ],
        severity=Severity.HIGH,
        case_type=CaseType.KNOWN_FAILURE,
        cohort_tags=["factual", "hallucination_trap"],
        must_not_have=["Chemistry"],
        failure_taxonomy=["hallucination", "false_premise"],
    )

    builder.add_case(
        title="Ambiguous request - needs clarification",
        input_messages=[{"role": "user", "content": "How much does it cost?"}],
        severity=Severity.MEDIUM,
        case_type=CaseType.KNOWN_FAILURE,
        cohort_tags=["clarification"],
        must_have=["clarify", "specific"],
        failure_taxonomy=["missing_context", "ambiguous_input"],
    )

    # Edge cases (20%)
    builder.add_case(
        title="Very long input handling",
        input_messages=[{"role": "user", "content": "Summarize: " + "word " * 1000}],
        severity=Severity.HIGH,
        case_type=CaseType.EDGE_CASE,
        cohort_tags=["long_input", "summarization"],
        failure_taxonomy=["truncation", "context_overflow"],
    )

    builder.add_case(
        title="Multi-language input",
        input_messages=[
            {
                "role": "user",
                "content": "Translate to English: Bonjour, comment allez-vous?",
            }
        ],
        expected_output="Hello, how are you?",
        severity=Severity.MEDIUM,
        case_type=CaseType.EDGE_CASE,
        cohort_tags=["translation", "multilingual"],
        must_have=["Hello"],
    )

    # High-value cohort cases (10%)
    builder.add_case(
        title="Enterprise compliance question",
        input_messages=[
            {
                "role": "system",
                "content": "You are assisting an enterprise customer with compliance questions.",
            },
            {
                "role": "user",
                "content": "What are GDPR requirements for data retention?",
            },
        ],
        severity=Severity.CRITICAL,
        case_type=CaseType.HIGH_VALUE,
        cohort_tags=["enterprise", "compliance", "legal"],
        must_have=["data", "retention", "consent"],
        must_not_have=["I'm not sure", "I don't know"],
    )

    # Format check cases (5%)
    builder.add_case(
        title="JSON output format",
        input_messages=[
            {
                "role": "user",
                "content": "Return a JSON object with fields 'name' and 'age' for a person named John who is 30.",
            }
        ],
        expected_output='{"name": "John", "age": 30}',
        severity=Severity.HIGH,
        case_type=CaseType.FORMAT_CHECK,
        cohort_tags=["json", "structured_output"],
        failure_taxonomy=["invalid_json", "missing_field"],
    )

    return builder.build()


if __name__ == "__main__":
    # Create and save sample dataset
    dataset = create_sample_golden_dataset()
    dataset.save("sample_golden_dataset.json")
    print(f"Created golden dataset with {len(dataset.test_cases)} cases")
    print(f"Statistics: {json.dumps(dataset.statistics(), indent=2)}")
