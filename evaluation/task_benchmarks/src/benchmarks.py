"""
Task Benchmarks Evaluation Framework

Implements standard benchmarking for LLM systems:
- MMLU: Massive Multitask Language Understanding (14K questions, 57 subjects)
- GSM8K: Grade School Math (8.5K problems with chain-of-thought)
- HumanEval: Code generation (164 problems with execution-based scoring)
- SWE-bench: Software Engineering (2.3K GitHub issues)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuestion:
    """Single benchmark question/problem."""

    question_id: str
    question: str
    choices: Optional[List[str]] = None  # For multiple choice (MMLU)
    answer: Optional[str] = None  # Ground truth
    expected_output: Optional[str] = None  # For code/math problems
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BenchmarkDataset:
    """Base benchmark dataset loader."""

    def __init__(self, name: str, path: Optional[str] = None):
        self.name = name
        self.path = path
        self.questions: List[BenchmarkQuestion] = []

    def load(self, path: str) -> None:
        """Load dataset from file."""
        self.path = path
        with open(path, "r") as f:
            data = json.load(f)
        self._parse_data(data)

    def _parse_data(self, data: Any) -> None:
        """Parse dataset-specific format. Override in subclasses."""
        pass

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> BenchmarkQuestion:
        return self.questions[idx]


class MMLUDataset(BenchmarkDataset):
    """MMLU (Massive Multitask Language Understanding) Dataset.

    14,042 questions across 57 subjects with multiple choice answers.
    Metrics: Accuracy
    """

    def __init__(self):
        super().__init__("MMLU")
        self.subjects = {}  # Subject -> questions mapping

    def _parse_data(self, data: List[Dict[str, Any]]) -> None:
        """Parse MMLU format: [{"subject": str, "question": str, "choices": list, "answer": str}, ...]"""
        for item in data:
            q = BenchmarkQuestion(
                question_id=f"{item.get('subject', '')}_{len(self.questions)}",
                question=item.get("question", ""),
                choices=item.get("choices", []),
                answer=item.get("answer", ""),
                metadata={"subject": item.get("subject", "")},
            )
            self.questions.append(q)

            # Index by subject
            subject = item.get("subject", "unknown")
            if subject not in self.subjects:
                self.subjects[subject] = []
            self.subjects[subject].append(q)

    def accuracy(self, predictions: List[str]) -> float:
        """Compute MMLU accuracy."""
        if len(predictions) != len(self.questions):
            return 0.0
        correct = sum(
            1
            for pred, q in zip(predictions, self.questions)
            if pred.strip().lower() == q.answer.strip().lower()
        )
        return correct / len(self.questions)

    def accuracy_by_subject(self, predictions: List[str]) -> Dict[str, float]:
        """Compute accuracy per subject."""
        subject_results = {}
        pred_idx = 0

        for subject, questions in self.subjects.items():
            subject_preds = predictions[pred_idx : pred_idx + len(questions)]
            correct = sum(
                1
                for pred, q in zip(subject_preds, questions)
                if pred.strip().lower() == q.answer.strip().lower()
            )
            subject_results[subject] = correct / len(questions) if questions else 0.0
            pred_idx += len(questions)

        return subject_results


class GSM8KDataset(BenchmarkDataset):
    """GSM8K (Grade School Math 8K) Dataset.

    8,500 grade school math problems with chain-of-thought solutions.
    Metrics: Accuracy (answer extraction with numeric comparison)
    """

    def __init__(self):
        super().__init__("GSM8K")

    def _parse_data(self, data: List[Dict[str, Any]]) -> None:
        """Parse GSM8K format with chain-of-thought reasoning."""
        for item in data:
            q = BenchmarkQuestion(
                question_id=f"gsm8k_{len(self.questions)}",
                question=item.get("question", ""),
                expected_output=item.get("answer", ""),
                metadata={"difficulty": item.get("difficulty", "medium")},
            )
            self.questions.append(q)

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """Extract numerical answer from model output."""
        # Look for pattern "####" or "**answer**"
        match = re.search(r"####\s*(\d+(?:\.\d+)?)", text)
        if match:
            return match.group(1)

        # Look for "answer is X"
        match = re.search(
            r"(?:answer|final answer)[:\s]+(-?\d+(?:\.\d+)?)", text.lower()
        )
        if match:
            return match.group(1)

        # Try to extract last number
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numbers[-1] if numbers else None

    def accuracy(self, predictions: List[str], ground_truth: List[str] = None) -> float:
        """
        Compute GSM8K accuracy by comparing extracted answers.
        """
        if ground_truth is None:
            ground_truth = [q.expected_output for q in self.questions]

        if len(predictions) != len(ground_truth):
            return 0.0

        correct = 0
        for pred, truth in zip(predictions, ground_truth):
            pred_answer = self.extract_answer(pred)
            truth_answer = self.extract_answer(truth)

            if pred_answer and truth_answer:
                try:
                    if float(pred_answer) == float(truth_answer):
                        correct += 1
                except (ValueError, TypeError):
                    if pred_answer.strip() == truth_answer.strip():
                        correct += 1

        return correct / len(predictions) if predictions else 0.0


class HumanEvalDataset(BenchmarkDataset):
    """HumanEval Code Generation Dataset.

    164 Python programming problems with function implementations.
    Metrics: Pass@k (fraction of problems solved by k samples)
    """

    def __init__(self):
        super().__init__("HumanEval")

    def _parse_data(self, data: List[Dict[str, Any]]) -> None:
        """Parse HumanEval format."""
        for item in data:
            q = BenchmarkQuestion(
                question_id=f"humaneval_{item.get('task_id', len(self.questions))}",
                question=item.get("prompt", ""),
                expected_output=item.get("canonical_solution", ""),
                metadata={
                    "entry_point": item.get("entry_point", "solution"),
                    "test_cases": item.get("test_list", []),
                },
            )
            self.questions.append(q)

    @staticmethod
    def pass_at_k(num_correct: int, num_total: int, k: int) -> float:
        """
        Estimate pass@k from single sample results.
        pass@k = 1 - C(n-c, k) / C(n, k)
        where n = num_total, c = num_correct
        """
        if k > num_total or num_total == 0:
            return 0.0

        from math import comb

        # If all passed, pass@k = 1.0
        if num_correct >= k:
            return 1.0

        # If none passed, pass@k = 0.0
        if num_correct == 0:
            return 0.0

        # Compute using combinatorial formula
        numerator = 1.0
        denominator = 1.0

        for i in range(1, k + 1):
            numerator *= num_correct - i + 1
            denominator *= num_total - i + 1

        return max(0.0, 1.0 - numerator / denominator) if denominator > 0 else 0.0

    def compute_pass_at_k(self, results: List[bool], k: int = 1) -> float:
        """
        Compute pass@k metric.

        Args:
            results: List of pass/fail for each test
            k: Number of samples to consider

        Returns:
            Estimated pass@k score
        """
        num_correct = sum(1 for r in results if r)
        return self.pass_at_k(num_correct, len(results), k)


class SWEBenchDataset(BenchmarkDataset):
    """SWE-bench Software Engineering Dataset.

    2,294 GitHub issues with repository context and expected solutions.
    Metrics: Resolution rate (issue fixed by generated patch)
    """

    def __init__(self):
        super().__init__("SWE-bench")

    def _parse_data(self, data: List[Dict[str, Any]]) -> None:
        """Parse SWE-bench format."""
        for item in data:
            q = BenchmarkQuestion(
                question_id=f"swe_{item.get('instance_id', len(self.questions))}",
                question=item.get("problem_statement", ""),
                expected_output=item.get("patch", ""),
                metadata={
                    "repo": item.get("repo", ""),
                    "base_commit": item.get("base_commit", ""),
                    "test_patch": item.get("test_patch", ""),
                },
            )
            self.questions.append(q)

    def resolution_rate(self, results: List[bool]) -> float:
        """
        Compute SWE-bench resolution rate.

        Args:
            results: List of True/False for each issue resolution

        Returns:
            Fraction of issues successfully resolved
        """
        if not results:
            return 0.0
        return sum(1 for r in results if r) / len(results)


# Convenience loaders
def load_mmlu(path: str) -> MMLUDataset:
    """Load MMLU dataset from JSON file."""
    dataset = MMLUDataset()
    dataset.load(path)
    return dataset


def load_gsm8k(path: str) -> GSM8KDataset:
    """Load GSM8K dataset from JSON file."""
    dataset = GSM8KDataset()
    dataset.load(path)
    return dataset


def load_humaneval(path: str) -> HumanEvalDataset:
    """Load HumanEval dataset from JSON file."""
    dataset = HumanEvalDataset()
    dataset.load(path)
    return dataset


def load_swe_bench(path: str) -> SWEBenchDataset:
    """Load SWE-bench dataset from JSON file."""
    dataset = SWEBenchDataset()
    dataset.load(path)
    return dataset
