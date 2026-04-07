"""
LLM-as-Judge Evaluation Framework

Implements LLM-based evaluation using judge models with standardized rubrics.
Supports multiple judge models: GPT-4, Claude, Prometheus, Llama, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class JudgeScore(Enum):
    """Standard scoring scale for LLM judges."""

    VERY_BAD = 1
    BAD = 2
    ACCEPTABLE = 3
    GOOD = 4
    VERY_GOOD = 5


@dataclass
class RubricCriterion:
    """Single criterion in a rubric."""

    name: str  # e.g., "Answer Relevance"
    description: str
    weight: float = 1.0  # For weighted scoring
    anchors: Dict[int, str] = field(default_factory=dict)  # Score -> example mapping

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "anchors": self.anchors,
        }


@dataclass
class JudgmentResult:
    """Result of a single LLM judgment."""

    content_id: str
    query: str
    response: str
    criterion: str
    score: int  # 1-5
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "query": self.query,
            "response": self.response,
            "criterion": self.criterion,
            "score": self.score,
            "rationale": self.rationale,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class AggregatedScore:
    """Aggregated scores across multiple judgments."""

    content_id: str
    criterion: str
    mean_score: float
    std_dev: float = 0.0
    min_score: int = 1
    max_score: int = 5
    agreement: float = 0.0  # Inter-rater agreement (0-1)
    judgments_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "criterion": self.criterion,
            "mean_score": self.mean_score,
            "std_dev": self.std_dev,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "agreement": self.agreement,
            "judgments_count": self.judgments_count,
        }


class Rubric:
    """Standard rubric for LLM evaluation."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.criteria: Dict[str, RubricCriterion] = {}

    def add_criterion(self, criterion: RubricCriterion) -> None:
        """Add a criterion to the rubric."""
        self.criteria[criterion.name] = criterion

    def get_criterion(self, name: str) -> Optional[RubricCriterion]:
        """Get a criterion by name."""
        return self.criteria.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "criteria": {name: c.to_dict() for name, c in self.criteria.items()},
        }

    def to_prompt(self) -> str:
        """Convert rubric to evaluation prompt."""
        prompt = f"# {self.name}\n\n{self.description}\n\n"
        prompt += "## Evaluation Criteria\n\n"

        for criterion in self.criteria.values():
            prompt += f"### {criterion.name}\n"
            prompt += f"{criterion.description}\n"

            if criterion.anchors:
                prompt += "\nAnchors:\n"
                for score, example in sorted(criterion.anchors.items()):
                    prompt += f"- Score {score}: {example}\n"
            prompt += "\n"

        return prompt


class Judge(ABC):
    """Abstract base class for LLM judges."""

    def __init__(
        self, name: str, model_id: str, config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.model_id = model_id
        self.config = config or {}
        self.cache: Dict[str, JudgmentResult] = {}

    @abstractmethod
    def judge(
        self, query: str, response: str, rubric: Rubric, criterion: str
    ) -> JudgmentResult:
        """Make a judgment on response quality."""
        pass

    def batch_judge(
        self, items: List[Dict[str, str]], rubric: Rubric, criterion: str
    ) -> List[JudgmentResult]:
        """Batch judgment across multiple items."""
        results = []
        for item in items:
            result = self.judge(
                query=item.get("query", ""),
                response=item.get("response", ""),
                rubric=rubric,
                criterion=criterion,
            )
            results.append(result)
        return results

    def _extract_score(self, judgment_text: str) -> int:
        """Extract numeric score (1-5) from judgment text."""
        # Look for pattern "Score: X" or "score X"
        import re

        match = re.search(r"(?:Score|score)[:\s]+(\d)", judgment_text)
        if match:
            return int(match.group(1))

        # Look for "X/5"
        match = re.search(r"(\d)/5", judgment_text)
        if match:
            return int(match.group(1))

        # Default to middle score if unable to parse
        return 3


class StandardRubrics:
    """Factory for standard evaluation rubrics."""

    @staticmethod
    def answer_relevance() -> Rubric:
        """Relevance of answer to question."""
        rubric = Rubric(
            "Answer Relevance",
            "Evaluate how well the response addresses the question/query.",
        )

        rubric.add_criterion(
            RubricCriterion(
                name="Relevance",
                description="Does the response directly address the query?",
                anchors={
                    1: "Response is completely off-topic",
                    2: "Response is tangentially related but misses main points",
                    3: "Response addresses the query but with some irrelevant parts",
                    4: "Response is mostly relevant with minor irrelevance",
                    5: "Response is perfectly relevant and focused",
                },
            )
        )

        return rubric

    @staticmethod
    def faithfulness() -> Rubric:
        """Faithfulness to provided context/facts."""
        rubric = Rubric(
            "Faithfulness",
            "Evaluate whether response is grounded in provided context and doesn't hallucinate.",
        )

        rubric.add_criterion(
            RubricCriterion(
                name="Factual Accuracy",
                description="Does response stick to facts from context? No hallucinations?",
                anchors={
                    1: "Multiple factual errors and hallucinations",
                    2: "Several inaccuracies or unsupported claims",
                    3: "Mostly accurate with minor errors",
                    4: "Accurate with very few unsupported claims",
                    5: "Completely faithful to provided context, no hallucinations",
                },
            )
        )

        return rubric

    @staticmethod
    def coherence() -> Rubric:
        """Coherence and clarity of response."""
        rubric = Rubric(
            "Coherence", "Evaluate logical flow, clarity, and organization of response."
        )

        rubric.add_criterion(
            RubricCriterion(
                name="Clarity",
                description="Is the response clear, well-organized, and easy to follow?",
                anchors={
                    1: "Incoherent and confusing",
                    2: "Difficult to follow with poor organization",
                    3: "Generally clear with some organizational issues",
                    4: "Clear and well-organized with minor issues",
                    5: "Perfectly clear, coherent, and well-structured",
                },
            )
        )

        return rubric

    @staticmethod
    def correctness() -> Rubric:
        """Correctness of technical/factual content."""
        rubric = Rubric(
            "Correctness",
            "Evaluate technical accuracy and correctness of the response.",
        )

        rubric.add_criterion(
            RubricCriterion(
                name="Technical Accuracy",
                description="Is the technical/domain content correct?",
                anchors={
                    1: "Fundamentally incorrect or dangerous",
                    2: "Multiple significant errors",
                    3: "Mostly correct with some errors",
                    4: "Correct with minor errors",
                    5: "Completely correct and accurate",
                },
            )
        )

        return rubric

    @staticmethod
    def completeness() -> Rubric:
        """Completeness and comprehensiveness of response."""
        rubric = Rubric(
            "Completeness",
            "Evaluate whether response covers all aspects of the question.",
        )

        rubric.add_criterion(
            RubricCriterion(
                name="Coverage",
                description="Does the response cover all key aspects and provide sufficient detail?",
                anchors={
                    1: "Extremely incomplete, missing most aspects",
                    2: "Significant gaps in coverage",
                    3: "Covers main points but lacks some details",
                    4: "Comprehensive with minor gaps",
                    5: "Complete and thorough coverage",
                },
            )
        )

        return rubric

    @staticmethod
    def helpfulness() -> Rubric:
        """Overall helpfulness and utility of response."""
        rubric = Rubric(
            "Helpfulness", "Evaluate whether response is helpful and actionable."
        )

        rubric.add_criterion(
            RubricCriterion(
                name="Practical Utility",
                description="How helpful and actionable is the response?",
                anchors={
                    1: "Not helpful at all, not actionable",
                    2: "Minimally helpful",
                    3: "Moderately helpful",
                    4: "Very helpful and mostly actionable",
                    5: "Extremely helpful, actionable, and insightful",
                },
            )
        )

        return rubric


def load_rubric(path: str) -> Rubric:
    """Load rubric from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    rubric = Rubric(data["name"], data["description"])
    for criterion_data in data.get("criteria", []):
        criterion = RubricCriterion(**criterion_data)
        rubric.add_criterion(criterion)

    return rubric


def save_rubric(rubric: Rubric, path: str) -> None:
    """Save rubric to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rubric.to_dict(), f, indent=2)
