"""
Base classes for evaluation framework.

Provides abstract base classes and interfaces for all evaluation categories.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class EvaluationMetric:
    """Base metric class."""

    name: str
    value: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "metadata": self.metadata}


@dataclass
class EvaluationResult:
    """Base result class for evaluation runs."""

    evaluator_name: str
    dataset_name: str
    metrics: List[EvaluationMetric]
    timestamp: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def get_metric(self, name: str) -> Optional[EvaluationMetric]:
        """Get a specific metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "dataset_name": self.dataset_name,
            "metrics": [m.to_dict() for m in self.metrics],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        """Save result to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "EvaluationResult":
        """Load result from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        metrics = [EvaluationMetric(**m) for m in data.pop("metrics", [])]
        return cls(metrics=metrics, **data)


class Evaluator(ABC):
    """Base class for all evaluators."""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> EvaluationResult:
        """Run evaluation and return results."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class MetricAggregator:
    """Utility for aggregating metrics across multiple results."""

    @staticmethod
    def compute_mean(results: List[EvaluationResult], metric_name: str) -> float:
        """Compute mean of a metric across results."""
        values = []
        for result in results:
            metric = result.get_metric(metric_name)
            if metric:
                values.append(metric.value)
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def compute_std(results: List[EvaluationResult], metric_name: str) -> float:
        """Compute standard deviation of a metric across results."""
        import statistics

        values = []
        for result in results:
            metric = result.get_metric(metric_name)
            if metric:
                values.append(metric.value)
        return statistics.stdev(values) if len(values) > 1 else 0.0

    @staticmethod
    def compute_min_max(results: List[EvaluationResult], metric_name: str) -> tuple:
        """Compute min and max of a metric across results."""
        values = []
        for result in results:
            metric = result.get_metric(metric_name)
            if metric:
                values.append(metric.value)
        return (min(values), max(values)) if values else (0.0, 0.0)
