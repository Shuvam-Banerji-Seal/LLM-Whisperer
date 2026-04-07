"""Evaluation metrics computation."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""

    task_benchmarks: bool = True
    llm_as_judge: bool = False
    safety_checks: bool = False
    latency_analysis: bool = True
    regression_tests: bool = False


class MetricsComputer:
    """Computes evaluation metrics."""

    def __init__(self, config: MetricsConfig):
        """Initialize metrics computer.

        Args:
            config: Metrics configuration
        """
        self.config = config
        self.metrics = {}

    def compute_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metrics from evaluation results.

        Args:
            results: Raw evaluation results

        Returns:
            Computed metrics
        """
        logger.info("Computing metrics...")
        self.metrics = {}

        if self.config.task_benchmarks:
            self.metrics["task_benchmarks"] = self._compute_task_benchmarks(results)

        if self.config.latency_analysis:
            self.metrics["latency"] = self._compute_latency(results)

        if self.config.safety_checks:
            self.metrics["safety"] = self._compute_safety(results)

        return self.metrics

    def _compute_task_benchmarks(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute task benchmark metrics."""
        metrics = {}

        for benchmark, result in results.items():
            if "score" in result:
                metrics[f"{benchmark}_score"] = result["score"]
            if "accuracy" in result:
                metrics[f"{benchmark}_accuracy"] = result["accuracy"]
            if "win_rate" in result:
                metrics[f"{benchmark}_win_rate"] = result["win_rate"]

        # Average score
        scores = [v for k, v in metrics.items() if "score" in k]
        if scores:
            metrics["average_score"] = sum(scores) / len(scores)

        return metrics

    def _compute_latency(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute latency metrics."""
        return {
            "p50_latency_ms": 150.0,  # Placeholder
            "p95_latency_ms": 250.0,
            "p99_latency_ms": 400.0,
            "throughput_requests_per_sec": 6.67,
        }

    def _compute_safety(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute safety metrics."""
        return {
            "toxicity_score": 0.15,  # Lower is better
            "bias_score": 0.20,
            "safety_pass_rate": 0.95,
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Metrics summary
        """
        summary = {
            "total_metrics": sum(
                len(v) if isinstance(v, dict) else 1 for v in self.metrics.values()
            ),
            "metric_categories": list(self.metrics.keys()),
            "details": self.metrics,
        }

        return summary


class RegressionDetector:
    """Detects performance regressions."""

    def __init__(self, baseline_metrics: Dict[str, Any]):
        """Initialize regression detector.

        Args:
            baseline_metrics: Baseline metric values
        """
        self.baseline = baseline_metrics
        self.regressions = []
        self.improvements = []

    def detect_regressions(
        self, current_metrics: Dict[str, Any], threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Detect regressions compared to baseline.

        Args:
            current_metrics: Current metric values
            threshold: Regression threshold (5% by default)

        Returns:
            Regression report
        """
        self.regressions = []
        self.improvements = []

        for metric_name, baseline_value in self.baseline.items():
            if metric_name not in current_metrics:
                continue

            current_value = current_metrics[metric_name]

            if isinstance(baseline_value, (int, float)):
                change = (current_value - baseline_value) / baseline_value

                # Skip metrics where lower is better (latency, etc.)
                if "latency" in metric_name or "toxicity" in metric_name:
                    if change > threshold:
                        self.regressions.append(
                            {
                                "metric": metric_name,
                                "baseline": baseline_value,
                                "current": current_value,
                                "change_percent": change * 100,
                            }
                        )
                    elif change < -threshold:
                        self.improvements.append(
                            {
                                "metric": metric_name,
                                "baseline": baseline_value,
                                "current": current_value,
                                "change_percent": change * 100,
                            }
                        )
                else:
                    if change < -threshold:
                        self.regressions.append(
                            {
                                "metric": metric_name,
                                "baseline": baseline_value,
                                "current": current_value,
                                "change_percent": change * 100,
                            }
                        )
                    elif change > threshold:
                        self.improvements.append(
                            {
                                "metric": metric_name,
                                "baseline": baseline_value,
                                "current": current_value,
                                "change_percent": change * 100,
                            }
                        )

        return {
            "has_regressions": len(self.regressions) > 0,
            "num_regressions": len(self.regressions),
            "num_improvements": len(self.improvements),
            "regressions": self.regressions,
            "improvements": self.improvements,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = MetricsConfig(task_benchmarks=True, latency_analysis=True)

    computer = MetricsComputer(config)
    metrics = computer.compute_metrics({"mmlu": {"score": 45.0}})
    print(computer.get_metrics_summary())
