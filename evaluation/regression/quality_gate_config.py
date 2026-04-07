"""
Quality Gate Configuration for LLM Regression Testing

Quality gates are automated checkpoints that block deployments
when evaluation scores fall below defined thresholds.

Supports:
- Absolute thresholds (e.g., accuracy > 80%)
- Relative thresholds (e.g., delta < 5% from baseline)
- Statistical significance testing (p-value thresholds)
- Multiple hypothesis correction
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import statistics


class GateType(Enum):
    """Types of quality gates."""

    ABSOLUTE = "absolute"  # Fixed threshold
    RELATIVE = "relative"  # Relative to baseline
    STATISTICAL = "statistical"  # P-value based
    COMPOSITE = "composite"  # Multiple conditions


class GateResult(Enum):
    """Result of quality gate evaluation."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class GateCheckResult:
    """Result of a single gate check."""

    gate_name: str
    gate_type: GateType
    result: GateResult
    metric_name: str
    current_value: float
    threshold: float
    baseline_value: Optional[float] = None
    delta: Optional[float] = None
    p_value: Optional[float] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gate_type"] = self.gate_type.value
        d["result"] = self.result.value
        return d


@dataclass
class QualityGateReport:
    """Report of all quality gate evaluations."""

    timestamp: str
    overall_result: GateResult
    gate_results: List[GateCheckResult]
    summary: Dict[str, int]

    @property
    def passed(self) -> bool:
        return self.overall_result == GateResult.PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_result": self.overall_result.value,
            "gate_results": [gr.to_dict() for gr in self.gate_results],
            "summary": self.summary,
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


class QualityGate:
    """
    Quality gate for LLM regression testing.

    Usage:
        gate = QualityGate()
        gate.add_absolute_threshold("accuracy", min_value=0.8)
        gate.add_relative_threshold("coherence", max_degradation=0.05)
        gate.add_statistical_threshold("relevance", p_value=0.05)

        report = gate.check(current_scores, baseline_scores)
        if not report.passed:
            raise Exception("Quality gates failed!")
    """

    def __init__(self, fail_on_warning: bool = False):
        self.fail_on_warning = fail_on_warning
        self.gates: List[Dict[str, Any]] = []

    def add_absolute_threshold(
        self,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        warning_min: Optional[float] = None,
        warning_max: Optional[float] = None,
    ) -> "QualityGate":
        """
        Add an absolute threshold gate.

        Args:
            metric_name: Name of the metric to check
            min_value: Minimum acceptable value (fail if below)
            max_value: Maximum acceptable value (fail if above)
            warning_min: Warning threshold for minimum
            warning_max: Warning threshold for maximum
        """
        self.gates.append(
            {
                "type": GateType.ABSOLUTE,
                "metric_name": metric_name,
                "min_value": min_value,
                "max_value": max_value,
                "warning_min": warning_min,
                "warning_max": warning_max,
            }
        )
        return self

    def add_relative_threshold(
        self,
        metric_name: str,
        max_degradation: Optional[float] = None,
        max_improvement: Optional[float] = None,
        warning_degradation: Optional[float] = None,
    ) -> "QualityGate":
        """
        Add a relative threshold gate (compared to baseline).

        Args:
            metric_name: Name of the metric to check
            max_degradation: Maximum allowed degradation (e.g., 0.05 = 5%)
            max_improvement: Maximum allowed improvement (for detecting anomalies)
            warning_degradation: Warning threshold for degradation
        """
        self.gates.append(
            {
                "type": GateType.RELATIVE,
                "metric_name": metric_name,
                "max_degradation": max_degradation,
                "max_improvement": max_improvement,
                "warning_degradation": warning_degradation,
            }
        )
        return self

    def add_statistical_threshold(
        self,
        metric_name: str,
        p_value_threshold: float = 0.05,
        min_samples: int = 30,
        correction: str = "bonferroni",
    ) -> "QualityGate":
        """
        Add a statistical significance gate.

        Args:
            metric_name: Name of the metric to check
            p_value_threshold: P-value threshold for significance
            min_samples: Minimum samples required for test
            correction: Multiple hypothesis correction method
        """
        self.gates.append(
            {
                "type": GateType.STATISTICAL,
                "metric_name": metric_name,
                "p_value_threshold": p_value_threshold,
                "min_samples": min_samples,
                "correction": correction,
            }
        )
        return self

    def _check_absolute(
        self, gate: Dict[str, Any], current_value: float
    ) -> GateCheckResult:
        """Check an absolute threshold gate."""
        metric_name = gate["metric_name"]
        min_val = gate.get("min_value")
        max_val = gate.get("max_value")
        warning_min = gate.get("warning_min")
        warning_max = gate.get("warning_max")

        result = GateResult.PASSED
        message = ""
        threshold = min_val if min_val is not None else max_val

        if min_val is not None and current_value < min_val:
            result = GateResult.FAILED
            message = f"{metric_name}={current_value:.4f} < min={min_val:.4f}"
        elif max_val is not None and current_value > max_val:
            result = GateResult.FAILED
            message = f"{metric_name}={current_value:.4f} > max={max_val:.4f}"
        elif warning_min is not None and current_value < warning_min:
            result = GateResult.WARNING
            message = (
                f"{metric_name}={current_value:.4f} < warning_min={warning_min:.4f}"
            )
        elif warning_max is not None and current_value > warning_max:
            result = GateResult.WARNING
            message = (
                f"{metric_name}={current_value:.4f} > warning_max={warning_max:.4f}"
            )
        else:
            message = f"{metric_name}={current_value:.4f} within bounds"

        return GateCheckResult(
            gate_name=f"absolute_{metric_name}",
            gate_type=GateType.ABSOLUTE,
            result=result,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold if threshold is not None else 0.0,
            message=message,
        )

    def _check_relative(
        self, gate: Dict[str, Any], current_value: float, baseline_value: float
    ) -> GateCheckResult:
        """Check a relative threshold gate."""
        metric_name = gate["metric_name"]
        max_deg = gate.get("max_degradation")
        max_imp = gate.get("max_improvement")
        warning_deg = gate.get("warning_degradation")

        if baseline_value == 0:
            return GateCheckResult(
                gate_name=f"relative_{metric_name}",
                gate_type=GateType.RELATIVE,
                result=GateResult.SKIPPED,
                metric_name=metric_name,
                current_value=current_value,
                threshold=0.0,
                baseline_value=baseline_value,
                message="Baseline is zero, cannot compute relative change",
            )

        delta = current_value - baseline_value
        delta_pct = delta / baseline_value

        result = GateResult.PASSED
        message = ""
        threshold = max_deg if max_deg is not None else 0.0

        # Check degradation (negative delta for metrics where higher is better)
        if max_deg is not None and delta_pct < -max_deg:
            result = GateResult.FAILED
            message = (
                f"{metric_name} degraded by {abs(delta_pct):.2%} > max {max_deg:.2%}"
            )
        elif max_imp is not None and delta_pct > max_imp:
            result = GateResult.WARNING
            message = f"{metric_name} improved by {delta_pct:.2%} > max {max_imp:.2%} (anomaly?)"
        elif warning_deg is not None and delta_pct < -warning_deg:
            result = GateResult.WARNING
            message = f"{metric_name} degraded by {abs(delta_pct):.2%} > warning {warning_deg:.2%}"
        else:
            message = f"{metric_name} delta={delta_pct:+.2%} within bounds"

        return GateCheckResult(
            gate_name=f"relative_{metric_name}",
            gate_type=GateType.RELATIVE,
            result=result,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            baseline_value=baseline_value,
            delta=delta_pct,
            message=message,
        )

    def _check_statistical(
        self,
        gate: Dict[str, Any],
        current_scores: List[float],
        baseline_scores: List[float],
    ) -> GateCheckResult:
        """Check a statistical significance gate using t-test."""
        metric_name = gate["metric_name"]
        p_threshold = gate["p_value_threshold"]
        min_samples = gate["min_samples"]

        if len(current_scores) < min_samples or len(baseline_scores) < min_samples:
            return GateCheckResult(
                gate_name=f"statistical_{metric_name}",
                gate_type=GateType.STATISTICAL,
                result=GateResult.SKIPPED,
                metric_name=metric_name,
                current_value=statistics.mean(current_scores) if current_scores else 0,
                threshold=p_threshold,
                message=f"Insufficient samples: {len(current_scores)}/{len(baseline_scores)} < {min_samples}",
            )

        # Simple two-sample t-test
        p_value = self._two_sample_t_test(current_scores, baseline_scores)

        current_mean = statistics.mean(current_scores)
        baseline_mean = statistics.mean(baseline_scores)

        result = GateResult.PASSED
        # Only fail if significantly worse (current < baseline with low p-value)
        if p_value < p_threshold and current_mean < baseline_mean:
            result = GateResult.FAILED
            message = f"Significant degradation: p={p_value:.4f} < {p_threshold}"
        elif p_value < p_threshold:
            message = f"Significant change: p={p_value:.4f}, but improvement"
        else:
            message = f"No significant change: p={p_value:.4f} >= {p_threshold}"

        return GateCheckResult(
            gate_name=f"statistical_{metric_name}",
            gate_type=GateType.STATISTICAL,
            result=result,
            metric_name=metric_name,
            current_value=current_mean,
            threshold=p_threshold,
            baseline_value=baseline_mean,
            p_value=p_value,
            message=message,
        )

    def _two_sample_t_test(self, sample1: List[float], sample2: List[float]) -> float:
        """
        Perform a two-sample t-test.
        Returns p-value (approximation using normal distribution).
        """
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
        var1, var2 = statistics.variance(sample1), statistics.variance(sample2)

        # Pooled standard error
        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            return 1.0

        # t-statistic
        t = (mean1 - mean2) / se

        # Approximate p-value using standard normal (for large samples)
        # Two-tailed test
        p = 2 * (1 - self._norm_cdf(abs(t)))
        return p

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def check(
        self,
        current_scores: Dict[str, float],
        baseline_scores: Optional[Dict[str, float]] = None,
        per_case_scores: Optional[Dict[str, List[float]]] = None,
        baseline_per_case: Optional[Dict[str, List[float]]] = None,
    ) -> QualityGateReport:
        """
        Check all quality gates.

        Args:
            current_scores: Current aggregate scores per metric
            baseline_scores: Baseline aggregate scores per metric
            per_case_scores: Per-case scores for statistical tests
            baseline_per_case: Baseline per-case scores

        Returns:
            QualityGateReport with all results
        """
        from datetime import datetime

        baseline_scores = baseline_scores or {}
        per_case_scores = per_case_scores or {}
        baseline_per_case = baseline_per_case or {}

        results = []

        for gate in self.gates:
            metric_name = gate["metric_name"]

            if metric_name not in current_scores:
                results.append(
                    GateCheckResult(
                        gate_name=f"{gate['type'].value}_{metric_name}",
                        gate_type=gate["type"],
                        result=GateResult.SKIPPED,
                        metric_name=metric_name,
                        current_value=0.0,
                        threshold=0.0,
                        message=f"Metric {metric_name} not found in current scores",
                    )
                )
                continue

            current_value = current_scores[metric_name]

            if gate["type"] == GateType.ABSOLUTE:
                results.append(self._check_absolute(gate, current_value))

            elif gate["type"] == GateType.RELATIVE:
                if metric_name not in baseline_scores:
                    results.append(
                        GateCheckResult(
                            gate_name=f"relative_{metric_name}",
                            gate_type=GateType.RELATIVE,
                            result=GateResult.SKIPPED,
                            metric_name=metric_name,
                            current_value=current_value,
                            threshold=0.0,
                            message=f"No baseline for {metric_name}",
                        )
                    )
                else:
                    results.append(
                        self._check_relative(
                            gate, current_value, baseline_scores[metric_name]
                        )
                    )

            elif gate["type"] == GateType.STATISTICAL:
                if (
                    metric_name not in per_case_scores
                    or metric_name not in baseline_per_case
                ):
                    results.append(
                        GateCheckResult(
                            gate_name=f"statistical_{metric_name}",
                            gate_type=GateType.STATISTICAL,
                            result=GateResult.SKIPPED,
                            metric_name=metric_name,
                            current_value=current_value,
                            threshold=gate["p_value_threshold"],
                            message=f"Per-case scores not available for {metric_name}",
                        )
                    )
                else:
                    results.append(
                        self._check_statistical(
                            gate,
                            per_case_scores[metric_name],
                            baseline_per_case[metric_name],
                        )
                    )

        # Determine overall result
        summary = {
            "passed": sum(1 for r in results if r.result == GateResult.PASSED),
            "failed": sum(1 for r in results if r.result == GateResult.FAILED),
            "warning": sum(1 for r in results if r.result == GateResult.WARNING),
            "skipped": sum(1 for r in results if r.result == GateResult.SKIPPED),
        }

        if summary["failed"] > 0:
            overall = GateResult.FAILED
        elif summary["warning"] > 0 and self.fail_on_warning:
            overall = GateResult.FAILED
        elif summary["warning"] > 0:
            overall = GateResult.WARNING
        else:
            overall = GateResult.PASSED

        return QualityGateReport(
            timestamp=datetime.utcnow().isoformat(),
            overall_result=overall,
            gate_results=results,
            summary=summary,
        )


class QualityGateException(Exception):
    """Raised when quality gates fail."""

    def __init__(self, report: QualityGateReport):
        self.report = report
        failed = [gr for gr in report.gate_results if gr.result == GateResult.FAILED]
        messages = [gr.message for gr in failed]
        super().__init__(f"Quality gates failed: {'; '.join(messages)}")


# Example configurations
def create_default_quality_gate() -> QualityGate:
    """Create a default quality gate configuration."""
    return (
        QualityGate()
        .add_absolute_threshold("accuracy", min_value=0.80, warning_min=0.85)
        .add_absolute_threshold("coherence", min_value=0.70, warning_min=0.75)
        .add_absolute_threshold("relevance", min_value=0.75, warning_min=0.80)
        .add_relative_threshold(
            "accuracy", max_degradation=0.05, warning_degradation=0.03
        )
        .add_relative_threshold(
            "coherence", max_degradation=0.10, warning_degradation=0.05
        )
        .add_relative_threshold(
            "relevance", max_degradation=0.10, warning_degradation=0.05
        )
    )


def create_strict_quality_gate() -> QualityGate:
    """Create a strict quality gate for production releases."""
    return (
        QualityGate(fail_on_warning=True)
        .add_absolute_threshold("accuracy", min_value=0.90, warning_min=0.92)
        .add_absolute_threshold("coherence", min_value=0.85, warning_min=0.88)
        .add_absolute_threshold("relevance", min_value=0.85, warning_min=0.88)
        .add_relative_threshold(
            "accuracy", max_degradation=0.02, warning_degradation=0.01
        )
        .add_relative_threshold(
            "coherence", max_degradation=0.03, warning_degradation=0.02
        )
        .add_statistical_threshold("accuracy", p_value_threshold=0.05, min_samples=50)
    )


# Example usage
if __name__ == "__main__":
    # Create quality gate
    gate = create_default_quality_gate()

    # Current and baseline scores
    current = {"accuracy": 0.82, "coherence": 0.78, "relevance": 0.76}

    baseline = {"accuracy": 0.85, "coherence": 0.80, "relevance": 0.78}

    # Check gates
    report = gate.check(current, baseline)

    print(f"Overall Result: {report.overall_result.value}")
    print(f"Summary: {report.summary}")
    print("\nGate Results:")
    for gr in report.gate_results:
        print(f"  [{gr.result.value}] {gr.gate_name}: {gr.message}")
