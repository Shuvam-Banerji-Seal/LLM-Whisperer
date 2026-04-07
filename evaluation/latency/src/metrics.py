"""
Latency Evaluation Framework

Comprehensive performance benchmarking for LLM systems:
- TTFT: Time to First Token
- TPOT: Time Per Output Token
- ITL: Inter-Token Latency
- Throughput: Requests per second
- Goodput: Successful tokens per second
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import time
import statistics
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RequestTrace:
    """Timing trace for a single request."""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float  # Time to first token
    tpot_ms: float  # Time per output token
    total_time_ms: float
    success: bool = True
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "total_time_ms": self.total_time_ms,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class LatencyMetrics:
    """Aggregated latency metrics."""

    name: str
    num_requests: int

    # TTFT metrics (ms)
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float

    # TPOT metrics (ms)
    tpot_mean: float
    tpot_p50: float
    tpot_p95: float
    tpot_p99: float

    # Throughput metrics
    throughput_req_per_sec: float
    throughput_tokens_per_sec: float
    goodput_tokens_per_sec: float  # Successful tokens/sec

    # Reliability
    success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_requests": self.num_requests,
            "ttft": {
                "mean": self.ttft_mean,
                "p50": self.ttft_p50,
                "p95": self.ttft_p95,
                "p99": self.ttft_p99,
            },
            "tpot": {
                "mean": self.tpot_mean,
                "p50": self.tpot_p50,
                "p95": self.tpot_p95,
                "p99": self.tpot_p99,
            },
            "throughput": {
                "req_per_sec": self.throughput_req_per_sec,
                "tokens_per_sec": self.throughput_tokens_per_sec,
                "goodput_tokens_per_sec": self.goodput_tokens_per_sec,
            },
            "success_rate": self.success_rate,
        }


class LatencyMetricsComputer:
    """Compute latency metrics from request traces."""

    @staticmethod
    def percentile(values: List[float], percentile: int) -> float:
        """Compute percentile (0-100)."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100.0)
        return sorted_values[min(index, len(sorted_values) - 1)]

    @staticmethod
    def compute_metrics(traces: List[RequestTrace]) -> LatencyMetrics:
        """Compute all latency metrics from traces."""
        if not traces:
            return LatencyMetrics(
                name="empty",
                num_requests=0,
                ttft_mean=0,
                ttft_p50=0,
                ttft_p95=0,
                ttft_p99=0,
                tpot_mean=0,
                tpot_p50=0,
                tpot_p95=0,
                tpot_p99=0,
                throughput_req_per_sec=0,
                throughput_tokens_per_sec=0,
                goodput_tokens_per_sec=0,
                success_rate=0,
            )

        # Extract metrics
        ttft_values = [t.ttft_ms for t in traces if t.success]
        tpot_values = [t.tpot_ms for t in traces if t.success]
        total_times = [t.total_time_ms for t in traces if t.success]

        # Success rate
        success_count = sum(1 for t in traces if t.success)
        success_rate = success_count / len(traces) if traces else 0.0

        # Total tokens
        total_prompt_tokens = sum(t.prompt_tokens for t in traces)
        total_completion_tokens = sum(t.completion_tokens for t in traces if t.success)
        total_time_sec = sum(total_times) / 1000.0 if total_times else 1.0

        # Compute percentiles
        ttft_p50 = LatencyMetricsComputer.percentile(ttft_values, 50)
        ttft_p95 = LatencyMetricsComputer.percentile(ttft_values, 95)
        ttft_p99 = LatencyMetricsComputer.percentile(ttft_values, 99)

        tpot_p50 = LatencyMetricsComputer.percentile(tpot_values, 50)
        tpot_p95 = LatencyMetricsComputer.percentile(tpot_values, 95)
        tpot_p99 = LatencyMetricsComputer.percentile(tpot_values, 99)

        return LatencyMetrics(
            name="benchmark",
            num_requests=len(traces),
            ttft_mean=statistics.mean(ttft_values) if ttft_values else 0,
            ttft_p50=ttft_p50,
            ttft_p95=ttft_p95,
            ttft_p99=ttft_p99,
            tpot_mean=statistics.mean(tpot_values) if tpot_values else 0,
            tpot_p50=tpot_p50,
            tpot_p95=tpot_p95,
            tpot_p99=tpot_p99,
            throughput_req_per_sec=len(traces) / total_time_sec
            if total_time_sec > 0
            else 0,
            throughput_tokens_per_sec=(total_prompt_tokens + total_completion_tokens)
            / total_time_sec
            if total_time_sec > 0
            else 0,
            goodput_tokens_per_sec=total_completion_tokens / total_time_sec
            if total_time_sec > 0
            else 0,
            success_rate=success_rate,
        )


class SLAChecker:
    """Check if metrics meet SLA requirements."""

    @dataclass
    class SLAThresholds:
        """SLA thresholds."""

        ttft_p99_ms: Optional[float] = None
        tpot_p99_ms: Optional[float] = None
        throughput_min_req_per_sec: Optional[float] = None
        goodput_min_tokens_per_sec: Optional[float] = None
        success_rate_min: Optional[float] = None

    def __init__(self, thresholds: SLAThresholds):
        self.thresholds = thresholds

    def check(self, metrics: LatencyMetrics) -> Dict[str, bool]:
        """Check if metrics pass SLA."""
        results = {}

        if self.thresholds.ttft_p99_ms is not None:
            results["ttft_p99"] = metrics.ttft_p99 <= self.thresholds.ttft_p99_ms

        if self.thresholds.tpot_p99_ms is not None:
            results["tpot_p99"] = metrics.tpot_p99 <= self.thresholds.tpot_p99_ms

        if self.thresholds.throughput_min_req_per_sec is not None:
            results["throughput"] = (
                metrics.throughput_req_per_sec
                >= self.thresholds.throughput_min_req_per_sec
            )

        if self.thresholds.goodput_min_tokens_per_sec is not None:
            results["goodput"] = (
                metrics.goodput_tokens_per_sec
                >= self.thresholds.goodput_min_tokens_per_sec
            )

        if self.thresholds.success_rate_min is not None:
            results["success_rate"] = (
                metrics.success_rate >= self.thresholds.success_rate_min
            )

        return results

    def all_passed(self, metrics: LatencyMetrics) -> bool:
        """Check if all SLAs passed."""
        results = self.check(metrics)
        return all(results.values()) if results else True
