"""
Complete LLM Production Monitoring & Observability Example
==========================================================

Demonstrates:
- Prometheus metrics collection
- Grafana-compatible metrics
- OpenTelemetry tracing
- Cost tracking and optimization
- Real-time dashboards
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
)
import logging


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class MonitoringConfig:
    """Configuration for monitoring"""

    enable_prometheus: bool = True
    enable_opentelemetry: bool = True
    metrics_port: int = 8000
    alert_latency_p95_ms: float = 2000.0
    alert_error_rate: float = 0.05
    cost_per_million_tokens: float = 0.01  # $0.01 per million tokens


# ============================================================================
# Prometheus Metrics Setup
# ============================================================================


class PrometheusMetrics:
    """Prometheus metrics for LLM inference"""

    def __init__(
        self, config: MonitoringConfig, registry: Optional[CollectorRegistry] = None
    ):
        if registry is None:
            registry = CollectorRegistry()

        self.config = config
        self.registry = registry

        # Request metrics
        self.request_count = Counter(
            "llm_requests_total",
            "Total number of inference requests",
            ["model", "status", "endpoint"],
            registry=registry,
        )

        self.request_duration = Histogram(
            "llm_request_duration_seconds",
            "Inference request latency (seconds)",
            ["model", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=registry,
        )

        self.tokens_generated = Counter(
            "llm_tokens_generated_total",
            "Total tokens generated",
            ["model"],
            registry=registry,
        )

        self.input_tokens = Counter(
            "llm_input_tokens_total", "Total input tokens", ["model"], registry=registry
        )

        # Resource metrics
        self.gpu_memory_used = Gauge(
            "gpu_memory_used_bytes",
            "GPU memory used in bytes",
            ["gpu_id", "model"],
            registry=registry,
        )

        self.gpu_memory_available = Gauge(
            "gpu_memory_available_bytes",
            "GPU memory available in bytes",
            ["gpu_id"],
            registry=registry,
        )

        self.gpu_utilization = Gauge(
            "gpu_utilization_percent",
            "GPU utilization percentage",
            ["gpu_id"],
            registry=registry,
        )

        # Queue metrics
        self.queue_length = Gauge(
            "llm_queue_length",
            "Number of pending inference requests",
            registry=registry,
        )

        self.queue_wait_time = Histogram(
            "llm_queue_wait_seconds",
            "Time spent waiting in queue",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
            registry=registry,
        )

        # Error metrics
        self.errors = Counter(
            "llm_errors_total",
            "Total inference errors",
            ["model", "error_type"],
            registry=registry,
        )

        self.error_rate = Gauge(
            "llm_error_rate", "Current error rate", registry=registry
        )

        # Cost metrics
        self.cost_total = Counter(
            "llm_cost_usd_total",
            "Cumulative inference cost in USD",
            ["model", "gpu_type"],
            registry=registry,
        )

        self.cost_per_token = Gauge(
            "llm_cost_per_million_tokens",
            "Cost per million tokens (USD)",
            ["model"],
            registry=registry,
        )

        # Model metrics
        self.model_load_time = Histogram(
            "llm_model_load_seconds", "Time to load model into GPU", registry=registry
        )

        self.active_models = Gauge(
            "llm_active_models", "Number of actively loaded models", registry=registry
        )

    def get_metrics_as_text(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry).decode("utf-8")


# ============================================================================
# Metrics Aggregator (for local tracking)
# ============================================================================


@dataclass
class RequestMetrics:
    """Metrics for a single request"""

    request_id: str
    model: str
    status: str  # 'success' or 'error'
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    gpu_memory_used_mb: float = 0.0
    error: Optional[str] = None

    def calculate_metrics(self):
        """Calculate derived metrics"""
        if self.end_time:
            self.latency_ms = (self.end_time - self.start_time).total_seconds() * 1000


class MetricsAggregator:
    """Aggregate metrics over time windows"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)

    def record(self, metric: RequestMetrics):
        """Record a request metric"""
        metric.calculate_metrics()
        self.metrics.append(metric)

    def get_stats(self) -> Dict:
        """Get aggregated statistics"""
        if not self.metrics:
            return {}

        latencies = [m.latency_ms for m in self.metrics]
        latencies.sort()

        successful = [m for m in self.metrics if m.status == "success"]
        errors = [m for m in self.metrics if m.status == "error"]

        return {
            "total_requests": len(self.metrics),
            "successful_requests": len(successful),
            "failed_requests": len(errors),
            "error_rate": len(errors) / len(self.metrics) if self.metrics else 0,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": latencies[len(latencies) // 2],
            "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "total_input_tokens": sum(m.input_tokens for m in self.metrics),
            "total_output_tokens": sum(m.output_tokens for m in self.metrics),
            "total_cost_usd": sum(m.cost_usd for m in self.metrics),
            "avg_cost_per_request": sum(m.cost_usd for m in self.metrics)
            / len(self.metrics)
            if self.metrics
            else 0,
        }


# ============================================================================
# OpenTelemetry Tracing
# ============================================================================


class TraceCollector:
    """Collect traces for distributed tracing"""

    def __init__(self):
        self.traces: deque = deque(maxlen=10000)
        self.logger = logging.getLogger(__name__)

    def record_trace(
        self,
        trace_id: str,
        span_name: str,
        duration_ms: float,
        attributes: Optional[Dict] = None,
    ):
        """Record a trace span"""
        trace = {
            "trace_id": trace_id,
            "span_name": span_name,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
        }
        self.traces.append(trace)


# ============================================================================
# Alert Management
# ============================================================================


@dataclass
class Alert:
    """Alert definition"""

    name: str
    condition: str  # e.g., "p95_latency > 2000"
    severity: str  # 'warning' or 'critical'
    threshold: float
    current_value: float = 0.0
    triggered: bool = False
    timestamp: Optional[datetime] = None


class AlertManager:
    """Manage and trigger alerts"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alerts: Dict[str, Alert] = {
            "high_latency": Alert(
                name="High Latency",
                condition="p95_latency_ms > threshold",
                severity="warning",
                threshold=config.alert_latency_p95_ms,
            ),
            "high_error_rate": Alert(
                name="High Error Rate",
                condition="error_rate > threshold",
                severity="critical",
                threshold=config.alert_error_rate,
            ),
        }
        self.logger = logging.getLogger(__name__)

    def check_alerts(self, stats: Dict) -> List[Alert]:
        """Check if any alerts should be triggered"""
        triggered = []

        # Check latency
        p95_latency = stats.get("p95_latency_ms", 0)
        if p95_latency > self.alerts["high_latency"].threshold:
            self.alerts["high_latency"].triggered = True
            self.alerts["high_latency"].current_value = p95_latency
            self.alerts["high_latency"].timestamp = datetime.now()
            triggered.append(self.alerts["high_latency"])
            self.logger.warning(
                f"ALERT: P95 Latency {p95_latency}ms exceeds threshold {self.alerts['high_latency'].threshold}ms"
            )

        # Check error rate
        error_rate = stats.get("error_rate", 0)
        if error_rate > self.alerts["high_error_rate"].threshold:
            self.alerts["high_error_rate"].triggered = True
            self.alerts["high_error_rate"].current_value = error_rate
            self.alerts["high_error_rate"].timestamp = datetime.now()
            triggered.append(self.alerts["high_error_rate"])
            self.logger.critical(
                f"ALERT: Error Rate {error_rate:.2%} exceeds threshold {self.alerts['high_error_rate'].threshold:.2%}"
            )

        return triggered


# ============================================================================
# Dashboard JSON Generator
# ============================================================================


def generate_grafana_dashboard(title: str = "LLM Inference Dashboard") -> Dict:
    """Generate Grafana dashboard JSON"""

    dashboard = {
        "dashboard": {
            "title": title,
            "panels": [
                {
                    "title": "Requests per Second",
                    "targets": [{"expr": "rate(llm_requests_total[1m])"}],
                    "type": "graph",
                },
                {
                    "title": "P95 Latency (ms)",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, llm_request_duration_seconds) * 1000"
                        }
                    ],
                    "type": "graph",
                },
                {
                    "title": "Error Rate",
                    "targets": [
                        {
                            "expr": "rate(llm_errors_total[5m]) / rate(llm_requests_total[5m])"
                        }
                    ],
                    "type": "gauge",
                },
                {
                    "title": "GPU Memory Usage",
                    "targets": [{"expr": "gpu_memory_used_bytes / 1e9"}],
                    "type": "gauge",
                    "unit": "GB",
                },
                {
                    "title": "GPU Utilization",
                    "targets": [{"expr": "gpu_utilization_percent"}],
                    "type": "gauge",
                    "unit": "percent",
                },
                {
                    "title": "Queue Length",
                    "targets": [{"expr": "llm_queue_length"}],
                    "type": "stat",
                },
                {
                    "title": "Cost per Million Tokens",
                    "targets": [{"expr": "llm_cost_per_million_tokens"}],
                    "type": "stat",
                },
                {
                    "title": "Tokens Generated",
                    "targets": [{"expr": "rate(llm_tokens_generated_total[1m])"}],
                    "type": "graph",
                },
            ],
        }
    }

    return dashboard


# ============================================================================
# Complete Monitoring System
# ============================================================================


class LLMMonitoringSystem:
    """Complete monitoring system for LLM inference"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.prometheus = PrometheusMetrics(config)
        self.aggregator = MetricsAggregator()
        self.tracer = TraceCollector()
        self.alerts = AlertManager(config)
        self.logger = logging.getLogger(__name__)

        logging.basicConfig(level=logging.INFO)

    def record_inference(self, metric: RequestMetrics):
        """Record an inference request"""
        self.aggregator.record(metric)

        # Update Prometheus
        self.prometheus.request_count.labels(
            model=metric.model, status=metric.status, endpoint="v1/completions"
        ).inc()

        self.prometheus.request_duration.labels(
            model=metric.model, endpoint="v1/completions"
        ).observe(metric.latency_ms / 1000.0)

        if metric.output_tokens > 0:
            self.prometheus.tokens_generated.labels(model=metric.model).inc(
                metric.output_tokens
            )

        if metric.cost_usd > 0:
            self.prometheus.cost_total.labels(model=metric.model, gpu_type="h100").inc(
                metric.cost_usd
            )

        # Check alerts
        stats = self.aggregator.get_stats()
        triggered = self.alerts.check_alerts(stats)

        if triggered:
            for alert in triggered:
                self._send_alert(alert)

    def _send_alert(self, alert: Alert):
        """Send alert (integrate with Slack, PagerDuty, etc.)"""
        self.logger.warning(f"ALERT TRIGGERED: {alert.name} ({alert.severity})")

    def get_dashboard_json(self) -> Dict:
        """Get Grafana dashboard JSON"""
        return generate_grafana_dashboard()

    def get_metrics_summary(self) -> Dict:
        """Get current metrics summary"""
        return {
            "aggregated_stats": self.aggregator.get_stats(),
            "prometheus_metrics": self.prometheus.get_metrics_as_text(),
            "alerts": {
                k: {"triggered": v.triggered, "value": v.current_value}
                for k, v in self.alerts.alerts.items()
            },
        }


# ============================================================================
# Example Usage
# ============================================================================


def example_monitoring():
    """Example monitoring setup"""

    config = MonitoringConfig(
        alert_latency_p95_ms=2000.0, alert_error_rate=0.05, cost_per_million_tokens=0.01
    )

    system = LLMMonitoringSystem(config)

    # Simulate inference requests
    import random

    for i in range(100):
        metric = RequestMetrics(
            request_id=f"req-{i}",
            model="llama-7b",
            status="success" if random.random() > 0.02 else "error",
            input_tokens=random.randint(50, 500),
            output_tokens=random.randint(100, 512),
            latency_ms=random.gauss(500, 200),
            gpu_memory_used_mb=random.randint(4000, 8000),
        )
        metric.end_time = metric.start_time + timedelta(milliseconds=metric.latency_ms)
        metric.cost_usd = (metric.output_tokens / 1_000_000) * 0.01

        system.record_inference(metric)

    # Print summary
    summary = system.get_metrics_summary()
    print(json.dumps(summary["aggregated_stats"], indent=2, default=str))

    # Print Grafana dashboard
    dashboard = system.get_dashboard_json()
    print("\nGrafana Dashboard Available:")
    print(json.dumps(dashboard, indent=2))


if __name__ == "__main__":
    example_monitoring()
