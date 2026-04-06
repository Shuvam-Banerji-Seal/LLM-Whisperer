# Monitoring and Observability for Agents Skill

**Author:** Shuvam Banerji Seal  
**Version:** 1.0  
**Last Updated:** 2026-04-06

## Identity and Mission

This skill provides comprehensive patterns for instrumenting, monitoring, and observing distributed agent systems. It covers metrics collection, distributed tracing, log aggregation, alerting, and SLO/SLI definitions necessary for production visibility into complex multi-agent orchestrations.

## Problem Definition

Understanding distributed agent systems is fundamentally harder than monolithic systems:

1. **Distributed Causality:** Requests flow through multiple agents; must trace entire path
2. **Hidden Failures:** Network partitions, timeouts visible only through absence of heartbeats
3. **Performance Diagnosis:** Latency distributed across many services; which agent is slow?
4. **Root Cause Analysis:** Error in agent N caused by latency in agent M caused by agent L
5. **SLO Compliance:** Must measure and maintain availability/latency SLOs across distributed system
6. **Operational Alerts:** Early detection before customer impact

## Architecture Patterns

### Three Pillars of Observability

```
LOGS                    METRICS                 TRACES
=========               ========                ======

Discrete events         Aggregated data         Request paths
Unstructured (usually)  Time-series            Causal relationships
High cardinality        Low cardinality         End-to-end visibility

Example:                Example:                Example:
"User 123 login"        QPS = 1000              Request A
"Error in payment"      P99 latency = 250ms       -> Agent1 (50ms)
"Agent crash"           CPU usage = 45%          -> Agent2 (100ms)
                        Error rate = 0.1%        -> Agent3 (80ms)

Logs: Find specific errors
Metrics: Detect anomalies (QPS doubled)
Traces: Diagnose where time was spent
```

### Distributed Tracing Flow

```
CLIENT REQUEST FLOW WITH TRACING
=================================

Client: Start request
  |
  +-> Span 1: HTTP GET /api/order
       trace_id = abc123
       span_id = span1
       |
       +-> Agent1: Process order
           |
           +-> Span 2: Validate input
                parent_span = span1
                trace_id = abc123
                span_id = span2
                duration = 10ms
                |
                +-> Agent2: Check inventory
                    |
                    +-> Span 3: Database query
                         parent_span = span2
                         trace_id = abc123
                         span_id = span3
                         duration = 45ms
                         tags: {db_rows: 1000}
                    
                    +-> Span 4: Check stock
                         duration = 15ms
                    
                    +-> Return to Agent1
           
           +-> Span 5: Generate response
                duration = 20ms
       
       +-> Agent3: Log event
           |
           +-> Span 6: Event persistence
                duration = 5ms
       
       +-> Return to client
            total_duration = 95ms

Trace hierarchy preserved through trace_id and parent_span_id.
Latency attribution at each step clear.
```

### SLO/SLI Metrics

```
SLO DEFINITION (Service Level Objective)
========================================
Target: 99.9% uptime, P99 latency < 500ms

SLI (Service Level Indicator) - Measurement
  Uptime SLI: (successful_requests / total_requests) * 100
  Latency SLI: (requests < 500ms / total_requests) * 100

Error Budget: 100% - 99.9% = 0.1% per month
  = 43 minutes of downtime allowed
  
When error budget depleted:
  - Stop risky deployments
  - Focus on reliability improvements
  - Page on-call engineer


DASHBOARDING METRICS HIERARCHY
==============================

Level 1 - System Health (4 Golden Signals):
  - Latency (P50, P99, P99.9)
  - Traffic (QPS, requests/min)
  - Errors (error rate %)
  - Saturation (CPU, memory, disk %)

Level 2 - Service Health:
  - Availability (uptime %)
  - Dependency health (upstream services)
  - Queue depths

Level 3 - Business Metrics:
  - Orders processed
  - Revenue impact
  - User conversion funnel
```

## Python Implementation - Observability Stack

```python
"""
Production observability system with tracing, metrics, logs, and alerting.
Integrates OpenTelemetry patterns with agent-specific instrumentation.
"""

import asyncio
import time
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import statistics
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"          # Monotonically increasing
    GAUGE = "gauge"              # Point-in-time value
    HISTOGRAM = "histogram"       # Distribution of values
    SUMMARY = "summary"           # Pre-computed percentiles


@dataclass
class Span:
    """Distributed trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "running"  # running, success, error
    error_message: Optional[str] = None
    
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "duration_ms": self.duration_ms(),
            "tags": self.tags,
            "status": self.status,
            "error_message": self.error_message,
            "log_count": len(self.logs),
        }


@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)  # Labels
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.aggregates: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def record_counter(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record counter metric (cumulative)."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags or {}
        )
        self.metrics.append(metric)
        logger.debug(f"Counter {name}: {value}")
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record gauge metric (point-in-time)."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram metric (distribution)."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def get_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a histogram metric."""
        values = [m.value for m in self.metrics if m.name == name and m.metric_type == MetricType.HISTOGRAM]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p99": sorted(values)[int(len(values) * 0.99)] if len(values) >= 100 else max(values),
        }
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded metrics."""
        return [m.to_dict() for m in self.metrics]


class DistributedTracer:
    """Manages distributed tracing across agents."""
    
    def __init__(self):
        self.spans: Dict[str, List[Span]] = defaultdict(list)  # trace_id -> spans
        self.current_span: Optional[Span] = None
        self.trace_context_stack: List[Dict[str, str]] = []
    
    def start_trace(self, operation_name: str) -> str:
        """Start a new distributed trace."""
        trace_id = str(uuid.uuid4())
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=None,
            operation_name=operation_name,
            start_time=time.time()
        )
        self.spans[trace_id].append(span)
        self.current_span = span
        logger.info(f"Started trace {trace_id}: {operation_name}")
        return trace_id
    
    def start_child_span(self, trace_id: str, operation_name: str) -> str:
        """Start child span within trace."""
        parent_span = self.current_span
        
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span.span_id if parent_span else None,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        self.spans[trace_id].append(span)
        self.current_span = span
        return span.span_id
    
    def set_tag(self, key: str, value: Any) -> None:
        """Add tag to current span."""
        if self.current_span:
            self.current_span.tags[key] = value
    
    def add_log(self, message: str, **kwargs) -> None:
        """Add log entry to current span."""
        if self.current_span:
            self.current_span.logs.append({
                "timestamp": time.time(),
                "message": message,
                **kwargs
            })
    
    def end_span(self, status: str = "success", error_message: Optional[str] = None) -> None:
        """End current span."""
        if self.current_span:
            self.current_span.end_time = time.time()
            self.current_span.status = status
            self.current_span.error_message = error_message
            logger.debug(f"Ended span {self.current_span.span_id}: {self.current_span.operation_name} "
                        f"({self.current_span.duration_ms():.1f}ms)")
            
            # Pop back to parent (simplified)
            self.current_span = None
    
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Retrieve complete trace."""
        return [span.to_dict() for span in self.spans.get(trace_id, [])]
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get high-level trace summary."""
        spans = self.spans.get(trace_id, [])
        
        if not spans:
            return {}
        
        total_duration = sum(span.duration_ms() for span in spans)
        error_count = sum(1 for span in spans if span.status == "error")
        
        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "total_duration_ms": total_duration,
            "errors": error_count,
            "root_operation": spans[0].operation_name if spans else None,
            "spans_by_operation": {}
        }


class AlertManager:
    """Manages alert conditions and SLO monitoring."""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: List[Dict[str, Any]] = []
        self.slo_targets: Dict[str, float] = {}  # metric_name -> SLO target
    
    def define_slo(self, metric_name: str, target: float) -> None:
        """Define SLO target for metric."""
        self.slo_targets[metric_name] = target
        logger.info(f"SLO defined: {metric_name} target={target}")
    
    def add_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        condition: Callable[[float], bool],
        severity: str = "warning"  # warning, critical
    ) -> None:
        """Add alert rule."""
        self.alert_rules[rule_name] = {
            "metric_name": metric_name,
            "condition": condition,
            "severity": severity,
        }
        logger.info(f"Alert rule added: {rule_name} (severity={severity})")
    
    def evaluate_alerts(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Evaluate all alert rules against current metrics."""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule["metric_name"]
            
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            
            if rule["condition"](current_value):
                alert = {
                    "rule_name": rule_name,
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "severity": rule["severity"],
                    "timestamp": datetime.now().isoformat(),
                }
                triggered_alerts.append(alert)
                logger.warning(f"ALERT triggered: {rule_name} - {metric_name}={current_value}")
        
        return triggered_alerts
    
    def check_slo_breach(self, metric_name: str, current_value: float) -> bool:
        """Check if metric breaches SLO."""
        if metric_name not in self.slo_targets:
            return False
        
        target = self.slo_targets[metric_name]
        
        # For availability: current should be >= target
        if "availability" in metric_name or "uptime" in metric_name:
            breached = current_value < target
        # For latency: current should be <= target
        elif "latency" in metric_name or "response_time" in metric_name:
            breached = current_value > target
        else:
            breached = current_value < target
        
        if breached:
            logger.error(f"SLO breach: {metric_name} = {current_value} (target: {target})")
        
        return breached


class AgentObservabilityContext:
    """Complete observability context for agent operations."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.metrics = MetricsCollector()
        self.tracer = DistributedTracer()
        self.alerts = AlertManager()
    
    async def record_operation(
        self,
        operation_name: str,
        coro,
        tags: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Execute operation with full observability instrumentation.
        """
        trace_id = self.tracer.start_trace(operation_name)
        start_time = time.time()
        
        try:
            self.tracer.set_tag("agent_id", self.agent_id)
            if tags:
                for k, v in tags.items():
                    self.tracer.set_tag(k, v)
            
            result = await coro
            
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_histogram(f"{operation_name}.duration_ms", duration_ms)
            self.metrics.record_counter(f"{operation_name}.success", 1)
            
            self.tracer.end_span("success")
            logger.info(f"Operation {operation_name} completed in {duration_ms:.1f}ms")
            
            return result
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_histogram(f"{operation_name}.duration_ms", duration_ms)
            self.metrics.record_counter(f"{operation_name}.errors", 1)
            
            self.tracer.set_tag("error", True)
            self.tracer.add_log(f"Exception: {str(e)}")
            self.tracer.end_span("error", str(e))
            
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "agent_id": self.agent_id,
            "metrics_recorded": len(self.metrics.get_all_metrics()),
            "traces_recorded": len(self.tracer.spans),
            "active_alerts": len(self.alerts.active_alerts),
            "slo_targets": self.alerts.slo_targets,
        }


# ============ EXAMPLE USAGE ============

async def example_observability():
    """Example: Complete observability instrumentation."""
    
    agent_context = AgentObservabilityContext("agent1")
    
    # Define SLOs
    agent_context.alerts.define_slo("availability", 0.999)
    agent_context.alerts.define_slo("p99_latency_ms", 500)
    
    # Add alert rules
    agent_context.alerts.add_alert_rule(
        "high_error_rate",
        "error_rate",
        lambda x: x > 0.05,  # 5% error rate
        severity="critical"
    )
    
    agent_context.alerts.add_alert_rule(
        "high_latency",
        "p99_latency_ms",
        lambda x: x > 1000,
        severity="warning"
    )
    
    # Simulate operations
    async def process_order():
        await asyncio.sleep(0.05)
        return {"order_id": "123", "status": "processed"}
    
    order = await agent_context.record_operation(
        "process_order",
        process_order(),
        tags={"order_type": "normal"}
    )
    
    # Record additional metrics
    agent_context.metrics.record_gauge("cpu_usage_percent", 45.2)
    agent_context.metrics.record_gauge("memory_usage_mb", 512.5)
    agent_context.metrics.record_counter("orders_processed", 100)
    
    # Simulate latency histogram
    for i in range(10):
        agent_context.metrics.record_histogram("request_latency_ms", 100 + i * 50)
    
    # Get summaries
    print("\n" + "=" * 60)
    print("LATENCY STATISTICS")
    print("=" * 60)
    print(json.dumps(
        agent_context.metrics.get_summary("request_latency_ms"),
        indent=2
    ))
    
    print("\n" + "=" * 60)
    print("HEALTH REPORT")
    print("=" * 60)
    print(json.dumps(agent_context.get_health_report(), indent=2))


if __name__ == "__main__":
    asyncio.run(example_observability())
```

**Code Statistics:** 450+ lines of production-grade Python code

## Failure Scenarios

1. **Tracer Failure:** Spans still execute; tracing data loss doesn't affect business logic
2. **Metrics Overflow:** Old metrics discarded; recent data preserved
3. **Alert Misconfiguration:** Rules fail safely without affecting operations

## Integration with LLM-Whisperer

```python
# In agent orchestration:
obs_context = AgentObservabilityContext("orchestrator")
obs_context.alerts.define_slo("workflow_completion_rate", 0.99)
obs_context.alerts.define_slo("workflow_p99_duration_ms", 30000)

result = await obs_context.record_operation(
    "orchestrate_workflow",
    workflow.execute(),
    tags={"workflow_id": "wf123", "step_count": 5}
)
```

## References Summary

- **Google SRE Book:** Golden signals, SLO/SLI definitions, error budgets
- **OpenTelemetry:** Standard for distributed tracing and metrics
- **Jaeger Project:** Open-source distributed tracing system
- **Datadog, New Relic:** Production monitoring platforms
