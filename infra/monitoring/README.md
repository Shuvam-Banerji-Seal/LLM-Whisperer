# Monitoring Module

Monitoring and observability tools for LLM-Whisperer.

## Overview

The Monitoring module provides comprehensive observability for LLM-Whisperer applications, including:

- **MetricsCollector**: Collect, store, and aggregate application metrics
- **AlertManager**: Create and manage alert rules and notifications
- **Dashboard**: Create and manage monitoring dashboards

## Main Classes

### MetricsCollector

Collects and manages application metrics with support for different metric types.

```python
from infra.monitoring.core import MetricsCollector
from infra.monitoring.config import MonitoringConfig, MetricType

config = MonitoringConfig()
collector = MetricsCollector(config)

# Record metrics
collector.record_metric(
    "api_latency_ms",
    150.5,
    metric_type=MetricType.HISTOGRAM,
    tags={"endpoint": "/api/v1", "method": "GET"}
)

# Get current metric
latency = collector.get_metric("api_latency_ms")

# Get metrics matching pattern
api_metrics = collector.get_metrics(pattern="api_")

# Get metric history
history = collector.get_metric_history("api_latency_ms", limit=100)

# Aggregate metrics
avg_latency = collector.aggregate_metrics("api_latency", aggregation="avg")
```

**Key Methods:**
- `record_metric(name, value, metric_type, tags, timestamp)`: Record a metric
- `get_metric(name)`: Get current metric value
- `get_metrics(pattern)`: Get metrics matching pattern
- `get_metric_history(name, limit)`: Get historical metric values
- `aggregate_metrics(pattern, aggregation)`: Aggregate metrics
- `clear_metrics(pattern)`: Clear metrics

### AlertManager

Create and manage alert rules with notification support.

```python
from infra.monitoring.core import AlertManager
from infra.monitoring.config import MonitoringConfig, AlertSeverity

config = MonitoringConfig()
alert_manager = AlertManager(config)

# Create alert rule
alert_manager.create_alert_rule(
    name="high_api_latency",
    condition="api_latency_ms > 1000",
    threshold=1000.0,
    severity=AlertSeverity.WARNING,
    duration_seconds=300
)

# Trigger alert
alert = alert_manager.trigger_alert(
    rule_name="high_api_latency",
    value=1500.0,
    message="API latency exceeded threshold"
)

# Resolve alert
resolved = alert_manager.resolve_alert(alert["id"])

# Get active alerts
active = alert_manager.get_active_alerts()
severe_alerts = alert_manager.get_active_alerts(severity=AlertSeverity.CRITICAL)

# Get alert history
history = alert_manager.get_alert_history(rule_name="high_api_latency", limit=50)

# List alert rules
rules = alert_manager.list_alert_rules()

# Disable alert rule
alert_manager.disable_alert_rule("high_api_latency")
```

**Key Methods:**
- `create_alert_rule(name, condition, threshold, severity, duration_seconds)`: Create rule
- `trigger_alert(rule_name, value, message)`: Trigger alert
- `resolve_alert(alert_id)`: Resolve alert
- `get_active_alerts(severity)`: Get active alerts
- `get_alert_history(rule_name, limit)`: Get alert history
- `list_alert_rules()`: List all rules
- `disable_alert_rule(name)`: Disable rule

### Dashboard

Create and manage monitoring dashboards.

```python
from infra.monitoring.core import Dashboard
from infra.monitoring.config import MonitoringConfig, DashboardConfig

config = MonitoringConfig()
dashboard_mgr = Dashboard(config)

# Create dashboard
dashboard_config = DashboardConfig(
    name="API Metrics",
    backend="grafana",
    time_range="1h",
    tags=["api", "production"]
)
dashboard = dashboard_mgr.create_dashboard(dashboard_config)

# Add panel to dashboard
panel_config = {
    "title": "Request Latency",
    "metrics": ["api_latency_ms"],
    "type": "graph"
}
dashboard_mgr.add_panel("API Metrics", "latency-panel", panel_config)

# Get dashboard
dashboard = dashboard_mgr.get_dashboard("API Metrics")

# List dashboards
all_dashboards = dashboard_mgr.list_dashboards()
api_dashboards = dashboard_mgr.list_dashboards(tag="api")

# Export dashboard configuration
config = dashboard_mgr.export_dashboard("API Metrics")

# Delete dashboard
deleted = dashboard_mgr.delete_dashboard("API Metrics")
```

**Key Methods:**
- `create_dashboard(dashboard_config)`: Create dashboard
- `add_panel(dashboard_name, panel_id, panel_config)`: Add panel
- `get_dashboard(name)`: Get dashboard
- `list_dashboards(tag)`: List dashboards
- `delete_dashboard(name)`: Delete dashboard
- `export_dashboard(name)`: Export configuration

## Configuration

### MonitoringConfig

Main monitoring configuration.

```python
from infra.monitoring.config import MonitoringConfig

config = MonitoringConfig(
    enabled=True,
    metrics=MetricsConfig(
        enabled=True,
        interval_seconds=60,
        backend="prometheus"
    ),
    alerts=AlertConfig(
        enabled=True,
        evaluation_interval_seconds=60
    ),
    retention_days=30,
    enable_distributed_tracing=True
)
```

**Fields:**
- `enabled`: Enable monitoring (default: True)
- `metrics`: Metrics configuration
- `alerts`: Alert configuration
- `dashboards`: Dashboard configurations
- `log_level`: Log level (default: "INFO")
- `storage_backend`: Storage backend (default: "tsdb")
- `retention_days`: Data retention in days
- `enable_distributed_tracing`: Enable distributed tracing
- `jaeger_endpoint`: Jaeger endpoint for tracing

### MetricsConfig

Metrics collection configuration.

```python
from infra.monitoring.config import MetricsConfig

config = MetricsConfig(
    enabled=True,
    interval_seconds=60,
    retention_days=30,
    backend="prometheus",
    metrics_prefix="llm_whisperer",
    scrape_interval="15s"
)
```

### AlertConfig

Alert configuration.

```python
from infra.monitoring.config import AlertConfig, AlertAction

config = AlertConfig(
    enabled=True,
    evaluation_interval_seconds=60,
    default_severity=AlertSeverity.WARNING,
    actions=[AlertAction.EMAIL, AlertAction.SLACK],
    email_recipients=["ops@example.com"],
    slack_webhook="https://hooks.slack.com/services/..."
)
```

### DashboardConfig

Dashboard configuration.

```python
from infra.monitoring.config import DashboardConfig

config = DashboardConfig(
    name="Performance Metrics",
    backend="grafana",
    url="https://grafana.example.com",
    auto_refresh_seconds=30,
    time_range="1h",
    tags=["performance", "production"]
)
```

## Error Handling

All classes validate input and raise `ValueError` for invalid configurations:

```python
try:
    collector.record_metric("", 100.0)
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Logging

Enable detailed logging for debugging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

## Metric Types

- **COUNTER**: Monotonically increasing counter
- **GAUGE**: Instantaneous measurement
- **HISTOGRAM**: Distribution of values
- **SUMMARY**: Quantiles of distributions

## Alert Severity

- **CRITICAL**: Immediate action required
- **WARNING**: Warning, investigation needed
- **INFO**: Informational, no action required

## Example: Complete Monitoring Setup

```python
from infra.monitoring.core import MetricsCollector, AlertManager, Dashboard
from infra.monitoring.config import (
    MonitoringConfig,
    MetricsConfig,
    AlertConfig,
    DashboardConfig,
    MetricType,
    AlertSeverity,
)

# Configure monitoring
monitoring_config = MonitoringConfig(
    metrics=MetricsConfig(retention_days=30),
    alerts=AlertConfig(
        email_recipients=["ops@example.com"],
        evaluation_interval_seconds=60
    )
)

# Set up metrics collector
collector = MetricsCollector(monitoring_config)

# Record application metrics
collector.record_metric(
    "request_count",
    1000.0,
    metric_type=MetricType.COUNTER,
    tags={"method": "GET", "path": "/api/v1"}
)
collector.record_metric(
    "response_latency_ms",
    150.5,
    metric_type=MetricType.HISTOGRAM
)
collector.record_metric(
    "gpu_utilization_percent",
    75.0,
    metric_type=MetricType.GAUGE
)

# Set up alert rules
alert_manager = AlertManager(monitoring_config)
alert_manager.create_alert_rule(
    name="high_latency",
    condition="response_latency_ms > 500",
    threshold=500.0,
    severity=AlertSeverity.WARNING
)
alert_manager.create_alert_rule(
    name="high_gpu_usage",
    condition="gpu_utilization_percent > 90",
    threshold=90.0,
    severity=AlertSeverity.CRITICAL
)

# Create dashboards
dashboard_mgr = Dashboard(monitoring_config)
api_dashboard = DashboardConfig(
    name="API Metrics",
    tags=["api", "production"],
    time_range="1h"
)
dashboard_mgr.create_dashboard(api_dashboard)

# Monitor metrics periodically
metrics = collector.get_metrics()
avg_latency = collector.aggregate_metrics("response_latency", "avg")
active_alerts = alert_manager.get_active_alerts()

print(f"Average latency: {avg_latency}ms")
print(f"Active alerts: {len(active_alerts)}")
```

## Testing

Run the module directly for basic examples:

```bash
python -m infra.monitoring.core
```

## Performance Considerations

- **Metric Retention**: Balance retention_days with storage capacity
- **Aggregation**: Use aggregation for high-volume metrics
- **Alert Evaluation**: Optimize evaluation_interval_seconds
- **Dashboard Refresh**: Use appropriate auto_refresh_seconds

## See Also

- [Docker Module](../docker/README.md)
- [Kubernetes Module](../kubernetes/README.md)
- [Terraform Module](../terraform/README.md)
