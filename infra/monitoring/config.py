"""Monitoring configuration dataclasses."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertAction(str, Enum):
    """Alert actions."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"


@dataclass
class MetricsConfig:
    """Metrics collection configuration."""

    enabled: bool = True
    interval_seconds: int = 60
    retention_days: int = 30
    backend: str = "prometheus"
    push_gateway_url: Optional[str] = None
    scrape_interval: str = "15s"
    scrape_timeout: str = "10s"
    metrics_prefix: str = "llm_whisperer"

    def __post_init__(self):
        """Validate metrics configuration."""
        if self.interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, got {self.interval_seconds}"
            )
        if self.retention_days <= 0:
            raise ValueError(
                f"retention_days must be positive, got {self.retention_days}"
            )


@dataclass
class AlertConfig:
    """Alert configuration."""

    enabled: bool = True
    evaluation_interval_seconds: int = 60
    alert_timeout_seconds: int = 300
    default_severity: AlertSeverity = AlertSeverity.WARNING
    silence_duration_minutes: int = 30
    repeat_interval_minutes: int = 60
    actions: List[AlertAction] = field(default_factory=lambda: [AlertAction.EMAIL])
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate alert configuration."""
        if self.evaluation_interval_seconds <= 0:
            raise ValueError(
                f"evaluation_interval_seconds must be positive, got {self.evaluation_interval_seconds}"
            )
        if self.alert_timeout_seconds <= 0:
            raise ValueError(
                f"alert_timeout_seconds must be positive, got {self.alert_timeout_seconds}"
            )


@dataclass
class DashboardConfig:
    """Dashboard configuration."""

    name: str
    backend: str = "grafana"
    url: Optional[str] = None
    auto_refresh_seconds: int = 30
    time_range: str = "1h"
    panels: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    refresh_interval: str = "30s"

    def __post_init__(self):
        """Validate dashboard configuration."""
        if not self.name:
            raise ValueError("Dashboard name must be provided")
        if self.auto_refresh_seconds < 0:
            raise ValueError(
                f"auto_refresh_seconds must be non-negative, got {self.auto_refresh_seconds}"
            )


@dataclass
class MonitoringConfig:
    """Main monitoring configuration."""

    enabled: bool = True
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    dashboards: Dict[str, DashboardConfig] = field(default_factory=dict)
    log_level: str = "INFO"
    storage_backend: str = "tsdb"
    retention_days: int = 30
    enable_distributed_tracing: bool = True
    jaeger_endpoint: Optional[str] = None

    def __post_init__(self):
        """Validate monitoring configuration."""
        if self.retention_days <= 0:
            raise ValueError(
                f"retention_days must be positive, got {self.retention_days}"
            )
