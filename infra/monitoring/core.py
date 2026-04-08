"""Monitoring and observability core functionality."""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
from datetime import datetime
from infra.monitoring.config import (
    MonitoringConfig,
    MetricsConfig,
    AlertConfig,
    DashboardConfig,
    MetricType,
    AlertSeverity,
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self, config: MonitoringConfig):
        """Initialize metrics collector.

        Args:
            config: Monitoring configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("Monitoring configuration must be provided")
        self.config = config
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.metric_history: Dict[str, List[Tuple[float, str]]] = {}
        logger.debug("Initialized MetricsCollector")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record a metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional metric tags
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Recorded metric metadata

        Raises:
            ValueError: If metric name is invalid
        """
        if not name:
            raise ValueError("Metric name must be provided")

        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()

        metric_key = f"{self.config.metrics.metrics_prefix}_{name}"

        self.metrics[metric_key] = {
            "name": name,
            "value": value,
            "type": metric_type.value,
            "tags": tags or {},
            "timestamp": timestamp,
        }

        if metric_key not in self.metric_history:
            self.metric_history[metric_key] = []
        self.metric_history[metric_key].append((value, self._get_timestamp()))

        logger.debug(f"Recorded metric: {metric_key} = {value}")

        return self.metrics[metric_key]

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get current metric value.

        Args:
            name: Metric name

        Returns:
            Metric data or None if not found
        """
        metric_key = f"{self.config.metrics.metrics_prefix}_{name}"
        return self.metrics.get(metric_key)

    def get_metrics(self, pattern: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get metrics matching pattern.

        Args:
            pattern: Optional pattern to filter metrics

        Returns:
            Dictionary of matching metrics
        """
        if not pattern:
            return self.metrics.copy()

        return {k: v for k, v in self.metrics.items() if pattern in k}

    def get_metric_history(
        self, name: str, limit: Optional[int] = None
    ) -> List[Tuple[float, str]]:
        """Get historical metric values.

        Args:
            name: Metric name
            limit: Limit number of entries

        Returns:
            List of (value, timestamp) tuples
        """
        metric_key = f"{self.config.metrics.metrics_prefix}_{name}"
        history = self.metric_history.get(metric_key, [])

        if limit:
            return history[-limit:]
        return history

    def aggregate_metrics(
        self, pattern: str, aggregation: str = "avg"
    ) -> Optional[float]:
        """Aggregate metrics matching pattern.

        Args:
            pattern: Pattern to match metrics
            aggregation: Aggregation function (avg, sum, min, max, count)

        Returns:
            Aggregated value or None

        Raises:
            ValueError: If aggregation function is invalid
        """
        matching_metrics = [v["value"] for k, v in self.metrics.items() if pattern in k]

        if not matching_metrics:
            return None

        if aggregation == "avg":
            return sum(matching_metrics) / len(matching_metrics)
        elif aggregation == "sum":
            return sum(matching_metrics)
        elif aggregation == "min":
            return min(matching_metrics)
        elif aggregation == "max":
            return max(matching_metrics)
        elif aggregation == "count":
            return float(len(matching_metrics))
        else:
            raise ValueError(f"Unknown aggregation function: {aggregation}")

    def clear_metrics(self, pattern: Optional[str] = None) -> int:
        """Clear metrics matching pattern.

        Args:
            pattern: Optional pattern to filter metrics

        Returns:
            Number of metrics cleared
        """
        if not pattern:
            count = len(self.metrics)
            self.metrics.clear()
            self.metric_history.clear()
            logger.info(f"Cleared all {count} metrics")
            return count

        keys_to_delete = [k for k in self.metrics.keys() if pattern in k]
        for key in keys_to_delete:
            del self.metrics[key]
            self.metric_history.pop(key, None)

        logger.info(
            f"Cleared {len(keys_to_delete)} metrics matching pattern: {pattern}"
        )
        return len(keys_to_delete)

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp.

        Returns:
            ISO format timestamp
        """
        return datetime.utcnow().isoformat() + "Z"


class AlertManager:
    """Manages alerts and alert rules."""

    def __init__(self, config: MonitoringConfig):
        """Initialize alert manager.

        Args:
            config: Monitoring configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("Monitoring configuration must be provided")
        self.config = config
        self.alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        logger.debug("Initialized AlertManager")

    def create_alert_rule(
        self,
        name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        duration_seconds: int = 300,
    ) -> Dict[str, Any]:
        """Create an alert rule.

        Args:
            name: Rule name
            condition: Condition expression
            threshold: Alert threshold value
            severity: Alert severity
            duration_seconds: Duration before triggering alert

        Returns:
            Created rule metadata

        Raises:
            ValueError: If configuration is invalid
        """
        if not name or not condition:
            raise ValueError("Rule name and condition must be provided")

        rule = {
            "name": name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity.value,
            "duration_seconds": duration_seconds,
            "enabled": True,
            "created_at": self._get_timestamp(),
        }

        self.alert_rules[name] = rule
        logger.info(f"Created alert rule: {name}")

        return rule

    def trigger_alert(
        self,
        rule_name: str,
        value: float,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger an alert.

        Args:
            rule_name: Alert rule name
            value: Current metric value
            message: Optional alert message

        Returns:
            Alert metadata

        Raises:
            ValueError: If rule name is invalid
        """
        if rule_name not in self.alert_rules:
            raise ValueError(f"Alert rule not found: {rule_name}")

        alert_id = f"alert-{len(self.alerts) + 1}"

        alert = {
            "id": alert_id,
            "rule_name": rule_name,
            "value": value,
            "message": message or f"Alert triggered for rule: {rule_name}",
            "severity": self.alert_rules[rule_name]["severity"],
            "triggered_at": self._get_timestamp(),
            "status": "active",
        }

        self.alerts[alert_id] = alert
        self.alert_history.append(alert.copy())

        logger.warning(f"Alert triggered: {alert_id} ({rule_name})")

        return alert

    def resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        """Resolve an active alert.

        Args:
            alert_id: Alert ID

        Returns:
            Resolved alert metadata

        Raises:
            ValueError: If alert not found
        """
        if alert_id not in self.alerts:
            raise ValueError(f"Alert not found: {alert_id}")

        alert = self.alerts[alert_id]
        alert["status"] = "resolved"
        alert["resolved_at"] = self._get_timestamp()

        logger.info(f"Alert resolved: {alert_id}")

        return alert

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[Dict[str, Any]]:
        """Get active alerts.

        Args:
            severity: Filter by severity (optional)

        Returns:
            List of active alerts
        """
        active = [a for a in self.alerts.values() if a["status"] == "active"]

        if severity:
            active = [a for a in active if a["severity"] == severity.value]

        return active

    def get_alert_history(
        self, rule_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get alert history.

        Args:
            rule_name: Filter by rule name (optional)
            limit: Limit number of entries (optional)

        Returns:
            List of historical alerts
        """
        history = self.alert_history

        if rule_name:
            history = [a for a in history if a["rule_name"] == rule_name]

        if limit:
            history = history[-limit:]

        return history

    def list_alert_rules(self) -> List[Dict[str, Any]]:
        """List all alert rules.

        Returns:
            List of alert rules
        """
        return list(self.alert_rules.values())

    def disable_alert_rule(self, name: str) -> Dict[str, Any]:
        """Disable an alert rule.

        Args:
            name: Rule name

        Returns:
            Updated rule metadata

        Raises:
            ValueError: If rule not found
        """
        if name not in self.alert_rules:
            raise ValueError(f"Alert rule not found: {name}")

        self.alert_rules[name]["enabled"] = False
        logger.info(f"Disabled alert rule: {name}")

        return self.alert_rules[name]

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp.

        Returns:
            ISO format timestamp
        """
        return datetime.utcnow().isoformat() + "Z"


class Dashboard:
    """Dashboard management."""

    def __init__(self, config: MonitoringConfig):
        """Initialize dashboard manager.

        Args:
            config: Monitoring configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("Monitoring configuration must be provided")
        self.config = config
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        logger.debug("Initialized Dashboard")

    def create_dashboard(self, dashboard_config: DashboardConfig) -> Dict[str, Any]:
        """Create a dashboard.

        Args:
            dashboard_config: Dashboard configuration

        Returns:
            Created dashboard metadata

        Raises:
            ValueError: If configuration is invalid
        """
        if not dashboard_config:
            raise ValueError("Dashboard configuration must be provided")

        dashboard = {
            "name": dashboard_config.name,
            "backend": dashboard_config.backend,
            "url": dashboard_config.url,
            "auto_refresh_seconds": dashboard_config.auto_refresh_seconds,
            "time_range": dashboard_config.time_range,
            "panels": dashboard_config.panels,
            "tags": dashboard_config.tags,
            "created_at": self._get_timestamp(),
        }

        self.dashboards[dashboard_config.name] = dashboard
        logger.info(f"Created dashboard: {dashboard_config.name}")

        return dashboard

    def add_panel(
        self,
        dashboard_name: str,
        panel_id: str,
        panel_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add a panel to dashboard.

        Args:
            dashboard_name: Dashboard name
            panel_id: Panel ID
            panel_config: Panel configuration

        Returns:
            Updated dashboard metadata

        Raises:
            ValueError: If dashboard not found
        """
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_name}")

        self.dashboards[dashboard_name]["panels"][panel_id] = panel_config
        logger.info(f"Added panel {panel_id} to dashboard: {dashboard_name}")

        return self.dashboards[dashboard_name]

    def get_dashboard(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dashboard by name.

        Args:
            name: Dashboard name

        Returns:
            Dashboard metadata or None if not found
        """
        return self.dashboards.get(name)

    def list_dashboards(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all dashboards.

        Args:
            tag: Filter by tag (optional)

        Returns:
            List of dashboards
        """
        dashboards = list(self.dashboards.values())

        if tag:
            dashboards = [d for d in dashboards if tag in d.get("tags", [])]

        return dashboards

    def delete_dashboard(self, name: str) -> Dict[str, Any]:
        """Delete a dashboard.

        Args:
            name: Dashboard name

        Returns:
            Deleted dashboard metadata

        Raises:
            ValueError: If dashboard not found
        """
        if name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {name}")

        dashboard = self.dashboards.pop(name)
        logger.info(f"Deleted dashboard: {name}")

        return dashboard

    def export_dashboard(self, name: str) -> Dict[str, Any]:
        """Export dashboard configuration.

        Args:
            name: Dashboard name

        Returns:
            Dashboard configuration

        Raises:
            ValueError: If dashboard not found
        """
        if name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {name}")

        return self.dashboards[name].copy()

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp.

        Returns:
            ISO format timestamp
        """
        return datetime.utcnow().isoformat() + "Z"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    monitoring_config = MonitoringConfig()

    # Create metrics collector
    collector = MetricsCollector(monitoring_config)
    collector.record_metric("api_latency_ms", 150.5, tags={"endpoint": "/api/v1"})
    collector.record_metric("gpu_memory_percent", 75.0)

    print(f"Metrics: {json.dumps(collector.get_metrics(), indent=2)}")

    # Create alert manager
    alert_manager = AlertManager(monitoring_config)
    alert_manager.create_alert_rule(
        "high_latency",
        "api_latency_ms > 1000",
        1000.0,
        severity=AlertSeverity.WARNING,
    )
    alert = alert_manager.trigger_alert("high_latency", 1500.0)
    print(f"Alert: {json.dumps(alert, indent=2)}")

    # Create dashboard
    dashboard_mgr = Dashboard(monitoring_config)
    dashboard_config = DashboardConfig(name="API Metrics")
    dashboard_mgr.create_dashboard(dashboard_config)
    print(f"Dashboards: {json.dumps(dashboard_mgr.list_dashboards(), indent=2)}")
