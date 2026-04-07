"""Infra utilities for deployment and monitoring."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environments."""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class InfraConfig:
    """Infrastructure configuration."""

    environment: DeploymentEnvironment
    region: str
    cpu_cores: int = 4
    memory_gb: int = 16
    gpu_enabled: bool = False
    gpu_type: Optional[str] = None
    enable_monitoring: bool = True
    enable_logging: bool = True


class DockerBuilder:
    """Builds Docker containers."""

    def __init__(self, config: InfraConfig):
        """Initialize Docker builder.

        Args:
            config: Infrastructure configuration
        """
        self.config = config

    def build_image(self, dockerfile_path: str, tag: str) -> Dict[str, Any]:
        """Build Docker image.

        Args:
            dockerfile_path: Path to Dockerfile
            tag: Image tag

        Returns:
            Build result
        """
        logger.info(f"Building Docker image: {tag}")

        return {
            "status": "built",
            "tag": tag,
            "environment": self.config.environment.value,
        }


class KubernetesDeployer:
    """Deploys to Kubernetes."""

    def __init__(self, config: InfraConfig):
        """Initialize Kubernetes deployer.

        Args:
            config: Infrastructure configuration
        """
        self.config = config

    def deploy(self, yaml_path: str, namespace: str = "default") -> Dict[str, Any]:
        """Deploy application to Kubernetes.

        Args:
            yaml_path: Path to deployment YAML
            namespace: Kubernetes namespace

        Returns:
            Deployment result
        """
        logger.info(f"Deploying to Kubernetes in namespace: {namespace}")

        return {
            "status": "deployed",
            "namespace": namespace,
            "environment": self.config.environment.value,
            "replicas": 3,
        }


class MonitoringSystem:
    """Monitoring system integration."""

    def __init__(self, config: InfraConfig):
        """Initialize monitoring system.

        Args:
            config: Infrastructure configuration
        """
        self.config = config
        self.metrics = {}

    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None):
        """Record a metric.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional metric tags
        """
        self.metrics[name] = {"value": value, "tags": tags or {}}
        logger.debug(f"Recorded metric: {name} = {value}")

    def get_metrics(self, name_pattern: Optional[str] = None) -> Dict[str, Any]:
        """Get recorded metrics.

        Args:
            name_pattern: Optional pattern to filter metrics

        Returns:
            Dictionary of metrics
        """
        if name_pattern:
            return {k: v for k, v in self.metrics.items() if name_pattern in k}

        return self.metrics


class LoggingManager:
    """Centralized logging management."""

    def __init__(self, config: InfraConfig):
        """Initialize logging manager.

        Args:
            config: Infrastructure configuration
        """
        self.config = config
        self.logs = []

    def log(self, level: str, message: str, context: Optional[Dict] = None):
        """Log a message.

        Args:
            level: Log level
            message: Log message
            context: Optional context data
        """
        log_entry = {
            "level": level,
            "message": message,
            "context": context or {},
            "environment": self.config.environment.value,
        }
        self.logs.append(log_entry)
        logger.log(getattr(logging, level), message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = InfraConfig(
        environment=DeploymentEnvironment.PRODUCTION, region="us-west-2"
    )

    monitoring = MonitoringSystem(config)
    monitoring.record_metric("inference_latency_ms", 150.0)
    print(monitoring.get_metrics())
