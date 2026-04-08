"""Kubernetes configuration dataclasses."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceType(str, Enum):
    """Kubernetes service types."""

    CLUSTERIP = "ClusterIP"
    NODEPORT = "NodePort"
    LOADBALANCER = "LoadBalancer"
    EXTERNALNAME = "ExternalName"


class RestartPolicy(str, Enum):
    """Pod restart policies."""

    ALWAYS = "Always"
    ONFAILURE = "OnFailure"
    NEVER = "Never"


class ImagePullPolicy(str, Enum):
    """Image pull policies."""

    ALWAYS = "Always"
    IFNOTPRESENT = "IfNotPresent"
    NEVER = "Never"


@dataclass
class ResourceRequirements:
    """Kubernetes resource requirements."""

    cpu_request: str = "100m"
    cpu_limit: str = "1000m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"

    def __post_init__(self):
        """Validate resource configuration."""
        if not self.cpu_request or not self.memory_request:
            raise ValueError("CPU and memory requests are required")


@dataclass
class HealthCheckProbe:
    """Kubernetes health check probe."""

    enabled: bool = True
    initial_delay_seconds: int = 10
    timeout_seconds: int = 5
    period_seconds: int = 10
    success_threshold: int = 1
    failure_threshold: int = 3

    def __post_init__(self):
        """Validate probe configuration."""
        if self.initial_delay_seconds < 0:
            raise ValueError(
                f"initial_delay_seconds must be non-negative, got {self.initial_delay_seconds}"
            )
        if self.period_seconds <= 0:
            raise ValueError(
                f"period_seconds must be positive, got {self.period_seconds}"
            )


@dataclass
class ContainerConfig:
    """Kubernetes container configuration."""

    name: str
    image: str
    image_pull_policy: ImagePullPolicy = ImagePullPolicy.IFNOTPRESENT
    ports: List[int] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    volume_mounts: Dict[str, str] = field(default_factory=dict)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    liveness_probe: Optional[HealthCheckProbe] = None
    readiness_probe: Optional[HealthCheckProbe] = None

    def __post_init__(self):
        """Validate container configuration."""
        if not self.name:
            raise ValueError("Container name must be provided")
        if not self.image:
            raise ValueError("Container image must be provided")


@dataclass
class DeploymentConfig:
    """Kubernetes deployment configuration."""

    name: str
    namespace: str = "default"
    replicas: int = 3
    containers: List[ContainerConfig] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    restart_policy: RestartPolicy = RestartPolicy.ALWAYS
    service_account_name: Optional[str] = None
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate deployment configuration."""
        if not self.name:
            raise ValueError("Deployment name must be provided")
        if self.replicas < 1:
            raise ValueError(f"replicas must be at least 1, got {self.replicas}")
        if not self.containers:
            raise ValueError("At least one container must be specified")


@dataclass
class ServiceConfig:
    """Kubernetes service configuration."""

    name: str
    namespace: str = "default"
    service_type: ServiceType = ServiceType.CLUSTERIP
    selector: Dict[str, str] = field(default_factory=dict)
    ports: Dict[int, int] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    cluster_ip: Optional[str] = None
    session_affinity: Optional[str] = None

    def __post_init__(self):
        """Validate service configuration."""
        if not self.name:
            raise ValueError("Service name must be provided")
        if not self.selector:
            raise ValueError("Service selector must be provided")


@dataclass
class IngressConfig:
    """Kubernetes ingress configuration."""

    name: str
    namespace: str = "default"
    ingress_class: str = "nginx"
    hosts: List[str] = field(default_factory=list)
    tls_enabled: bool = False
    tls_secret_name: Optional[str] = None
    rules: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate ingress configuration."""
        if not self.name:
            raise ValueError("Ingress name must be provided")
        if not self.hosts:
            raise ValueError("At least one host must be specified")


@dataclass
class HelmChartConfig:
    """Helm chart configuration."""

    name: str
    chart_name: str
    namespace: str = "default"
    release_name: Optional[str] = None
    version: Optional[str] = None
    values: Dict[str, Any] = field(default_factory=dict)
    values_files: List[str] = field(default_factory=list)
    create_namespace: bool = True
    wait: bool = True
    timeout_seconds: int = 300

    def __post_init__(self):
        """Validate Helm chart configuration."""
        if not self.name:
            raise ValueError("Helm name must be provided")
        if not self.chart_name:
            raise ValueError("chart_name must be provided")
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )


@dataclass
class K8sConfig:
    """Main Kubernetes configuration."""

    cluster_name: str
    context: str
    namespace: str = "default"
    kubeconfig_path: Optional[str] = None
    api_server: Optional[str] = None
    verify_ssl: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3

    def __post_init__(self):
        """Validate K8s configuration."""
        if not self.cluster_name:
            raise ValueError("cluster_name must be provided")
        if not self.context:
            raise ValueError("context must be provided")
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
