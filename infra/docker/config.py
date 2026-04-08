"""Docker configuration dataclasses."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class PullPolicy(str, Enum):
    """Container pull policy."""

    ALWAYS = "Always"
    IFNOTPRESENT = "IfNotPresent"
    NEVER = "Never"


class DockerNetworkMode(str, Enum):
    """Docker network modes."""

    BRIDGE = "bridge"
    HOST = "host"
    NONE = "none"
    CONTAINER = "container"


@dataclass
class RegistryConfig:
    """Docker registry configuration."""

    registry_url: str = "docker.io"
    username: Optional[str] = None
    password: Optional[str] = None
    email: Optional[str] = None
    insecure_skip_tls_verify: bool = False

    def __post_init__(self):
        """Validate registry configuration."""
        if not self.registry_url:
            raise ValueError("registry_url must be provided")
        if (self.username and not self.password) or (
            self.password and not self.username
        ):
            raise ValueError(
                "Both username and password must be provided for authentication"
            )


@dataclass
class ImageBuildConfig:
    """Docker image build configuration."""

    dockerfile_path: str
    context_path: str
    tag: str
    buildargs: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    cache: bool = True
    platform: Optional[str] = None
    target: Optional[str] = None
    no_cache: bool = False

    def __post_init__(self):
        """Validate build configuration."""
        if not self.dockerfile_path:
            raise ValueError("dockerfile_path must be provided")
        if not self.context_path:
            raise ValueError("context_path must be provided")
        if not self.tag:
            raise ValueError("tag must be provided")


@dataclass
class DockerConfig:
    """Main Docker configuration."""

    registry: RegistryConfig = field(default_factory=RegistryConfig)
    image_build: ImageBuildConfig = field(
        default_factory=lambda: ImageBuildConfig(
            dockerfile_path="Dockerfile", context_path=".", tag="latest"
        )
    )
    enable_logging: bool = True
    log_driver: str = "json-file"
    max_log_size: str = "10m"
    pull_policy: PullPolicy = PullPolicy.IFNOTPRESENT
    network_mode: DockerNetworkMode = DockerNetworkMode.BRIDGE
    dns_servers: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate Docker configuration."""
        if not self.registry:
            raise ValueError("Registry configuration is required")
        if not self.image_build:
            raise ValueError("Image build configuration is required")


@dataclass
class ContainerResourceConfig:
    """Container resource limits and requests."""

    cpu_cores: float = 1.0
    memory_mb: int = 512
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    ephemeral_storage_mb: Optional[int] = None

    def __post_init__(self):
        """Validate resource configuration."""
        if self.cpu_cores <= 0:
            raise ValueError(f"cpu_cores must be positive, got {self.cpu_cores}")
        if self.memory_mb <= 0:
            raise ValueError(f"memory_mb must be positive, got {self.memory_mb}")
        if self.gpu_count < 0:
            raise ValueError(f"gpu_count must be non-negative, got {self.gpu_count}")


@dataclass
class ContainerHealthCheckConfig:
    """Container health check configuration."""

    enabled: bool = True
    command: List[str] = field(
        default_factory=lambda: ["CMD", "curl", "-f", "http://localhost/health"]
    )
    interval_seconds: int = 30
    timeout_seconds: int = 5
    start_period_seconds: int = 10
    retries: int = 3

    def __post_init__(self):
        """Validate health check configuration."""
        if self.interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, got {self.interval_seconds}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
        if self.retries < 0:
            raise ValueError(f"retries must be non-negative, got {self.retries}")
