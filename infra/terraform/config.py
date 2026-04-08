"""Terraform configuration dataclasses."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Terraform backend types."""

    LOCAL = "local"
    S3 = "s3"
    AZURERM = "azurerm"
    GCS = "gcs"
    CONSUL = "consul"
    TERRAFORM_CLOUD = "terraform_cloud"


class DestroyBehavior(str, Enum):
    """Destroy behavior."""

    APPLY = "apply"
    DESTROY = "destroy"
    PLAN = "plan"


@dataclass
class BackendConfig:
    """Terraform backend configuration."""

    backend_type: BackendType = BackendType.LOCAL
    bucket: Optional[str] = None
    key: Optional[str] = None
    region: Optional[str] = None
    dynamodb_table: Optional[str] = None
    encrypt: bool = True
    skip_credentials_validation: bool = False
    skip_region_validation: bool = False

    def __post_init__(self):
        """Validate backend configuration."""
        if self.backend_type in [BackendType.S3, BackendType.GCS] and not self.bucket:
            raise ValueError(
                f"bucket is required for {self.backend_type.value} backend"
            )


@dataclass
class VarsConfig:
    """Terraform variables configuration."""

    variables: Dict[str, Any] = field(default_factory=dict)
    variables_file: Optional[str] = None
    var_files: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    sensitive_variables: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate variables configuration."""
        # Validate that sensitive variables are actually defined
        for var in self.sensitive_variables:
            if var not in self.variables and var not in self.environment_variables:
                logger.warning(f"Sensitive variable '{var}' not defined in variables")


@dataclass
class StackConfig:
    """Terraform stack configuration."""

    name: str
    working_dir: str
    backend: BackendConfig = field(default_factory=BackendConfig)
    vars: VarsConfig = field(default_factory=VarsConfig)
    environment: str = "production"
    auto_approve: bool = False
    parallelism: int = 10
    lock: bool = True
    input: bool = False
    refresh: bool = True
    version: Optional[str] = None

    def __post_init__(self):
        """Validate stack configuration."""
        if not self.name:
            raise ValueError("Stack name must be provided")
        if not self.working_dir:
            raise ValueError("working_dir must be provided")
        if self.parallelism <= 0:
            raise ValueError(f"parallelism must be positive, got {self.parallelism}")


@dataclass
class TerraformConfig:
    """Main Terraform configuration."""

    version: Optional[str] = None
    stacks: Dict[str, StackConfig] = field(default_factory=dict)
    global_vars: Dict[str, Any] = field(default_factory=dict)
    backend: BackendConfig = field(default_factory=BackendConfig)
    providers: List[str] = field(default_factory=list)
    enable_remote_state: bool = True
    workspace: str = "default"
    log_level: str = "INFO"
    max_retries: int = 3

    def __post_init__(self):
        """Validate Terraform configuration."""
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )
