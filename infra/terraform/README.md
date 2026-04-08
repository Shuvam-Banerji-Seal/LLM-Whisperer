# Terraform Module

Infrastructure as Code with Terraform for LLM-Whisperer.

## Overview

The Terraform module provides comprehensive Infrastructure as Code tools for managing cloud infrastructure. It includes:

- **TerraformApplier**: Main orchestrator for infrastructure changes
- **StackManager**: Manage individual Terraform stacks
- **VarsManager**: Manage Terraform variables and configurations

## Main Classes

### TerraformApplier

Main orchestrator for applying and managing infrastructure.

```python
from infra.terraform.core import TerraformApplier
from infra.terraform.config import TerraformConfig, BackendConfig, BackendType

config = TerraformConfig(
    workspace="production",
    backend=BackendConfig(
        backend_type=BackendType.S3,
        bucket="terraform-state",
        region="us-west-2"
    )
)
applier = TerraformApplier(config)

# Initialize workspace
init_result = applier.init_workspace()

# Plan infrastructure
plan_result = applier.plan_all()

# Apply changes
apply_result = applier.apply_infrastructure(auto_approve=False)

# Destroy infrastructure
destroy_result = applier.destroy_infrastructure(force=False)

# Get outputs
outputs = applier.get_outputs()

# Get execution history
history = applier.get_execution_history(limit=10)
```

**Key Methods:**
- `init_workspace(workspace)`: Initialize Terraform workspace
- `apply_infrastructure(stack_names, auto_approve)`: Apply infrastructure
- `destroy_infrastructure(stack_names, force)`: Destroy infrastructure
- `plan_all()`: Plan all stacks
- `get_outputs(stack_name)`: Get Terraform outputs
- `get_execution_history(limit)`: Get execution history

### StackManager

Manage individual Terraform stacks.

```python
from infra.terraform.core import StackManager
from infra.terraform.config import TerraformConfig, StackConfig

config = TerraformConfig()
manager = StackManager(config)

# Create stack
stack_config = StackConfig(
    name="production",
    working_dir="./terraform/production",
    environment="production",
    parallelism=10
)
stack = manager.create_stack(stack_config)

# Plan stack
plan_result = manager.plan_stack("production", destroy=False)

# Apply stack
apply_result = manager.apply_stack("production")

# Destroy stack
destroy_result = manager.destroy_stack("production")

# List stacks
stacks = manager.list_stacks()

# Get stack state
state = manager.get_stack_state("production")

# Refresh stack
refresh_result = manager.refresh_stack("production")
```

**Key Methods:**
- `create_stack(stack_config)`: Create stack
- `plan_stack(name, destroy)`: Plan stack changes
- `apply_stack(name)`: Apply stack changes
- `destroy_stack(name)`: Destroy stack
- `list_stacks()`: List all stacks
- `get_stack_state(name)`: Get stack state
- `refresh_stack(name)`: Refresh stack state

### VarsManager

Manage Terraform variables and configurations.

```python
from infra.terraform.core import VarsManager
from infra.terraform.config import VarsConfig

vars_config = VarsConfig()
vars_manager = VarsManager(vars_config)

# Set variable
vars_manager.set_variable("instance_count", 3)
vars_manager.set_variable("db_password", "secret123", sensitive=True)

# Get variable
count = vars_manager.get_variable("instance_count")

# Get variables matching pattern
app_vars = vars_manager.get_variables(pattern="app_")

# Set environment variable for Terraform
vars_manager.set_environment_variable("api_key", "sk-123456")

# Validate variables
validation = vars_manager.validate_variables()

# Clear variables
cleared = vars_manager.clear_variables(pattern="temp_")
```

**Key Methods:**
- `set_variable(name, value, sensitive)`: Set variable
- `get_variable(name)`: Get variable value
- `get_variables(pattern)`: Get variables matching pattern
- `set_environment_variable(key, value)`: Set environment variable
- `validate_variables()`: Validate all variables
- `clear_variables(pattern)`: Clear variables

## Configuration

### TerraformConfig

Main Terraform configuration.

```python
from infra.terraform.config import TerraformConfig, BackendConfig, BackendType

config = TerraformConfig(
    version="1.5.0",
    workspace="production",
    backend=BackendConfig(
        backend_type=BackendType.S3,
        bucket="terraform-state",
        key="prod/terraform.tfstate",
        region="us-west-2",
        encrypt=True
    ),
    global_vars={
        "environment": "production",
        "region": "us-west-2",
        "tags": {"project": "llm-whisperer"}
    },
    enable_remote_state=True,
    log_level="INFO"
)
```

**Fields:**
- `version`: Terraform version
- `stacks`: Stack configurations
- `global_vars`: Global variables
- `backend`: Backend configuration
- `providers`: List of providers
- `enable_remote_state`: Enable remote state
- `workspace`: Workspace name
- `log_level`: Log level
- `max_retries`: Maximum retries

### StackConfig

Terraform stack configuration.

```python
from infra.terraform.config import StackConfig, BackendConfig

stack_config = StackConfig(
    name="production",
    working_dir="./terraform/production",
    backend=BackendConfig(backend_type="s3"),
    environment="production",
    auto_approve=False,
    parallelism=10,
    lock=True,
    refresh=True
)
```

### VarsConfig

Variables configuration.

```python
from infra.terraform.config import VarsConfig

vars_config = VarsConfig(
    variables={
        "instance_count": 3,
        "instance_type": "t3.xlarge",
        "environment": "production"
    },
    variables_file="terraform.tfvars",
    var_files=["common.tfvars", "production.tfvars"],
    environment_variables={},
    sensitive_variables=["db_password", "api_key"]
)
```

### BackendConfig

Backend storage configuration.

```python
from infra.terraform.config import BackendConfig, BackendType

# S3 Backend
s3_backend = BackendConfig(
    backend_type=BackendType.S3,
    bucket="terraform-state",
    key="prod/terraform.tfstate",
    region="us-west-2",
    dynamodb_table="terraform-locks",
    encrypt=True
)

# Terraform Cloud Backend
cloud_backend = BackendConfig(
    backend_type=BackendType.TERRAFORM_CLOUD,
    bucket="organization/workspace"
)

# Local Backend
local_backend = BackendConfig(
    backend_type=BackendType.LOCAL
)
```

## Error Handling

All classes validate input and raise `ValueError` for invalid configurations:

```python
try:
    config = TerraformConfig(max_retries=-1)
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

## Example: Complete Infrastructure Workflow

```python
from infra.terraform.core import TerraformApplier, VarsManager
from infra.terraform.config import (
    TerraformConfig,
    StackConfig,
    VarsConfig,
    BackendConfig,
    BackendType,
)

# Configure Terraform with remote state
tf_config = TerraformConfig(
    workspace="production",
    backend=BackendConfig(
        backend_type=BackendType.S3,
        bucket="llm-whisperer-state",
        key="prod/terraform.tfstate",
        region="us-west-2",
        encrypt=True
    ),
    global_vars={
        "environment": "production",
        "region": "us-west-2",
        "project": "llm-whisperer"
    }
)

# Initialize Terraform applier
applier = TerraformApplier(tf_config)

# Initialize workspace
applier.init_workspace("production")

# Manage variables
vars_config = VarsConfig(
    variables={
        "instance_count": 5,
        "instance_type": "p3.8xlarge",
        "enable_autoscaling": True
    },
    sensitive_variables=["db_password", "api_key"]
)
vars_manager = VarsManager(vars_config)
vars_manager.set_variable("db_password", "secure-password", sensitive=True)

# Create infrastructure stack
stack_config = StackConfig(
    name="api-infrastructure",
    working_dir="./terraform/api",
    environment="production",
    parallelism=10,
    auto_approve=False
)
applier.stack_manager.create_stack(stack_config)

# Plan infrastructure changes
plan_result = applier.plan_all()
print(f"Planned changes: {plan_result['summary']['total_changes']}")

# Apply infrastructure
apply_result = applier.apply_infrastructure(auto_approve=False)
print(f"Applied stacks: {apply_result['summary']['stacks_applied']}")

# Get outputs
outputs = applier.get_outputs()
print(f"API endpoint: {outputs['api_endpoint']}")

# Monitor execution
history = applier.get_execution_history(limit=5)
for execution in history:
    print(f"Execution: {execution['timestamp']} - {execution['status']}")
```

## Supported Backends

- **local**: Local filesystem state
- **s3**: AWS S3 state storage
- **azurerm**: Azure Remote State
- **gcs**: Google Cloud Storage
- **consul**: Consul KV store
- **terraform_cloud**: Terraform Cloud

## Testing

Run the module directly for basic examples:

```bash
python -m infra.terraform.core
```

## Best Practices

1. **State Management**: Use remote backends for production
2. **Locking**: Enable state locking to prevent concurrent modifications
3. **Variables**: Use sensitive variables for secrets
4. **Parallelism**: Adjust parallelism based on resources
5. **Planning**: Always plan before applying in production
6. **Tagging**: Use consistent tagging for resource management

## Environment Variables

Set Terraform-specific environment variables:

```bash
export TF_LOG=DEBUG
export TF_INPUT=false
export TF_PARALLELISM=10
```

## See Also

- [Docker Module](../docker/README.md)
- [Kubernetes Module](../kubernetes/README.md)
- [Monitoring Module](../monitoring/README.md)
