# Docker Module

Docker containerization utilities and builders for LLM-Whisperer.

## Overview

The Docker module provides comprehensive tools for building, managing, and pushing Docker images. It includes:

- **DockerBuilder**: Build Docker images with advanced configuration options
- **DockerRegistry**: Manage image push/pull operations with registry support
- **ContainerConfig**: Configure containers with resources, health checks, volumes, and ports

## Main Classes

### DockerBuilder

Builds Docker images with support for build arguments, labels, and platform-specific builds.

```python
from infra.docker.core import DockerBuilder
from infra.docker.config import DockerConfig, ImageBuildConfig

config = DockerConfig()
builder = DockerBuilder(config)

build_config = ImageBuildConfig(
    dockerfile_path="./Dockerfile",
    context_path=".",
    tag="my-app:v1.0.0",
    buildargs={"PYTHON_VERSION": "3.11"},
    labels={"app": "my-app", "version": "1.0.0"}
)

result = builder.build_image(build_config)
# Build and push to registry
push_result = builder.build_and_push(build_config)
```

**Key Methods:**
- `build_image(build_config)`: Build Docker image
- `build_and_push(build_config)`: Build and push to registry
- `list_built_images()`: Get list of built images
- `get_image_info(tag)`: Get info about specific image
- `validate_build_config(build_config)`: Validate configuration

### DockerRegistry

Manage Docker registry operations including push, pull, and delete.

```python
from infra.docker.core import DockerRegistry
from infra.docker.config import RegistryConfig

registry_config = RegistryConfig(
    registry_url="docker.io",
    username="your-username",
    password="your-password"
)
registry = DockerRegistry(registry_config)

# Push image to registry
push_result = registry.push_image("my-app:v1.0.0")

# Pull image from registry
pull_result = registry.pull_image("my-app:v1.0.0")

# List pushed images
pushed = registry.list_pushed_images()

# Delete image
delete_result = registry.delete_image("my-app:v1.0.0")
```

**Key Methods:**
- `push_image(image_tag)`: Push image to registry
- `pull_image(image_tag, pull_policy)`: Pull image from registry
- `delete_image(image_tag)`: Delete image from registry
- `list_pushed_images()`: List all pushed images
- `list_pulled_images()`: List all pulled images
- `get_image_details(image_tag)`: Get image metadata

### ContainerConfig

Configure containers with resources, environment variables, volumes, and port mappings.

```python
from infra.docker.core import ContainerConfig
from infra.docker.config import ContainerResourceConfig, ContainerHealthCheckConfig

container = ContainerConfig(
    name="api-server",
    image="my-app:v1.0.0",
    resources=ContainerResourceConfig(cpu_cores=2.0, memory_mb=4096),
    health_check=ContainerHealthCheckConfig(interval_seconds=30)
)

# Add configuration
container.add_port_mapping(8000, 8000)
container.add_environment_variable("LOG_LEVEL", "INFO")
container.add_volume("/data", "/app/data")

# Get configuration as dictionary
config_dict = container.to_dict()
```

**Key Methods:**
- `add_environment_variable(key, value)`: Add environment variable
- `add_volume(source, target)`: Add volume mapping
- `add_port_mapping(host_port, container_port)`: Add port mapping
- `to_dict()`: Convert to dictionary representation

## Configuration

### DockerConfig

Main Docker configuration dataclass.

```python
from infra.docker.config import DockerConfig, RegistryConfig, ImageBuildConfig

config = DockerConfig(
    registry=RegistryConfig(registry_url="docker.io"),
    pull_policy=PullPolicy.IFNOTPRESENT,
    network_mode=DockerNetworkMode.BRIDGE,
    dns_servers=["8.8.8.8"],
    environment_variables={"HTTP_PROXY": "http://proxy:8080"}
)
```

**Fields:**
- `registry`: Registry configuration
- `image_build`: Default image build configuration
- `enable_logging`: Enable logging (default: True)
- `log_driver`: Log driver (default: "json-file")
- `max_log_size`: Max log size (default: "10m")
- `pull_policy`: Image pull policy
- `network_mode`: Network mode
- `dns_servers`: DNS servers
- `environment_variables`: Global environment variables

### ImageBuildConfig

Docker image build configuration.

```python
from infra.docker.config import ImageBuildConfig

config = ImageBuildConfig(
    dockerfile_path="./Dockerfile",
    context_path=".",
    tag="my-app:v1.0.0",
    buildargs={"PYTHON_VERSION": "3.11"},
    labels={"app": "my-app"},
    platform="linux/amd64",
    target="production",
    cache=True,
    no_cache=False
)
```

### RegistryConfig

Docker registry configuration.

```python
from infra.docker.config import RegistryConfig

config = RegistryConfig(
    registry_url="docker.io",
    username="your-username",
    password="your-password",
    email="your-email@example.com",
    insecure_skip_tls_verify=False
)
```

### ContainerResourceConfig

Container resource limits and requests.

```python
from infra.docker.config import ContainerResourceConfig

resources = ContainerResourceConfig(
    cpu_cores=2.0,
    memory_mb=4096,
    gpu_count=1,
    gpu_type="nvidia-tesla-v100"
)
```

### ContainerHealthCheckConfig

Container health check configuration.

```python
from infra.docker.config import ContainerHealthCheckConfig

health_check = ContainerHealthCheckConfig(
    enabled=True,
    command=["CMD", "curl", "-f", "http://localhost/health"],
    interval_seconds=30,
    timeout_seconds=5,
    start_period_seconds=10,
    retries=3
)
```

## Error Handling

All classes validate input and raise `ValueError` for invalid configurations:

```python
try:
    config = ContainerConfig(name="", image="my-app:v1.0.0")
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

## Example: Complete Workflow

```python
from infra.docker.core import DockerBuilder, DockerRegistry, ContainerConfig
from infra.docker.config import (
    DockerConfig,
    ImageBuildConfig,
    RegistryConfig,
    ContainerResourceConfig,
)

# Configure Docker
docker_config = DockerConfig(
    registry=RegistryConfig(registry_url="docker.io")
)

# Build image
builder = DockerBuilder(docker_config)
build_config = ImageBuildConfig(
    dockerfile_path="./Dockerfile",
    context_path=".",
    tag="llm-whisperer:v1.0.0"
)
build_result = builder.build_image(build_config)
print(f"Build status: {build_result['status']}")

# Configure container
container = ContainerConfig(
    name="llm-whisperer-api",
    image="llm-whisperer:v1.0.0",
    resources=ContainerResourceConfig(cpu_cores=4.0, memory_mb=8192)
)
container.add_port_mapping(8000, 8000)
container.add_environment_variable("LOG_LEVEL", "INFO")

# Push to registry
registry = DockerRegistry(docker_config.registry)
push_result = registry.push_image("llm-whisperer:v1.0.0")
print(f"Push status: {push_result['status']}")
```

## Testing

Run the module directly for basic examples:

```bash
python -m infra.docker.core
```

## Performance Considerations

- **Image Caching**: Enable caching for faster builds
- **Multi-stage Builds**: Use target parameter for production builds
- **Image Size**: Use buildargs to exclude unnecessary files
- **Registry Optimization**: Use IfNotPresent pull policy to reduce registry calls

## Error Codes

- `ValueError`: Invalid configuration or input
- Registry authentication failures are logged with details

## See Also

- [Kubernetes Module](../kubernetes/README.md)
- [Terraform Module](../terraform/README.md)
- [Monitoring Module](../monitoring/README.md)
