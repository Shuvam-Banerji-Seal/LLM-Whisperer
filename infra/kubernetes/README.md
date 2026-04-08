# Kubernetes Module

Kubernetes deployment configurations and operators for LLM-Whisperer.

## Overview

The Kubernetes module provides comprehensive tools for managing Kubernetes deployments, services, and ingress configurations. It includes:

- **K8sDeployer**: Deploy and manage applications on Kubernetes clusters
- **ResourceManager**: Create and manage Kubernetes resources (deployments, services, ingresses)
- **HelmChart**: Install and manage Helm charts

## Main Classes

### K8sDeployer

Deploy applications to Kubernetes clusters with support for scaling and status monitoring.

```python
from infra.kubernetes.core import K8sDeployer
from infra.kubernetes.config import K8sConfig

config = K8sConfig(
    cluster_name="production",
    context="prod-context",
    namespace="default"
)
deployer = K8sDeployer(config)

# Deploy from YAML
result = deployer.deploy("./deployment.yaml", namespace="default")

# Scale a deployment
scale_result = deployer.scale_deployment("api-server", replicas=5)

# Get rollout status
status = deployer.rollout_status("api-server")

# List deployments
deployments = deployer.list_deployments(namespace="default")
```

**Key Methods:**
- `deploy(yaml_path, namespace)`: Deploy from YAML manifest
- `scale_deployment(name, replicas, namespace)`: Scale deployment
- `rollout_status(name, namespace)`: Get rollout status
- `list_deployments(namespace)`: List all deployments

### ResourceManager

Create and manage Kubernetes resources including deployments, services, and ingresses.

```python
from infra.kubernetes.core import ResourceManager
from infra.kubernetes.config import (
    K8sConfig,
    DeploymentConfig,
    ServiceConfig,
    IngressConfig,
    ContainerConfig,
)

config = K8sConfig(
    cluster_name="production",
    context="prod-context"
)
manager = ResourceManager(config)

# Create deployment
deployment_config = DeploymentConfig(
    name="api-server",
    replicas=3,
    containers=[
        ContainerConfig(
            name="api",
            image="llm-whisperer:v1.0.0",
            ports=[8000]
        )
    ]
)
deploy_result = manager.create_deployment(deployment_config)

# Create service
service_config = ServiceConfig(
    name="api-service",
    selector={"app": "api-server"},
    ports={8000: 8000}
)
service_result = manager.create_service(service_config)

# Create ingress
ingress_config = IngressConfig(
    name="api-ingress",
    hosts=["api.example.com"],
    rules={"api.example.com": "api-service"}
)
ingress_result = manager.create_ingress(ingress_config)
```

**Key Methods:**
- `create_deployment(deployment_config)`: Create deployment
- `create_service(service_config)`: Create service
- `create_ingress(ingress_config)`: Create ingress
- `delete_deployment(name, namespace)`: Delete deployment
- `delete_service(name, namespace)`: Delete service
- `list_deployments(namespace)`: List deployments
- `list_services(namespace)`: List services
- `list_ingresses(namespace)`: List ingresses
- `get_resource_status(kind, name, namespace)`: Get resource status

### HelmChart

Install and manage Helm charts for complex deployments.

```python
from infra.kubernetes.core import HelmChart
from infra.kubernetes.config import K8sConfig, HelmChartConfig

config = K8sConfig(
    cluster_name="production",
    context="prod-context"
)
helm = HelmChart(config)

# Install chart
chart_config = HelmChartConfig(
    name="prometheus",
    chart_name="kube-prometheus-stack",
    version="45.0.0",
    namespace="monitoring",
    values={"prometheus": {"enabled": True}}
)
install_result = helm.install(chart_config)

# Upgrade release
upgrade_result = helm.upgrade(chart_config)

# Uninstall release
uninstall_result = helm.uninstall("prometheus", namespace="monitoring")

# List releases
releases = helm.list_releases(namespace="monitoring")

# Get release values
values = helm.get_values("prometheus")
```

**Key Methods:**
- `install(chart_config)`: Install Helm chart
- `upgrade(chart_config)`: Upgrade Helm release
- `uninstall(release_name, namespace)`: Uninstall release
- `list_releases(namespace)`: List all releases
- `get_values(release_name)`: Get release values

## Configuration

### K8sConfig

Main Kubernetes configuration.

```python
from infra.kubernetes.config import K8sConfig

config = K8sConfig(
    cluster_name="production",
    context="prod-context",
    namespace="default",
    kubeconfig_path="~/.kube/config",
    api_server="https://api.example.com",
    verify_ssl=True,
    timeout_seconds=30,
    max_retries=3
)
```

**Fields:**
- `cluster_name`: Cluster name
- `context`: Kubectl context name
- `namespace`: Default namespace
- `kubeconfig_path`: Path to kubeconfig file
- `api_server`: Kubernetes API server URL
- `verify_ssl`: Verify SSL certificates
- `timeout_seconds`: Request timeout
- `max_retries`: Maximum retry attempts

### DeploymentConfig

Kubernetes deployment configuration.

```python
from infra.kubernetes.config import (
    DeploymentConfig,
    ContainerConfig,
    ResourceRequirements,
)

container = ContainerConfig(
    name="api",
    image="llm-whisperer:v1.0.0",
    ports=[8000],
    environment={"LOG_LEVEL": "INFO"},
    resources=ResourceRequirements(
        cpu_request="100m",
        cpu_limit="1000m",
        memory_request="256Mi",
        memory_limit="1Gi"
    )
)

deployment = DeploymentConfig(
    name="api-server",
    namespace="production",
    replicas=3,
    containers=[container],
    labels={"app": "api-server", "version": "v1"},
    restart_policy=RestartPolicy.ALWAYS
)
```

### ServiceConfig

Kubernetes service configuration.

```python
from infra.kubernetes.config import ServiceConfig, ServiceType

service = ServiceConfig(
    name="api-service",
    namespace="production",
    service_type=ServiceType.LOADBALANCER,
    selector={"app": "api-server"},
    ports={8000: 8000},
    labels={"app": "api-service"}
)
```

### IngressConfig

Kubernetes ingress configuration.

```python
from infra.kubernetes.config import IngressConfig

ingress = IngressConfig(
    name="api-ingress",
    namespace="production",
    hosts=["api.example.com", "api-v2.example.com"],
    tls_enabled=True,
    tls_secret_name="api-tls",
    rules={
        "api.example.com": "api-service",
        "api-v2.example.com": "api-v2-service"
    }
)
```

### HelmChartConfig

Helm chart configuration.

```python
from infra.kubernetes.config import HelmChartConfig

chart = HelmChartConfig(
    name="prometheus",
    chart_name="kube-prometheus-stack",
    version="45.0.0",
    namespace="monitoring",
    values={
        "prometheus": {"enabled": True},
        "grafana": {"enabled": True}
    },
    create_namespace=True,
    wait=True,
    timeout_seconds=300
)
```

### ContainerConfig

Kubernetes container configuration.

```python
from infra.kubernetes.config import (
    ContainerConfig,
    ResourceRequirements,
    HealthCheckProbe,
    ImagePullPolicy,
)

container = ContainerConfig(
    name="api",
    image="llm-whisperer:v1.0.0",
    image_pull_policy=ImagePullPolicy.IFNOTPRESENT,
    ports=[8000, 9000],
    environment={"LOG_LEVEL": "DEBUG"},
    volume_mounts={"/data": "/app/data"},
    resources=ResourceRequirements(
        cpu_request="100m",
        cpu_limit="1000m",
        memory_request="256Mi",
        memory_limit="1Gi"
    ),
    liveness_probe=HealthCheckProbe(initial_delay_seconds=30),
    readiness_probe=HealthCheckProbe(initial_delay_seconds=10)
)
```

## Error Handling

All classes validate input and raise `ValueError` for invalid configurations:

```python
try:
    config = K8sConfig(cluster_name="", context="prod")
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
from infra.kubernetes.core import K8sDeployer, ResourceManager, HelmChart
from infra.kubernetes.config import (
    K8sConfig,
    DeploymentConfig,
    ServiceConfig,
    ContainerConfig,
    ResourceRequirements,
    HelmChartConfig,
)

# Configure Kubernetes
k8s_config = K8sConfig(
    cluster_name="production",
    context="prod-context",
    namespace="llm-whisperer"
)

# Deploy using ResourceManager
manager = ResourceManager(k8s_config)

container = ContainerConfig(
    name="api",
    image="llm-whisperer:v1.0.0",
    ports=[8000],
    resources=ResourceRequirements(
        cpu_request="500m",
        cpu_limit="2000m",
        memory_request="1Gi",
        memory_limit="4Gi"
    )
)

deployment = DeploymentConfig(
    name="llm-whisperer-api",
    replicas=3,
    containers=[container],
    labels={"app": "llm-whisperer"}
)

deploy_result = manager.create_deployment(deployment)

service = ServiceConfig(
    name="llm-whisperer-service",
    selector={"app": "llm-whisperer"},
    ports={8000: 8000}
)

service_result = manager.create_service(service)

# Deploy Helm chart for monitoring
helm = HelmChart(k8s_config)
chart_config = HelmChartConfig(
    name="prometheus",
    chart_name="kube-prometheus-stack",
    namespace="monitoring"
)
helm.install(chart_config)

# Deploy using K8sDeployer
deployer = K8sDeployer(k8s_config)
deployer.deploy("./deployment.yaml")
```

## Testing

Run the module directly for basic examples:

```bash
python -m infra.kubernetes.core
```

## Performance Considerations

- **Replica Count**: Use appropriate replica counts for load balancing
- **Resource Limits**: Set proper CPU and memory limits
- **Health Checks**: Configure liveness and readiness probes
- **Service Type**: Use LoadBalancer for external access, ClusterIP for internal

## See Also

- [Docker Module](../docker/README.md)
- [Terraform Module](../terraform/README.md)
- [Monitoring Module](../monitoring/README.md)
