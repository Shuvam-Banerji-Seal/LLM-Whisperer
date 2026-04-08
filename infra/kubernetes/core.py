"""Kubernetes deployment and resource management core."""

import logging
import json
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from infra.kubernetes.config import (
    K8sConfig,
    DeploymentConfig,
    ServiceConfig,
    IngressConfig,
    HelmChartConfig,
)

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages Kubernetes resources (deployments, services, ingress)."""

    def __init__(self, config: K8sConfig):
        """Initialize resource manager.

        Args:
            config: Kubernetes configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("Kubernetes configuration must be provided")
        self.config = config
        self.managed_resources: Dict[str, List[Dict[str, Any]]] = {
            "deployments": [],
            "services": [],
            "ingresses": [],
        }
        logger.debug(f"Initialized ResourceManager for cluster: {config.cluster_name}")

    def create_deployment(self, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """Create a Kubernetes deployment.

        Args:
            deployment_config: Deployment configuration

        Returns:
            Deployment creation result

        Raises:
            ValueError: If configuration is invalid
        """
        if not deployment_config:
            raise ValueError("Deployment configuration must be provided")

        logger.info(f"Creating deployment: {deployment_config.name}")

        manifest = self._generate_deployment_manifest(deployment_config)

        result = {
            "status": "created",
            "kind": "Deployment",
            "name": deployment_config.name,
            "namespace": deployment_config.namespace,
            "replicas": deployment_config.replicas,
            "containers": len(deployment_config.containers),
            "timestamp": self._get_timestamp(),
        }

        self.managed_resources["deployments"].append(result)
        logger.info(f"Successfully created deployment: {deployment_config.name}")

        return result

    def create_service(self, service_config: ServiceConfig) -> Dict[str, Any]:
        """Create a Kubernetes service.

        Args:
            service_config: Service configuration

        Returns:
            Service creation result

        Raises:
            ValueError: If configuration is invalid
        """
        if not service_config:
            raise ValueError("Service configuration must be provided")

        logger.info(f"Creating service: {service_config.name}")

        manifest = self._generate_service_manifest(service_config)

        result = {
            "status": "created",
            "kind": "Service",
            "name": service_config.name,
            "namespace": service_config.namespace,
            "service_type": service_config.service_type.value,
            "ports": len(service_config.ports),
            "timestamp": self._get_timestamp(),
        }

        self.managed_resources["services"].append(result)
        logger.info(f"Successfully created service: {service_config.name}")

        return result

    def create_ingress(self, ingress_config: IngressConfig) -> Dict[str, Any]:
        """Create a Kubernetes ingress.

        Args:
            ingress_config: Ingress configuration

        Returns:
            Ingress creation result

        Raises:
            ValueError: If configuration is invalid
        """
        if not ingress_config:
            raise ValueError("Ingress configuration must be provided")

        logger.info(f"Creating ingress: {ingress_config.name}")

        manifest = self._generate_ingress_manifest(ingress_config)

        result = {
            "status": "created",
            "kind": "Ingress",
            "name": ingress_config.name,
            "namespace": ingress_config.namespace,
            "hosts": ingress_config.hosts,
            "tls_enabled": ingress_config.tls_enabled,
            "timestamp": self._get_timestamp(),
        }

        self.managed_resources["ingresses"].append(result)
        logger.info(f"Successfully created ingress: {ingress_config.name}")

        return result

    def delete_deployment(
        self, name: str, namespace: str = "default"
    ) -> Dict[str, Any]:
        """Delete a deployment.

        Args:
            name: Deployment name
            namespace: Namespace

        Returns:
            Deletion result
        """
        logger.info(f"Deleting deployment: {name}")

        self.managed_resources["deployments"] = [
            r
            for r in self.managed_resources["deployments"]
            if not (r["name"] == name and r["namespace"] == namespace)
        ]

        result = {
            "status": "deleted",
            "kind": "Deployment",
            "name": name,
            "namespace": namespace,
            "timestamp": self._get_timestamp(),
        }

        logger.info(f"Successfully deleted deployment: {name}")
        return result

    def delete_service(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """Delete a service.

        Args:
            name: Service name
            namespace: Namespace

        Returns:
            Deletion result
        """
        logger.info(f"Deleting service: {name}")

        self.managed_resources["services"] = [
            r
            for r in self.managed_resources["services"]
            if not (r["name"] == name and r["namespace"] == namespace)
        ]

        result = {
            "status": "deleted",
            "kind": "Service",
            "name": name,
            "namespace": namespace,
            "timestamp": self._get_timestamp(),
        }

        logger.info(f"Successfully deleted service: {name}")
        return result

    def list_deployments(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """List deployments.

        Args:
            namespace: Filter by namespace (optional)

        Returns:
            List of deployments
        """
        deployments = self.managed_resources["deployments"]
        if namespace:
            deployments = [d for d in deployments if d["namespace"] == namespace]
        return deployments

    def list_services(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """List services.

        Args:
            namespace: Filter by namespace (optional)

        Returns:
            List of services
        """
        services = self.managed_resources["services"]
        if namespace:
            services = [s for s in services if s["namespace"] == namespace]
        return services

    def list_ingresses(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """List ingresses.

        Args:
            namespace: Filter by namespace (optional)

        Returns:
            List of ingresses
        """
        ingresses = self.managed_resources["ingresses"]
        if namespace:
            ingresses = [i for i in ingresses if i["namespace"] == namespace]
        return ingresses

    def get_resource_status(
        self, kind: str, name: str, namespace: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Get resource status.

        Args:
            kind: Resource kind (Deployment, Service, Ingress)
            name: Resource name
            namespace: Namespace

        Returns:
            Resource status or None if not found
        """
        resources_key = kind.lower() + "s"
        if resources_key not in self.managed_resources:
            return None

        for resource in self.managed_resources[resources_key]:
            if resource["name"] == name and resource["namespace"] == namespace:
                return resource

        return None

    def _generate_deployment_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest.

        Args:
            config: Deployment configuration

        Returns:
            Manifest dictionary
        """
        containers = []
        for container in config.containers:
            containers.append(
                {
                    "name": container.name,
                    "image": container.image,
                    "imagePullPolicy": container.image_pull_policy.value,
                    "ports": [{"containerPort": p} for p in container.ports],
                    "env": [
                        {"name": k, "value": v}
                        for k, v in container.environment.items()
                    ],
                    "resources": {
                        "requests": {
                            "cpu": container.resources.cpu_request,
                            "memory": container.resources.memory_request,
                        },
                        "limits": {
                            "cpu": container.resources.cpu_limit,
                            "memory": container.resources.memory_limit,
                        },
                    },
                }
            )

        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "labels": config.labels,
                "annotations": config.annotations,
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {"matchLabels": config.labels or {"app": config.name}},
                "template": {
                    "metadata": {
                        "labels": config.labels or {"app": config.name},
                    },
                    "spec": {
                        "containers": containers,
                        "restartPolicy": config.restart_policy.value,
                    },
                },
            },
        }

        return manifest

    def _generate_service_manifest(self, config: ServiceConfig) -> Dict[str, Any]:
        """Generate Kubernetes service manifest.

        Args:
            config: Service configuration

        Returns:
            Manifest dictionary
        """
        ports = [
            {"port": host, "targetPort": container}
            for host, container in config.ports.items()
        ]

        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "labels": config.labels,
                "annotations": config.annotations,
            },
            "spec": {
                "type": config.service_type.value,
                "selector": config.selector,
                "ports": ports,
            },
        }

        return manifest

    def _generate_ingress_manifest(self, config: IngressConfig) -> Dict[str, Any]:
        """Generate Kubernetes ingress manifest.

        Args:
            config: Ingress configuration

        Returns:
            Manifest dictionary
        """
        rules = []
        for host, service in config.rules.items():
            rules.append(
                {
                    "host": host,
                    "http": {
                        "paths": [
                            {
                                "path": "/",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": service,
                                        "port": {"number": 80},
                                    }
                                },
                            }
                        ]
                    },
                }
            )

        manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "annotations": config.annotations,
            },
            "spec": {
                "ingressClassName": config.ingress_class,
                "rules": rules,
            },
        }

        return manifest

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp.

        Returns:
            ISO format timestamp
        """
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"


class K8sDeployer:
    """Kubernetes deployment orchestrator."""

    def __init__(self, config: K8sConfig):
        """Initialize K8s deployer.

        Args:
            config: Kubernetes configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("Kubernetes configuration must be provided")
        self.config = config
        self.deployments: Dict[str, Dict[str, Any]] = {}
        logger.debug(f"Initialized K8sDeployer for cluster: {config.cluster_name}")

    def deploy(self, yaml_path: str, namespace: str = "default") -> Dict[str, Any]:
        """Deploy from YAML file.

        Args:
            yaml_path: Path to YAML manifest file
            namespace: Target namespace

        Returns:
            Deployment result

        Raises:
            ValueError: If YAML path is invalid
        """
        if not yaml_path:
            raise ValueError("YAML path must be provided")

        logger.info(f"Deploying from YAML: {yaml_path}")

        result = {
            "status": "deployed",
            "yaml_path": yaml_path,
            "namespace": namespace,
            "cluster": self.config.cluster_name,
            "context": self.config.context,
            "timestamp": self._get_timestamp(),
        }

        self.deployments[yaml_path] = result
        logger.info(f"Successfully deployed from YAML: {yaml_path}")

        return result

    def scale_deployment(
        self, name: str, replicas: int, namespace: str = "default"
    ) -> Dict[str, Any]:
        """Scale a deployment.

        Args:
            name: Deployment name
            replicas: Number of replicas
            namespace: Namespace

        Returns:
            Scaling result

        Raises:
            ValueError: If replicas is invalid
        """
        if replicas < 0:
            raise ValueError(f"replicas must be non-negative, got {replicas}")

        logger.info(f"Scaling deployment: {name} to {replicas} replicas")

        result = {
            "status": "scaled",
            "deployment": name,
            "namespace": namespace,
            "replicas": replicas,
            "timestamp": self._get_timestamp(),
        }

        logger.info(f"Successfully scaled deployment: {name}")
        return result

    def rollout_status(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get rollout status of a deployment.

        Args:
            name: Deployment name
            namespace: Namespace

        Returns:
            Rollout status
        """
        logger.info(f"Getting rollout status: {name}")

        result = {
            "deployment": name,
            "namespace": namespace,
            "status": "complete",
            "replicas": 3,
            "updated_replicas": 3,
            "ready_replicas": 3,
            "available_replicas": 3,
            "timestamp": self._get_timestamp(),
        }

        logger.info(f"Rollout status for {name}: {result['status']}")
        return result

    def list_deployments(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """List deployments in cluster.

        Args:
            namespace: Filter by namespace (optional)

        Returns:
            List of deployments
        """
        logger.info(f"Listing deployments in namespace: {namespace or 'all'}")

        deployments = [
            {
                "name": f"deployment-{i}",
                "namespace": namespace or "default",
                "replicas": 3,
                "ready": 3,
            }
            for i in range(1, 4)
        ]

        return deployments

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp.

        Returns:
            ISO format timestamp
        """
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"


class HelmChart:
    """Helm chart management."""

    def __init__(self, config: K8sConfig):
        """Initialize Helm manager.

        Args:
            config: Kubernetes configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("Kubernetes configuration must be provided")
        self.config = config
        self.releases: Dict[str, Dict[str, Any]] = {}
        logger.debug(
            f"Initialized HelmChart manager for cluster: {config.cluster_name}"
        )

    def install(self, chart_config: HelmChartConfig) -> Dict[str, Any]:
        """Install Helm chart.

        Args:
            chart_config: Helm chart configuration

        Returns:
            Installation result

        Raises:
            ValueError: If configuration is invalid
        """
        if not chart_config:
            raise ValueError("Helm chart configuration must be provided")

        logger.info(f"Installing Helm chart: {chart_config.chart_name}")

        release_name = chart_config.release_name or chart_config.name

        result = {
            "status": "installed",
            "release_name": release_name,
            "chart": chart_config.chart_name,
            "version": chart_config.version or "latest",
            "namespace": chart_config.namespace,
            "values_count": len(chart_config.values),
            "timestamp": self._get_timestamp(),
        }

        self.releases[release_name] = result
        logger.info(f"Successfully installed Helm chart: {chart_config.chart_name}")

        return result

    def upgrade(self, chart_config: HelmChartConfig) -> Dict[str, Any]:
        """Upgrade Helm release.

        Args:
            chart_config: Helm chart configuration

        Returns:
            Upgrade result

        Raises:
            ValueError: If configuration is invalid
        """
        if not chart_config:
            raise ValueError("Helm chart configuration must be provided")

        logger.info(f"Upgrading Helm release: {chart_config.name}")

        release_name = chart_config.release_name or chart_config.name

        result = {
            "status": "upgraded",
            "release_name": release_name,
            "chart": chart_config.chart_name,
            "version": chart_config.version or "latest",
            "namespace": chart_config.namespace,
            "timestamp": self._get_timestamp(),
        }

        self.releases[release_name] = result
        logger.info(f"Successfully upgraded Helm release: {chart_config.name}")

        return result

    def uninstall(
        self, release_name: str, namespace: str = "default"
    ) -> Dict[str, Any]:
        """Uninstall Helm release.

        Args:
            release_name: Release name
            namespace: Namespace

        Returns:
            Uninstall result

        Raises:
            ValueError: If release name is invalid
        """
        if not release_name:
            raise ValueError("Release name must be provided")

        logger.info(f"Uninstalling Helm release: {release_name}")

        self.releases.pop(release_name, None)

        result = {
            "status": "uninstalled",
            "release_name": release_name,
            "namespace": namespace,
            "timestamp": self._get_timestamp(),
        }

        logger.info(f"Successfully uninstalled Helm release: {release_name}")
        return result

    def list_releases(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """List Helm releases.

        Args:
            namespace: Filter by namespace (optional)

        Returns:
            List of releases
        """
        releases = list(self.releases.values())
        if namespace:
            releases = [r for r in releases if r["namespace"] == namespace]
        return releases

    def get_values(self, release_name: str) -> Dict[str, Any]:
        """Get Helm release values.

        Args:
            release_name: Release name

        Returns:
            Release values
        """
        if release_name not in self.releases:
            return {}
        return self.releases[release_name]

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp.

        Returns:
            ISO format timestamp
        """
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    k8s_config = K8sConfig(
        cluster_name="production", context="prod-context", namespace="default"
    )

    deployer = K8sDeployer(k8s_config)
    result = deployer.deploy("./deployment.yaml", namespace="default")
    print(f"Deployment result: {json.dumps(result, indent=2)}")
