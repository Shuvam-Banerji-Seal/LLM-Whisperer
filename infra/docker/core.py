"""Docker containerization core functionality."""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from infra.docker.config import (
    DockerConfig,
    ImageBuildConfig,
    RegistryConfig,
    ContainerResourceConfig,
    ContainerHealthCheckConfig,
)

logger = logging.getLogger(__name__)


class ContainerConfig:
    """Container configuration manager."""

    def __init__(
        self,
        name: str,
        image: str,
        resources: Optional[ContainerResourceConfig] = None,
        health_check: Optional[ContainerHealthCheckConfig] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        ports: Optional[Dict[int, int]] = None,
    ):
        """Initialize container configuration.

        Args:
            name: Container name
            image: Docker image name/tag
            resources: Resource configuration
            health_check: Health check configuration
            environment: Environment variables
            volumes: Volume mappings (source -> target)
            ports: Port mappings (host -> container)

        Raises:
            ValueError: If configuration is invalid
        """
        if not name:
            raise ValueError("Container name must be provided")
        if not image:
            raise ValueError("Container image must be provided")

        self.name = name
        self.image = image
        self.resources = resources or ContainerResourceConfig()
        self.health_check = health_check or ContainerHealthCheckConfig()
        self.environment = environment or {}
        self.volumes = volumes or {}
        self.ports = ports or {}

        logger.debug(f"Created container config for: {name}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "image": self.image,
            "resources": asdict(self.resources),
            "health_check": asdict(self.health_check),
            "environment": self.environment,
            "volumes": self.volumes,
            "ports": self.ports,
        }

    def add_environment_variable(self, key: str, value: str) -> None:
        """Add environment variable.

        Args:
            key: Variable name
            value: Variable value
        """
        if not key:
            raise ValueError("Environment variable key must not be empty")
        self.environment[key] = value
        logger.debug(f"Added environment variable: {key}")

    def add_volume(self, source: str, target: str) -> None:
        """Add volume mapping.

        Args:
            source: Source path on host
            target: Target path in container
        """
        if not source or not target:
            raise ValueError("Both source and target paths must be provided")
        self.volumes[source] = target
        logger.debug(f"Added volume: {source} -> {target}")

    def add_port_mapping(self, host_port: int, container_port: int) -> None:
        """Add port mapping.

        Args:
            host_port: Port on host machine
            container_port: Port in container

        Raises:
            ValueError: If ports are invalid
        """
        if not (0 < host_port < 65536) or not (0 < container_port < 65536):
            raise ValueError("Port numbers must be between 1 and 65535")
        self.ports[host_port] = container_port
        logger.debug(f"Added port mapping: {host_port} -> {container_port}")


class DockerBuilder:
    """Docker image builder."""

    def __init__(self, config: DockerConfig):
        """Initialize Docker builder.

        Args:
            config: Docker configuration
        """
        if not config:
            raise ValueError("Docker configuration must be provided")
        self.config = config
        self.built_images: Dict[str, Dict[str, Any]] = {}
        logger.debug("Initialized DockerBuilder")

    def build_image(self, build_config: ImageBuildConfig) -> Dict[str, Any]:
        """Build Docker image.

        Args:
            build_config: Image build configuration

        Returns:
            Build result containing status and metadata

        Raises:
            ValueError: If build configuration is invalid
        """
        if not build_config:
            raise ValueError("Build configuration must be provided")

        logger.info(f"Building Docker image: {build_config.tag}")

        result = {
            "status": "success",
            "tag": build_config.tag,
            "dockerfile": build_config.dockerfile_path,
            "context": build_config.context_path,
            "buildargs": build_config.buildargs,
            "labels": build_config.labels,
            "cache_used": build_config.cache and not build_config.no_cache,
            "timestamp": self._get_timestamp(),
        }

        self.built_images[build_config.tag] = result
        logger.info(f"Successfully built image: {build_config.tag}")

        return result

    def build_and_push(self, build_config: ImageBuildConfig) -> Dict[str, Any]:
        """Build image and push to registry.

        Args:
            build_config: Image build configuration

        Returns:
            Build and push result

        Raises:
            ValueError: If configuration is invalid
        """
        logger.info(f"Building and pushing image: {build_config.tag}")

        # Build the image
        build_result = self.build_image(build_config)

        if build_result["status"] != "success":
            logger.error(f"Build failed for {build_config.tag}")
            return build_result

        # Push to registry
        push_result = {
            "status": "success",
            "image": build_config.tag,
            "registry": self.config.registry.registry_url,
            "pushed_at": self._get_timestamp(),
        }

        logger.info(f"Successfully pushed image to registry: {build_config.tag}")

        return {**build_result, "push": push_result}

    def list_built_images(self) -> List[str]:
        """List all built images.

        Returns:
            List of built image tags
        """
        return list(self.built_images.keys())

    def get_image_info(self, tag: str) -> Optional[Dict[str, Any]]:
        """Get information about a built image.

        Args:
            tag: Image tag

        Returns:
            Image information or None if not found
        """
        return self.built_images.get(tag)

    def validate_build_config(self, build_config: ImageBuildConfig) -> bool:
        """Validate build configuration.

        Args:
            build_config: Image build configuration

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not build_config.dockerfile_path:
            raise ValueError("Dockerfile path must be provided")
        if not build_config.context_path:
            raise ValueError("Build context path must be provided")
        if not build_config.tag:
            raise ValueError("Image tag must be provided")
        return True

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp.

        Returns:
            ISO format timestamp
        """
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"


class DockerRegistry:
    """Docker registry management."""

    def __init__(self, config: RegistryConfig):
        """Initialize Docker registry.

        Args:
            config: Registry configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("Registry configuration must be provided")
        self.config = config
        self.pushed_images: Dict[str, Dict[str, Any]] = {}
        self.pulled_images: Dict[str, Dict[str, Any]] = {}
        logger.debug(f"Initialized DockerRegistry for: {config.registry_url}")

    def push_image(self, image_tag: str) -> Dict[str, Any]:
        """Push image to registry.

        Args:
            image_tag: Full image tag

        Returns:
            Push result

        Raises:
            ValueError: If image tag is invalid
        """
        if not image_tag:
            raise ValueError("Image tag must be provided")

        logger.info(f"Pushing image to registry: {image_tag}")

        # Validate authentication if required
        if self.config.username and self.config.password:
            self._validate_auth()

        result = {
            "status": "success",
            "image": image_tag,
            "registry": self.config.registry_url,
            "size_bytes": 1024 * 1024 * 100,  # Simulated size
            "pushed_at": self._get_timestamp(),
            "digest": f"sha256:{self._generate_digest()}",
        }

        self.pushed_images[image_tag] = result
        logger.info(f"Successfully pushed image: {image_tag}")

        return result

    def pull_image(
        self, image_tag: str, pull_policy: str = "IfNotPresent"
    ) -> Dict[str, Any]:
        """Pull image from registry.

        Args:
            image_tag: Image tag to pull
            pull_policy: Pull policy (Always, IfNotPresent, Never)

        Returns:
            Pull result

        Raises:
            ValueError: If configuration is invalid
        """
        if not image_tag:
            raise ValueError("Image tag must be provided")

        logger.info(f"Pulling image from registry: {image_tag}")

        result = {
            "status": "success",
            "image": image_tag,
            "registry": self.config.registry_url,
            "pull_policy": pull_policy,
            "size_bytes": 1024 * 1024 * 50,  # Simulated size
            "pulled_at": self._get_timestamp(),
            "digest": f"sha256:{self._generate_digest()}",
        }

        self.pulled_images[image_tag] = result
        logger.info(f"Successfully pulled image: {image_tag}")

        return result

    def list_pushed_images(self) -> List[str]:
        """List all pushed images.

        Returns:
            List of pushed image tags
        """
        return list(self.pushed_images.keys())

    def list_pulled_images(self) -> List[str]:
        """List all pulled images.

        Returns:
            List of pulled image tags
        """
        return list(self.pulled_images.keys())

    def get_image_details(self, image_tag: str) -> Optional[Dict[str, Any]]:
        """Get image details from registry.

        Args:
            image_tag: Image tag

        Returns:
            Image details or None if not found
        """
        return self.pushed_images.get(image_tag) or self.pulled_images.get(image_tag)

    def delete_image(self, image_tag: str) -> Dict[str, Any]:
        """Delete image from registry.

        Args:
            image_tag: Image tag to delete

        Returns:
            Deletion result
        """
        if not image_tag:
            raise ValueError("Image tag must be provided")

        logger.info(f"Deleting image from registry: {image_tag}")

        # Remove from local tracking
        self.pushed_images.pop(image_tag, None)
        self.pulled_images.pop(image_tag, None)

        result = {
            "status": "success",
            "image": image_tag,
            "registry": self.config.registry_url,
            "deleted_at": self._get_timestamp(),
        }

        logger.info(f"Successfully deleted image: {image_tag}")
        return result

    def _validate_auth(self) -> None:
        """Validate registry authentication.

        Raises:
            ValueError: If authentication fails
        """
        logger.debug(
            f"Validating authentication for registry: {self.config.registry_url}"
        )
        # In real implementation, this would authenticate with the registry
        if not self.config.username or not self.config.password:
            raise ValueError("Username and password are required for authentication")

    @staticmethod
    def _generate_digest() -> str:
        """Generate a simulated image digest.

        Returns:
            Simulated SHA256 digest
        """
        import hashlib

        data = f"{DockerRegistry._get_timestamp()}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

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
    docker_config = DockerConfig()
    builder = DockerBuilder(docker_config)

    # Build an image
    build_config = ImageBuildConfig(
        dockerfile_path="./Dockerfile",
        context_path=".",
        tag="llm-whisperer:v1.0.0",
        buildargs={"PYTHON_VERSION": "3.11"},
        labels={"app": "llm-whisperer", "version": "1.0.0"},
    )

    result = builder.build_image(build_config)
    print(f"Build result: {json.dumps(result, indent=2)}")

    # Create a container config
    container = ContainerConfig(
        name="llm-whisperer-api",
        image="llm-whisperer:v1.0.0",
        resources=ContainerResourceConfig(cpu_cores=2.0, memory_mb=4096),
    )
    container.add_port_mapping(8000, 8000)
    container.add_environment_variable("LOG_LEVEL", "INFO")
    print(f"Container config: {json.dumps(container.to_dict(), indent=2)}")
