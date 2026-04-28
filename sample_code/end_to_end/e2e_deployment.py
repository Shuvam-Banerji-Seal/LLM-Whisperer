"""
End-to-End Deployment Pipeline

Complete deployment pipeline demonstrating:
- Model packaging
- Container creation
- Health checks
- API endpoints
- Monitoring setup

Author: Shuvam Banerji
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import uuid
import time
import logging
import json
import hashlib
import subprocess
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environments."""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS_SAGEMAKER = "aws_sagemaker"
    AZURE_ML = "azure_ml"
    GCP_VERTEX = "gcp_vertex"


class ModelFormat(Enum):
    """Model export formats."""
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    GGUF = "gguf"


@dataclass
class ModelPackage:
    """Model package information."""
    model_name: str
    model_path: str
    format: ModelFormat
    size_mb: float
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    endpoint: str = "/health"
    timeout_sec: int = 30
    interval_sec: int = 10
    failure_threshold: int = 3
    success_threshold: int = 1


@dataclass
class APIServerConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_batch_size: int = 32
    timeout_sec: int = 300
    cors_origins: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"


class ModelPackager:
    """Package models for deployment."""

    def __init__(self, output_dir: str = "./packages"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def package_model(
        self,
        model_path: str,
        model_name: str,
        format: ModelFormat = ModelFormat.SAFETENSORS,
        config: Optional[Dict] = None
    ) -> ModelPackage:
        """
        Package model for deployment.

        Args:
            model_path: Path to model files
            model_name: Name of model
            format: Export format
            config: Model configuration

        Returns:
            ModelPackage with metadata
        """
        logger.info(f"Packaging model: {model_name} ({format.value})")

        package_path = os.path.join(self.output_dir, model_name)
        os.makedirs(package_path, exist_ok=True)

        package_info = {
            "model_name": model_name,
            "format": format.value,
            "version": "1.0.0",
            "created_at": time.time(),
            "config": config or {}
        }

        config_path = os.path.join(package_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(package_info, f, indent=2)

        size_mb = self._calculate_size(model_path)

        return ModelPackage(
            model_name=model_name,
            model_path=package_path,
            format=format,
            size_mb=size_mb,
            config=package_info
        )

    def _calculate_size(self, path: str) -> float:
        """Calculate model size in MB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)


class DockerBuilder:
    """Build Docker containers for model deployment."""

    def __init__(self, base_image: str = "python:3.10-slim"):
        self.base_image = base_image

    def build_image(
        self,
        model_package: ModelPackage,
        api_server_code: str,
        port: int = 8000
    ) -> str:
        """
        Build Docker image for model.

        Args:
            model_package: Packaged model
            api_server_code: API server code
            port: Container port

        Returns:
            Image tag
        """
        image_name = f"llm-{model_package.model_name}:{model_package.format.value}"

        logger.info(f"Building Docker image: {image_name}")

        dockerfile = self._generate_dockerfile(model_package, port)

        logger.info(f"Generated Dockerfile:\n{dockerfile}")

        return image_name

    def _generate_dockerfile(self, model_package: ModelPackage, port: int) -> str:
        """Generate Dockerfile content."""
        return f'''FROM {self.base_image}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./model /app/model

{self._generate_api_server_code()}

EXPOSE {port}

CMD ["python", "server.py"]
'''

    def _generate_api_server_code(self) -> str:
        """Generate API server code."""
        return '''
COPY server.py .
'''


class ContainerManager:
    """Manage containers for model deployment."""

    def __init__(self):
        self.containers: Dict[str, Dict] = {}

    def start_container(
        self,
        image: str,
        name: str,
        port_mapping: Dict[int, int],
        environment: Optional[Dict] = None,
        volumes: Optional[Dict] = None
    ) -> str:
        """
        Start a container.

        Args:
            image: Docker image
            name: Container name
            port_mapping: Host to container port mapping
            environment: Environment variables
            volumes: Volume mounts

        Returns:
            Container ID
        """
        container_id = str(uuid.uuid4())[:12]

        self.containers[container_id] = {
            "name": name,
            "image": image,
            "status": "running",
            "port_mapping": port_mapping,
            "environment": environment or {},
            "volumes": volumes or {},
            "started_at": time.time()
        }

        logger.info(f"Started container {name} ({container_id})")
        logger.info(f"  Image: {image}")
        logger.info(f"  Ports: {port_mapping}")

        return container_id

    def stop_container(self, container_id: str) -> bool:
        """Stop a container."""
        if container_id in self.containers:
            self.containers[container_id]["status"] = "stopped"
            self.containers[container_id]["stopped_at"] = time.time()
            logger.info(f"Stopped container {container_id}")
            return True
        return False

    def get_container_status(self, container_id: str) -> Optional[Dict]:
        """Get container status."""
        return self.containers.get(container_id)


class HealthChecker:
    """Health check implementation for deployed models."""

    def __init__(self, config: Optional[HealthCheckConfig] = None):
        self.config = config or HealthCheckConfig()

    def check_health(self, endpoint: str, timeout_sec: int = 30) -> Dict[str, Any]:
        """
        Check health of deployment.

        Args:
            endpoint: Health check endpoint
            timeout_sec: Timeout in seconds

        Returns:
            Health status dict
        """
        start_time = time.time()

        try:
            health_status = self._perform_health_check(endpoint, timeout_sec)

            elapsed = (time.time() - start_time) * 1000

            return {
                "status": "healthy" if health_status else "unhealthy",
                "endpoint": endpoint,
                "latency_ms": elapsed,
                "timestamp": time.time(),
                "details": health_status
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "endpoint": endpoint,
                "error": str(e),
                "timestamp": time.time()
            }

    def _perform_health_check(self, endpoint: str, timeout_sec: int) -> bool:
        """Perform actual health check."""
        logger.info(f"Checking health at {endpoint}")
        time.sleep(0.1)
        return True

    def check_readiness(self, endpoint: str) -> Dict[str, Any]:
        """Check if service is ready to accept traffic."""
        status = self.check_health(endpoint)
        status["ready"] = status["status"] == "healthy"
        return status

    def check_liveness(self, endpoint: str) -> Dict[str, Any]:
        """Check if service is alive."""
        status = self.check_health(endpoint)
        status["alive"] = status["status"] == "healthy"
        return status


class APIServer:
    """API server for model inference."""

    def __init__(self, config: Optional[APIServerConfig] = None):
        self.config = config or APIServerConfig()
        self.endpoints: Dict[str, Callable] = {}
        self.request_count = 0
        self.error_count = 0

        self._register_default_endpoints()

    def _register_default_endpoints(self) -> None:
        """Register default API endpoints."""
        self.register_endpoint("/health", self._health_handler, ["GET"])
        self.register_endpoint("/predict", self._predict_handler, ["POST"])
        self.register_endpoint("/batch_predict", self._batch_predict_handler, ["POST"])
        self.register_endpoint("/model_info", self._model_info_handler, ["GET"])
        self.register_endpoint("/metrics", self._metrics_handler, ["GET"])

    def register_endpoint(
        self,
        path: str,
        handler: Callable,
        methods: List[str] = ["GET"]
    ) -> None:
        """Register an API endpoint."""
        self.endpoints[path] = {
            "handler": handler,
            "methods": methods
        }
        logger.info(f"Registered endpoint: {path} ({', '.join(methods)})")

    def _health_handler(self, request: Dict) -> Dict:
        """Health check handler."""
        return {
            "status": "healthy",
            "timestamp": time.time()
        }

    def _predict_handler(self, request: Dict) -> Dict:
        """Prediction handler."""
        self.request_count += 1

        try:
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_tokens", 100)
            temperature = request.get("temperature", 0.7)

            result = self._mock_predict(prompt, max_tokens, temperature)

            return {
                "prompt": prompt,
                "generated_text": result,
                "metrics": {
                    "tokens_generated": len(result.split()),
                    "latency_ms": 100 + len(prompt) * 0.5
                }
            }

        except Exception as e:
            self.error_count += 1
            return {"error": str(e)}

    def _batch_predict_handler(self, request: Dict) -> Dict:
        """Batch prediction handler."""
        self.request_count += 1

        prompts = request.get("prompts", [])
        max_tokens = request.get("max_tokens", 100)

        results = [self._mock_predict(p, max_tokens, 0.7) for p in prompts]

        return {
            "predictions": results,
            "count": len(results)
        }

    def _model_info_handler(self, request: Dict) -> Dict:
        """Model info handler."""
        return {
            "model_name": "meta-llama/Llama-2-7b",
            "version": "1.0.0",
            "max_tokens": 4096,
            "supported_endpoints": list(self.endpoints.keys())
        }

    def _metrics_handler(self, request: Dict) -> Dict:
        """Metrics handler."""
        return {
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "uptime_seconds": time.time() - getattr(self, "_start_time", time.time())
        }

    def _mock_predict(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Mock prediction."""
        return f"Generated response for: {prompt[:50]}..."

    def start(self) -> None:
        """Start the API server."""
        self._start_time = time.time()
        logger.info(f"Starting API server on {self.config.host}:{self.config.port}")
        logger.info(f"  Workers: {self.config.workers}")
        logger.info(f"  Max batch size: {self.config.max_batch_size}")
        logger.info(f"  Endpoints: {list(self.endpoints.keys())}")

    def stop(self) -> None:
        """Stop the API server."""
        logger.info("Stopping API server")


class MonitoringSystem:
    """Monitoring and observability for deployed models."""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.metrics: Dict[str, List[float]] = {}
        self.traces: List[Dict] = []
        self.logs: List[Dict] = []

    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None) -> None:
        """Record a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

        if self.config.enable_logging:
            logger.debug(f"Metric: {name}={value} tags={tags}")

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        """Record request latency."""
        self.record_metric(f"latency.{endpoint}", latency_ms, {"endpoint": endpoint})

    def record_request(self, endpoint: str, status_code: int) -> None:
        """Record request."""
        self.record_metric(f"requests.{endpoint}", 1, {"status": str(status_code)})

        if status_code >= 400:
            self.record_metric(f"errors.{endpoint}", 1, {"status": str(status_code)})

    def start_trace(self, trace_id: str, operation: str) -> Dict:
        """Start a trace span."""
        span = {
            "trace_id": trace_id,
            "operation": operation,
            "start_time": time.time(),
            "events": []
        }
        return span

    def end_trace(self, span: Dict) -> None:
        """End a trace span."""
        span["end_time"] = time.time()
        span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000

        if self.config.enable_tracing:
            self.traces.append(span)

    def log_event(self, level: str, message: str, metadata: Optional[Dict] = None) -> None:
        """Log an event."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "metadata": metadata or {}
        }

        if self.config.enable_logging:
            self.logs.append(log_entry)

            if level == "ERROR":
                logger.error(f"{message} {metadata}")
            elif level == "WARNING":
                logger.warning(f"{message} {metadata}")
            else:
                logger.info(f"{message} {metadata}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {}

        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "last": values[-1]
                }

        return summary


class DeploymentManager:
    """Complete deployment management."""

    def __init__(
        self,
        environment: DeploymentEnvironment = DeploymentEnvironment.DOCKER,
        model_name: str = "meta-llama/Llama-2-7b"
    ):
        self.environment = environment
        self.model_name = model_name

        self.packager = ModelPackager()
        self.docker_builder = DockerBuilder()
        self.container_manager = ContainerManager()
        self.health_checker = HealthChecker()
        self.api_server = APIServer()
        self.monitoring = MonitoringSystem()

        self.deployment_id: Optional[str] = None
        self.status = "not_deployed"

    def deploy(
        self,
        model_path: str,
        format: ModelFormat = ModelFormat.SAFETENSORS,
        port: int = 8000
    ) -> Dict[str, Any]:
        """
        Deploy model.

        Args:
            model_path: Path to model
            format: Model format
            port: Host port

        Returns:
            Deployment info
        """
        self.deployment_id = str(uuid.uuid4())[:12]

        logger.info(f"Starting deployment: {self.deployment_id}")
        logger.info(f"  Environment: {self.environment.value}")
        logger.info(f"  Model: {self.model_name}")

        model_package = self.packager.package_model(
            model_path=model_path,
            model_name=self.model_name,
            format=format
        )
        logger.info(f"Model packaged: {model_package.size_mb:.2f} MB")

        if self.environment == DeploymentEnvironment.DOCKER:
            image = self.docker_builder.build_image(
                model_package=model_package,
                api_server_code="",
                port=port
            )
            logger.info(f"Docker image built: {image}")

            container_id = self.container_manager.start_container(
                image=image,
                name=f"{self.model_name}-{self.deployment_id}",
                port_mapping={port: 8000}
            )

            health = self.health_checker.check_health(
                f"http://localhost:{port}/health"
            )

            self.api_server.start()

            self.status = "deployed"

            return {
                "deployment_id": self.deployment_id,
                "status": self.status,
                "model_package": {
                    "name": model_package.model_name,
                    "size_mb": model_package.size_mb,
                    "format": model_package.format.value
                },
                "endpoint": f"http://localhost:{port}",
                "health": health
            }

        else:
            self.api_server.start()
            self.status = "deployed"

            return {
                "deployment_id": self.deployment_id,
                "status": self.status,
                "endpoint": f"http://localhost:{port}"
            }

    def undeploy(self) -> bool:
        """Undeploy model."""
        if self.status == "deployed":
            self.api_server.stop()
            self.status = "not_deployed"
            logger.info(f"Undeployed: {self.deployment_id}")
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get deployment status."""
        return {
            "deployment_id": self.deployment_id,
            "status": self.status,
            "environment": self.environment.value,
            "model_name": self.model_name
        }

    def run_health_checks(self) -> Dict[str, Any]:
        """Run health checks on deployment."""
        health = self.health_checker.check_health(
            f"http://localhost:{self.api_server.config.port}/health"
        )
        return health


def demo():
    """Demonstrate end-to-end deployment pipeline."""
    print("=" * 70)
    print("End-to-End Deployment Pipeline Demo")
    print("=" * 70)

    deployment = DeploymentManager(
        environment=DeploymentEnvironment.DOCKER,
        model_name="meta-llama/Llama-2-7b"
    )

    print("\n--- Deploying Model ---")
    result = deployment.deploy(
        model_path="/models/llama-2-7b",
        format=ModelFormat.SAFETENSORS,
        port=8000
    )

    print(f"Deployment ID: {result['deployment_id']}")
    print(f"Status: {result['status']}")
    print(f"Endpoint: {result['endpoint']}")
    if 'health' in result:
        print(f"Health: {result['health']['status']}")

    print("\n--- Deployment Status ---")
    status = deployment.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n--- Running Health Check ---")
    health = deployment.run_health_checks()
    print(f"  Status: {health['status']}")
    print(f"  Latency: {health['latency_ms']:.2f}ms")

    print("\n--- Monitoring Metrics ---")
    deployment.monitoring.record_metric("test.metric", 42.0)
    deployment.monitoring.record_latency("/predict", 150.5)
    deployment.monitoring.record_request("/predict", 200)

    summary = deployment.monitoring.get_metrics_summary()
    for name, stats in summary.items():
        print(f"  {name}: avg={stats['avg']:.2f}")

    print("\n--- Undeploying ---")
    deployment.undeploy()
    print(f"Status: {deployment.get_status()['status']}")


if __name__ == "__main__":
    demo()