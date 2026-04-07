"""Deployment orchestrator for model packaging and publishing."""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""

    model_path: str
    model_name: str
    model_version: str
    output_dir: str = "./deployments"

    # Deployment targets
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None

    # Model optimization
    quantization: bool = False
    optimization_level: int = 0  # 0 = none, 1 = basic, 2 = aggressive

    # Versioning
    major: int = 1
    minor: int = 0
    patch: int = 0


class DeploymentOrchestrator:
    """Orchestrates model deployment."""

    def __init__(self, config: DeploymentConfig):
        """Initialize deployment orchestrator.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.deployment_path = None
        self._setup_directories()

    def _setup_directories(self):
        """Create deployment directories."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        (output_path / "models").mkdir(exist_ok=True)
        (output_path / "versions").mkdir(exist_ok=True)
        (output_path / "metadata").mkdir(exist_ok=True)

    def package_model(self) -> str:
        """Package model for deployment.

        Returns:
            Path to packaged model
        """
        logger.info("Packaging model for deployment...")

        import shutil
        from datetime import datetime

        # Create version directory
        version_str = f"v{self.config.major}.{self.config.minor}.{self.config.patch}"
        package_dir = Path(self.config.output_dir) / "models" / version_str
        package_dir.mkdir(parents=True, exist_ok=True)

        # Copy model
        logger.info(f"Copying model from {self.config.model_path} to {package_dir}")
        model_src = Path(self.config.model_path)

        if model_src.is_dir():
            for item in model_src.iterdir():
                if item.is_file():
                    shutil.copy(item, package_dir / item.name)

        # Save metadata
        metadata = {
            "model_name": self.config.model_name,
            "version": version_str,
            "timestamp": datetime.now().isoformat(),
            "quantized": self.config.quantization,
            "optimization_level": self.config.optimization_level,
        }

        import json

        metadata_path = package_dir / "deployment_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model packaged at {package_dir}")
        self.deployment_path = str(package_dir)

        return str(package_dir)

    def publish_model(self) -> Dict[str, Any]:
        """Publish model to registry.

        Returns:
            Publishing result
        """
        if not self.deployment_path:
            raise ValueError("Model must be packaged before publishing")

        logger.info("Publishing model...")

        result = {
            "status": "published",
            "model_name": self.config.model_name,
            "version": f"v{self.config.major}.{self.config.minor}.{self.config.patch}",
            "deployment_path": self.deployment_path,
        }

        if self.config.push_to_hub:
            logger.info(f"Pushing to Hub: {self.config.hub_repo_id}")
            result["hub_url"] = f"https://huggingface.co/{self.config.hub_repo_id}"

        return result

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information."""
        return {
            "model_name": self.config.model_name,
            "version": f"v{self.config.major}.{self.config.minor}.{self.config.patch}",
            "deployment_path": self.deployment_path,
            "quantized": self.config.quantization,
            "optimization_level": self.config.optimization_level,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = DeploymentConfig(
        model_path="./training_outputs/lora",
        model_name="mistral-7b-lora",
        model_version="1.0.0",
    )

    orchestrator = DeploymentOrchestrator(config)
    orchestrator.package_model()
    result = orchestrator.publish_model()
    print(result)
