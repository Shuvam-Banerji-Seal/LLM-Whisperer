"""Deployment orchestrator script."""

import logging
import argparse
from pathlib import Path

from src.orchestrator import DeploymentOrchestrator, DeploymentConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_deployment(
    model_path: str,
    model_name: str,
    version: str,
    output_dir: str = "./deployments",
    push_to_hub: bool = False,
    hub_repo_id: str = None,
):
    """Run model deployment."""
    logger.info("=" * 80)
    logger.info("LLM-Whisperer Deployment Pipeline")
    logger.info("=" * 80)

    # Parse version
    parts = version.split(".")
    major, minor, patch = (
        int(parts[0]),
        int(parts[1]) if len(parts) > 1 else 0,
        int(parts[2]) if len(parts) > 2 else 0,
    )

    # Configure deployment
    config = DeploymentConfig(
        model_path=model_path,
        model_name=model_name,
        model_version=version,
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
        major=major,
        minor=minor,
        patch=patch,
    )

    # Run deployment
    orchestrator = DeploymentOrchestrator(config)

    logger.info("\n[STEP 1] Package Model")
    logger.info("-" * 40)
    package_path = orchestrator.package_model()

    logger.info("\n[STEP 2] Publish Model")
    logger.info("-" * 40)
    result = orchestrator.publish_model()

    # Print deployment info
    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete")
    logger.info("=" * 80)

    for key, value in result.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--name", type=str, required=True, help="Model name")
    parser.add_argument("--version", type=str, default="1.0.0", help="Version")
    parser.add_argument("--output-dir", type=str, default="./deployments")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", type=str)

    args = parser.parse_args()

    run_deployment(
        args.model,
        args.name,
        args.version,
        args.output_dir,
        args.push_to_hub,
        args.hub_repo_id,
    )
