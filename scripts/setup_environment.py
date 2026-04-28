#!/usr/bin/env python3
"""
Complete environment setup.

This script sets up a complete development environment for
LLM Whisperer including:
- System dependencies
- Python package installation
- Model downloading
- Configuration setup
- Environment validation

Usage:
    python setup_environment.py --task full --models meta-llama/Llama-2-7b
    python setup_environment.py --task gpu --models gpt2
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SetupConfig:
    """Configuration for environment setup."""
    task: str = "full"
    models: List[str] = field(default_factory=list)
    install_dir: str = "./llm_whisperer_env"
    cache_dir: Optional[str] = None
    python_version: str = "3.10"
    force: bool = False
    skip_conda: bool = False
    skip_models: bool = False


def run_command(
    cmd: List[str],
    description: str,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a shell command.

    Args:
        cmd: Command to run
        description: Description for logging
        check: Whether to check return code
        capture: Whether to capture output

    Returns:
        Completed process
    """
    logger.info(f"Running: {description}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=check,
        )

        if result.returncode == 0:
            logger.info(f"Success: {description}")
        else:
            logger.error(f"Failed: {description}")
            if result.stderr:
                logger.error(f"Error: {result.stderr[:500]}")

        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Command not found: {e}")
        raise


def check_python_version() -> bool:
    """Check Python version.

    Returns:
        True if version is compatible, False otherwise
    """
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 9:
        return True

    logger.warning(f"Python 3.9+ recommended, found {version.major}.{version.minor}")
    return version.major == 3 and version.minor >= 8


def check_cuda_available() -> Dict[str, Any]:
    """Check CUDA availability.

    Returns:
        CUDA information dictionary
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            cuda_version = torch.version.cuda

            return {
                "available": True,
                "device_count": device_count,
                "device_name": device_name,
                "cuda_version": cuda_version,
            }
        else:
            return {"available": False}

    except ImportError:
        logger.warning("PyTorch not installed, cannot check CUDA")
        return {"available": False, "error": "PyTorch not installed"}


def install_system_dependencies() -> None:
    """Install system dependencies."""
    logger.info("Installing system dependencies")

    if os.path.exists("/etc/debian_version"):
        cmd = [
            "sudo", "apt-get", "update"
        ]
        run_command(cmd, "Update apt")

        dependencies = [
            "build-essential",
            "git",
            "curl",
            "wget",
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "libsm6",
            "libxext6",
            "libxrender-dev",
        ]

        cmd = [
            "sudo", "apt-get", "install", "-y"
        ] + dependencies

        run_command(cmd, "Install system packages")

    elif os.path.exists("/etc/redhat-release"):
        cmd = [
            "sudo", "yum", "install", "-y",
            "gcc", "gcc-c++", "make", "git", "curl",
        ]
        run_command(cmd, "Install system packages (RHEL/CentOS)")

    else:
        logger.warning("Unknown distribution, skipping system dependencies")


def create_venv(config: SetupConfig) -> None:
    """Create Python virtual environment.

    Args:
        config: Setup configuration
    """
    install_dir = Path(config.install_dir)

    if install_dir.exists() and not config.force:
        logger.info(f"Virtual environment already exists at {install_dir}")
        return

    logger.info(f"Creating virtual environment at {install_dir}")

    install_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "venv",
        config.install_dir,
    ]

    run_command(cmd, "Create virtual environment")

    python_path = Path(config.install_dir) / "bin" / "python"

    upgrade_cmd = [str(python_path), "-m", "pip", "upgrade", "pip"]
    run_command(upgrade_cmd, "Upgrade pip")


def install_python_packages(config: SetupConfig) -> None:
    """Install Python packages.

    Args:
        config: Setup configuration
    """
    python_path = Path(config.install_dir) / "bin" / "python"

    base_packages = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        " scipy>=1.11.0",
    ]

    logger.info("Installing base packages")
    cmd = [str(python_path), "-m", "pip", "install"] + base_packages
    run_command(cmd, "Install base packages")

    ml_packages = [
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "huggingface_hub>=0.21.0",
    ]

    logger.info("Installing ML packages")
    cmd = [str(python_path), "-m", "pip", "install"] + ml_packages
    run_command(cmd, "Install ML packages")

    inference_packages = [
        "vllm>=0.4.0",
        "optimum>=1.18.0",
        "onnxruntime>=1.17.0",
    ]

    logger.info("Installing inference packages")
    cmd = [str(python_path), "-m", "pip", "install"] + inference_packages
    run_command(cmd, "Install inference packages")

    dev_packages = [
        "black>=24.0.0",
        "ruff>=0.3.0",
        "mypy>=1.9.0",
        "pytest>=8.0.0",
        "pytest-cov>=4.1.0",
    ]

    logger.info("Installing dev packages")
    cmd = [str(python_path), "-m", "pip", "install"] + dev_packages
    run_command(cmd, "Install dev packages")


def download_models(model_ids: List[str], cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Download models from HuggingFace.

    Args:
        model_ids: List of model IDs
        cache_dir: Optional cache directory

    Returns:
        Download results
    """
    logger.info(f"Downloading {len(model_ids)} models")

    results = {}

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed")
        return {"error": "transformers not installed"}

    for model_id in model_ids:
        logger.info(f"Downloading model: {model_id}")

        try:
            kwargs = {"trust_remote_code": True}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir

            tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)

            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

            results[model_id] = {
                "status": "success",
                "model_type": model.config.model_type,
                "vocab_size": model.config.vocab_size,
            }

            logger.info(f"Successfully downloaded {model_id}")

        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            results[model_id] = {
                "status": "error",
                "error": str(e),
            }

    return results


def create_config_file(config: SetupConfig) -> None:
    """Create configuration file.

    Args:
        config: Setup configuration
    """
    config_path = Path(config.install_dir) / "config.json"

    setup_config = {
        "install_dir": config.install_dir,
        "python_version": config.python_version,
        "cache_dir": config.cache_dir,
        "models": config.models,
    }

    with open(config_path, "w") as f:
        json.dump(setup_config, f, indent=2)

    logger.info(f"Created config file: {config_path}")


def create_activation_script(config: SetupConfig) -> None:
    """Create environment activation script.

    Args:
        config: Setup configuration
    """
    scripts_dir = Path(config.install_dir) / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    activation_script = scripts_dir / "activate.sh"
    with open(activation_script, "w") as f:
        f.write(f"""#!/bin/bash
source {config.install_dir}/bin/activate
export PYTHONPATH="{config.install_dir}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages:$PYTHONPATH"
echo "Activated LLM Whisperer environment"
""")

    os.chmod(activation_script, 0o755)
    logger.info(f"Created activation script: {activation_script}")


def validate_environment(config: SetupConfig) -> bool:
    """Validate environment setup.

    Args:
        config: Setup configuration

    Returns:
        True if valid, False otherwise
    """
    logger.info("Validating environment setup")

    python_path = Path(config.install_dir) / "bin" / "python"

    if not python_path.exists():
        logger.error("Python executable not found")
        return False

    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
    ]

    all_valid = True

    for package in required_packages:
        cmd = [str(python_path), "-c", f"import {package}"]
        result = run_command(cmd, f"Check {package}", check=False)

        if result.returncode == 0:
            logger.info(f"  {package}: OK")
        else:
            logger.warning(f"  {package}: Not found")
            all_valid = False

    return all_valid


def main() -> int:
    """Main entry point for environment setup."""
    parser = argparse.ArgumentParser(
        description="Complete environment setup for LLM Whisperer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full setup with GPU support
    python setup_environment.py --task full --models meta-llama/Llama-2-7b

    # CPU-only setup
    python setup_environment.py --task cpu

    # Quick setup (skip model download)
    python setup_environment.py --task full --skip-models

    # Custom installation
    python setup_environment.py --task full --install-dir ./my_env --models gpt2 llama-2-7b
        """
    )

    parser.add_argument(
        "--task",
        choices=["full", "cpu", "gpu", "inference", "fine-tuning"],
        default="full",
        help="Setup task type (default: full)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to download"
    )
    parser.add_argument(
        "--install-dir",
        default="./llm_whisperer_env",
        help="Installation directory (default: ./llm_whisperer_env)"
    )
    parser.add_argument(
        "--cache-dir",
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstallation"
    )
    parser.add_argument(
        "--skip-conda",
        action="store_true",
        help="Skip conda installation"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model download"
    )

    args = parser.parse_args()

    config = SetupConfig(
        task=args.task,
        models=args.models or [],
        install_dir=args.install_dir,
        cache_dir=args.cache_dir,
        force=args.force,
        skip_conda=args.skip_conda,
        skip_models=args.skip_models,
    )

    try:
        if not check_python_version():
            logger.warning("Python version check failed")

        if config.task in ["full", "gpu"]:
            cuda_info = check_cuda_available()
            if config.task == "gpu" and not cuda_info.get("available"):
                logger.error("CUDA not available but GPU task requested")
                return 1

            logger.info(f"CUDA available: {cuda_info.get('available', False)}")
            if cuda_info.get("available"):
                logger.info(f"  Device: {cuda_info.get('device_name', 'unknown')}")
                logger.info(f"  CUDA version: {cuda_info.get('cuda_version', 'unknown')}")

        if config.task in ["full", "gpu", "cpu"]:
            install_system_dependencies()

        if config.task != "inference":
            create_venv(config)
            install_python_packages(config)

        if not config.skip_models and config.models:
            download_results = download_models(config.models, config.cache_dir)
            logger.info(f"Model download results: {json.dumps(download_results, indent=2)}")

        create_config_file(config)
        create_activation_script(config)

        if not validate_environment(config):
            logger.warning("Environment validation failed")
        else:
            logger.info("Environment validation passed")

        setup_info = {
            "status": "success",
            "install_dir": config.install_dir,
            "task": config.task,
            "models": config.models,
        }

        print(json.dumps(setup_info, indent=2))
        return 0

    except KeyboardInterrupt:
        logger.info("Setup cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())