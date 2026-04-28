#!/usr/bin/env python3
"""
Build Docker container for serving.

This script builds Docker containers for deploying LLM models
with optimized inference serving. Supports vLLM, TensorRT-LLM,
and other serving frameworks.

Usage:
    python build_container.py --model-id meta-llama/Llama-2-7b --framework vllm --output-tag llm-server:v1.0
    python build_container.py --model-id gpt2 --framework vllm --output-tag custom-model:latest --port 8000
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
class ContainerBuildConfig:
    """Configuration for container build."""
    model_id: str
    output_tag: str
    framework: str = "vllm"
    base_image: str = "nvidia/cuda:12.1.0-devel-ubuntu22.04"
    python_version: str = "3.10"
    port: int = 8000
    gpu_enabled: bool = True
    max_model_len: int = 4096
    tensor_parallel: int = 1
    build_args: Dict[str, str] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)


DOCKERFILE_TEMPLATES = {
    "vllm": """
FROM {base_image}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \\
    python3.10 python3-pip git curl \\
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app/
RUN pip3.10 install -r requirements.txt

COPY scripts/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENV MODEL_ID={model_id}
ENV PORT={port}
ENV MAX_MODEL_LEN={max_model_len}
ENV TENSOR_PARALLEL={tensor_parallel}

EXPOSE {port}

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--model", "$MODEL_ID", "--port", "$PORT", "--max-model-len", "$MAX_MODEL_LEN"]
""",
    "tensorrt_llm": """
FROM {base_image}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \\
    python3.10 python3-pip git curl libcurldev \\
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app/
RUN pip3.10 install -r requirements.txt

COPY scripts/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENV MODEL_ID={model_id}
ENV PORT={port}

EXPOSE {port}

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--model", "$MODEL_ID", "--port", "$PORT"]
""",
    "transformers": """
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \\
    git curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY scripts/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENV MODEL_ID={model_id}
ENV PORT={port}
ENV MAX_MODEL_LEN={max_model_len}

EXPOSE {port}

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--model", "$MODEL_ID", "--port", "$PORT"]
""",
}


REQUIREMENTS_TEMPLATES = {
    "vllm": [
        "vllm>=0.4.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "ray>=2.10.0",
    ],
    "tensorrt_llm": [
        "tensorrt_llm>=0.10.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
    ],
    "transformers": [
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.41.0",
        "torch>=2.2.0",
    ],
}


def validate_framework(framework: str) -> bool:
    """Validate serving framework.

    Args:
        framework: Framework name

    Returns:
        True if valid, False otherwise
    """
    return framework in DOCKERFILE_TEMPLATES


def create_requirements_file(output_dir: str, framework: str, extra_requirements: List[str]) -> None:
    """Create requirements.txt file.

    Args:
        output_dir: Output directory
        framework: Serving framework
        extra_requirements: Additional requirements
    """
    requirements = REQUIREMENTS_TEMPLATES.get(framework, [])
    requirements.extend(extra_requirements)

    requirements_path = Path(output_dir) / "requirements.txt"
    with open(requirements_path, "w") as f:
        f.write("\n".join(requirements) + "\n")

    logger.info(f"Created requirements.txt: {requirements_path}")


def create_entrypoint_script(output_dir: str, framework: str) -> None:
    """Create container entrypoint script.

    Args:
        output_dir: Output directory
        framework: Serving framework
    """
    scripts_dir = Path(output_dir) / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    if framework == "vllm":
        entrypoint = """#!/bin/bash
set -e

echo "Starting vLLM server with model: $MODEL_ID"
echo "Port: $PORT"
echo "Max model length: $MAX_MODEL_LEN"

if [ -n "$TENSOR_PARALLEL" ]; then
    TP_ARGS="--tensor-parallel-size $TENSOR_PARALLEL"
fi

exec python3.10 -m vllm.entrypoints.openai.api_server \\
    --model "$MODEL_ID" \\
    --port "$PORT" \\
    --host 0.0.0.0 \\
    --max-model-len "${MAX_MODEL_LEN:-4096}" \\
    $TP_ARGS \\
    "$@"
"""
    elif framework == "tensorrt_llm":
        entrypoint = """#!/bin/bash
set -e

echo "Starting TensorRT-LLM server with model: $MODEL_ID"
echo "Port: $PORT"

exec python3.10 -m tensorrt_llm.serving \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --host 0.0.0.0 \
    "$@"
"""
    else:
        entrypoint = """#!/bin/bash
set -e

echo "Starting transformers server with model: $MODEL_ID"
echo "Port: $PORT"

exec python3.10 -m transformers_server \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    "$@"
"""

    entrypoint_path = scripts_dir / "entrypoint.sh"
    with open(entrypoint_path, "w") as f:
        f.write(entrypoint)

    logger.info(f"Created entrypoint script: {entrypoint_path}")


def create_dockerfile(output_dir: str, config: ContainerBuildConfig) -> None:
    """Create Dockerfile.

    Args:
        output_dir: Output directory
        config: Build configuration
    """
    template = DOCKERFILE_TEMPLATES.get(config.framework, DOCKERFILE_TEMPLATES["transformers"])

    dockerfile_content = template.format(
        base_image=config.base_image,
        model_id=config.model_id,
        port=config.port,
        max_model_len=config.max_model_len,
        tensor_parallel=config.tensor_parallel,
    )

    dockerfile_path = Path(output_dir) / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    logger.info(f"Created Dockerfile: {dockerfile_path}")


def build_container(config: ContainerBuildConfig) -> Dict[str, Any]:
    """Build Docker container.

    Args:
        config: Build configuration

    Returns:
        Build result
    """
    output_dir = Path(config.output_tag.split(":")[0].replace("/", "_"))
    output_dir = Path("./container_build") / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating build context in: {output_dir}")

    create_dockerfile(str(output_dir), config)
    create_requirements_file(str(output_dir), config.framework, config.requirements)
    create_entrypoint_script(str(output_dir), config.framework)

    build_args = []
    for key, value in config.build_args.items():
        build_args.extend(["--build-arg", f"{key}={value}"])

    dockerfile_path = output_dir / "Dockerfile"

    cmd = [
        "docker", "build",
        "-t", config.output_tag,
        "-f", str(dockerfile_path),
    ]
    cmd.extend(build_args)
    cmd.append(str(output_dir))

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info(f"Container built successfully: {config.output_tag}")

        return {
            "status": "success",
            "tag": config.output_tag,
            "framework": config.framework,
            "model_id": config.model_id,
            "output_dir": str(output_dir),
            "build_output": result.stdout[-1000:] if result.stdout else "",
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Container build failed: {e.stderr}")

        return {
            "status": "error",
            "tag": config.output_tag,
            "error": e.stderr,
            "build_output": e.stdout[-1000:] if e.stdout else "",
        }


def get_container_info(tag: str) -> Optional[Dict[str, Any]]:
    """Get information about a built container.

    Args:
        tag: Container tag

    Returns:
        Container information or None
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", tag],
            capture_output=True,
            text=True,
            check=True,
        )

        info = json.loads(result.stdout)
        if info:
            return info[0]

    except Exception as e:
        logger.warning(f"Failed to get container info: {e}")

    return None


def main() -> int:
    """Main entry point for container building."""
    parser = argparse.ArgumentParser(
        description="Build Docker container for serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build vLLM container
    python build_container.py --model-id meta-llama/Llama-2-7b --framework vllm --output-tag llm-server:v1.0

    # Build with custom settings
    python build_container.py --model-id gpt2 --framework vllm --output-tag gpt2-server:latest --port 8080 --max-model-len 2048

    # Build with GPU support
    python build_container.py --model-id meta-llama/Llama-2-7b --framework tensorrt_llm --output-tag trt-server:v1 --gpu-enabled
        """
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID to serve"
    )
    parser.add_argument(
        "--framework",
        choices=["vllm", "tensorrt_llm", "transformers"],
        default="vllm",
        help="Serving framework (default: vllm)"
    )
    parser.add_argument(
        "--output-tag",
        required=True,
        help="Docker image tag (e.g., llm-server:v1.0)"
    )
    parser.add_argument(
        "--base-image",
        default="nvidia/cuda:12.1.0-devel-ubuntu22.04",
        help="Base Docker image (default: nvidia/cuda:12.1.0-devel-ubuntu22.04)"
    )
    parser.add_argument(
        "--python-version",
        default="3.10",
        help="Python version (default: 3.10)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Container port (default: 8000)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model length (default: 4096)"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU support"
    )
    parser.add_argument(
        "--build-arg",
        nargs=2,
        action="append",
        help="Build arguments (key value)"
    )
    parser.add_argument(
        "--requirement",
        action="append",
        dest="requirements",
        help="Additional requirements"
    )

    args = parser.parse_args()

    build_args = {}
    if args.build_arg:
        build_args = dict(args.build_arg)

    config = ContainerBuildConfig(
        model_id=args.model_id,
        output_tag=args.output_tag,
        framework=args.framework,
        base_image=args.base_image,
        python_version=args.python_version,
        port=args.port,
        gpu_enabled=not args.no_gpu,
        max_model_len=args.max_model_len,
        tensor_parallel=args.tensor_parallel,
        build_args=build_args,
        requirements=args.requirements or [],
    )

    try:
        result = build_container(config)

        if result["status"] == "success":
            print(json.dumps({
                "status": "success",
                "tag": config.output_tag,
                "framework": config.framework,
                "model_id": config.model_id,
            }, indent=2))

            info = get_container_info(config.output_tag)
            if info:
                logger.info(f"Container ID: {info.get('Id', 'unknown')[:12]}")

            return 0
        else:
            logger.error(f"Build failed: {result.get('error', 'Unknown error')}")
            return 1

    except KeyboardInterrupt:
        logger.info("Build cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())