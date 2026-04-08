"""Utility functions for fine-tuning."""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Setup logging configuration.

    Args:
        log_file: Optional path to log file
        level: Logging level
    """
    logging_config = {
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "level": level,
    }

    if log_file:
        logging.basicConfig(**logging_config, filename=log_file)
    else:
        logging.basicConfig(**logging_config)

    logger.info("Logging setup complete")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def get_device(use_cpu: bool = False) -> torch.device:
    """Get the appropriate device (CPU or CUDA).

    Args:
        use_cpu: Force CPU usage

    Returns:
        torch.device object
    """
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    config: Dict[str, Any],
    state: Dict[str, Any],
    checkpoint_path: str,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        config: Configuration dictionary
        state: Training state dictionary
        checkpoint_path: Path to save checkpoint
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config,
        "state": state,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint onto

    Returns:
        Checkpoint dictionary
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Checkpoint loaded from {checkpoint_path}")

    return checkpoint


def create_output_directory(output_dir: str) -> Path:
    """Create output directory.

    Args:
        output_dir: Path to output directory

    Returns:
        Path object
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created at {output_path}")
    return output_path


def log_config(config: Dict[str, Any], prefix: str = "Config") -> None:
    """Log configuration dictionary.

    Args:
        config: Configuration dictionary
        prefix: Prefix for log message
    """
    logger.info(f"{prefix}:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_info(model: torch.nn.Module, model_name: str = "Model") -> None:
    """Print model information.

    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    size_mb = get_model_size_mb(model)

    logger.info(f"\n{model_name} Information:")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable Parameters: {total_params - trainable_params:,}")
    logger.info(f"  Model Size: {size_mb:.2f} MB")
    logger.info(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")


def cleanup_checkpoints(checkpoint_dir: str, keep_last_n: int = 3) -> None:
    """Cleanup old checkpoints keeping only the last N.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return

    checkpoints = sorted(checkpoint_path.glob("checkpoint_*"))

    if len(checkpoints) > keep_last_n:
        checkpoints_to_remove = checkpoints[:-keep_last_n]
        for checkpoint in checkpoints_to_remove:
            if checkpoint.is_file():
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
            elif checkpoint.is_dir():
                import shutil

                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint directory: {checkpoint}")
