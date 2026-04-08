"""Base fine-tuning utilities and configurations for LLM-Whisperer.

This module provides core abstractions, configurations, and utilities for fine-tuning
large language models, including base classes, configuration management, and common utilities.
"""

from .core import (
    BaseFinetuner,
    FinetuningCallback,
    FinetuningMetrics,
    FinetuningState,
)
from .config import (
    BaseFinetuningConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .utils import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    get_device,
    set_seed,
)

__version__ = "0.1.0"
__all__ = [
    "BaseFinetuner",
    "FinetuningCallback",
    "FinetuningMetrics",
    "FinetuningState",
    "BaseFinetuningConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "setup_logging",
    "save_checkpoint",
    "load_checkpoint",
    "get_device",
    "set_seed",
]
