"""Configuration dataclasses for fine-tuning."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAMW_8BIT = "adamw_8bit"


class SchedulerType(str, Enum):
    """Supported learning rate scheduler types."""

    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    def __post_init__(self):
        """Validate optimizer configuration."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}"
            )


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: SchedulerType = SchedulerType.LINEAR
    num_warmup_steps: int = 500
    num_training_steps: int = 10000
    num_cycles: float = 0.5
    last_epoch: int = -1

    def __post_init__(self):
        """Validate scheduler configuration."""
        if self.num_warmup_steps < 0:
            raise ValueError(
                f"num_warmup_steps must be non-negative, got {self.num_warmup_steps}"
            )
        if self.num_training_steps <= 0:
            raise ValueError(
                f"num_training_steps must be positive, got {self.num_training_steps}"
            )


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    num_epochs: int = 3
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    eval_batch_size: int = 32
    max_seq_length: int = 512
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 100
    eval_steps: Optional[int] = None
    seed: int = 42
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    use_cpu: bool = False
    device_map: Optional[str] = None
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    def __post_init__(self):
        """Validate training configuration."""
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_seq_length <= 0:
            raise ValueError(
                f"max_seq_length must be positive, got {self.max_seq_length}"
            )
        if self.mixed_precision not in ["no", "fp16", "bf16"]:
            raise ValueError(
                f"mixed_precision must be 'no', 'fp16', or 'bf16', "
                f"got {self.mixed_precision}"
            )


@dataclass
class BaseFinetuningConfig:
    """Base configuration for fine-tuning."""

    model_name: str
    output_dir: str
    model_type: str = "automodel"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    use_auth_token: Optional[str] = None
    trust_remote_code: bool = True
    resume_from_checkpoint: Optional[str] = None
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    meta_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate base configuration."""
        if not self.model_name:
            raise ValueError("model_name is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        logger.info(f"Initialized BaseFinetuningConfig for model: {self.model_name}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "model_type": self.model_type,
            "tokenizer_name": self.tokenizer_name,
            "cache_dir": self.cache_dir,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "training": self.training.__dict__,
            "use_auth_token": self.use_auth_token is not None,
            "trust_remote_code": self.trust_remote_code,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "meta_data": self.meta_data,
        }
