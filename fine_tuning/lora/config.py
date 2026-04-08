"""LoRA configuration."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from fine_tuning.base.config import BaseFinetuningConfig

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig(BaseFinetuningConfig):
    """Configuration for LoRA fine-tuning.

    LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning by adding
    trainable low-rank decomposition matrices to selected layers.
    """

    # LoRA specific parameters
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling factor
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )  # Modules to apply LoRA
    lora_dropout: float = 0.1  # Dropout for LoRA layers
    bias: str = "none"  # Bias configuration: "none", "all", or "lora_only"
    modules_to_save: Optional[List[str]] = None  # Modules to save full parameters
    task_type: Optional[str] = None  # Task type for prompt tuning variants
    inference_mode: bool = False  # Whether in inference mode

    def __post_init__(self):
        """Validate LoRA configuration."""
        super().__post_init__()

        if self.r <= 0:
            raise ValueError(f"r must be positive, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(
                f"bias must be 'none', 'all', or 'lora_only', got {self.bias}"
            )

        logger.info(f"LoRA configuration: r={self.r}, alpha={self.lora_alpha}")
