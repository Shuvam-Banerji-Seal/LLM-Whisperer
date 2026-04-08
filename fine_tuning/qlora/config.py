"""QLoRA configuration."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from fine_tuning.lora.config import LoRAConfig

logger = logging.getLogger(__name__)


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA fine-tuning.

    QLoRA combines LoRA with 4-bit quantization for extreme memory efficiency.
    """

    # Quantization parameters
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"  # "float16", "bfloat16"
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True

    def __post_init__(self):
        """Validate QLoRA configuration."""
        super().__post_init__()

        if self.bnb_4bit_compute_dtype not in ["float16", "bfloat16"]:
            raise ValueError(
                f"bnb_4bit_compute_dtype must be 'float16' or 'bfloat16', "
                f"got {self.bnb_4bit_compute_dtype}"
            )
        if self.bnb_4bit_quant_type not in ["nf4", "fp4"]:
            raise ValueError(
                f"bnb_4bit_quant_type must be 'nf4' or 'fp4', "
                f"got {self.bnb_4bit_quant_type}"
            )

        logger.info(
            f"QLoRA configuration: 4bit_quant={self.bnb_4bit_quant_type}, "
            f"compute_dtype={self.bnb_4bit_compute_dtype}"
        )
