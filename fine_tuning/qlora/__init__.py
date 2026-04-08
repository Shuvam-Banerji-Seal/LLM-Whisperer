"""QLoRA (Quantized LoRA) fine-tuning module for LLM-Whisperer.

Combines LoRA with quantization for memory-efficient fine-tuning of large models.
"""

from .core import QLoRAFinetuner
from .config import QLoRAConfig

__version__ = "0.1.0"
__all__ = [
    "QLoRAFinetuner",
    "QLoRAConfig",
]
