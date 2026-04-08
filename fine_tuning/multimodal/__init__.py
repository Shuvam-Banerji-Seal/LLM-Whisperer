"""Multimodal model fine-tuning module for LLM-Whisperer.

Fine-tuning utilities for vision-language models, video-language models,
and other multimodal architectures.
"""

from .core import MultimodalFinetuner
from .config import MultimodalTuningConfig

__version__ = "0.1.0"
__all__ = [
    "MultimodalFinetuner",
    "MultimodalTuningConfig",
]
