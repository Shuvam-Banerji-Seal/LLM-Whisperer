"""Behavior-specific fine-tuning module for LLM-Whisperer.

Fine-tuning utilities for training models on specific behaviors,
styles, and domain-specific tasks.
"""

from .core import BehaviorFinetuner
from .config import BehaviorTuningConfig

__version__ = "0.1.0"
__all__ = [
    "BehaviorFinetuner",
    "BehaviorTuningConfig",
]
