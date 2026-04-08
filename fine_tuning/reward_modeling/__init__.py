"""Reward model training module for LLM-Whisperer.

Fine-tuning utilities for training reward models used in RLHF and similar methods.
"""

from .core import RewardModelFinetuner
from .config import RewardModelingConfig

__version__ = "0.1.0"
__all__ = ["RewardModelFinetuner", "RewardModelingConfig"]
