"""Behavior tuning configuration."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from fine_tuning.base.config import BaseFinetuningConfig

logger = logging.getLogger(__name__)


@dataclass
class BehaviorTuningConfig(BaseFinetuningConfig):
    """Configuration for behavior-specific fine-tuning.

    Specialized for training models to exhibit specific behaviors,
    communication styles, and domain-specific knowledge.
    """

    # Behavior-specific parameters
    behavior_name: str = "custom"
    tone: str = "professional"  # "professional", "casual", "formal", "friendly"
    domain: str = "general"  # "medical", "legal", "technical", "creative", etc.
    include_examples: bool = True
    example_weight: float = 1.0
    behavior_consistency_weight: float = 0.5
    style_transfer_mode: bool = False
    target_behaviors: List[str] = None

    def __post_init__(self):
        """Validate behavior configuration."""
        super().__post_init__()

        if self.target_behaviors is None:
            self.target_behaviors = []

        logger.info(
            f"Behavior tuning for: {self.behavior_name} ({self.domain}, {self.tone})"
        )
