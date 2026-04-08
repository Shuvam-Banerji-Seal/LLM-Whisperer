"""Reward modeling configuration."""

import logging
from dataclasses import dataclass
from typing import Optional

from fine_tuning.base.config import BaseFinetuningConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardModelingConfig(BaseFinetuningConfig):
    """Configuration for reward model training."""

    use_pairwise_loss: bool = True
    margin: float = 1.0
    temperature: float = 0.1
    normalize_scores: bool = True
    score_range: tuple = (0.0, 1.0)

    def __post_init__(self):
        """Validate reward modeling configuration."""
        super().__post_init__()
        logger.info(
            f"Reward modeling: pairwise={self.use_pairwise_loss}, margin={self.margin}"
        )
