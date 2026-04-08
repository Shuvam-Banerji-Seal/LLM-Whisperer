"""Agentic tuning configuration."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fine_tuning.base.config import BaseFinetuningConfig

logger = logging.getLogger(__name__)


@dataclass
class AgenticTuningConfig(BaseFinetuningConfig):
    """Configuration for agentic system fine-tuning.

    Specialized configuration for fine-tuning models for agentic tasks including
    tool use, reasoning, planning, and sequential decision-making.
    """

    # Agentic-specific parameters
    num_tools: int = 5
    max_reasoning_steps: int = 10
    reasoning_loss_weight: float = 0.5
    action_loss_weight: float = 0.3
    reward_loss_weight: float = 0.2
    include_tool_descriptions: bool = True
    tool_call_format: str = "json"  # "json", "xml", "text"
    trajectory_sampling: str = "full"  # "full", "recent", "important"
    trajectory_max_length: int = 50  # Max steps in a trajectory
    use_trajectory_weighting: bool = True
    include_failure_examples: bool = True
    failure_example_ratio: float = 0.2

    def __post_init__(self):
        """Validate agentic configuration."""
        super().__post_init__()

        if self.num_tools <= 0:
            raise ValueError(f"num_tools must be positive, got {self.num_tools}")
        if self.max_reasoning_steps <= 0:
            raise ValueError(
                f"max_reasoning_steps must be positive, got {self.max_reasoning_steps}"
            )
        if not (0 < self.reasoning_loss_weight <= 1):
            raise ValueError(
                f"reasoning_loss_weight must be in (0, 1], got {self.reasoning_loss_weight}"
            )
        if not (0 <= self.failure_example_ratio <= 1):
            raise ValueError(
                f"failure_example_ratio must be in [0, 1], "
                f"got {self.failure_example_ratio}"
            )

        logger.info(
            f"Agentic tuning: {self.num_tools} tools, "
            f"max {self.max_reasoning_steps} reasoning steps"
        )
