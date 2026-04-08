"""Configuration for reranking."""

import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RerankerType(str, Enum):
    """Supported reranker types."""

    CROSS_ENCODER = "cross_encoder"
    LLM_JUDGE = "llm_judge"
    DIVERSITY = "diversity"
    MMR = "mmr"


@dataclass
class RerankerConfig:
    """Configuration for reranking."""

    reranker_type: RerankerType = RerankerType.CROSS_ENCODER
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    batch_size: int = 32
    top_k: int = 10
    score_threshold: float = 0.5
    use_gpu: bool = True
    normalize_scores: bool = True
    diversity_penalty: float = 0.0

    def __post_init__(self):
        """Validate reranker configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if not (0 <= self.score_threshold <= 1):
            raise ValueError(
                f"score_threshold must be between 0 and 1, got {self.score_threshold}"
            )
        if not (0 <= self.diversity_penalty <= 1):
            raise ValueError(
                f"diversity_penalty must be between 0 and 1, got {self.diversity_penalty}"
            )
