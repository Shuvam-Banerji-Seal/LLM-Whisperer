"""RAG tuning configuration."""

import logging
from dataclasses import dataclass
from typing import Optional

from fine_tuning.base.config import BaseFinetuningConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGTuningConfig(BaseFinetuningConfig):
    """Configuration for RAG system fine-tuning."""

    retriever_model: Optional[str] = None
    retrieval_loss_weight: float = 0.3
    generation_loss_weight: float = 0.7
    top_k_retrieved: int = 5
    use_dense_retrieval: bool = True
    joint_training: bool = True

    def __post_init__(self):
        """Validate RAG configuration."""
        super().__post_init__()
        logger.info(
            f"RAG tuning: top_k={self.top_k_retrieved}, joint={self.joint_training}"
        )
