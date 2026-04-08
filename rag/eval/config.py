"""Configuration for RAG evaluation."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class EvalMetric(str, Enum):
    """RAG evaluation metrics."""

    RETRIEVAL_PRECISION = "retrieval_precision"
    RETRIEVAL_RECALL = "retrieval_recall"
    RETRIEVAL_NDCG = "retrieval_ndcg"
    RETRIEVAL_MRR = "retrieval_mrr"
    RETRIEVAL_HITS = "retrieval_hits"
    GENERATION_BLEU = "generation_bleu"
    GENERATION_ROUGE = "generation_rouge"
    GENERATION_BERTSCORE = "generation_bertscore"
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"


@dataclass
class EvalConfig:
    """Configuration for RAG evaluation."""

    metrics: List[EvalMetric] = field(
        default_factory=lambda: [
            EvalMetric.RETRIEVAL_NDCG,
            EvalMetric.GENERATION_ROUGE,
            EvalMetric.FAITHFULNESS,
        ]
    )
    top_k: int = 5
    relevance_threshold: float = 0.5
    faithfulness_threshold: float = 0.7
    batch_size: int = 32
    use_gpu: bool = True
    num_workers: int = 4
    cache_results: bool = True

    def __post_init__(self):
        """Validate evaluation configuration."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if not (0 <= self.relevance_threshold <= 1):
            raise ValueError(
                f"relevance_threshold must be between 0 and 1, got {self.relevance_threshold}"
            )
        if not (0 <= self.faithfulness_threshold <= 1):
            raise ValueError(
                f"faithfulness_threshold must be between 0 and 1, got {self.faithfulness_threshold}"
            )
