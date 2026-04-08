"""Configuration for document retrieval."""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class RetrieverType(str, Enum):
    """Supported retriever types."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    BM25 = "bm25"
    ENSEMBLE = "ensemble"


@dataclass
class RetrieverType:
    """Configuration for document retrieval."""

    retriever_type: RetrieverType = RetrieverType.DENSE
    top_k: int = 5
    similarity_threshold: float = 0.5
    use_reranking: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None
    search_timeout_seconds: int = 30
    batch_size: int = 32
    cache_results: bool = True
    fusion_method: str = "rrf"  # reciprocal rank fusion

    def __post_init__(self):
        """Validate retriever configuration."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if not (0 <= self.similarity_threshold <= 1):
            raise ValueError(
                f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}"
            )
        if self.search_timeout_seconds <= 0:
            raise ValueError(
                f"search_timeout_seconds must be positive, got {self.search_timeout_seconds}"
            )
