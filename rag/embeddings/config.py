"""Configuration for embedding models."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class EmbeddingType(str, Enum):
    """Supported embedding types."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_type: EmbeddingType = EmbeddingType.DENSE
    embedding_dim: int = 384
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    use_cuda: bool = True
    cache_embeddings: bool = True
    cache_size_mb: int = 1000
    quantize: bool = False
    quantization_type: Optional[str] = None
    model_kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate embedding configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )
        if self.max_seq_length <= 0:
            raise ValueError(
                f"max_seq_length must be positive, got {self.max_seq_length}"
            )
        if self.cache_size_mb < 0:
            raise ValueError(
                f"cache_size_mb must be non-negative, got {self.cache_size_mb}"
            )
