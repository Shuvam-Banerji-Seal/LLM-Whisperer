"""Configuration for indexing."""

import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """Supported index types."""

    FLAT = "flat"
    IVFFLAT = "ivfflat"
    HNSW = "hnsw"
    ANNOY = "annoy"
    FAISS = "faiss"


@dataclass
class IndexConfig:
    """Configuration for vector indexing."""

    index_type: IndexType = IndexType.HNSW
    embedding_dim: int = 384
    max_elements: int = 1000000
    ef_construction: int = 200  # For HNSW
    ef: int = 50  # For HNSW search
    m: int = 16  # For HNSW
    num_clusters: Optional[int] = None  # For IVF
    nprobe: Optional[int] = None  # For IVF
    metric: str = "cosine"
    index_path: Optional[str] = None
    save_on_disk: bool = False

    def __post_init__(self):
        """Validate indexing configuration."""
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )
        if self.max_elements <= 0:
            raise ValueError(f"max_elements must be positive, got {self.max_elements}")
        if self.ef_construction <= 0:
            raise ValueError(
                f"ef_construction must be positive, got {self.ef_construction}"
            )
