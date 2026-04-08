"""Configuration for document chunking."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkingMethod(str, Enum):
    """Supported chunking methods."""

    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"
    DOCUMENT_AWARE = "document_aware"
    CODE_AWARE = "code_aware"
    SEMANTIC = "semantic"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking strategies."""

    method: ChunkingMethod = ChunkingMethod.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunks_per_document: Optional[int] = None
    preserve_separators: bool = True
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " ", ""])

    # Code-aware chunking
    code_aware: bool = False
    language: Optional[str] = None

    # Semantic chunking
    semantic_chunking: bool = False
    semantic_threshold: float = 0.5

    # Metadata handling
    preserve_metadata: bool = True
    metadata_fields: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate chunking configuration."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_size < 0:
            raise ValueError(
                f"min_chunk_size must be non-negative, got {self.min_chunk_size}"
            )
        if self.semantic_threshold < 0 or self.semantic_threshold > 1:
            raise ValueError(
                f"semantic_threshold must be between 0 and 1, "
                f"got {self.semantic_threshold}"
            )
