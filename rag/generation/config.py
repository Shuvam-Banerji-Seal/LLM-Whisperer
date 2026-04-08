"""Configuration for generation and augmentation."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class GenerationMode(str, Enum):
    """Generation modes."""

    GROUNDED = "grounded"
    OPEN_ENDED = "open_ended"
    COMPARATIVE = "comparative"
    ABSTRACTIVE = "abstractive"


@dataclass
class GenerationConfig:
    """Configuration for generation."""

    mode: GenerationMode = GenerationMode.GROUNDED
    llm_model: str = "mistral-7b"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    include_citations: bool = True
    include_confidence: bool = True
    context_window_size: int = 4
    prompt_template: Optional[str] = None
    use_retrieval_context: bool = True
    enable_fact_checking: bool = False

    def __post_init__(self):
        """Validate generation configuration."""
        if not (0 <= self.temperature <= 2):
            raise ValueError(
                f"temperature must be between 0 and 2, got {self.temperature}"
            )
        if not (0 <= self.top_p <= 1):
            raise ValueError(f"top_p must be between 0 and 1, got {self.top_p}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.context_window_size <= 0:
            raise ValueError(
                f"context_window_size must be positive, got {self.context_window_size}"
            )
