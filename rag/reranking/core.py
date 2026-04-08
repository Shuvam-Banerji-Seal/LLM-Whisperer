"""Core reranking implementations."""

import logging
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from .config import RerankerConfig, RerankerType

logger = logging.getLogger(__name__)


class RankingStrategy(ABC):
    """Abstract ranking strategy."""

    @abstractmethod
    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Rerank candidates for query.

        Args:
            query: Query text
            candidates: List of candidate texts

        Returns:
            List of (candidate, score) tuples
        """
        pass


class Reranker:
    """Main reranker using cross-encoders."""

    def __init__(self, config: RerankerConfig):
        """Initialize reranker.

        Args:
            config: Reranker configuration
        """
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load reranking model."""
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker: {self.config.model_name}")
            self.model = CrossEncoder(
                self.config.model_name,
                max_length=512,
            )
        except ImportError:
            logger.warning("sentence-transformers not installed for reranking")

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Rerank candidates using cross-encoder.

        Args:
            query: Query text
            candidates: List of candidate texts

        Returns:
            List of (candidate, score) tuples sorted by score
        """
        if self.model is None:
            logger.warning("Model not loaded, returning original order")
            return [(c, 1.0 / (i + 1)) for i, c in enumerate(candidates)]

        pairs = [[query, candidate] for candidate in candidates]
        scores = self.model.predict(pairs)

        ranked = list(zip(candidates, scores.tolist()))
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked[: self.config.top_k]


class RerankerFactory:
    """Factory for creating rerankers."""

    @staticmethod
    def create(config: RerankerConfig) -> Reranker:
        """Create reranker instance.

        Args:
            config: Reranker configuration

        Returns:
            Reranker instance
        """
        if config.reranker_type == RerankerType.CROSS_ENCODER:
            return Reranker(config)
        else:
            raise ValueError(f"Unsupported reranker type: {config.reranker_type}")
