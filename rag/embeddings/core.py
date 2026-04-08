"""Core embedding model implementations."""

import logging
from typing import List, Dict, Any, Optional, Hashable
from abc import ABC, abstractmethod
import hashlib
import pickle

from .config import EmbeddingConfig, EmbeddingType

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Main embedding model wrapper supporting multiple embedding types.

    Handles loading, initialization, and inference for embedding models.
    Supports dense embeddings with optional caching and quantization.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding model.

        Args:
            config: Embedding configuration

        Raises:
            ImportError: If required libraries not available
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load embedding model from huggingface."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.config.model_name}")
            self.model = SentenceTransformer(
                self.config.model_name,
                device="cuda" if self.config.use_cuda else "cpu",
                **self.config.model_kwargs,
            )
            logger.info(
                f"Model loaded successfully with dimension {self.config.embedding_dim}"
            )
        except ImportError as e:
            logger.warning(f"Failed to load embedding model: {e}")
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If model failed to load
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )

        logger.debug(f"Generated embeddings for {len(texts)} texts")
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []


class BatchEmbedder:
    """Batch embedding processor for efficient large-scale embedding.

    Handles batching, progress tracking, and error handling for embedding
    large document collections.
    """

    def __init__(self, embedding_model: EmbeddingModel, batch_size: int = 32):
        """Initialize batch embedder.

        Args:
            embedding_model: EmbeddingModel instance
            batch_size: Batch size for processing
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def embed_batch(self, texts: List[str]) -> Dict[str, List[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Dictionary mapping text (hashed) to embedding vector
        """
        if not texts:
            return {}

        embeddings = self.embedding_model.embed(texts)

        result = {}
        for text, embedding in zip(texts, embeddings):
            text_hash = self._hash_text(text)
            result[text_hash] = embedding

        logger.info(f"Embedded batch of {len(texts)} texts")
        return result

    def embed_multiple_batches(
        self,
        texts_batches: List[List[str]],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, List[float]]:
        """Embed multiple batches with progress tracking.

        Args:
            texts_batches: List of text batches
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping text hash to embedding vector
        """
        all_embeddings = {}

        for i, batch in enumerate(texts_batches):
            batch_embeddings = self.embed_batch(batch)
            all_embeddings.update(batch_embeddings)

            if progress_callback:
                progress_callback(i + 1, len(texts_batches))

        logger.info(f"Completed embedding {len(all_embeddings)} texts")
        return all_embeddings

    @staticmethod
    def _hash_text(text: str) -> str:
        """Hash text for use as dictionary key.

        Args:
            text: Text to hash

        Returns:
            SHA256 hash of text
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class EmbeddingCache:
    """Caching layer for embedding vectors.

    Implements LRU cache with configurable memory limit to avoid
    recomputing embeddings for repeated texts.
    """

    def __init__(self, max_size_mb: int = 1000):
        """Initialize embedding cache.

        Args:
            max_size_mb: Maximum cache size in MB
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, List[float]] = {}
        self.access_counts: Dict[str, int] = {}
        self.current_size_bytes = 0

    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache.

        Args:
            text: Text to look up

        Returns:
            Embedding vector if cached, None otherwise
        """
        text_hash = self._hash_text(text)

        if text_hash in self.cache:
            self.access_counts[text_hash] += 1
            logger.debug(f"Cache hit for text: {text_hash}")
            return self.cache[text_hash]

        return None

    def put(self, text: str, embedding: List[float]) -> bool:
        """Cache embedding for text.

        Args:
            text: Text to cache
            embedding: Embedding vector

        Returns:
            True if cached, False if cache full and eviction failed
        """
        text_hash = self._hash_text(text)

        if text_hash in self.cache:
            return True

        # Estimate size of embedding
        embedding_size = len(embedding) * 8  # float64
        text_size = len(text.encode())
        total_size = embedding_size + text_size

        # Make room if needed
        while self.current_size_bytes + total_size > self.max_size_bytes:
            if not self._evict_lru():
                logger.warning("Cache full, could not evict items")
                return False

        self.cache[text_hash] = embedding
        self.access_counts[text_hash] = 0
        self.current_size_bytes += total_size

        logger.debug(f"Cached embedding for text: {text_hash}")
        return True

    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
        self.access_counts.clear()
        self.current_size_bytes = 0
        logger.info("Embedding cache cleared")

    def _evict_lru(self) -> bool:
        """Evict least recently used item from cache.

        Returns:
            True if item evicted, False if cache empty
        """
        if not self.cache:
            return False

        lru_key = min(self.access_counts, key=self.access_counts.get)
        embedding = self.cache.pop(lru_key)
        del self.access_counts[lru_key]

        self.current_size_bytes -= len(embedding) * 8 + 100  # Estimate
        logger.debug(f"Evicted LRU item: {lru_key}")
        return True

    @staticmethod
    def _hash_text(text: str) -> str:
        """Hash text for use as cache key.

        Args:
            text: Text to hash

        Returns:
            SHA256 hash of text
        """
        import hashlib

        return hashlib.sha256(text.encode()).hexdigest()[:16]
