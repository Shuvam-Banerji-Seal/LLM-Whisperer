"""Core vector indexing implementations."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .config import IndexConfig, IndexType

logger = logging.getLogger(__name__)


class VectorIndex:
    """In-memory vector index for RAG."""

    def __init__(self, config: IndexConfig):
        """Initialize vector index.

        Args:
            config: Index configuration
        """
        self.config = config
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.document_ids: List[str] = []

    def add(
        self,
        vectors: List[List[float]],
        doc_ids: List[str],
        metadata: Optional[List[Dict]] = None,
    ):
        """Add vectors to index.

        Args:
            vectors: List of embedding vectors
            doc_ids: List of document IDs
            metadata: Optional metadata dicts
        """
        for i, vec in enumerate(vectors):
            self.vectors.append(np.array(vec, dtype=np.float32))
            self.document_ids.append(doc_ids[i])
            if metadata:
                self.metadata.append(metadata[i])
            else:
                self.metadata.append({})

        logger.info(f"Added {len(vectors)} vectors to index")

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Search index for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (doc_id, similarity) tuples
        """
        query_vec = np.array(query_vector, dtype=np.float32)

        similarities = []
        for i, vec in enumerate(self.vectors):
            # Cosine similarity
            sim = np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8
            )
            similarities.append((self.document_ids[i], float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class IndexBuilder:
    """Builds and manages vector indices."""

    def __init__(self, config: IndexConfig):
        """Initialize index builder.

        Args:
            config: Index configuration
        """
        self.config = config
        self.index = VectorIndex(config)

    def build(
        self,
        vectors: List[List[float]],
        doc_ids: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> VectorIndex:
        """Build index from vectors.

        Args:
            vectors: List of embedding vectors
            doc_ids: List of document IDs
            metadata: Optional metadata

        Returns:
            Built VectorIndex
        """
        self.index.add(vectors, doc_ids, metadata)
        logger.info(f"Built index with {len(vectors)} vectors")
        return self.index


class IndexOptimizer:
    """Optimizes vector indices."""

    @staticmethod
    def optimize_search_params(
        index: VectorIndex, query_distribution: List[float]
    ) -> Dict[str, Any]:
        """Optimize search parameters based on query distribution.

        Args:
            index: VectorIndex to optimize
            query_distribution: Distribution of query types

        Returns:
            Optimized parameters
        """
        return {
            "ef": max(50, min(200, int(len(index.vectors) * 0.1))),
            "top_k": 5,
        }
