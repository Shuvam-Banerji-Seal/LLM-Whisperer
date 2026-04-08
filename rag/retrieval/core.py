"""Core document retrieval implementations."""

import logging
from typing import List, Dict, Any, Optional, Tuple

from .config import RetrieverType

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Main document retriever using dense embeddings."""

    def __init__(self, index, embedder, config):
        """Initialize retriever.

        Args:
            index: Vector index for search
            embedder: Embedding model
            config: Retriever config
        """
        self.index = index
        self.embedder = embedder
        self.config = config

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Retrieve documents for query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        if top_k is None:
            top_k = self.config.top_k

        # Embed query
        query_embedding = self.embedder.embed_single(query)

        # Search index
        results = self.index.search(query_embedding, top_k)

        logger.info(f"Retrieved {len(results)} documents for query")
        return [doc_id for doc_id, _ in results]


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search."""

    def __init__(
        self,
        dense_retriever: DocumentRetriever,
        sparse_retriever: Optional["DocumentRetriever"] = None,
        alpha: float = 0.5,
    ):
        """Initialize hybrid retriever.

        Args:
            dense_retriever: Dense retriever instance
            sparse_retriever: Optional sparse retriever
            alpha: Weight for dense results (1-alpha for sparse)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve using hybrid approach.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of retrieved documents
        """
        dense_results = self.dense_retriever.retrieve(query, top_k * 2)

        if self.sparse_retriever:
            sparse_results = self.sparse_retriever.retrieve(query, top_k * 2)
            # Fuse results
            combined = set(dense_results) | set(sparse_results)
            return list(combined)[:top_k]

        return dense_results[:top_k]


class RetrieverConfig:
    """Configuration for retriever."""

    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.5):
        """Initialize config.

        Args:
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
