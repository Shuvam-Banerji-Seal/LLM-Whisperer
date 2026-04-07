"""RAG system core components."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document for RAG."""

    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.5


class DocumentChunker:
    """Chunks documents into smaller pieces."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """Initialize chunker.

        Args:
            chunk_size: Size of chunks
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Chunk text into smaller pieces.

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks


class EmbeddingModel:
    """Generates embeddings for texts."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding model.

        Args:
            model_name: Name of embedding model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            logger.warning("Model not loaded, returning dummy embeddings")
            return [[0.0] * 384 for _ in texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class VectorDatabase:
    """In-memory vector database for RAG."""

    def __init__(self):
        """Initialize vector database."""
        self.documents: Dict[str, Document] = {}
        self.embeddings: List[np.ndarray] = []

    def add_document(self, document: Document):
        """Add document to database.

        Args:
            document: Document to add
        """
        self.documents[document.id] = document

        if document.embedding:
            self.embeddings.append(np.array(document.embedding))

        logger.info(f"Added document: {document.id}")

    def add_documents(self, documents: List[Document]):
        """Add multiple documents.

        Args:
            documents: List of documents
        """
        for doc in documents:
            self.add_document(doc)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results

        Returns:
            List of (document_id, similarity) tuples
        """
        query_vec = np.array(query_embedding)

        similarities = []
        for doc_id, doc in self.documents.items():
            if doc.embedding is None:
                continue

            doc_vec = np.array(doc.embedding)
            # Cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8
            )
            similarities.append((doc_id, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class RAGSystem:
    """Complete RAG system."""

    def __init__(self, config: RAGConfig):
        """Initialize RAG system.

        Args:
            config: RAG configuration
        """
        self.config = config
        self.chunker = DocumentChunker(config.chunk_size, config.chunk_overlap)
        self.embedding_model = EmbeddingModel(config.embedding_model)
        self.vector_db = VectorDatabase()

    def add_documents(
        self, documents: List[str], metadata: Optional[List[Dict]] = None
    ):
        """Add documents to RAG system.

        Args:
            documents: List of document texts
            metadata: Optional metadata for documents
        """
        for i, doc_text in enumerate(documents):
            chunks = self.chunker.chunk(doc_text)
            chunk_embeddings = self.embedding_model.embed(chunks)

            meta = metadata[i] if metadata else {}

            for j, chunk in enumerate(chunks):
                doc = Document(
                    id=f"doc_{i}_chunk_{j}",
                    content=chunk,
                    metadata=meta,
                    embedding=chunk_embeddings[j],
                )
                self.vector_db.add_document(doc)

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for query.

        Args:
            query: Query text

        Returns:
            List of relevant documents
        """
        query_embedding = self.embedding_model.embed([query])[0]

        results = self.vector_db.search(query_embedding, self.config.top_k)

        documents = []
        for doc_id, similarity in results:
            if similarity >= self.config.similarity_threshold:
                documents.append(self.vector_db.documents[doc_id])

        logger.info(f"Retrieved {len(documents)} documents for query")
        return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = RAGConfig()
    rag = RAGSystem(config)

    # Add sample documents
    rag.add_documents(
        [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
        ]
    )

    # Retrieve documents
    results = rag.retrieve("What is machine learning?")
    print(f"Found {len(results)} relevant documents")
