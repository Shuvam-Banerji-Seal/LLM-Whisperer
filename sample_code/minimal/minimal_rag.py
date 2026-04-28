"""
Minimal RAG Implementation

A minimal but complete RAG (Retrieval-Augmented Generation) pipeline demonstrating:
- Document loading and chunking
- Embedding generation
- Vector search
- Answer generation

This implementation uses in-memory storage for simplicity but follows production patterns.

Author: Shuvam Banerji
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
import time
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the RAG system."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {"created_at": time.time()}


@dataclass
class Chunk:
    """Represents a chunk of a document."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    chunk_index: int = 0


class ChunkingStrategy(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document, chunk_size: int, overlap: int) -> List[Chunk]:
        """Split document into chunks."""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """Split documents into fixed-size chunks with overlap."""

    def chunk(self, document: Document, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
        """
        Split document into fixed-size chunks.

        Args:
            document: Document to chunk
            chunk_size: Target size of each chunk (characters)
            overlap: Number of overlapping characters between chunks

        Returns:
            List of Chunk objects
        """
        content = document.content
        chunks = []

        if len(content) <= chunk_size:
            chunks.append(Chunk(
                content=content,
                metadata=document.metadata.copy(),
                doc_id=document.doc_id,
                chunk_index=0
            ))
            return chunks

        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]

            if chunk_index > 0:
                prev_start = start - overlap
                if prev_start >= 0:
                    chunk_text = content[prev_start:start] + chunk_text

            chunks.append(Chunk(
                content=chunk_text.strip(),
                metadata={
                    **document.metadata,
                    "start_char": start,
                    "end_char": min(end, len(content))
                },
                doc_id=document.doc_id,
                chunk_index=chunk_index
            ))

            start = end - overlap if overlap > 0 else end
            chunk_index += 1

            if start >= len(content):
                break

        return chunks


class SemanticChunker(ChunkingStrategy):
    """Split documents based on semantic boundaries (sentences/paragraphs)."""

    def __init__(self, sentences_per_chunk: int = 5):
        """
        Initialize semantic chunker.

        Args:
            sentences_per_chunk: Number of sentences per chunk
        """
        self.sentences_per_chunk = sentences_per_chunk

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        import re
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, document: Document, chunk_size: int = 500, overlap: int = 1) -> List[Chunk]:
        """
        Split document by sentences.

        Args:
            document: Document to chunk
            chunk_size: Target chunk size (used for grouping)
            overlap: Number of sentences to overlap

        Returns:
            List of Chunk objects
        """
        sentences = self._split_sentences(document.content)
        chunks = []
        chunk_index = 0

        for i in range(0, len(sentences), self.sentences_per_chunk - overlap):
            if i >= len(sentences):
                break

            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)

            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "sentence_range": (i, min(i + self.sentences_per_chunk, len(sentences)))
                },
                doc_id=document.doc_id,
                chunk_index=chunk_index
            ))

            chunk_index += 1

            if i + self.sentences_per_chunk >= len(sentences):
                break

        return chunks


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        pass


class SimpleEmbeddingModel(EmbeddingModel):
    """Simple TF-IDF based embedding for demonstration."""

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize embedding model.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) > 2]

    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate term frequency."""
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        total = len(tokens) if tokens else 1
        return {k: v / total for k, v in tf.items()}

    def _build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        self.vocab = {}
        for text in texts:
            tokens = self._tokenize(text)
            for token in set(tokens):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def _calculate_idf(self, texts: List[str]) -> None:
        """Calculate inverse document frequency."""
        num_docs = len(texts)
        doc_freq = {}

        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.idf = {
            token: math.log(num_docs / (df + 1)) + 1
            for token, df in doc_freq.items()
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate TF-IDF embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self.vocab:
            self._build_vocab(texts)
            self._calculate_idf(texts)

        embeddings = []

        for text in texts:
            tokens = self._tokenize(text)
            tf = self._calculate_tf(tokens)

            embedding = [0.0] * min(self.embedding_dim, len(self.vocab))

            for token, freq in tf.items():
                if token in self.vocab and self.vocab[token] < self.embedding_dim:
                    idx = self.vocab[token]
                    idf_val = self.idf.get(token, 1.0)
                    embedding[idx] = freq * idf_val

            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

            embeddings.append(embedding)

        return embeddings

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity."""
        if len(embedding1) != len(embedding2):
            min_len = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_len]
            embedding2 = embedding2[:min_len]

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        return dot_product


class VectorStore:
    """In-memory vector store for document retrieval."""

    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize vector store.

        Args:
            embedding_model: Model to use for embeddings
        """
        self.embedding_model = embedding_model
        self.chunks: List[Chunk] = []
        self.embeddings: List[List[float]] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of chunks to add
        """
        if not chunks:
            return

        self.chunks.extend(chunks)
        texts = [chunk.content for chunk in chunks]
        new_embeddings = self.embedding_model.embed(texts)
        self.embeddings.extend(new_embeddings)

        logger.info(f"Added {len(chunks)} chunks to vector store")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for most similar chunks to query.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (Chunk, similarity_score) tuples
        """
        if not self.chunks:
            logger.warning("Vector store is empty")
            return []

        query_embedding = self.embedding_model.embed([query])[0]

        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            sim = self.embedding_model.similarity(query_embedding, chunk_embedding)
            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in similarities[:top_k]:
            if idx < len(self.chunks):
                results.append((self.chunks[idx], score))

        return results

    def count(self) -> int:
        """Return number of chunks in store."""
        return len(self.chunks)


class Generator:
    """Simple generator that formats retrieved context into answers."""

    def __init__(self, model_name: str = "mock-llm"):
        """
        Initialize generator.

        Args:
            model_name: Name of LLM to use
        """
        self.model_name = model_name

    def generate(self, query: str, context_chunks: List[Tuple[Chunk, float]]) -> str:
        """
        Generate answer based on query and retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks with scores

        Returns:
            Generated answer
        """
        if not context_chunks:
            return "I don't have enough information to answer that question."

        context_text = "\n\n".join([
            f"[Source {i+1}] {chunk.content}"
            for i, (chunk, score) in enumerate(context_chunks)
        ])

        answer = (
            f"Based on the retrieved documents, here's my answer to your query '{query}':\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            f"This information should help answer your question. "
            f"I found {len(context_chunks)} relevant pieces of information."
        )

        return answer


class MinimalRAG:
    """Minimal RAG pipeline combining all components."""

    def __init__(
        self,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize minimal RAG pipeline.

        Args:
            chunking_strategy: Strategy for chunking documents
            embedding_model: Model for generating embeddings
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunker = chunking_strategy or FixedSizeChunker()
        self.embedding_model = embedding_model or SimpleEmbeddingModel()
        self.vector_store = VectorStore(self.embedding_model)
        self.generator = Generator()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info("Minimal RAG pipeline initialized")

    def load_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        """
        Load and index documents.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts

        Returns:
            Number of chunks created
        """
        docs = [
            Document(content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadatas or [{}] * len(documents))
        ]

        all_chunks = []
        for doc in docs:
            chunks = self.chunker.chunk(doc, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(chunks)

        self.vector_store.add_chunks(all_chunks)

        logger.info(f"Indexed {len(docs)} documents into {len(all_chunks)} chunks")
        return len(all_chunks)

    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            query: User query
            top_k: Number of results to retrieve

        Returns:
            Dict containing answer and source chunks
        """
        start_time = time.time()

        results = self.vector_store.search(query, top_k)

        if not results:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "query_time_ms": (time.time() - start_time) * 1000
            }

        answer = self.generator.generate(query, results)

        return {
            "answer": answer,
            "sources": [
                {
                    "content": chunk.content,
                    "doc_id": chunk.doc_id,
                    "score": score,
                    "chunk_index": chunk.chunk_index
                }
                for chunk, score in results
            ],
            "query_time_ms": (time.time() - start_time) * 1000
        }


def demo():
    """Demonstrate minimal RAG pipeline."""
    print("=" * 70)
    print("Minimal RAG Implementation Demo")
    print("=" * 70)

    documents = [
        "Python is a high-level programming language known for its simplicity and readability. "
        "It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",

        "Machine learning is a subset of artificial intelligence that enables systems to learn from data. "
        "Key algorithms include supervised learning, unsupervised learning, and reinforcement learning.",

        "Large Language Models (LLMs) are neural networks trained on vast amounts of text data. "
        "They can understand and generate human language, code, and other forms of content.",

        "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. "
        "This approach allows models to cite sources and provide more accurate answers.",

        "Vector databases store high-dimensional embeddings that enable semantic search. "
        "Popular options include Pinecone, Weaviate, and FAISS for efficient similarity search."
    ]

    metadatas = [
        {"source": "python_docs.txt", "category": "programming"},
        {"source": "ml_intro.txt", "category": "ai"},
        {"source": "llm_overview.txt", "category": "nlp"},
        {"source": "rag_guide.txt", "category": "ai"},
        {"source": "vector_db.txt", "category": "databases"}
    ]

    rag = MinimalRAG(chunk_size=200, chunk_overlap=30)
    num_chunks = rag.load_documents(documents, metadatas)
    print(f"\nIndexed {len(documents)} documents into {num_chunks} chunks\n")

    queries = [
        "What is Python programming?",
        "How does machine learning work?",
        "What is RAG?"
    ]

    for query in queries:
        print(f"Query: {query}")
        result = rag.query(query, top_k=2)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources found: {len(result['sources'])}")
        print(f"Query time: {result['query_time_ms']:.2f}ms")
        print("-" * 70)


if __name__ == "__main__":
    demo()