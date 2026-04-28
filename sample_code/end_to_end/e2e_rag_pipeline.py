"""
End-to-End RAG Pipeline

Complete RAG pipeline demonstrating:
- Data ingestion from multiple sources
- Chunking strategies comparison
- Embedding model selection
- Vector database setup (in-memory)
- Query processing and reranking
- Evaluation metrics

Author: Shuvam Banerji
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import uuid
import time
import logging
import math
import json
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources for ingestion."""
    TEXT = "text"
    FILE = "file"
    URL = "url"
    DATABASE = "database"
    API = "api"


@dataclass
class Document:
    """Document representation."""
    content: str
    source: DataSource
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                "source_type": self.source.value,
                "created_at": self.created_at
            }


@dataclass
class Chunk:
    """Document chunk."""
    content: str
    doc_id: str
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RetrievalResult:
    """Retrieval result with score."""
    chunk: Chunk
    score: float
    rank: int = 0


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC_SENTENCE = "semantic_sentence"
    SEMANTIC_PARAGRAPH = "semantic_paragraph"
    RECURSIVE = "recursive"
    SEMANTIC_CHUNKING = "semantic_chunking"


class EmbeddingModelType(Enum):
    """Embedding model types."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    Colbert = "colbert"


class IngestionPipeline:
    """Data ingestion from multiple sources."""

    def __init__(self):
        self.documents: List[Document] = []

    def ingest_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """
        Ingest text documents.

        Args:
            texts: List of text contents
            metadatas: Optional metadata for each text

        Returns:
            List of Document objects
        """
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            doc = Document(
                content=text,
                source=DataSource.TEXT,
                metadata={**meta, "ingestion_index": i}
            )
            docs.append(doc)

        self.documents.extend(docs)
        logger.info(f"Ingested {len(docs)} text documents")
        return docs

    def ingest_files(self, file_paths: List[str]) -> List[Document]:
        """
        Ingest files (mock implementation).

        Args:
            file_paths: List of file paths

        Returns:
            List of Document objects
        """
        docs = []
        for path in file_paths:
            doc = Document(
                content=f"Content from {path}",
                source=DataSource.FILE,
                metadata={"file_path": path}
            )
            docs.append(doc)

        self.documents.extend(docs)
        logger.info(f"Ingested {len(docs)} files")
        return docs

    def ingest_urls(self, urls: List[str]) -> List[Document]:
        """
        Ingest from URLs (mock implementation).

        Args:
            urls: List of URLs

        Returns:
            List of Document objects
        """
        docs = []
        for url in urls:
            doc = Document(
                content=f"Content fetched from {url}",
                source=DataSource.URL,
                metadata={"url": url}
            )
            docs.append(doc)

        self.documents.extend(docs)
        logger.info(f"Ingested {len(docs)} URLs")
        return docs

    def get_documents(self) -> List[Document]:
        """Get all ingested documents."""
        return self.documents.copy()

    def clear(self) -> None:
        """Clear all documents."""
        self.documents.clear()


class ChunkingPipeline:
    """Pipeline for chunking documents with multiple strategies."""

    def __init__(self):
        self.strategies = {
            ChunkingStrategy.FIXED_SIZE: self._fixed_size_chunk,
            ChunkingStrategy.SEMANTIC_SENTENCE: self._sentence_chunk,
            ChunkingStrategy.SEMANTIC_PARAGRAPH: self._paragraph_chunk,
            ChunkingStrategy.RECURSIVE: self._recursive_chunk,
            ChunkingStrategy.SEMANTIC_CHUNKING: self._semantic_chunk,
        }

    def chunk_documents(
        self,
        documents: List[Document],
        strategy: ChunkingStrategy,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Chunk]:
        """
        Chunk documents using specified strategy.

        Args:
            documents: Documents to chunk
            strategy: Chunking strategy
            chunk_size: Target chunk size
            overlap: Overlap between chunks

        Returns:
            List of chunks
        """
        strategy_func = self.strategies.get(strategy, self._fixed_size_chunk)
        all_chunks = []

        for doc in documents:
            chunks = strategy_func(doc, chunk_size, overlap)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks using {strategy.value}")
        return all_chunks

    def _fixed_size_chunk(self, doc: Document, size: int, overlap: int) -> List[Chunk]:
        """Fixed-size chunking."""
        content = doc.content
        chunks = []

        for i in range(0, len(content), size - overlap):
            chunk_text = content[i:i + size]
            if chunk_text.strip():
                chunks.append(Chunk(
                    content=chunk_text.strip(),
                    doc_id=doc.doc_id,
                    chunk_index=len(chunks),
                    start_char=i,
                    end_char=min(i + size, len(content)),
                    metadata={"strategy": "fixed_size", **doc.metadata}
                ))
            if i + size >= len(content):
                break

        return chunks

    def _sentence_chunk(self, doc: Document, size: int, overlap: int) -> List[Chunk]:
        """Sentence-based chunking."""
        sentences = re.split(r'(?<=[.!?])\s+', doc.content)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    content=chunk_text,
                    doc_id=doc.doc_id,
                    chunk_index=chunk_index,
                    metadata={"strategy": "sentence", **doc.metadata}
                ))
                chunk_index += 1

                overlap_count = max(1, overlap // 50)
                current_chunk = current_chunk[-overlap_count:]
                current_size = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(Chunk(
                content=" ".join(current_chunk),
                doc_id=doc.doc_id,
                chunk_index=chunk_index,
                metadata={"strategy": "sentence", **doc.metadata}
            ))

        return chunks

    def _paragraph_chunk(self, doc: Document, size: int, overlap: int) -> List[Chunk]:
        """Paragraph-based chunking."""
        paragraphs = re.split(r'\n\s*\n', doc.content)
        chunks = []

        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append(Chunk(
                    content=para.strip(),
                    doc_id=doc.doc_id,
                    chunk_index=i,
                    metadata={"strategy": "paragraph", **doc.metadata}
                ))

        return chunks

    def _recursive_chunk(self, doc: Document, size: int, overlap: int) -> List[Chunk]:
        """Recursive character-based chunking."""
        content = doc.content
        chunks = []

        def split_text(text: str, start: int) -> None:
            if len(text) <= size:
                if text.strip():
                    chunks.append(Chunk(
                        content=text.strip(),
                        doc_id=doc.doc_id,
                        chunk_index=len(chunks),
                        start_char=start,
                        end_char=start + len(text),
                        metadata={"strategy": "recursive", **doc.metadata}
                    ))
                return

            split_point = size
            for sep in ['\n\n', '\n', '. ', ' ']:
                last_sep = text.rfind(sep, 0, split_point)
                if last_sep > size // 2:
                    split_point = last_sep + len(sep)
                    break

            chunks.append(Chunk(
                content=text[:split_point].strip(),
                doc_id=doc.doc_id,
                chunk_index=len(chunks),
                start_char=start,
                end_char=start + split_point,
                metadata={"strategy": "recursive", **doc.metadata}
            ))

            remaining = text[split_point - overlap:split_point].strip()
            if remaining:
                split_text(text[split_point - overlap:], start + split_point - overlap)
            else:
                split_text(text[split_point:], start + split_point)

        if content.strip():
            split_text(content, 0)

        return chunks

    def _semantic_chunk(self, doc: Document, size: int, overlap: int) -> List[Chunk]:
        """Semantic chunking based on topic coherence."""
        sentences = re.split(r'(?<=[.!?])\s+', doc.content)
        chunks = []
        current_chunk = []
        current_embedding = [0.0] * 384

        for sentence in sentences:
            sentence_embedding = self._mock_embed([sentence])[0]

            if current_chunk:
                avg_embedding = [
                    sum(x) / len(current_chunk) for x in zip(*[self._mock_embed([s])[0] for s in current_chunk])
                ]
                similarity = sum(a * b for a, b in zip(avg_embedding, sentence_embedding))

                if similarity < 0.7 and len(" ".join(current_chunk)) > size // 2:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(Chunk(
                        content=chunk_text,
                        doc_id=doc.doc_id,
                        chunk_index=len(chunks),
                        metadata={"strategy": "semantic", "similarity_threshold": 0.7, **doc.metadata}
                    ))
                    current_chunk = current_chunk[-2:]
                    current_embedding = [sum(x) / len(current_chunk) for x in zip(*[self._mock_embed([s])[0] for s in current_chunk])]

            current_chunk.append(sentence)

        if current_chunk:
            chunks.append(Chunk(
                content=" ".join(current_chunk),
                doc_id=doc.doc_id,
                chunk_index=len(chunks),
                metadata={"strategy": "semantic", **doc.metadata}
            ))

        return chunks

    def _mock_embed(self, texts: List[str]) -> List[List[float]]:
        """Mock embedding for semantic chunking."""
        import hashlib
        embeddings = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            import random
            random.seed(seed)
            embedding = [random.gauss(0, 1) for _ in range(384)]
            norm = math.sqrt(sum(x * x for x in embedding))
            embedding = [x / norm for x in embedding]
            embeddings.append(embedding)
        return embeddings


class EmbeddingPipeline:
    """Embedding generation with multiple model options."""

    def __init__(self, model_type: EmbeddingModelType = EmbeddingModelType.DENSE):
        """
        Initialize embedding pipeline.

        Args:
            model_type: Type of embedding model
        """
        self.model_type = model_type
        self.embedding_dim = 384
        self.vocab: Dict[str, int] = {}

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: Chunks to embed

        Returns:
            List of embedding vectors
        """
        if self.model_type == EmbeddingModelType.DENSE:
            return self._dense_embed(chunks)
        elif self.model_type == EmbeddingModelType.SPARSE:
            return self._sparse_embed(chunks)
        else:
            return self._dense_embed(chunks)

    def _dense_embed(self, chunks: List[Chunk]) -> List[List[float]]:
        """Dense embedding generation."""
        import hashlib
        embeddings = []

        for chunk in chunks:
            seed = int(hashlib.md5(chunk.content.encode()).hexdigest()[:8], 16)
            import random
            random.seed(seed)
            embedding = [random.gauss(0, 1) for _ in range(self.embedding_dim)]
            norm = math.sqrt(sum(x * x for x in embedding))
            embedding = [x / norm for x in embedding]
            embeddings.append(embedding)

        return embeddings

    def _sparse_embed(self, chunks: List[Chunk]) -> List[List[float]]:
        """Sparse TF-IDF embedding."""
        all_terms = set()
        for chunk in chunks:
            terms = re.findall(r'\b\w+\b', chunk.content.lower())
            all_terms.update(terms)

        self.vocab = {term: i for i, term in enumerate(all_terms)}

        embeddings = []
        for chunk in chunks:
            terms = re.findall(r'\b\w+\b', chunk.content.lower())
            term_freq = defaultdict(int)
            for term in terms:
                term_freq[term] += 1

            embedding = [0.0] * len(self.vocab)
            for term, freq in term_freq.items():
                if term in self.vocab:
                    embedding[self.vocab[term]] = freq

            embeddings.append(embedding)

        return embeddings


class VectorDatabase:
    """In-memory vector database for document retrieval."""

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize vector database.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.chunks: List[Chunk] = []
        self.embeddings: List[List[float]] = []
        self.chunk_to_doc: Dict[str, Document] = {}

    def index(self, chunks: List[Chunk], embeddings: List[List[float]], documents: Dict[str, Document]) -> None:
        """
        Index chunks with embeddings.

        Args:
            chunks: Chunks to index
            embeddings: Corresponding embeddings
            documents: Document lookup
        """
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        self.chunk_to_doc.update({c.chunk_id: documents.get(c.doc_id) for c in chunks})

        logger.info(f"Indexed {len(chunks)} chunks into vector database")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding
            top_k: Number of results
            min_score: Minimum similarity score

        Returns:
            List of retrieval results
        """
        if not self.embeddings:
            return []

        similarities = []
        for i, embedding in enumerate(self.embeddings):
            score = self._cosine_similarity(query_embedding, embedding)
            if score >= min_score:
                similarities.append((i, score))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(similarities[:top_k]):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=score,
                rank=rank + 1
            ))

        return results

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        min_len = min(len(a), len(b))
        dot_product = sum(a[i] * b[i] for i in range(min_len))
        return dot_product

    def count(self) -> int:
        """Get number of indexed chunks."""
        return len(self.chunks)


class Reranker:
    """Cross-encoder reranker for improving retrieval results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker.

        Args:
            model_name: Model name for reranking
        """
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results.

        Args:
            query: Query text
            results: Initial retrieval results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        reranked = []

        for result in results:
            score = self._score_query_document(query, result.chunk.content)
            reranked.append(RetrievalResult(
                chunk=result.chunk,
                score=score,
                rank=result.rank
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)

        for i, result in enumerate(reranked[:top_k]):
            result.rank = i + 1

        return reranked[:top_k]

    def _score_query_document(self, query: str, document: str) -> float:
        """Score query-document relevance."""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        doc_terms = set(re.findall(r'\b\w+\b', document.lower()))

        overlap = len(query_terms & doc_terms)
        union = len(query_terms | doc_terms)

        return overlap / union if union > 0 else 0.0


class QueryProcessor:
    """Process and expand queries."""

    def __init__(self):
        pass

    def process(self, query: str) -> str:
        """
        Process query.

        Args:
            query: Raw query

        Returns:
            Processed query
        """
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        return query

    def expand(self, query: str) -> List[str]:
        """
        Expand query with synonyms.

        Args:
            query: Query to expand

        Returns:
            List of expanded queries
        """
        expansions = [query]

        synonym_map = {
            "python": ["python programming", "python language"],
            "ml": ["machine learning", "ml"],
            "ai": ["artificial intelligence", "ai"],
            "nn": ["neural network", "neural networks"],
        }

        query_lower = query.lower()
        for key, synonyms in synonym_map.items():
            if key in query_lower:
                for syn in synonyms:
                    if syn != query:
                        expansions.append(syn.replace(key, synonyms[0]))

        return expansions


class AnswerGenerator:
    """Generate answers from retrieved context."""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b"):
        """
        Initialize answer generator.

        Args:
            model_name: LLM model name
        """
        self.model_name = model_name

    def generate(
        self,
        query: str,
        context: List[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate answer from context.

        Args:
            query: User query
            context: Retrieved context chunks
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated answer with metadata
        """
        if not context:
            return {
                "answer": "I don't have enough information to answer your question.",
                "sources": [],
                "tokens_used": 0,
                "generation_time_ms": 0
            }

        start_time = time.time()

        context_text = "\n\n".join([
            f"[Source {r.rank}] {r.chunk.content}"
            for r in context
        ])

        answer = self._mock_generate(query, context_text)

        generation_time = (time.time() - start_time) * 1000

        return {
            "answer": answer,
            "sources": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "content": r.chunk.content[:200] + "...",
                    "score": r.score,
                    "rank": r.rank
                }
                for r in context
            ],
            "tokens_used": len(answer.split()) * 1.3,
            "generation_time_ms": generation_time
        }

    def _mock_generate(self, query: str, context: str) -> str:
        """Mock answer generation."""
        return (
            f"Based on the retrieved documents, here is my answer to your query '{query}':\n\n"
            f"Context summary: Found {len(context.split())} words of relevant information.\n\n"
            f"The answer is generated by {self.model_name} using the provided context. "
            f"In production, this would be an actual LLM response grounded in the retrieved information."
        )


class RAGEvaluator:
    """Evaluate RAG system performance."""

    def __init__(self):
        pass

    def evaluate(
        self,
        queries: List[str],
        expected_answers: List[str],
        retrieved_contexts: List[List[RetrievalResult]]
    ) -> Dict[str, float]:
        """
        Evaluate RAG performance.

        Args:
            queries: Test queries
            expected_answers: Expected answers
            retrieved_contexts: Retrieved contexts for each query

        Returns:
            Evaluation metrics
        """
        metrics = {
            "precision_at_1": [],
            "precision_at_5": [],
            "recall_at_k": [],
            "mrr": [],
            "ndcg_at_k": [],
        }

        for i, (query, expected, contexts) in enumerate(zip(queries, expected_answers, retrieved_contexts)):
            metrics["mrr"].append(self._calculate_mrr(contexts, k=10))
            metrics["ndcg_at_k"].append(self._calculate_ndcg(contexts, k=10))

            if len(contexts) >= 1:
                metrics["precision_at_1"].append(1.0 if contexts[0].score > 0.5 else 0.0)

            if len(contexts) >= 5:
                relevant_count = sum(1 for c in contexts[:5] if c.score > 0.5)
                metrics["precision_at_5"].append(relevant_count / 5)

        return {
            "precision_at_1": sum(metrics["precision_at_1"]) / max(1, len(metrics["precision_at_1"])),
            "precision_at_5": sum(metrics["precision_at_5"]) / max(1, len(metrics["precision_at_5"])),
            "mrr": sum(metrics["mrr"]) / max(1, len(metrics["mrr"])),
            "ndcg_at_10": sum(metrics["ndcg_at_k"]) / max(1, len(metrics["ndcg_at_k"])),
        }

    def _calculate_mrr(self, contexts: List[RetrievalResult], k: int) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, ctx in enumerate(contexts[:k]):
            if ctx.score > 0.5:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_ndcg(self, contexts: List[RetrievalResult], k: int) -> float:
        """Calculate NDCG@k."""
        relevances = [1.0 if c.score > 0.5 else 0.0 for c in contexts[:k]]

        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

        ideal = sorted(relevances, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))

        return dcg / idcg if idcg > 0 else 0.0


class EndToEndRAGPipeline:
    """Complete end-to-end RAG pipeline."""

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        embedding_model: EmbeddingModelType = EmbeddingModelType.DENSE,
        use_reranking: bool = True
    ):
        """
        Initialize E2E RAG pipeline.

        Args:
            chunking_strategy: Document chunking strategy
            embedding_model: Embedding model type
            use_reranking: Whether to use reranking
        """
        self.ingestion = IngestionPipeline()
        self.chunker = ChunkingPipeline()
        self.chunking_strategy = chunking_strategy
        self.embedder = EmbeddingPipeline(embedding_model)
        self.vector_db = VectorDatabase()
        self.reranker = Reranker() if use_reranking else None
        self.query_processor = QueryProcessor()
        self.generator = AnswerGenerator()
        self.evaluator = RAGEvaluator()

        self.documents: Dict[str, Document] = {}

    def ingest(
        self,
        texts: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> int:
        """
        Ingest data from sources.

        Args:
            texts: Text documents
            files: File paths
            urls: URLs
            metadatas: Metadata for texts

        Returns:
            Number of chunks created
        """
        all_docs = []

        if texts:
            docs = self.ingestion.ingest_texts(texts, metadatas)
            all_docs.extend(docs)

        if files:
            docs = self.ingestion.ingest_files(files)
            all_docs.extend(docs)

        if urls:
            docs = self.ingestion.ingest_urls(urls)
            all_docs.extend(docs)

        for doc in all_docs:
            self.documents[doc.doc_id] = doc

        chunks = self.chunker.chunk_documents(
            all_docs,
            self.chunking_strategy,
            chunk_size=500,
            overlap=50
        )

        embeddings = self.embedder.embed_chunks(chunks)

        self.vector_db.index(chunks, embeddings, self.documents)

        return len(chunks)

    def query(self, query: str, top_k: int = 10, rerank: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            query: User query
            top_k: Number of results
            rerank: Whether to rerank results

        Returns:
            Query results
        """
        processed_query = self.query_processor.process(query)

        query_embedding = self.embedder._dense_embed([Chunk(
            content=processed_query,
            doc_id="",
            chunk_index=0
        )])[0]

        results = self.vector_db.search(query_embedding, top_k=top_k * 2 if rerank else top_k)

        if rerank and self.reranker:
            results = self.reranker.rerank(processed_query, results, top_k=top_k)
        else:
            results = results[:top_k]

        answer = self.generator.generate(processed_query, results)

        return {
            "query": query,
            "processed_query": processed_query,
            "answer": answer["answer"],
            "sources": answer["sources"],
            "total_chunks_indexed": self.vector_db.count()
        }


def demo():
    """Demonstrate end-to-end RAG pipeline."""
    print("=" * 70)
    print("End-to-End RAG Pipeline Demo")
    print("=" * 70)

    documents = [
        "Python is a high-level programming language. Python was created by Guido van Rossum. "
        "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",

        "Machine learning is a subset of artificial intelligence. Machine learning algorithms learn patterns from data. "
        "Common machine learning types include supervised learning, unsupervised learning, and reinforcement learning.",

        "Neural networks are computing systems inspired by biological neural networks. Deep learning uses multiple neural network layers. "
        "Convolutional neural networks are used for image recognition. Recurrent neural networks process sequential data.",

        "Large Language Models (LLMs) are neural networks trained on vast text corpora. Examples include GPT, BERT, and Llama models. "
        "LLMs can understand and generate human language, code, and various other content types.",

        "Retrieval-Augmented Generation (RAG) combines information retrieval with generative AI. "
        "RAG systems can cite sources and provide more accurate, up-to-date answers than standalone LLMs."
    ]

    metadatas = [
        {"source": "python_guide.txt", "category": "programming"},
        {"source": "ml_intro.txt", "category": "ai"},
        {"source": "neural_nets.txt", "category": "deep_learning"},
        {"source": "llms.txt", "category": "nlp"},
        {"source": "rag_guide.txt", "category": "ai"}
    ]

    pipeline = EndToEndRAGPipeline(
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        embedding_model=EmbeddingModelType.DENSE,
        use_reranking=True
    )

    num_chunks = pipeline.ingest(texts=documents, metadatas=metadatas)
    print(f"\nIngested {len(documents)} documents into {num_chunks} chunks")

    test_queries = [
        "What is Python programming?",
        "How do neural networks work?",
        "What is RAG?"
    ]

    print("\n" + "-" * 70)
    print("Query Results:")
    print("-" * 70)

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = pipeline.query(query, top_k=3)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources found: {len(result['sources'])}")
        print("-" * 70)


if __name__ == "__main__":
    demo()