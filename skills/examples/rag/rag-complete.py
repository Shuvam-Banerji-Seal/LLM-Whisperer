"""
Complete RAG (Retrieval-Augmented Generation) Implementation
Comprehensive example covering dense/sparse retrieval, reranking, and evaluation.

Key Components:
- DenseRetriever: Using embeddings for semantic search
- SparseRetriever: Using BM25 for lexical search
- HyDEQueryExpansion: Query expansion with hypothetical documents
- CrossEncoderReranker: Re-ranking retrieved documents
- LLMAsJudge: Evaluation using LLM as judge
- RAGPipeline: End-to-end retrieval + generation pipeline

Usage:
    from rag_complete import RAGPipeline

    # Initialize pipeline
    rag = RAGPipeline(
        model_name="meta-llama/Llama-2-7b-hf",
        embedding_model="sentence-transformers/all-mpnet-base-v2"
    )

    # Add documents to retrieval index
    documents = ["Document 1", "Document 2", ...]
    rag.index_documents(documents)

    # Generate answer with retrieval
    answer = rag.generate(query="What is quantum computing?")

    # Evaluate with metrics
    metrics = rag.evaluate(test_qa_pairs)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import heapq
import math

# Placeholder for external dependencies (would be installed in real implementation)
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch


@dataclass
class Document:
    """Represents a document in the retrieval system."""

    content: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""

    documents: List[Document]
    scores: List[float]
    retrieval_method: str


class DenseRetriever(ABC):
    """
    Dense retrieval using embeddings.

    Approach:
    - Convert queries and documents to dense vectors (embeddings)
    - Use cosine similarity or L2 distance for retrieval
    - Typically 768-1024 dimensional vectors
    - Fast approximate nearest neighbor search with FAISS/ANNOY

    Performance:
    - Query latency: 1-10ms for 1M documents
    - Memory: ~1GB per 1M documents (768-dim embeddings)
    - Typical recall@10: 70-85%
    """

    def __init__(
        self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def _mock_embed(self, texts: List[str]) -> np.ndarray:
        """Mock embedding function for demonstration."""
        # In real implementation, use SentenceTransformer:
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(self.embedding_model)
        # return model.encode(texts, convert_to_numpy=True)

        # Mock: create random embeddings with cosine similarity
        batch_size = len(texts)
        embedding_dim = 384
        embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)
        # Normalize to unit vectors for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def index(self, documents: List[Document]) -> None:
        """Index documents for retrieval."""
        self.documents = documents
        texts = [doc.content for doc in documents]
        self.embeddings = self._mock_embed(texts)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve top-k documents using dense similarity."""
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index() first.")

        query_embedding = self._mock_embed([query])[0]

        # Cosine similarity: dot product for normalized vectors
        scores = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return RetrievalResult(
            documents=[self.documents[i] for i in top_indices],
            scores=[float(scores[i]) for i in top_indices],
            retrieval_method="dense",
        )


class SparseRetriever:
    """
    Sparse retrieval using BM25 (Best Matching 25).

    Approach:
    - Tokenize documents and queries
    - TF-IDF weighted term matching
    - Fast exact string matching
    - Typically 1000s-10000s dimensional vectors (bag of words)

    Advantages over dense:
    - Interpretable (based on exact term matches)
    - No embedding model needed
    - Better for rare domain-specific terms
    - Fast indexing

    Performance:
    - Query latency: <1ms for 1M documents
    - Memory: minimal
    - Typical recall@10: 40-60% (lower than dense)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        BM25 parameters:
        - k1: Controls term frequency saturation (default 1.5)
        - b: Controls length normalization (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.avg_doc_len = 0.0
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def index(self, documents: List[Document]) -> None:
        """Index documents using BM25."""
        self.documents = documents
        self.doc_freqs = []
        all_terms = set()

        # Count term frequencies per document
        for doc in documents:
            tokens = self._tokenize(doc.content)
            freq_map = {}
            for token in tokens:
                freq_map[token] = freq_map.get(token, 0) + 1
            self.doc_freqs.append(freq_map)
            all_terms.update(freq_map.keys())

        # Calculate IDF (inverse document frequency)
        num_docs = len(documents)
        for term in all_terms:
            doc_count = sum(1 for freq_map in self.doc_freqs if term in freq_map)
            self.idf[term] = math.log(
                (num_docs - doc_count + 0.5) / (doc_count + 0.5) + 1
            )

        # Calculate average document length
        doc_lens = [len(self._tokenize(doc.content)) for doc in documents]
        self.avg_doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        doc_len = len(self._tokenize(self.documents[doc_idx].content))
        score = 0.0

        for token in set(query_tokens):
            if token not in self.idf:
                continue

            freq = self.doc_freqs[doc_idx].get(token, 0)
            idf = self.idf[token]

            # BM25 formula
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * (doc_len / self.avg_doc_len)
            )
            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve top-k documents using BM25."""
        if not self.documents:
            raise ValueError("No documents indexed. Call index() first.")

        query_tokens = self._tokenize(query)
        scores = []

        for i in range(len(self.documents)):
            score = self._bm25_score(query_tokens, i)
            scores.append(score)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return RetrievalResult(
            documents=[self.documents[i] for i in top_indices],
            scores=[float(scores[i]) for i in top_indices],
            retrieval_method="sparse",
        )


class HyDEQueryExpansion:
    """
    Hypothetical Document Embeddings (HyDE) for query expansion.

    Idea:
    - Generate hypothetical documents that would answer the query
    - Use embeddings of hypothetical docs to retrieve real documents
    - Improves recall by expanding query semantic space

    Process:
    1. For query "What are the benefits of exercise?", generate:
       - "Exercise improves cardiovascular health..."
       - "Regular exercise enhances mental wellbeing..."
       - "Physical activity strengthens bones..."
    2. Embed all hypothetical documents
    3. Use them to retrieve similar real documents

    Benefits:
    - 5-10% improvement in recall@10
    - Better for open-domain questions
    - Language model handles semantic expansion
    """

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.num_hypotheticals = 3

    def _generate_hypotheticals(self, query: str) -> List[str]:
        """Generate hypothetical documents for a query."""
        # In real implementation, use LLM to generate
        # Prompt: f"Generate 3 sample documents that would answer this query: {query}"

        # Mock implementation for demonstration
        hypotheticals = [
            f"Document about: {query}. This document discusses key aspects.",
            f"Answer to query: {query}. Key findings show important results.",
            f"Research on: {query}. Evidence indicates significant impact.",
        ]
        return hypotheticals

    def expand_query(self, query: str, retriever: DenseRetriever) -> List[str]:
        """Expand query using hypothetical documents."""
        hypotheticals = self._generate_hypotheticals(query)

        # In real implementation, would get embeddings and use for retrieval
        # For now, return original query + hypotheticals
        return [query] + hypotheticals


class CrossEncoderReranker:
    """
    Cross-encoder for reranking retrieved documents.

    Approach:
    - Use cross-encoder model (e.g., mmarco-bert-base) to score (query, doc) pairs
    - Score all pairs jointly (not independent embeddings)
    - More accurate but slower than dual-encoders

    Key difference from dual-encoders:
    - Dual-encoder: embed query and docs separately, compare independently
    - Cross-encoder: jointly score query + doc concatenation

    Performance:
    - Accuracy: 2-5% NDCG@10 improvement over dense retrieval
    - Latency: 5-50ms per reranking (depends on top-k size)
    - Typical: rerank top-100 dense results to get top-10

    Models:
    - cross-encoder/mmarco-TinyBERT-L-6
    - cross-encoder/mmarco-MiniLMv2-L-6-H-384
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str = "cross-encoder/mmarco-MiniLMv2-L-6-H-384"):
        self.model_name = model_name

    def _score_pair(self, query: str, document: str) -> float:
        """Score a query-document pair."""
        # In real implementation:
        # model = CrossEncoder(self.model_name)
        # scores = model.predict([[query, document]])
        # return float(scores[0])

        # Mock: score based on overlap and length
        query_tokens = set(query.lower().split())
        doc_tokens = set(document.lower().split())
        overlap = len(query_tokens & doc_tokens)
        max_tokens = max(len(query_tokens), len(doc_tokens))
        return overlap / max_tokens if max_tokens > 0 else 0.0

    def rerank(
        self, query: str, results: RetrievalResult, top_k: int = 5
    ) -> RetrievalResult:
        """Rerank retrieved documents."""
        # Score all pairs
        scores = []
        for doc in results.documents:
            score = self._score_pair(query, doc.content)
            scores.append(score)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return RetrievalResult(
            documents=[results.documents[i] for i in top_indices],
            scores=[float(scores[i]) for i in top_indices],
            retrieval_method=f"{results.retrieval_method}_reranked",
        )


class LLMAsJudge:
    """
    Use LLM as a judge to evaluate answer quality.

    Approach:
    - Provide query, retrieved documents, and generated answer to LLM
    - Ask LLM to evaluate answer quality on multiple dimensions
    - More nuanced than automatic metrics

    Evaluation Dimensions:
    - Relevance: Does answer address the query?
    - Faithfulness: Is answer grounded in retrieved documents?
    - Completeness: Does answer cover all important aspects?
    - Accuracy: Is information factually correct?

    Metrics Used:
    - NDCG (Normalized Discounted Cumulative Gain): 0-1
    - MRR (Mean Reciprocal Rank): 0-1
    - Recall@K: 0-1
    - Precision@K: 0-1
    """

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name

    def evaluate_answer(
        self, query: str, retrieved_docs: List[Document], answer: str
    ) -> Dict[str, float]:
        """Evaluate answer quality using LLM as judge."""
        # In real implementation, would use LLM
        # Prompt structure:
        # Query: {query}
        # Retrieved Documents: {retrieved_docs}
        # Generated Answer: {answer}
        # Evaluate on: relevance, faithfulness, completeness, accuracy

        # Mock evaluation
        evaluation = {
            "relevance": 0.85,
            "faithfulness": 0.80,
            "completeness": 0.75,
            "accuracy": 0.88,
            "overall": 0.82,
        }
        return evaluation


class RAGEvaluator:
    """
    Complete evaluation framework for RAG systems.

    Metrics:
    - NDCG: Discounted gain at different cutoffs (1, 5, 10)
    - MRR: Reciprocal rank of first relevant document
    - Recall@K: Fraction of relevant docs in top-k
    - Precision@K: Fraction of top-k docs that are relevant
    - Map (Mean Average Precision): Average precision across queries
    """

    @staticmethod
    def calculate_ndcg(relevance_scores: List[float], k: int = 10) -> float:
        """Calculate NDCG@k."""
        relevance_scores = relevance_scores[:k]
        if not relevance_scores:
            return 0.0

        # DCG: discounted cumulative gain
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))

        # Ideal DCG (perfect ranking)
        ideal = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_mrr(is_relevant: List[bool]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, rel in enumerate(is_relevant, 1):
            if rel:
                return 1.0 / i
        return 0.0

    @staticmethod
    def calculate_recall(
        is_relevant: List[bool], k: int = 10, total_relevant: int = 1
    ) -> float:
        """Calculate Recall@k."""
        if total_relevant == 0:
            return 0.0
        relevant_in_k = sum(is_relevant[:k])
        return relevant_in_k / total_relevant

    @staticmethod
    def calculate_precision(is_relevant: List[bool], k: int = 10) -> float:
        """Calculate Precision@k."""
        if k == 0:
            return 0.0
        return sum(is_relevant[:k]) / k


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.

    Architecture:
    1. Query Expansion (HyDE)
    2. Dense Retrieval (semantic search)
    3. Sparse Retrieval (lexical search)
    4. Fusion (combine dense + sparse results)
    5. Reranking (cross-encoder)
    6. Answer Generation (LLM with retrieved context)

    Performance Characteristics:
    - Query latency: 50-200ms (depends on top-k and reranking)
    - NDCG@10: 0.65-0.75 (typical open-domain QA)
    - Hallucination rate: 5-15% (grounding in retrieval helps)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        use_reranking: bool = True,
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.use_reranking = use_reranking

        # Initialize retrievers
        self.dense_retriever = DenseRetriever(embedding_model)
        self.sparse_retriever = SparseRetriever()
        self.query_expander = HyDEQueryExpansion(model_name)
        self.reranker = CrossEncoderReranker()
        self.evaluator = RAGEvaluator()

        self.documents: List[Document] = []

    def index_documents(
        self, documents: List[str], metadatas: Optional[List[Dict]] = None
    ) -> None:
        """Index documents for retrieval."""
        doc_objects = []
        for i, doc_content in enumerate(documents):
            metadata = metadatas[i] if metadatas else {"index": i}
            doc_objects.append(Document(content=doc_content, metadata=metadata))

        self.documents = doc_objects
        self.dense_retriever.index(doc_objects)
        self.sparse_retriever.index(doc_objects)

    def retrieve(
        self, query: str, top_k: int = 5, method: str = "hybrid"
    ) -> RetrievalResult:
        """Retrieve documents using specified method."""
        if method == "dense":
            return self.dense_retriever.retrieve(query, top_k)
        elif method == "sparse":
            return self.sparse_retriever.retrieve(query, top_k)
        elif method == "hybrid":
            # Combine dense and sparse results
            dense_results = self.dense_retriever.retrieve(query, top_k)
            sparse_results = self.sparse_retriever.retrieve(query, top_k)

            # Normalize and combine scores
            combined_scores = {}
            for doc, score in zip(dense_results.documents, dense_results.scores):
                doc_id = id(doc)
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 0.5 * score

            for doc, score in zip(sparse_results.documents, sparse_results.scores):
                doc_id = id(doc)
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 0.5 * score

            # Rerank with cross-encoder if enabled
            if self.use_reranking and combined_scores:
                # Create temporary result for reranking
                all_docs = dense_results.documents + sparse_results.documents
                temp_result = RetrievalResult(
                    documents=all_docs,
                    scores=[combined_scores.get(id(d), 0) for d in all_docs],
                    retrieval_method="hybrid",
                )
                return self.reranker.rerank(query, temp_result, top_k)

            # Sort by combined score
            sorted_items = sorted(
                combined_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_doc_ids = [item[0] for item in sorted_items[:top_k]]

            all_unique_docs = []
            seen_ids = set()
            for doc in dense_results.documents + sparse_results.documents:
                doc_id = id(doc)
                if doc_id not in seen_ids and doc_id in top_doc_ids:
                    all_unique_docs.append(doc)
                    seen_ids.add(doc_id)

            return RetrievalResult(
                documents=all_unique_docs,
                scores=[combined_scores.get(id(d), 0) for d in all_unique_docs],
                retrieval_method="hybrid",
            )

    def generate(self, query: str, top_k: int = 5, method: str = "hybrid") -> str:
        """Generate answer using retrieved documents."""
        # Retrieve documents
        results = self.retrieve(query, top_k, method)

        # Format context from retrieved documents
        context = "\n".join(
            [f"[{i + 1}] {doc.content}" for i, doc in enumerate(results.documents)]
        )

        # In real implementation, would use LLM to generate answer
        # Prompt: f"""Based on the following context, answer the query.
        # Context: {context}
        # Query: {query}
        # Answer:"""

        # Mock response
        answer = f"Based on the retrieved documents, this is an answer to '{query}'."
        return answer

    def evaluate(self, qa_pairs: List[Tuple[str, str, List[int]]]) -> Dict[str, float]:
        """
        Evaluate RAG system on QA pairs.

        Args:
            qa_pairs: List of (query, answer, relevant_doc_indices)

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            "ndcg_5": [],
            "ndcg_10": [],
            "mrr": [],
            "recall_10": [],
            "precision_5": [],
        }

        for query, expected_answer, relevant_indices in qa_pairs:
            # Retrieve documents
            results = self.retrieve(query, top_k=10)

            # Create relevance scores
            is_relevant = [
                i in relevant_indices for i, doc in enumerate(results.documents)
            ]

            # Calculate metrics
            metrics["ndcg_5"].append(
                self.evaluator.calculate_ndcg([float(r) for r in is_relevant], k=5)
            )
            metrics["ndcg_10"].append(
                self.evaluator.calculate_ndcg([float(r) for r in is_relevant], k=10)
            )
            metrics["mrr"].append(self.evaluator.calculate_mrr(is_relevant))
            metrics["recall_10"].append(
                self.evaluator.calculate_recall(
                    is_relevant, k=10, total_relevant=len(relevant_indices)
                )
            )
            metrics["precision_5"].append(
                self.evaluator.calculate_precision(is_relevant, k=5)
            )

        # Average metrics
        return {
            "ndcg_5": np.mean(metrics["ndcg_5"]),
            "ndcg_10": np.mean(metrics["ndcg_10"]),
            "mrr": np.mean(metrics["mrr"]),
            "recall_10": np.mean(metrics["recall_10"]),
            "precision_5": np.mean(metrics["precision_5"]),
        }


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Quantum computing uses quantum bits (qubits) to process information exponentially faster than classical computers.",
        "Machine learning models learn patterns from data without being explicitly programmed.",
        "Neural networks are inspired by the structure and function of biological neural networks.",
        "Deep learning uses multiple layers of neural networks to extract features from raw data.",
        "Natural language processing enables computers to understand and generate human language.",
        "Retrieval-augmented generation combines information retrieval with language generation.",
    ]

    # Initialize pipeline
    rag = RAGPipeline()
    rag.index_documents(documents)

    # Test retrieval
    query = "How does quantum computing work?"
    results = rag.retrieve(query, top_k=3, method="hybrid")

    print(f"Query: {query}")
    print(f"\nTop-3 Retrieved Documents:")
    for i, (doc, score) in enumerate(zip(results.documents, results.scores)):
        print(f"{i + 1}. [{results.retrieval_method}] Score: {score:.4f}")
        print(f"   {doc.content[:100]}...\n")

    # Test generation
    answer = rag.generate(query)
    print(f"Generated Answer: {answer}")

    # Test evaluation
    qa_pairs = [
        ("How does quantum computing work?", "Quantum computers use qubits", [0]),
        ("What is machine learning?", "ML learns from data", [1]),
        ("Explain neural networks", "Networks with multiple layers", [2, 3]),
    ]

    eval_metrics = rag.evaluate(qa_pairs)
    print(f"\nEvaluation Metrics:")
    for metric, value in eval_metrics.items():
        print(f"  {metric}: {value:.4f}")
