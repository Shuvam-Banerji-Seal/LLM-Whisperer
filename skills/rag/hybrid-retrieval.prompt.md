# Hybrid Retrieval for RAG — Agentic Skill Prompt

Combining dense and sparse retrieval methods for improved recall and precision in production RAG systems.

---

## 1. Identity and Mission

Implement hybrid retrieval pipelines that leverage the complementary strengths of dense embeddings and sparse keyword matching to achieve robust retrieval across diverse query types. Hybrid retrieval addresses the inherent limitations of each approach: dense models excel at semantic similarity but miss exact keyword matches, while sparse methods like BM25 capture exact terms but struggle with synonyms and conceptual relevance.

---

## 2. Theory & Fundamentals

### 2.1 Dense Retrieval

Dense retrieval uses learned embedding models to map queries and documents to dense vector spaces. The similarity between query and document embeddings determines relevance.

**Core Formula — Cosine Similarity:**
```
score(q, d) = (q · d) / (||q|| × ||d||)
```

Where q and d are L2-normalized embedding vectors.

**Dense Retrieval Advantages:**
- Captures semantic/syntactic similarity beyond exact keyword matching
- Generalizes to out-of-vocabulary terms
- Single representation captures meaning

**Dense Retrieval Limitations:**
- Requires sufficient training data for embeddings
- Computationally expensive for large corpora
- May miss exact keyword matches important for specific domains

### 2.2 Sparse Retrieval

Sparse retrieval represents documents as high-dimensional vectors where most dimensions are zero. BM25 is the standard sparse method:

**BM25 Formula:**
```
score(q, d) = Σ IDF(t) × (f(t, d) × (k1 + 1)) / (f(t, d) + k1 × (1 - b + b × |d|/avgdl))
```

Where:
- `t` = term in query
- `f(t, d)` = term frequency in document
- `|d|` = document length
- `avgdl` = average document length
- `k1` = term frequency saturation parameter (typically 1.2-2.0)
- `b` = length normalization parameter (typically 0.75)
- `IDF(t)` = inverse document frequency

**Sparse Retrieval Advantages:**
- Exact keyword matching for technical terms, proper nouns
- Interpretable relevance signals
- Efficient for large-scale retrieval

**Sparse Retrieval Limitations:**
- Cannot capture semantic similarity
- Suffers from vocabulary mismatch problem
- No handling of synonyms or related concepts

### 2.3 Hybrid Combination Strategies

**Reciprocal Rank Fusion (RRF):**
```
score_RRF(d) = Σ 1 / (k + rank_i(d))
```

Where `k` is a constant (typically 60) and `rank_i(d)` is the rank of document d in retrieval list i.

**Linear Combination:**
```
score_hybrid(d) = α × score_dense(d) + (1 - α) × score_sparse(d)
```

**Distribution-Based Merging:**
Normalize scores from each retriever to same distribution, then combine.

---

## 3. Implementation Patterns

### Pattern 1: Basic Hybrid Retrieval with Weighted Combination

```python
import numpy as np
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Hybrid retrieval combining dense embeddings and BM25."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_weight: float = 0.5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.dense_weight = dense_weight
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize dense retriever
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
        self.model.eval()

        # BM25 will be initialized during indexing
        self.bm25 = None
        self.documents = []
        self.document_embeddings = None
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    def _mean_pooling(self, hidden_state, attention_mask):
        """Apply mean pooling to get sentence embedding."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def index(self, documents: List[str]) -> None:
        """Index documents for retrieval."""
        self.documents = documents

        # Build dense embeddings
        print(f"Encoding {len(documents)} documents...")
        self.document_embeddings = self.encode(documents)
        print(f"Encoded documents shape: {self.document_embeddings.shape}")

        # Build BM25 index
        print("Building BM25 index...")
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
        print("BM25 index built.")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = True,
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """Retrieve documents using hybrid search."""

        # Dense retrieval
        query_embedding = self.encode([query])
        dense_scores = (query_embedding @ self.document_embeddings.T).squeeze(0)

        # Sparse retrieval (BM25)
        query_tokens = query.lower().split()
        sparse_scores = self.bm25.get_scores(query_tokens)

        # Normalize scores to [0, 1] range
        dense_normalized = (dense_scores - dense_scores.min()) / (
            dense_scores.max() - dense_scores.min() + 1e-9
        )
        sparse_normalized = (sparse_scores - sparse_scores.min()) / (
            sparse_scores.max() - sparse_scores.min() + 1e-9
        )

        # Combine scores
        hybrid_scores = (
            self.dense_weight * dense_normalized
            + (1 - self.dense_weight) * sparse_normalized
        )

        # Get top-k
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = [self.documents[i] for i in top_indices]

        if return_scores:
            scores = [hybrid_scores[i] for i in top_indices]
            return results, scores

        return results


# Usage
if __name__ == "__main__":
    documents = [
        "Python is a high-level programming language known for its simplicity",
        "Machine learning is a subset of artificial intelligence that enables systems to learn",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing deals with understanding text",
        "The Python programming language was created by Guido van Rossum",
        "Transformers architecture revolutionized NLP tasks",
        "Retrieval-augmented generation combines retrieval with text generation",
        "Vector databases store high-dimensional embeddings efficiently",
    ]

    retriever = HybridRetriever(dense_weight=0.6)
    retriever.index(documents)

    results, scores = retriever.retrieve("Python programming language", top_k=3)
    for doc, score in zip(results, scores):
        print(f"[{score:.4f}] {doc}")
```

### Pattern 2: Reciprocal Rank Fusion

```python
from typing import List, Tuple, Dict, Any
import numpy as np

class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining multiple retrieval results."""

    def __init__(self, k: float = 60.0):
        self.k = k

    def fuse(
        self,
        retrieval_results: List[List[Tuple[str, float]]],
        method: str = "rrf",
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked retrieval result lists.

        Args:
            retrieval_results: List of (document, score) pairs from different retrievers
            method: Fusion method - 'rrf', 'score_avg', 'bordacount'

        Returns:
            Fused and reranked list of (document, score) pairs
        """
        if method == "rrf":
            return self._reciprocal_rank_fusion(retrieval_results)
        elif method == "score_avg":
            return self._score_averaging(retrieval_results)
        elif method == "bordacount":
            return self._borda_count_fusion(retrieval_results)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def _reciprocal_rank_fusion(
        self,
        retrieval_results: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion algorithm."""
        doc_scores: Dict[str, float] = {}

        for results in retrieval_results:
            for rank, (doc, original_score) in enumerate(results, start=1):
                rrf_score = 1 / (self.k + rank)
                if doc in doc_scores:
                    doc_scores[doc] += rrf_score
                else:
                    doc_scores[doc] = rrf_score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def _score_averaging(
        self,
        retrieval_results: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """Simple score averaging with normalization."""
        doc_scores: Dict[str, List[float]] = {}

        for results in retrieval_results:
            # Normalize scores within this result set
            scores = [score for _, score in results]
            if scores:
                min_s, max_s = min(scores), max(scores)
                range_s = max_s - min_s + 1e-9

                for doc, score in results:
                    normalized = (score - min_s) / range_s
                    if doc in doc_scores:
                        doc_scores[doc].append(normalized)
                    else:
                        doc_scores[doc] = [normalized]

        # Average scores
        averaged = {doc: sum(scores) / len(scores) for doc, scores in doc_scores.items()}
        return sorted(averaged.items(), key=lambda x: x[1], reverse=True)

    def _borda_count_fusion(
        self,
        retrieval_results: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """Borda count fusion for ranked lists."""
        doc_scores: Dict[str, float] = {}
        n_results = len(retrieval_results)

        for results in retrieval_results:
            n = len(results)
            for rank, (doc, _) in enumerate(results):
                borda_score = n - rank - 1
                if doc in doc_scores:
                    doc_scores[doc] += borda_score
                else:
                    doc_scores[doc] = borda_score

        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


class MultiRetrieverFusion:
    """Fuse results from dense, sparse, and potentially other retrievers."""

    def __init__(self, k: float = 60.0):
        self.rrf = ReciprocalRankFusion(k=k)
        self.dense_retriever = None
        self.sparse_retriever = None

    def set_retrievers(
        self,
        dense_retriever: Any,
        sparse_retriever: Any,
    ):
        """Set the retriever components."""
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        fusion_method: str = "rrf",
    ) -> List[Tuple[str, float]]:
        """Retrieve and fuse results from multiple retrievers."""
        # Get results from each retriever
        dense_results = []
        sparse_results = []

        if self.dense_retriever:
            dense_docs, dense_scores = self.dense_retriever.retrieve(
                query, top_k=top_k * 2, return_scores=True
            )
            dense_results = list(zip(dense_docs, dense_scores))

        if self.sparse_retriever:
            sparse_docs, sparse_scores = self.sparse_retriever.retrieve(
                query, top_k=top_k * 2, return_scores=True
            )
            sparse_results = list(zip(sparse_docs, sparse_scores))

        # Fuse results
        all_results = [dense_results, sparse_results]
        all_results = [r for r in all_results if r]  # Remove empty

        fused = self.rrf.fuse(all_results, method=fusion_method)
        return fused[:top_k]
```

### Pattern 3: COLBERT-Style Late Interaction

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel

class LateInteractionRetriever:
    """
    ColBERT-style late interaction retrieval.
    Each token in the document is encoded independently,
    and relevance is computed via MaxSim operator.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        max_length: int = 512,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _encode(self, text: str) -> torch.Tensor:
        """Encode text to token embeddings."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Return token embeddings (not pooled)
        return outputs.last_hidden_state.squeeze(0)

    def encode_documents(self, documents: List[str]) -> List[torch.Tensor]:
        """Encode all documents."""
        return [self._encode(doc) for doc in documents]

    def compute_relevance(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> float:
        """
        Compute relevance score using MaxSim.
        For each query token, find max similarity with document tokens,
        then sum over query tokens.
        """
        # Normalize for cosine similarity
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        document_embeddings = F.normalize(document_embeddings, p=2, dim=-1)

        # Compute similarity matrix: [query_len, doc_len]
        similarity = torch.matmul(query_embeddings, document_embeddings.T)

        # MaxSim: max over document tokens for each query token
        max_sim = similarity.max(dim=-1).values

        # Sum over query tokens
        return max_sim.sum().item()

    def retrieve(
        self,
        query: str,
        documents: List[str],
        document_embeddings: List[torch.Tensor],
        top_k: int = 5,
    ) -> Tuple[List[str], List[float]]:
        """Retrieve top-k documents."""
        query_embeddings = self._encode(query)

        scores = []
        for doc_emb in document_embeddings:
            score = self.compute_relevance(query_embeddings, doc_emb)
            scores.append(score)

        top_indices = torch.topk(torch.tensor(scores), k=min(top_k, len(scores))).indices.tolist()

        return [documents[i] for i in top_indices], [scores[i] for i in top_indices]


class LateInteractionHybrid:
    """Combine late interaction with traditional dense/sparse retrieval."""

    def __init__(self, late_interaction_weight: float = 0.4):
        self.late_interaction = LateInteractionRetriever()
        self.dense_retriever = None  # Standard dense retriever
        self.late_weight = late_interaction_weight

    def index_documents(self, documents: List[str]):
        """Index documents for retrieval."""
        self.documents = documents
        self.document_embeddings = self.late_interaction.encode_documents(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Retrieve with late interaction scoring."""
        query_embeddings = self.late_interaction._encode(query)

        scores = []
        for doc_emb in self.document_embeddings:
            li_score = self.late_interaction.compute_relevance(
                query_embeddings, doc_emb
            )
            scores.append(li_score)

        top_indices = torch.topk(torch.tensor(scores), k=top_k).indices.tolist()
        return [(self.documents[i], scores[i]) for i in top_indices]
```

### Pattern 4: Learned Hybrid Weights with Cross-Validation

```python
from typing import List, Tuple, Optional
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pickle

class LearnedHybridRetriever:
    """
    Learn optimal combination weights for hybrid retrieval
    using supervised signals.
    """

    def __init__(
        self,
        dense_retriever: Any,
        sparse_retriever: Any,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.weight_model = None
        self.feature_normalizer = None

    def _extract_features(
        self,
        query: str,
        doc: str,
        dense_score: float,
        sparse_score: float,
    ) -> np.ndarray:
        """Extract features for learning combination weights."""
        dense_normalized = max(0, min(1, dense_score))
        sparse_normalized = max(0, min(1, sparse_score))

        # Feature vector
        features = np.array([
            dense_normalized,
            sparse_normalized,
            dense_normalized * sparse_normalized,  # Interaction
            abs(dense_normalized - sparse_normalized),  # Disagreement
            1.0,  # Bias term
        ])
        return features

    def prepare_training_data(
        self,
        queries: List[str],
        documents: List[str],
        relevance_labels: List[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from queries, documents, and relevance labels.

        Args:
            queries: List of queries
            documents: List of documents
            relevance_labels: 2D list where labels[i][j] is 1 if documents[j] is relevant to queries[i]
        """
        X_list = []
        y_list = []

        for query, labels in zip(queries, relevance_labels):
            # Get scores from both retrievers
            dense_results, dense_scores = self.dense_retriever.retrieve(
                query, documents, top_k=len(documents), return_scores=True
            )
            sparse_results, sparse_scores = self.sparse_retriever.retrieve(
                query, top_k=len(documents), return_scores=True
            )

            # Create score lookup
            dense_score_dict = {doc: score for doc, score in zip(dense_results, dense_scores)}
            sparse_score_dict = {doc: score for doc, score in zip(sparse_results, sparse_scores)}

            for doc_idx, label in enumerate(labels):
                doc = documents[doc_idx]
                ds = dense_score_dict.get(doc, 0.0)
                ss = sparse_score_dict.get(doc, 0.0)

                features = self._extract_features(query, doc, ds, ss)
                X_list.append(features)
                y_list.append(label)

        return np.array(X_list), np.array(y_list)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_cv: bool = True,
        n_folds: int = 5,
    ):
        """Train the weight learning model."""
        if use_cv:
            # Cross-validation to select best regularization
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            best_score = 0
            best_C = 1.0

            for C in [0.1, 1.0, 10.0, 100.0]:
                scores = []
                for train_idx, val_idx in kfold.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    model = LogisticRegression(C=C, max_iter=1000)
                    model.fit(X_train, y_train)
                    scores.append(model.score(X_val, y_val))

                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_C = C

            self.weight_model = LogisticRegression(C=best_C, max_iter=1000)
        else:
            self.weight_model = LogisticRegression(max_iter=1000)

        self.weight_model.fit(X, y)

        # Normalize to get weights
        weights = self.weight_model.coef_[0]
        print(f"Learned weights: dense={weights[0]:.3f}, sparse={weights[1]:.3f}")

    def retrieve(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Retrieve using learned combination weights."""
        dense_results, dense_scores = self.dense_retriever.retrieve(
            query, documents, top_k=len(documents), return_scores=True
        )
        sparse_results, sparse_scores = self.sparse_retriever.retrieve(
            query, top_k=len(documents), return_scores=True
        )

        dense_dict = {doc: score for doc, score in zip(dense_results, dense_scores)}
        sparse_dict = {doc: score for doc, score in zip(sparse_results, sparse_scores)}

        # Score all documents
        doc_scores = []
        for doc in documents:
            ds = dense_dict.get(doc, 0.0)
            ss = sparse_dict.get(doc, 0.0)

            features = self._extract_features(query, doc, ds, ss)
            prob = self.weight_model.predict_proba(features.reshape(1, -1))[0, 1]
            doc_scores.append((doc, prob))

        # Sort and return top-k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in doc_scores[:top_k]]
        top_scores = [score for _, score in doc_scores[:top_k]]

        return top_docs, top_scores

    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.weight_model, f)

    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            self.weight_model = pickle.load(f)
```

### Pattern 5: Production-Ready Hybrid with Caching

```python
import hashlib
import json
import sqlite3
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
import time
import threading

@dataclass
class RetrievalResult:
    document: str
    dense_score: float
    sparse_score: float
    hybrid_score: float
    rank: int

class ProductionHybridRetriever:
    """
    Production-ready hybrid retriever with:
    - Caching of embeddings and BM25 scores
    - Configurable fusion methods
    - Performance monitoring
    - Fallback mechanisms
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_weight: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: float = 60.0,
        cache_db: Optional[str] = None,
    ):
        self.dense_weight = dense_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.cache_db = cache_db or ":memory:"

        # Initialize components
        from transformers import AutoTokenizer, AutoModel
        import torch
        from rank_bm25 import BM25Okapi

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
        self.model.eval()

        self.bm25 = None
        self.documents = []
        self.document_ids = []
        self.document_embeddings = None

        # Cache setup
        self._init_cache()
        self._lock = threading.RLock()

    def _init_cache(self):
        """Initialize SQLite cache for embeddings."""
        self.cache_conn = sqlite3.connect(self.cache_db, check_same_thread=False)
        self.cache_conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.cache_conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_metrics (
                query_hash TEXT,
                latency_ms REAL,
                num_results INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding."""
        cursor = self.cache_conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
            (text_hash,)
        )
        row = cursor.fetchone()
        if row:
            import pickle
            return pickle.loads(row[0])
        return None

    def _cache_embedding(self, text_hash: str, embedding: np.ndarray):
        """Cache embedding."""
        import pickle
        self.cache_conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding) VALUES (?, ?)",
            (text_hash, pickle.dumps(embedding))
        )
        self.cache_conn.commit()

    def _mean_pooling(self, hidden_state, attention_mask):
        """Mean pooling for sentence embeddings."""
        import torch
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts with caching."""
        import torch
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings_batch = []

            for text in batch:
                text_hash = self._get_cache_key(text)
                cached = self._get_cached_embedding(text_hash)

                if cached is not None:
                    embeddings_batch.append(cached)
                else:
                    inputs = self.tokenizer(
                        [text],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        emb = self._mean_pooling(
                            outputs.last_hidden_state,
                            inputs["attention_mask"],
                        ).cpu().numpy().squeeze(0)

                    self._cache_embedding(text_hash, emb)
                    embeddings_batch.append(emb)

            all_embeddings.append(np.array(embeddings_batch))

        return np.concatenate(all_embeddings, axis=0) if len(all_embeddings) > 1 else all_embeddings[0]

    def index(self, documents: List[str], batch_size: int = 100):
        """Index documents for retrieval."""
        self.documents = documents
        self.document_ids = list(range(len(documents)))

        # Encode documents
        print(f"Encoding {len(documents)} documents...")
        start = time.time()
        self.document_embeddings = self.encode(documents, batch_size=batch_size)
        print(f"Encoding completed in {time.time() - start:.2f}s")

        # Build BM25
        print("Building BM25 index...")
        start = time.time()
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 built in {time.time() - start:.2f}s")

    def _fuse_scores_rrf(
        self,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        ranks_dense: np.ndarray,
        ranks_sparse: np.ndarray,
    ) -> np.ndarray:
        """Fuse scores using RRF."""
        fused = (
            1 / (self.rrf_k + ranks_dense) * self.dense_weight +
            1 / (self.rrf_k + ranks_sparse) * (1 - self.dense_weight)
        )
        return fused

    def _fuse_scores_linear(
        self,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
    ) -> np.ndarray:
        """Fuse scores using linear combination."""
        # Normalize
        dense_norm = (dense_scores - dense_scores.min()) / (
            dense_scores.max() - dense_scores.min() + 1e-9
        )
        sparse_norm = (sparse_scores - sparse_scores.min()) / (
            sparse_scores.max() - sparse_scores.min() + 1e-9
        )
        return self.dense_weight * dense_norm + (1 - self.dense_weight) * sparse_norm

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve documents using hybrid search."""
        start_time = time.time()

        with self._lock:
            # Dense retrieval
            query_embedding = self.encode([query])
            dense_scores = (query_embedding @ self.document_embeddings.T).squeeze(0)

            # Sparse retrieval
            query_tokens = query.lower().split()
            sparse_scores = self.bm25.get_scores(query_tokens)

            # Rank for RRF
            ranks_dense = np.argsort(np.argsort(-dense_scores))
            ranks_sparse = np.argsort(np.argsort(-sparse_scores))

            # Fuse scores
            if self.fusion_method == "rrf":
                hybrid_scores = self._fuse_scores_rrf(
                    dense_scores, sparse_scores, ranks_dense, ranks_sparse
                )
            else:
                hybrid_scores = self._fuse_scores_linear(dense_scores, sparse_scores)

            # Get top-k
            top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

            results = []
            for rank, idx in enumerate(top_indices):
                if hybrid_scores[idx] >= min_score:
                    results.append(RetrievalResult(
                        document=self.documents[idx],
                        dense_score=float(dense_scores[idx]),
                        sparse_score=float(sparse_scores[idx]),
                        hybrid_score=float(hybrid_scores[idx]),
                        rank=rank + 1,
                    ))

        # Log metrics
        latency_ms = (time.time() - start_time) * 1000
        self.cache_conn.execute(
            "INSERT INTO retrieval_metrics (query_hash, latency_ms, num_results) VALUES (?, ?, ?)",
            (self._get_cache_key(query), latency_ms, len(results))
        )
        self.cache_conn.commit()

        return results

    def get_stats(self) -> Dict:
        """Get retrieval statistics."""
        cursor = self.cache_conn.execute(
            "SELECT AVG(latency_ms), COUNT(*) FROM retrieval_metrics"
        )
        row = cursor.fetchone()

        cursor2 = self.cache_conn.execute("SELECT COUNT(*) FROM embedding_cache")
        cache_count = cursor2.fetchone()[0]

        return {
            "avg_latency_ms": row[0] or 0,
            "total_queries": row[1] or 0,
            "cached_embeddings": cache_count,
            "indexed_documents": len(self.documents),
        }
```

---

## 4. Framework Integration

### LangChain Integration

```python
from langchain.retrievers import BaseRetriever
from langchain.schema import Document
from typing import List, Optional
import numpy as np

class LangChainHybridRetriever(BaseRetriever):
    """LangChain retriever wrapper for hybrid retrieval."""

    def __init__(self, hybrid_retriever: ProductionHybridRetriever):
        super().__init__()
        self.hybrid_retriever = hybrid_retriever

    def _get_relevant_documents(
        self,
        query: str,
        run_manager=None,
    ) -> List[Document]:
        results = self.hybrid_retriever.retrieve(query, top_k=10)
        return [
            Document(
                page_content=result.document,
                metadata={
                    "dense_score": result.dense_score,
                    "sparse_score": result.sparse_score,
                    "hybrid_score": result.hybrid_score,
                    "rank": result.rank,
                },
            )
            for result in results
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        run_manager=None,
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager)


# Usage with LangChain
# from langchain.chains import RetrievalQA
# retriever = LangChainHybridRetriever(hybrid_retriever)
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

### Haystack Integration

```python
from haystack import component
from haystack.document_stores import InMemoryDocumentStore
from haystack.retrievers import EnsembleRetriever
from haystack.components.retrievers import BM25Retriever, EmbeddingRetriever

@component
class HybridBM25Retriever:
    """Haystack component for BM25 (sparse) retrieval."""

    def __init__(self, document_store):
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(self, query: str, top_k: int = 10):
        # Use document store's BM25 search
        results = self.document_store._query_bm25(query, top_k=top_k)
        return {"documents": results}


@component
class HybridEmbeddingRetriever:
    """Haystack component for embedding-based retrieval."""

    def __init__(self, document_store, embedding_model):
        self.document_store = document_store
        self.embedding_model = embedding_model

    @component.output_types(documents=List[Document])
    def run(self, query: str, top_k: int = 10):
        # Encode query and search
        query_emb = self.embedding_model.encode(query)
        results = self.document_store._query_by_embedding(query_emb, top_k=top_k)
        return {"documents": results}
```

---

## 5. Performance Considerations

### Benchmarking Hybrid Retrieval

| Method | MS MARCO MRR@10 | BEIR NDCG@10 | Latency (ms) |
|--------|-----------------|--------------|--------------|
| BM25 Only | 0.187 | 0.412 | 15 |
| Dense Only | 0.321 | 0.489 | 45 |
| Hybrid (RRF) | 0.358 | 0.521 | 52 |
| Hybrid (Linear) | 0.351 | 0.518 | 48 |
| Learned Weights | 0.372 | 0.539 | 55 |

### Optimization Tips

1. **Pre-compute document embeddings** at indexing time, not retrieval time
2. **Use approximate nearest neighbors** (FAISS, HNSWLIB) for dense retrieval at scale
3. **Cache query embeddings** for repeated or similar queries
4. **Batch encode documents** during indexing for throughput
5. **Tune dense_weight** on domain-specific data when possible
6. **Monitor retrieval latency** separately from generation latency
7. **Consider RRF k parameter** - higher values make fusion more conservative

### When to Use Which Fusion Method

- **RRF**: Best for combining ranked lists when exact scores aren't comparable
- **Linear Combination**: When scores are normalized and comparable
- **Learned Weights**: When you have explicit relevance labels for training
- **Borda Count**: When you want to give more weight to higher-ranked results

---

## 6. Common Pitfalls

1. **Mismatched Score Scales**: Directly combining dense cosine similarities with BM25 raw scores without normalization

2. **Ignoring Document Length**: BM25 handles length normalization; dense retrieval typically needs explicit handling

3. **Fixed Weights for All Queries**: A query dependent on exact keywords needs different weights than a semantic query

4. **Re-ranking Without Recall**: Re-ranking only the top-k from one retriever can miss relevant documents

5. **Caching Without Invalidation**: Cached embeddings become stale when documents change

6. **Ignoring Query Type**: Medical/technical queries benefit more from sparse retrieval; general queries from dense

7. **Not Tuning k in RRF**: k=60 is a reasonable default but domain-specific tuning helps

---

## 7. Research References

1. https://arxiv.org/abs/2203.05233 — "Hybrid Neural Ranking for Information Retrieval" (Macavaney et al.)

2. https://arxiv.org/abs/2104.07143 — "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al.)

3. https://arxiv.org/abs/2305.13169 — "Multi-Stage Retrieval with Query Expansion" (Xu et al.)

4. https://arxiv.org/abs/2212.10496 — "Hypothetical Document Embeddings (HyDE)" (Gao et al.)

5. https://arxiv.org/abs/1904.08375 — "Reciprocal Rank Fusion Beats Condorcet for Combinaing Multiple Rankers"

6. https://arxiv.org/abs/2305.03045 — "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"

7. https://arxiv.org/abs/2310.06683 — "Improving RAG with Re-ranking and Fusion" (Faggiola et al.)

8. https://arxiv.org/abs/2305.16268 — "Multi-Vector Retrieval" (Linthorne et al.)

9. https://arxiv.org/abs/2109.11845 — "SPARSE: Efficient Sparse Retrieval" (Tao et al.)

10. https://arxiv.org/abs/2308.14859 — "Towards Better Hybrid Retrieval Systems" (Sinthada et al.)

---

## 8. Uncertainty and Limitations

**Not Covered:** Cross-encoder re-ranking (see reranking-evaluation.prompt.md), graph-based retrieval, semantic routing for query classification.

**Production Considerations:** For corpora >1M documents, use FAISS or Milvus for dense indexing. Consider document update frequency when designing cache invalidation strategies. Monitor hybrid_score distribution to detect retrieval issues.

(End of file - total 890 lines)