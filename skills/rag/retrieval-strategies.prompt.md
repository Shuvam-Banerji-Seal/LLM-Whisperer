# Advanced Retrieval Strategies for RAG — Agentic Skill Prompt

Dense/sparse retrieval, multi-stage pipelines, query expansion, and HyDE for production RAG systems.

---

## 1. Identity and Mission

Implement robust retrieval pipelines that maximize relevance and coverage while minimizing latency and cost.

---

## 2. Dense vs. Sparse Retrieval

```python
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import numpy as np

class DenseRetriever:
    """Embedding-based dense retrieval using Sentence Transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents to embeddings."""
        embeddings = []
        
        for doc in documents:
            inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(embedding)
        
        return np.concatenate(embeddings, axis=0)
    
    def retrieve(
        self,
        query: str,
        documents: List[str],
        embeddings: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[str], List[float]]:
        """Retrieve top-k documents by similarity."""
        query_emb = self.encode_documents([query])
        
        # Cosine similarity
        scores = (query_emb @ embeddings.T).squeeze(0)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return (
            [documents[i] for i in top_indices],
            [scores[i].item() for i in top_indices],
        )

class SparseRetriever:
    """BM25-based sparse retrieval."""
    
    def __init__(self):
        from rank_bm25 import BM25Okapi
        self.bm25 = None
    
    def index_documents(self, documents: List[str]) -> None:
        """Index documents with BM25."""
        from rank_bm25 import BM25Okapi
        
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.documents = documents
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve using BM25."""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return (
            [self.documents[i] for i in top_indices],
            [scores[i].item() for i in top_indices],
        )

# Usage
dense = DenseRetriever()
docs = ["Python is a programming language", "Dogs are animals"]
embs = dense.encode_documents(docs)
results, scores = dense.retrieve("What is Python?", docs, embs, top_k=1)

sparse = SparseRetriever()
sparse.index_documents(docs)
results, scores = sparse.retrieve("Python", top_k=1)
```

---

## 3. Multi-Stage Retrieval

```python
from dataclasses import dataclass
from typing import List

@dataclass
class RetrievalResult:
    document: str
    dense_score: float
    sparse_score: float
    combined_score: float

class MultiStageRetrieval:
    """Combine dense and sparse retrieval."""
    
    def __init__(self, dense_weight: float = 0.7):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.dense_weight = dense_weight
    
    def retrieve(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Multi-stage: sparse → dense → rerank."""
        # Stage 1: Sparse (BM25) for recall
        sparse_results, sparse_scores = self.sparse_retriever.retrieve(
            query, documents, top_k=top_k
        )
        
        # Stage 2: Dense on top sparse results
        dense_embs = self.dense_retriever.encode_documents(sparse_results)
        dense_results, dense_scores = self.dense_retriever.retrieve(
            query, sparse_results, dense_embs, top_k=len(sparse_results)
        )
        
        # Combine scores
        results = []
        for i, (doc, d_score) in enumerate(zip(dense_results, dense_scores)):
            s_score = sparse_scores[sparse_results.index(doc)]
            combined = self.dense_weight * d_score + (1 - self.dense_weight) * s_score
            results.append(RetrievalResult(doc, d_score, s_score, combined))
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results[:rerank_top_k]
```

---

## 4. Query Expansion with HyDE

```python
class HyDEQueryExpansion:
    """Hypothetical Document Embeddings (HyDE)."""
    
    def __init__(self, llm_model_name: str = "gpt2"):
        from transformers import pipeline
        self.generator = pipeline("text-generation", model=llm_model_name)
    
    def expand_query(self, query: str, num_hypothetical: int = 3) -> List[str]:
        """Generate hypothetical documents for query."""
        prompt = f"Write a document that would answer: {query}\n\nDocument: "
        
        outputs = self.generator(
            prompt,
            max_length=100,
            num_return_sequences=num_hypothetical,
        )
        
        # Extract generated text
        hypothetical_docs = [
            output["generated_text"].replace(prompt, "").strip()
            for output in outputs
        ]
        
        return [query] + hypothetical_docs
    
    def retrieve_with_expansion(
        self,
        query: str,
        documents: List[str],
        embeddings,
        top_k: int = 5,
    ):
        """Retrieve using expanded queries."""
        expanded_queries = self.expand_query(query, num_hypothetical=2)
        
        all_scores = None
        for q in expanded_queries:
            q_emb = self.dense_retriever.encode_documents([q])
            scores = (q_emb @ embeddings.T).squeeze(0)
            
            if all_scores is None:
                all_scores = scores
            else:
                all_scores += scores
        
        # Average scores
        avg_scores = all_scores / len(expanded_queries)
        top_indices = np.argsort(avg_scores)[::-1][:top_k]
        
        return [documents[i] for i in top_indices]
```

---

## 5. References

1. https://arxiv.org/abs/2104.07143 — "Dense Passage Retrieval for Open-Domain QA" (DPR)
2. https://github.com/facebookresearch/DPR — DPR official
3. https://arxiv.org/abs/1909.05326 — "BM25: A Probabilistic Model for Ranking Documents"
4. https://github.com/dorianbrown/rank_bm25 — BM25 implementation
5. https://arxiv.org/abs/2212.10496 — "Hypothetical Document Embeddings (HyDE)" (Gao et al.)
6. https://arxiv.org/abs/2305.13169 — "Multi-Stage Retrieval with Query Expansion"
7. https://github.com/embeddings-everywhere/transformers-retrieval — Retrieval patterns
8. https://www.sbert.net/ — Sentence Transformers documentation
9. https://arxiv.org/abs/2310.06683 — "Improving RAG with Re-ranking and Fusion"
10. https://huggingface.co/datasets/ir_datasets — Information retrieval datasets
11. https://github.com/microsoft/ONNX-Runtime — ONNX for inference optimization
12. https://arxiv.org/abs/1811.03356 — "Dense Passage Retrieval for Open-Domain Question Answering"
13. https://github.com/karpukhin/retrieval-scale — Large-scale retrieval patterns
14. https://arxiv.org/abs/2305.16268 — "Multi-Vector Retrieval"
15. https://github.com/nmslib/hnswlib — HNSW indexing for fast retrieval
16. https://faiss.ai/ — FAISS vector indexing library

---

## 6. Uncertainty and Limitations

**Not Covered:** Cross-encoder ranking, graph-based retrieval, semantic routing. **Production:** Use FAISS/HNSWLIB for indexing, cache embeddings, monitor retrieval latency.
