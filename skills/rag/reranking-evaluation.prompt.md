# RAG Reranking and Evaluation — Agentic Skill Prompt

Cross-encoder reranking, LLM-as-judge evaluation, and RAG metrics (NDCG, MRR, Recall@K).

---

## 1. Identity and Mission

Optimize retrieval quality through intelligent reranking and measure RAG system performance rigorously.

---

## 2. Cross-Encoder Reranking

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple

class CrossEncoderReranker:
    """Rerank documents using cross-encoders."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> Tuple[List[str], List[float]]:
        """Rerank documents using cross-encoder."""
        scores = []
        
        for doc in documents:
            inputs = self.tokenizer(
                query, doc, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            score = torch.sigmoid(logits).item()
            scores.append(score)
        
        # Sort by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        return (
            [documents[i] for i in ranked_indices[:top_k]],
            [scores[i] for i in ranked_indices[:top_k]],
        )

# Usage
reranker = CrossEncoderReranker()
query = "What is machine learning?"
docs = ["ML is...", "Python is...", "Deep learning is..."]
reranked, scores = reranker.rerank(query, docs, top_k=2)
```

---

## 3. LLM-as-Judge Evaluation

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class RelevanceJudgment:
    document: str
    relevance_score: float  # 0-5
    reasoning: str

class LLMAsJudge:
    """Use LLM to evaluate relevance."""
    
    def __init__(self, llm_model_name: str = "gpt2"):
        # In production, use OpenAI/Claude API
        self.model_name = llm_model_name
    
    def judge_relevance(
        self,
        query: str,
        document: str,
        rubric: str = "Does this document answer the query?",
    ) -> RelevanceJudgment:
        """Judge document relevance using LLM."""
        
        prompt = f"""
Query: {query}
Document: {document}

{rubric}

Score (0-5): 
Reasoning: 
"""
        
        # In production: call LLM API
        # response = llm_call(prompt)
        # Parse response to extract score and reasoning
        
        return RelevanceJudgment(
            document=document,
            relevance_score=4.0,  # Placeholder
            reasoning="Document is highly relevant.",
        )
    
    def batch_judge(
        self,
        query: str,
        documents: List[str],
    ) -> List[RelevanceJudgment]:
        """Judge multiple documents."""
        return [
            self.judge_relevance(query, doc)
            for doc in documents
        ]
```

---

## 4. RAG Evaluation Metrics

```python
from typing import List
import numpy as np

class RAGMetrics:
    """Compute standard information retrieval metrics."""
    
    @staticmethod
    def ndcg(
        relevances: List[float],
        k: int = 10,
    ) -> float:
        """Normalized Discounted Cumulative Gain."""
        dcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(relevances[:k])
        )
        
        ideal = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(sorted(relevances, reverse=True)[:k])
        )
        
        return dcg / ideal if ideal > 0 else 0.0
    
    @staticmethod
    def mrr(
        relevances: List[float],
        threshold: float = 0.5,
    ) -> float:
        """Mean Reciprocal Rank."""
        for i, rel in enumerate(relevances):
            if rel >= threshold:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def recall_at_k(
        relevances: List[float],
        k: int = 10,
        threshold: float = 0.5,
    ) -> float:
        """Recall@k."""
        total_relevant = sum(1 for r in relevances if r >= threshold)
        if total_relevant == 0:
            return 0.0
        
        retrieved_relevant = sum(
            1 for r in relevances[:k] if r >= threshold
        )
        
        return retrieved_relevant / total_relevant
    
    @staticmethod
    def precision_at_k(
        relevances: List[float],
        k: int = 10,
        threshold: float = 0.5,
    ) -> float:
        """Precision@k."""
        retrieved_relevant = sum(
            1 for r in relevances[:k] if r >= threshold
        )
        return retrieved_relevant / k if k > 0 else 0.0

# Usage
relevances = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
print(f"NDCG@5: {RAGMetrics.ndcg(relevances, k=5):.3f}")
print(f"MRR: {RAGMetrics.mrr(relevances):.3f}")
print(f"Recall@5: {RAGMetrics.recall_at_k(relevances, k=5):.3f}")
```

---

## 5. References

1. https://arxiv.org/abs/2110.07305 — "Cross-Encoders for Ranking" (SBERT)
2. https://www.sbert.net/docs/pretrained_cross-encoders.html — Pre-trained cross-encoders
3. https://github.com/niderhoff/nlp-metrics — NLP metrics implementations
4. https://arxiv.org/abs/2004.07159 — "Understanding and Improving Retrieval Ranking"
5. https://github.com/castorini/anserini — Dense/sparse retrieval toolkit
6. https://arxiv.org/abs/2307.10045 — "LLM-as-Judge for RAG Evaluation"
7. https://huggingface.co/spaces/mteb/leaderboard — MTEB retrieval leaderboard
8. https://arxiv.org/abs/2305.14283 — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
9. https://github.com/embeddings-everywhere/evaluator — RAG evaluation framework
10. https://arxiv.org/abs/2309.04947 — "Context Precision and Recall for RAG"
11. https://github.com/zetavg/LlamaIndex — Query/document matching patterns
12. https://huggingface.co/datasets/ir_datasets — IR datasets for evaluation
13. https://arxiv.org/abs/2304.09848 — "Retrieval Augmented Generation with FAISS"
14. https://github.com/microsoft/pygaggle — Ranking and evaluation tools
15. https://arxiv.org/abs/2105.01601 — "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"
16. https://github.com/stanford-futuredata/ColBERT — ColBERT implementation

---

## 6. Uncertainty and Limitations

**Not Covered:** LLM hallucination evaluation, multi-hop reasoning metrics, graph-based RAG. **Production:** Implement automated evaluation loops, track metric drift, A/B test retrieval strategies.
