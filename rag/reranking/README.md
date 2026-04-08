# Reranking Module

Reranking retrieved documents for improved relevance.

## Overview

Rerank retrieved documents using multiple strategies:
- Cross-encoder models (MS MARCO trained)
- LLM-as-judge reranking
- Diversity-aware reranking (MMR)
- Fusion with retrieval scores

## Key Classes

### Reranker
Main reranker using cross-encoder models.

### RankingStrategy
Abstract strategy for custom ranking algorithms.

### RerankerFactory
Factory for creating reranker instances.

## Usage

```python
from rag.reranking import Reranker, RerankerConfig, RerankerType

config = RerankerConfig(
    reranker_type=RerankerType.CROSS_ENCODER,
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_k=5,
)

reranker = Reranker(config)
ranked = reranker.rerank(query, candidates)

for doc, score in ranked:
    print(f"{doc}: {score:.3f}")
```

## Performance

- Reranking adds 5-10% latency overhead
- Significantly improves retrieval quality (often +20-30% NDCG)
- Essential for production RAG systems

## References

- RAG main README: `../README.md`
- Configuration: `config.py`
- Core implementations: `core.py`
