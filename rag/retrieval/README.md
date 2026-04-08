# Document Retrieval Module

Document retrieval from vector stores for RAG systems.

## Overview

Retrieve relevant documents using multiple strategies:
- Dense embedding search
- Sparse BM25 keyword search
- Hybrid dense+sparse fusion
- Ensemble retrieval methods
- Metadata filtering

## Key Classes

### DocumentRetriever
Main retriever using dense embeddings.

### HybridRetriever
Combines dense and sparse retrieval strategies.

### RetrieverConfig
Configuration for retrieval parameters.

## Usage

```python
from rag.retrieval import DocumentRetriever, HybridRetriever

# Dense retrieval
retriever = DocumentRetriever(index, embedder, config)
docs = retriever.retrieve("What is machine learning?", top_k=5)

# Hybrid retrieval
hybrid = HybridRetriever(dense_retriever, sparse_retriever, alpha=0.5)
docs = hybrid.retrieve(query, top_k=5)
```

## Fusion Methods

- **Reciprocal Rank Fusion (RRF)**: Default, score-agnostic
- **Score Normalization**: Min-max or z-score normalization
- **Weighted Sum**: Manual weight tuning for dense vs sparse

## Performance Tips

1. Use dense retrieval for semantic similarity
2. Add BM25 for keyword matching
3. Adjust alpha (0.3-0.7) for fusion balance
4. Filter by metadata before ranking

## References

- RAG main README: `../README.md`
- Configuration: `config.py`
- Core implementations: `core.py`
