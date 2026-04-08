# Vector Indexing Module

Vector index creation and management for RAG systems.

## Overview

Build and manage vector indices for efficient similarity search:
- Multiple index types (HNSW, IVF, FLAT)
- Configurable index parameters
- Index optimization and parameter tuning

## Key Classes

### VectorIndex
In-memory vector index with cosine similarity search.

### IndexBuilder
Builds indices from embedding vectors.

### IndexOptimizer
Optimizes index parameters for query patterns.

## Usage

```python
from rag.indexing import IndexBuilder, IndexConfig, IndexType

config = IndexConfig(
    index_type=IndexType.HNSW,
    embedding_dim=384,
)

builder = IndexBuilder(config)
index = builder.build(vectors, doc_ids, metadata)

# Search
results = index.search(query_vector, top_k=5)
```

## References

- RAG main README: `../README.md`
- Configuration: `config.py`
- Core implementations: `core.py`
