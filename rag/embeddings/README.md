# Embedding Models Module

Embedding model management with support for dense embeddings, batch processing, and caching.

## Overview

This module provides tools for generating and managing embeddings:

- **EmbeddingModel**: Main wrapper for loading and using embedding models
- **BatchEmbedder**: Efficient batch processing of large text collections
- **EmbeddingCache**: LRU caching to avoid recomputing embeddings

## Key Classes

### EmbeddingModel
Main embedding model wrapper with:
- Support for sentence-transformers models
- Optional GPU acceleration
- Normalization and quantization options
- Single and batch embedding methods

**Supported Models:**
- all-MiniLM-L6-v2 (384d, fast, good quality)
- all-mpnet-base-v2 (768d, high quality)
- bge-base-en (768d, optimized for dense retrieval)
- bge-large-en (1024d, highest quality)

### BatchEmbedder
Handles batch embedding of large document collections:
- Configurable batch size
- Progress tracking
- Error handling
- Text hashing for deduplication

### EmbeddingCache
LRU-based embedding cache:
- Configurable memory limit
- Automatic eviction of least-used items
- Fast retrieval of cached embeddings
- Reduces computation on repeated texts

## Configuration

```python
from rag.embeddings import EmbeddingConfig, EmbeddingType

config = EmbeddingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_type=EmbeddingType.DENSE,
    batch_size=32,
    normalize_embeddings=True,
    cache_embeddings=True,
    cache_size_mb=1000,
)
```

## Usage Examples

### Basic Embedding
```python
from rag.embeddings import EmbeddingModel, EmbeddingConfig

config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedder = EmbeddingModel(config)

# Single text
embedding = embedder.embed_single("Hello world")

# Multiple texts
embeddings = embedder.embed(["Text 1", "Text 2", "Text 3"])
```

### Batch Processing
```python
from rag.embeddings import BatchEmbedder

batch_embedder = BatchEmbedder(embedder, batch_size=64)

# Progress tracking
def progress_callback(current, total):
    print(f"Progress: {current}/{total}")

embeddings = batch_embedder.embed_multiple_batches(
    [["text1", "text2"], ["text3", "text4"]],
    progress_callback=progress_callback
)
```

### Caching Embeddings
```python
from rag.embeddings import EmbeddingCache

cache = EmbeddingCache(max_size_mb=500)

# Check cache
cached = cache.get("Some text")
if cached is None:
    # Generate embedding
    embedding = embedder.embed_single("Some text")
    cache.put("Some text", embedding)
```

## Model Selection Guide

| Model | Dimension | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | Default, low latency |
| all-mpnet-base-v2 | 768 | Fast | Excellent | Balanced |
| bge-base-en | 768 | Fast | Excellent | Dense retrieval |
| bge-large-en | 1024 | Slower | Best | High-quality retrieval |

## Performance Tips

1. **Batch Size**: Use larger batches (64-128) for GPU, smaller (8-16) for CPU
2. **GPU**: Enable `use_cuda=True` if GPU available (10x+ speedup)
3. **Normalization**: Keep enabled for cosine similarity search
4. **Caching**: Enable for production with sufficient memory
5. **Model Size**: Smaller models (384d) are faster but less accurate

## Memory Management

- Cache size is configured in MB
- Automatically evicts least-recently-used items when full
- Each cached embedding takes ~embedding_dim * 8 bytes + text overhead
- Monitor cache hits/misses in logs

## Integration with RAG Pipeline

Embeddings are central to RAG:

1. **Chunking** → Create document chunks
2. **Embedding** → Generate embeddings (this module)
3. **Indexing** → Store embeddings in vector DB
4. **Retrieval** → Find similar chunks via embedding search

## Error Handling

- Validates configuration on initialization
- Graceful fallback if model fails to load
- Proper error messages for missing dependencies
- Logging at all stages

## References

- RAG main README: `../README.md`
- Configuration details: `config.py`
- Core implementations: `core.py`
