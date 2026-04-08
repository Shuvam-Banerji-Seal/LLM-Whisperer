# Document Chunking Module

Document chunking strategies for splitting large documents into manageable chunks for RAG systems.

## Overview

This module provides multiple chunking strategies optimized for different document types and use cases:

- **Recursive Chunking**: Semantic-aware chunking using progressive separator fallback
- **Sliding Window**: Fixed-size overlapping chunks to preserve context
- **Document Aware**: Respects document boundaries and structure
- **Code Aware**: Special handling for code with language-specific logic

## Key Classes

### ChunkingStrategy
Abstract base class for all chunking strategies.

### RecursiveChunker
Chunks text recursively by semantic boundaries. Attempts to preserve meaning by using progressively smaller separators (paragraph → sentence → word → character).

**Best for:**
- General-purpose document chunking
- Mixed content (paragraphs, lists, code)
- Maintaining semantic coherence

### SlidingWindowChunker
Creates overlapping chunks using a fixed-size sliding window. Useful for preserving context at chunk boundaries.

**Best for:**
- Dense text requiring maximum context
- Time-series or sequential data
- When overlap is critical

### DocumentChunker
Factory class that creates appropriate chunker based on configuration.

### ChunkMerger
Merges small or similar chunks to avoid fragmentation.

## Configuration

See `config.py` for detailed configuration options:

```python
from rag.chunking import ChunkingConfig, ChunkingMethod

config = ChunkingConfig(
    method=ChunkingMethod.RECURSIVE,
    chunk_size=512,
    chunk_overlap=50,
    min_chunk_size=100,
)
```

## Usage Examples

### Basic Chunking
```python
from rag.chunking import DocumentChunker, ChunkingConfig

config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
chunker = DocumentChunker(config)

chunks = chunker.chunk(
    "Your long document text here...",
    metadata={"doc_id": "doc1", "source": "pdf"}
)

for chunk in chunks:
    print(f"Chunk {chunk.id}: {len(chunk.content)} chars")
```

### Multiple Documents
```python
texts = ["Document 1...", "Document 2..."]
metadata = [
    {"doc_id": "doc1"},
    {"doc_id": "doc2"}
]

all_chunks = chunker.chunk_multiple(texts, metadata)
```

### Chunk Merging
```python
from rag.chunking import ChunkMerger

merger = ChunkMerger(min_size=100)
merged_chunks = merger.merge_by_size(chunks)
```

## Performance Considerations

1. **Chunk Size**: 
   - Small chunks (256-512): Better for dense retrieval, more chunks
   - Large chunks (1024+): Preserve more context, fewer chunks

2. **Overlap**: 
   - 0%: No overlap, fast but may miss context at boundaries
   - 10-20%: Recommended for most cases
   - 30%+: More expensive but better boundary coverage

3. **Separator Priority**: 
   - Recursive chunker tries larger separators first
   - Falls back to smaller ones if chunks exceed size limit
   - Empty string falls back to character-level splitting

## Advanced Features

### Semantic Chunking
Preserve chunks at semantic boundaries by analyzing content similarity.

### Code-Aware Chunking
Respects programming language structures (functions, classes, blocks).

### Metadata Preservation
Maintains document metadata and adds chunk-specific metadata:
- `chunk_index`: Position in document
- `chunk_text_length`: Length of chunk content
- `start_idx`: Character position in original text
- `end_idx`: Character position in original text

## Integration with RAG Pipeline

Chunking is typically the second step in a RAG pipeline:

1. **Document Ingestion** → Load documents
2. **Chunking** → Segment into chunks (this module)
3. **Embedding** → Generate embeddings
4. **Indexing** → Store in vector database
5. **Retrieval** → Find relevant chunks
6. **Reranking** → Order by relevance
7. **Generation** → Create response

## Error Handling

All classes include proper error handling:
- Invalid configurations raise `ValueError` during initialization
- Logging at appropriate levels (info, warning, error)
- Graceful handling of edge cases (empty text, tiny chunks)

## References

- RAG main README: `../README.md`
- Configuration details: `config.py`
- Core implementations: `core.py`
