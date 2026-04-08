# RAG Module Component Reference Guide

**Complete Implementation Reference for All 8 RAG Components**

## Quick Navigation

1. [Chunking](#1-chunking) - Document segmentation
2. [Embeddings](#2-embeddings) - Vector representations
3. [Evaluation](#3-evaluation) - Quality metrics
4. [Generation](#4-generation) - Response assembly
5. [Indexing](#5-indexing) - Vector database
6. [Ingestion](#6-ingestion) - Document loading
7. [Reranking](#7-reranking) - Result improvement
8. [Retrieval](#8-retrieval) - Document search

---

## 1. Chunking

**Path**: `rag/chunking/`  
**Purpose**: Split documents into semantic chunks for embedding

### Main Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `ChunkingStrategy` | Abstract strategy interface | `chunk(text, metadata)` |
| `RecursiveChunker` | Recursive splitting with fallback separators | `chunk()`, `_recursive_split()`, `_merge_splits()` |
| `SlidingWindowChunker` | Fixed-size overlapping windows | `chunk()` |
| `DocumentChunker` | Factory and orchestrator | `chunk()`, `chunk_multiple()` |
| `ChunkMerger` | Merges small chunks | `merge_by_size()`, `merge_by_separator()` |

### Configuration

```python
from rag.chunking import ChunkingConfig, ChunkingMethod

config = ChunkingConfig(
    method=ChunkingMethod.RECURSIVE,      # chunking strategy
    chunk_size=512,                        # default chunk size
    chunk_overlap=50,                      # overlap between chunks
    min_chunk_size=100,                    # minimum chunk size
    preserve_separators=True,              # keep separators
    separators=["\n\n", "\n", " ", ""],   # fallback separators
)
```

### Usage Example

```python
from rag.chunking import DocumentChunker, ChunkingConfig

config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
chunker = DocumentChunker(config)

chunks = chunker.chunk(
    "Your long document text...",
    metadata={"doc_id": "doc1", "source": "pdf"}
)

for chunk in chunks:
    print(f"ID: {chunk.id}")
    print(f"Content: {chunk.content[:100]}")
    print(f"Position: {chunk.start_idx}-{chunk.end_idx}")
```

---

## 2. Embeddings

**Path**: `rag/embeddings/`  
**Purpose**: Generate and manage embedding vectors

### Main Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `EmbeddingModel` | Load and use embedding models | `embed()`, `embed_single()` |
| `BatchEmbedder` | Efficient batch processing | `embed_batch()`, `embed_multiple_batches()` |
| `EmbeddingCache` | LRU-based caching | `get()`, `put()`, `clear()`, `_evict_lru()` |

### Configuration

```python
from rag.embeddings import EmbeddingConfig, EmbeddingType

config = EmbeddingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_type=EmbeddingType.DENSE,
    embedding_dim=384,
    batch_size=32,
    normalize_embeddings=True,
    use_cuda=True,
    cache_embeddings=True,
    cache_size_mb=1000,
)
```

### Usage Example

```python
from rag.embeddings import EmbeddingModel, BatchEmbedder, EmbeddingCache

# Single embedding
embedder = EmbeddingModel(config)
embedding = embedder.embed_single("Hello world")

# Batch processing
batch_embedder = BatchEmbedder(embedder, batch_size=64)
embeddings_dict = batch_embedder.embed_batch(["text1", "text2"])

# Caching
cache = EmbeddingCache(max_size_mb=500)
cached_emb = cache.get("text")
if cached_emb is None:
    emb = embedder.embed_single("text")
    cache.put("text", emb)
```

---

## 3. Evaluation

**Path**: `rag/eval/`  
**Purpose**: Evaluate RAG system quality

### Main Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `RAGEvaluator` | Main evaluation coordinator | `evaluate_retrieval()`, `evaluate_generation()`, `evaluate_faithfulness()` |
| `MetricCalculator` | Computes standard metrics | `calculate_ndcg()`, `calculate_recall()`, `calculate_rouge()`, `calculate_faithfulness()` |
| `BenchmarkRunner` | Runs comprehensive benchmarks | `run_benchmark()` |

### Configuration

```python
from rag.eval import EvalConfig, EvalMetric

config = EvalConfig(
    metrics=[
        EvalMetric.RETRIEVAL_NDCG,
        EvalMetric.GENERATION_ROUGE,
        EvalMetric.FAITHFULNESS,
    ],
    top_k=5,
    relevance_threshold=0.5,
    faithfulness_threshold=0.7,
)
```

### Metrics Available

- **Retrieval**: NDCG, Recall, Precision, MRR, Hit Rate
- **Generation**: BLEU, ROUGE, BERTScore
- **Faithfulness**: Hallucination detection, grounding score

### Usage Example

```python
from rag.eval import RAGEvaluator, BenchmarkRunner

evaluator = RAGEvaluator(config)

# Evaluate retrieval
retrieval_metrics = evaluator.evaluate_retrieval(
    retrieved_docs=["doc1", "doc2", "doc3"],
    relevant_docs=["doc1", "doc3"],
    query="what is AI?"
)

# Benchmark multiple queries
runner = BenchmarkRunner(evaluator)
results = runner.run_benchmark(
    queries=["q1", "q2"],
    retrieved_results=[["d1", "d2"], ["d3"]],
    relevant_results=[["d1"], ["d3"]],
    generated_texts=["response1", "response2"],
    reference_texts=["ref1", "ref2"]
)
```

---

## 4. Generation

**Path**: `rag/generation/`  
**Purpose**: Assemble prompts and generate responses

### Main Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `TextAugmenter` | Augments text with context | `augment()` |
| `PromptAssembler` | Constructs prompts | `assemble()` |
| `GenerationOrchestrator` | Orchestrates full pipeline | `generate()` |

### Configuration

```python
from rag.generation import GenerationConfig, GenerationMode

config = GenerationConfig(
    mode=GenerationMode.GROUNDED,
    llm_model="mistral-7b",
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    include_citations=True,
    include_confidence=True,
    context_window_size=4,
)
```

### Usage Example

```python
from rag.generation import PromptAssembler, GenerationOrchestrator

# Simple prompt assembly
assembler = PromptAssembler(config)
prompt = assembler.assemble(
    query="What is machine learning?",
    context_docs=["ML is a branch of AI...", "Deep learning..."],
    citations=True
)

# Full orchestration
orchestrator = GenerationOrchestrator(
    retriever=retriever,
    reranker=reranker,
    prompt_assembler=assembler,
    config=config
)

result = orchestrator.generate(
    query="What is ML?",
    llm_generate_fn=lambda prompt, **kw: llm.generate(prompt, **kw)
)

print(result["response"])
print(result["context_docs"])
```

---

## 5. Indexing

**Path**: `rag/indexing/`  
**Purpose**: Store and search embeddings

### Main Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `VectorIndex` | In-memory vector index | `add()`, `search()` |
| `IndexBuilder` | Builds indices | `build()` |
| `IndexOptimizer` | Optimizes parameters | `optimize_search_params()` |

### Configuration

```python
from rag.indexing import IndexConfig, IndexType

config = IndexConfig(
    index_type=IndexType.HNSW,
    embedding_dim=384,
    max_elements=1_000_000,
    ef_construction=200,  # HNSW param
    ef=50,                # HNSW param
    m=16,                 # HNSW param
    metric="cosine",
)
```

### Index Types

- **FLAT**: Exact search, good for small datasets
- **HNSW**: Fast approximate search (recommended)
- **IVF**: Inverted file, good for large datasets
- **FAISS**: Advanced indexing (requires faiss package)
- **ANNOY**: Spotify's approximate search

### Usage Example

```python
from rag.indexing import IndexBuilder

builder = IndexBuilder(config)
index = builder.build(
    vectors=[[0.1, 0.2, ...], ...],
    doc_ids=["doc1", "doc2", ...],
    metadata=[{"source": "pdf"}, ...]
)

# Search
results = index.search(query_vector, top_k=5)
for doc_id, similarity in results:
    print(f"{doc_id}: {similarity:.3f}")
```

---

## 6. Ingestion

**Path**: `rag/ingestion/`  
**Purpose**: Load documents from various sources

### Main Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `DocumentLoader` | Load from multiple formats | `load()` |
| `MetadataExtractor` | Extract document metadata | `extract()` |
| `DocumentPipeline` | Complete ingestion workflow | `process()` |

### Configuration

```python
from rag.ingestion import IngestionConfig, LoaderType

config = IngestionConfig(
    loader_type=LoaderType.PDF,
    extract_metadata=True,
    preserve_formatting=True,
    encoding="utf-8",
    max_file_size_mb=100,
    timeout_seconds=30,
    remove_duplicates=True,
)
```

### Supported Formats

- PDF, Text, Markdown, HTML
- CSV, Database, Web
- Directories (batch loading)

### Usage Example

```python
from rag.ingestion import DocumentPipeline

pipeline = DocumentPipeline(config)

documents = pipeline.process("documents.pdf")

for doc in documents:
    print(f"Content: {doc['content'][:100]}")
    print(f"Metadata: {doc['metadata']}")
    print(f"Word count: {doc['metadata']['word_count']}")
```

---

## 7. Reranking

**Path**: `rag/reranking/`  
**Purpose**: Improve retrieval results through reranking

### Main Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `RankingStrategy` | Abstract ranking interface | `rerank()` |
| `Reranker` | Cross-encoder reranker | `rerank()` |
| `RerankerFactory` | Factory for rerankers | `create()` |

### Configuration

```python
from rag.reranking import RerankerConfig, RerankerType

config = RerankerConfig(
    reranker_type=RerankerType.CROSS_ENCODER,
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    batch_size=32,
    top_k=10,
    score_threshold=0.5,
    normalize_scores=True,
)
```

### Reranker Types

- **CROSS_ENCODER**: MS MARCO trained cross-encoders
- **LLM_JUDGE**: Use LLM for ranking (custom)
- **DIVERSITY**: Maximize diversity (MMR)
- **RANKING**: Custom ranking strategies

### Usage Example

```python
from rag.reranking import Reranker

reranker = Reranker(config)

ranked = reranker.rerank(
    query="What is machine learning?",
    candidates=["ML is...", "Machine learning refers to...", "AI and ML..."]
)

for doc, score in ranked:
    print(f"Score {score:.3f}: {doc[:50]}")
```

---

## 8. Retrieval

**Path**: `rag/retrieval/`  
**Purpose**: Find relevant documents

### Main Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `DocumentRetriever` | Dense embedding search | `retrieve()` |
| `HybridRetriever` | Combines dense + sparse | `retrieve()` |
| `RetrieverConfig` | Configuration | (dataclass) |

### Configuration

```python
from rag.retrieval import RetrieverConfig, RetrieverType

# Note: RetrieverType is defined as an enum
# RetrieverConfig is a dataclass with these fields:
config = type('Config', (), {
    'retriever_type': 'dense',
    'top_k': 5,
    'similarity_threshold': 0.5,
    'use_reranking': True,
    'fusion_method': 'rrf',  # reciprocal rank fusion
})()
```

### Retriever Types

- **DENSE**: Vector similarity search
- **SPARSE**: BM25 keyword search
- **HYBRID**: Combined dense + sparse
- **BM25**: Pure keyword search
- **ENSEMBLE**: Multiple strategies

### Usage Example

```python
from rag.retrieval import DocumentRetriever, HybridRetriever

# Dense retrieval
dense_ret = DocumentRetriever(index, embedder, config)
results = dense_ret.retrieve("What is AI?", top_k=5)

# Hybrid retrieval
hybrid = HybridRetriever(dense_ret, sparse_ret, alpha=0.5)
hybrid_results = hybrid.retrieve("What is AI?", top_k=5)
```

---

## Integration Example: Complete RAG Pipeline

```python
from rag.ingestion import DocumentPipeline, IngestionConfig
from rag.chunking import DocumentChunker, ChunkingConfig
from rag.embeddings import EmbeddingModel, EmbeddingConfig
from rag.indexing import IndexBuilder, IndexConfig
from rag.retrieval import DocumentRetriever
from rag.reranking import Reranker, RerankerConfig
from rag.generation import GenerationOrchestrator, PromptAssembler, GenerationConfig
from rag.eval import RAGEvaluator, EvalConfig

# 1. Ingest documents
ingest_config = IngestionConfig()
pipeline = DocumentPipeline(ingest_config)
docs = pipeline.process("documents/")

# 2. Chunk documents
chunk_config = ChunkingConfig(chunk_size=512)
chunker = DocumentChunker(chunk_config)
all_chunks = []
for doc in docs:
    chunks = chunker.chunk(doc['content'], doc['metadata'])
    all_chunks.extend(chunks)

# 3. Generate embeddings
embed_config = EmbeddingConfig()
embedder = EmbeddingModel(embed_config)
embeddings = embedder.embed([c.content for c in all_chunks])

# 4. Build index
index_config = IndexConfig()
builder = IndexBuilder(index_config)
index = builder.build(
    embeddings,
    [c.id for c in all_chunks],
    [c.metadata for c in all_chunks]
)

# 5. Create retriever
retriever = DocumentRetriever(index, embedder, config)

# 6. Create reranker
rerank_config = RerankerConfig()
reranker = Reranker(rerank_config)

# 7. Create generator
gen_config = GenerationConfig()
assembler = PromptAssembler(gen_config)
orchestrator = GenerationOrchestrator(
    retriever, reranker, assembler, gen_config
)

# 8. Process queries
query = "What is machine learning?"
response = orchestrator.generate(query, llm.generate)

# 9. Evaluate
eval_config = EvalConfig()
evaluator = RAGEvaluator(eval_config)
metrics = evaluator.evaluate_retrieval(
    response['context_docs'],
    ground_truth_docs,
    query
)

print(f"Response: {response['response']}")
print(f"Metrics: {metrics}")
```

---

## Configuration Best Practices

### Chunking
- Small chunks (256-512): Better for dense search, more chunks
- Large chunks (1024+): Preserve more context, fewer chunks
- Overlap: 10-20% recommended for boundary coherence

### Embeddings
- Batch size: 64-128 for GPU, 8-16 for CPU
- Enable GPU if available (10x+ speedup)
- Use caching for repeated texts
- Normalize embeddings for cosine similarity

### Indexing
- HNSW: Recommended for 1K-10M vectors
- IVF: For >10M vectors
- Tune ef_construction (200-400) for quality vs speed

### Retrieval
- Dense alone: Fast, less accurate
- Sparse alone: Slow, keyword-dependent
- Hybrid: Best quality, moderate speed

### Reranking
- Essential for production (5-10% overhead, +20-30% quality)
- Cross-encoder: Best quality
- Always include in retrieval pipeline

### Generation
- GROUNDED: Most faithful, safest
- OPEN_ENDED: More flexible
- Include citations for accountability
- Fact-checking for critical applications

---

## Error Handling

All components include proper error handling:
- Configuration validation on initialization
- Type hints for static checking
- Logging at all levels
- Graceful fallbacks for missing dependencies

---

## Performance Tuning

- **Caching**: Enable embedding cache for repeated texts
- **Batching**: Process multiple items together
- **GPU**: Use GPU acceleration for embeddings/reranking
- **Index**: Choose appropriate index type for dataset size
- **Fusion**: Tune hybrid retrieval weights (0.3-0.7)

---

**Last Updated**: April 8, 2026  
**Version**: 1.0  
**Status**: Production Ready
