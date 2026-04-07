# RAG (Retrieval-Augmented Generation)

Comprehensive retrieval-augmented generation pipeline implementation, from document ingestion through reranked generation.

## Overview

This module provides production-ready implementations for building RAG systems:
- Document loading and preprocessing
- Chunking and segmentation strategies
- Embedding generation and indexing
- Vector search and hybrid retrieval
- Semantic reranking
- Prompt assembly and grounded generation
- Quality evaluation and faithfulness testing

## Structure

```
rag/
├── README.md (this file)
├── ingestion/        # Document loaders and connectors
├── chunking/         # Text segmentation strategies
├── embeddings/       # Embedding model wrappers
├── indexing/         # Vector index creation/maintenance
├── retrieval/        # Retriever implementations and fusion
├── reranking/        # Cross-encoder rerankers
├── generation/       # Prompt assembly and generation
└── eval/             # Retrieval and generation evaluation
```

## Quick Start

### 1. Document Ingestion
```python
from rag.ingestion import DocumentLoader

loader = DocumentLoader()
docs = loader.load_pdf("document.pdf")
```

### 2. Chunking
```python
from rag.chunking import RecursiveChunker

chunker = RecursiveChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk(docs)
```

### 3. Embedding & Indexing
```python
from rag.embeddings import EmbeddingModel
from rag.indexing import VectorIndex

embedder = EmbeddingModel()
index = VectorIndex()
index.add(chunks, embedder)
```

### 4. Retrieval
```python
from rag.retrieval import DenseRetriever

retriever = DenseRetriever(index)
results = retriever.retrieve("query", k=5)
```

### 5. Generation
```python
from rag.generation import PromptAssembler

assembler = PromptAssembler()
prompt = assembler.assemble(results, query)
```

## Key Concepts

### Document Ingestion
- Multiple loader types (PDF, HTML, Markdown, databases)
- Metadata preservation
- Format normalization

### Chunking Strategies
- Recursive chunking (semantic boundaries)
- Sliding window
- Document-aware segmentation
- Code-aware splitting

### Embeddings
- Dense embeddings (e.g., all-MiniLM, BGE)
- Sparse embeddings (BM25)
- Hybrid dense+sparse
- Quantization and caching

### Retrieval
- Dense similarity search
- Sparse keyword search
- Hybrid fusion methods
- Metadata filtering
- Time-aware retrieval

### Reranking
- Cross-encoder models
- LLM-as-judge reranking
- Diversity-aware reranking
- Fusion with retrieval scores

### Generation
- In-context learning with retrieved documents
- Citation and grounding
- Faithfulness checking
- Multi-hop reasoning

## Core Components

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| ingestion | Load diverse document types | `DocumentLoader`, `PDFLoader`, `WebLoader` |
| chunking | Segment documents optimally | `RecursiveChunker`, `SlidingWindowChunker` |
| embeddings | Generate dense/sparse vectors | `EmbeddingModel`, `SparseEmbedder` |
| indexing | Store and organize vectors | `VectorIndex`, `HybridIndex` |
| retrieval | Find relevant documents | `DenseRetriever`, `HybridRetriever` |
| reranking | Reorder and filter results | `CrossEncoderReranker`, `LLMReranker` |
| generation | Assemble and generate responses | `PromptAssembler`, `GroundedGenerator` |
| eval | Measure quality | `RetrievalEvaluator`, `FaithfulnessChecker` |

## Configuration

See `../configs/` for example configurations:
- `rag_basic.yaml` - Minimal setup
- `rag_production.yaml` - Full pipeline with optimization
- `rag_hybrid.yaml` - Dense + sparse retrieval

## Advanced Topics

### Hybrid Search
- Combining dense and sparse retrieval
- Normalization and fusion strategies
- Parameter tuning for balance

### Filtering
- Metadata filtering during retrieval
- Temporal constraints
- Source-based filtering
- Type-based filtering

### Multimodal RAG
- Image and table indexing
- Cross-modal retrieval
- Multimodal embedding models

### Agents with RAG
- Tool-augmented retrieval
- Iterative refinement
- Multi-source synthesis

## Performance Tips

1. **Chunking**: Balance between too small (<256) and too large (>1024)
2. **Embedding Model**: Use BGE or similar for best retrieval quality
3. **Index Type**: Use HNSW for <10M docs, IVF for larger scales
4. **Reranking**: Always include for production (5-10% overhead, major quality gain)
5. **Caching**: Cache embeddings and reranker scores

## Evaluation

Key metrics:
- **Retrieval**: NDCG, MRR, Recall@k, Hit Rate
- **Generation**: BLEU, ROUGE, BERTScore
- **Faithfulness**: Hallucination detection, grounding score
- **Latency**: End-to-end query latency, per-component breakdown

## References

- See `../skills/rag-advanced/` for comprehensive research documents
- Main README: `../../README.md`
- Evaluation guide: `../evaluation/README.md`
- Research index: `../skills/research-archive/rag-advanced/`

## Contributing

When adding new retrieval or generation methods:
1. Add implementation to appropriate subdirectory
2. Include unit tests
3. Add integration example
4. Document configuration options
5. Include performance benchmarks

## License

See LICENSE file in repository root.
