# Document Ingestion Module

Document loading and ingestion pipelines for RAG systems.

## Overview

Load and process documents from multiple sources:
- Text files, PDFs, Markdown, HTML
- CSV and database sources
- Directories and web sources
- Format normalization and metadata extraction

## Key Classes

### DocumentLoader
Loads documents from various sources.

### MetadataExtractor
Extracts metadata (word count, source, etc.) from content.

### DocumentPipeline
Complete ingestion pipeline with validation and enrichment.

## Usage

```python
from rag.ingestion import DocumentPipeline, IngestionConfig, LoaderType

config = IngestionConfig(loader_type=LoaderType.PDF)
pipeline = DocumentPipeline(config)

docs = pipeline.process("document.pdf")
for doc in docs:
    print(f"Content: {doc['content'][:100]}")
    print(f"Metadata: {doc['metadata']}")
```

## References

- RAG main README: `../README.md`
- Configuration: `config.py`
- Core implementations: `core.py`
