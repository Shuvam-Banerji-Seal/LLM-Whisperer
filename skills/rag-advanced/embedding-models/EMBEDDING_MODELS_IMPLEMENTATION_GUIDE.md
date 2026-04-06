# Embedding Models Implementation Guide: Production-Ready Setup (2026)

**Document Type**: Implementation Reference Guide  
**Target Audience**: ML Engineers, DevOps, Product Teams  
**Scope**: Setup, deployment, optimization, and troubleshooting

---

## Table of Contents

1. [Quick Setup Guide](#quick-setup-guide)
2. [Model Selection Decision Tree](#model-selection-decision-tree)
3. [Self-Hosted Deployment](#self-hosted-deployment)
4. [API-Based Deployment](#api-based-deployment)
5. [Vector Database Configuration](#vector-database-configuration)
6. [Fine-Tuning Pipeline](#fine-tuning-pipeline)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring and Evaluation](#monitoring-and-evaluation)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## Quick Setup Guide

### 5-Minute Setup: OpenAI text-embedding-3

```bash
# Install dependencies
pip install openai python-dotenv

# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Quick test
python -c "
from openai import OpenAI
client = OpenAI()
result = client.embeddings.create(
    model='text-embedding-3-small',
    input='Test query'
)
print(f'Embedding dimensions: {len(result.data[0].embedding)}')
print(f'Success! Ready for production.')
"
```

### 15-Minute Setup: Self-Hosted BGE-M3

```bash
# Install dependencies
pip install sentence-transformers torch qdrant-client

# Download model (first run only, ~1.3GB)
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
embedding = model.encode('Test query')
print(f'Model loaded. Embedding shape: {embedding.shape}')
"

# Start Qdrant (if using)
docker run -p 6333:6333 qdrant/qdrant

# Integration test
python example_bge_m3.py
```

**example_bge_m3.py**:
```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load model
model = SentenceTransformer('BAAI/bge-m3')

# Embed sample documents
docs = ["How to fix bugs?", "Database optimization", "Python best practices"]
embeddings = model.encode(docs)

# Connect to Qdrant
client = QdrantClient(":memory:")
client.create_collection("test", VectorParams(1024, Distance.COSINE))

# Index documents
points = [PointStruct(id=i, vector=emb.tolist()) for i, emb in enumerate(embeddings)]
client.upsert("test", points)

# Search
query_emb = model.encode("How to debug?").tolist()
results = client.search("test", query_emb, limit=2)
for res in results:
    print(f"ID: {res.id}, Score: {res.score:.3f}")
```

---

## Model Selection Decision Tree

```
START HERE: What's your primary constraint?
│
├─→ [DATA SOVEREIGNTY REQUIRED]
│   ├─ Open source license needed?
│   │  ├─ YES → BGE-M3 (MIT) or Qwen3-8B (Apache 2.0)
│   │  └─ NO  → On-premises option available (Cohere VPC)
│   │
│   └─→ Budget/Infrastructure?
│       ├─ Limited → BGE-M3 (smallest, 568M params)
│       └─ Available → Qwen3-8B (best multilingual)
│
├─→ [MAXIMUM RETRIEVAL QUALITY]
│   ├─ API acceptable?
│   │  ├─ YES → Voyage-3-large ($0.06/1M, best NDCG@10)
│   │  └─ NO  → Self-host Qwen3-8B or NV-Embed-v2
│   │
│   └─→ Multilingual needed?
│       ├─ YES → Qwen3-8B (MTEB 70.58)
│       └─ NO  → NV-Embed-v2 (MTEB 69.32, non-commercial)
│
├─→ [COST OPTIMIZATION]
│   ├─ Scale (100M+ documents)?
│   │  ├─ YES → Self-host (BGE-M3 or Qwen3)
│   │  └─ NO  → text-embedding-3-small ($0.02/1M)
│   │
│   └─→ Engineering resources available?
│       ├─ Limited → Use API
│       └─ Available → Self-host and save 80% at scale
│
├─→ [DOMAIN SPECIALIZATION]
│   ├─ Specialized domain (legal/finance/code)?
│   │  ├─ YES → Use domain variant (Voyage-law-2, etc.)
│   │  └─ NO  → General-purpose model adequate
│   │
│   └─→ Need fine-tuning after?
│       ├─ Maybe → Start with domain variant (avoid FT overhead)
│       └─ Definitely → Self-host + fine-tune
│
├─→ [LONG DOCUMENTS (>8K tokens)]
│   ├─ Context window needed?
│   │  ├─ 32K+  → Cohere (128K), Qwen3/Voyage (32K)
│   │  ├─ 8-32K → Most models support
│   │  └─ <8K   → any model
│   │
│   └─→ Avoid chunking?
│       ├─ YES → Cohere embed-v4 only (128K)
│       └─ NO  → Any model works
│
├─→ [MULTILINGUAL (100+ languages)]
│   ├─ Cross-lingual alignment critical?
│   │  ├─ YES → Gemini Embedding 2 (0.997 R@1)
│   │  └─ NO  → Cohere or BGE-M3
│   │
│   └─→ Self-host required?
│       ├─ YES → Qwen3-8B (MTEB 70.58)
│       └─ NO  → Gemini or Cohere
│
├─→ [MULTIMODAL (text + image/video/audio)]
│   ├─ Modalities needed?
│   │  ├─ 5 (all) → Gemini Embedding 2
│   │  ├─ 3 (text+image+video) → Voyage Multimodal 3.5
│   │  └─ 2 (text+image) → Cohere Embed v4, Jina CLIP
│   │
│   └─→ Trade-off: Quality vs cost
│       ├─ Quality → Gemini
│       └─ Cost   → Voyage MM 3.5
│
└─→ [ECOSYSTEM INTEGRATION]
    ├─ Using OpenAI APIs elsewhere?
    │  ├─ YES → text-embedding-3 (ecosystem simplicity)
    │  └─ NO  → Choose on merits above
    │
    └─→ Using Anthropic Claude?
        ├─ YES → Voyage AI (Anthropic-recommended)
        └─ NO  → Any model works

═══════════════════════════════════════════════════════════════════
DECISION MATRIX (Quick Lookup)
═══════════════════════════════════════════════════════════════════

Scenario                          | Recommended Model(s)
─────────────────────────────────┼───────────────────────────────
Startup MVP                       | text-embedding-3-small
General RAG (self-hosted)        | BGE-M3 or Qwen3-8B
Maximum retrieval quality        | Voyage-3-large
Cost at 100M+ documents          | Qwen3-8B (self) or Voyage (API)
Multilingual + privacy           | Qwen3-8B
Domain-specific (legal/fin)      | Voyage-domain OR fine-tune
Long docs no chunking            | Cohere (128K only)
Multimodal + cost-effective      | Voyage MM 3.5
Regulated industry               | BGE-M3 or Cohere VPC
Research/non-commercial          | NV-Embed-v2
Compliance/auditability          | nomic-embed-text
Edge/low-latency                 | all-MiniLM-L6-v2
```

---

## Self-Hosted Deployment

### 3.1 Complete Qwen3-Embedding-8B Setup

**Hardware Requirements**:
- GPU: 1x A100 (40GB) or 2x A10G (24GB each) or equivalent
- Memory: 32GB+ system RAM
- Storage: 50GB (model + cache)
- Network: 10Gbps recommended for bulk ingestion

**Installation**:

```bash
# 1. Create virtual environment
python -m venv venv_embedding
source venv_embedding/bin/activate

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentence-transformers accelerate
pip install fastapi uvicorn python-multipart
pip install numpy scikit-learn

# 3. Create embedding server
cat > embedding_server.py << 'EOF'
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model on startup
model = SentenceTransformer(
    'Alibaba-NLP/gte-qwen2-7b-instruct',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    trust_remote_code=True
)

app = FastAPI(title="Qwen3 Embedding Server")

@app.post("/embed")
async def embed(texts: List[str], task_type: str = "retrieval"):
    """
    Embed texts using Qwen3-Embedding-8B
    
    task_type options:
    - "retrieval": for documents and queries
    - "classification": for short texts
    - "clustering": for grouping
    """
    try:
        # Add task prefix if needed
        if task_type == "retrieval.query":
            texts = [f"Instruct: Represent this sentence for searching: {t}" 
                    for t in texts]
        
        embeddings = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return {
            "embeddings": embeddings.tolist(),
            "model": "qwen3-embedding-8b",
            "dimensions": embeddings.shape[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwen3-embedding-8b"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
EOF

# 4. Start server
python embedding_server.py
# Server ready at http://localhost:8000

# 5. Test with client
python -c "
import requests
import json

response = requests.post(
    'http://localhost:8000/embed',
    json={
        'texts': ['How to fix memory leaks?', 'Database optimization'],
        'task_type': 'retrieval'
    }
)
result = response.json()
print(f'Embedded {len(result[\"embeddings\"])} texts')
print(f'Dimensions: {result[\"dimensions\"]}')
"
```

**Docker Setup** (Recommended for Production):

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers sentence-transformers fastapi uvicorn accelerate

# Copy server code
COPY embedding_server.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "embedding_server.py"]
```

**Docker Compose** (Multi-GPU):

```yaml
version: '3.8'

services:
  embedding-server:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - huggingface_cache:/root/.cache/huggingface
      - model_cache:/app/model_cache
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - TRANSFORMERS_CACHE=/app/model_cache
      - HF_HOME=/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]

volumes:
  huggingface_cache:
  model_cache:
```

**Launch**:

```bash
# Build image
docker build -t embedding-server:latest .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f embedding-server

# Monitor GPU usage
nvidia-smi -l 1
```

### 3.2 BGE-M3 Self-Hosting (Lightweight Alternative)

```python
from sentence_transformers import SentenceTransformer
import torch

# Load BGE-M3 (smaller, 568M params)
model = SentenceTransformer('BAAI/bge-m3', device='cuda')

# Use with dense retrieval
dense_embeddings = model.encode(
    documents,
    batch_size=64,
    normalize_embeddings=True
)

# Also get sparse embeddings (lexical)
sparse_embeddings = model.encode(
    documents,
    batch_size=32,
    convert_to_sparse_embeddings=True
)

# And multi-vector (ColBERT-style)
multi_embeddings = model.encode(
    documents,
    convert_to_multi_embeddings=True
)
```

---

## API-Based Deployment

### 4.1 OpenAI text-embedding-3 Integration

```python
from openai import OpenAI
import os
from typing import List
import numpy as np
from functools import lru_cache

class EmbeddingClient:
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        
    def embed(self, text: str, dimensions: int = None) -> List[float]:
        """Embed single text."""
        params = {
            "model": self.model,
            "input": text
        }
        if dimensions:
            params["dimensions"] = dimensions
            
        response = self.client.embeddings.create(**params)
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str], dimensions: int = None) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        params = {
            "model": self.model,
            "input": texts
        }
        if dimensions:
            params["dimensions"] = dimensions
            
        response = self.client.embeddings.create(**params)
        # Sort by index to maintain order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]

# Usage
client = EmbeddingClient(model="text-embedding-3-large")

# Single embedding
embedding = client.embed("How to fix bugs?")
print(f"Dimensions: {len(embedding)}")

# Batch embedding (more efficient)
texts = ["How to fix bugs?", "Database optimization", "Python tips"]
embeddings = client.embed_batch(texts, dimensions=1024)  # 67% storage savings

# Calculate similarity
similarity = np.dot(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.3f}")
```

### 4.2 Voyage AI Integration

```python
import voyageai
from typing import List

class VoyageEmbedding:
    def __init__(self, api_key: str = None, model: str = "voyage-3-large"):
        voyageai.api_key = api_key
        self.model = model
        
    def embed(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """
        input_type: "query" or "document" (affects output)
        """
        result = voyageai.Client().embed(
            model=self.model,
            input=texts,
            input_type=input_type
        )
        return result.embeddings

# Usage
voyage = VoyageEmbedding()

# Embed documents
docs = ["Bug fixing guide", "Python optimization"]
doc_embeddings = voyage.embed(docs, input_type="document")

# Embed query differently
query_emb = voyage.embed(["How to fix bugs?"], input_type="query")[0]

# Find most similar
import numpy as np
similarities = [np.dot(query_emb, doc_emb) for doc_emb in doc_embeddings]
best_match = docs[np.argmax(similarities)]
print(f"Best match: {best_match}")
```

### 4.3 Cohere embed-v4 Integration

```python
import cohere
from typing import List

class CohereEmbedding:
    def __init__(self, api_key: str = None):
        self.client = cohere.ClientV2(api_key=api_key)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents for storage."""
        response = self.client.embed(
            model="embed-v4",
            texts=texts,
            input_type="search_document",
            embedding_types=["float"]
        )
        return response.embeddings.float
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query for search (different input type!)."""
        response = self.client.embed(
            model="embed-v4",
            texts=[query],
            input_type="search_query",
            embedding_types=["float"]
        )
        return response.embeddings.float[0]

# Usage
cohere = CohereEmbedding()

# Index documents
docs = ["Python async programming", "Database indexing strategies"]
doc_embeddings = cohere.embed_documents(docs)

# Search with query
query_emb = cohere.embed_query("How to optimize async code?")

# Find similar
import numpy as np
scores = [np.dot(query_emb, doc_emb) for doc_emb in doc_embeddings]
print(f"Top match score: {max(scores):.3f}")
```

---

## Vector Database Configuration

### 5.1 Qdrant Setup and Integration

```bash
# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant

# Or with persistent storage
docker run -p 6333:6333 \
  -v ./qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Production Configuration** (qdrant-config.yaml):

```yaml
debug: false
log_level: info

server:
  host: 0.0.0.0
  port: 6333
  grpc_port: 6334

storage:
  # Snapshots and backups
  snapshots_path: ./snapshots
  wal_capacity_mb: 200
  
  # Performance tuning
  max_optimization_threads: 4
  
performance:
  # Increase search performance
  max_search_batch_size: 100
  search_timeout_sec: 30

cluster:
  enabled: false  # Set to true for clustering
```

**Python Integration**:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
import numpy as np

class QdrantVectorDB:
    def __init__(self, url: str = "http://localhost:6333", 
                 collection_name: str = "documents"):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        
    def create_collection(self, vector_size: int = 1024):
        """Create collection with vector configuration."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE  # Best for normalized embeddings
            ),
            optimizers_config={
                "default": {
                    "enabled": True,
                    "default": {"work_dir": "/tmp"}
                }
            }
        )
        
    def index_documents(self, texts: List[str], embeddings: List[List[float]]):
        """Index documents with embeddings."""
        points = [
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": text,
                    "text_length": len(text),
                    "indexed_at": datetime.now().isoformat()
                }
            )
            for i, (embedding, text) in enumerate(zip(embeddings, texts))
        ]
        
        # Upsert in batches for large datasets
        batch_size = 1000
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"Indexed {i + len(batch)}/{len(points)} documents")
    
    def search(self, query_embedding: List[float], top_k: int = 10):
        """Search for similar documents."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            # Optional: filter by metadata
            # query_filter=Filter(
            #     must=[
            #         HasIdCondition(has_id=[1, 2, 3])
            #     ]
            # )
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text"),
                "metadata": result.payload
            }
            for result in results
        ]
    
    def delete_collection(self):
        """Clean up."""
        self.client.delete_collection(collection_name=self.collection_name)

# Usage
from datetime import datetime

db = QdrantVectorDB(url="http://localhost:6333")
db.create_collection(vector_size=1024)

# Index documents
texts = ["Bug fixing guide", "Database optimization", "Python best practices"]
embeddings = np.random.randn(len(texts), 1024).tolist()  # Replace with real embeddings
db.index_documents(texts, embeddings)

# Search
query_emb = np.random.randn(1024).tolist()  # Replace with real query embedding
results = db.search(query_emb, top_k=5)
for res in results:
    print(f"ID: {res['id']}, Score: {res['score']:.3f}, Text: {res['text'][:50]}...")
```

### 5.2 Weaviate Configuration for BGE-M3 Hybrid Search

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from typing import List

class WeaviateVectorDB:
    def __init__(self, url: str = "http://localhost:8080"):
        self.client = weaviate.connect_to_local(url=url)
        
    def create_collection_hybrid(self, collection_name: str, vector_size: int = 1024):
        """Create collection supporting hybrid search (dense + BM25)."""
        self.client.collections.create(
            name=collection_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.OBJECT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
            vector_index_config=Configure.VectorIndex.hnsw(
                ef_construction=400,
                ef=1024,
                max_connections=64,
            ),
            inverted_index_config=Configure.inverted_index(
                bm25_b=0.75,
                bm25_k1=1.25,
            ),
        )
    
    def index_with_hybrid(self, collection_name: str, 
                         texts: List[str], embeddings: List[List[float]]):
        """Index documents with both dense and sparse (BM25) indexing."""
        collection = self.client.collections.get(collection_name)
        
        with collection.batch.fixed_size(batch_size=100) as batch:
            for text, embedding in zip(texts, embeddings):
                batch.add_object(
                    properties={"text": text},
                    vector=embedding
                )
    
    def hybrid_search(self, collection_name: str, query: str, 
                     query_vector: List[float], top_k: int = 10):
        """Hybrid search combining vector and BM25."""
        collection = self.client.collections.get(collection_name)
        
        results = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=0.5,  # Balance between text (0) and vector (1)
            limit=top_k,
        )
        
        return [
            {
                "text": obj.properties["text"],
                "score": obj.metadata.score
            }
            for obj in results.objects
        ]

# Usage
db = WeaviateVectorDB()
db.create_collection_hybrid("documents", vector_size=1024)

# Index with hybrid support
texts = ["text1...", "text2...", "text3..."]
embeddings = []  # Get from BGE-M3
db.index_with_hybrid("documents", texts, embeddings)

# Search with hybrid
results = db.hybrid_search("documents", "query text", query_vector=[...], top_k=10)
```

---

## Fine-Tuning Pipeline

### 6.1 Complete Fine-Tuning Setup

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingFineTuner:
    def __init__(self, base_model: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(base_model)
        self.base_model = base_model
        
    def prepare_training_data(self, 
                             query_doc_pairs: List[tuple],
                             negative_docs: List[str] = None) -> List[InputExample]:
        """
        Prepare triplet training data.
        
        Args:
            query_doc_pairs: List of (query, relevant_doc) tuples
            negative_docs: List of irrelevant documents
        """
        if negative_docs is None:
            negative_docs = []
            
        examples = []
        for query, positive_doc in query_doc_pairs:
            # Find hard negative (BM25 top hit that's not relevant)
            # For now, use random from negative_docs
            negative_doc = np.random.choice(negative_docs) if negative_docs else "unrelated text"
            
            examples.append(InputExample(
                texts=[query, positive_doc, negative_doc],
                label=0  # Triplet loss uses label=0 for all
            ))
        
        return examples
    
    def mine_hard_negatives(self, 
                          queries: List[str],
                          documents: List[str],
                          gold_pairs: List[tuple],
                          top_k: int = 5) -> List[str]:
        """
        Mine hard negatives using retrieval.
        Returns documents that are retrieved but not relevant.
        """
        # Embed everything
        query_embeddings = self.model.encode(queries, normalize_embeddings=True)
        doc_embeddings = self.model.encode(documents, normalize_embeddings=True)
        
        hard_negatives = set()
        gold_docs = {doc for _, doc in gold_pairs}
        
        for query_emb in query_embeddings:
            # Find top-k most similar documents
            similarities = cosine_similarity([query_emb], doc_embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:]
            
            # Collect non-gold documents
            for idx in top_indices:
                doc = documents[idx]
                if doc not in gold_docs:
                    hard_negatives.add(doc)
        
        return list(hard_negatives)
    
    def fine_tune(self,
                  training_examples: List[InputExample],
                  output_path: str = "./fine-tuned-model",
                  epochs: int = 5,
                  batch_size: int = 16,
                  warmup_steps: int = 100,
                  learning_rate: float = 2e-5):
        """Fine-tune the model."""
        
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.TripletLoss(self.model)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate, 'eps': 1e-6},
            weight_decay=0.01,
            show_progress_bar=True,
            use_amp=True,  # Mixed precision for speedup
        )
        
        # Save model
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.model.save(output_path)
        print(f"Model saved to {output_path}")
        
        return self.model
    
    def evaluate(self,
                 test_queries: List[str],
                 test_docs: List[str],
                 gold_pairs: List[tuple],
                 top_k: int = 10) -> dict:
        """Evaluate fine-tuned model on test set."""
        query_embeddings = self.model.encode(test_queries, normalize_embeddings=True)
        doc_embeddings = self.model.encode(test_docs, normalize_embeddings=True)
        
        # Calculate metrics
        ndcg_scores = []
        recall_scores = []
        
        for query, query_emb in zip(test_queries, query_embeddings):
            # Get ground truth relevant docs
            relevant = {doc for q, doc in gold_pairs if q == query}
            
            if not relevant:
                continue
            
            # Rank docs by similarity
            similarities = cosine_similarity([query_emb], doc_embeddings)[0]
            ranked_docs = [test_docs[i] for i in np.argsort(similarities)[::-1][:top_k]]
            
            # Calculate NDCG@10
            dcg = sum(
                1 / np.log2(i + 2) for i, doc in enumerate(ranked_docs)
                if doc in relevant
            )
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), top_k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
            
            # Calculate Recall@10
            recall = len([d for d in ranked_docs if d in relevant]) / len(relevant)
            recall_scores.append(recall)
        
        return {
            "ndcg@10": np.mean(ndcg_scores),
            "recall@10": np.mean(recall_scores),
            "sample_count": len(ndcg_scores)
        }

# Usage
finetuner = EmbeddingFineTuner("BAAI/bge-m3")

# Prepare data
queries = ["How to fix bugs?", "Database optimization"]
documents = ["Bug fixing guide", "Debug tips", "Database indexing"]
gold_pairs = [("How to fix bugs?", "Bug fixing guide"),
              ("Database optimization", "Database indexing")]
negatives = ["Weather forecast", "Sports news"]

# Mine hard negatives
hard_negs = finetuner.mine_hard_negatives(queries, documents, gold_pairs)
all_negatives = negatives + hard_negs

# Prepare training
training_data = finetuner.prepare_training_data(gold_pairs, all_negatives)

# Fine-tune
finetuner.fine_tune(training_data, epochs=3, batch_size=8)

# Evaluate
metrics = finetuner.evaluate(queries, documents, gold_pairs)
print(f"NDCG@10: {metrics['ndcg@10']:.3f}")
print(f"Recall@10: {metrics['recall@10']:.3f}")
```

---

## Performance Optimization

### 7.1 Caching Strategy

```python
import redis
import hashlib
import json
from typing import List, Optional
from functools import wraps

class EmbeddingCache:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379,
                 ttl_hours: int = 24):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.ttl_seconds = ttl_hours * 3600
        
    def _cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        hash_val = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"emb:{model}:{hash_val}"
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._cache_key(text, model)
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, text: str, model: str, embedding: List[float]):
        """Cache embedding."""
        key = self._cache_key(text, model)
        self.redis_client.setex(
            key, 
            self.ttl_seconds,
            json.dumps(embedding)
        )
    
    def batch_get_missing(self, texts: List[str], model: str) -> tuple:
        """Get cached embeddings and return uncached texts."""
        cached_results = {}
        uncached_texts = []
        
        for text in texts:
            embedding = self.get(text, model)
            if embedding:
                cached_results[text] = embedding
            else:
                uncached_texts.append(text)
        
        return cached_results, uncached_texts
    
    def batch_set(self, texts: List[str], model: str, embeddings: List[List[float]]):
        """Cache multiple embeddings."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, model, embedding)

# Usage with OpenAI
from openai import OpenAI

cache = EmbeddingCache(redis_host="localhost", ttl_hours=24)
client = OpenAI()

def embed_with_cache(texts: List[str], model: str = "text-embedding-3-large"):
    """Embed texts with caching."""
    
    # Check cache
    cached, uncached = cache.batch_get_missing(texts, model)
    
    # Embed uncached
    if uncached:
        response = client.embeddings.create(
            model=model,
            input=uncached
        )
        new_embeddings = [item.embedding for item in response.data]
        cache.batch_set(uncached, model, new_embeddings)
        
        # Merge results
        embeddings = {}
        embeddings.update(cached)
        for text, emb in zip(uncached, new_embeddings):
            embeddings[text] = emb
    else:
        embeddings = cached
    
    # Return in original order
    return [embeddings[text] for text in texts]

# Test
texts = ["How to fix bugs?", "Database optimization"]
embeddings = embed_with_cache(texts)
print(f"Embedded {len(embeddings)} texts with caching")
```

### 7.2 Batch Processing for Throughput

```python
from typing import List, Generator
import time

def batch_embed_with_rate_limit(
    texts: List[str],
    embed_fn,
    batch_size: int = 100,
    delay_between_batches: float = 1.0) -> List[List[float]]:
    """
    Process embeddings in batches with rate limiting.
    
    Args:
        texts: Texts to embed
        embed_fn: Function that takes list of texts and returns embeddings
        batch_size: Size of each batch
        delay_between_batches: Delay in seconds between batches
    """
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_fn(batch)
        all_embeddings.extend(embeddings)
        
        # Rate limiting
        if i + batch_size < len(texts):
            time.sleep(delay_between_batches)
        
        # Progress
        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")
    
    return all_embeddings

# Example: Index 1M documents from a file
def read_documents(file_path: str, chunk_size: int = 10000) -> Generator[List[str], None, None]:
    """Generator for reading documents in chunks."""
    batch = []
    with open(file_path, 'r') as f:
        for line in f:
            batch.append(line.strip())
            if len(batch) == chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Process in batches
from openai import OpenAI
client = OpenAI()

all_embeddings = []
for doc_batch in read_documents("documents.jsonl", chunk_size=10000):
    embeddings = batch_embed_with_rate_limit(
        doc_batch,
        lambda texts: client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        ).data,
        batch_size=100,
        delay_between_batches=1.0
    )
    all_embeddings.extend(embeddings)
    print(f"Total processed: {len(all_embeddings)}")
```

---

## Monitoring and Evaluation

### 8.1 Production Monitoring

```python
import logging
from datetime import datetime
import time
from typing import List
import numpy as np

class EmbeddingMonitor:
    def __init__(self, log_file: str = "embedding_metrics.log"):
        self.logger = logging.getLogger("EmbeddingMonitor")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def log_embedding_latency(self, text_length: int, latency_ms: float):
        """Log embedding latency."""
        self.logger.info(f"LATENCY,text_length={text_length},latency_ms={latency_ms:.2f}")
    
    def log_cache_hit(self, hit: bool):
        """Log cache hit/miss."""
        self.logger.info(f"CACHE_HIT,hit={hit}")
    
    def log_dimension_stats(self, embeddings: List[List[float]]):
        """Log embedding statistics."""
        embeddings_array = np.array(embeddings)
        self.logger.info(
            f"EMBEDDING_STATS,"
            f"dimensions={embeddings_array.shape[1]},"
            f"mean_norm={np.mean(np.linalg.norm(embeddings_array, axis=1)):.4f},"
            f"std_norm={np.std(np.linalg.norm(embeddings_array, axis=1)):.4f}"
        )

# Usage
monitor = EmbeddingMonitor()

# Track embedding operation
start = time.time()
embeddings = embed_function("Some text")
latency = (time.time() - start) * 1000
monitor.log_embedding_latency(len("Some text"), latency)
monitor.log_dimension_stats(embeddings)
```

### 8.2 Continuous Evaluation

```python
from datetime import datetime, timedelta
from typing import Dict, List

class EmbeddingEvaluation:
    def __init__(self, eval_data_file: str):
        """Load evaluation dataset."""
        self.eval_data = self._load_eval_data(eval_data_file)
        self.historical_scores = []
        
    def _load_eval_data(self, file_path: str) -> List[Dict]:
        """Load query-document pairs with relevance."""
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def evaluate_model(self, embed_fn) -> Dict[str, float]:
        """Evaluate embedding model on test set."""
        queries = [item['query'] for item in self.eval_data]
        documents = [item['document'] for item in self.eval_data]
        labels = [item['relevant'] for item in self.eval_data]
        
        # Embed
        query_embeddings = embed_fn([q for q in queries])
        doc_embeddings = embed_fn([d for d in documents])
        
        # Calculate metrics
        from sklearn.metrics import ndcg_score
        
        # Create relevance matrix (simplified)
        ndcg_scores = []
        for q_emb, label in zip(query_embeddings, labels):
            similarities = [np.dot(q_emb, d_emb) for d_emb in doc_embeddings]
            # NDCG calculation
            ranked_relevance = [labels[i] for i in np.argsort(similarities)[::-1]]
            ndcg = ndcg_score([[1 if l else 0 for l in ranked_relevance]], [similarities])
            ndcg_scores.append(ndcg)
        
        metrics = {
            "ndcg@10": np.mean(ndcg_scores),
            "evaluated_at": datetime.now().isoformat()
        }
        
        self.historical_scores.append(metrics)
        return metrics
    
    def detect_degradation(self, threshold_pct: float = 5.0) -> bool:
        """Detect model degradation."""
        if len(self.historical_scores) < 2:
            return False
        
        prev_score = self.historical_scores[-2]["ndcg@10"]
        curr_score = self.historical_scores[-1]["ndcg@10"]
        
        degradation_pct = (prev_score - curr_score) / prev_score * 100
        return degradation_pct > threshold_pct

# Schedule evaluation
import schedule

evaluator = EmbeddingEvaluation("eval_dataset.json")

def daily_evaluation():
    metrics = evaluator.evaluate_model(embed_function)
    print(f"NDCG@10: {metrics['ndcg@10']:.3f}")
    
    if evaluator.detect_degradation():
        print("WARNING: Model performance degraded!")

schedule.every().day.at("02:00").do(daily_evaluation)
```

---

## Troubleshooting Guide

### 9.1 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Low retrieval quality** | NDCG@10 < 0.6 | 1. Check MTEB score of model<br>2. Evaluate on domain data (not just MTEB)<br>3. Consider fine-tuning<br>4. Try hybrid search (BM25+dense) |
| **High latency** | Embedding >100ms | 1. Use smaller model variant<br>2. Batch process<br>3. Quantize (INT8 on CPU)<br>4. Add caching |
| **GPU memory error** | CUDA OOM | 1. Reduce batch size<br>2. Use smaller model (BGE-small)<br>3. Use multi-GPU with data parallelism<br>4. Use int8 quantization |
| **Cache misses increasing** | Hit rate <5% | 1. Increase TTL<br>2. Pre-populate cache<br>3. Check cache key generation<br>4. Implement query deduplication |
| **Inconsistent embeddings** | Different results same input | 1. Check random seed<br>2. Verify normalization<br>3. Check batch size effects<br>4. Validate model versioning |
| **Vector DB lag** | Search latency >500ms | 1. Reduce top_k if possible<br>2. Tune HNSW ef parameter<br>3. Shard database<br>4. Add read replicas |

### 9.2 Debug Checklist

```python
def debug_embedding_pipeline():
    """Comprehensive debugging checklist."""
    
    print("=== EMBEDDING PIPELINE DEBUG ===\n")
    
    # 1. Model loading
    print("1. Model Loading")
    try:
        model = SentenceTransformer("BAAI/bge-m3")
        print(f"   ✓ Model loaded: {model}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return
    
    # 2. Single embedding
    print("\n2. Single Embedding")
    try:
        embedding = model.encode("Test")
        print(f"   ✓ Embedding shape: {np.array(embedding).shape}")
        print(f"   ✓ Norm: {np.linalg.norm(embedding):.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 3. Batch embedding
    print("\n3. Batch Embedding")
    try:
        embeddings = model.encode(["Test1", "Test2", "Test3"], batch_size=2)
        print(f"   ✓ Batch shape: {np.array(embeddings).shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 4. Consistency
    print("\n4. Consistency Check")
    emb1 = model.encode("Test")
    emb2 = model.encode("Test")
    similarity = np.dot(emb1, emb2)
    print(f"   {'✓' if similarity > 0.999 else '✗'} Same input similarity: {similarity:.4f}")
    
    # 5. Vector DB connectivity
    print("\n5. Vector Database")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        info = client.get_collections()
        print(f"   ✓ Connected. Collections: {len(info.collections)}")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
    
    # 6. Cache (if configured)
    print("\n6. Caching")
    try:
        import redis
        r = redis.Redis()
        r.ping()
        print(f"   ✓ Redis connected")
    except Exception as e:
        print(f"   ✗ Redis not available: {e}")
    
    print("\n=== DEBUG COMPLETE ===")

# Run diagnostic
debug_embedding_pipeline()
```

---

**End of Implementation Guide**

*Companion to: EMBEDDING_MODELS_COMPREHENSIVE_RESEARCH.md*
