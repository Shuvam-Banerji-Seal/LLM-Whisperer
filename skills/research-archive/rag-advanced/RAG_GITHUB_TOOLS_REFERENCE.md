# RAG & Synthetic Data: GitHub Repositories & Tools

**Comprehensive Resource Guide 2026**

---

## 1. Core RAG Frameworks & Libraries

### 1.1 High-Level RAG Frameworks

| Framework | GitHub | Stars | Best For | Key Strength |
|-----------|--------|-------|----------|--------------|
| **LangChain** | github.com/langchain-ai/langchain | 90K+ | General RAG, LLM chains | Ecosystem & integrations |
| **LlamaIndex** | github.com/run-llama/llama_index | 65K+ | Document indexing | Structure-aware processing |
| **Haystack** | github.com/deepset-ai/haystack | 18K+ | NLP pipelines | Pipeline composition |
| **RAGstack** | github.com/datastax/ragstack-ai | 2K+ | Production RAG | Cassandra integration |
| **Verba** | github.com/weaviate/Verba | 8K+ | Semantic search | Vector DB native |

### 1.2 Retrieval & Indexing

| Library | GitHub | Purpose | Tech Stack |
|---------|--------|---------|-----------|
| **FAISS** | github.com/facebookresearch/faiss | Dense vector search | C++, HNSW/IVF implementation |
| **HNSWLIB** | github.com/nmslib/hnswlib | Approximate nearest neighbor | C++ library |
| **BM25-Okapi** | github.com/dorianbrown/rank_bm25 | Sparse retrieval | Pure Python |
| **Pyserini** | github.com/castorini/pyserini | Information retrieval | Lucene Java binding + Python |
| **Weaviate** | github.com/weaviate/weaviate | Vector database | Go backend, Python client |
| **Qdrant** | github.com/qdrant/qdrant | Vector database | Rust backend, Python SDK |
| **Milvus** | github.com/milvus-io/milvus | Vector database | C++ with Python bindings |
| **Pinecone Python** | github.com/pinecone-io/pinecone-python | Vector DB client | Python wrapper |
| **Vespa** | github.com/vespa-engine/vespa | Search engine | Java, supports dense+sparse |

### 1.3 Reranking & Ranking

| Library | GitHub | Purpose | Model Type |
|---------|--------|---------|-----------|
| **Sentence Transformers** | github.com/UKPLab/sentence-transformers | Embedding & cross-encoder | PyTorch models |
| **Cross-Encoder** | huggingface.co/cross-encoder | Pre-trained rerankers | Hugging Face models |
| **LLMRanker** | github.com/microsoft/LMOps | LLM-based ranking | Proprietary LLM ranking |
| **ColBERT** | github.com/stanford-future/ColBERT | Late interaction ranking | Token-level similarity |

---

## 2. Vector Databases (Self-Hosted & Managed)

### 2.1 Self-Hosted Vector Databases

```
QDRANT
├─ Language: Rust
├─ Storage: Local/S3/persistent
├─ Features: HNSW, sparse vectors, payload filtering
├─ Python: pip install qdrant-client
├─ Docker: docker run -p 6333:6333 qdrant/qdrant
├─ Deployment: Single node or cluster
└─ Cost: Self-hosted (compute only)

WEAVIATE
├─ Language: Go
├─ Storage: Local/S3/GCS
├─ Features: HNSW, BM25F, GraphQL API
├─ Python: pip install weaviate-client
├─ Docker: docker run -p 8080:8080 semitechnologies/weaviate
├─ Deployment: Single node or Kubernetes
└─ Cost: Self-hosted or managed cloud

MILVUS
├─ Language: C++ (Go coordinator)
├─ Storage: MinIO/etcd/Pulsar backend
├─ Features: HNSW, IVF, GPU acceleration
├─ Python: pip install pymilvus
├─ Docker: docker run -p 19530:19530 milvusdb/milvus
├─ Deployment: Distributed cluster required
└─ Cost: Self-hosted cloud infrastructure

REDIS
├─ Language: C with extensions
├─ Storage: In-memory (optional RDB/AOF)
├─ Features: Vector search (Redis 8.0+), hash operations
├─ Python: pip install redis
├─ Docker: docker run -p 6379:6379 redis/redis
├─ Deployment: Single node or cluster
└─ Cost: Self-hosted (memory only)
```

### 2.2 Managed/SaaS Vector Databases

```
PINECONE
├─ Pros: Fully managed, high availability, pay-per-use
├─ Cons: Proprietary, lock-in, expensive at scale
├─ Pricing: ~$0.10-0.50 per million vectors
└─ Integration: Excellent LangChain/LlamaIndex support

SUPABASE VECTOR
├─ Pros: PostgreSQL vector (pgvector), affordable, integrated auth
├─ Cons: Requires PostgreSQL knowledge, not pure vector DB
└─ Pricing: $5-100/month for compute

ATLAS (MongoDB)
├─ Pros: Vector search in MongoDB, integrated with Atlas
├─ Cons: JSON-focused, not optimized for pure vector search
└─ Pricing: MongoDB Atlas pricing + vector features

ANTHROPIC VECTOR DB
├─ Status: Announced but not yet released (2026)
├─ Expected: Tight integration with Claude APIs
└─ Pricing: TBD
```

---

## 3. Search Engines & Full-Text Indexing

### 3.1 Production Search Engines

```
ELASTICSEARCH
├─ Version: 8.9+ (with hybrid search support)
├─ Hybrid: RRF fusion built-in
├─ Setup: docker run -p 9200:9200 docker.elastic.co/elasticsearch/elasticsearch:8.9.0
├─ Python Client: pip install elasticsearch
├─ Storage: Shards for scale
├─ Cost: Self-hosted or Elastic Cloud ($15+/month)
└─ Competitors: OpenSearch (AWS fork)

OPENSEARCH
├─ Open source fork of Elasticsearch
├─ Hybrid: Supports score-based fusion
├─ Setup: docker run -p 9200:9200 opensearchproject/opensearch:latest
├─ Cost: Self-hosted (no vendor lock-in)
└─ Maintained by AWS

SOLR
├─ Legacy but stable
├─ Hybrid: Can be configured with custom handlers
├─ Java-based, enterprise-proven
└─ Cost: Self-hosted (Apache license)

MEILISEARCH
├─ Fast, user-friendly search
├─ Cons: Limited to full-text, no vector search (yet)
├─ Good for: E-commerce product search
└─ Cloud: Meilisearch Cloud ($20+/month)
```

### 3.2 Comparative Setup

```python
# ELASTICSEARCH (Hybrid Search Example)
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# Create index with dense + sparse vectors
es.indices.create(
    index="hybrid_index",
    body={
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True
                },
                "sparse_vector": {
                    "type": "sparse_vector"
                }
            }
        }
    }
)

# Hybrid search with RRF
result = es.search(
    index="hybrid_index",
    body={
        "query": {
            "rrf": {
                "retrievers": [
                    {
                        "standard": {
                            "query": {
                                "match": {
                                    "text": "database optimization"
                                }
                            }
                        }
                    },
                    {
                        "knn": {
                            "field": "dense_vector",
                            "query_vector": [0.1, 0.2, ...],
                            "k": 20
                        }
                    }
                ]
            }
        }
    }
)

# WEAVIATE (Hybrid Search Example)
import weaviate

client = weaviate.Client("http://localhost:8080")

# Query
result = client.query.get(
    "Document",
    ["text", "score"]
).with_hybrid(
    query="database optimization",
    alpha=0.5
).with_limit(10).do()

# QDRANT (Native Hybrid with SPLADE)
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, SparseVectorParams

client = QdrantClient("localhost", port=6333)

# Hybrid query
results = client.query_points(
    collection_name="documents",
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=20),
        Prefetch(query=sparse_vector, using="sparse", limit=20)
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10
)
```

---

## 4. LLM Inference & Generation

### 4.1 Self-Hosted LLM Serving

```
VLLM (vLLM)
├─ GitHub: github.com/vllm-project/vllm
├─ Speed: 10-40x faster than naive inference
├─ Models: LLaMA, Mistral, Qwen, Yi, etc.
├─ Setup: pip install vllm && vllm serve meta-llama/Llama-2-7b-hf
├─ API: OpenAI-compatible REST
├─ GPU: Single or multi-GPU support
└─ Scaling: vLLM cluster for load balancing

TEXT GENERATION INFERENCE (TGI)
├─ GitHub: github.com/huggingface/text-generation-inference
├─ Speed: Competitive with vLLM
├─ Quantization: GPTQ, AWQ support
├─ Setup: docker run ghcr.io/huggingface/text-generation-inference
├─ Models: Hugging Face models
└─ Enterprise: Offered as service by Hugging Face

OLLAMA
├─ GitHub: github.com/ollama/ollama
├─ Simplicity: Single binary, easy to use
├─ Local: Download models once, run without internet
├─ Models: 100+ available (LLaMA, Mistral, etc.)
├─ Setup: ollama pull llama2 && ollama serve
├─ Use Case: Development, edge deployment
└─ Limitation: Not production-grade for scale

LLAMA.CPP
├─ GitHub: github.com/ggerganov/llama.cpp
├─ Speed: Optimized for CPUs
├─ Format: GGUF quantized models
├─ Perfect for: Laptops, servers without GPU
└─ Integration: Python bindings available
```

### 4.2 Cloud LLM APIs

```
OPENAI
├─ Models: GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5
├─ API: REST + WebSockets
├─ Cost: $10-30 per 1M input tokens, $40-120 per 1M output
├─ Limits: Rate limits, regional availability
└─ Integration: Excellent with LangChain

ANTHROPIC (Claude)
├─ Models: Claude 3 Opus, Sonnet, Haiku
├─ Context: Up to 200K tokens (Opus)
├─ Cost: $3-15 per 1M input, $15-75 per 1M output
├─ Strengths: Long context, instruction-following
└─ Integration: Good with LangChain

TOGETHER AI
├─ Models: Open source models (LLaMA, Mistral, etc.)
├─ Cost: $1-2 per 1M tokens (much cheaper)
├─ Speed: Optimized inference
└─ Good for: Cost-sensitive applications

ANYSCALE (Ray Serve)
├─ Models: Open source focus
├─ Cost: Competitive
└─ Good for: Distributed inference

REPLICATE
├─ Models: Open source via Replicate API
├─ Cost: ~$0.0005-0.001 per input, ~$0.0015-0.003 per output
├─ Setup: pip install replicate
└─ Use Case: Serverless generation
```

---

## 5. Synthetic Data Generation Tools

### 5.1 Data Generation Libraries

```
SYNTHETIC DATA VAULT (SDV)
├─ GitHub: github.com/sdv-dev/SDV
├─ Purpose: Synthetic data generation with privacy preservation
├─ Methods: TVAE, GaussianCopula, etc.
├─ Python: pip install sdv
└─ Use: Tabular + time series data

TEXTSYNTH
├─ GitHub: github.com/kvyatkovskys/TextSynth
├─ Purpose: Generate synthetic text data
├─ Methods: LLM-based generation, paraphrasing
└─ Use: NLG tasks

OUTLINES
├─ GitHub: github.com/outlines-ai/outlines
├─ Purpose: Structured generation from LLMs
├─ Features: JSON mode, regex constraints
├─ Use: Structured synthetic data (JSON, XML)
```

### 5.2 Synthetic Data Quality Tools

```
GREAT EXPECTATIONS
├─ GitHub: github.com/great-expectations/great_expectations
├─ Purpose: Data validation and quality checks
├─ Features: Expectation suites, checkpoints
└─ Integration: DataOps best practice

EVIDENTLY AI
├─ GitHub: github.com/evidentlyai/evidently
├─ Purpose: ML data/model quality monitoring
├─ Features: Data drift detection, report generation
└─ Use: Monitor synthetic data quality over time

PANDERA
├─ GitHub: github.com/unionai-oss/pandera
├─ Purpose: Statistical validation for pandas
├─ Syntax: Easy schema definition
└─ Use: Quick validation of generated data
```

---

## 6. Evaluation & Benchmarking

### 6.1 RAG Evaluation Frameworks

```
RAGAS (RAG Assessment)
├─ GitHub: github.com/explodinggradients/ragas
├─ Metrics: Faithfulness, relevance, context adherence
├─ LLM-based: Uses LLM to evaluate quality
├─ Setup: pip install ragas
├─ Integration: Works with any RAG pipeline
└─ Recommended: Yes, widely adopted

TRULENS
├─ GitHub: github.com/truera/trulens
├─ Company: TruEra
├─ Purpose: Evaluate RAG/LLM quality at scale
├─ Metrics: Groundedness, helpfulness, relevance
├─ Cost: SaaS platform with free tier
└─ Strength: Good for production monitoring

MLFLOW
├─ GitHub: github.com/mlflow/mlflow
├─ Purpose: General ML experiment tracking
├─ RAG: Can track RAG metrics and logs
├─ Setup: pip install mlflow
└─ Integration: Works with LangChain, LlamaIndex

LANGSMITH
├─ By: LangChain team
├─ Purpose: LLM/RAG monitoring and debugging
├─ Features: Traces, metrics, comparison
└─ Cost: Free tier available
```

### 6.2 Benchmark Datasets

```
HOTPOTQA
├─ Purpose: Multi-hop question answering
├─ Size: 113K QA pairs
├─ GitHub: github.com/hotpotqa/hotpot
├─ Benchmark: Standard for multi-hop evaluation

MSMARCO
├─ Purpose: Large-scale ranking/retrieval
├─ Size: 1M+ queries, 100M+ passages
├─ Task: Document ranking, passage retrieval
└─ Source: Microsoft Research

BEIR
├─ Purpose: Information retrieval benchmark
├─ Datasets: 15+ diverse IR datasets
├─ Metrics: NDCG, MAP, MRR
├─ GitHub: github.com/beir-cellar/beir

SCIFACT
├─ Purpose: Fact verification with evidence
├─ Size: 1.4K claims, 200K papers
└─ Task: Retrieve evidence, verify claims
```

---

## 7. Monitoring & Observability

### 7.1 Monitoring Stacks

```
PROMETHEUS + GRAFANA + LOKI
├─ Prometheus: github.com/prometheus/prometheus
├─ Grafana: github.com/grafana/grafana
├─ Loki: github.com/grafana/loki
├─ Stack: Best for metrics + logs + traces
├─ Cost: Self-hosted (free), Grafana Cloud ($10+)
└─ Recommended: Best for production

ELK STACK (Elasticsearch, Logstash, Kibana)
├─ Elasticsearch: github.com/elastic/elasticsearch
├─ Kibana: github.com/elastic/kibana
├─ Purpose: Log aggregation + visualization
├─ Cost: Elastic Cloud ($15+), self-hosted free
└─ Alternative: OpenSearch (AWS fork)

DATADOG
├─ Cost: $0.23-$0.46 per host/month
├─ Features: Metrics, logs, traces, APM
└─ Good for: Small to medium teams

NEW RELIC
├─ Cost: $100-500+/month
├─ Features: Full observability
└─ Good for: Enterprise
```

### 7.2 Tracing for RAG

```
JAEGER (Distributed Tracing)
├─ GitHub: github.com/jaegertracing/jaeger
├─ OpenTelemetry: Standard instrumentation
├─ Traces: Full request path through RAG pipeline
├─ Setup: docker run -p 16686:16686 jaegertracing/all-in-one
└─ Cost: Self-hosted (free)

OPENTELEMETRY
├─ Standard: Universal instrumentation
├─ Languages: Python, Go, Java, JavaScript, etc.
├─ RAG Use: Trace retrieval, reranking, generation stages
└─ Integration: Works with Jaeger, Datadog, etc.
```

---

## 8. Complete Tech Stack Examples

### 8.1 Budget Self-Hosted Stack (~$50-100/month)

```
Compute: AWS EC2 t3.large ($30/month)
├─ Qdrant (self-hosted): Vector storage
├─ PostgreSQL + pgvector: Hybrid search
├─ vLLM: LLM inference (7B-13B model)
├─ Redis: Caching
└─ Prometheus + Grafana: Monitoring

Observability: Self-hosted Prometheus/Loki
├─ Logs: Loki on same instance
├─ Metrics: Prometheus
└─ Visualization: Grafana

Vector Embeddings: Local or Hugging Face Inference
├─ sentence-transformers: All-MiniLM-L6-v2
└─ No external API calls

LLM Generation: Local with vLLM
├─ Mistral-7B or LLaMA-2-7B
├─ GPU not required (CPU inference acceptable)
└─ Cost: Just compute

Storage: EBS (20GB, $2/month)
```

**Cost Breakdown:**
- EC2: $30
- Storage: $2
- Data transfer: $5 (average)
- Total: ~$37-50/month

---

### 8.2 Production Stack (~$500-2000/month)

```
Vector DB: Qdrant Cloud
├─ Cost: $100-500/month depending on scale
├─ Managed: No ops burden
├─ Backup: Included
└─ Scaling: Automatic

Search: Elasticsearch (Elastic Cloud)
├─ Cost: $100+/month
├─ Hybrid: Built-in RRF
└─ Scaling: Managed

LLM: OpenAI API
├─ Cost: $200-1000/month
├─ Model: GPT-4o for complex, GPT-4o-mini for simple
└─ No infrastructure needed

Inference Services: AWS SageMaker or Replicate
├─ Reranking: $100-200/month
├─ Embeddings: ~$50/month
└─ Total: $150-250/month

Infrastructure: AWS RDS (PostgreSQL + pgvector)
├─ Cost: $100+/month
├─ Managed backups, HA
└─ Suitable for metadata/cache

Monitoring: Datadog or New Relic
├─ Cost: $100-500/month
├─ Full observability
└─ Essential for production

**Total: $550-2250/month**

Cost Optimization:
- Hybrid models: Use cheaper LLM for simple queries (GPT-4o-mini)
- Cache aggressively: Reduce API calls by 30-50%
- Batch operations: Reduce reranking calls
- Self-host optional components: Embeddings, reranking on EC2
```

---

### 8.3 Enterprise Stack (~$10K+/month)

```
Vector DB: Qdrant Cloud Enterprise
├─ Cost: $1000+/month
├─ SLA: 99.9%
├─ Support: Dedicated account manager
└─ Features: Custom models, GPU acceleration

Search: Elasticsearch/Opensearch managed
├─ Cost: $500+/month
├─ Multi-region: Available
└─ Security: SOC2, HIPAA compliance

LLM: Hybrid (OpenAI + Self-hosted)
├─ OpenAI: GPT-4o for complex ($500+/month)
├─ Self-hosted: vLLM on Kubernetes ($2000+/month)
└─ Cost: $2500+/month

Kubernetes Infrastructure: EKS/GKE/AKS
├─ Compute: $3000+/month (production nodes)
├─ GPU nodes: Additional $500-2000/month
├─ Managed Kubernetes: $70/month (EKS)
└─ Total: $3570+/month

Monitoring & Security:
├─ Datadog Enterprise: $500+/month
├─ VPC, security groups: Included
├─ Compliance: PCI-DSS, SOC2
└─ Cost: $500+/month

**Total: $10K-15K+/month**

Enterprise Features:
- Multi-region deployment
- 99.95% SLA
- Dedicated support
- Custom model training
- Advanced security & compliance
```

---

## 9. Recommended GitHub Resources

### Essential Reading

1. **facebook/faiss** - Dense vector search Bible
   - Clone: `git clone github.com/facebookresearch/faiss`
   - Learn: HNSW/IVF algorithms
   - Use: For understanding vector indexing

2. **langchain-ai/langchain** - RAG orchestration gold standard
   - Clone: `git clone github.com/langchain-ai/langchain`
   - Examples: 100+ RAG examples
   - Integration: Every major service

3. **huggingface/transformers** - State-of-the-art models
   - Clone: `git clone github.com/huggingface/transformers`
   - Models: 100K+ pre-trained
   - Use: Embeddings, cross-encoders

4. **vllm-project/vllm** - Production LLM serving
   - Clone: `git clone github.com/vllm-project/vllm`
   - Deploy: Optimized inference server
   - Scale: Multi-GPU support

5. **explodinggradients/ragas** - RAG evaluation framework
   - Install: `pip install ragas`
   - Metrics: Faithfulness, relevance, coherence
   - Production: Essential for monitoring

---

## 10. Quick Start Command Reference

```bash
# Vector Database Setup
docker run -p 6333:6333 qdrant/qdrant  # Qdrant
docker run -p 8080:8080 semitechnologies/weaviate  # Weaviate
docker run -p 9200:9200 docker.elastic.co/elasticsearch/elasticsearch:8.9.0  # Elasticsearch

# LLM Serving
vllm serve meta-llama/Llama-2-7b-hf  # vLLM
ollama pull llama2 && ollama serve  # Ollama

# Python Setup
python -m venv venv && source venv/bin/activate
pip install langchain llama-index faiss-cpu sentence-transformers qdrant-client elasticsearch

# Quick RAG Test
python -c "
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
print('RAG ready!')
"

# Monitoring
docker run -p 9090:9090 prom/prometheus  # Prometheus
docker run -p 3000:3000 grafana/grafana  # Grafana
```

---

## 11. Community Resources

### Discord/Communities
- LangChain Discord: discord.gg/langchain
- LLaMA Discord: discord.gg/llamaindex
- Qdrant Community: discord.gg/qdrant
- Together AI Community: discord.gg/togetherai

### Conferences & Workshops
- NeurIPS (December) - Annual AI conference
- ICLR (May) - Learning representations conference
- ACL (June) - Language processing conference
- Webinars: Weekly from LangChain, Hugging Face, etc.

### Publications
- ArXiv: arxiv.org (search "retrieval augmented generation")
- Papers with Code: paperswithcode.com
- Hugging Face Blog: huggingface.co/blog

---

