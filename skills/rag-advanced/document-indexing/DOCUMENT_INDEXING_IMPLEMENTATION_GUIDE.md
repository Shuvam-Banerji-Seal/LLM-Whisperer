# Document Indexing & Retrieval: Implementation Guide & Code Reference

**Date:** April 2026  
**Focus:** Practical implementation of chunking, indexing, and retrieval strategies

---

## Quick Reference: When to Use Each Strategy

```
Use Case → Chunking Strategy → Implementation

1. Blog posts, articles → Recursive splitter → LangChain RecursiveCharacterTextSplitter
2. PDFs, reports → Page-level → Unstructured.io partition_pdf
3. Multi-topic documents → Semantic chunking → LangChain SemanticChunker
4. Long manuals → Hierarchical → LlamaIndex HierarchicalNodeParser
5. Code repositories → AST-aware → Custom ast module or tree-sitter
6. Cross-referential docs → Late chunking → Jina API with late_chunking=True
7. Mixed media → Element-based → Docling HierarchicalChunker
```

---

## Section 1: Chunking Implementation Examples

### 1.1 Recursive Character Splitting (Python)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from pathlib import Path

# Load document
doc_path = Path("documents/article.txt")
text = doc_path.read_text()

# Configure splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,           # Target chunk size in characters
    chunk_overlap=50,         # Overlap to preserve context
    length_function=len,      # Use character length (can use token length)
    separators=[
        "\n\n",              # Paragraph breaks (highest priority)
        "\n",                # Line breaks
        ". ",                # Sentence boundaries
        " ",                 # Word boundaries
        ""                   # Character level (fallback)
    ]
)

# Split document
chunks = splitter.split_text(text)

# Save results
output = []
for i, chunk in enumerate(chunks):
    output.append({
        "id": f"chunk_{i:04d}",
        "text": chunk,
        "length": len(chunk),
        "order": i
    })

Path("output/chunks.json").write_text(json.dumps(output, indent=2))
print(f"Created {len(chunks)} chunks")
```

**For Token-Based Splitting:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Load tokenizer matching your embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def count_tokens(text):
    """Count tokens using transformer tokenizer"""
    return len(tokenizer.encode(text, add_special_tokens=False))

# Token-aware splitter
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=512,          # 512 tokens
    chunk_overlap=50         # 50 token overlap
)

chunks = splitter.split_text(text)
```

---

### 1.2 Sentence-Based Chunking (Python)

```python
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

nltk.download('punkt', quiet=True)

def chunk_by_sentences(text, max_tokens=1024, overlap_sentences=1):
    """
    Split text by sentences, respecting token budget.
    Sentences are never split; groups are combined up to max_tokens.
    """
    
    # Split into sentences
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    
    if len(sentences) <= 1:
        return [text]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        
        # If single sentence exceeds limit, it becomes its own chunk
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # Overlap: keep last N sentences
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_tokens = sum(len(tokenizer.encode(s)) for s in current_chunk)
            chunks.append(sentence)
            continue
        
        # If adding this sentence would exceed limit, start new chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Overlap: keep last N sentences
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_tokens = sum(len(tokenizer.encode(s)) for s in current_chunk)
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Don't forget last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Usage
text = Path("documents/research_paper.txt").read_text()
chunks = chunk_by_sentences(text, max_tokens=512, overlap_sentences=1)
print(f"Created {len(chunks)} sentence-based chunks")
```

---

### 1.3 Semantic Chunking (Python)

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Configure semantic chunker
chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # Options: percentile, standard_deviation, interquartile
    breakpoint_threshold_amount=95           # 95th percentile similarity drop = new chunk
)

# Chunk document
text = Path("documents/article.md").read_text()
chunks = chunker.split_text(text)

# Cost estimate: Every sentence gets embedded
# 5000-word doc = 200+ sentence embeddings
# OpenAI: ~$0.0002-0.0005 per document
print(f"Created {len(chunks)} semantic chunks")
print(f"Estimated cost: ${len(chunks) * 0.00005:.4f}")
```

---

### 1.4 Hierarchical Chunking (Python)

```python
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore import DocumentStore

# Create document
doc = Document(text=Path("documents/manual.txt").read_text())

# Configure hierarchical parser
parser = HierarchicalNodeParser.from_defaults(
    chunk_size=2048,      # Parent chunk size
    chunk_overlap=512,
    chunk_sizes=[2048, 512, 128]  # Coarse to fine granularity
)

# Parse into hierarchy
nodes = parser.get_nodes_from_documents([doc])

# Nodes are organized: parent → children relationship
# For storage:
docstore = SimpleDocumentStore()
for node in nodes:
    docstore.add_documents([node])

print(f"Created {len(nodes)} hierarchical nodes")
print(f"Node types: {set(type(n).__name__ for n in nodes)}")
```

---

### 1.5 Page-Level Chunking from PDFs (Python)

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# Parse PDF with high-res strategy (better for complex layouts)
elements = partition_pdf(
    filename="documents/report.pdf",
    strategy="hi_res",  # Options: hi_res, fast, auto
    extract_images_in_pdf=True,  # Extract embedded images
)

# Chunk by title, respecting page boundaries
chunks = chunk_by_title(
    elements,
    multipage_sections=False,  # Keep chunks within page boundaries
    combine_text_under_n_chars=200,  # Minimum chunk size
    max_characters=2000  # Maximum chunk size
)

# Attach metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"page_chunk_{i}"
    chunk.metadata["source"] = "report.pdf"

print(f"Created {len(chunks)} page-level chunks from PDF")
```

---

### 1.6 Code-Aware Chunking (Python + AST)

```python
import ast
from pathlib import Path

def chunk_python_code(filepath, include_docstrings=True):
    """
    Chunk Python code by functions and classes.
    """
    code = Path(filepath).read_text()
    tree = ast.parse(code)
    
    chunks = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Get source code for this node
            start_line = node.lineno - 1
            end_line = node.end_lineno
            
            source_lines = code.split('\n')[start_line:end_line]
            source = '\n'.join(source_lines)
            
            # Extract docstring if present
            docstring = ast.get_docstring(node)
            
            chunk_type = 'class' if isinstance(node, ast.ClassDef) else 'function'
            
            chunks.append({
                "id": f"{chunk_type}_{node.name}",
                "type": chunk_type,
                "name": node.name,
                "source": source,
                "docstring": docstring,
                "start_line": start_line + 1,
                "end_line": end_line,
                "length": len(source)
            })
    
    return chunks

# Usage
repo_chunks = chunk_python_code("path/to/script.py")
for chunk in repo_chunks:
    print(f"{chunk['type'].upper()}: {chunk['name']} ({chunk['length']} chars)")
```

---

## Section 2: Vector Index Implementation

### 2.1 FAISS Indexing (Python)

```python
import faiss
import numpy as np
from typing import List, Tuple

class FAISSIndex:
    """Simple FAISS index wrapper."""
    
    def __init__(self, dimension=1536, metric='l2'):
        """
        metric: 'l2' for Euclidean, 'ip' for inner product (cosine if normalized)
        """
        if metric == 'ip':
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.metric = metric
        self.chunk_store = {}  # Map index ID to chunk text
        self.metadata_store = {}
    
    def add_chunks(self, chunks: List[str], embeddings: np.ndarray, 
                   metadata: List[dict] = None):
        """
        Add chunks with their embeddings.
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize for inner product (cosine similarity)
        if self.metric == 'ip':
            faiss.normalize_L2(embeddings)
        
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Store chunk text and metadata
        for i, chunk in enumerate(chunks):
            self.chunk_store[start_idx + i] = chunk
            if metadata:
                self.metadata_store[start_idx + i] = metadata[i]
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Search and return (chunk, similarity_score) pairs."""
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        if self.metric == 'ip':
            faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # Valid result
                chunk = self.chunk_store[idx]
                results.append((chunk, float(score)))
        
        return results

# Usage
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Create index
index = FAISSIndex(dimension=384, metric='ip')

# Embed and add chunks
chunks = ["chunk1 text", "chunk2 text", "chunk3 text"]
embeddings = model.encode(chunks)
index.add_chunks(chunks, embeddings)

# Search
query = "search query"
query_embedding = model.encode(query)
results = index.search(query_embedding, k=3)

for chunk, score in results:
    print(f"Score: {score:.4f} | {chunk[:100]}...")
```

---

### 2.2 HNSW Indexing with Hnswlib (Python)

```python
import hnswlib
import numpy as np
from pathlib import Path
import json

class HNSWIndex:
    """Production-ready HNSW index."""
    
    def __init__(self, dimension=1536, max_elements=100000, 
                 m=16, ef_construction=200):
        """
        dimension: Embedding dimension
        m: Number of connections (trade-off: memory vs recall)
        ef_construction: Beam width during index building
        """
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=m
        )
        
        self.chunk_store = {}
        self.next_id = 0
    
    def add_chunks(self, chunks: list, embeddings: np.ndarray, 
                   metadata: dict = None):
        """Add chunks to index."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Generate IDs
        ids = np.arange(self.next_id, self.next_id + len(chunks))
        
        # Add to index
        self.index.add_items(embeddings, ids)
        
        # Store metadata
        for idx, chunk in zip(ids, chunks):
            self.chunk_store[int(idx)] = {
                "text": chunk,
                "metadata": metadata[int(idx)] if metadata else {}
            }
        
        self.next_id += len(chunks)
    
    def search(self, query_embedding: np.ndarray, k: int = 5, ef: int = 100):
        """
        Search for nearest neighbors.
        ef: Beam width during search (higher = better recall, slower)
        """
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        self.index.set_ef(ef)  # Set search parameter
        ids, scores = self.index.knn_query(query_embedding, k=k)
        
        results = []
        for idx, score in zip(ids[0], scores[0]):
            if idx in self.chunk_store:
                chunk_data = self.chunk_store[idx]
                results.append({
                    "id": int(idx),
                    "text": chunk_data["text"],
                    "score": float(score),
                    "metadata": chunk_data["metadata"]
                })
        
        return results
    
    def save(self, path: str):
        """Save index to disk."""
        self.index.save_index(f"{path}/hnsw.bin")
        Path(f"{path}/metadata.json").write_text(
            json.dumps(self.chunk_store, indent=2)
        )
    
    def load(self, path: str):
        """Load index from disk."""
        self.index.load_index(f"{path}/hnsw.bin")
        self.chunk_store = json.loads(
            Path(f"{path}/metadata.json").read_text()
        )

# Usage
index = HNSWIndex(dimension=384, max_elements=10000, m=16)

# Build index
chunks = ["text1", "text2", "text3"]
embeddings = model.encode(chunks)
index.add_chunks(chunks, embeddings)

# Search with different beam widths for speed/quality trade-off
results = index.search(query_embedding, k=5, ef=50)  # Fast
results = index.search(query_embedding, k=5, ef=200)  # Better recall

# Save/load
index.save("./my_index")
# Later...
index.load("./my_index")
```

---

### 2.3 Pinecone Vector Database (Python)

```python
from pinecone import Pinecone, ServerlessSpec
import json

# Initialize Pinecone
pc = Pinecone(api_key="your-api-key")

# Create index (serverless, fast startup)
index_name = "my-rag-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

index = pc.Index(index_name)

# Upsert vectors with metadata
vectors_to_upsert = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    vectors_to_upsert.append((
        f"chunk_{i}",  # ID
        embedding,  # Vector
        {  # Metadata
            "text": chunk,
            "source": "document.pdf",
            "page": i // 10  # Example: every 10 chunks is a page
        }
    ))

# Batch upsert
batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i+batch_size]
    index.upsert(vectors=batch)

print(f"Upserted {len(vectors_to_upsert)} vectors")

# Query
query_embedding = model.encode(query)
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

for match in results['matches']:
    print(f"Score: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text'][:100]}...")
```

---

## Section 3: Query Optimization & Retrieval

### 3.1 Hybrid Search (BM25 + Dense Vectors)

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """Combines BM25 (sparse) and dense vector search."""
    
    def __init__(self, chunks: list, embeddings: np.ndarray, 
                 embedding_model):
        """Initialize with chunks and pre-computed embeddings."""
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = embedding_model
        
        # Build BM25 index
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def retrieve_hybrid(self, query: str, k: int = 10, alpha: float = 0.5):
        """
        Hybrid retrieval combining BM25 and dense vectors.
        alpha: 0 = pure BM25, 0.5 = balanced, 1 = pure dense
        """
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        
        # Dense vector search
        query_embedding = self.model.encode(query)
        dense_scores = np.dot(self.embeddings, query_embedding)
        
        # Normalize scores to [0, 1]
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-10)
        
        # Combine with Reciprocal Rank Fusion
        combined_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
        
        # Get top-k
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "bm25_score": float(bm25_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "combined_score": float(combined_scores[idx])
            })
        
        return results

# Usage
retriever = HybridRetriever(chunks, embeddings, model)
results = retriever.retrieve_hybrid("search query", k=5, alpha=0.5)
```

---

### 3.2 Query Expansion & Decomposition

```python
import re
from typing import List

class QueryOptimizer:
    """Query expansion and decomposition for better retrieval."""
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """
        Expand query with synonyms and variations.
        Simple approach: token-based expansion (production would use
        a thesaurus or LLM).
        """
        synonyms = {
            "car": ["automobile", "vehicle", "car"],
            "neural": ["neural", "deep learning", "AI"],
            "retrieval": ["retrieval", "search", "information retrieval"]
        }
        
        expanded = query
        for token, syn_list in synonyms.items():
            if token in query.lower():
                expanded = query + " " + " OR ".join(syn_list)
        
        return [expanded]
    
    @staticmethod
    def decompose_query(query: str) -> List[str]:
        """
        Decompose complex query into sub-queries.
        Example: "What are side effects of drug X in patients over 65?"
        → ["side effects drug X", "drug X elderly patients", "dosage over 65"]
        """
        # This is simplified; production would use LLM or NLP pipeline
        
        if "and" in query.lower():
            # Split on 'and'
            sub_queries = query.lower().split(" and ")
            return [q.strip() for q in sub_queries]
        
        if "?" in query:
            # Extract key phrases
            parts = re.split(r'[?;:]', query)
            return [p.strip() for p in parts if p.strip()]
        
        return [query]

# Usage
optimizer = QueryOptimizer()

query = "What are side effects of drug X in elderly patients?"
expanded = optimizer.expand_query(query)
decomposed = optimizer.decompose_query(query)

print(f"Expanded: {expanded}")
print(f"Decomposed: {decomposed}")
```

---

### 3.3 Reranking with Cross-Encoder

```python
from sentence_transformers import CrossEncoder

class RerankRetriever:
    """Rerank dense retrieval results with cross-encoder."""
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """Initialize cross-encoder for reranking."""
        self.reranker = CrossEncoder(model_name)
    
    def rerank(self, query: str, candidates: List[str], k: int = 5):
        """
        Rerank dense retrieval candidates using cross-encoder.
        """
        # Score each candidate with cross-encoder
        pairs = [[query, chunk] for chunk in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k
        return [
            {
                "text": chunk,
                "rerank_score": float(score)
            }
            for chunk, score in ranked[:k]
        ]

# Usage
dense_retriever = HybridRetriever(chunks, embeddings, model)
reranker = RerankRetriever()

# Step 1: Dense retrieval (get 100 candidates)
dense_results = dense_retriever.retrieve_hybrid(query, k=100)
dense_chunks = [r["chunk"] for r in dense_results]

# Step 2: Rerank with cross-encoder (get top 5)
reranked = reranker.rerank(query, dense_chunks, k=5)
```

---

## Section 4: Evaluation & Benchmarking

### 4.1 Retrieval Metrics Implementation

```python
import numpy as np
from typing import List, Set

class RetrievalEvaluator:
    """Compute standard retrieval metrics."""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Precision@K: fraction of top-k that are relevant."""
        top_k = retrieved[:k]
        return len(set(top_k) & relevant) / k
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Recall@K: fraction of relevant items found in top-k."""
        top_k = retrieved[:k]
        if not relevant:
            return 0
        return len(set(top_k) & relevant) / len(relevant)
    
    @staticmethod
    def mrr_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Mean Reciprocal Rank@K: position of first relevant result."""
        top_k = retrieved[:k]
        for rank, item in enumerate(top_k, 1):
            if item in relevant:
                return 1 / rank
        return 0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant_scores: dict, k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        relevant_scores: dict of {item: relevance_score}
        """
        top_k = retrieved[:k]
        
        # Compute DCG
        dcg = 0
        for rank, item in enumerate(top_k, 1):
            relevance = relevant_scores.get(item, 0)
            dcg += relevance / np.log2(rank + 1)
        
        # Compute ideal DCG (perfect ranking)
        sorted_scores = sorted(relevant_scores.values(), reverse=True)
        idcg = 0
        for rank, relevance in enumerate(sorted_scores[:k], 1):
            idcg += relevance / np.log2(rank + 1)
        
        if idcg == 0:
            return 0
        
        return dcg / idcg

# Usage
evaluator = RetrievalEvaluator()

# Mock retrieval results and ground truth
retrieved = ["doc1", "doc3", "doc2", "doc5", "doc4"]
relevant = {"doc1", "doc2", "doc3"}  # Ground truth relevant docs
relevant_scores = {"doc1": 2, "doc2": 1, "doc3": 2, "doc5": 1}  # Graded relevance

# Compute metrics
p5 = evaluator.precision_at_k(retrieved, relevant, 5)
r5 = evaluator.recall_at_k(retrieved, relevant, 5)
mrr = evaluator.mrr_at_k(retrieved, relevant, 5)
ndcg = evaluator.ndcg_at_k(retrieved, relevant_scores, 5)

print(f"Precision@5: {p5:.3f}")
print(f"Recall@5: {r5:.3f}")
print(f"MRR@5: {mrr:.3f}")
print(f"NDCG@5: {ndcg:.3f}")
```

---

### 4.2 Benchmark Script

```python
import json
from pathlib import Path
from datetime import datetime

class RAGBenchmark:
    """Benchmark RAG system performance."""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.results = {
            "system": system_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
    
    def run_benchmark(self, test_queries: dict, retriever, evaluator):
        """
        Run benchmark on test query set.
        test_queries: {query: {relevant_docs, ground_truth}}
        """
        all_metrics = {
            "precision@5": [],
            "recall@5": [],
            "ndcg@10": [],
            "mrr": []
        }
        
        for query, ground_truth in test_queries.items():
            # Retrieve
            results = retriever.search(query, k=10)
            retrieved = [r["text"] for r in results]
            
            # Evaluate
            p5 = evaluator.precision_at_k(retrieved, 
                                         ground_truth["relevant"], 5)
            r5 = evaluator.recall_at_k(retrieved, 
                                      ground_truth["relevant"], 5)
            ndcg = evaluator.ndcg_at_k(retrieved, 
                                      ground_truth["scores"], 10)
            mrr = evaluator.mrr_at_k(retrieved, 
                                    ground_truth["relevant"], 10)
            
            all_metrics["precision@5"].append(p5)
            all_metrics["recall@5"].append(r5)
            all_metrics["ndcg@10"].append(ndcg)
            all_metrics["mrr"].append(mrr)
        
        # Average metrics
        self.results["metrics"] = {
            k: {
                "mean": np.mean(v),
                "std": np.std(v),
                "min": np.min(v),
                "max": np.max(v)
            }
            for k, v in all_metrics.items()
        }
        
        return self.results
    
    def save_results(self, path: str):
        """Save benchmark results to file."""
        Path(path).write_text(json.dumps(self.results, indent=2))

# Usage
test_queries = {
    "query 1": {
        "relevant": {"doc1", "doc2", "doc3"},
        "scores": {"doc1": 2, "doc2": 2, "doc3": 1}
    },
    "query 2": {
        "relevant": {"doc4", "doc5"},
        "scores": {"doc4": 2, "doc5": 1}
    }
}

benchmark = RAGBenchmark("my-rag-system")
results = benchmark.run_benchmark(test_queries, retriever, evaluator)
benchmark.save_results("benchmark_results.json")

print(json.dumps(results, indent=2))
```

---

## Section 5: Production Deployment Checklist

### Deployment Checklist

```
CHUNKING:
[ ] Chunk size tuned for embedding model and use case
[ ] Overlap percentage tested (default 10-20%)
[ ] Separators customized for document types
[ ] Metadata attached (source, section, date)
[ ] Chunks deduplicated and validated

INDEXING:
[ ] Index type chosen (HNSW for <10M, IVF+PQ for scale)
[ ] Parameters tuned: M=16, ef_construction=200
[ ] Index built and tested with sample queries
[ ] Index monitored for corruption/consistency
[ ] Backup strategy implemented

RETRIEVAL:
[ ] Hybrid search (BM25 + dense) enabled
[ ] Query optimization pipeline (expansion, decomposition)
[ ] Reranking (optional, cost vs accuracy)
[ ] Fallback strategies for edge cases

EVALUATION:
[ ] Test set of 50-100 representative queries
[ ] Ground truth relevance judgments collected
[ ] NDCG@10, Recall@K, Precision@K computed
[ ] Latency benchmarks measured (p50, p95, p99)
[ ] Cost tracking (embeddings, storage, compute)

MONITORING:
[ ] Recall tracked (catch degradation)
[ ] Latency monitored (detect slowdowns)
[ ] Error rate tracked (timeouts, failures)
[ ] Query volume logged
[ ] Cost dashboard set up

OPTIMIZATION:
[ ] Periodic re-evaluation (monthly)
[ ] Index refreshed on data changes
[ ] Parameters tuned based on metrics
[ ] New chunking strategies tested
[ ] Cost optimization analyzed
```

---

## Section 6: Common Pitfalls & Solutions

### Pitfall 1: Boundary Loss Without Overlap

**Problem:** Important facts split across chunk boundaries

**Solution:** Add overlap
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50  # 10% overlap reduces boundary loss
)
```

### Pitfall 2: Chunk Size Too Large

**Problem:** LLM context window used inefficiently

**Solution:** Tune to query patterns
```python
# Factoid QA: smaller chunks
chunk_size = 256

# Analytical reasoning: larger chunks
chunk_size = 1024
```

### Pitfall 3: Vector-Only Retrieval Fails on Exact Matches

**Problem:** Acronyms, IDs, proper nouns missed

**Solution:** Hybrid retrieval
```python
retriever = HybridRetriever(chunks, embeddings, model)
results = retriever.retrieve_hybrid(query, alpha=0.5)  # 50/50 BM25 + dense
```

### Pitfall 4: Latency Grows with Index Size

**Problem:** Linear scaling with FLAT or poorly tuned HNSW

**Solution:** Right index type
```python
if total_vectors < 10_000_000:
    index = HNSW(m=16, ef_search=100)
else:
    index = IVF(nlist=1000) + PQ()  # For billion-scale
```

### Pitfall 5: Recall Drops Without Monitoring

**Problem:** Silent degradation in retrieval quality

**Solution:** Track metrics continuously
```python
# Monthly evaluation
results = benchmark.run_benchmark(test_queries, retriever, evaluator)
if results["metrics"]["ndcg@10"]["mean"] < 0.70:
    alert("Recall degradation detected")
```

---

## Conclusion

This implementation guide provides production-ready code patterns for:
- Chunking strategies (recursive, semantic, hierarchical, page-level)
- Index implementations (FAISS, HNSW, Pinecone)
- Hybrid retrieval (BM25 + dense)
- Query optimization
- Evaluation and benchmarking

For most projects, start with recursive chunking + HNSW indexing + hybrid search. Optimize based on measured metrics.

---

**Last Updated:** April 6, 2026
