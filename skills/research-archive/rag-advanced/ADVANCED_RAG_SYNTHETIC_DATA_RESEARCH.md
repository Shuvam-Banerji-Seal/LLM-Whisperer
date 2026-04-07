# Advanced RAG Techniques & Synthetic Data Generation: Comprehensive Research Guide

**Last Updated:** April 2026  
**Research Scope:** 25+ research papers, 20+ repositories, production patterns, code examples, and quality frameworks

---

## Executive Summary

This comprehensive guide covers the latest advances in Retrieval-Augmented Generation (RAG) systems and synthetic data generation for LLMs. The research consolidates emerging patterns from 2024-2026, including multi-hop retrieval optimization, hybrid search implementations, vector database strategies, and production-scale RAG architectures.

**Key Findings:**
- Hybrid search (BM25 + dense) consistently outperforms single-method retrieval by 18-31% NDCG
- Structured decomposition with consensus-based validation reduces error propagation by 7-13% on multi-hop tasks
- SPLADE sparse retrieval outperforms BM25 on most BEIR benchmarks while maintaining interpretability
- Production RAG systems require integrated caching, monitoring, and cost optimization strategies
- Synthetic data generation quality directly correlates with downstream LLM performance

---

## 1. Advanced RAG Patterns & Architectures

### 1.1 Multi-Hop & Iterative Retrieval

#### Recursive Retrieval Types

Research identifies three distinct patterns for recursive/iterative retrieval:

**1. Page-Based Recursive Retrieval**
- Follows explicit cross-references between documents
- Optimal for technical manuals with numbered references
- Example: Manufacturing specs, legal documents with page citations
- Benefit: Deterministic exploration paths
- Limitation: Requires explicit cross-reference structure

**2. Information-Centric Recursive Retrieval**
- Fixes seed node (concept) and expands knowledge graph around it
- Ideal for relationship tracking across documents
- Example: Liability tracking for specific client across multiple legal documents
- Benefit: Creates reusable knowledge base for related queries
- Limitation: Requires manual relationship definition or LLM extraction

**3. Concept-Centric Recursive Retrieval**
- LLM determines relevant nodes based on query context
- Handles multi-hop questions with unknown intermediate steps
- Example: "Which entities are affected by X's actions and how?"
- Benefit: Most flexible, handles open-ended exploration
- Challenge: Risk of exploration drift from original intent

#### Reasoning Tree Guided RAG (RT-RAG)

**Architecture:** Hierarchical decomposition with consensus-based tree selection

**Key Innovation:** Three-stage approach addressing error propagation

```
Stage 1: Question Decomposition
├── Extract Core Query (what is being asked)
├── Identify Known Entities (explicit anchors)
├── Discover Unknown Entities (to be retrieved)
└── Generate multiple candidate trees with consensus selection

Stage 2: Bottom-Up Retrieval
├── Post-order traversal of decomposition tree
├── Rejection sampling for answer consistency
├── Query rewriting for insufficient results
└── Adaptive node reconfiguration when evidence missing

Stage 3: Hierarchical Answer Integration
├── Respect tree structure dependencies
├── Iterative refinement on failures
└── Maintain contextual coherence across hops
```

**Performance Gains (vs. state-of-the-art):**
- MuSiQue: +3.9% F1, +3.0% EM (GPT-4o-mini) / +13.0% F1, +11.5% EM (Qwen-14B)
- 2WikiMQA: +12.5% F1, +11.0% EM (GPT-4o-mini) / +13.2% F1, +14.0% EM (Qwen-14B)
- HotpotQA: +0.7% F1, +1.5% EM (modest but consistent gains)

**Consensus Mechanism:**
```python
# Consensus-based tree selection strategy
def select_optimal_tree(candidate_trees: list[Tree]) -> Tree:
    """
    Statistical validation of tree structure robustness
    Selects most prevalent tree configuration by frequency
    """
    tree_frequencies = {}
    for tree in candidate_trees:
        structure = (tree.depth, tree.node_count, tree.decomposition_type)
        tree_frequencies[structure] = tree_frequencies.get(structure, 0) + 1
    
    # Select tree with most common structure pattern
    best_structure = max(tree_frequencies, key=tree_frequencies.get)
    return [t for t in candidate_trees if (t.depth, t.node_count, t.decomposition_type) == best_structure][0]
```

### 1.2 Hybrid Search Implementations

#### Theory & Benchmarks

**Why Hybrid Search Matters:**
- Dense retrieval (embeddings) fails on exact matches: error codes, APIs, product SKUs
- Sparse retrieval (BM25) misses semantic similarity: "car" ≠ "automobile"
- Hybrid combines both in parallel, fusing results with mathematical fusion strategies

**BEIR Benchmark Results (2024):**
| Dataset | Metric | Dense/Keyword Only | Hybrid | Gain |
|---------|--------|-------------------|--------|------|
| BEIR aggregate (13 datasets) | NDCG | Baseline | +26-31% | 26-31% |
| BEIR aggregate | MRR | 0.410 | 0.486 | +18.5% |
| WANDS furniture | Mean NDCG | 0.695 (KNN) | 0.708 (RRF) | +1.7% |
| BRIGHT Biology | Recall | Baseline | +24% | +24% |
| OpenSearch real-world | MAP | 0.55 | 0.60 | +9% |
| OpenSearch real-world | NDCG | 0.69 | 0.82 | +19% |

**Critical Insight:** Hybrid gains correlate with vocabulary mismatch between queries and documents. High vocabulary mismatch = higher gains (BRIGHT Biology +24%). Low mismatch = minimal gains (WANDS +1.7%).

#### Sparse Retrievers: BM25 vs SPLADE

| Aspect | BM25 | SPLADE |
|--------|------|--------|
| Vocabulary Expansion | None | Yes (learned) |
| Exact Keyword Matching | Excellent | Good |
| Semantic Matching | None | Partial |
| Index Type | Inverted Index | Inverted Index |
| Inference Cost | None | One encoder pass per doc |
| GPU Required | No | For indexing only |
| Best Use Case | SKUs, error codes, legal IDs | Mixed vocab, enterprise docs |
| BEIR Benchmark Winner | ~70% datasets | ~70% datasets |

**SPLADE Advantages:**
- Learned sparse vectors with semantic term expansion
- Handles vocabulary mismatch (how users ask vs. how docs written)
- Compatible with standard inverted index infrastructure
- Outperforms BM25 on most BEIR datasets while maintaining interpretability

**When to Use Each:**
- **BM25 Only:** Product catalogs, financial instruments, exact identifiers
- **SPLADE:** Enterprise knowledge bases, customer support, mixed terminology
- **Both (Hybrid):** General-purpose RAG, technical documentation, multi-domain corpora

#### Fusion Strategies

**1. Reciprocal Rank Fusion (RRF) - Zero-Config Default**

```
Formula: RRF_score(d) = Σ(1 / (k + rank_r(d)))

k = 60 (industry standard from Cormack et al. 2009)
rank_r(d) = position in ranked list r

Example:
Doc A ranked #1 in dense, #5 in BM25:
Score = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0308 = 0.0472

Doc B ranked #5 in both:
Score = 1/(60+5) + 1/(60+5) = 0.0308 + 0.0308 = 0.0616 (higher!)
```

**Properties:**
- Score-agnostic (uses ranks, not raw scores)
- No normalization needed (handles BM25 unbounded vs cosine -1 to 1)
- Works with zero labeled data
- Conservative fusion that rewards consistency across lists

**Implementation:**

```python
def reciprocal_rank_fusion(result_lists: list[list], k: int = 60) -> list:
    """
    Fuse multiple ranked result lists using RRF.
    Each result_list: [(doc_id, score), ...] sorted by relevance descending
    Returns: [(doc_id, rrf_score), ...] sorted descending
    """
    rrf_scores = {}
    for result_list in result_lists:
        for rank, (doc_id, _score) in enumerate(result_list, start=1):
            rrf_scores.setdefault(doc_id, 0.0)
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

**2. Convex Combination (Tuned Alpha)**

```
Formula: score(d) = α × normalized_dense(d) + (1-α) × normalized_sparse(d)

α = 1.0  → pure vector search
α = 0.0  → pure keyword search
α ≈ 0.5  → balanced (default starting point)
```

**Alpha Tuning Guidelines (from LlamaIndex):**
- Technical docs with error codes/API names/SKUs: α ≈ 0.3 (favor sparse)
- Customer support chatbots (conversational): α ≈ 0.7 (favor dense)
- Balanced starting point: α = 0.5

**Research Finding (Bruch et al., ACM TOIS 2023):**
- Convex combination outperforms RRF when alpha is tuned
- Sample efficiency: 50-100 labeled query pairs sufficient for tuning
- Specific normalization method matters less than having normalization

**Implementation with Min-Max Normalization:**

```python
def hybrid_convex_combination(
    bm25_results: dict,    # {doc_id: raw_bm25_score}
    dense_results: dict,   # {doc_id: cosine_score}
    alpha: float = 0.5     # 1.0 = pure dense, 0.0 = pure BM25
) -> list:
    """
    Combine BM25 and dense scores via convex combination.
    Requires min-max normalization to handle different score ranges.
    """
    def min_max_normalize(scores: list[float]) -> list[float]:
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
    bm25_ids = list(bm25_results.keys())
    bm25_norm = dict(zip(bm25_ids, min_max_normalize(list(bm25_results.values()))))
    
    dense_ids = list(dense_results.keys())
    dense_norm = dict(zip(dense_ids, min_max_normalize(list(dense_results.values()))))
    
    all_docs = set(bm25_ids) | set(dense_ids)
    combined = {}
    for doc_id in all_docs:
        d = dense_norm.get(doc_id, 0.0)
        b = bm25_norm.get(doc_id, 0.0)
        combined[doc_id] = alpha * d + (1 - alpha) * b
    
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

**3. Distribution-Based Score Fusion (DBSF)**

Developed by Qdrant team - adapts to query-specific score distributions

```python
def distribution_based_score_fusion(dense_scores, sparse_scores):
    """
    DBSF: Calculates mean and std of each retriever's scores
    Normalizes using mean ± 3σ bounds (adaptive per query)
    """
    def get_bounds(scores):
        mean = np.mean(scores)
        std = np.std(scores)
        return max(0, mean - 3*std), mean + 3*std
    
    dense_min, dense_max = get_bounds(dense_scores.values())
    sparse_min, sparse_max = get_bounds(sparse_scores.values())
    
    normalized_dense = {
        k: (v - dense_min) / (dense_max - dense_min)
        for k, v in dense_scores.items()
    }
    normalized_sparse = {
        k: (v - sparse_min) / (sparse_max - sparse_min)
        for k, v in sparse_scores.items()
    }
    
    # Combine normalized scores
    all_docs = set(normalized_dense.keys()) | set(normalized_sparse.keys())
    combined = {}
    for doc_id in all_docs:
        combined[doc_id] = (
            normalized_dense.get(doc_id, 0.0) + 
            normalized_sparse.get(doc_id, 0.0)
        )
    
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

**When to Use DBSF:**
- Score magnitudes vary significantly between retrievers across different queries
- Query-specific adaptive normalization outperforms static min-max
- Particularly useful with SPLADE (which has different distribution than BM25)

**4. RAG-Fusion (Multi-Query RRF)**

```
Process:
1. Generate K rewritten variants of original query using LLM
   - Different phrasings of the same question
   - Alternative framings of intent
2. Run dense + sparse retrieval on each variant
3. Apply RRF across ALL result lists (original + K variants)
4. Select top-K from fused results
```

**Advantage:** Improves recall by approaching query from multiple semantic angles  
**Cost:** K × LLM calls per query + K × 2 retrieval calls  
**Tradeoff:** Significant latency increase, occasional off-topic drift if generated queries diverge from original intent

### 1.3 Corrective RAG (CRAG)

**Problem:** Basic RAG's linear pipeline can't recover from retrieval failures

**Solution:** Self-correcting loops that detect and fix hallucinations

**Workflow:**

```
Input Query
    ↓
Retrieve Documents
    ↓
Relevance Check (LLM evaluates if docs support answer)
    ├─ RELEVANT → Generate Answer
    │            ↓
    │         Output
    │
    ├─ PARTIALLY RELEVANT → Query Rewriting
    │                       ↓
    │                    Re-retrieve with reformulated query
    │
    └─ IRRELEVANT → Expand Context or Alternative Search
                    ├─ Expand search scope
                    ├─ Add web search
                    ├─ Query expansion with synonyms
                    └─ Fallback generation with uncertain flag
```

**Key Components:**

1. **Relevance Assessment**
   - LLM judges whether retrieved docs answer the query
   - Avoid hallucinations from unrelated context
   
2. **Query Reformulation**
   - Rewrite query if docs don't match
   - Alternative phrasings
   - Synonym expansion
   
3. **Adaptive Retrieval**
   - Increase k on reformulation
   - Switch retrievers if primary fails
   - Expand search scope (web search, knowledge graphs)

**Implementation Pattern:**

```python
class CorrectiveRAG:
    def answer_query(self, query: str, max_corrections: int = 3):
        current_query = query
        for attempt in range(max_corrections):
            # Retrieve
            docs = self.retriever.search(current_query, k=10)
            
            # Assess relevance
            relevance = self.assess_relevance(current_query, docs)
            
            if relevance == "RELEVANT":
                # Generate answer from relevant docs
                return self.generator.answer(current_query, docs)
            
            elif relevance == "PARTIALLY_RELEVANT":
                # Reformulate query
                current_query = self.reformulate_query(query, docs)
                # Continue loop
            
            else:  # "IRRELEVANT"
                if attempt < max_corrections - 1:
                    # Try alternative retrieval strategy
                    docs = self.alternative_retriever.search(query, k=15)
                else:
                    # Fallback with uncertainty
                    return self.generator.answer_with_uncertainty(query, docs)
        
        return None
```

### 1.4 Query Expansion & Reformulation

**Purpose:** Handle vocabulary mismatch and improve recall

**Techniques:**

1. **Synonym-Based Expansion**
   ```python
   original: "database query optimization"
   expanded: [
       "database query optimization",
       "SQL query optimization", 
       "PostgreSQL performance tuning",
       "database indexing strategies",
       "query execution plan improvement"
   ]
   ```
   - Retrieve on all variants
   - Combine with RRF or convex combination

2. **LLM-Based Query Reformulation**
   ```
   Original: "How to fix connection refused error?"
   
   Reformulations:
   - "ECONNREFUSED error solution"
   - "connection reset by peer troubleshooting"
   - "TCP connection timeout diagnosis"
   - "socket connection failure debugging"
   ```
   - Better handles paraphrasing
   - Captures domain-specific terminology

3. **Sub-Query Decomposition**
   ```
   Original: "Which Italian explorer navigated South America's coast in the 1500s?"
   
   Sub-queries:
   - "Italian explorer 1500s"
   - "South America coast exploration 1500s"
   - "Portuguese/Spanish navigators South America"
   ```
   - Search each independently
   - Merge results with evidence linkage

**Trade-offs:**
- Expanded queries → Higher recall, potential noise
- More API calls → Increased latency and cost
- LLM-based expansion → Quality dependent on LLM capability

### 1.5 Knowledge Graph Integration

**Architectural Patterns:**

1. **Iterative Knowledge Graph Construction**
   - Build KG on-the-fly from retrieved documents
   - Semantic relationships extracted by LLM
   - Evolves with new documents
   - Reusable for related queries

2. **Multi-Hop Graph Traversal**
   ```
   User Query: "Which entities are harmed by X's actions?"
   
   KG Traversal:
   X (client) 
   ├─ actions → [Actions A1, A2, A3]
   │   ├─ A1 → affected_entities → [Entity1, Entity2]
   │   ├─ A2 → affected_entities → [Entity3]
   │   └─ A3 → affected_entities → []
   └─ Impact Analysis
       ├─ Entity1 → harm_type → "Financial loss"
       ├─ Entity2 → harm_type → "Operational disruption"
       └─ Entity3 → harm_type → "Reputational damage"
   ```

3. **Contextual Dictionary for Concept Mapping**
   - Pre-processed map of concepts → chunks containing them
   - Accelerates concept-centric retrieval
   - Prevents over-exploration
   - Similar to traditional index

**Implementation with WhyHow.AI Pattern:**

```python
class IterativeKnowledgeGraphRAG:
    def build_kg_for_entity(self, entity: str, seed_query: str):
        """
        Iteratively build knowledge graph around seed entity
        across multiple documents
        """
        kg = KnowledgeGraph()
        visited_chunks = set()
        
        # Define relationships to track (domain-specific)
        relationships = [
            "causes", "affects", "involves", 
            "located_in", "occurs_in", "related_to"
        ]
        
        for chunk in self.corpus:
            if chunk.id in visited_chunks:
                continue
            
            # Extract relationships involving entity
            extracted = self.llm.extract_relationships(
                entity, chunk.text, relationships
            )
            
            for rel_type, target_entity, evidence in extracted:
                kg.add_edge(
                    entity, target_entity,
                    relationship=rel_type,
                    evidence=evidence,
                    chunk_id=chunk.id
                )
            
            visited_chunks.add(chunk.id)
        
        # Build contextual dictionary for future queries
        for concept in kg.nodes:
            kg.concept_dictionary[concept] = [
                chunk_id for chunk_id in kg.get_evidence(concept)
            ]
        
        return kg
    
    def query_with_kg(self, query: str, kg: KnowledgeGraph):
        """
        Use constructed KG to guide retrieval
        """
        # Identify relevant concept from query
        seed_concept = self.llm.identify_seed_concept(query, kg)
        
        # Traverse KG to find related information
        traversal_paths = kg.find_relevant_paths(seed_concept, query)
        
        # Retrieve evidence for each path
        all_evidence = []
        for path in traversal_paths:
            for chunk_id in path.evidence_chunks:
                all_evidence.append(self.corpus[chunk_id])
        
        # Generate answer with evidence links
        return self.generator.answer_with_evidence(query, all_evidence)
```

---

## 2. Retrieval Optimization Techniques

### 2.1 Re-Ranking with Cross-Encoders

**Two-Stage Retrieval Pattern:**

```
Stage 1: Fast, Broad Retrieval
├─ Hybrid search (BM25 + dense)
├─ Low latency requirements
├─ Goal: Maximize recall (get relevant docs into candidate set)
└─ Output: Top-20 candidates

Stage 2: Slow, Precise Re-Ranking
├─ Cross-encoder scores each query-document pair jointly
├─ More accurate than embedding similarity
├─ Higher latency acceptable (scores only 20 docs)
├─ Goal: Maximize precision (order for LLM)
└─ Output: Top-5 re-ranked results → LLM
```

**Why Cross-Encoders Win:**

| Method | Query Representation | Doc Representation | Joint Scoring |
|--------|-------------------|-------------------|---|
| Dense Embedding | Single vector | Single vector | Cosine similarity (fast) |
| Cross-Encoder | Joint input | Joint input | Transformer forward pass (accurate) |

**Example:** Query: "ECONNREFUSED connection error fix"

```
Hybrid Search (Recall maximization):
1. [0.78] "ECONNREFUSED error occurs when server..."
2. [0.76] "Connection reset by peer troubleshooting..."
3. [0.75] "TCP timeout handling in Node.js..."
4. [0.72] "Port not listening error debugging..."
5. [0.68] "Network socket connection basics..."

Cross-Encoder Reranking:
1. [8.9/10] "ECONNREFUSED error occurs when server..."
2. [7.2/10] "TCP timeout handling in Node.js..."
3. [6.8/10] "Connection reset by peer troubleshooting..."
4. [5.1/10] "Network socket connection basics..."
5. [2.3/10] "Port not listening error debugging..."
```

**Popular Cross-Encoder Models:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, accurate, recommended)
- `cross-encoder/qnli-distilroberta-base` (domain: NLI)
- `cross-encoder/stsb-roberta-base` (domain: semantic similarity)

**Production Implementation:**

```python
from sentence_transformers import CrossEncoder
import numpy as np

class HybridSearchWithReranking:
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def retrieve_and_rerank(
        self,
        query: str,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5
    ) -> list[RerankedDoc]:
        # Stage 1: Hybrid retrieval for recall
        dense_results = self.dense_retriever.search(query, k=top_k_retrieve)
        sparse_results = self.sparse_retriever.search(query, k=top_k_retrieve)
        
        # Fusion with RRF
        fused = rrf_fusion([dense_results, sparse_results], k=60)
        candidate_docs = [doc for doc, score in fused[:top_k_retrieve]]
        
        # Stage 2: Cross-encoder reranking for precision
        pairs = [[query, doc.text] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(pairs)
        
        # Sort by rerank scores
        ranked = sorted(
            zip(candidate_docs, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k_rerank]
        
        return [
            RerankedDoc(doc=d, score=float(s))
            for d, s in ranked
        ]
```

**Latency Profile:**
- Hybrid retrieval (top-20): ~50-100ms
- Cross-encoder reranking (20 docs): ~20-50ms
- Total: ~70-150ms vs. dense-only ~50-100ms

**Cost-Benefit:** +20-50ms latency for +8-12% precision improvement on top-5 results

### 2.2 Semantic & Sparse-Dense Fusion

**Mathematical Foundations:**

**Cosine Similarity (Dense):**
```
similarity(q, d) = (q · d) / (||q|| × ||d||)
Range: [-1, 1], typically [0, 1] after normalization
Optimized: Fast dot product with HNSW/FAISS
```

**BM25 Score (Sparse):**
```
BM25(D, Q) = Σ(IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl)))

where:
- IDF: Inverse document frequency (term rarity)
- f(qi, D): Frequency of term in document
- |D|: Document length
- k1, b: Tuning parameters (typically k1=1.5, b=0.75)
- Range: [0, ∞) unbounded

Key insight: Saturates on term frequency (diminishing returns for repetition)
Avoids: Penalizing documents with more occurrences of query terms
```

**Fusion Challenge:** 
- Dense: normalized similarity score [0,1]
- Sparse: unbounded raw score [0, ∞)
- Direct combination impossible without normalization

**Normalization Approaches:**

1. **Min-Max Normalization (Most Common)**
   ```python
   normalized = (score - min_score) / (max_score - min_score)
   # Handles extreme outliers poorly
   # Sensitive to corpus composition
   ```

2. **Z-Score Normalization**
   ```python
   normalized = (score - mean) / std_dev
   # Better handles outliers
   # Assumes normal distribution
   ```

3. **Quantile-Based Normalization**
   ```python
   normalized = (score - percentile_25) / (percentile_75 - percentile_25)
   # Robust to outliers
   # Recommended for production
   ```

4. **Distribution-Based Fusion (DBSF)** - Query-Adaptive
   ```python
   # Qdrant approach: uses mean ± 3σ bounds per query
   bounds = (mean - 3*std, mean + 3*std)
   normalized = clip((score - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
   ```

### 2.3 Chunking Strategies & Optimization

**Problem:** Document chunking directly impacts retrieval quality

**Standard Approach (Suboptimal):**
```python
# Fixed token window with overlap
docs = split_into_chunks(
    text=full_document,
    chunk_size=256,          # tokens
    overlap=50               # tokens
)
```

**Issues:**
- Arbitrary chunk boundaries cut off semantic context
- Overlap doesn't guarantee coherence
- No consideration of document structure
- Same chunk size for all document types

**Advanced Chunking Strategies:**

1. **Semantic/Hierarchical Chunking**
   ```python
   def semantic_chunking(document: str, similarity_threshold: float = 0.7):
       """
       Split document where semantic similarity drops below threshold
       Preserves semantic boundaries better than token-based
       """
       sentences = sentence_tokenize(document)
       sentence_embeddings = embed_model.encode(sentences)
       
       chunks = []
       current_chunk = [sentences[0]]
       
       for i in range(1, len(sentences)):
           similarity = cosine_similarity(
               sentence_embeddings[i-1],
               sentence_embeddings[i]
           )
           
           if similarity < similarity_threshold:
               # Semantic boundary detected
               chunks.append(" ".join(current_chunk))
               current_chunk = [sentences[i]]
           else:
               current_chunk.append(sentences[i])
       
       chunks.append(" ".join(current_chunk))
       return chunks
   ```

2. **Document Structure-Aware Chunking**
   ```python
   def structure_aware_chunking(document: dict):
       """
       Respect document structure: sections, headings, tables
       """
       chunks = []
       
       for section in document['sections']:
           chunk = {
               'text': section['content'],
               'metadata': {
                   'section': section['title'],
                   'level': section['level'],
                   'document': document['title']
               }
           }
           chunks.append(chunk)
       
       # Process within-section content separately
       for chunk in chunks:
           if len(chunk['text']) > MAX_TOKEN_LENGTH:
               # Only split large sections semantically
               chunk['text'] = semantic_chunk(chunk['text'])
       
       return chunks
   ```

3. **Context Window Aware Chunking**
   ```python
   def context_aware_chunking(
       document: str,
       chunk_size: int = 256,
       context_size: int = 100,
       model_context_window: int = 4096
   ):
       """
       Ensure chunks fit within LLM context window with other context
       """
       # Reserve tokens for:
       # - User query: ~100 tokens
       # - System prompt: ~50 tokens
       # - Metadata/formatting: ~100 tokens
       available_for_content = model_context_window - 250
       
       # Multiple chunks can fit
       num_chunks_per_context = available_for_content // chunk_size
       
       chunks = split_into_chunks(document, chunk_size, context_size)
       
       return chunks, num_chunks_per_context
   ```

**Optimal Settings by Document Type:**

| Document Type | Chunk Size | Overlap | Strategy |
|--------------|-----------|---------|----------|
| Code | 100-200 tokens | 20% | Structure-aware (functions, classes) |
| Technical Docs | 256-512 tokens | 10-20% | Semantic + hierarchy |
| Legal Contracts | 256 tokens | 20% | Structure-aware (clauses, sections) |
| News/Articles | 256-512 tokens | 10% | Semantic boundaries |
| Academic Papers | 512 tokens | 15% | Section-aware with overlap |
| Long Form (Books) | 512-1024 tokens | 10% | Semantic + hierarchical |

### 2.4 Context Window Management

**Challenge:** LLMs have finite context windows; retrieving too much context degrades performance

**Window Degradation Research:**
```
Typical pattern (from recent studies):
- Optimal: Use 60-70% of context window
- 70-90% context: -2-5% performance degradation
- 90%+ context: -10-20% performance degradation
- 100%+ (exceeds window): Truncation, lost information
```

**Strategies:**

1. **Lost-in-the-Middle Problem**
   - Relevant information at start/end: usually found
   - Relevant information in middle: often missed
   - Solution: Place most important chunks at boundaries

2. **Dynamic Context Selection**
   ```python
   def select_optimal_context(
       retrieved_chunks: list[str],
       context_budget: int = 2048,  # tokens for retrieved content
       model_context_window: int = 4096
   ) -> list[str]:
       """
       Select subset of chunks that maximize quality within budget
       """
       scored_chunks = []
       for i, chunk in enumerate(retrieved_chunks):
           token_count = count_tokens(chunk)
           
           # Scoring factors:
           relevance_score = get_relevance_score(chunk)  # from reranker
           recency_score = 1.0 if is_recent(chunk) else 0.5
           position_score = 1.0 / (1 + abs(i - len(retrieved_chunks)/2))
           
           combined_score = (
               0.6 * relevance_score + 
               0.2 * recency_score + 
               0.2 * position_score
           )
           
           scored_chunks.append({
               'chunk': chunk,
               'score': combined_score,
               'tokens': token_count
           })
       
       # Select greedily by score-to-token ratio
       sorted_chunks = sorted(
           scored_chunks,
           key=lambda x: x['score'] / max(x['tokens'], 1),
           reverse=True
       )
       
       selected = []
       total_tokens = 0
       for item in sorted_chunks:
           if total_tokens + item['tokens'] <= context_budget:
               selected.append(item['chunk'])
               total_tokens += item['tokens']
       
       return selected
   ```

3. **Recursive Summarization**
   ```
   For very long contexts:
   
   Original chunks → Summarize groups → Recursive summarization
   
   Example (Legal Document):
   1000 sentences
     ↓ (summarize 100-sentence groups)
   10 summaries (100 sentences)
     ↓ (summarize 2-summary groups)
   5 summaries (50 sentences)
     ↓ (use these + top-K relevant chunks)
   Final context: 50 + top-5 = 55 sentences
   ```

---

## 3. Vector Databases & Search Infrastructure

### 3.1 Vector Database Comparison (2026)

| Feature | Qdrant | Weaviate | Milvus | Pinecone | Elasticsearch |
|---------|--------|----------|--------|----------|---|
| **Open Source** | Yes | Yes | Yes | No | Yes |
| **Dense Vectors** | Yes | Yes | Yes | Yes | Yes |
| **Sparse Vectors** | SPLADE native | BM25F | Yes | Client-side | BM25 native |
| **Hybrid Search** | RRF, DBSF | alpha-fusion | Weighted | Convex combo | RRF (v8.9+) |
| **Indexing** | HNSW, HNSWF | HNSW | HNSW, IVF | Proprietary | Hierarchical |
| **Filtering** | Exact, range | Exact, range | Pre-filter | Namespace | Query context |
| **Multi-tenancy** | Collections | Namespaces | Partitions | Namespaces | Indices |
| **Reranking** | LLM-compatible | Custom | LLM-compatible | Supported | Post-process |
| **Production Ready** | Yes (v1.7+) | Yes | Yes | Yes | Yes |
| **Best For** | Hybrid RAG | Enterprise | Scale | Managed service | Full-text + vector |

### 3.2 Approximate Nearest Neighbor (ANN) Algorithms

**HNSW (Hierarchical Navigable Small World)**

```
Tree-like structure with multiple layers (0 to ln(n))
- Layer 0: All nodes connected (dense base graph)
- Higher layers: Progressively fewer nodes (sparse graph)

Search process:
1. Start at layer M (top)
2. Greedy search to nearest node
3. Drop to lower layer
4. Repeat until reaching layer 0
5. Return nearest neighbors from layer 0

Complexity:
- Build: O(n log n) with proper parameter tuning
- Search: O(log n)
- Memory: O(n × efConstruction)

Tuning parameters:
- M: Max connections per node (default 16, increase for recall)
- efConstruction: Quality during build (default 200, higher = better quality)
- efSearch: Quality during search (default 100, can increase for better recall)
```

**IVF (Inverted File Index)**

```
Quantizes space into clusters (inverted lists)
1. K-means clustering of vectors
2. Each vector assigned to nearest cluster
3. Query: Search K clusters containing nearest centroids
4. Within clusters: Linear search or PQ

Complexity:
- Build: O(n × K) for k-means
- Search: O(K × n/K) = O(n/clustering_quality)
- Memory: Minimal (cluster assignments only)

IVF-PQ variant:
- Adds Product Quantization
- Further compresses vectors
- Trade: Compression for accuracy
```

**Comparison (2026 benchmarks):**

| Metric | HNSW | IVF | IVF-PQ |
|--------|------|-----|--------|
| Recall @10 | 99.5% | 97.2% | 95.1% |
| QPS (1M vectors) | 500-1000 | 2000-5000 | 5000-10000 |
| Memory per vector | ~1KB | ~64B | ~32B |
| Build time (1M) | ~60s | ~30s | ~20s |
| Tuning difficulty | Medium | Medium | High |

**Production Guidance:**
- HNSW: < 10M vectors, recall > 98% needed, budget for memory
- IVF: 10M-1B vectors, balance recall/speed/memory
- IVF-PQ: > 100M vectors, can tolerate slight recall loss

### 3.3 Quantization Techniques

**Product Quantization (PQ)**

```
Goal: Compress vectors from 1536D to 32-128 bytes

Process:
1. Divide 1536D into 16 subspaces (96D each)
2. K-means cluster each subspace (256 clusters)
3. Replace each 96D subspace with 1-byte cluster ID
4. Result: 1536D → 16 bytes (96x compression)

Example:
Original vector (1536D, fp32): 6144 bytes
PQ-compressed (16 bytes): 384x compression
Memory: 1M vectors × 16 bytes = 16MB (vs 6.4GB)

Trade: accuracy vs compression
Typical: 8-16 byte representation ≈ 95% of full accuracy
```

**Scalar Quantization (SQ)**

```
Simpler than PQ:
1. Find min/max value in vector
2. Linearly quantize to 8-bits per dimension
3. Result: 1536D → 192 bytes (32x compression)

Cost-benefit:
- Build: Extremely fast
- Memory: 32x reduction
- Accuracy: 90-95% of full vectors
- Use case: When speed/memory critical, slight accuracy loss acceptable
```

**Binary Quantization (BQ)**

```
Extreme compression: 1-bit per dimension

1. Threshold each dimension at 0
2. 1 = positive, 0 = negative
3. Pack 8 bits per byte
4. Result: 1536D → 192 bits = 24 bytes (256x compression)

Hamming distance search: Single CPU instruction

Accuracy: 75-85% of full vectors
Use case: Scale to billions, or pre-filter before full vectors
```

**Implementation Pattern:**

```python
def quantize_vectors_for_storage(vectors: np.ndarray, method: str = "pq"):
    """
    Compress vectors for memory-efficient storage
    """
    if method == "pq":
        # Product Quantization
        n_subspaces = 16
        subspace_size = len(vectors[0]) // n_subspaces
        
        encoded = []
        for vector in vectors:
            code = []
            for i in range(n_subspaces):
                subspace = vector[i*subspace_size:(i+1)*subspace_size]
                # Find nearest cluster in pre-trained codebook
                nearest_cluster = find_nearest_cluster(subspace)
                code.append(nearest_cluster)
            encoded.append(bytes(code))
        return encoded
    
    elif method == "sq":
        # Scalar Quantization
        return vectors.astype(np.int8)  # Quantize to int8
    
    elif method == "bq":
        # Binary Quantization
        binary = (vectors > 0).astype(np.uint8)
        # Pack into bytes
        packed = np.packbits(binary)
        return packed
```

### 3.4 Indexing Strategies & Performance

**Single-Vector-Index vs Composite Index**

```
Single Dense Index (Traditional):
- One HNSW index for embeddings
- Fast dense search (500-5000 QPS)
- Limited to similarity queries
- Recall limited by embedding model

Composite Index (Advanced):
- Dense index: HNSW on embeddings
- Sparse index: Inverted list on tokens
- Metadata index: Structured filters
- Hybrid fusion at query time

Trade:
- Storage: +2-3x (maintain multiple indexes)
- Query: More complex logic, but better results
- Maintenance: More moving parts
```

**Caching Strategies for Search**

```
Three-Level Cache (Production Pattern):

L1: Query Result Cache (LRU, in-memory)
├─ Cache top-100 results for exact query matches
├─ Hit rate: 10-30% (repeated queries common)
├─ Latency: <1ms on hit
└─ Size: 10-100MB for typical workloads

L2: Chunk Cache (Semantic hash)
├─ Cache embeddings + chunks for popular documents
├─ Hit rate: 20-40% (chunked content)
├─ Latency: 5-10ms on hit (dict lookup)
└─ Size: 1-10GB

L3: Vector Index Cache (OS page cache)
├─ OS caches frequently accessed index pages
├─ Automatic, no explicit management
├─ Hit rate: 50-80% for HNSW
└─ Latency: 50-200ms on hit vs 500ms+ on disk
```

---

## 4. Synthetic Data Generation for LLMs

### 4.1 LLM-Based Data Generation Techniques

**Problem:** High-quality labeled data is expensive and hard to obtain

**Solution:** Use existing LLMs to generate synthetic training data

**Approaches:**

1. **Instruction-Following Data Generation**
   ```python
   def generate_qa_pairs(documents: list[str], num_pairs: int = 1000):
       """
       Generate diverse Q&A pairs from documents
       """
       generated_pairs = []
       
       for doc in documents:
           prompt = f"""
           Given the following document, generate 5 diverse question-answer pairs
           that test comprehension and reasoning.
           
           Vary the questions:
           - Direct factual questions
           - Multi-hop reasoning questions
           - Inference-based questions
           - Contradictory scenarios
           - Open-ended exploration
           
           Document:
           {doc}
           
           Format:
           Q1: [question]
           A1: [answer with evidence]
           Q2: [question]
           A2: [answer with evidence]
           ...
           """
           
           response = llm.generate(prompt)
           pairs = parse_qa_pairs(response)
           generated_pairs.extend(pairs)
       
       return generated_pairs[:num_pairs]
   ```

2. **Contrastive Data Generation**
   ```python
   def generate_contrastive_pairs(
       positive_examples: list[str],
       num_negatives: int = 5
   ):
       """
       Create positive and negative examples for contrastive learning
       """
       contrastive_data = []
       
       for example in positive_examples:
           # Generate hard negatives (similar but wrong)
           negatives = llm.generate(f"""
           Given this example: "{example}"
           
           Generate {num_negatives} similar-looking but incorrect variants.
           These should be plausible wrong answers that a model might confuse.
           
           Example: "Paris is the capital of France"
           Negatives might be:
           - "Paris is the capital of Germany"
           - "Lyon is the capital of France"
           - "Paris is the largest city in Europe"
           """)
           
           hard_negatives = parse_negatives(negatives)
           
           contrastive_data.append({
               'anchor': example,
               'positives': [example],  # Original is positive
               'negatives': hard_negatives
           })
       
       return contrastive_data
   ```

3. **Chain-of-Thought Data Generation**
   ```python
   def generate_reasoning_chains(
       problems: list[str],
       num_chains: int = 3
   ):
       """
       Generate multiple reasoning chains per problem
       Captures reasoning diversity
       """
       chains = []
       
       for problem in problems:
           for attempt in range(num_chains):
               # Temperature > 0 for diversity
               prompt = f"""
               Solve this problem step-by-step:
               {problem}
               
               Show each reasoning step clearly.
               Consider multiple approaches if applicable.
               """
               
               chain = llm.generate(
                   prompt,
                   temperature=0.8,  # Higher for diversity
                   max_tokens=500
               )
               
               chains.append({
                   'problem': problem,
                   'reasoning_chain': chain,
                   'attempt': attempt + 1
               })
       
       return chains
   ```

### 4.2 Distillation & Knowledge Transfer

**Purpose:** Compress knowledge from large models into smaller ones

**Process:**

```
Teacher Model (Large, Expensive)
    ↓ (generates soft targets)
Synthetic Data with Confidence Scores
    ↓ (trains on)
Student Model (Small, Fast)
```

**Temperature-Scaled Softening:**

```python
def knowledge_distillation(
    teacher_model,
    student_model,
    training_data,
    temperature: float = 4.0,
    alpha: float = 0.7
):
    """
    Knowledge distillation: student learns from teacher
    
    temperature: Higher = softer probability distribution
                 Exposes more of teacher's reasoning
    alpha: Weight between distillation loss and task loss
    """
    
    for batch in training_data:
        # Teacher forward pass (soft targets)
        with torch.no_grad():
            teacher_logits = teacher_model(batch)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # Student forward pass
        student_logits = student_model(batch)
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        
        # Distillation loss
        kl_div_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            teacher_probs,
            reduction='batchmean'
        )
        
        # Task loss (ground truth labels)
        task_loss = cross_entropy(student_logits, batch.labels)
        
        # Combined loss
        total_loss = alpha * kl_div_loss + (1 - alpha) * task_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    return student_model
```

**Selecting Good Temperature:**
- Low (1-2): Focus on high-confidence predictions
- Medium (3-5): Balance teacher knowledge + task learning
- High (10+): Soften all predictions, learn from subtle differences

### 4.3 Data Augmentation Strategies

**1. Paraphrasing Augmentation**
```python
def augment_with_paraphrasing(
    examples: list[str],
    num_paraphrases: int = 3
):
    """
    Generate multiple paraphrases of each example
    """
    augmented = []
    
    for example in examples:
        paraphrases = llm.generate(f"""
        Generate {num_paraphrases} semantically equivalent paraphrases
        of this text. Vary the phrasing, vocabulary, and structure.
        
        Original: "{example}"
        
        Paraphrase 1:
        Paraphrase 2:
        Paraphrase 3:
        """, temperature=0.7)
        
        for para in parse_paraphrases(paraphrases):
            augmented.append({
                'original': example,
                'paraphrase': para,
                'type': 'paraphrase'
            })
    
    return augmented
```

**2. Back-Translation Augmentation**
```
Original: "How do you fix a connection error?"
    ↓ (translate to German)
"Wie behebt man einen Verbindungsfehler?"
    ↓ (translate back to English)
"How do you correct a connection error?"
    ↓ (keep if different, filter if same)
Output: Semantically similar but differently phrased
```

**3. Mixup & Interpolation**
```python
def mixup_examples(example1: str, example2: str, alpha: float = 0.5):
    """
    Blend two examples for in-between representation
    Particularly useful for embeddings/vectors
    """
    embed1 = embed_model.encode(example1)
    embed2 = embed_model.encode(example2)
    
    # Linear interpolation in embedding space
    mixed_embed = alpha * embed1 + (1 - alpha) * embed2
    
    # Decode back (if possible)
    mixed_text = decode_embedding(mixed_embed)
    
    return mixed_text, mixed_embed
```

### 4.4 Consistency & Quality Verification

**Problem:** Synthetic data can contain hallucinations, inconsistencies, duplicates

**Solutions:**

1. **Self-Consistency Filtering**
   ```python
   def filter_by_self_consistency(
       generated_examples: list[dict],
       num_checks: int = 3,
       consistency_threshold: float = 0.8
   ):
       """
       Re-generate answers for consistency
       Keep only examples where LLM agrees with itself
       """
       filtered = []
       
       for example in generated_examples:
           answers = []
           
           for _ in range(num_checks):
               answer = llm.generate(
                   example['question'],
                   temperature=0.0  # Deterministic generation
               )
               answers.append(answer)
           
           # Check consistency
           consistency_score = compute_string_similarity(answers)
           
           if consistency_score > consistency_threshold:
               # Keep and store agreement count
               example['consistency'] = consistency_score
               filtered.append(example)
       
       return filtered
   ```

2. **Semantic Similarity Deduplication**
   ```python
   def deduplicate_semantic(examples: list[str], threshold: float = 0.95):
       """
       Remove near-duplicate examples
       Based on semantic similarity, not string matching
       """
       embeddings = embed_model.encode(examples)
       
       # Compute pairwise similarity
       similarity_matrix = cosine_similarity(embeddings)
       
       kept_indices = []
       for i in range(len(examples)):
           # Check if similar to any kept example
           is_duplicate = False
           for j in kept_indices:
               if similarity_matrix[i][j] > threshold:
                   is_duplicate = True
                   break
           
           if not is_duplicate:
               kept_indices.append(i)
       
       return [examples[i] for i in kept_indices]
   ```

3. **Quality Scoring Pipeline**
   ```python
   def score_synthetic_data_quality(example: dict) -> float:
       """
       Multi-factor quality assessment
       """
       scores = {}
       
       # Factor 1: Self-consistency (0-1)
       scores['consistency'] = example.get('consistency', 0.5)
       
       # Factor 2: Factual correctness (verified against knowledge base)
       kb_matches = knowledge_base.verify(example['answer'])
       scores['factuality'] = min(1.0, len(kb_matches) / max(1, len(example['facts'])))
       
       # Factor 3: Linguistic quality (no grammatical errors)
       linguistic_check = language_model.check_grammar(example['question'])
       scores['linguistic'] = 1.0 if linguistic_check.errors == 0 else 0.8
       
       # Factor 4: Diversity (not similar to training data)
       train_similarity = max(
           cosine_similarity(
               embed_model.encode([example['question']]),
               embed_model.encode(training_data['questions'])
           )[0]
       )
       scores['diversity'] = 1.0 - train_similarity
       
       # Weighted combination
       total_score = (
           0.3 * scores['consistency'] +
           0.3 * scores['factuality'] +
           0.2 * scores['linguistic'] +
           0.2 * scores['diversity']
       )
       
       return total_score
   ```

### 4.5 Self-Supervised Learning with Synthetic Data

**Pattern:** Use synthetic data as weak supervision, refine with unlabeled data

```
Generate Synthetic Data (LLM)
    ↓ Quality Filtering
High-Quality Synthetic Data
    ↓ Use as Weak Labels
Train on Synthetic Labels
    ↓ Apply to Unlabeled Data
Generate Pseudo-Labels
    ↓ Confidence-Weighted Training
Fine-tune Model
    ↓ Evaluate on Real Labeled Data
```

**Implementation:**

```python
class SelfSupervisedRefinement:
    def __init__(self, synthetic_data, unlabeled_data):
        self.synthetic = synthetic_data
        self.unlabeled = unlabeled_data
        self.model = initialize_model()
    
    def refine_with_pseudo_labels(self, confidence_threshold: float = 0.8):
        """
        Generate pseudo-labels on unlabeled data using model trained on synthetic data
        """
        # Step 1: Train on filtered synthetic data
        self.model.train(
            self.synthetic,
            epochs=3,
            learning_rate=1e-4
        )
        
        # Step 2: Generate pseudo-labels on unlabeled data
        pseudo_labels = []
        confidences = []
        
        self.model.eval()
        for example in self.unlabeled:
            predictions, confidence = self.model.predict_with_confidence(example)
            
            if confidence > confidence_threshold:
                pseudo_labels.append({
                    'text': example,
                    'label': predictions,
                    'confidence': confidence,
                    'source': 'pseudo'
                })
                confidences.append(confidence)
        
        # Step 3: Combine synthetic + pseudo-labeled data
        combined_data = self.synthetic + pseudo_labels
        
        # Step 4: Confidence-weighted training
        weights = [
            1.0 for _ in self.synthetic  # Full weight for synthetic
        ] + [
            c for c in confidences  # Confidence weight for pseudo
        ]
        
        self.model.train(
            combined_data,
            sample_weights=weights,
            epochs=5,
            learning_rate=5e-5
        )
        
        return self.model, {
            'pseudo_labeled_count': len(pseudo_labels),
            'average_confidence': np.mean(confidences),
            'high_confidence_ratio': sum(1 for c in confidences if c > 0.9) / len(confidences)
        }
```

---

## 5. Quality Assurance & Validation

### 5.1 Synthetic Data Validation Methods

**Multi-Layer Validation Framework:**

```
Layer 1: Structural Validation
├─ Format checks (valid JSON, field presence)
├─ Type checks (strings, numbers, lists)
├─ Length checks (within reasonable bounds)
└─ Schema validation against definition

Layer 2: Semantic Validation
├─ Language detection (matches expected language)
├─ Grammar checking
├─ Entity recognition (expected entities present)
└─ Semantic coherence

Layer 3: Factual Validation
├─ Knowledge base verification
├─ Consistency with training data
├─ Contradiction detection
└─ Evidence backing

Layer 4: Statistical Validation
├─ Distribution matching (synthetic vs real)
├─ Diversity metrics
├─ Outlier detection
└─ Statistical hypothesis testing
```

**Implementation:**

```python
def validate_synthetic_batch(
    batch: list[dict],
    validation_rules: dict,
    knowledge_base: KB = None
) -> ValidationReport:
    """
    Comprehensive validation of synthetic data batch
    """
    report = ValidationReport()
    
    for idx, example in enumerate(batch):
        # Layer 1: Structural
        structural = validate_structure(example, validation_rules['schema'])
        if not structural.valid:
            report.add_error(idx, 'structural', structural.errors)
            continue
        
        # Layer 2: Semantic
        semantic = validate_semantics(example)
        if not semantic.valid:
            report.add_warning(idx, 'semantic', semantic.warnings)
        
        # Layer 3: Factual (if KB available)
        if knowledge_base:
            factual = validate_against_kb(example, knowledge_base)
            if not factual.valid:
                report.add_error(idx, 'factual', factual.errors)
        
        # Layer 4: Statistical
        statistical = check_for_anomalies(example, batch[:idx])
        if statistical.is_outlier:
            report.add_warning(idx, 'statistical', f'Outlier detected: {statistical.reason}')
    
    return report
```

### 5.2 Duplicate & Near-Duplicate Detection

**Problem:** Synthetic data often contains unintended duplicates

**Solution: Three-Layer Detection**

```python
def detect_duplicates(
    examples: list[dict],
    exact_match_fields: list[str],
    semantic_similarity_threshold: float = 0.95,
    fuzzy_threshold: float = 0.85
) -> DuplicateReport:
    """
    Detect exact, semantic, and fuzzy duplicates
    """
    report = DuplicateReport()
    
    # Layer 1: Exact string matching
    seen_exact = {}
    for idx, example in enumerate(examples):
        key = tuple(example.get(f, '') for f in exact_match_fields)
        if key in seen_exact:
            report.add_duplicate(idx, seen_exact[key], 'exact')
        else:
            seen_exact[key] = idx
    
    # Layer 2: Fuzzy string matching (token-based)
    for i in range(len(examples)):
        for j in range(i + 1, len(examples)):
            text_i = examples[i].get('text', '')
            text_j = examples[j].get('text', '')
            
            if fuzz.ratio(text_i, text_j) > fuzzy_threshold * 100:
                report.add_duplicate(j, i, 'fuzzy')
    
    # Layer 3: Semantic similarity (embedding-based)
    embeddings = embed_model.encode([ex.get('text', '') for ex in examples])
    similarity_matrix = cosine_similarity(embeddings)
    
    for i in range(len(examples)):
        for j in range(i + 1, len(examples)):
            if similarity_matrix[i][j] > semantic_similarity_threshold:
                report.add_duplicate(j, i, 'semantic')
    
    return report
```

### 5.3 Quality Metrics for Generated Data

**Quantitative Metrics:**

1. **Diversity Score**
   ```python
   def calculate_diversity_score(examples: list[str]) -> float:
       """
       Measure semantic diversity of generated examples
       Higher = more diverse
       """
       embeddings = embed_model.encode(examples)
       
       # Compute pairwise distances
       distances = []
       for i in range(len(embeddings)):
           for j in range(i + 1, len(embeddings)):
               dist = 1 - cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
               distances.append(dist)
       
       # Average pairwise distance = diversity
       return np.mean(distances) if distances else 0.0
   ```

2. **Correctness Score**
   ```python
   def calculate_correctness_score(
       examples: list[dict],
       eval_fn  # Function to evaluate correctness
   ) -> float:
       """
       Percentage of examples that pass evaluation
       """
       correct = sum(1 for ex in examples if eval_fn(ex))
       return correct / len(examples) if examples else 0.0
   ```

3. **Relevance Score**
   ```python
   def calculate_relevance_score(
       examples: list[dict],
       queries: list[str]
   ) -> float:
       """
       Average relevance of examples to queries
       Using BM25 + embedding similarity
       """
       scores = []
       
       for query in queries:
           query_embed = embed_model.encode([query])[0]
           
           for example in examples:
               example_text = example.get('text', '')
               example_embed = embed_model.encode([example_text])[0]
               
               # Combine BM25 + semantic similarity
               bm25_score = bm25_scorer.score(query, example_text)
               semantic_score = cosine_similarity([query_embed], [example_embed])[0][0]
               
               combined = 0.3 * bm25_score + 0.7 * semantic_score
               scores.append(combined)
       
       return np.mean(scores) if scores else 0.0
   ```

### 5.4 Bias Detection in Synthetic Data

**Bias Sources in Synthetic Data:**

1. **Model Bias Propagation**
   - LLM training reflects biases in training data
   - Synthetic data inherits these biases

2. **Prompt Engineering Bias**
   - Biased prompts generate biased data
   - Subtle framing affects outputs

3. **Demographic Bias**
   - Representation imbalance
   - Stereotyping in generation

**Detection Methods:**

```python
def detect_demographic_bias(
    examples: list[dict],
    sensitive_attributes: dict  # e.g., {'gender': ['male', 'female']}
) -> BiasReport:
    """
    Detect demographic representation imbalance
    """
    report = BiasReport()
    
    for attribute, values in sensitive_attributes.items():
        value_counts = {v: 0 for v in values}
        
        # Count occurrences of each value
        for example in examples:
            for value in values:
                if value.lower() in example.get('text', '').lower():
                    value_counts[value] += 1
        
        # Calculate representation percentages
        total = sum(value_counts.values())
        if total == 0:
            report.add_finding(attribute, 'No instances found', confidence=1.0)
            continue
        
        percentages = {k: v / total for k, v in value_counts.items()}
        
        # Chi-square test for deviation from expected uniform distribution
        expected = total / len(values)
        chi_square = sum((v - expected) ** 2 / expected for v in value_counts.values())
        
        report.add_finding(
            attribute,
            f'Distribution: {percentages}',
            chi_square_statistic=chi_square,
            is_imbalanced=chi_square > 3.84  # Critical value for p=0.05, df=1
        )
    
    return report
```

### 5.5 Statistical Validation Techniques

**Distribution Matching:**

```python
def validate_distribution_match(
    synthetic_data: list[float],
    real_data: list[float],
    test_type: str = 'ks'  # Kolmogorov-Smirnov
) -> ValidationResult:
    """
    Test whether synthetic and real data distributions match
    """
    result = ValidationResult()
    
    if test_type == 'ks':
        # Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(synthetic_data, real_data)
        result.test = 'KS'
        result.statistic = statistic
        result.p_value = p_value
        # Fails if p < 0.05 (distributions differ significantly)
        result.pass = p_value > 0.05
    
    elif test_type == 'wasserstein':
        # Wasserstein distance (better for continuous distributions)
        distance = wasserstein_distance(synthetic_data, real_data)
        result.test = 'Wasserstein'
        result.statistic = distance
        # Pass if distance < threshold (domain-dependent)
        result.pass = distance < 0.15  # Tune based on use case
    
    elif test_type == 'anderson':
        # Anderson-Darling test
        result_ad = anderson_ksamp([synthetic_data, real_data])
        result.test = 'Anderson-Darling'
        result.statistic = result_ad.statistic
        # Pass if within critical values
        result.pass = result_ad.statistic < result_ad.critical_values[2]
    
    return result
```

---

## 6. Production RAG Systems

### 6.1 Production RAG Architecture

**Scalable Multi-Layer Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Request Router & Load Balancer             │
│  - Route to appropriate RAG pipeline                    │
│  - Handle scaling & failover                           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│         RAG Orchestration & Caching Layer              │
│  - Query result cache (L1)                             │
│  - Decompose into retrieval tasks                      │
│  - Parallel batch processing                           │
└────┬─────────────────────┬─────────────────────┬────────┘
     │                     │                     │
┌────▼────────┐   ┌────────▼─────────┐  ┌──────▼─────┐
│  Dense       │   │  Sparse          │  │  Metadata  │
│  Retriever   │   │  Retriever       │  │  Filter    │
│  (HNSW)      │   │  (BM25/SPLADE)   │  │  Service   │
├─────────────┤   ├──────────────────┤  ├────────────┤
│ Vector DB   │   │ Search Engine    │  │ Rules DB   │
│(Qdrant/     │   │(Elasticsearch/   │  │            │
│ Milvus)     │   │ Opensearch)      │  │            │
└────┬────────┘   └────┬─────────────┘  └──────┬─────┘
     │                 │                       │
     └─────────────────┼───────────────────────┘
                       │
                ┌──────▼──────────┐
                │  Result Fusion  │
                │  & Reranking    │
                └──────┬──────────┘
                       │
                ┌──────▼──────────────────┐
                │ Cross-Encoder Reranker  │
                │ (Inference Service)     │
                └──────┬──────────────────┘
                       │
                ┌──────▼──────────────────┐
                │ Context Window Manager  │
                │ & Optimization          │
                └──────┬──────────────────┘
                       │
                ┌──────▼──────────────────┐
                │ LLM Generation          │
                │ (Inference API)         │
                └──────┬──────────────────┘
                       │
                ┌──────▼──────────────────┐
                │ Post-Generation         │
                │ - Validation            │
                │ - Fact-checking         │
                │ - Context grounding     │
                └──────┬──────────────────┘
                       │
                ┌──────▼──────────────────┐
                │ Monitoring & Logging    │
                │ - Latency tracking      │
                │ - Quality metrics       │
                │ - Cost attribution      │
                └──────────────────────────┘
```

### 6.2 Caching & Optimization Strategies

**Three-Tier Cache Hierarchy:**

```python
class RAGCachingStrategy:
    def __init__(self):
        # L1: Query Result Cache (exact match, short TTL)
        self.l1_cache = LRUCache(max_size=10000, ttl_seconds=3600)
        
        # L2: Chunk Cache (semantic hash, medium TTL)
        self.l2_cache = SemanticHashCache(
            max_size=100000,
            ttl_seconds=86400,  # 24 hours
            similarity_threshold=0.98
        )
        
        # L3: Vector Index Cache (OS page cache, implicit)
        # Configure via OS settings
    
    def get_or_compute(self, query: str, retriever_fn) -> list:
        # L1: Exact query match
        cache_key = hash_query(query)
        if cache_key in self.l1_cache:
            self.metrics.cache_hit('l1')
            return self.l1_cache[cache_key]
        
        # L2: Semantic similar queries
        similar_query_key = self.find_semantic_similar(query)
        if similar_query_key in self.l2_cache:
            self.metrics.cache_hit('l2')
            cached_results = self.l2_cache[similar_query_key]
            # Re-rank if needed, but avoid re-retrieval
            return re_rank_cached(cached_results, query)
        
        # Cache miss: Retrieve
        results = retriever_fn(query)
        
        # Cache for future
        self.l1_cache[cache_key] = results
        self.l2_cache[similar_query_key] = results
        self.metrics.cache_miss()
        
        return results
```

**Query Caching Patterns:**

1. **Exact Match Caching**
   - Cache entry: hash(query) → results
   - TTL: 1 hour (user preferences change)
   - Hit rate: 10-30%

2. **Semantic Bucketing**
   - Cluster similar queries
   - Share cache entry across cluster
   - TTL: 24 hours
   - Hit rate: 20-40%

3. **Fragment Caching**
   - Cache sub-query results
   - Compose for multi-hop queries
   - TTL: 24-48 hours
   - Hit rate: 30-50% on multi-hop

**Cache Invalidation:**

```python
class CacheInvalidationStrategy:
    def __init__(self):
        self.document_version_map = {}  # doc_id -> version
        self.query_doc_mapping = defaultdict(set)  # query_hash -> [doc_ids]
    
    def invalidate_on_document_update(self, doc_id: str):
        """
        When document changes, invalidate all queries using it
        """
        # Increment version
        self.document_version_map[doc_id] = (
            self.document_version_map.get(doc_id, 0) + 1
        )
        
        # Find all cached queries using this document
        queries_to_invalidate = [
            q for q, docs in self.query_doc_mapping.items()
            if doc_id in docs
        ]
        
        # Invalidate from caches
        for query_hash in queries_to_invalidate:
            self.l1_cache.invalidate(query_hash)
            self.l2_cache.invalidate(query_hash)
        
        return len(queries_to_invalidate)
    
    def batch_invalidate(self, doc_ids: list[str], invalidation_type: str = 'full'):
        """
        Efficient batch invalidation
        """
        if invalidation_type == 'full':
            # Clear entire cache (expensive but safe)
            self.l1_cache.clear()
            self.l2_cache.clear()
        elif invalidation_type == 'selective':
            # Only invalidate affected queries
            for doc_id in doc_ids:
                self.invalidate_on_document_update(doc_id)
```

### 6.3 Scalability Patterns

**Horizontal Scaling:**

```
Problem: Single RAG instance bottleneck
Solution: Distributed retrieval & generation

┌─────────────────────────────────────────┐
│       Load Balancer (API Gateway)       │
├─────────────────────────────────────────┤
│  Routes:                                │
│  - Retrieval-heavy queries → Retrieval  │
│  - Generation-heavy queries → LLM Pool  │
│  - Mixed → Distributed                  │
└────┬──────────────────────┬─────────────┘
     │                      │
┌────▼──────────────┐  ┌───▼─────────────────┐
│ Retrieval Cluster │  │ LLM Generation Pool │
│  (Replicated)     │  │  (Multiple instances)
├───────────────────┤  ├─────────────────────┤
│ Instance 1        │  │ GPT-4o instance     │
│ Instance 2        │  │ Qwen instance       │
│ Instance 3        │  │ Claude instance     │
│                   │  │ Backup instance     │
└───────────────────┘  └─────────────────────┘
```

**Shared Infrastructure (Stateless Design):**

```python
class ScalableRAGPipeline:
    """
    Stateless, horizontally scalable RAG
    """
    def __init__(self, config):
        # Shared services (single instance each)
        self.vector_db = qdrant_client.QdrantClient(
            host="vector-db-service.prod.svc",  # K8s service
            port=6333
        )
        self.search_engine = elasticsearch_client(
            hosts=["search-1.prod.svc", "search-2.prod.svc", "search-3.prod.svc"]
        )
        self.redis_cache = redis.Redis(
            host="redis-cluster.prod.svc",
            port=6379
        )
        
        # Load balanced services
        self.reranker_pool = pool_of_inference_servers([
            "reranker-1.prod.svc:8000",
            "reranker-2.prod.svc:8000",
            "reranker-3.prod.svc:8000"
        ])
        
        self.llm_pool = pool_of_llm_services([
            LLMServiceClient("llm-1.prod.svc:5000"),
            LLMServiceClient("llm-2.prod.svc:5000"),
            LLMServiceClient("llm-3.prod.svc:5000")
        ])
    
    async def answer_query(self, query: str) -> str:
        """
        Stateless query handling - can run on any instance
        """
        # Retrieve
        dense_results = await self.vector_db.search_async(query)
        sparse_results = await self.search_engine.search_async(query)
        fused = rrf_fusion([dense_results, sparse_results])
        
        # Rerank (parallel across pool)
        reranked = await self.reranker_pool.rerank(query, fused[:20])
        
        # Generate (parallel across LLM pool)
        final_answer = await self.llm_pool.generate(query, reranked[:5])
        
        return final_answer
```

### 6.4 Cost Optimization for RAG

**Bottleneck Analysis:**

```
Typical RAG Cost Breakdown:

LLM Inference (Generation)    55% (most expensive)
├─ Token usage for answer     30%
├─ Long context overhead      15%
├─ Multiple retries           10%

Vector Database              15%
├─ Embedding computation      8%
├─ Search queries             5%
├─ Storage                    2%

Reranker Inference            15%
├─ Cross-encoder scoring      12%
├─ Recompute on cache miss    3%

Infrastructure & Overhead    15%
├─ Compute instances          8%
├─ Caching systems            4%
├─ Monitoring                 3%
```

**Optimization Strategies:**

```python
class CostOptimizedRAG:
    def __init__(self):
        self.query_complexity_classifier = QueryComplexityModel()
        self.cost_budget = {'low_priority': $0.05, 'normal': $0.20, 'high': $0.50}
    
    def route_by_cost(self, query: str, user_tier: str) -> ExecutionPlan:
        """
        Route query to appropriate cost tier
        """
        # Classify query complexity
        complexity = self.query_complexity_classifier.predict(query)
        
        execution_plan = ExecutionPlan()
        
        if complexity == 'simple':
            # Simple: Use fast, cheap LLM
            execution_plan.llm_choice = 'fast-cheap'  # Smaller model
            execution_plan.reranking = False  # Skip expensive reranking
            execution_plan.context_budget = 1024  # tokens
        
        elif complexity == 'moderate':
            # Moderate: Balanced approach
            execution_plan.llm_choice = 'balanced'
            execution_plan.reranking = True
            execution_plan.context_budget = 2048
        
        else:  # 'complex'
            # Complex: Use best model
            execution_plan.llm_choice = 'best'
            execution_plan.reranking = True
            execution_plan.context_budget = 4096
        
        # Adjust for user tier
        if user_tier == 'premium':
            execution_plan.llm_choice = max(
                execution_plan.llm_choice, 'best'
            )
        
        return execution_plan
```

**Token Reduction Techniques:**

1. **Prompt Compression**
   ```python
   def compress_prompt(full_prompt: str, target_tokens: int = 100) -> str:
       """
       Extract key information, discard redundancy
       """
       # Use LLM to identify critical parts
       compression_prompt = f"""
       Compress this prompt to {target_tokens} tokens while preserving:
       - Core question/intent
       - Key constraints
       - Critical examples
       
       Original prompt ({count_tokens(full_prompt)} tokens):
       {full_prompt}
       
       Compressed:
       """
       
       compressed = llm.generate(compression_prompt, max_tokens=target_tokens)
       return compressed
   ```

2. **Context Pruning**
   ```python
   def prune_low_value_context(
       context_chunks: list[str],
       query: str,
       importance_threshold: float = 0.5
   ) -> list[str]:
       """
       Remove chunks that minimally contribute to answer
       """
       # Score each chunk's importance
       scores = []
       for chunk in context_chunks:
           # Factor 1: Relevance to query
           relevance = cosine_similarity(
               embed_model.encode([query]),
               embed_model.encode([chunk])
           )[0][0]
           
           # Factor 2: Information density
           # (use entropy or TF-IDF)
           density = calculate_information_density(chunk)
           
           # Factor 3: Early in search ranking
           # (assume later chunks less important)
           position_score = 0.5  # Adjust based on position
           
           combined_score = (
               0.5 * relevance +
               0.3 * density +
               0.2 * position_score
           )
           scores.append(combined_score)
       
       # Keep only above-threshold
       pruned = [
           chunk for chunk, score in zip(context_chunks, scores)
           if score >= importance_threshold
       ]
       
       return pruned
   ```

### 6.5 Monitoring & Observability for RAG

**Key Metrics:**

```python
class RAGMonitoring:
    """
    Production monitoring for RAG systems
    """
    
    # 1. Retrieval Quality Metrics
    retrieval_metrics = {
        'recall@k': 'What % of relevant docs are in top-k?',
        'mrr': 'Mean reciprocal rank of first relevant',
        'ndcg@10': 'Normalized discounted cumulative gain',
        'mean_avg_precision': 'MAP across queries',
        'hit_rate@k': '% of queries with >= 1 relevant doc'
    }
    
    # 2. Generation Quality Metrics
    generation_metrics = {
        'faithfulness': 'Does answer cite retrieved context?',
        'relevance': 'Is answer relevant to query?',
        'coherence': 'Is answer well-structured?',
        'completeness': 'Does answer fully address question?',
        'hallucination_rate': '% of answers with false info'
    }
    
    # 3. Performance Metrics
    performance_metrics = {
        'retrieval_latency_p50': '50th percentile latency',
        'retrieval_latency_p99': '99th percentile latency',
        'reranking_latency': 'Cross-encoder inference time',
        'llm_latency': 'Generation latency',
        'total_latency': 'End-to-end latency',
        'throughput': 'Queries per second'
    }
    
    # 4. Cost Metrics
    cost_metrics = {
        'cost_per_query': 'Total cost (LLM + retrieval + rerank)',
        'token_efficiency': 'Answer tokens / input tokens',
        'cache_hit_rate': '% queries served from cache',
        'llm_cost_per_query': 'LLM tokens × rate'
    }
    
    # 5. System Health
    health_metrics = {
        'vector_db_health': 'Uptime, query success rate',
        'search_engine_health': 'Query latency, error rate',
        'cache_hit_rate': 'L1, L2 cache hit rates',
        'error_rate': '% of queries returning errors',
        'retry_rate': 'How often queries retry'
    }
```

**Implementation with Prometheus:**

```python
from prometheus_client import Counter, Histogram, Gauge

class RAGMetricsCollector:
    def __init__(self):
        # Counter: Total queries, total tokens, total errors
        self.queries_total = Counter(
            'rag_queries_total',
            'Total queries processed',
            ['status']  # 'success', 'error', 'timeout'
        )
        
        # Histogram: Latency buckets
        self.retrieval_latency = Histogram(
            'rag_retrieval_latency_seconds',
            'Retrieval latency',
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        # Gauge: Current cache size
        self.cache_size = Gauge(
            'rag_cache_size_bytes',
            'Current cache size',
            ['cache_level']
        )
        
        # Histogram: Retrieval quality
        self.mrr = Histogram(
            'rag_mrr',
            'Mean reciprocal rank',
            buckets=(0.1, 0.25, 0.5, 0.75, 1.0)
        )
    
    def record_query(
        self,
        query: str,
        retrieval_latency: float,
        llm_latency: float,
        mrr: float,
        cache_hit: bool,
        tokens_used: int,
        cost: float
    ):
        """Record metrics for single query"""
        self.queries_total.labels(status='success').inc()
        self.retrieval_latency.observe(retrieval_latency)
        self.mrr.observe(mrr)
        # ... etc
```

---

## 7. Production Deployment & Integration

### 7.1 RAG Framework Integration

**Popular Frameworks & Ecosystem:**

| Framework | Best For | Key Strength |
|-----------|----------|--------------|
| LangChain | General RAG | Excellent ecosystem integrations |
| LlamaIndex | Document indexing | Structure-aware document processing |
| Haystack | NLP pipelines | Pipeline composition |
| RAGAS | RAG evaluation | Comprehensive evaluation framework |
| DSPy | Prompt optimization | Algorithmic prompt learning |
| AutoGPT/Agentic RAG | Complex agents | Multi-step reasoning |

**Sample Integration (LangChain with Hybrid Search):**

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialize retrievers
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Qdrant.from_documents(
    docs,
    embeddings,
    url="http://qdrant:6333",
    collection_name="hybrid_docs"
)

dense_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10

# 2. Ensemble retriever with RRF
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]
)

# 3. Generation with RAG
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:
""")

chain = (
    {"context": ensemble_retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
)

# 4. Execute
answer = chain.invoke("How do I optimize database queries?")
```

---

## 8. Code Examples & Templates

### 8.1 Complete Hybrid RAG System

```python
# complete_hybrid_rag.py

from typing import Optional, List
import asyncio
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float
    source: str  # 'dense', 'sparse', 'fused'

class DenseRetriever:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.doc_ids = []
    
    def index_documents(self, docs: List[str], doc_ids: Optional[List[str]] = None):
        """Index documents for dense search"""
        self.documents = docs
        self.doc_ids = doc_ids or [f"doc_{i}" for i in range(len(docs))]
        
        # Embed and index
        embeddings = self.embedding_model.encode(docs, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Search using dense retrieval"""
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append(RetrievalResult(
                doc_id=self.doc_ids[idx],
                text=self.documents[idx],
                score=float(score),
                source='dense'
            ))
        
        return results

class SparseRetriever:
    def __init__(self):
        self.bm25_model = None
        self.documents = []
        self.doc_ids = []
    
    def index_documents(self, docs: List[str], doc_ids: Optional[List[str]] = None):
        """Index documents for sparse search"""
        self.documents = docs
        self.doc_ids = doc_ids or [f"doc_{i}" for i in range(len(docs))]
        
        # Tokenize and index
        tokenized = [doc.lower().split() for doc in docs]
        self.bm25_model = BM25Okapi(tokenized)
    
    def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Search using BM25"""
        tokens = query.lower().split()
        scores = self.bm25_model.get_scores(tokens)
        
        # Sort by score and get top-k
        top_indices = np.argsort(-scores)[:k]
        
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                doc_id=self.doc_ids[idx],
                text=self.documents[idx],
                score=float(scores[idx]),
                source='sparse'
            ))
        
        return results

class HybridRAG:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.dense = DenseRetriever(embedding_model)
        self.sparse = SparseRetriever()
        self.reranker = CrossEncoder(reranker_model)
    
    def index_documents(self, docs: List[str], doc_ids: Optional[List[str]] = None):
        """Index documents in both retrievers"""
        self.dense.index_documents(docs, doc_ids)
        self.sparse.index_documents(docs, doc_ids)
    
    def rrf_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        k: int = 60
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion"""
        rrf_scores = {}
        
        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                if result.doc_id not in rrf_scores:
                    rrf_scores[result.doc_id] = {'doc': result, 'score': 0.0}
                rrf_scores[result.doc_id]['score'] += 1.0 / (k + rank)
        
        # Sort by RRF score
        fused = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['doc'] for item in fused]
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        fusion_method: str = 'rrf'
    ) -> List[RetrievalResult]:
        """Hybrid retrieval with optional reranking"""
        # Get results from both retrievers
        dense_results = self.dense.search(query, k=k)
        sparse_results = self.sparse.search(query, k=k)
        
        # Fuse results
        if fusion_method == 'rrf':
            fused = self.rrf_fusion([dense_results, sparse_results], k=60)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Mark fused results
        for result in fused:
            result.source = 'fused'
        
        return fused[:k]
    
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        k: int = 5
    ) -> List[RetrievalResult]:
        """Rerank using cross-encoder"""
        # Prepare pairs
        pairs = [[query, result.text] for result in candidates]
        
        # Score
        scores = self.reranker.predict(pairs)
        
        # Sort by rerank score
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Update results
        reranked = []
        for result, score in ranked[:k]:
            result.score = float(score)
            reranked.append(result)
        
        return reranked

# Usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "PostgreSQL uses B-tree indexes for efficient query execution.",
        "Database optimization involves proper indexing strategies.",
        "ECONNREFUSED error occurs when the connection is refused.",
        "Connection pooling improves database performance.",
        "Query optimization reduces latency significantly.",
    ]
    
    # Initialize RAG
    rag = HybridRAG()
    rag.index_documents(documents)
    
    # Query
    query = "How to optimize database queries?"
    
    # Retrieve
    retrieved = rag.retrieve(query, k=5)
    print("Retrieved results:")
    for result in retrieved:
        print(f"  [{result.score:.4f}] {result.text[:60]}...")
    
    # Rerank
    reranked = rag.rerank(query, retrieved, k=3)
    print("\nReranked results:")
    for result in reranked:
        print(f"  [{result.score:.4f}] {result.text[:60]}...")
```

---

## References & Research Papers

### Key Papers on Advanced RAG

1. **Multi-Hop Question Answering**
   - "Reasoning in Trees: Improving Retrieval-Augmented Generation for Multi-Hop Question Answering" (2026)
     - Authors: Shi et al., Shanghai Jiao Tong University
     - Key innovation: Consensus-based tree decomposition, bottom-up synthesis
     - Benchmark gains: +7.0% F1, +6.0% EM across HotpotQA, MuSiQue, 2WikiMQA
   - "Retrieve, Summarize, Plan: Advancing Multi-hop Question Answering with an Iterative Approach" (2024)
   - "GRITHopper: Decomposition-Free Multi-Hop Dense Retrieval" (2025)

2. **Hybrid Search & Fusion**
   - "Hybrid Search for RAG: BM25, SPLADE, and Vector Search Combined" (2026)
     - Comprehensive comparison of fusion methods: RRF, convex combination, DBSF
     - Benchmark: 26-31% NDCG improvement on BEIR
   - "Dense vs Sparse Retrieval: Mastering FAISS, BM25, and Hybrid Search" (2025)
   - "Reciprocal Rank Fusion" (Cormack et al., SIGIR 2009)

3. **Corrective RAG & Self-Correction**
   - "Corrective RAG: The Missing Layer Between Smart and Trustworthy AI" (2026)
   - "Building Self-Corrective Agentic RAG Systems" (2026)
   - "Agentic RAG Pipelines: Complete Guide" (2026)

4. **Knowledge Graphs in RAG**
   - "Injecting Knowledge Graphs in Different RAG Stages" (Medium, 2024)
   - "Harry Potter and the Self-Learning Knowledge Graph RAG" (Demo, 2024)

### Vector Databases & Indexing

5. **BEIR Benchmark**
   - Thakur et al., NeurIPS 2021
   - Standard benchmark for retrieval evaluation
   - 15+ datasets across domains

6. **Approximate Nearest Neighbor Algorithms**
   - HNSW (Hierarchical Navigable Small World)
   - IVF (Inverted File Index)
   - Research on trade-offs between recall and latency

### Synthetic Data Generation

7. **Data Generation from LLMs**
   - Knowledge distillation techniques
   - Contrastive learning with synthetic negatives
   - Self-supervised learning on synthetic data

8. **Data Quality Assessment**
   - Statistical validation frameworks
   - Duplicate detection (exact, fuzzy, semantic)
   - Bias detection in synthetic data

---

## Recommended Learning Path

**Phase 1: Foundations (Week 1-2)**
- Understand basic RAG pipeline
- Study embedding models and vector similarity
- Learn BM25 ranking algorithm

**Phase 2: Hybrid Search (Week 3-4)**
- Implement hybrid search with RRF
- Benchmark against baselines
- Fine-tune fusion parameters

**Phase 3: Advanced Patterns (Week 5-8)**
- Multi-hop retrieval with query decomposition
- Knowledge graph integration
- Corrective RAG loops

**Phase 4: Production (Week 9-12)**
- Implement caching strategies
- Set up monitoring and observability
- Optimize for cost and latency
- Deploy to production infrastructure

---

## Conclusion

Advanced RAG techniques represent a significant shift from simple retrieval toward intelligent, reasoning-based systems. The combination of hybrid search, structured decomposition, and corrective loops addresses fundamental limitations of basic RAG.

**Key Takeaways:**
1. Hybrid search (BM25 + dense) provides 18-31% improvement through fusion
2. Consensus-based tree decomposition reduces multi-hop error by 7-13%
3. Cross-encoder reranking improves precision by 8-12% with modest latency increase
4. Caching at multiple levels (query, chunk, index) provides 10-50% efficiency gains
5. Production RAG requires integrated monitoring, cost optimization, and graceful degradation

The field is rapidly evolving. Recent architectures increasingly emphasize:
- Agentic RAG with self-correction loops
- Structured reasoning with explicit decomposition
- Cost-aware retrieval routing
- Retrieval as a learned component (not just retrieval then generation)

Success in production RAG systems depends on:
- Understanding your domain's vocabulary mismatch patterns
- Building robust evaluation frameworks before deployment
- Iterating on fusion parameters with your actual data
- Monitoring both performance and cost metrics continuously

