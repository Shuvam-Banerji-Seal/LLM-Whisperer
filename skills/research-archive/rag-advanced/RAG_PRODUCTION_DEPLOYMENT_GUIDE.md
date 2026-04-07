# RAG Production Deployment & Optimization Playbook

**Version:** 2026  
**Last Updated:** April 2026  
**Audience:** ML Engineers, LLM Architects, DevOps

---

## 1. Pre-Deployment Checklist

### Retrieval Layer
- [ ] Hybrid search implemented with both dense and sparse retrievers
- [ ] Fusion method evaluated (RRF vs. convex combination vs. DBSF)
- [ ] Re-ranking with cross-encoder validated on eval set
- [ ] Chunking strategy optimized for your document types
- [ ] Vector index created with appropriate parameters (HNSW, IVF)
- [ ] Sparse index built (BM25 or SPLADE)
- [ ] Batch retrieval pipeline for efficiency
- [ ] Query caching infrastructure ready

### Data Quality
- [ ] Evaluation dataset created (50+ labeled query-doc pairs minimum)
- [ ] Synthetic data filtered for quality
- [ ] Near-duplicate detection run
- [ ] Bias analysis completed
- [ ] Document versioning strategy implemented

### Generation Layer
- [ ] LLM API rate limits understood
- [ ] Token counting accurate
- [ ] Prompt template validated
- [ ] Fallback generation strategies defined
- [ ] Hallucination detection implemented

### Infrastructure
- [ ] Vector database deployment (Qdrant, Weaviate, Milvus, etc.)
- [ ] Search engine deployment (Elasticsearch, Opensearch, etc.)
- [ ] Redis cache layer for L1/L2 caching
- [ ] LLM inference service (vLLM, TGI, proprietary API)
- [ ] Reranker inference service
- [ ] Load balancer and API gateway configured

### Monitoring & Observability
- [ ] Metrics collection implemented (Prometheus/CloudWatch)
- [ ] Dashboards created for key metrics
- [ ] Alerting rules configured
- [ ] Logging aggregation setup (ELK, Loki, etc.)
- [ ] Tracing configured (Jaeger, Tempo)
- [ ] Cost tracking enabled
- [ ] Error rate monitoring

### Testing
- [ ] Unit tests for retrieval components
- [ ] Integration tests for end-to-end pipeline
- [ ] Load testing completed (target QPS achieved)
- [ ] Latency benchmarks established
- [ ] Cost analysis completed
- [ ] Failover scenarios tested

---

## 2. Eval Set Creation & Benchmark

### Minimal Eval Set (50 examples)

```python
def create_minimal_eval_set(
    corpus: List[str],
    llm: LLMClient,
    num_examples: int = 50
) -> List[dict]:
    """
    Quickly create evaluation set from corpus
    Fast path: ~2-3 hours for 50 examples
    """
    eval_set = []
    
    for doc_idx in range(0, len(corpus), len(corpus) // num_examples):
        doc = corpus[doc_idx]
        
        # Generate questions about this doc
        prompt = f"""
        Generate 2-3 factual questions about this text that would require
        understanding of the content. Include expected answers.
        
        Text:
        {doc}
        
        Format:
        Q1: [question]
        A1: [answer]
        Q2: ...
        """
        
        response = llm.generate(prompt)
        qa_pairs = parse_qa_pairs(response)
        
        for q, a in qa_pairs:
            eval_set.append({
                'query': q,
                'relevant_docs': [doc],  # Ground truth
                'expected_answer': a,
                'source': 'generated'
            })
    
    return eval_set[:num_examples]

# Usage
eval_set = create_minimal_eval_set(corpus, llm, num_examples=50)
# Manually review and correct: ~30 minutes
# Use for baseline evaluation
```

### Production Eval Set (500+ examples)

```python
class ProductionEvalSetBuilder:
    """
    Build large, diverse evaluation set
    Time: ~2-3 weeks with human review
    """
    
    def build_from_user_queries(self, user_queries_log: List[str]):
        """
        Approach 1: Use real user queries
        - Pros: True distribution of user needs
        - Cons: May not cover all document aspects
        """
        eval_examples = []
        
        for query in user_queries_log:
            # Retrieve documents
            docs = self.retriever.retrieve(query, k=5)
            
            # Human labels: which docs are relevant?
            # Use interface like Prodigy or Label Studio
            annotations = self.labeling_service.get_annotations(query)
            
            eval_examples.append({
                'query': query,
                'relevant_doc_ids': annotations['relevant_ids'],
                'num_relevant': len(annotations['relevant_ids']),
                'source': 'user_queries'
            })
        
        return eval_examples
    
    def build_from_document_sampling(self, documents: List[str]):
        """
        Approach 2: Systematic document sampling
        - Pros: Balanced coverage of all docs
        - Cons: May not match real query distribution
        """
        eval_examples = []
        
        # Stratified sampling by document length
        length_bins = defaultdict(list)
        for doc in documents:
            length = len(doc.split())
            bin_size = 500
            bin_idx = length // bin_size
            length_bins[bin_idx].append(doc)
        
        for bin_idx, docs in length_bins.items():
            # Sample from each bin
            sampled = random.sample(docs, min(10, len(docs)))
            
            for doc in sampled:
                # Generate questions
                questions = self.llm.generate_questions(doc)
                
                for q in questions:
                    eval_examples.append({
                        'query': q,
                        'relevant_docs': [doc],
                        'source': 'generated_from_docs'
                    })
        
        return eval_examples
    
    def build_multi_hop_subset(self):
        """
        Approach 3: Explicit multi-hop queries
        - Critical for evaluating complex RAG systems
        """
        eval_examples = []
        
        # Generate multi-hop questions linking documents
        for doc1, doc2 in combinations(self.documents, 2):
            # Does relationship exist between these docs?
            has_relationship = self.llm.check_relationship(doc1, doc2)
            
            if has_relationship:
                # Generate multi-hop question
                question = self.llm.generate_multi_hop_question(doc1, doc2)
                
                eval_examples.append({
                    'query': question,
                    'relevant_docs': [doc1, doc2],
                    'num_hops': 2,
                    'source': 'multi_hop'
                })
        
        return eval_examples
```

### Eval Metrics Implementation

```python
class RAGEvaluator:
    def __init__(self, eval_set: List[dict]):
        self.eval_set = eval_set
        self.retriever = None
        self.generator = None
    
    def evaluate_retrieval(self, retriever) -> dict:
        """
        Evaluate retrieval quality
        """
        self.retriever = retriever
        results = {
            'hit_rate@5': [],
            'hit_rate@10': [],
            'mrr': [],
            'ndcg@10': [],
        }
        
        for example in self.eval_set:
            retrieved = retriever.retrieve(example['query'], k=10)
            retrieved_ids = {r.doc_id for r in retrieved}
            relevant_ids = set(example['relevant_doc_ids'])
            
            # Hit rate @5
            hit_at_5 = any(
                rid in relevant_ids
                for rid, _ in retrieved[:5]
            )
            results['hit_rate@5'].append(hit_at_5)
            
            # Hit rate @10
            hit_at_10 = any(
                rid in relevant_ids
                for rid, _ in retrieved
            )
            results['hit_rate@10'].append(hit_at_10)
            
            # MRR
            mrr = 0.0
            for rank, (rid, _) in enumerate(retrieved, start=1):
                if rid in relevant_ids:
                    mrr = 1.0 / rank
                    break
            results['mrr'].append(mrr)
            
            # NDCG@10
            ndcg = self.compute_ndcg(retrieved_ids, relevant_ids, k=10)
            results['ndcg@10'].append(ndcg)
        
        # Average metrics
        return {
            k: np.mean(v) for k, v in results.items()
        }
    
    def evaluate_generation(self, generator) -> dict:
        """
        Evaluate generation quality against retrieved context
        """
        self.generator = generator
        results = {
            'faithfulness': [],
            'relevance': [],
            'contains_hallucination': []
        }
        
        for example in self.eval_set:
            # Retrieve
            context = self.retriever.retrieve(
                example['query'], k=5
            )
            context_text = "\n".join(r.text for r in context)
            
            # Generate
            answer = generator.generate(
                query=example['query'],
                context=context_text
            )
            
            # Evaluate
            # Faithfulness: answer grounded in context?
            faithfulness = self.evaluate_faithfulness(answer, context_text)
            results['faithfulness'].append(faithfulness)
            
            # Relevance: answer relevant to query?
            relevance = self.evaluate_relevance(answer, example['query'])
            results['relevance'].append(relevance)
            
            # Hallucination check
            has_hallucination = self.check_hallucination(answer, context_text)
            results['contains_hallucination'].append(has_hallucination)
        
        return {
            'faithfulness': np.mean(results['faithfulness']),
            'relevance': np.mean(results['relevance']),
            'hallucination_rate': np.mean(results['contains_hallucination'])
        }
    
    def compute_ndcg(self, retrieved_ids, relevant_ids, k=10):
        """NDCG computation"""
        # Truncate to k
        retrieved_ids = list(retrieved_ids)[:k]
        
        # Compute DCG
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                dcg += 1.0 / np.log2(i + 1)
        
        # Compute IDCG
        ideal_relevant_count = min(len(relevant_ids), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_relevant_count + 1))
        
        # NDCG
        return dcg / idcg if idcg > 0 else 0.0
```

---

## 3. Latency Optimization

### Profiling & Bottleneck Analysis

```python
class RAGProfiler:
    def profile_full_pipeline(self, query: str, num_runs: int = 10) -> dict:
        """
        Break down latency across pipeline stages
        """
        timings = defaultdict(list)
        
        for _ in range(num_runs):
            # Stage 1: Dense retrieval
            start = time.time()
            dense_results = self.dense_retriever.search(query, k=10)
            timings['dense_retrieval'].append(time.time() - start)
            
            # Stage 2: Sparse retrieval (parallel)
            start = time.time()
            sparse_results = self.sparse_retriever.search(query, k=10)
            timings['sparse_retrieval'].append(time.time() - start)
            
            # Stage 3: Fusion
            start = time.time()
            fused = self.fusion_fn([dense_results, sparse_results])
            timings['fusion'].append(time.time() - start)
            
            # Stage 4: Reranking
            start = time.time()
            reranked = self.reranker.rerank(query, fused[:20])
            timings['reranking'].append(time.time() - start)
            
            # Stage 5: LLM generation
            start = time.time()
            answer = self.llm.generate(query, reranked[:5])
            timings['generation'].append(time.time() - start)
        
        # Summarize
        summary = {}
        total_time = 0
        for stage, times in timings.items():
            avg_time = np.mean(times)
            summary[stage] = {
                'mean_ms': avg_time * 1000,
                'p95_ms': np.percentile(times, 95) * 1000,
                'p99_ms': np.percentile(times, 99) * 1000,
                'percentage': avg_time / sum(sum(times) for times in timings.values())
            }
            total_time += avg_time
        
        return summary
```

### Latency Optimization Techniques

```python
class LatencyOptimizer:
    
    def parallelize_retrieval(self):
        """
        Run dense + sparse retrieval in parallel
        Cost: ~0ms overhead, benefit: retrieval_latency
        """
        async def parallel_retrieve(query, k):
            dense_task = asyncio.create_task(
                self.dense_retriever.search_async(query, k)
            )
            sparse_task = asyncio.create_task(
                self.sparse_retriever.search_async(query, k)
            )
            
            dense_results, sparse_results = await asyncio.gather(
                dense_task,
                sparse_task
            )
            
            return [dense_results, sparse_results]
        
        # Latency improvement: ~40-50% (serial to parallel)
    
    def batch_embedding_computation(self, queries: List[str]):
        """
        Batch embedding computation for efficiency
        Cost: Increased memory, benefit: better throughput
        """
        # Instead of embedding queries one-by-one:
        # Batch 32-64 queries together
        embeddings = self.embedding_model.encode(
            queries,
            batch_size=64,
            show_progress_bar=False
        )
        
        # Latency improvement: ~30% for batch of 64
    
    def adaptive_reranking_depth(self, query: str, num_candidates: int):
        """
        Use query complexity to decide reranking depth
        Simple query: rerank top-5
        Complex query: rerank top-20
        """
        complexity = self.classify_complexity(query)
        
        if complexity == 'simple':
            rerank_k = 5
        elif complexity == 'moderate':
            rerank_k = 10
        else:
            rerank_k = 20
        
        # Latency improvement: ~20-30% for simple queries
    
    def early_stopping_generation(self):
        """
        Stop LLM generation early if answer is complete
        Monitor generation for answer completion signal
        """
        # Generation parameters:
        max_tokens = 500
        stop_sequences = ['\n\nQ:', 'Q:', 'Human:']
        
        # Latency improvement: ~15-25% on average
```

---

## 4. Cost Optimization

### Cost Breakdown & Attribution

```python
class CostAnalyzer:
    def __init__(self, pricing_config):
        self.pricing = pricing_config  # Per-token rates, API costs
    
    def analyze_query_cost(self, query: str, execution_log: dict) -> dict:
        """
        Break down cost for single query
        
        execution_log = {
            'query_tokens': 50,
            'embedding_model': 'text-embedding-3-small',
            'retrieved_docs': 5,
            'rerank_docs': 5,
            'generation_tokens': 200,
            'llm_model': 'gpt-4o',
            'cache_hit': False
        }
        """
        cost = {}
        
        # Embedding cost
        cost['embedding'] = (
            execution_log['query_tokens'] * 
            self.pricing['embedding']['per_1k_tokens']
        ) / 1000
        
        # Vector search (fixed per query)
        cost['vector_search'] = self.pricing['vector_db']['per_query']
        
        # Reranking cost
        rerank_token_count = (
            execution_log['query_tokens'] * execution_log['rerank_docs']
        )
        cost['reranking'] = (
            rerank_token_count * 
            self.pricing['reranker']['per_1k_tokens']
        ) / 1000
        
        # LLM cost
        llm = execution_log['llm_model']
        input_tokens = execution_log['query_tokens'] + 200  # Assume ~200 context tokens
        output_tokens = execution_log['generation_tokens']
        
        cost['llm_input'] = (
            input_tokens * 
            self.pricing['llm'][llm]['input_per_1k']
        ) / 1000
        
        cost['llm_output'] = (
            output_tokens * 
            self.pricing['llm'][llm]['output_per_1k']
        ) / 1000
        
        # Apply cache discount if hit
        if execution_log['cache_hit']:
            # Cache hit: skip vector search, embedding
            cost['vector_search'] = 0
            cost['embedding'] = 0
            cost['cache_lookup'] = 0.0001  # Minimal
        
        cost['total'] = sum(cost.values())
        
        return cost
    
    def optimize_for_cost(self, target_cost_per_query: float):
        """
        Strategies to reduce cost to target
        """
        strategies = {
            'reduce_rerank_depth': {
                'description': 'Rerank only top-10 instead of top-20',
                'cost_reduction_percent': 20,
                'quality_impact': 'Minor precision loss'
            },
            'use_cheaper_llm': {
                'description': 'Switch from GPT-4o to GPT-4o-mini',
                'cost_reduction_percent': 60,
                'quality_impact': 'Moderate quality loss'
            },
            'increase_cache_ttl': {
                'description': 'Extend cache TTL from 1h to 24h',
                'cost_reduction_percent': 30,
                'quality_impact': 'Slight staleness'
            },
            'context_compression': {
                'description': 'Compress context with summarization',
                'cost_reduction_percent': 25,
                'quality_impact': 'Potential info loss'
            },
            'batch_processing': {
                'description': 'Batch 10 queries for reranking',
                'cost_reduction_percent': 15,
                'quality_impact': 'None (latency increase)'
            }
        }
        
        return strategies
```

---

## 5. Monitoring Dashboard Configuration

### Prometheus Metrics

```yaml
# prometheus-rag.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-system'
    static_configs:
      - targets: ['localhost:8000']

# Alerting rules
groups:
  - name: rag_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, rag_retrieval_latency_seconds) > 2
        for: 5m
        annotations:
          summary: "RAG retrieval latency exceeded 2s"
      
      - alert: LowHitRate
        expr: rag_hit_rate < 0.8
        for: 10m
        annotations:
          summary: "RAG hit rate below 80%"
      
      - alert: HighCostPerQuery
        expr: rag_cost_per_query > 0.50
        for: 5m
        annotations:
          summary: "Cost per query exceeded threshold"
      
      - alert: VectorDBDown
        expr: up{job="vector-db"} == 0
        for: 1m
        annotations:
          summary: "Vector database is down"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "RAG System Monitoring",
    "panels": [
      {
        "title": "Query Latency (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rag_query_latency_seconds)",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rag_query_latency_seconds)",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rag_query_latency_seconds)",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(rag_cache_hits_total[5m]) / rate(rag_cache_queries_total[5m])",
            "legendFormat": "Hit rate"
          }
        ]
      },
      {
        "title": "Cost Per Query",
        "targets": [
          {
            "expr": "increase(rag_total_cost_cents[1h]) / increase(rag_queries_total[1h])",
            "legendFormat": "Cost"
          }
        ]
      },
      {
        "title": "Retrieval Hit Rate by Retriever",
        "targets": [
          {
            "expr": "rag_hit_rate{retriever='hybrid'}",
            "legendFormat": "Hybrid"
          },
          {
            "expr": "rag_hit_rate{retriever='dense'}",
            "legendFormat": "Dense"
          },
          {
            "expr": "rag_hit_rate{retriever='sparse'}",
            "legendFormat": "Sparse"
          }
        ]
      }
    ]
  }
}
```

---

## 6. Troubleshooting Guide

### Common Issues & Solutions

| Issue | Symptom | Root Cause | Solution |
|-------|---------|-----------|----------|
| **Low Hit Rate** | Relevant docs not retrieved | Poor embedding model or chunking | 1. Evaluate embedding model on eval set<br>2. Adjust chunk size<br>3. Use hybrid search instead of dense only |
| **High Latency** | p99 > 2 seconds | LLM generation or reranking bottleneck | 1. Profile each stage<br>2. Use cheaper LLM for simple queries<br>3. Reduce rerank depth<br>4. Implement parallel retrieval |
| **High Cost** | Cost per query > $1 | Using expensive LLM or large context | 1. Switch to cheaper model variant<br>2. Compress context<br>3. Reduce retrieved context size<br>4. Increase cache effectiveness |
| **Hallucinations** | Model generates false info | Insufficient retrieval or grounding | 1. Implement faithfulness check<br>2. Increase retrieval top-k<br>3. Add fallback to admitting uncertainty<br>4. Implement fact-checking |
| **Degraded Performance After Update** | Quality drops on redeployment | Vector index mismatch or model change | 1. Compare embeddings before/after<br>2. Reindex documents<br>3. A/B test before full rollout |
| **Vector DB OOM** | Out of memory errors | Index too large for available RAM | 1. Implement IVF-PQ compression<br>2. Use sparse vectors<br>3. Shard across multiple instances |

---

## 7. Deployment Checklist Summary

- [ ] **Retrieval**: Hybrid search, reranking, chunking optimized
- [ ] **Data**: Eval set created (500+ examples), quality validated
- [ ] **Infrastructure**: Vector DB, search engine, LLM service, monitoring
- [ ] **Caching**: Multi-tier cache with invalidation strategy
- [ ] **Monitoring**: Dashboards, alerts, cost tracking
- [ ] **Testing**: Load tests passed, latency acceptable, cost within budget
- [ ] **Documentation**: Runbooks for common issues, disaster recovery
- [ ] **Gradual Rollout**: Start at 5% traffic, monitor for 48h, scale to 100%

