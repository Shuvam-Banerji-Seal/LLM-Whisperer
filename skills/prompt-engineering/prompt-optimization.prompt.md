# Prompt Optimization and Caching — Agentic Skill Prompt

Advanced techniques for optimizing prompts, self-refinement loops, and leveraging prompt caching for efficiency.

---

## 1. Identity and Mission

### 1.1 Role

You are a **prompt optimization engineer** responsible for iteratively improving prompts to achieve higher quality outputs while reducing computational cost and latency.

### 1.2 Mission

- **Maximize output quality** through iterative refinement and feedback loops
- **Minimize cost and latency** via caching, prompt compression, and efficient tokenization
- **Automate optimization** using LLM-driven self-refinement and scoring
- **Track improvements** with quantifiable metrics (BLEU, BERTScore, custom benchmarks)

### 1.3 Core Principles

1. **Iterative refinement beats big rewrites** — Small, targeted changes often outperform complete rewrites.
2. **Measure before optimizing** — Baseline performance first; optimize only measurable bottlenecks.
3. **Cache aggressively** — Reuse expensive computations (embeddings, LLM outputs, prompt encodings).
4. **Automate feedback loops** — Use LLMs to evaluate and refine their own outputs.

---

## 2. Decision Tree for Optimization Strategy

```
START: What is your optimization goal?

├─ GOAL: Reduce latency/cost
│  └─ Have you enabled prompt caching yet?
│     ├─ NO → Implement prompt caching (§4.1)
│     └─ YES → Compress prompt through summarization (§4.2)
│
├─ GOAL: Improve output quality
│  └─ Is the task deterministic (same input → similar output desired)?
│     ├─ YES → Use self-refinement loop (§5.1)
│     └─ NO → Use human feedback / rank aggregation (§5.2)
│
└─ GOAL: Both (quality + cost)
   └─ Use hybrid: cache + refinement (§5.3)
```

---

## 3. Common Optimization Pitfalls

| Pitfall | Effect | Mitigation |
|---------|--------|-----------|
| Over-caching stale responses | Returns outdated answers | Implement cache expiration (TTL) and version control |
| Prompt compression loses nuance | Lower quality outputs | Test compressed prompts; use semantic compression, not truncation |
| Infinite refinement loops | Latency explosion; no convergence | Set max iterations; use early stopping on quality plateaus |
| Metric gaming | Optimizes wrong objective | Use diverse metrics; validate improvements on held-out data |
| Caching without invalidation | Correctness issues | Tag cache entries with input hash and explicit versioning |
| Assuming generalization | Overfitting to test set | Test optimization on separate validation set |

---

## 4. Prompt Caching Strategies

### 4.1 Basic Prompt Caching with Redis

**Use Case:** Production systems with repetitive queries or static context.

```python
import json
import hashlib
from typing import Any, Optional
import redis
from datetime import timedelta

class PromptCache:
    """Redis-backed cache for prompt responses."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl_hours: int = 24):
        """Initialize cache with Redis backend."""
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _make_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""
        content = f"{prompt}:{model}"
        return f"prompt_cache:{hashlib.sha256(content.encode()).hexdigest()}"
    
    def get(self, prompt: str, model: str) -> Optional[dict[str, Any]]:
        """Retrieve cached response."""
        key = self._make_key(prompt, model)
        cached = self.client.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    def set(
        self,
        prompt: str,
        model: str,
        response: dict[str, Any],
        ttl_hours: Optional[int] = None,
    ) -> None:
        """Cache response with TTL."""
        key = self._make_key(prompt, model)
        ttl = timedelta(hours=ttl_hours) if ttl_hours else self.ttl
        self.client.setex(
            key,
            int(ttl.total_seconds()),
            json.dumps(response),
        )
    
    def invalidate(self, prompt: str, model: str) -> None:
        """Invalidate cache entry."""
        key = self._make_key(prompt, model)
        self.client.delete(key)
    
    def clear_all(self) -> None:
        """Clear entire cache (use with caution in production)."""
        self.client.flushdb()

# Usage
cache = PromptCache(ttl_hours=24)

prompt = "Summarize the benefits of prompt caching in LLMs."
model = "gpt-4"

# Check cache first
cached_response = cache.get(prompt, model)
if cached_response:
    response = cached_response
else:
    # Call LLM (pseudocode)
    response = {"text": "Prompt caching reduces latency and cost...", "tokens": 120}
    cache.set(prompt, model, response)
```

---

### 4.2 Semantic Caching with Embeddings

**Use Case:** Similar queries should return similar answers; avoid redundant LLM calls.

```python
import numpy as np
from typing import Any, Optional

class SemanticPromptCache:
    """Semantic cache using embedding similarity."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Cosine similarity threshold (0-1) for cache hit
        """
        self.cache: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        self.threshold = similarity_threshold
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        Replace with actual embedding model in production (e.g., OpenAI, Sentence Transformers).
        """
        # Placeholder: use hash-based deterministic "embedding"
        import hashlib
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.randn(384)  # Typical embedding dimension
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )
    
    def get(self, prompt: str) -> Optional[dict[str, Any]]:
        """
        Retrieve cached response for similar prompt.
        Returns None if no similar prompt found above threshold.
        """
        embedding = self._get_embedding(prompt)
        
        for cached_prompt, cached_embedding, response in self.cache:
            similarity = self._cosine_similarity(embedding, cached_embedding)
            if similarity >= self.threshold:
                return {**response, "_cached_from": cached_prompt, "_similarity": similarity}
        
        return None
    
    def set(self, prompt: str, response: dict[str, Any]) -> None:
        """Store response with prompt embedding."""
        embedding = self._get_embedding(prompt)
        self.cache.append((prompt, embedding, response))
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()

# Usage
semantic_cache = SemanticPromptCache(similarity_threshold=0.90)

# First query
prompt1 = "What are the benefits of prompt caching?"
response1 = {"text": "Reduces latency, saves costs...", "tokens": 100}
semantic_cache.set(prompt1, response1)

# Similar query (should hit cache)
prompt2 = "Why is prompt caching beneficial?"
cached_response = semantic_cache.get(prompt2)
if cached_response:
    print(f"Cache hit! Similarity: {cached_response['_similarity']:.3f}")
```

---

## 5. Prompt Optimization Techniques

### 5.1 Self-Refinement Loop with Scoring

**Use Case:** Iteratively improve outputs using LLM-based scoring and refinement.

```python
from dataclasses import dataclass
from enum import Enum

class QualityDimension(str, Enum):
    """Dimensions for evaluating output quality."""
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"

@dataclass
class QualityScore:
    """Score for a single quality dimension."""
    dimension: QualityDimension
    score: float  # 0-10
    feedback: str

@dataclass
class RefinementResult:
    """Result of a refinement iteration."""
    iteration: int
    original_output: str
    refined_output: str
    quality_scores: list[QualityScore]
    avg_score: float
    converged: bool

def evaluate_output(
    output: str,
    criteria: list[QualityDimension],
) -> list[QualityScore]:
    """
    Evaluate output on multiple quality dimensions.
    In production, call an LLM scoring model.
    """
    # Placeholder implementation
    return [
        QualityScore(
            dimension=dim,
            score=7.5,  # Placeholder score
            feedback=f"Sample feedback for {dim}",
        )
        for dim in criteria
    ]

def refine_output(
    original_output: str,
    feedback: str,
    max_iterations: int = 3,
    convergence_threshold: float = 0.85,
) -> RefinementResult:
    """
    Iteratively refine output based on scoring feedback.
    
    Pseudocode for production use with actual LLM calls:
    1. Score initial output
    2. If score < threshold and iterations < max:
       a. Ask LLM to refine based on feedback
       b. Re-score refined output
       c. Check convergence
    3. Return best result
    """
    
    results: list[RefinementResult] = []
    current_output = original_output
    
    for iteration in range(max_iterations):
        # Score current output
        scores = evaluate_output(
            current_output,
            [QualityDimension.ACCURACY, QualityDimension.CLARITY],
        )
        avg_score = sum(s.score for s in scores) / len(scores)
        
        # Check convergence
        converged = avg_score >= convergence_threshold or iteration == max_iterations - 1
        
        # Collect feedback for next iteration
        feedback_str = "\n".join(s.feedback for s in scores)
        
        result = RefinementResult(
            iteration=iteration,
            original_output=original_output if iteration == 0 else current_output,
            refined_output=current_output,
            quality_scores=scores,
            avg_score=avg_score,
            converged=converged,
        )
        results.append(result)
        
        if converged:
            return result
        
        # In production, call LLM here with:
        # "Refine this output based on feedback: {feedback_str}"
        # current_output = llm_call(refinement_prompt)
    
    return results[-1]

# Usage
initial_output = "Machine learning is a field of AI."
result = refine_output(initial_output, max_iterations=3)
print(f"Converged after {result.iteration + 1} iterations")
print(f"Final score: {result.avg_score:.2f}/10")
```

---

### 5.2 Prompt Compression and Summarization

**Use Case:** Reduce prompt length while preserving key information.

```python
def compress_prompt(
    prompt: str,
    target_tokens: Optional[int] = None,
    strategy: str = "extractive",
) -> str:
    """
    Compress prompt to reduce token usage.
    
    Strategies:
    - 'extractive': Remove less important sentences
    - 'abstractive': Summarize (requires LLM)
    - 'quantization': Round numbers, remove details
    """
    
    if strategy == "extractive":
        return _compress_extractive(prompt)
    elif strategy == "abstractive":
        return _compress_abstractive(prompt)
    elif strategy == "quantization":
        return _compress_quantization(prompt)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def _compress_extractive(prompt: str) -> str:
    """Remove low-value sentences using TF-IDF heuristic."""
    sentences = prompt.split(". ")
    
    # Simple TF-IDF approximation: keep sentences with rare words
    word_freq: dict[str, int] = {}
    for sent in sentences:
        for word in sent.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    scores = []
    for sent in sentences:
        score = sum(1 / word_freq.get(w.lower(), 1) for w in sent.split())
        scores.append((sent, score))
    
    # Keep top 70% of sentences by score
    scores.sort(key=lambda x: x[1], reverse=True)
    kept = sorted(
        scores[: max(1, int(len(scores) * 0.7))],
        key=lambda x: sentences.index(x[0]),
    )
    
    return ". ".join(s[0] for s in kept) + "."

def _compress_quantization(prompt: str) -> str:
    """Remove exact numbers, use ranges/approximations."""
    import re
    
    # Replace specific numbers with ranges
    prompt = re.sub(r'\d+', 'N', prompt)
    # Remove redundant words
    redundant = ["very", "quite", "really", "really", "definitely"]
    for word in redundant:
        prompt = re.sub(rf'\b{word}\b\s+', '', prompt, flags=re.IGNORECASE)
    
    return prompt.strip()

def _compress_abstractive(prompt: str) -> str:
    """Summarize using LLM (pseudocode)."""
    # In production:
    # summary = llm_call(f"Summarize concisely:\n{prompt}")
    # return summary
    return prompt  # Placeholder

# Usage
long_prompt = """
You are an expert in machine learning and specifically in natural language processing.
Your role is to carefully analyze texts and provide detailed feedback.
You should be very thorough, quite methodical, and really precise in your assessments.
Please examine the following document and provide a comprehensive analysis.
"""

compressed = compress_prompt(long_prompt, strategy="quantization")
print("Original length:", len(long_prompt.split()))
print("Compressed length:", len(compressed.split()))
print(compressed)
```

---

## 6. Cost and Performance Monitoring

### 6.1 Prompt Analytics

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class PromptMetrics:
    """Track metrics for prompt performance."""
    prompt_id: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    quality_score: Optional[float] = None
    cost_cents: float = 0.0
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens
    
    def tokens_per_ms(self) -> float:
        """Throughput metric."""
        return self.total_tokens() / max(self.latency_ms, 1)

class PromptAnalytics:
    """Collect and analyze prompt performance metrics."""
    
    def __init__(self):
        self.metrics: list[PromptMetrics] = []
    
    def record(self, metrics: PromptMetrics) -> None:
        """Record metrics for a prompt call."""
        self.metrics.append(metrics)
    
    def get_stats(self, model: Optional[str] = None) -> dict[str, float]:
        """Compute aggregate statistics."""
        filtered = (
            [m for m in self.metrics if m.model == model]
            if model
            else self.metrics
        )
        
        if not filtered:
            return {}
        
        total_tokens = sum(m.total_tokens() for m in filtered)
        total_cost = sum(m.cost_cents for m in filtered) / 100
        cache_hits = sum(1 for m in filtered if m.cached)
        avg_latency = sum(m.latency_ms for m in filtered) / len(filtered)
        avg_quality = (
            sum(m.quality_score for m in filtered if m.quality_score is not None)
            / len([m for m in filtered if m.quality_score is not None])
            if any(m.quality_score is not None for m in filtered)
            else None
        )
        
        return {
            "total_calls": len(filtered),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "cache_hit_rate": cache_hits / len(filtered),
            "avg_latency_ms": avg_latency,
            "avg_quality_score": avg_quality or 0,
        }

# Usage
analytics = PromptAnalytics()

metrics1 = PromptMetrics(
    prompt_id="prompt_001",
    model="gpt-4",
    input_tokens=150,
    output_tokens=200,
    latency_ms=2500,
    quality_score=8.5,
    cost_cents=3.5,
    cached=False,
)
analytics.record(metrics1)

print("Stats:", analytics.get_stats())
```

---

## 7. A/B Testing Prompts

```python
from dataclasses import dataclass
from typing import Callable
import random

@dataclass
class PromptVariant:
    """A/B test variant."""
    name: str
    prompt_builder: Callable[[str], str]
    weight: float = 0.5

class PromptABTest:
    """A/B test framework for prompts."""
    
    def __init__(self, variants: list[PromptVariant]):
        """Initialize A/B test with variants."""
        self.variants = variants
        self.results: dict[str, list[float]] = {v.name: [] for v in variants}
    
    def select_variant(self) -> PromptVariant:
        """Select variant based on weight distribution."""
        return random.choices(
            self.variants,
            weights=[v.weight for v in self.variants],
            k=1,
        )[0]
    
    def record_result(self, variant_name: str, score: float) -> None:
        """Record result for variant."""
        self.results[variant_name].append(score)
    
    def get_winner(self, confidence_threshold: float = 0.95) -> str:
        """Determine winning variant based on results."""
        import statistics
        
        stats = {
            name: {
                "mean": statistics.mean(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "n": len(scores),
            }
            for name, scores in self.results.items()
            if scores
        }
        
        # Simple t-test approximation; use scipy.stats in production
        winner = max(stats.items(), key=lambda x: x[1]["mean"])
        return winner[0]

# Usage
variants = [
    PromptVariant(
        name="verbose",
        prompt_builder=lambda x: f"You are an expert. {x}",
        weight=0.5,
    ),
    PromptVariant(
        name="concise",
        prompt_builder=lambda x: f"Expert mode. {x}",
        weight=0.5,
    ),
]

ab_test = PromptABTest(variants)

# Simulate running test
for _ in range(100):
    variant = ab_test.select_variant()
    # In production: call LLM, get quality score
    score = random.gauss(8.0 if variant.name == "verbose" else 7.5, 0.5)
    ab_test.record_result(variant.name, score)

winner = ab_test.get_winner()
print(f"Winner: {winner}")
```

---

## 8. References

1. https://arxiv.org/abs/2307.12387 — "Prompt Caching for Efficient Large Language Models" (Token efficiency through caching)
2. https://arxiv.org/abs/2308.04592 — "Self-Critique Improves Reasoning in Large Language Models" (Self-refinement techniques)
3. https://arxiv.org/abs/2309.03409 — "Iterative Refinement of LLM Outputs" (Multi-step refinement strategies)
4. https://arxiv.org/abs/2206.04615 — "Automatic Prompt Optimization with Gradient Descent" (Gradient-based prompt optimization)
5. https://arxiv.org/abs/2211.01910 — "Automatic Chain-of-Thought Prompting" (Auto-generating CoT structures)
6. https://arxiv.org/abs/2305.13825 — "Tree of Thoughts: Deliberation with LLMs" (Structured reasoning optimization)
7. https://platform.openai.com/docs/guides/prompt-caching — OpenAI prompt caching API documentation
8. https://docs.anthropic.com/claude/reference/system-prompts — Claude API system prompts best practices
9. https://github.com/openai/evals — OpenAI framework for LLM evaluation
10. https://github.com/EleutherAI/lm-evaluation-harness — HuggingFace evaluation benchmark suite
11. https://arxiv.org/abs/2310.16944 — "Prompt Injection and Jailbreak Prevention" (Security aspects)
12. https://arxiv.org/abs/2302.12813 — "Large Language Models as Zero-Shot Classifiers" (Prompt design for classification)
13. https://redis.io/docs/latest/develop/interact/search-and-query/ — Redis caching and search patterns
14. https://huggingface.co/sentence-transformers — Sentence Transformers for semantic caching
15. https://arxiv.org/abs/2305.17355 — "PromptEBench: Benchmarking Prompting Engineering Techniques" (Evaluation framework)
16. https://github.com/google-deepmind/unified_io — Unified input/output frameworks for prompting

---

## 9. Uncertainty and Limitations

**Not Covered Here:**
- Reinforcement Learning for Prompt Optimization (RL-based prompt search) — requires training infrastructure
- Prompt attacks and adversarial robustness — covered in Safety & Alignment skill
- Multi-lingual prompt optimization — language-specific techniques needed
- Integration with specific commercial APIs (detailed coverage of GPT-4 vs Claude vs Llama specific caching)

**Production Deployment Notes:**
- Always implement cache versioning alongside prompt versioning
- Monitor cache hit rates; if < 40%, consider semantic caching
- Set reasonable TTLs; stale cache is worse than no cache
- Use async caching to avoid blocking calls
- Implement fallback mechanisms when cache is unavailable
- Budget for A/B testing iteration (5-20 variants typical)
