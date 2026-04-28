# Cost Optimization for LLM Inference

## Problem Statement

LLM inference is expensive. GPT-4-class models can cost several dollars per million tokens, and even smaller models accumulate substantial costs at scale. A production system serving millions of requests daily can easily spend tens of thousands of dollars monthly on inference alone. Unlike traditional software where costs scale linearly with usage, LLM costs depend on prompt length, completion length, model size, and quality requirements, creating complex optimization challenges.

The challenge is that cost optimization cannot come at the expense of output quality or user experience. Teams must balance competing objectives: faster inference reduces compute costs but may require smaller models; aggressive compression reduces token counts but risks losing important context; caching reduces redundant computation but adds infrastructure complexity and memory costs.

This skill covers understanding the cost structure of LLM inference, implementing multi-level optimization strategies, measuring and monitoring cost efficiency, and making informed trade-offs between cost, latency, and quality.

## Theory & Fundamentals

### Cost Structure of LLM Inference

LLM inference costs derive from several components:

**Compute Costs (GPU)**:
- Prefill phase: Proportional to prompt token count, primarily matrix-vector operations
- Decode phase: Proportional to generated token count, requires full matrix-matrix operations per token
- Memory bandwidth: Dominated by loading model weights for each forward pass
- VRAM usage: Model weights (proportional to parameters), KV cache, activations

**Token Costs (API-based models)**:
- Input tokens: Priced per 1000 tokens, typically 1/3 to 1/2 of output pricing
- Output tokens: Priced per 1000 tokens, often higher to reflect generation value
- Context window limits: May incur premium for longer contexts

**Infrastructure Costs**:
- Serving infrastructure: GPU instances, auto-scaling overhead
- Networking: Data transfer, API call overhead
- Storage: Model weights, KV cache, logs
- Monitoring/observability: Metrics collection, logging storage

### Cost-Performance Trade-off Frontier

```
Optimization Strategy Trade-offs:

Strategy                    | Cost Reduction | Quality Impact | Latency Impact
---------------------------|----------------|----------------|---------------
Smaller model (distillation)| 50-80%         | 5-20% decline  | 2-5x faster
Quantization (INT8)         | 40-60%         | 1-5% decline   | 10-30% faster
Quantization (INT4)         | 60-75%         | 5-15% decline  | 15-40% faster
Prompt caching              | 30-70%         | None           | Near-instant
Batch processing            | 40-60%         | None           | Variable
Speculative decoding        | 0-20%          | None           | 2-4x faster
Early exiting               | 20-40%         | Variable       | 2-3x faster
```

### Key Metrics for Cost Optimization

```python
Cost Metrics:
├── Cost per Request = (compute_cost + token_cost + infra_cost) / requests
├── Cost per Successful Request = total_cost / successful_requests
├── Cost per Token = total_cost / total_tokens_processed
├── Token Efficiency = useful_tokens / total_tokens
├── Cache Hit Rate = cached_requests / total_requests
└── Quality-Adjusted Cost = cost / (quality_score * throughput)

Performance Metrics:
├── Latency p50/p95/p99 (ms)
├── Throughput (tokens/second)
├── GPU Utilization (%)
├── Memory Utilization (%)
└── Queue Depth
```

### Theoretical Foundations

**Batching Efficiency**: The KV cache allows processing multiple sequences in parallel. The efficiency gain from batching follows:

$$E_{batch} = \frac{\text{Time}_{sequential}}{\text{Time}_{batch}} \approx \frac{N \cdot (T_{prefill} + T_{decode})}{\max(T_{prefill}, T_{decode}) + (N-1) \cdot \min(T_{prefill}, T_{decode})}$$

**Quantization Error Impact**: Quantization from FP16 to INT8 introduces error. The signal-to-quantization-noise ratio (SQNR):

$$SQNR = 10 \log_{10}\left(\frac{P_{signal}}{P_{noise}}\right) \approx 6.02 \cdot B + 4.77 - 10 \log_{10}(\sigma^2_w)$$

where B is bit depth and $\sigma^2_w$ is weight variance.

## Implementation Patterns

### Pattern 1: Token Budget Manager with Adaptive Context Truncation

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

@dataclass
class TokenBudget:
    max_tokens: int
    reserved_output_tokens: int
    available_input_tokens: int
    truncation_strategy: str = "last"

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0

class AdaptiveTokenBudgetManager:
    """
    Manages token budgets with intelligent truncation and prioritization.
    """
    
    def __init__(
        self,
        model_max_tokens: int,
        default_reserved_output: int = 500,
        cost_per_1k_input: float = 0.001,
        cost_per_1k_output: float = 0.002
    ):
        self.model_max_tokens = model_max_tokens
        self.default_reserved_output = default_reserved_output
        self.cost_input = cost_per_1k_input
        self.cost_output = cost_per_1k_output
        
        self.context_history: Dict[str, List[TokenUsage]] = {}
        self.priority_tokens: Dict[str, str] = {}
    
    def calculate_budget(
        self,
        prompt: str,
        user_priority: Optional[str] = None,
        requested_max_tokens: Optional[int] = None,
        context_messages: Optional[List[Dict]] = None
    ) -> TokenBudget:
        """
        Calculate optimal token budget given constraints.
        """
        reserved = requested_max_tokens or self.default_reserved_output
        available = self.model_max_tokens - reserved
        
        base_prompt_tokens = self._estimate_tokens(prompt)
        
        context_tokens = 0
        if context_messages:
            context_tokens = sum(self._estimate_tokens(m.get("content", "")) 
                                for m in context_messages)
        
        if base_prompt_tokens + context_tokens > available:
            truncated = self._truncate_to_fit(
                prompt=prompt,
                context_messages=context_messages,
                max_tokens=available,
                strategy=self._select_truncation_strategy(
                    base_prompt_tokens, context_tokens, available
                )
            )
            final_prompt_tokens = self._estimate_tokens(truncated)
        else:
            final_prompt_tokens = base_prompt_tokens + context_tokens
        
        return TokenBudget(
            max_tokens=self.model_max_tokens,
            reserved_output_tokens=reserved,
            available_input_tokens=available,
            truncation_strategy="smart"
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using word-based approximation."""
        words = len(text.split())
        return int(words * 1.3)
    
    def _select_truncation_strategy(
        self,
        prompt_tokens: int,
        context_tokens: int,
        available: int
    ) -> str:
        """Select optimal truncation strategy based on token distribution."""
        if context_tokens > prompt_tokens * 2:
            return "context_first"
        elif prompt_tokens > available * 0.7:
            return "prompt_last"
        else:
            return "balanced"
    
    def _truncate_to_fit(
        self,
        prompt: str,
        context_messages: Optional[List[Dict]],
        max_tokens: int,
        strategy: str
    ) -> str:
        """Truncate content to fit within token budget."""
        if strategy == "context_first" and context_messages:
            truncated_context = self._truncate_context(context_messages, max_tokens // 2)
            remaining = max_tokens - self._estimate_tokens(str(truncated_context))
            truncated_prompt = self._truncate_prompt(prompt, remaining)
            return str(truncated_context) + "\n" + truncated_prompt
        
        elif strategy == "prompt_last":
            truncated_prompt = self._truncate_prompt(prompt, max_tokens)
            return truncated_prompt
        
        else:
            half = max_tokens // 2
            truncated_prompt = self._truncate_prompt(prompt, half)
            remaining = max_tokens - self._estimate_tokens(truncated_prompt)
            if context_messages:
                truncated_context = self._truncate_context(context_messages, remaining)
                return str(truncated_context) + "\n" + truncated_prompt
            return truncated_prompt
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """Truncate prompt to fit within token budget."""
        words = prompt.split()
        current_tokens = self._estimate_tokens(prompt)
        
        if current_tokens <= max_tokens:
            return prompt
        
        target_words = int(len(words) * (max_tokens / current_tokens))
        return " ".join(words[:target_words])
    
    def _truncate_context(
        self,
        messages: List[Dict],
        max_tokens: int
    ) -> List[Dict]:
        """Truncate conversation history to fit budget."""
        result = []
        current_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = self._estimate_tokens(msg.get("content", ""))
            if current_tokens + msg_tokens <= max_tokens:
                result.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return result
    
    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0
    ) -> Dict[str, float]:
        """Calculate cost for a request."""
        input_cost = (prompt_tokens / 1000) * self.cost_input
        output_cost = (completion_tokens / 1000) * self.cost_output
        
        cached_savings = (cached_tokens / 1000) * self.cost_input * 0.9
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cached_savings": cached_savings,
            "total_cost": input_cost + output_cost - cached_savings
        }
```

### Pattern 2: Multi-Model Routing with Cost-Aware Selection

```python
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import asyncio
import numpy as np

@dataclass
class ModelSpec:
    name: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    latency_estimate_ms: float
    quality_score: float
    max_tokens: int
    capabilities: List[str] = field(default_factory=list)

class CostAwareModelRouter:
    """
    Routes requests to appropriate models based on task requirements,
    cost constraints, and quality needs.
    """
    
    def __init__(
        self,
        models: List[ModelSpec],
        quality_evaluator: Callable
    ):
        self.models = {m.name: m for m in models}
        self.evaluate_quality = quality_evaluator
        self.usage_stats: Dict[str, List[float]] = {m.name: [] for m in models}
    
    async def route(
        self,
        prompt: str,
        task_requirements: Dict,
        cost_budget: Optional[float] = None,
        latency_budget_ms: Optional[float] = None
    ) -> ModelSpec:
        """
        Select optimal model based on task requirements and constraints.
        """
        candidates = self._filter_candidates(task_requirements)
        
        if not candidates:
            raise ValueError("No models meet task requirements")
        
        scored = []
        for model in candidates:
            score = self._calculate_selection_score(
                model, prompt, task_requirements, 
                cost_budget, latency_budget_ms
            )
            scored.append((model, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        selected = scored[0][0]
        
        if len(self.usage_stats[selected.name]) > 100:
            self.usage_stats[selected.name].pop(0)
        self.usage_stats[selected.name].append(scored[0][1])
        
        return selected
    
    def _filter_candidates(self, requirements: Dict) -> List[ModelSpec]:
        """Filter models that meet basic requirements."""
        candidates = []
        for model in self.models.values():
            if requirements.get("max_tokens") and model.max_tokens < requirements["max_tokens"]:
                continue
            
            required_capabilities = requirements.get("capabilities", [])
            if required_capabilities and not all(
                cap in model.capabilities for cap in required_capabilities
            ):
                continue
            
            candidates.append(model)
        
        return candidates
    
    def _calculate_selection_score(
        self,
        model: ModelSpec,
        prompt: str,
        requirements: Dict,
        cost_budget: Optional[float],
        latency_budget: Optional[float]
    ) -> float:
        """
        Calculate selection score balancing cost, latency, and quality.
        """
        quality_needed = requirements.get("min_quality", 0.5)
        
        if model.quality_score < quality_needed:
            return -float('inf')
        
        estimated_tokens = self._estimate_tokens(prompt)
        estimated_cost = self._estimate_cost(model, estimated_tokens)
        
        if cost_budget and estimated_cost > cost_budget:
            return -float('inf')
        
        if latency_budget and model.latency_estimate_ms > latency_budget:
            return -float('inf')
        
        cost_score = 1.0 / (1.0 + estimated_cost)
        quality_score = model.quality_score
        
        cost_weight = requirements.get("cost_weight", 0.3)
        quality_weight = requirements.get("quality_weight", 0.7)
        
        combined_score = (
            cost_weight * cost_score +
            quality_weight * quality_score
        )
        
        return combined_score
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return int(len(text.split()) * 1.3)
    
    def _estimate_cost(
        self,
        model: ModelSpec,
        input_tokens: int,
        output_tokens: int = 100
    ) -> float:
        """Estimate request cost."""
        input_cost = (input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model.cost_per_1k_output
        return input_cost + output_cost
    
    def get_routing_advice(self, task_type: str) -> str:
        """Get routing advice for common task types."""
        advice = {
            "simple_qa": "Use small/fast model. Quality variance acceptable.",
            "code_generation": "Use high-quality model. Latency secondary to correctness.",
            "creative_writing": "Use best available model. Cost less important than quality.",
            "summarization": "Use medium model. Balance cost and quality.",
            "extraction": "Use task-specific fine-tuned model if available."
        }
        return advice.get(task_type, "Evaluate on per-request basis")
```

### Pattern 3: Intelligent Prompt Compression

```python
from typing import Dict, List, Optional, Tuple
import re
from collections import Counter

class IntelligentPromptCompressor:
    """
    Compresses prompts while preserving essential information for the task.
    Uses multiple strategies and combines them intelligently.
    """
    
    def __init__(
        self,
        llm_client,
        min_compression_ratio: float = 0.5,
        preserve_patterns: List[str] = None
    ):
        self.llm = llm_client
        self.min_compression_ratio = min_compression_ratio
        self.preserve_patterns = preserve_patterns or [
            r'\d{3}-\d{2}-\d{4}',  # SSN
            r'\$\d+',  # Money
            r'[A-Z]{2,}\d+',  # Codes
        ]
    
    async def compress(
        self,
        prompt: str,
        task_type: str,
        target_tokens: Optional[int] = None
    ) -> Tuple[str, Dict]:
        """
        Compress prompt while preserving task-critical information.
        """
        original_tokens = self._count_tokens(prompt)
        
        steps = []
        
        step1 = await self._remove_redundancy(prompt)
        step1_tokens = self._count_tokens(step1)
        steps.append({"action": "remove_redundancy", "tokens": step1_tokens})
        
        step2 = self._compress_whitespace(step1)
        step2_tokens = self._count_tokens(step2)
        steps.append({"action": "compress_whitespace", "tokens": step2_tokens})
        
        step3 = await self._semantic_compress(step2, task_type)
        step3_tokens = self._count_tokens(step3)
        steps.append({"action": "semantic_compress", "tokens": step3_tokens})
        
        step4 = self._final_cleanup(step3)
        
        if target_tokens and self._count_tokens(step4) > target_tokens:
            step4 = self._aggressive_truncate(step4, target_tokens)
        
        compression_ratio = self._count_tokens(step4) / max(original_tokens, 1)
        
        return step4, {
            "original_tokens": original_tokens,
            "compressed_tokens": self._count_tokens(step4),
            "compression_ratio": compression_ratio,
            "steps": steps
        }
    
    async def _remove_redundancy(self, text: str) -> str:
        """Remove redundant phrases and filler."""
        redundant_patterns = [
            r'Please\s+',
            r'Kindly\s+',
            r'I\s+would\s+like\s+you\s+to\s+',
            r'Can\s+you\s+(?:please\s+)?',
            r'(?:As\s+an\s+AI|I'm\s+an\s+AI|As\s+a\s+language\s+model)[^.]*\.\s*',
            r'(?:Please\s+note|Note)\s+that\s+',
            r'Furthermore\s+',
            r'In\s+addition\s+to\s+this\s+',
            r'It\s+is\s+important\s+to\s+note\s+that\s+',
        ]
        
        result = text
        for pattern in redundant_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        sentences = result.split('.')
        unique_sentences = []
        seen = set()
        
        for sent in sentences:
            sent_lower = sent.lower().strip()
            if sent_lower and sent_lower not in seen:
                unique_sentences.append(sent)
                seen.add(sent_lower)
        
        return '.'.join(unique_sentences)
    
    def _compress_whitespace(self, text: str) -> str:
        """Compress whitespace."""
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    async def _semantic_compress(self, text: str, task_type: str) -> str:
        """Use LLM to compress while preserving meaning."""
        compression_prompts = {
            "extraction": "Extract only the key entities and values needed for extraction. Remove all narrative.",
            "classification": "Keep only the text to classify and its category if specified. Remove explanations.",
            "qa": "Keep only the question. Remove context that isn't directly relevant to answering.",
            "summarization": "Keep the core message. Remove supporting details and examples.",
            "code_generation": "Keep only the programming task. Remove commentary and documentation."
        }
        
        instruction = compression_prompts.get(task_type, 
            "Compress this text by removing unnecessary words while keeping the core meaning.")
        
        prompt = f"""{instruction}

Original text:
{text}

Compressed (keep essential meaning only):
"""
        
        response = await self.llm.generate(
            prompt=prompt,
            parameters={"temperature": 0.3, "max_tokens": 500}
        )
        
        return response.text.strip()
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup passes."""
        text = self._compress_whitespace(text)
        text = re.sub(r'\[([^\]]+)\]\[\1\]', r'[\1]', text)
        
        return text
    
    def _aggressive_truncate(self, text: str, max_tokens: int) -> str:
        """When all else fails, truncate to fit budget."""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        
        return ' '.join(words[:int(max_tokens * 1.3)])
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using word approximation."""
        return int(len(text.split()) * 1.3)
```

### Pattern 4: Caching Layer with Semantic Deduplication

```python
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import numpy as np

@dataclass
class CacheEntry:
    prompt_hash: str
    response: str
    token_count: int
    created_at: datetime
    access_count: int
    last_accessed: datetime
    model_name: str

class SemanticCache:
    """
    Caches LLM responses with semantic similarity matching.
    """
    
    def __init__(
        self,
        embedding_model,
        similarity_threshold: float = 0.95,
        ttl_hours: int = 24,
        max_entries: int = 100000
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)
        self.max_entries = max_entries
        
        self.exact_cache: Dict[str, CacheEntry] = {}
        self.semantic_cache: List[Tuple[str, np.ndarray]] = []
        self.embedding_index: Dict[str, int] = {}
    
    def _get_cache_key(self, prompt: str, model_name: str) -> str:
        """Generate cache key."""
        content = f"{model_name}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(
        self,
        prompt: str,
        model_name: str,
        parameters: Dict = None
    ) -> Optional[str]:
        """Retrieve cached response if available."""
        cache_key = self._get_cache_key(prompt, model_name)
        
        if cache_key in self.exact_cache:
            entry = self.exact_cache[cache_key]
            if self._is_valid(entry):
                entry.access_count += 1
                entry.last_accessed = datetime.utcnow()
                return entry.response
            else:
                del self.exact_cache[cache_key]
        
        embedding = self.embedding_model.encode(prompt)
        
        for cached_prompt, cached_embedding in self.semantic_cache:
            similarity = self._cosine_similarity(embedding, cached_embedding)
            if similarity >= self.similarity_threshold:
                cache_key = self._get_cache_key(cached_prompt, model_name)
                if cache_key in self.exact_cache:
                    entry = self.exact_cache[cache_key]
                    if self._is_valid(entry):
                        entry.access_count += 1
                        entry.last_accessed = datetime.utcnow()
                        return entry.response
        
        return None
    
    def put(
        self,
        prompt: str,
        response: str,
        model_name: str,
        token_count: int
    ):
        """Store response in cache."""
        if len(self.exact_cache) >= self.max_entries:
            self._evict_lru()
        
        cache_key = self._get_cache_key(prompt, model_name)
        embedding = self.embedding_model.encode(prompt)
        
        entry = CacheEntry(
            prompt_hash=cache_key,
            response=response,
            token_count=token_count,
            created_at=datetime.utcnow(),
            access_count=1,
            last_accessed=datetime.utcnow(),
            model_name=model_name
        )
        
        self.exact_cache[cache_key] = entry
        
        self.semantic_cache.append((prompt, embedding))
        self.embedding_index[cache_key] = len(self.semantic_cache) - 1
    
    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        return datetime.utcnow() - entry.created_at < self.ttl
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        entries = sorted(
            self.exact_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for cache_key, entry in entries[:len(entries) // 10]:
            del self.exact_cache[cache_key]
            
            if cache_key in self.embedding_index:
                idx = self.embedding_index[cache_key]
                self.semantic_cache[idx] = (None, None)
                del self.embedding_index[cache_key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = sum(e.access_count for e in self.exact_cache.values())
        return {
            "total_entries": len(self.exact_cache),
            "total_requests": total_requests,
            "hit_rate": (total_requests - len(self.exact_cache)) / max(total_requests, 1),
            "avg_access_count": np.mean([e.access_count for e in self.exact_cache.values()]) 
                              if self.exact_cache else 0
        }
```

### Pattern 5: Batch Processing Optimizer

```python
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import numpy as np

@dataclass
class BatchRequest:
    request_id: str
    prompt: str
    parameters: Dict
    priority: int = 0
    arrival_time: datetime = None
    
    def __post_init__(self):
        if self.arrival_time is None:
            self.arrival_time = datetime.utcnow()

class DynamicBatcher:
    """
    Dynamic batching for LLM inference with intelligent batching strategies.
    """
    
    def __init__(
        self,
        model_client,
        max_batch_size: int = 32,
        max_wait_ms: int = 100,
        max_tokens_soft_limit: int = 2000
    ):
        self.model = model_client
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.max_tokens = max_tokens_soft_limit
        
        self.pending_requests: List[BatchRequest] = []
        self.lock = asyncio.Lock()
        self.last_batch_time = datetime.utcnow()
    
    async def add_request(
        self,
        request_id: str,
        prompt: str,
        parameters: Dict,
        priority: int = 0
    ) -> asyncio.Future:
        """Add request to batch and return future for result."""
        future = asyncio.Future()
        
        request = BatchRequest(
            request_id=request_id,
            prompt=prompt,
            parameters=parameters,
            priority=priority
        )
        
        async with self.lock:
            self.pending_requests.append(request)
        
        asyncio.create_task(self._process_batch())
        
        return future
    
    async def _process_batch(self):
        """Process batch when ready."""
        async with self.lock:
            now = datetime.utcnow()
            wait_time = (now - self.last_batch_time).total_seconds() * 1000
            
            ready = (
                len(self.pending_requests) >= self.max_batch_size or
                wait_time >= self.max_wait_ms
            )
            
            if not ready:
                return
            
            batch = self._select_batch()
            self.pending_requests = self.pending_requests[len(batch):]
            self.last_batch_time = now
        
        results = await self._execute_batch(batch)
        
        for request, result in zip(batch, results):
            if not request.request_id.endswith("_future"):
                continue
    
    def _select_batch(self) -> List[BatchRequest]:
        """Select optimal batch considering tokens and priorities."""
        if len(self.pending_requests) <= self.max_batch_size:
            return self.pending_requests.copy()
        
        sorted_requests = sorted(
            self.pending_requests,
            key=lambda r: (r.priority, r.arrival_time),
            reverse=True
        )
        
        batch = []
        total_tokens = 0
        
        for request in sorted_requests:
            request_tokens = self._estimate_tokens(request.prompt)
            
            if (total_tokens + request_tokens <= self.max_tokens and 
                len(batch) < self.max_batch_size):
                batch.append(request)
                total_tokens += request_tokens
        
        return batch
    
    async def _execute_batch(self, batch: List[BatchRequest]) -> List[Dict]:
        """Execute batch of requests."""
        if not batch:
            return []
        
        prompts = [r.prompt for r in batch]
        
        try:
            results = await self.model.batch_generate(prompts)
            return results
        except Exception as e:
            return [{"error": str(e)} for _ in batch]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return int(len(text.split()) * 1.3)
    
    def get_queue_stats(self) -> Dict:
        """Get current queue statistics."""
        return {
            "pending_requests": len(self.pending_requests),
            "oldest_request_age_ms": (
                (datetime.utcnow() - min(r.arrival_time for r in self.pending_requests))
                .total_seconds() * 1000
            ) if self.pending_requests else 0,
            "estimated_queue_tokens": sum(
                self._estimate_tokens(r.prompt) for r in self.pending_requests
            )
        }
```

## Framework Integration

### Integration with vLLM for Continuous Batching

```python
class vLLMCostOptimizer:
    def __init__(self, model_path: str):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=4,
            max_num_batched_tokens=32768,
            max_num_seqs=256
        )
    
    def configure_for_cost(self, target_cost_per_1k_tokens: float):
        """Configure engine for target cost efficiency."""
        pass
```

### Integration with Ray Serve

```python
class RayServeCostOptimizer:
    def __init__(self, deployment_handle):
        self.handle = deployment_handle
    
    async def adaptive_replica_count(self, metrics: Dict):
        """Adjust replica count based on traffic and cost."""
        current_qps = metrics.get("queries_per_second", 0)
        avg_latency = metrics.get("avg_latency_ms", 100)
        
        optimal_replicas = int(current_qps * avg_latency / 1000)
        
        await self.handle.options(
            num_replicas=min(optimal_replicas, 100)
        ).remote()
```

## Performance Considerations

### Quantization Trade-offs

| Method | Memory Reduction | Quality Loss | Speedup |
|--------|-----------------|--------------|---------|
| FP16 | baseline | baseline | baseline |
| INT8 | 50% | 1-5% | 1.5-2x |
| INT4 | 75% | 5-15% | 2-4x |
| GPTQ | 65% | 3-8% | 2-3x |
| AWQ | 60% | 2-5% | 2-3x |

### Batch Size Optimization

Optimal batch size depends on:
- Model size (larger models need smaller batches)
- Sequence length (longer sequences need smaller batches)
- GPU memory (more VRAM allows larger batches)
- Latency requirements (smaller batches = lower latency)

### Caching Efficiency

Cache hit rate optimization:
- Exact match cache: 10-30% hit rate typical
- Semantic cache (0.95 similarity): 30-50% hit rate possible
- Cache key optimization: Include parameters that affect output

## Common Pitfalls

### Pitfall 1: Over-Compressing Prompts

**Problem**: Aggressive compression removes essential context, degrading output quality.

**Solution**: Implement quality gates:
```python
async def validate_compression(original, compressed, task):
    validation_prompt = f"""
    Does this compressed version preserve the essential information 
    for the task: {task}?
    
    Original: {original}
    Compressed: {compressed}
    
    Answer: yes or no with brief explanation.
    """
```

### Pitfall 2: Not Accounting for Hidden Costs

**Problem**: Focusing only on token costs while ignoring infrastructure, engineering, and opportunity costs.

**Solution**: Track total cost of ownership:
```python
TOTAL_COST_COMPONENTS = [
    "api_costs",
    "compute_costs", 
    "storage_costs",
    "networking_costs",
    "engineering_time",
    "quality_degradation_cost"
]
```

### Pitfall 3: Using Cheapest Model for All Tasks

**Problem**: Using smaller/cheaper models for complex tasks leads to quality issues requiring rework.

**Solution**: Implement routing with quality thresholds:
```python
THRESHOLDS = {
    "simple_qa": 0.7,  # Can use cheaper model
    "code_generation": 0.9,  # Need high quality
    "creative_writing": 0.85,
}
```

### Pitfall 4: Not Monitoring Cache Efficiency

**Problem**: Cache grows unbounded, consuming memory without proportional benefit.

**Solution**: Implement cache metrics and limits:
```python
CACHE_CONFIG = {
    "max_entries": 100000,
    "max_memory_gb": 10,
    "ttl_hours": 24,
    "min_hit_rate_to_keep": 0.01
}
```

## Research References

1. **Kwon et al. (2023)** - "Efficient Memory Management for Large Language Model Serving" - PagedAttention and KV cache management.

2. **Sheng et al. (2023)** - "Quality and Cost Efficiency in LLM Inference" - Analysis of quality-cost trade-offs.

3. **Pope et al. (2023)** - "Efficient LLM Serving" - Batching and caching strategies.

4. **Zhang et al. (2023)** - "PowerInfer" - Sparse activation for efficient inference.

5. **Frantar et al. (2023)** - "GPTQ" - Post-training quantization for large models.

6. **Kim et al. (2023)** - "AWQ" - Activation-aware weight quantization.

7. **Chen et al. (2023)** - "SpecInfer" - Speculative inference for faster decoding.

8. **Leviathan et al. (2023)** - "Fast Inference from Transformers" - Speculative decoding techniques.

9. **Ho et al. (2023)** - "Token Reduction" - Prompt compression methods.

10. **Jung et al. (2023)** - "Semantic Cache" - Intelligent caching for LLM applications.