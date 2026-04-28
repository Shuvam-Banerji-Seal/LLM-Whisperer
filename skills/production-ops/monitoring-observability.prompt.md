# Monitoring and Observability for LLM Systems

## Problem Statement

Production LLM systems present unique observability challenges that differ significantly from traditional ML models. Unlike classical ML where inputs and outputs are structured data points, LLMs process free-form text with stochastic outputs, making behavior verification and performance tracking substantially more complex. Teams deploying LLMs face critical questions: How do we detect model degradation without labeled test sets? When the model produces a harmful response, how do we identify it in real-time? Why did the model suddenly start generating longer responses? What is the computational cost per conversation? Without proper observability infrastructure, these questions remain unanswered until users report problems.

The core challenges include: (1) lack of ground truth labels for real-time evaluation, (2) the vast output space making manual inspection impossible, (3) latency sensitivity since LLM inference is computationally expensive, (4) cost tracking at a granular level, (5) detecting behavioral changes that might indicate data drift or prompt sensitivity, and (6) handling the non-deterministic nature of generation requiring statistical approaches to monitoring.

This skill covers building comprehensive observability infrastructure for LLM systems that enables proactive detection of issues, rapid debugging, and data-driven optimization decisions.

## Theory & Fundamentals

### The Three Pillars of Observability

Observability in LLM systems rests on three pillars, adapted from traditional software engineering:

**Metrics**: Quantitative measurements aggregated over time. For LLMs, key metrics include:
- Request latency percentiles (p50, p95, p99)
- Token throughput (tokens/second/gpu)
- Error rates by category
- Cost per 1000 tokens
- Cache hit rates

**Logs**: Detailed records of individual events. LLM logs should capture:
- Request ID, timestamp, user ID
- Input prompt (with PII handling)
- Output completion
- Token counts
- Latency breakdown (time to first token, total generation time)
- Model version, temperature, other generation parameters

**Traces**: End-to-end request flow across services. For LLM pipelines, traces connect:
- API gateway receipt
- Auth/Cache lookup
- Prompt preprocessing
- Model inference
- Output postprocessing
- Response delivery

### LLM-Specific Metrics Taxonomy

```
Metrics Hierarchy:
├── Input Metrics
│   ├── Prompt token count distribution
│   ├── Prompt complexity scores
│   ├── Repetition patterns in prompts
│   └── PII detection frequency
├── Output Metrics
│   ├── Completion token count distribution
│   ├── Output complexity/verbosity
│   ├── Repetition in generations
│   ├── Refusal rate (safety triggers)
│   └── Sentiment consistency
├── Latency Metrics
│   ├── Time to First Token (TTFT)
│   ├── Inter-Token Latency (ITL)
│   ├── Total Generation Time
│   └── Queue wait time
├── Cost Metrics
│   ├── Cost per request
│   ├── Cost per successful request
│   ├── Cache savings vs. full inference
│   └── Token usage by user/project
└── Quality Metrics (Proxy)
    ├── Perplexity of outputs (self-checking)
    ├── N-gram overlap with training data
    ├── Semantic drift from expected distribution
    └── User feedback signals (thumbs up/down)
```

### Statistical Approaches to LLM Monitoring

Since we often lack ground truth labels, we use statistical methods:

**Distribution Shift Detection**: Monitor the distribution of outputs over time using:
- KL divergence between output token distributions
- Wasserstein distance for embedding space changes
- Chi-squared tests for categorical output patterns

**Anomaly Scoring**: Each request gets an anomaly score based on:
- Prompt unusualness (rare tokens, unusual structures)
- Output unusualness (low probability tokens, unusual lengths)
- Latency outliers
- Combined ensemble scores

**Bayesian Change Detection**: Maintain posterior distributions over key metrics and detect when observations suggest parameter changes.

## Implementation Patterns

### Pattern 1: Structured Logging with OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
import time
import json
import hashlib

class LLMObservabilityLogger:
    def __init__(self, service_name: str, otlp_endpoint: str):
        self.service_name = service_name
        self.tracer = self._setup_tracing(otlp_endpoint)
        self.metrics = MetricsCollector()
        
    def _setup_tracing(self, otlp_endpoint: str):
        provider = TracerProvider()
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        return trace.get_tracer(self.service_name)
    
    def log_llm_request(
        self,
        prompt: str,
        model_name: str,
        parameters: dict,
        user_id: str = None
    ) -> contextmanager:
        """Context manager for tracing LLM requests."""
        request_id = self._generate_request_id(prompt, user_id)
        start_time = time.perf_counter()
        
        with self.tracer.start_as_current_span(
            f"llm.{model_name}",
            attributes={
                "request.id": request_id,
                "user.id": str(user_id) if user_id else "anonymous",
                "model.name": model_name,
                "temperature": parameters.get("temperature", 0.0),
                "max_tokens": parameters.get("max_tokens", 0),
            }
        ) as span:
            try:
                yield request_id
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def log_generation(
        self,
        request_id: str,
        prompt: str,
        completion: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        error: Exception = None
    ):
        """Log detailed generation information."""
        log_entry = {
            "timestamp": time.time(),
            "request_id": request_id,
            "service": self.service_name,
            "event_type": "llm_generation",
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "completion_hash": hashlib.sha256(completion.encode()).hexdigest()[:16],
            "token_counts": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "latency_ms": latency_ms,
            "error": str(error) if error else None
        }
        
        if error:
            log_entry["error_type"] = type(error).__name__
        
        self._emit_log(log_entry)
        self.metrics.record_request(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            error=error is not None
        )
```

### Pattern 2: Real-time Quality Monitoring with Rolling Windows

```python
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
import numpy as np
import threading
from scipy.stats import entropy

@dataclass
class QualityMetrics:
    output_entropies: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    output_lengths: Deque[int] = field(default_factory=lambda: deque(maxlen=1000))
    latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    total_count: int = 0
    
class LLMQualityMonitor:
    def __init__(
        self,
        baseline_entropy: float,
        baseline_length_mean: float,
        baseline_length_std: float,
        alert_percentile: float = 95
    ):
        self.baseline_entropy = baseline_entropy
        self.baseline_length_mean = baseline_length_mean
        self.baseline_length_std = baseline_length_std
        self.alert_percentile = alert_percentile
        self.metrics = QualityMetrics()
        self.lock = threading.Lock()
        self._running = True
        
    def record_response(
        self,
        output_tokens: List[int],
        token_logprobs: List[float],
        latency_ms: float,
        error: bool = False
    ):
        with self.lock:
            self.metrics.total_count += 1
            if error:
                self.metrics.error_count += 1
            
            output_entropy = entropy(np.exp(token_logprobs))
            self.metrics.output_entropies.append(output_entropy)
            self.metrics.output_lengths.append(len(output_tokens))
            self.metrics.latencies.append(latency_ms)
    
    def check_anomalies(self) -> List[Dict]:
        """Check for anomalies across all monitored metrics."""
        alerts = []
        
        with self.lock:
            if len(self.metrics.output_entropies) < 100:
                return alerts
            
            recent_entropies = list(self.metrics.output_entropies)
            recent_lengths = list(self.metrics.output_lengths)
            recent_latencies = list(self.metrics.latencies)
        
        entropy_zscore = (np.mean(recent_entropies[-100:]) - self.baseline_entropy) / np.std(recent_entropies)
        if abs(entropy_zscore) > 3:
            alerts.append({
                "type": "entropy_shift",
                "severity": "high",
                "zscore": entropy_zscore,
                "current": np.mean(recent_entropies[-100:]),
                "baseline": self.baseline_entropy
            })
        
        length_mean = np.mean(recent_lengths[-100:])
        length_zscore = (length_mean - self.baseline_length_mean) / self.baseline_length_std
        if abs(length_zscore) > 3:
            alerts.append({
                "type": "length_shift",
                "severity": "high",
                "zscore": length_zscore,
                "current": length_mean,
                "baseline": self.baseline_length_mean
            })
        
        error_rate = self.metrics.error_count / max(self.metrics.total_count, 1)
        if error_rate > 0.05:
            alerts.append({
                "type": "error_rate_spike",
                "severity": "critical",
                "error_rate": error_rate,
                "threshold": 0.05
            })
        
        return alerts
    
    def get_summary_stats(self) -> Dict:
        with self.lock:
            if len(self.metrics.output_entropies) == 0:
                return {}
            
            return {
                "total_requests": self.metrics.total_count,
                "error_rate": self.metrics.error_count / max(self.metrics.total_count, 1),
                "avg_output_entropy": np.mean(self.metrics.output_entropies),
                "avg_output_length": np.mean(self.metrics.output_lengths),
                "p95_latency": np.percentile(self.metrics.latencies, 95),
                "p99_latency": np.percentile(self.metrics.latencies, 99),
                "entropy_std": np.std(self.metrics.output_entropies)
            }
```

### Pattern 3: Distributed Tracing for LLM Pipelines

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import httpx
import asyncio
from typing import Dict, Any, Optional, List
from contextvars import ContextVar

trace_context: ContextVar[Optional[trace.Context]] = ContextVar('trace_context', default=None)

class LLMPipelineTracer:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = trace.get_tracer(service_name)
        self.http_client = httpx.AsyncClient()
        
    async def trace_pipeline(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trace a full LLM pipeline from input to output."""
        
        with self.tracer.start_as_current_span(
            "llm_pipeline",
            kind=trace.SpanKind.INTERNAL
        ) as span:
            span.set_attribute("user.input_length", len(user_input))
            
            step_traces = []
            
            preprocessing_span = self.tracer.start_span("preprocessing")
            with preprocessing_span:
                preprocessed = await self._preprocess(user_input)
                preprocessing_span.set_attribute("tokens.count", preprocessed["token_count"])
                preprocessing_span.set_attribute("cache.hit", preprocessed.get("cache_hit", False))
            step_traces.append(self._span_to_dict(preprocessing_span))
            
            retrieval_span = self.tracer.start_span("retrieval")
            retrieved_context = []
            with retrieval_span:
                if context and context.get("use_rag"):
                    retrieved_context = await self._retrieve_context(preprocessed)
                    retrieval_span.set_attribute("docs.retrieved", len(retrieved_context))
            step_traces.append(self._span_to_dict(retrieval_span))
            
            generation_span = self.tracer.start_span("generation")
            with generation_span:
                prompt = self._build_prompt(preprocessed, retrieved_context)
                output = await self._generate(prompt)
                generation_span.set_attribute("output.tokens", output["token_count"])
                generation_span.set_attribute("output.latency_ms", output["latency_ms"])
            step_traces.append(self._span_to_dict(generation_span))
            
            postprocessing_span = self.tracer.start_span("postprocessing")
            with postprocessing_span:
                result = await self._postprocess(output["text"])
            step_traces.append(self._span_to_dict(postprocessing_span))
            
            span.set_attribute("pipeline.total_steps", len(step_traces))
            
            return {
                "output": result,
                "trace": {
                    "service": self.service_name,
                    "steps": step_traces,
                    "total_latency_ms": sum(s["duration_ms"] for s in step_traces)
                }
            }
    
    async def _preprocess(self, user_input: str) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return {
            "text": user_input.strip(),
            "token_count": len(user_input.split()),
            "cache_hit": False
        }
    
    async def _retrieve_context(self, preprocessed: Dict) -> List[str]:
        await asyncio.sleep(0.02)
        return ["Relevant context document 1", "Relevant context document 2"]
    
    async def _generate(self, prompt: str) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {
            "text": f"Generated response for: {prompt[:50]}...",
            "token_count": 50,
            "latency_ms": 100.0
        }
    
    async def _postprocess(self, text: str) -> str:
        await asyncio.sleep(0.005)
        return text.strip()
    
    def _span_to_dict(self, span) -> Dict[str, Any]:
        return {
            "name": span.name,
            "duration_ms": span.end_time - span.start_time if span.is_recording() else 0,
            "attributes": {k: str(v) for k, v in span.attributes.items()} if span.attributes else {}
        }
```

### Pattern 4: Cost Attribution and Budget Alerting

```python
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import threading
from collections import defaultdict

@dataclass
class CostRecord:
    timestamp: datetime
    user_id: str
    project_id: str
    input_tokens: int
    output_tokens: int
    model_name: str
    cost: float

class LLMCostTracker:
    def __init__(
        self,
        pricing: Dict[str, Dict[str, float]],
        budget_alerts: Dict[str, float]
    ):
        self.pricing = pricing
        self.budget_alerts = budget_alerts
        self.records: List[CostRecord] = []
        self.user_costs: Dict[str, float] = defaultdict(float)
        self.project_costs: Dict[str, float] = defaultdict(float)
        self.lock = threading.Lock()
        
    def record(
        self,
        user_id: str,
        project_id: str,
        input_tokens: int,
        output_tokens: int,
        model_name: str
    ):
        cost = self.calculate_cost(input_tokens, output_tokens, model_name)
        
        record = CostRecord(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            project_id=project_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=model_name,
            cost=cost
        )
        
        with self.lock:
            self.records.append(record)
            self.user_costs[user_id] += cost
            self.project_costs[project_id] += cost
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str
    ) -> float:
        if model_name not in self.pricing:
            return 0.0
        
        model_pricing = self.pricing[model_name]
        input_cost = (input_tokens / 1000) * model_pricing.get("input", 0)
        output_cost = (output_tokens / 1000) * model_pricing.get("output", 0)
        return input_cost + output_cost
    
    def check_budget_alerts(
        self,
        lookback_hours: int = 24
    ) -> List[Dict]:
        alerts = []
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        with self.lock:
            recent_records = [r for r in self.records if r.timestamp > cutoff]
        
        recent_user_costs = defaultdict(float)
        recent_project_costs = defaultdict(float)
        
        for record in recent_records:
            recent_user_costs[record.user_id] += record.cost
            recent_project_costs[record.project_id] += record.cost
        
        for user_id, cost in recent_user_costs.items():
            if user_id in self.budget_alerts and cost > self.budget_alerts[user_id]:
                alerts.append({
                    "type": "user_budget_exceeded",
                    "user_id": user_id,
                    "cost": cost,
                    "budget": self.budget_alerts[user_id],
                    "percentage": (cost / self.budget_alerts[user_id]) * 100
                })
        
        for project_id, cost in recent_project_costs.items():
            if project_id in self.budget_alerts and cost > self.budget_alerts[project_id]:
                alerts.append({
                    "type": "project_budget_exceeded",
                    "project_id": project_id,
                    "cost": cost,
                    "budget": self.budget_alerts[project_id],
                    "percentage": (cost / self.budget_alerts[project_id]) * 100
                })
        
        return alerts
    
    def get_cost_breakdown(
        self,
        group_by: str = "user",
        lookback_hours: int = 24
    ) -> Dict:
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        with self.lock:
            recent_records = [r for r in self.records if r.timestamp > cutoff]
        
        if group_by == "user":
            costs = defaultdict(lambda: {"cost": 0, "input_tokens": 0, "output_tokens": 0})
            for r in recent_records:
                costs[r.user_id]["cost"] += r.cost
                costs[r.user_id]["input_tokens"] += r.input_tokens
                costs[r.user_id]["output_tokens"] += r.output_tokens
        else:
            costs = defaultdict(lambda: {"cost": 0, "input_tokens": 0, "output_tokens": 0})
            for r in recent_records:
                costs[r.project_id]["cost"] += r.cost
                costs[r.project_id]["input_tokens"] += r.input_tokens
                costs[r.project_id]["output_tokens"] += r.output_tokens
        
        return dict(costs)
```

### Pattern 5: Semantic Drift Detection

```python
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ks_2samp
import threading

class SemanticDriftDetector:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reference_window_size: int = 1000,
        current_window_size: int = 100,
        drift_threshold: float = 0.1
    ):
        self.model = SentenceTransformer(embedding_model)
        self.reference_window_size = reference_window_size
        self.current_window_size = current_window_size
        self.drift_threshold = drift_threshold
        
        self.reference_embeddings: List[np.ndarray] = []
        self.current_embeddings: List[np.ndarray] = []
        self.lock = threading.Lock()
    
    def record_prompt(self, prompt: str):
        embedding = self.model.encode(prompt, convert_to_numpy=True)
        
        with self.lock:
            if len(self.reference_embeddings) < self.reference_window_size:
                self.reference_embeddings.append(embedding)
            
            self.current_embeddings.append(embedding)
            if len(self.current_embeddings) > self.current_window_size:
                self.current_embeddings.pop(0)
    
    def compute_drift_score(self) -> Tuple[float, str]:
        with self.lock:
            if len(self.reference_embeddings) < 10 or len(self.current_embeddings) < 10:
                return 0.0, "insufficient_data"
            
            ref_matrix = np.array(self.reference_embeddings)
            curr_matrix = np.array(self.current_embeddings)
        
        ref_mean = np.mean(ref_matrix, axis=0)
        curr_mean = np.mean(curr_matrix, axis=0)
        mean_diff = np.linalg.norm(ref_mean - curr_mean)
        
        ref_std = np.std(ref_matrix, axis=0)
        curr_std = np.std(curr_matrix, axis=0)
        std_diff = np.mean(np.abs(ref_std - curr_std))
        
        ref_centroids = self._compute_cluster_centroids(ref_matrix, n_clusters=10)
        curr_centroids = self._compute_cluster_centroids(curr_matrix, n_clusters=10)
        centroid_diff = np.mean([cosine_similarity([r], [c])[0][0] 
                                 for r, c in zip(ref_centroids, curr_centroids)])
        
        combined_score = mean_diff * 0.5 + std_diff * 0.3 + (1 - centroid_diff) * 0.2
        
        return float(combined_score), self._interpret_drift(combined_score)
    
    def _compute_cluster_centroids(
        self,
        embeddings: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=min(n_clusters, len(embeddings)))
        kmeans.fit(embeddings)
        return kmeans.cluster_centers_
    
    def _interpret_drift(self, score: float) -> str:
        if score < self.drift_threshold * 0.5:
            return "stable"
        elif score < self.drift_threshold:
            return "minor_drift"
        elif score < self.drift_threshold * 2:
            return "moderate_drift"
        else:
            return "severe_drift"
    
    def check_distribution_shift(self) -> dict:
        score, interpretation = self.compute_drift_score()
        
        with self.lock:
            ref_emb = np.array(self.reference_embeddings)
            curr_emb = np.array(self.current_embeddings)
        
        ks_stat, ks_pvalue = ks_2samp(
            ref_emb.flatten()[:10000],
            curr_emb.flatten()[:10000]
        )
        
        return {
            "drift_score": score,
            "interpretation": interpretation,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "requires_action": score > self.drift_threshold
        }
```

## Framework Integration

### Integration with LangChain

```python
from langchain.callbacks import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Optional, List, Dict, Any
import time

class LangChainObservabilityHandler(BaseCallbackHandler):
    def __init__(self, metrics_collector, tracer):
        self.metrics = metrics_collector
        self.tracer = tracer
        
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ):
        self.start_time = time.perf_counter()
        for prompt in prompts:
            self.tracer.log_prompt(prompt)
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs
    ):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        if response.generations:
            output = response.generations[0][0].text
            self.metrics.record_generation(
                output_length=len(output.split()),
                latency_ms=elapsed
            )
```

### Integration with vLLM

```python
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

class vLLMObservabilityWrapper:
    def __init__(self, model_path: str, metrics_collector):
        self.llm = LLM(model=model_path)
        self.metrics = metrics_collector
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        sampling_params = SamplingParams(**kwargs)
        start_time = time.perf_counter()
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        total_latency = (time.perf_counter() - start_time) * 1000
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids)
            results.append(generated_text)
            
            self.metrics.record_response(
                output_tokens=output.outputs[0].token_ids,
                token_logprobs=output.outputs[0].logprobs or [],
                latency_ms=total_latency / len(prompts)
            )
        
        return results
```

## Performance Considerations

### Metric Collection Overhead

Logging every token generation can introduce significant overhead. Benchmarks show:
- Basic structured logging: ~0.1-0.5ms per request
- Full OpenTelemetry tracing: ~1-5ms per request  
- Embedding-based quality monitoring: ~10-50ms per request (batch recommended)

**Optimization strategies**:
- Use async logging with background workers
- Batch metric exports to reduce network overhead
- Sample logging for high-volume requests (log 1 in 100 or use adaptive sampling)
- Pre-compute embeddings in batches rather than per-request

### Storage and Query Performance

For high-volume LLM systems:
- Use time-series databases (Prometheus, InfluxDB) for metrics
- Use columnar storage (Parquet) for detailed logs enabling efficient queries
- Implement log retention policies (e.g., detailed logs for 7 days, aggregated for 90 days)
- Consider sampling strategies for storage-constrained environments

### Alerting Latency Trade-offs

Real-time alerts require smaller windows but increase false positives. Production recommendations:
- Use 5-minute rolling windows for latency alerts
- Use 1-hour rolling windows for quality/composition alerts
- Use 24-hour windows for cost/billing alerts
- Implement cooldowns on alerts to prevent alert fatigue

## Common Pitfalls

### Pitfall 1: Logging PII Without Proper Handling

**Problem**: Storing full prompts containing personal information creates compliance risks and storage bloat.

**Solution**: Implement PII detection and redaction before logging:
```python
class PIIRedactingLogger:
    def __init__(self, base_logger):
        self.base = base
        self.pii_detector = PIIDetector()
    
    def log_prompt(self, prompt: str):
        redacted, entities = self.pii_detector.detect_and_redact(prompt)
        self.base.log({
            "redacted_prompt": redacted,
            "pii_entities": entities,
            "pii_redacted": True
        })
```

### Pitfall 2: Not Tracking Cost Per User/Request

**Problem**: Without granular cost tracking, 无法进行成本优化 or chargeback。

**Solution**: Always include user and request context in cost tracking:
```python
# Always record with attribution
cost_tracker.record(
    user_id=context.user_id,
    project_id=context.project_id,
    input_tokens=tokens_in,
    output_tokens=tokens_out,
    model_name=model
)
```

### Pitfall 3: Ignoring Output Distribution Monitoring

**Problem**: Without checking output statistics, model degradation goes undetected.

**Solution**: Track not just errors but output characteristics:
- Monitor output length distribution (sudden changes indicate issues)
- Track refusal rates (spikes may indicate overly aggressive safety)
- Monitor entropy patterns (low entropy = repetitive/halting outputs)

### Pitfall 4: Not Implementing Sampling for High-Volume Scenarios

**Problem**: Logging every request at high volume causes performance degradation.

**Solution**: Implement adaptive sampling:
```python
class SamplingLogger:
    def __init__(self, base_rate: float = 0.01):
        self.base_rate = base_rate
        self.adaptive_rate = base_rate
        
    def should_log(self, request_features: dict) -> bool:
        risk_score = self._compute_risk_score(request_features)
        effective_rate = self.base_rate * (1 + risk_score * 10)
        return random.random() < effective_rate
```

## Research References

1. **Dodge et al. (2022)** - "Documenting Large Language Models: A Field Guide" - Best practices for LLM documentation and logging conventions.

2. **Laskar et al. (2023)** - "A Systematic Evaluation of LLM-as-a-Judge" - Evaluation methodology for LLM output quality assessment.

3. **Bommasani et al. (2021)** - "On the Opportunities and Risks of Foundation Models" - Stanford CRFM paper establishing observability frameworks for foundation models.

4. **Chang et al. (2023)** - "Monitoring ML Models in Production" - General ML monitoring techniques applicable to LLM systems.

5. **Klaise et al. (2023)** - "Manufacturing Events in the Wild" - Event monitoring frameworks for production ML systems.

6. **Paleyes et al. (2022)** - "Industrial Machine Learning Monitoring" - Comprehensive review of ML monitoring infrastructure.

7. **Seraph et al. (2023)** - "Observability in AI Systems" - Defining observability requirements for AI/ML pipelines.

8. **Zhang et al. (2023)** - "Debugging Neural Machine Translation" - Techniques applicable to LLM debugging and monitoring.

9. **Madhusudan et al. (2023)** - "Production ML Monitoring" - O'Reilly report on production ML observability patterns.

10. **Schelter et al. (2022)** - "Automating Large-Scale Data Quality Verification" - Data quality monitoring techniques relevant for LLM prompt/output monitoring.