# LLMOps and Production Monitoring: Comprehensive Guide (2026)

## Executive Summary

LLM Operations (LLMOps) and production monitoring have evolved from optional features to essential requirements for deploying large language models at scale. Unlike traditional application monitoring, LLM systems fail in ways that infrastructure metrics cannot detect—silent queue buildup, GPU memory saturation, hallucinations in successful API responses, and cost explosions hidden in token consumption patterns.

This guide provides production teams with:
- **20+ monitoring tools** (open-source and commercial) with detailed descriptions
- **15+ authoritative guides and blog posts** with URLs and key insights
- **Configuration templates** and code examples for implementation
- **Monitoring dashboards** and alert policies
- **SLO definitions** and burn-rate measurement approaches
- **Cost optimization strategies** with real-world examples

---

## Part 1: LLM Monitoring Frameworks

### 1.1 Monitoring Tool Ecosystem

#### Infrastructure & Metrics Collection

| Tool | Type | License | Best For | Key Features |
|------|------|---------|----------|--------------|
| **Prometheus** | Metrics Backend | Open Source (Apache 2.0) | Pull-based metrics, time-series storage | Efficient storage, PromQL query language, multi-dimensional data, alerting engine |
| **Grafana** | Visualization | Open Source (AGPL) + Commercial | Dashboards, alerting, visualization | Multi-datasource support, automated dashboards, native alerting, 40+ plugins |
| **OpenTelemetry** | Instrumentation | Open Source (Apache 2.0) | Vendor-neutral telemetry | Metrics, traces, logs, OpenTelemetry Protocol (OTLP), GenAI semantic conventions |
| **Datadog** | Commercial SaaS | Proprietary | Full-stack monitoring | APM, logs, metrics, traces, real-time alerts, ML-based anomaly detection |
| **New Relic** | Commercial SaaS | Proprietary | Cloud-native monitoring | Unified platform, NRQL query language, entity relationships, error tracking |
| **Grafana Cloud** | Managed SaaS | Freemium | Managed observability stack | Prometheus + Tempo + Loki + Grafana, AI Observability features, OpenLIT integration |

#### LLM-Specific Observability Platforms

| Tool | Type | License | Best For | Key Features |
|------|------|---------|----------|--------------|
| **Langfuse** | Commercial SaaS + Open Source | Dual Licensed | LLM application tracing | OpenTelemetry integration, cost tracking, evaluation, prompt management |
| **Braintrust** | Commercial SaaS | Proprietary | LLM evaluation & monitoring | Production monitoring, quality evaluation, experimentation platform, cost analytics |
| **LangSmith** | Commercial SaaS | Proprietary | LangChain ecosystem | Tracing, evaluation, playground debugging, prompt versioning |
| **Helicone** | Commercial SaaS + Open Source | Freemium | LLM API monitoring | Multi-provider support, cost tracking, response caching, fallback routing |
| **Arize AI** | Commercial SaaS | Proprietary | Model monitoring & evaluation | ML observability, response drift detection, embedded evaluations |
| **Galileo** | Commercial SaaS | Proprietary | GenAI evaluation intelligence | Evaluation platform, safety monitoring, quality scoring, hallucination detection |
| **Confident AI** | Commercial SaaS | Proprietary | LLM evaluation framework | Automated testing, evaluation scoring, CI/CD integration |
| **Traceloop** | Commercial SaaS + Open Source | Dual Licensed | LLM observability | Per-user/feature monitoring, distributed tracing, cost attribution |
| **OpenLIT** | Open Source | Apache 2.0 | OpenTelemetry LLM instrumentation | Auto-instrumentation, 50+ GenAI tools, semantic conventions |
| **Phoenix** | Open Source | Apache 2.0 | ML observability | Traces, rankings, evaluations, LLM-specific scorers |

#### Inference Server & GPU Monitoring

| Tool | Type | License | Best For | Key Features |
|------|------|---------|----------|--------------|
| **vLLM** | Open Source | Apache 2.0 | LLM inference serving | Prometheus `/metrics` endpoint, token tracking, KV cache monitoring |
| **TGI (Text Generation Inference)** | Open Source | Apache 2.0 | HuggingFace-compatible serving | Queue metrics, batch metrics, per-token latency, distributed tracing |
| **llama.cpp** | Open Source | MIT | Lightweight inference | Built-in Prometheus metrics, GPU monitoring, low resource overhead |
| **NVIDIA Triton** | Commercial + Open Source | Free | Multi-model serving | GPU metrics, multi-framework support, dynamic batching |
| **NVIDIA DCGM Exporter** | Open Source | Apache 2.0 | GPU telemetry | Direct GPU metrics (utilization, memory, temperature, power), Prometheus-compatible |
| **Ollama** | Open Source | MIT | Local LLM serving | Simple deployment, model management, REST API, metrics support |

#### Log Aggregation & Trace Analysis

| Tool | Type | License | Best For | Key Features |
|------|------|---------|----------|--------------|
| **Grafana Loki** | Open Source | AGPL + Commercial | Log aggregation | Label-only indexing, low cost, LogQL query language, plugin ecosystem |
| **Grafana Tempo** | Open Source | AGPL + Commercial | Distributed tracing | High-scale tracing, metrics-generator, span-to-metrics conversion |
| **Jaeger** | Open Source | Apache 2.0 | Distributed tracing | OTLP support, multiple backends, trace sampling, UI exploration |
| **Elasticsearch/OpenSearch** | Open Source/Commercial | Elastic License | Full-text search on logs | Powerful search, Kibana visualization, ML features |
| **Splunk** | Commercial SaaS | Proprietary | Enterprise log management | Real-time analytics, machine learning, data models, compliance reporting |

#### Alerting & SLO Management

| Tool | Type | License | Best For | Key Features |
|------|------|---------|----------|--------------|
| **Prometheus AlertManager** | Open Source | Apache 2.0 | Alert routing & deduplication | Grouping, routing, silencing, integration with 50+ notification channels |
| **OpenObserve** | Open Source + SaaS | AGPL + Commercial | Integrated observability | Logs, metrics, traces in one platform, AI anomaly detection, SLO-based alerting |
| **Elastic Observability** | Commercial SaaS | Elastic License | Full-stack observability | End-to-end tracing, log analytics, APM, alerting |

---

## Part 2: Essential Metrics & KPIs for LLMs

### 2.1 Core Metrics Framework

#### Golden Signals for LLM Systems

```
TRAFFIC          → Requests/sec, Tokens/sec, Throughput (generated tokens/min)
ERRORS           → Error rate %, Timeouts, Rate limits (429), Out-of-memory
LATENCY          → P50/P95/P99 end-to-end latency
                 → Time-to-first-token (TTFT)
                 → Time-per-output-token (TPOT)
SATURATION       → GPU utilization %, GPU memory usage
                 → KV cache usage, Queue size
                 → Batch size
```

#### Token Usage Tracking

```yaml
Input Tokens Metrics:
  - gen_ai.client.token.usage (counter, input_token attribute)
  - gen_ai.server.input_tokens (histogram, per-request)
  - llm_prompt_tokens_total (Prometheus counter)

Output Tokens Metrics:
  - gen_ai.client.token.usage (counter, output_token attribute)
  - gen_ai.server.output_tokens (histogram, per-request)
  - llm_completion_tokens_total (Prometheus counter)

Token Efficiency:
  - tokens_per_request (average output / input ratio)
  - token_waste_rate (duplicated or unused tokens)
  - cache_hit_ratio (prompt caching effectiveness)
```

#### Latency Metrics Breakdown

```yaml
Request Latency Components:
  - TTFT (Time-to-First-Token):
      Definition: Time from request arrival to first output token
      Ideal: <100ms for interactive applications
      Metric: gen_ai.server.time_to_first_token (histogram)
      
  - TPOT (Time-Per-Output-Token):
      Definition: Average time between consecutive output tokens
      Ideal: <50ms for streaming UX
      Metric: gen_ai.server.time_per_output_token (histogram)
      
  - End-to-End Latency:
      Definition: Request arrival to final token delivery
      Ideal: <500ms for most applications
      Metric: llm_request_latency_seconds (histogram)
      
  - Queue Latency:
      Definition: Time spent waiting for inference
      Ideal: <100ms under normal load
      Metric: tgi_request_queue_duration (TGI)
      
  - Inference Latency:
      Definition: Actual model computation time
      Ideal: <300ms for typical requests
      Metric: vllm_e2e_request_latency_seconds (vLLM)
```

#### Cost Per Token Calculations

```python
# Basic cost tracking
cost_per_request = (input_tokens * cost_per_input_token + 
                   output_tokens * cost_per_output_token)

# OpenAI API example (March 2026 pricing)
GPT-4-Turbo:
  input:  $0.01 / 1K tokens
  output: $0.03 / 1K tokens
  
Claude 3 Haiku:
  input:  $0.25 / 1M tokens
  output: $1.25 / 1M tokens

# Cost per user/feature attribution
cost_per_user = sum(cost_per_request) / unique_users
cost_per_feature = cost_per_feature_requests / feature_invocations
cost_efficiency = (successful_requests / total_requests) * cost_per_successful_request
```

#### Quality Metric Definitions

```yaml
Accuracy Metrics:
  - Hallucination Rate: % of responses containing ungrounded claims
  - Groundedness Score: % of claims supported by retrieved context
  - Factual Accuracy: % of verifiable claims that are correct
  - Reference Accuracy: % of claims properly attributed

Relevance Metrics:
  - Query Relevance: Does response address the user's question?
  - Context Relevance: % of retrieved documents relevant to query
  - Retrieval Precision: Relevant results / Total retrieved results
  - Retrieval Recall: Relevant retrieved / Total relevant results

Content Quality:
  - Coherence Score: Response logical consistency (0-1 scale)
  - Completeness Score: Question fully answered (0-1 scale)
  - Toxicity Score: Absence of harmful content (0-1 scale)
  - Bias Detection: Systematic fairness across demographic groups

User Satisfaction:
  - Thumbs Up Rate: % positive explicit feedback
  - Task Completion Rate: % of interactions achieving user goal
  - Escalation Rate: % requiring human intervention
  - Session Abandonment: % of incomplete conversations
```

#### GPU/Resource Utilization Monitoring

```yaml
GPU Metrics (DCGM/nvidia-smi):
  GPU Utilization:
    - Definition: % of GPU clock cycles with work
    - Target: 80-95% for optimal throughput
    - Metric: DCGM_FI_DEV_GPU_UTIL or nvidia_gpu_utilization
    
  GPU Memory Usage:
    - Definition: Allocated VRAM / Total VRAM
    - Target: 85-95% for maximum capacity
    - Metric: DCGM_FI_DEV_FB_FREE or nvidia_gpu_memory_allocated_mb
    
  GPU Temperature:
    - Definition: GPU die temperature in Celsius
    - Target: <80°C, Warning >85°C
    - Metric: DCGM_FI_DEV_GPU_TEMP
    
  Power Consumption:
    - Definition: Instantaneous power draw in Watts
    - Target: 80-95% of TDP for efficiency
    - Metric: DCGM_FI_DEV_POWER_USAGE

System Metrics:
  - CPU Utilization: Preprocessing, tokenization bottlenecks
  - Memory Usage: Context length impact on system RAM
  - Network Bandwidth: Multi-GPU throughput constraints
  - Disk I/O: Model loading, batch processing
```

---

## Part 3: Alerting & SLOs

### 3.1 SLO Definition Framework

#### Step 1: Define Service Level Indicators (SLI)

```yaml
# Example SLI for customer support chatbot
SLI Definition:
  - Numerator: Requests returning 2xx status in <500ms
  - Denominator: Total requests
  - Calculation: (good_requests / total_requests) × 100
  - Target Window: Rolling 30 days

# Multi-dimensional SLI
SLI_by_feature:
  chat_support: 99.5% (more critical)
  document_analysis: 99.0% (background task)
  code_generation: 98.5% (creative workload)
```

#### Step 2: Set Service Level Objectives (SLO)

```yaml
# Example SLOs for production LLM system
Latency SLO:
  - Target: 99.5% of requests <500ms (p95)
  - Time Window: 30 days rolling
  - Error Budget: 0.5% = ~216 minutes/month = 3h 36min

Availability SLO:
  - Target: 99.9% uptime
  - Time Window: 30 days rolling
  - Error Budget: 0.1% = ~43 minutes/month

Quality SLO:
  - Target: 95% of responses score >0.8 relevance
  - Measurement: Automated + sampled human review
  - Time Window: 7 days rolling

Cost SLO:
  - Target: $0.02 average cost per interaction
  - Alert: >$0.025 sustained
  - Review: Weekly trend analysis
```

#### Step 3: Calculate Error Budget & Burn Rate

```python
# Error budget calculation
slo_target = 0.995  # 99.5%
error_budget = 1 - slo_target  # 0.005 = 0.5%

# 30-day window (43,200 minutes)
monthly_budget_minutes = 43200 * error_budget  # 216 minutes

# Burn rate definition
# Burn Rate = (Current Error Rate) / (1 - SLO Target)

burn_rate_mapping = {
    "1.0": "on_pace (30 days to exhaust budget)",
    "2.0": "fast (15 days to exhaust)",
    "6.0": "significant_degradation (5 days)",
    "14.0": "major_incident (2 days / page_immediately)",
}

# Alert configuration (multi-window)
fast_burn_alert:
  # Alert when 14× burn rate for 5 minutes
  condition: burn_rate_5m > 14.0 AND burn_rate_1h > 14.0
  action: "Page on-call immediately"
  
slow_burn_alert:
  # Alert when 6× burn rate for 1 hour
  condition: burn_rate_30m > 6.0 AND burn_rate_6h > 6.0
  action: "Slack notification to team"
```

### 3.2 Recommended Alert Thresholds for LLMs

```yaml
Latency Alerts:
  P95 Latency > 500ms:
    duration: 5 minutes
    severity: warning
  P99 Latency > 1000ms:
    duration: 10 minutes
    severity: critical
  TTFT > 200ms:
    duration: sustained 2 min
    severity: warning

Throughput Alerts:
  Requests/sec drop >20% baseline:
    duration: 5 minutes
    severity: warning
  Tokens/sec < expected * 0.8:
    duration: 10 minutes
    severity: critical

Error Rate Alerts:
  Error rate > 1%:
    duration: 2 minutes
    severity: critical
  Timeouts > 0.5%:
    duration: 5 minutes
    severity: warning

Quality Alerts:
  Hallucination rate > 5%:
    duration: 1 hour rolling
    severity: page
  Relevance score avg < 0.75:
    duration: 1 hour rolling
    severity: warning

Cost Alerts:
  Cost/request > baseline * 1.5:
    duration: 30 minutes
    severity: warning
  Daily cost > budget * 1.1:
    duration: sustained
    severity: critical

Resource Alerts:
  GPU utilization > 95%:
    duration: 10 minutes
    severity: warning
  GPU memory > 90% for >2 min:
    duration: immediate
    severity: page
  Queue size > 100:
    duration: 5 minutes
    severity: warning
```

---

## Part 4: Incident Response & Anomaly Detection

### 4.1 Anomaly Detection Strategies

```yaml
Statistical Anomaly Detection:
  Method: Standard deviation from rolling baseline
  Implementation:
    - Calculate mean + 3σ (3 standard deviations)
    - Alert when metric exceeds threshold
    - Suitable for: Latency, throughput, error rates
    
  Prometheus example:
    avg(rate(requests_total[5m])) > 
    (avg_over_time(avg(rate(requests_total[5m]))[1h]) + 
     3 * stddev_over_time(...))

Seasonal Pattern Detection:
  Method: Compare current vs same time period (1 week/1 month ago)
  Use case: Traffic patterns, model quality drift
  Tools: OpenObserve ML anomaly detection, Datadog Anomaly Detection

Isolation Forest / ML-Based:
  Method: Unsupervised learning for multivariate anomalies
  Use case: Complex correlations (latency + error rate + memory)
  Tools: Datadog ML algorithms, New Relic Applied Intelligence

LLM-Specific Anomalies:
  - Token consumption spike: May indicate prompt injection
  - Quality score drop: Possible model degradation
  - Latency increase with no traffic change: KV cache/queue saturation
  - Hallucination rate jump: Input distribution shift
```

### 4.2 Incident Response Playbook

```yaml
Stage 1: Detection (0-2 minutes)
  Triggers:
    - Critical alert fires
    - User reports issue
    - Anomaly detected
  
  Immediate Actions:
    1. Page on-call team
    2. Create incident ticket
    3. Grab most recent logs/traces
    4. Record incident start time

Stage 2: Assessment (2-10 minutes)
  Questions to Answer:
    - What changed? (Deploy, config, traffic pattern)
    - User impact? (% of requests affected, feature impact)
    - Root cause area? (Inference, retrieval, infrastructure)
  
  Tools:
    - Dashboards: Error rate, latency, resource graphs
    - Recent deploys: Git log, deployment tracker
    - Logs: Structured logs with trace IDs
    - Traces: Distributed trace waterfall for slow requests

Stage 3: Mitigation (5-30 minutes)
  Quick Fixes (in order):
    1. Increase concurrency limits / rate limiting
    2. Scale inference servers horizontally
    3. Switch to lower-cost model variant
    4. Enable cached responses
    5. Rollback recent deployment
    6. Failover to backup region
  
  Escalation:
    - If not resolved in 15 min: Page incident commander
    - If P1 impact: Page senior engineer + manager

Stage 4: Resolution & Postmortem (30 min - 24 hours)
  During Incident:
    - Update status page every 15 minutes
    - Maintain incident channel for coordination
  
  Post-Incident:
    - Write root cause analysis
    - Identify prevention opportunities
    - Update runbooks
    - Assign action items with owners
```

---

## Part 5: Production Deployment Strategies

### 5.1 Canary Deployment for LLMs

```yaml
Phased Rollout Strategy:
  Phase 1: Shadow Testing (0% traffic)
    - Deploy new model alongside current
    - Route duplicate traffic to both
    - Compare outputs without affecting users
    - Duration: 1-24 hours
    - Success Criteria: Same quality scores, no new errors
    
  Phase 2: Canary (5% traffic)
    - Route small % of real traffic to new model
    - Monitor: Latency, errors, quality metrics
    - Duration: 1-4 hours
    - Success Criteria: Metrics within SLO
    
  Phase 3: Gradual Rollout (25% → 50% → 75% → 100%)
    - Increase traffic incrementally every 1-2 hours
    - Monitor error budget burn rate
    - Stop if burn rate > 2×
    - Duration: 4-8 hours total
    
  Automatic Rollback Trigger:
    - Error rate > 1% for >2 min
    - P95 latency > 500ms for >5 min
    - Quality score < 0.85 for >10 min (sampled)

Implementation (Kubernetes):
  apiVersion: v1
  kind: Service
  metadata:
    name: llm-inference
  spec:
    selector:
      app: llm-inference
    ports:
    - name: http
      port: 8000
  ---
  apiVersion: v1
  kind: DestinationRule
  metadata:
    name: llm-inference
  spec:
    host: llm-inference
    trafficPolicy:
      connectionPool:
        http:
          http1MaxPendingRequests: 1024
          http2MaxRequests: 1000
    subsets:
    - name: stable
      labels:
        version: v1
    - name: canary
      labels:
        version: v2
  ---
  apiVersion: v1
  kind: VirtualService
  metadata:
    name: llm-inference
  spec:
    hosts:
    - llm-inference
    http:
    - match:
      - uri:
          prefix: /
      route:
      - destination:
          host: llm-inference
          subset: stable
        weight: 95
      - destination:
          host: llm-inference
          subset: canary
        weight: 5
```

### 5.2 A/B Testing Framework

```yaml
Request Routing:
  Method 1: User Bucketing
    bucket_id = hash(user_id) % 100
    if bucket_id < 50:
      model_version = "control"
    else:
      model_version = "treatment"
  
  Method 2: Feature Flag
    is_variant = feature_flags.get("new_model_variant")
    model = "variant_model" if is_variant else "baseline"
  
  Method 3: Request Parameter
    model = request.query_params.get("model", default="baseline")

Statistical Analysis:
  Sample Size Calculation:
    n = (Z_α + Z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²
    
    Example (99% confidence, 80% power, 2% baseline):
    - Baseline: 2% error rate
    - Minimum detectable effect: 0.5% improvement
    - Required sample size: ~15,000 requests per variant
    
  Duration Estimation:
    1,000 req/min × 2 variants = 2,000 req/min total
    time_to_complete = 15,000 requests / (2,000 req/min) = 7.5 minutes
    
    (In practice, run for longer to account for variance)

Evaluation Metrics:
  Primary:
    - Quality score (hallucination, relevance, coherence)
    - User satisfaction (thumbs up rate)
    - Task completion rate
  
  Secondary:
    - Latency (TTFT, end-to-end)
    - Cost per request
    - Token efficiency
  
  Guardrails:
    - Error rate < 2%
    - Latency p95 < 2× baseline
    - No significant quality decrease

Result Analysis:
  If treatment wins:
    - Promotion pathway to canary → rollout
    - Document improvements for knowledge base
    - Measure long-term impact after 1 week
  
  If control wins:
    - Debug treatment to understand failure
    - If inconclusive: Run longer test
    - Document decision for future reference
```

### 5.3 Rollback Procedures

```yaml
Automatic Rollback Triggers:
  Critical:
    - Error rate > 2% for >1 minute: IMMEDIATE
    - P99 latency > 2× baseline for >5 min: IMMEDIATE
    - Quality score drop > 20%: IMMEDIATE
  
  Automated Action:
    1. Pause traffic to new version (0% canary weight)
    2. Route all traffic to previous stable
    3. Page on-call team
    4. Create incident ticket

Manual Rollback (Kubernetes):
  kubectl rollout undo deployment/llm-inference-v2
  
  Verification:
    1. Confirm traffic routing changed
    2. Monitor metrics for 5 minutes
    3. Verify alerts cleared
    4. Notify team in Slack

Post-Rollback Checklist:
  - [ ] User impact assessment
  - [ ] Root cause identified
  - [ ] Prevent recurrence (fix code, tests, monitoring)
  - [ ] Retry decision (same deployment or redesign)
  - [ ] Update runbook with lessons learned
```

---

## Part 6: Logging & Debugging

### 6.1 Logging Best Practices for LLMs

```yaml
Structured Logging Format:
  - Use JSON format for machine parsing
  - Include trace_id for distributed tracing correlation
  - Log levels: DEBUG < INFO < WARN < ERROR < CRITICAL
  
  Example:
    {
      "timestamp": "2026-04-07T12:34:56.789Z",
      "level": "INFO",
      "service": "llm-inference",
      "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
      "span_id": "00f067aa0ba902b7",
      "message": "LLM inference completed",
      "model": "gpt-4-turbo",
      "tokens_input": 150,
      "tokens_output": 45,
      "latency_ms": 234,
      "cost_usd": 0.0045,
      "user_id": "user_abc123",
      "feature": "chat_support"
    }

Privacy-Safe Logging:
  DO:
    - Log token counts (numerical)
    - Log model names and versions
    - Log latency, cost, quality scores
    - Log error types and codes
    - Log user segments (not raw user_id if possible)
  
  DON'T:
    - Raw prompts (tokenization info OK, prompt text NO)
    - Raw responses (unless explicitly required)
    - API keys, credentials, tokens
    - PII (email, phone, SSN)
    - Raw function arguments
  
  Redaction Example:
    prompt_hash = hash(prompt)  # Instead of prompt
    contains_pii = pii_detector.check(response)  # Boolean flag

Log Levels Strategy:
  DEBUG:
    - Token-level streaming details
    - Memory allocation/deallocation
    - Cache hit/miss details
    - Full span attributes
  
  INFO:
    - Request arrival/completion
    - Model selection decision
    - Errors (with categorization)
    - Cost significant changes
  
  WARN:
    - High latency (approaching SLO)
    - Quality score drop >10%
    - Token usage >expected
    - Cache miss rate high
  
  ERROR:
    - Failed inference
    - Timeout (>SLO threshold)
    - OOM/resource exhaustion
    - API rate limit hit
```

### 6.2 Error Tracking & Debugging Tools

```yaml
Open Source Solutions:
  Sentry (Self-Hosted):
    - Real-time error aggregation
    - Grouping and deduplication
    - Session replay integration
    - SourceMap support
    - API: https://github.com/getsentry/sentry
    
  Elastic Stack:
    - Elasticsearch for log storage
    - Kibana for visualization
    - ML features for anomaly detection
    - Integration with observability data
    
  OpenSearch:
    - Open-source Elasticsearch fork
    - Log analytics, visualization
    - Security plugins built-in
    - Cost-effective for large scale

Commercial Solutions:
  Datadog Error Tracking:
    - Automatic error grouping
    - Root cause analysis (RCA)
    - Source code integration
    - Error budget calculation
  
  New Relic Error Analytics:
    - Error rate and trend analysis
    - Error fingerprinting
    - Error profile comparisons
    - Integration with APM

LLM-Specific Debugging:
  What Went Wrong Checklist:
    Model Quality:
      - Is hallucination rate normal?
      - Did quality score drop?
      - Check logs for context retrieval issues
      - Compare against baseline model
    
    Latency Issue:
      - Is TTFT normal? (If high: model is cold/overloaded)
      - Is TPOT normal? (If high: decode bottleneck)
      - Queue depth? (If high: scale up)
      - Trace the request through pipeline
    
    Cost Explosion:
      - Token count per request (increased prompt size?)
      - Retry loops (tokens wasted on retries)
      - Model switch (switched to more expensive model?)
      - Token leakage (are we generating unnecessarily?)
    
    Errors:
      - 5xx errors: Model server issue, check logs
      - 429 errors: Rate limit hit, scale or implement queuing
      - Timeouts: Queue too deep or model too slow
      - OOM: Increase batch size limits, reduce context length
```

---

## Part 7: Cost Optimization Strategies

### 7.1 Cost Tracking Implementation

```yaml
Token-Level Attribution:
  Cost Model:
    base_cost = (input_tokens * input_cost_per_1k + 
                output_tokens * output_cost_per_1k) / 1000
    
    # Multi-provider support
    provider_rates = {
      "gpt-4-turbo": {
        "input": 0.01 / 1000,
        "output": 0.03 / 1000,
      },
      "claude-3-haiku": {
        "input": 0.25 / 1_000_000,
        "output": 1.25 / 1_000_000,
      },
      "gemini-pro": {
        "input": 0.00075 / 1000,
        "output": 0.00225 / 1000,
      },
    }

Per-User Attribution:
  def calculate_user_cost(user_id: str, period: str = "month"):
    requests = query_requests(user_id, period)
    total_cost = sum(
      (r.input_tokens * rate[r.model]["input"] +
       r.output_tokens * rate[r.model]["output"]) / 1000
      for r in requests
    )
    return total_cost

Per-Feature Attribution:
  def calculate_feature_cost(feature: str, period: str = "month"):
    requests = query_requests(feature=feature, period=period)
    cost_by_model = defaultdict(float)
    for r in requests:
      cost = (r.input_tokens * rate[r.model]["input"] +
              r.output_tokens * rate[r.model]["output"]) / 1000
      cost_by_model[r.model] += cost
    return cost_by_model

Dashboard Queries (PromQL):
  Daily cost trend:
    sum(rate(llm_cost_usd_total[1d])) * 86400
  
  Cost by model:
    sum by (model) (rate(llm_cost_usd_total[1d])) * 86400
  
  Average cost per request:
    (sum(llm_cost_usd_total) / sum(llm_requests_total))
```

### 7.2 Cost Optimization Techniques

```yaml
Prompt Optimization:
  1. Prompt Compression:
     - Remove redundant context
     - Distill instructions to essentials
     - Use placeholder variables instead of repeating data
     - Savings: 20-40% input token reduction
     
     Example:
       # Before (longer)
       """
       You are a helpful customer support agent. Your goal is to help
       customers with their questions. Be polite and professional...
       """
       
       # After (compressed)
       """
       You are a helpful customer support agent.
       
       Guidelines:
       - Be polite and professional
       - Solve issues directly
       """
  
  2. System Prompt Caching (Claude):
     - Cache common system prompts
     - Reuse across multiple requests
     - Savings: 80% cost on cached context
     
     Implementation:
       cache_control = {
         "type": "ephemeral"  # 5 minute cache
       }
       system = [
         {
           "type": "text",
           "text": "You are a helpful assistant",
           "cache_control": cache_control
         }
       ]
  
  3. Context Limit Tuning:
     - Reduce max_tokens for non-streaming use cases
     - Set context_length based on requirements
     - Savings: 15-30% fewer tokens generated
     
     Guidelines:
       - Chat: max_tokens = 256 (typical response)
       - Summarization: max_tokens = 512
       - Code generation: max_tokens = 1024
       - Analysis: max_tokens = 2048

Model Selection Strategy:
  Cost-Performance Matrix:
    Tier 1: Fastest & Cheapest
      - Claude 3 Haiku ($0.25-$1.25 per 1M)
      - GPT-3.5 Turbo ($0.0005-$0.0015 per 1K)
      - Use for: Summarization, classification, simple QA
      - Savings: 90% vs GPT-4
    
    Tier 2: Balanced
      - Claude 3 Sonnet ($3-$15 per 1M)
      - Mixtral 8x7B ($0.54-$1.08 per 1M)
      - Use for: Complex reasoning, content generation
      - Savings: 70% vs GPT-4
    
    Tier 3: Premium
      - GPT-4 Turbo ($0.01-$0.03 per 1K)
      - Claude 3 Opus ($15-$75 per 1M)
      - Use for: Critical decisions, complex analysis
      - Cost: Full price for quality

  Router Implementation:
    def select_model(query: str) -> str:
      complexity_score = analyze_complexity(query)
      
      if complexity_score < 0.3:
        return "gpt-3.5-turbo"  # Fast & cheap
      elif complexity_score < 0.7:
        return "claude-3-haiku"  # Balanced
      else:
        return "gpt-4-turbo"  # Premium quality
    
    # Track model selection
    model_selected = select_model(user_query)
    query(llm, model=model_selected, ...)
    
    # Estimated savings: 40-60% cost reduction

Batch Processing:
  1. Request Batching:
     - Group requests in non-time-critical scenarios
     - Send in single API call
     - Savings: 20% via batch endpoints
     
     Example:
       # Non-batched
       for item in items:
         result = llm.complete(item)
         results.append(result)
       
       # Batched
       results = llm.batch_complete([
         item for item in items
       ], batch_size=100)
  
  2. Scheduled Processing:
     - Defer non-urgent tasks to off-peak hours
     - Process in bulk during lower-cost windows
     - Savings: 10-30% with reduced inference load
  
  3. Caching & Memoization:
     - Cache identical prompt responses
     - Probability of cache hit: 30-50% in real workloads
     - Savings: 80% on cache hits
     
     Redis Implementation:
       cache_key = hash(model + prompt)
       
       if cache.exists(cache_key):
         return cache.get(cache_key)
       
       result = llm.complete(prompt)
       cache.set(cache_key, result, ttl=3600)
       return result

Retrieval-Augmented Generation (RAG) Optimization:
  1. Reranking:
     - Use lightweight reranking model before LLM
     - Reduce context window by filtering irrelevant docs
     - Savings: 10-20% fewer tokens in prompt
  
  2. Chunking Strategy:
     - Smaller chunks: Higher retrieval cost, lower LLM cost
     - Larger chunks: Lower retrieval cost, higher LLM cost
     - Optimal: ~300-500 tokens per chunk
  
  3. Embedding Caching:
     - Cache vector embeddings for documents
     - Recompute only on document updates
     - Savings: 80-90% on embedding API calls

Infrastructure Cost Optimization:
  1. GPU Utilization:
     - Continuous batching: Keep GPUs >80% utilized
     - Dynamic batching: Adaptive batch size
     - Savings: 30-50% via higher throughput
  
  2. Model Quantization:
     - 8-bit quantization: 50% memory reduction
     - 4-bit quantization: 75% memory reduction
     - Quality impact: <1% degradation
     - Hardware requirement: Runs on cheaper GPUs
  
  3. Specification Downsizing:
     - Smaller models for simpler tasks
     - Example: Haiku vs Opus (30% cost reduction)
     - Trade-off: Quality on complex tasks

Example Optimization Case Study:
  Before Optimization:
    - 1M requests/month
    - 50/50 GPT-4 vs GPT-3.5
    - $5,000/month LLM API cost
    - 87% relevant responses
  
  After Optimization:
    - Prompt compression: -20% tokens
    - Smart routing: 70% GPT-3.5, 30% GPT-4
    - Prompt caching: -15% repeated prompts
    - Reranking: -10% context tokens
    
    Calculation:
      Base: $5,000
      Prompt compression: -$1,000 (20%)
      Smart routing: -$1,500 (30%)
      Caching: -$750 (15%)
      Reranking: -$500 (10%)
      New total: $1,250/month (75% reduction)
```

---

## Part 8: Configuration Templates & Code Examples

### 8.1 Prometheus Configuration

```yaml
# prometheus.yml - Complete LLM monitoring setup
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'
    environment: 'prod'

# Alertmanager configuration
alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - localhost:9093

# Alert rule files
rule_files:
  - '/etc/prometheus/rules/llm_alerts.yml'
  - '/etc/prometheus/rules/slo_alerts.yml'

scrape_configs:
  # vLLM inference server
  - job_name: 'vllm'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['vllm:8000']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'vllm:.*'
        action: keep
    scrape_interval: 5s
    scrape_timeout: 10s

  # TGI inference server
  - job_name: 'tgi'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['tgi:8080']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'tgi_.*'
        action: keep
    scrape_interval: 5s

  # DCGM GPU metrics
  - job_name: 'gpu'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:9400']  # DCGM exporter
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'DCGM_FI_DEV_.*'
        action: keep

  # Application metrics (FastAPI/Flask with Prometheus client)
  - job_name: 'llm-app'
    static_configs:
      - targets: ['localhost:8001']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'llm_.*'
        action: keep

  # Kubernetes service discovery
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: "true"
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      target_label: __address__
```

### 8.2 Grafana Dashboard Definition

```json
{
  "dashboard": {
    "title": "LLM Inference Production Monitoring",
    "tags": ["llm", "inference", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate (RPS)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(llm_requests_total[5m])) by (model)",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "P95 Latency (ms)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(llm_request_latency_seconds_bucket[5m])) by (le, model)) * 1000",
            "legendFormat": "{{model}}"
          }
        ],
        "thresholds": [
          {
            "value": 500,
            "color": "red",
            "op": "gt"
          }
        ]
      },
      {
        "id": 3,
        "title": "TTFT (Time-to-First-Token)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(gen_ai_server_time_to_first_token_bucket[5m])) by (le, model)) * 1000",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Error Rate (%)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "100 * (sum(rate(llm_errors_total[5m])) by (model) / sum(rate(llm_requests_total[5m])) by (model))",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "id": 5,
        "title": "GPU Utilization (%)",
        "type": "gauge",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_UTIL",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ],
        "thresholds": [
          {"value": 0, "color": "green"},
          {"value": 80, "color": "yellow"},
          {"value": 95, "color": "red"}
        ]
      },
      {
        "id": 6,
        "title": "GPU Memory (GB)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_FB_USED / 1024",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ]
      },
      {
        "id": 7,
        "title": "Queue Size",
        "type": "timeseries",
        "targets": [
          {
            "expr": "max(tgi_queue_size) by (instance)",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "id": 8,
        "title": "Tokens Per Second",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(llm_tokens_generated_total[5m])) by (model)",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "id": 9,
        "title": "Cost Per Request ($)",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(llm_cost_usd_total) / sum(llm_requests_total)",
            "legendFormat": "Avg Cost"
          }
        ]
      },
      {
        "id": 10,
        "title": "SLO Error Budget Remaining",
        "type": "gauge",
        "targets": [
          {
            "expr": "100 * (1 - (sum(rate(llm_errors_total[30d])) / sum(rate(llm_requests_total[30d]))))",
            "legendFormat": "Budget %"
          }
        ],
        "thresholds": [
          {"value": 0, "color": "red"},
          {"value": 10, "color": "yellow"},
          {"value": 50, "color": "green"}
        ]
      }
    ]
  }
}
```

### 8.3 OpenTelemetry Instrumentation Example

```python
# FastAPI + OpenTelemetry instrumentation
import os
import time
from typing import Callable
from fastapi import FastAPI, Request
from pydantic import BaseModel

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Initialize resource
resource = Resource.create({
    "service.name": "llm-inference-api",
    "service.version": "1.0.0",
    "deployment.environment": "production",
})

# Initialize tracer provider
trace_exporter = OTLPSpanExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
)
trace_provider = TracerProvider(resource=resource)
trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(trace_provider)

# Initialize meter provider
metrics_exporter = OTLPMetricExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
)
metrics_reader = PeriodicExportingMetricReader(metrics_exporter)
meter_provider = MeterProvider(resource=resource, metric_readers=[metrics_reader])
metrics.set_meter_provider(meter_provider)

# Auto-instrumentation
FastAPIInstrumentor.instrument_app(FastAPI())
HTTPXClientInstrumentor().instrument()
RequestsInstrumentor().instrument()

# Create tracer and meter
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Create metrics
request_counter = meter.create_counter(
    name="gen_ai.client.request.total",
    description="Total number of LLM requests",
    unit="1"
)

token_counter = meter.create_counter(
    name="gen_ai.client.token.usage",
    description="Token usage (input and output)",
    unit="1"
)

latency_histogram = meter.create_histogram(
    name="gen_ai.client.operation.duration",
    description="LLM operation latency",
    unit="ms"
)

cost_counter = meter.create_counter(
    name="llm.cost.usd",
    description="Cost in USD",
    unit="$"
)

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gpt-4-turbo"
    max_tokens: int = 256

@app.post("/v1/completions")
async def generate(request: GenerateRequest):
    start_time = time.time()
    
    with tracer.start_as_current_span("llm.generate") as span:
        # Set span attributes (OpenTelemetry GenAI semantic conventions)
        span.set_attribute("gen_ai.request.model", request.model)
        span.set_attribute("gen_ai.request.max_tokens", request.max_tokens)
        span.set_attribute("gen_ai.system", "OpenAI")
        
        try:
            # Call LLM API (pseudo-code)
            response = call_llm_api(
                model=request.model,
                prompt=request.prompt,
                max_tokens=request.max_tokens
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Set response attributes
            span.set_attribute("gen_ai.response.finish_reason", response.finish_reason)
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
            
            # Update metrics
            request_counter.add(
                1,
                {"gen_ai.request.model": request.model, "status": "success"}
            )
            
            token_counter.add(
                response.usage.prompt_tokens,
                {"gen_ai.request.model": request.model, "type": "input"}
            )
            
            token_counter.add(
                response.usage.completion_tokens,
                {"gen_ai.request.model": request.model, "type": "output"}
            )
            
            latency_histogram.record(
                latency_ms,
                {"gen_ai.request.model": request.model}
            )
            
            # Calculate cost
            cost = calculate_cost(
                request.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            
            cost_counter.add(
                cost,
                {"gen_ai.request.model": request.model}
            )
            
            return {
                "model": request.model,
                "response": response.choices[0].text,
                "usage": response.usage.dict(),
                "cost": cost
            }
        
        except Exception as e:
            request_counter.add(
                1,
                {"gen_ai.request.model": request.model, "status": "error"}
            )
            span.record_exception(e)
            raise

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = {
        "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
    }
    rate = rates.get(model, {})
    return (input_tokens * rate.get("input", 0) +
            output_tokens * rate.get("output", 0))

def call_llm_api(*args, **kwargs):
    # Placeholder - implement actual API call
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 8.4 Alert Rules Configuration

```yaml
# llm_alerts.yml - Production alert rules
groups:
  - name: llm_inference
    interval: 30s
    rules:
      # Latency SLO burn rate alerts
      - alert: LLMHighP95LatencyFastBurn
        expr: |
          (
            histogram_quantile(0.95, sum(rate(llm_request_latency_seconds_bucket[5m])) by (le))
            > 0.5  # 500ms threshold
          )
          and
          (
            histogram_quantile(0.95, sum(rate(llm_request_latency_seconds_bucket[1h])) by (le))
            > 0.5
          )
        for: 5m
        labels:
          severity: critical
          slo: latency
        annotations:
          summary: "P95 latency {{ $value }}s exceeds SLO (5m + 1h burn)"
          runbook: "https://wiki.example.com/runbooks/latency"

      - alert: LLMHighP95LatencySlowBurn
        expr: |
          (
            histogram_quantile(0.95, sum(rate(llm_request_latency_seconds_bucket[30m])) by (le))
            > 0.5
          )
          and
          (
            histogram_quantile(0.95, sum(rate(llm_request_latency_seconds_bucket[6h])) by (le))
            > 0.5
          )
        for: 30m
        labels:
          severity: warning
          slo: latency
        annotations:
          summary: "P95 latency trending high (30m + 6h burn)"

      # Error rate alerts
      - alert: LLMHighErrorRate
        expr: |
          sum(rate(llm_errors_total[5m])) / sum(rate(llm_requests_total[5m])) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate {{ $value | humanizePercentage }} exceeds 1%"

      # GPU utilization alerts
      - alert: GPUHighUtilization
        expr: DCGM_FI_DEV_GPU_UTIL > 95
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu_id }} utilization at {{ $value }}%"
          runbook: "https://wiki.example.com/runbooks/gpu-scaling"

      - alert: GPUMemoryPressure
        expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU {{ $labels.gpu_id }} memory pressure critical"

      # Queue depth alerts
      - alert: LLMQueueBackup
        expr: tgi_queue_size > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM queue size {{ $value }} requests (capacity constraint)"

      # Quality degradation alerts
      - alert: LLMHalluccinationRateSpiking
        expr: |
          (
            sum(rate(llm_hallucination_detected_total[1h])) /
            sum(rate(llm_requests_total[1h]))
          ) > 0.1
        for: 30m
        labels:
          severity: critical
        annotations:
          summary: "Hallucination rate {{ $value | humanizePercentage }} exceeds threshold"

      # Cost alerts
      - alert: LLMCostAnomaly
        expr: |
          abs(
            rate(llm_cost_usd_total[1h]) - 
            avg_over_time(rate(llm_cost_usd_total[24h])[7d:1h])
          ) / avg_over_time(rate(llm_cost_usd_total[24h])[7d:1h]) > 0.3
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Hourly cost {{ $value | humanizeCurrency }} deviates 30% from baseline"

      # TTFT (Time-to-First-Token) alert
      - alert: LLMHighTimeToFirstToken
        expr: |
          histogram_quantile(0.95, 
            sum(rate(gen_ai_server_time_to_first_token_bucket[5m])) by (le)
          ) > 0.2  # 200ms threshold
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "TTFT p95 {{ $value }}s indicates model saturation"
```

---

## Part 9: Authoritative Resources & References

### Blog Posts & Guides (2026)

1. **Monitor LLM Inference in Production: Prometheus & Grafana**
   - URL: https://glukhov.org/observability/monitoring-llm-inference-prometheus-grafana/
   - By: Rost Glukhov
   - Focus: vLLM, TGI, llama.cpp metrics, PromQL examples
   - Date: March 2026

2. **How to Monitor LLMs in Production with Grafana Cloud, OpenLIT, and OpenTelemetry**
   - URL: https://grafana.com/blog/ai-observability-llms-in-production
   - By: Grafana Labs
   - Focus: OpenTelemetry integration, cost management, quality evaluation
   - Date: March 2026

3. **What is LLM Monitoring? (Quality, Cost, Latency, and Drift in Production)**
   - URL: https://www.braintrust.dev/articles/what-is-llm-monitoring
   - By: Braintrust Team
   - Focus: Comprehensive monitoring framework, quality evaluation
   - Date: February 2026

4. **LLM Observability: The Complete Guide to Monitoring LLMs in Production**
   - URL: https://www.swept.ai/post/llm-observability-complete-guide
   - By: Swept AI
   - Focus: Operational + quality + safety observability
   - Date: February 2026

5. **Observability for LLM Systems: Metrics, Traces, Logs, and Testing in Production**
   - URL: https://www.glukhov.org/observability/observability-for-llm-systems/
   - By: Rost Glukhov
   - Focus: End-to-end observability strategy, deployment patterns
   - Date: February 2026

6. **How to Actually Set Meaningful SLOs (Most Teams Are Doing It Wrong)**
   - URL: https://openobserve.ai/blog/set-meaningful-slos/
   - By: OpenObserve
   - Focus: SLO definition, burn rate alerting, practical implementation
   - Date: March 2026

7. **How to Track Token Usage, Prompt Costs, and Model Latency with OpenTelemetry**
   - URL: https://oneuptime.com/blog/post/2026-02-06-track-token-usage-prompt-costs-model-latency-opentelemetry/
   - By: OneUptime
   - Focus: Token tracking, cost attribution, latency measurement
   - Date: February 2026

8. **Tracking LLM Token Usage Across Providers, Teams, and Workloads**
   - URL: https://www.getmaxim.ai/articles/tracking-llm-token-usage-across-providers-teams-and-workloads/
   - By: Maxim AI
   - Focus: Multi-provider cost tracking, billing
   - Date: February 2026

9. **Granular LLM Monitoring: Tracking Token Usage and Latency per User and Feature**
   - URL: https://www.traceloop.com/blog/granular-llm-monitoring-for-tracking-token-usage-and-latency-per-user-and-feature
   - By: Traceloop
   - Focus: Per-user and per-feature attribution
   - Date: October 2025

10. **AI Anomaly Detection: Complete Guide for DevOps & SRE 2026**
    - URL: https://openobserve.ai/blog/ai-anomaly-detection-guide/
    - By: OpenObserve
    - Focus: Anomaly detection techniques, implementation
    - Date: April 2026

11. **Canary Deployment for LLM Features**
    - URL: https://www.featbit.co/ai-llm-canary
    - By: FeatBit
    - Focus: Safe model rollout, shadow testing, gradual deployment
    - Date: March 2026

12. **Best Practices for SLO Scalability in AI Systems**
    - URL: https://techvzero.com/slo-scalability-best-practices-ai-systems/
    - By: TechVZero
    - Focus: SLO scaling for unpredictable AI workloads
    - Date: February 2026

13. **OpenTelemetry LLM Monitoring: Architecture, Implementation, and Tool Comparison**
    - URL: https://spanora.ai/blog/opentelemetry-llm-monitoring
    - By: Spanora
    - Focus: OpenTelemetry architecture, tool selection
    - Date: February 2026

14. **How to Create Latency Monitoring** (LLMOps Series)
    - URL: https://oneuptime.com/blog/post/2026-01-30-llmops-latency-monitoring/
    - By: OneUptime
    - Focus: Latency monitoring setup, SLA management
    - Date: January 2026

15. **Top 7 LLM Observability Tools in 2026**
    - URL: https://www.confident-ai.com/knowledge-base/top-7-llm-observability-tools
    - By: Confident AI
    - Focus: Tool comparison, feature matrix
    - Date: February 2026

### Key Technical Papers

- **Revisiting Service Level Objectives and System Level Metrics in Large Language Model Serving**
  - arXiv: 2410.14257
  - Authors: Zhibin Wang et al., Nanjing University
  - Focus: SLO design for LLM inference, system-level metrics

- **SLOs-Serve: Optimized Serving of Multi-SLO LLMs**
  - arXiv: 2504.08784
  - Authors: Siyuan Chen, Zhipeng Jia (Google), Samira Khan
  - Focus: Multi-tier SLO serving architecture

---

## Part 10: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

```
Goals:
  - Basic operational monitoring in place
  - Understand current baseline metrics
  - Alert system operational

Tasks:
  1. Deploy Prometheus
     - Setup scrape configs for vLLM/TGI/GPU metrics
     - Configure 7-day retention
     - Test metric ingestion
  
  2. Deploy Grafana
     - Add Prometheus as data source
     - Create basic dashboard: latency, error rate, GPU
     - Import community dashboards
  
  3. Instrument application
     - Add OpenTelemetry SDK
     - Emit token counts, latency, cost metrics
     - Configure local trace exporter (stdout)
  
  4. Setup alerts
     - Create AlertManager configuration
     - Alert on error rate > 1%, latency > SLO
     - Integrate with Slack/PagerDuty

  Timeline: 3-5 days
  Resources: 1-2 engineers
```

### Phase 2: Quality Observability (Week 3-4)

```
Goals:
  - Track quality metrics alongside operational metrics
  - Implement cost tracking
  - Establish baseline SLOs

Tasks:
  1. Setup quality evaluation
     - Implement hallucination detection
     - Relevance scoring for RAG
     - Quality dashboard panel
  
  2. Cost attribution
     - Track tokens by model/feature/user
     - Implement cost dashboard
     - Set up weekly cost reports
  
  3. Define SLOs
     - Measure current latency distribution
     - Set initial SLO targets (slightly tighter than baseline)
     - Calculate error budgets
  
  4. Logging & tracing
     - Deploy Tempo or Jaeger
     - Configure OpenTelemetry OTLP export
     - Setup structured logging pipeline

  Timeline: 1-2 weeks
  Resources: 2-3 engineers
```

### Phase 3: Production Safeguards (Week 5-8)

```
Goals:
  - Implement canary deployments
  - Automate error detection
  - Cost optimization in place

Tasks:
  1. Canary deployment system
     - Setup traffic splitting (Istio/FluxCD)
     - Implement automated rollback logic
     - Shadow traffic testing
  
  2. Anomaly detection
     - Implement statistical anomaly detection
     - ML-based anomaly models
     - Automated alerting on drift
  
  3. Cost optimization
     - Prompt compression
     - Smart model routing
     - Prompt caching
     - Implement batch processing
  
  4. Incident response
     - Create runbooks for common issues
     - Regular incident simulations
     - Post-mortem process

  Timeline: 2-4 weeks
  Resources: 3-4 engineers
```

### Phase 4: Scale & Maturity (Week 9+)

```
Goals:
  - Multi-region monitoring
  - Automated cost management
  - Predictive scaling

Tasks:
  1. Multi-region/federation
     - Setup Prometheus federation
     - Cross-region SLO tracking
     - Geo-distributed alerting
  
  2. Advanced cost management
     - ML-based cost forecasting
     - Automated budget alerts
     - Usage-based scaling
  
  3. Predictive monitoring
     - Forecast capacity needs
     - Predict quality degradation
     - Cost trend analysis
  
  4. Continuous improvement
     - Regular SLO reviews
     - Cost optimization sprints
     - Tool consolidation

  Timeline: Ongoing
  Resources: 1-2 engineers (maintenance)
```

---

## Conclusion

LLMOps and production monitoring in 2026 require a multi-layered approach combining infrastructure monitoring, application instrumentation, quality evaluation, and cost tracking. Success comes from:

1. **Early implementation**: Instrument from day one—retrofitting is expensive
2. **User-centric metrics**: Track what users experience, not just infrastructure health
3. **Automation**: Alert on burn rate, not raw thresholds
4. **Cost discipline**: Every token has a cost; make it visible
5. **Quality vigilance**: LLMs fail silently—quality metrics are critical
6. **Production-grade SLOs**: SLOs drive behavior and culture

The tools ecosystem is mature. The challenge is discipline: defining clear metrics, setting achievable targets, and building the observability culture that makes production LLM systems reliable, cost-effective, and trustworthy.

---

## Appendix: Quick Reference Checklists

### Pre-Production Checklist

- [ ] Prometheus/Grafana deployed
- [ ] Token/cost tracking implemented
- [ ] Latency SLOs defined
- [ ] Quality baseline established
- [ ] Alerting rules configured
- [ ] Runbooks written for common failures
- [ ] Incident response process defined
- [ ] Cost budget approved
- [ ] Canary deployment tested
- [ ] Logs/traces pipeline working

### Monthly Review Checklist

- [ ] SLO attainment vs targets
- [ ] Error budget status
- [ ] Cost trends vs budget
- [ ] Quality metric trends
- [ ] Incident postmortems completed
- [ ] Alert fatigue evaluation
- [ ] Capacity planning for next quarter
- [ ] Cost optimization opportunities
- [ ] Tool/service changes required
- [ ] Training needs identified
