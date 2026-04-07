# Production Operations: LLM Monitoring & Observability

Comprehensive monitoring, logging, and observability for production LLM systems.

## Overview

This implementation covers production monitoring essentials:
- **Prometheus Metrics** - Standard metrics for observability
- **Grafana Dashboards** - Real-time visualization
- **OpenTelemetry Tracing** - Distributed tracing across services
- **Cost Tracking** - Token usage and cost monitoring
- **Alert Management** - Proactive issue detection
- **Performance Analysis** - Latency and throughput tracking
- **Quality Metrics** - Model accuracy and hallucination tracking

## Files Included

```
production-ops/
├── llm-monitoring-complete.py    # Complete implementation (519 lines)
├── README.md                     # This file
└── Examples:
    ├── Prometheus setup
    ├── Grafana dashboards
    ├── OpenTelemetry tracing
    ├── Cost tracking
    ├── Alert configuration
    ├── Performance analysis
    └── Quality monitoring
```

## Key Components

### 1. Prometheus Metrics

Standard metrics for LLM inference:

```python
from llm_monitoring_complete import PrometheusMetrics, MonitoringConfig

config = MonitoringConfig(
    enable_prometheus=True,
    metrics_port=8000,
    alert_latency_p95_ms=2000.0,
    cost_per_million_tokens=0.01
)

metrics = PrometheusMetrics(config)

# Metrics automatically tracked:
# - llm_requests_total (by model, status, endpoint)
# - llm_request_duration_seconds (p50, p95, p99)
# - llm_tokens_generated_total (by model)
# - llm_input_tokens_total (by model)
# - gpu_memory_used_bytes (by GPU)
# - gpu_memory_available_bytes (by GPU)
# - llm_cost_usd_total (cumulative cost)

# Record inference
result = model.generate(prompt, max_tokens=256)
metrics.record_inference(
    model_name="llama-7b",
    input_tokens=len(prompt.split()),
    output_tokens=len(result.split()),
    latency_ms=145,
    status="success"
)

# Expose metrics endpoint
# GET http://localhost:8000/metrics
```

**Key Metrics to Track**:

| Metric | Alert Threshold | Good Value | Action |
|--------|-----------------|-----------|--------|
| P95 Latency | >2000ms | <500ms | Scale up, optimize |
| Error Rate | >5% | <1% | Investigate, rollback |
| GPU Memory | >95% | 70-85% | Scale or quantize |
| Throughput | <1000 tok/s | >5000 tok/s | Load balance |
| Cost/M Tokens | >$0.05 | <$0.02 | Optimize model |

### 2. Grafana Dashboards

Real-time visualization of metrics:

```python
# Example dashboard queries:

# Latency over time
query: histogram_quantile(0.95, rate(llm_request_duration_seconds[5m]))

# Throughput
query: rate(llm_tokens_generated_total[5m])

# Error rate
query: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m])

# Cost tracking
query: increase(llm_cost_usd_total[1h])

# GPU utilization
query: gpu_utilization_percent{gpu_id="0"}
```

**Dashboard Setup**:
1. Install Grafana
2. Add Prometheus data source
3. Import dashboard JSON
4. Configure alerts
5. Share with team

### 3. OpenTelemetry Tracing

Distributed tracing for request journey:

```python
from llm_monitoring_complete import OpenTelemetryTracer

tracer = OpenTelemetryTracer(
    service_name="llm-inference-server",
    exporter_endpoint="http://jaeger:6831"
)

# Trace request through system
with tracer.trace_request("generate", request_id="req_123") as span:
    span.add_attribute("model", "llama-7b")
    span.add_attribute("prompt_length", 150)
    
    # Sub-spans for components
    with tracer.start_span("tokenization"):
        tokens = tokenize(prompt)
    
    with tracer.start_span("inference"):
        logits = model(tokens)
    
    with tracer.start_span("decoding"):
        output = decode(logits)
    
    span.add_attribute("output_length", len(output))
```

**Benefits**:
- ✅ See exact latency breakdown
- ✅ Identify bottlenecks (which step is slow?)
- ✅ Correlate errors to system components
- ✅ Debug performance issues

### 4. Cost Tracking

Monitor token usage and costs:

```python
from llm_monitoring_complete import CostTracker

tracker = CostTracker(
    cost_per_input_token=0.0005,    # $0.0005 per input token
    cost_per_output_token=0.0015,   # $0.0015 per output token
    cost_per_million_tokens=15.0    # Or use flat rate
)

# Track each inference
result = model.generate(prompt, max_tokens=256)
cost = tracker.track_inference(
    input_tokens=len(prompt.split()),
    output_tokens=len(result.split()),
    model="llama-7b"
)

# Generate cost reports
daily_costs = tracker.get_daily_costs()
model_costs = tracker.get_costs_by_model()
endpoint_costs = tracker.get_costs_by_endpoint()

print(f"Today's costs: ${daily_costs['total']:.2f}")
print(f"Cost per request: ${daily_costs['average']:.4f}")
```

**Cost Optimization**:
- Monitor cost per model
- Identify expensive endpoints
- Set budgets and alerts
- Optimize expensive patterns

### 5. Quality Metrics

Track model accuracy and hallucination:

```python
from llm_monitoring_complete import QualityMetrics

quality = QualityMetrics()

# Track correctness on evaluation set
for prompt, reference_output in eval_set:
    prediction = model.generate(prompt)
    
    # Compute metrics
    bleu = quality.compute_bleu(prediction, reference_output)
    rouge = quality.compute_rouge(prediction, reference_output)
    f1 = quality.compute_f1(prediction, reference_output)
    
    # Track
    quality.record(
        model="llama-7b",
        bleu_score=bleu,
        rouge_score=rouge,
        f1_score=f1
    )

# Get quality dashboard
report = quality.get_quality_report()
print(f"Average BLEU: {report['bleu_mean']:.3f}")
```

### 6. Alert Management

Automated alerting for issues:

```python
from llm_monitoring_complete import AlertManager

alerts = AlertManager(
    slack_webhook="https://hooks.slack.com/...",
    pagerduty_key="https://events.pagerduty.com/..."
)

# Define alerts
alerts.add_alert(
    name="high_latency",
    condition="p95_latency > 2000",
    severity="warning",
    notification="Slack"
)

alerts.add_alert(
    name="high_error_rate",
    condition="error_rate > 0.05",
    severity="critical",
    notification="PagerDuty"
)

# Alerts trigger automatically based on metrics
# → Slack notification if P95 latency > 2 seconds
# → PagerDuty page if error rate > 5%
```

## Quick Start

### Local Monitoring Setup

```bash
# 1. Start Prometheus
docker run -d -p 9090:9090 prom/prometheus

# 2. Start Grafana
docker run -d -p 3000:3000 grafana/grafana

# 3. Run LLM with metrics
from llm_monitoring_complete import PrometheusMetrics
metrics = PrometheusMetrics()

# 4. Access
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

### Production Monitoring Stack

```yaml
version: '3'
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
  
  jaeger:
    image: jaegertracing/all-in-one
    ports:
      - "6831:6831/udp"
      - "16686:16686"
  
  llm-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PROMETHEUS_PORT=8000
      - JAEGER_ENDPOINT=jaeger:6831
```

## Key Monitoring Dashboards

### Dashboard 1: Overview (Top-level health)
- P50/P95/P99 latency over time
- Request success rate
- Current throughput (tokens/sec)
- Error log (last 10 errors)

### Dashboard 2: Performance (Latency breakdown)
- Time-to-first-token (TTFT)
- Time-per-token (TPS)
- Batch size over time
- GPU utilization

### Dashboard 3: Cost (Usage and spend)
- Tokens generated per hour
- Daily cost trend
- Cost per model
- Cost per endpoint

### Dashboard 4: Quality (Model behavior)
- Accuracy metrics over time
- Hallucination rate
- Token length distribution
- Common error patterns

## Common Patterns

### Pattern 1: Baseline & Monitoring
```python
# Start with basic metrics
metrics = PrometheusMetrics()

# Track each request
for request in requests:
    result = model(request)
    metrics.record_inference(...)

# Visualize in Grafana
# Alert if issues detected
```

### Pattern 2: Progressive Monitoring
```
Week 1: Collect metrics
Week 2: Set baseline values
Week 3: Create dashboards  
Week 4: Configure alerts
```

### Pattern 3: Incident Response
```
1. Alert triggered (e.g., P95 > 2s)
2. Grafana dashboard shows root cause
3. Jaeger trace identifies bottleneck
4. Auto-scale or rollback triggered
5. Incident resolved in <5 minutes
```

## Best Practices

### 1. Metric Cardinality Control
```python
# Bad: Unlimited dimensions
labels=["request_id", "user_id", "timestamp"]  # Millions of combinations

# Good: Limited, meaningful dimensions
labels=["model", "endpoint", "status"]  # Hundreds of combinations
```

### 2. Alert Tuning
```python
# Too many false positives:
alert: p95_latency > 100ms  # Too sensitive

# Good threshold based on baseline:
alert: p95_latency > baseline * 1.5  # 50% increase
```

### 3. Dashboard Design
```
1. High-level health (1 dashboard)
2. Detailed performance (1 dashboard)  
3. Cost tracking (1 dashboard)
4. Quality metrics (1 dashboard)
5. On-call runbook (linked from alerts)
```

## Troubleshooting

**Q: Too many alerts?**
- Increase thresholds (reduce sensitivity)
- Add time windows (alert only if sustained)
- Filter to critical metrics only

**Q: Metrics missing?**
- Check if collection is running
- Verify network connectivity to prometheus
- Enable debug logging

**Q: High cardinality causing issues?**
- Remove high-cardinality labels
- Aggregate dimensions
- Use label dropping rules

## References

- **Prometheus**: [Monitoring with Prometheus](https://prometheus.io/docs/introduction/overview/)
- **Grafana**: [Creating Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- **OpenTelemetry**: [Distributed Tracing](https://opentelemetry.io/docs/)
- **Jaeger**: [Getting Started](https://www.jaegertracing.io/docs/getting-started/)

## Integration with Other Skills

- **Infrastructure**: Deploy server with monitoring
- **Fast Inference**: Track optimization impact on latency
- **Fine-Tuning**: Monitor quality metrics post-tuning
- **Quantization**: Track accuracy vs compression trade-off
- **RAG**: Monitor retrieval quality and accuracy

