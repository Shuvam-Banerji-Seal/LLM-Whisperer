# Latency Evaluation

Comprehensive performance benchmarking for LLM systems with latency metrics, throughput measurement, and SLA monitoring.

## Overview

Latency evaluation tracks key performance indicators:

| Metric | Unit | Definition | Target |
|--------|------|-----------|--------|
| **TTFT** | ms | Time to First Token | <100ms |
| **TPOT** | ms | Time Per Output Token | <20ms |
| **ITL** | ms | Inter-Token Latency | <20ms |
| **Throughput** | req/s | Requests per second | >10 req/s |
| **Goodput** | tokens/s | Successful tokens/sec | >100 tokens/s |

## Quick Start

### Basic Latency Measurement

```python
from evaluation.latency import RequestTrace, LatencyMetricsComputer, LatencyRunner

# Create request traces (from actual server logs)
traces = [
    RequestTrace(
        request_id="req_1",
        prompt_tokens=100,
        completion_tokens=50,
        ttft_ms=45.2,
        tpot_ms=15.3,
        total_time_ms=810.0,
        success=True
    ),
    # ... more traces
]

# Compute metrics
metrics = LatencyMetricsComputer.compute_metrics(traces)
print(f"TTFT P95: {metrics.ttft_p95:.2f}ms")
print(f"Throughput: {metrics.throughput_req_per_sec:.2f} req/s")

# Run with runner
runner = LatencyRunner()
result = runner.run_benchmark(traces)
runner.print_summary(result)
```

### SLA Monitoring

```python
from evaluation.latency import SLAChecker

# Define SLA thresholds
sla = SLAChecker(SLAChecker.SLAThresholds(
    ttft_p99_ms=200.0,
    tpot_p99_ms=50.0,
    throughput_min_req_per_sec=10.0,
    goodput_min_tokens_per_sec=100.0,
    success_rate_min=0.99
))

# Check metrics against SLA
passed = sla.all_passed(metrics)
print(f"SLA Status: {'PASSED' if passed else 'FAILED'}")

# Run with SLA check
result = runner.run_benchmark(traces, sla_checker=sla)
```

## Key Metrics

### Time to First Token (TTFT)

Time from request submission to receiving the first output token.

**Formula:** 
```
TTFT = end_time_first_token - request_submit_time
```

**Typical values:**
- Fast: <50ms
- Normal: 50-100ms  
- Slow: >100ms

**P95/P99 targets:**
- Interactive apps: P99 <200ms
- Batch processing: P99 <500ms

### Time Per Output Token (TPOT)

Average time to generate each output token after the first.

**Formula:**
```
TPOT = (total_completion_time - ttft) / num_completion_tokens
```

**Typical values:**
- Fast: <15ms
- Normal: 15-30ms
- Slow: >30ms

**P95/P99 targets:**
- 32K context: P99 <100ms
- Streaming: P99 <50ms

### Throughput

Number of requests and tokens processed per second.

**Formulas:**
```
Throughput (req/s) = num_requests / total_time_seconds
Throughput (tokens/s) = total_tokens / total_time_seconds
Goodput (tokens/s) = successful_tokens / total_time_seconds
```

**Typical values:**
- Per GPU: 10-100 req/s
- Per instance: 100-1000 tokens/s

## Configuration

### sla_config.yaml

```yaml
sla:
  name: "Production SLA"
  ttft:
    p99_ms: 200
    p95_ms: 100
  tpot:
    p99_ms: 50
    p95_ms: 30
  throughput:
    min_req_per_sec: 10
    min_tokens_per_sec: 100
  goodput:
    min_tokens_per_sec: 90
  success_rate_min: 0.99

monitoring:
  sample_interval_seconds: 10
  alert_on_breach: true
  dashboard_url: "http://grafana:3000"
```

## Profiling Request Traces

### Collecting TTFT

```python
import time

start = time.time()
# ... call LLM
first_token_time = time.time()
ttft_ms = (first_token_time - start) * 1000
```

### Collecting TPOT

```python
first_token_time = time.time()
# ... generate remaining tokens
last_token_time = time.time()

remaining_time = (last_token_time - first_token_time) * 1000
num_remaining_tokens = num_completion_tokens - 1
tpot_ms = remaining_time / num_remaining_tokens
```

### Computing from vLLM

vLLM provides detailed timing information:

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-2-7b-hf")
request_id = llm.generate(...)

# Access metrics
metrics = llm.llm_engine.get_num_unfinished_requests()
```

## CI/CD Integration

Monitor latency regressions in CI/CD:

```yaml
name: Latency Check

on: [push]

jobs:
  latency:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run latency benchmark
        run: |
          python -m pytest evaluation/latency/tests/ -v --tb=short
      
      - name: Check SLA compliance
        run: |
          python evaluation/latency/check_sla.py \
            --baseline baseline_metrics.json \
            --current results/current_metrics.json \
            --max-regression 10
      
      - name: Upload metrics
        uses: actions/upload-artifact@v2
        with:
          name: latency-metrics
          path: results/
```

## References

- vLLM Benchmarking: https://docs.vllm.ai/en/latest/performance.html
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- LLM Locust: https://github.com/ray-project/llm-perf
- GenAI Perf: https://github.com/NVIDIA/GenerativeAI-Perf
