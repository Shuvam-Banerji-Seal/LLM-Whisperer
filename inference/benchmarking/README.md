# Benchmarking Suite

Author: Shuvam Banerji Seal

This suite measures latency, throughput, and reliability for OpenAI-compatible
inference endpoints.

## Included Assets

- Workload template: `workloads/chat_small.jsonl`
- Scenario matrix: `configs/benchmark_matrix.yaml`
- Metric schemas: `schemas/*.json`
- Runner: `scripts/run_openai_benchmark.py`
- Aggregator: `scripts/summarize_results.py`
- Comparator: `scripts/compare_runs.py`

## Metrics

- `latency_ms`: full request latency.
- `ttft_ms`: approximated TTFT when streaming is unavailable.
- `tpot_ms`: approximate time per output token.
- `success`: request success indicator.
- `status_code`: HTTP status code.
- `completion_tokens`: generated token count when available.

## Run

```bash
python scripts/run_openai_benchmark.py \
  --base-url http://localhost:8000/v1 \
  --api-key local-dev-key \
  --model local-model \
  --workload workloads/chat_small.jsonl \
  --output results/latest.ndjson \
  --concurrency 8

python scripts/summarize_results.py \
  --input results/latest.ndjson \
  --output results/latest_summary.json
```

## Compare

```bash
python scripts/compare_runs.py \
  --baseline results/baseline_summary.json \
  --candidate results/latest_summary.json
```
