# Task Prompt: RAG Improvement

## Objective

Improve retrieval and grounded generation quality for {{goal_dataset_or_domain}}.

## Required Inputs

- baseline metrics (at least recall@k and groundedness)
- corpus manifest
- target query set
- latency and cost bounds

## Execution Checklist

1. Measure baseline quality.
2. Diagnose likely bottlenecks: ingestion, chunking, retrieval, reranking, or generation prompt.
3. Apply one controlled change at a time where possible.
4. Re-evaluate against baseline.
5. Run safety checks for hallucination and unsupported citation risk.

## Completion Criteria

- measurable quality gain vs baseline
- no safety regressions
- no unacceptable latency or cost regression
