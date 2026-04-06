# Runtime Configs

This directory defines execution behavior for core platform components.

## Files

- agent_runtime.yaml: orchestration behavior and safety gates for agent workflows.
- rag_runtime.yaml: retrieval and generation defaults for RAG systems.
- inference_runtime.yaml: serving limits and decoding defaults.
- observability.yaml: logging, tracing, metrics, and alerting defaults.

## Policy

- Runtime values may be overridden by environment overlays only where explicitly allowed.
- High-risk changes should require approval and evaluation reruns.
