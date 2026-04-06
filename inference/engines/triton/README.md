# Triton Inference Server Engine

Author: Shuvam Banerji Seal

Triton provides a production model-serving control plane with model repository
management and HTTP/gRPC APIs.

## Features Covered

- Server startup template with HTTP, gRPC, and metrics ports.
- Minimal Python backend model repository example (`echo`).
- HTTP infer smoke test against `/v2/models/<name>/infer`.

## Key References

- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/
- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html
- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/perf_analyzer.html

## Run

```bash
bash scripts/start_triton_server.sh
```

## Smoke Test

```bash
python scripts/smoke_infer.py --url http://localhost:8001 --model echo
```
