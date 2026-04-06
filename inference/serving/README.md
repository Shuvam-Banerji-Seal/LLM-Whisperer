# Serving Templates

Author: Shuvam Banerji Seal

This module provides deployment templates and operational probes for
OpenAI-compatible serving stacks.

## Included Artifacts

- Docker template for vLLM serving.
- Docker Compose runtime example with GPU reservation.
- Kubernetes Deployment + Service template.
- Health check and lightweight load probe scripts.
- Local mock OpenAI-compatible server for benchmark dry-runs.

## Run Local Mock Endpoint

```bash
python scripts/mock_openai_server.py --host 127.0.0.1 --port 18000 --model mock-local-model
```

Then run smoke and benchmark scripts against `http://127.0.0.1:18000/v1`.

## Run Local Container Stack

```bash
docker compose -f docker/docker-compose.vllm.yaml up --build
```

## Validate Health

```bash
python scripts/healthcheck.py --base-url http://localhost:8000
```

## Run Load Probe

```bash
python scripts/load_probe.py \
  --base-url http://localhost:8000/v1 \
  --api-key local-dev-key \
  --model local-model \
  --requests 50 \
  --concurrency 8
```
