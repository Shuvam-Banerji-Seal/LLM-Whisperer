# vLLM Engine

Author: Shuvam Banerji Seal

vLLM is the default serving path in this scaffold because it combines strong
throughput/latency behavior with OpenAI-compatible endpoints.

## Features Covered

- OpenAI-compatible server startup.
- Scheduler and memory controls exposed as environment variables.
- Basic chat-completion smoke validation.

## Key References

- https://docs.vllm.ai/en/stable/
- https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- https://docs.vllm.ai/en/stable/configuration/serve_args/

## Run

```bash
bash scripts/start_vllm_server.sh
```

## Smoke Test

```bash
python scripts/smoke_openai_chat.py \
  --base-url http://localhost:8000/v1 \
  --model local-model
```
