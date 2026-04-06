# Model Configs

This directory defines model catalogs and profile presets used by training and inference flows.

## Files

- model_registry.yaml: canonical inventory of base, embedding, and reranker models.
- fine_tuning_profiles.yaml: reproducible fine-tuning parameter presets.
- inference_profiles.yaml: serving/runtime presets for latency-throughput tradeoffs.

## Rules

- Every model entry should have license metadata and intended use.
- Use immutable model identifiers (revision pins where possible).
- Keep profile names stable to avoid pipeline drift.
