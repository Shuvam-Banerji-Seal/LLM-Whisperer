# Configuration System

This directory holds reusable, environment-aware configuration for LLM Whisperer.

## Structure

- environments/: deployment/environment overlays (local, dev, staging, prod).
- models/: model catalogs and training/inference profile presets.
- datasets/: dataset registry, split policy, and quality thresholds.
- runtime/: execution controls for agents, RAG, inference, and observability.

## Layering Strategy

Apply configuration in this order:

1. global defaults from each domain config file
2. environment overlay from environments/*.yaml
3. runtime overrides from CLI or CI variables (if explicitly allowed)

## Naming Rules

- Use snake_case keys.
- Keep numeric thresholds explicit and unit-labeled where relevant.
- Keep secrets out of files; pass via environment variables.

## Secret Handling

Use placeholders like ${VAR_NAME} in config files and resolve in runtime bootstrap code.

## Recommended Validation

- YAML syntax validation in CI
- Schema validation for machine-enforced configs
- Drift checks between environment overlays and production runtime
