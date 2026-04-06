# Environment Overlays

These files define environment-specific values that should not be hard-coded in code.

## Files

- local.yaml: single-machine development settings.
- dev.yaml: shared development/staging-like defaults.
- staging.yaml: pre-production integration and load testing.
- prod.yaml: production-safe constraints and stricter controls.

## Usage

Load one overlay and merge with domain defaults from models/, datasets/, and runtime/.

## Guardrails

- Keep production more restrictive than development.
- Keep endpoint URLs and bucket names explicit per environment.
- Keep credentials external via environment variables.
