# Agent Configs

This directory contains machine-readable controls for agent behavior.

## File Groups

- schemas/: JSON-Schema-style contracts for config validation.
- profiles/: environment overlays.
- top-level YAML files: active config values.

## Precedence

1. base config (top-level)
2. environment profile overlay
3. runtime override (if explicitly allowed)

## Validation Rule

All top-level config files should validate against the corresponding schema in schemas/.

## Security Notes

- Keep secrets out of this directory.
- Put credentials in environment-specific secret managers.
- Treat tool_policy.yaml as an enforcement artifact, not documentation.
