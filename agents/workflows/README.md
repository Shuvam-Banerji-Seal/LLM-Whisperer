# Workflows

This directory defines reusable multi-step orchestration graphs.

## Design Rules

1. Keep step IDs stable for traceability.
2. Keep role assignment explicit per step.
3. Put reusable gates in shared/gates.yaml.
4. Keep workflow definitions declarative.

## Workflow Lifecycle

- planned
- executing
- blocked
- completed
- failed

## Validation

Each workflow should conform to workflow.schema.yaml.
