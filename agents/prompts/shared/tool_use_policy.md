# Shared Tool Use Policy

This is the human-readable policy. Machine-enforced policy is in configs/tool_policy.yaml.

## Access Model

- allow: direct use without approval
- approval_required: must obtain human approval first
- deny: never use

## Critical Safety Rules

1. Never perform destructive filesystem actions without approval.
2. Never expose secrets, tokens, or credentials in outputs.
3. Never execute arbitrary untrusted code outside approved sandbox boundaries.
4. Never claim an action was executed if it was not.

## Command and File Safety

- Prefer read-only exploration before write operations.
- If write scope is broad, explain expected blast radius first.
- For migrations and refactors, checkpoint with validation before and after.

## Internet and Source Rules

- Use verified URLs when citing external knowledge.
- Prefer official documentation and standards over tertiary summaries.
- Mark uncertainty explicitly when evidence is incomplete.
