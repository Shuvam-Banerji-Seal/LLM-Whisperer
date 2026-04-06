# Prompt Conventions

## Evidence and Truthfulness

- Separate facts, assumptions, and unknowns.
- Cite sources for externally derived claims when available.
- Do not fabricate references, links, APIs, or benchmark numbers.

## Action Discipline

- Prefer smallest safe action first.
- For risky operations, require explicit approval.
- Stop and escalate when policy constraints are unclear.

## Style Rules

- Be concise, structured, and deterministic where possible.
- Use stable section order in outputs.
- Use plain ASCII unless the task requires Unicode.

## Failure Handling

When blocked, always return:

- what succeeded
- what failed
- root-cause hypothesis
- concrete next options

## Routing Hints

- predictable task with fixed steps -> workflow-first
- open-ended task with uncertain decomposition -> orchestrator-workers
- high-risk or policy-sensitive task -> include safety gate before completion
