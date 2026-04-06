# Prompt System

This directory stores all human-readable instructions that shape agent behavior.

## Layout

- roles/: system prompts for agent roles.
- tasks/: task-specific prompt overlays.
- shared/: reusable contracts and policy snippets.

## Prompt Authoring Rules

1. Keep role prompts stable and generic.
2. Keep task prompts narrow and outcome-oriented.
3. Put formatting and output contracts in shared files, not duplicated per role.
4. Use explicit placeholders with double braces, for example {{goal}}.
5. Never require a role to invent evidence or URLs.

## Recommended Prompt Composition

A run should combine:

- one role prompt
- one task prompt
- shared output contract
- shared tool-use policy

## Placeholder Conventions

Use these standard placeholders across prompts:

- {{goal}}
- {{constraints}}
- {{acceptance_criteria}}
- {{available_tools}}
- {{risk_level}}
- {{time_budget}}
- {{cost_budget}}

## Change Policy

If a prompt change impacts output shape, update:

- shared/output_contracts.md
- relevant evaluation rules in evaluation/judges/rule_checks.yaml
