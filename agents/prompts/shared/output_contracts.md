# Shared Output Contracts

Use these role-specific contracts unless a task file provides stricter requirements.

## Orchestrator Contract

```yaml
status: planned|executing|blocked|done
workflow_id: string
next_step: string
delegations:
  - role: string
    objective: string
    done_definition: string
risks:
  - string
```

## Planner Contract

```yaml
assumptions:
  - string
task_breakdown:
  - id: string
    action: string
    owner_role: string
acceptance_criteria:
  - string
validation_plan:
  - string
rollback_plan:
  - string
```

## Researcher Contract

```yaml
findings:
  - claim: string
    evidence_url: string
    confidence: low|medium|high
alternatives:
  - option: string
    tradeoffs: string
recommendation: string
open_questions:
  - string
```

## Implementer Contract

```yaml
changes:
  - target: string
    action: create|modify|remove
    rationale: string
tests:
  - name: string
    scope: unit|integration|e2e
known_risks:
  - string
```

## Reviewer Contract

```yaml
findings:
  - severity: critical|high|medium|low
    issue: string
    evidence: string
    suggested_fix: string
approval: true|false
required_followups:
  - string
```

## Safety Contract

```yaml
policy_status: pass|fail|needs_human_review
violations:
  - id: string
    reason: string
required_controls:
  - string
```
