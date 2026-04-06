# Evaluation Layer

This directory defines objective quality gates for agent workflows.

## Goals

- Prevent regressions before promotion.
- Standardize scoring across workflows.
- Combine deterministic checks and rubric-based grading.

## Components

- rubric.md: human-readable scoring criteria.
- score_weights.yaml: machine-readable weights and threshold profiles.
- benchmark.schema.yaml: schema for benchmark case files.
- benchmark_manifest.yaml: source of truth for benchmark suites.
- cases/: benchmark cases per workflow family.
- judges/: deterministic and LLM-judge policies.
- reports/: standard report templates.
- scripts/: helper code for local scoring and CI hooks.

## Reference Basis

This layer is informed by:

- HELM multi-metric evaluation perspective: https://arxiv.org/abs/2211.09110
- Reliability and incident discipline: https://sre.google/sre-book/table-of-contents/
- Security governance references: https://www.nist.gov/itl/ai-risk-management-framework and https://owasp.org/www-project-top-10-for-large-language-model-applications/

## Release Rules

A run should not be promoted if any of the following is true:

- overall weighted score is below threshold
- safety score is below threshold
- any critical reviewer finding remains unresolved
