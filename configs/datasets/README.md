# Dataset Configs

This directory tracks dataset metadata, split strategy, and quality checks.

## Files

- dataset_registry.yaml: authoritative metadata for source and processed datasets.
- split_policy.yaml: default split behavior and leakage prevention rules.
- quality_gates.yaml: thresholds that must pass before training/evaluation.

## Rules

- Keep lineage from raw to processed datasets.
- Record pii_risk and license for every dataset.
- Use stable dataset IDs in experiments.
