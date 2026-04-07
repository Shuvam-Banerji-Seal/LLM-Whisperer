# LLM Regression Testing & Quality Gates

## Overview

Regression testing for LLM systems ensures that model updates, prompt changes, or pipeline modifications don't degrade performance over time. Unlike traditional software testing, LLM regression requires statistical approaches that account for the probabilistic nature of model outputs.

## Key Concepts

### Golden Datasets
Golden datasets are curated test sets that represent real production traffic and critical use cases. They serve as the baseline for comparing model performance across releases.

**Characteristics of effective golden datasets:**
- Representative of actual production traffic
- Stratified by intent, severity, and business value
- Versioned and reproducible
- Privacy-safe (redacted PII)
- Labeled with expected outcomes

### Quality Gates
Quality gates are automated checkpoints in CI/CD pipelines that block deployments when evaluation scores fall below defined thresholds.

**Types of thresholds:**
- **Absolute thresholds**: e.g., accuracy > 80%
- **Relative thresholds**: e.g., delta < 5% from baseline
- **Statistical significance**: p-value testing

### Drift Detection
Monitoring for distribution shifts in input data or model behavior over time using statistical methods.

## Contents

1. `golden_dataset_builder.py` - Tools for creating golden datasets from production logs
2. `regression_test_framework.py` - Pytest-based regression testing framework
3. `quality_gate_config.py` - Quality gate configuration and threshold management
4. `drift_detection.py` - Statistical drift detection methods
5. `ci_cd_integration/` - CI/CD pipeline examples (GitHub Actions, GitLab CI)
6. `reporting/` - Score visualization and reporting utilities

## Quick Start

```python
from regression import RegressionTestSuite, GoldenDataset, QualityGate

# Load golden dataset
dataset = GoldenDataset.load("golden_v1.0.json")

# Run regression tests
suite = RegressionTestSuite(
    dataset=dataset,
    metrics=["accuracy", "coherence", "relevance"],
    baseline_version="v1.2.0"
)

# Check quality gates
gate = QualityGate(
    absolute_threshold={"accuracy": 0.80},
    relative_threshold={"coherence": -0.05},  # Max 5% degradation
    p_value_threshold=0.05
)

results = suite.run()
gate.check(results)  # Raises if gates fail
```

## References

- [DeepEval CI/CD Guide](https://docs.confident-ai.com/guides/guides-regression-testing-in-cicd)
- [OptyxStack Golden Dataset Guide](https://optyxstack.com/llm-evaluation/golden-dataset-real-user-logs-regression-testing)
- [Microsoft GenAI Evals](https://github.com/microsoft/genai-evals)
