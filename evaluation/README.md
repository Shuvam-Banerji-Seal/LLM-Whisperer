# Evaluation

Comprehensive LLM evaluation framework covering task benchmarks, LLM-as-judge methodology, safety checks, latency monitoring, and regression testing.

**Author:** Shuvam Banerji Seal

## Overview

This module provides a complete evaluation stack for assessing language models across multiple dimensions:

| Dimension | Purpose | Key Metrics |
|-----------|---------|-------------|
| Task Benchmarks | Domain-specific performance assessment | Accuracy, F1, Exact Match, Pass@k |
| LLM-as-Judge | Scalable quality assessment using AI evaluators | Relevance, Coherence, Helpfulness, Faithfulness |
| Safety | Harm prevention, policy compliance, jailbreak resistance | Toxicity, Bias, Refusal Rate, Jailbreak Success Rate |
| Latency | Runtime performance and SLA compliance | P50, P95, P99 latency, Throughput, TTFT |
| Regression | CI-compatible quality gates | Score delta, Pass/Fail rate, Drift detection |

## Quick Start

```python
from evaluation.llm_as_judge import LLMJudgeConfig, LLMJudgeEvaluator
from evaluation.task_benchmarks import BenchmarkRunner
from evaluation.safety import SafetyEvaluator
from evaluation.latency import LatencyBenchmark

# Run LLM-as-Judge evaluation
judge = LLMJudgeEvaluator(config=LLMJudgeConfig(
    judge_model="gpt-4o",
    rubrics=["answer_relevance", "task_completion", "faithfulness"],
    scoring_scale=5
))
results = judge.evaluate(dataset="eval_dataset.jsonl")

# Run safety checks
safety = SafetyEvaluator()
safety_results = safety.run_checks(model_outputs)
```

## Folder Structure

```
evaluation/
├── README.md                          # This file
├── task_benchmarks/                   # Domain/task-specific benchmark suites
│   ├── README.md
│   ├── src/
│   │   ├── __init__.py
│   │   ├── benchmark_runner.py        # Main benchmark orchestration
│   │   ├── academic.py                # MMLU, GPQA, GSM8K, MATH
│   │   ├── code.py                    # HumanEval, SWE-bench, APPS
│   │   ├── reasoning.py               # HellaSwag, ARC, Big-Bench
│   │   ├── factuality.py              # SimpleQA, TruthfulQA
│   │   └── multilingual.py            # XGLUE, Belebele, FLORES
│   ├── configs/
│   │   ├── mmlu.yaml
│   │   ├── gpqa.yaml
│   │   ├── humaneval.yaml
│   │   ├── swe_bench.yaml
│   │   ├── simpleqa.yaml
│   │   └── benchmark_presets.yaml
│   ├── scripts/
│   │   ├── run_benchmark.py
│   │   └── compare_models.py
│   └── tests/
│       ├── test_benchmark_runner.py
│       └── test_academic_benchmarks.py
│
├── llm_as_judge/                      # Rubrics and evaluator pipelines
│   ├── README.md
│   ├── src/
│   │   ├── __init__.py
│   │   ├── judge.py                   # Core LLM-as-judge engine
│   │   ├── rubrics.py                 # Scoring rubrics and criteria
│   │   ├── evaluators.py              # Pre-built evaluators
│   │   ├── prompt_templates.py        # Evaluation prompt templates
│   │   ├── calibration.py             # Judge calibration utilities
│   │   └── aggregation.py             # Score aggregation and smoothing
│   ├── configs/
│   │   ├── judge_config.yaml
│   │   └── rubric_presets.yaml
│   ├── rubrics/
│   │   ├── answer_relevance.json
│   │   ├── task_completion.json
│   │   ├── faithfulness.json
│   │   ├── coherence.json
│   │   ├── helpfulness.json
│   │   └── prompt_adhesion.json
│   ├── scripts/
│   │   ├── run_judge.py
│   │   └── calibrate_judge.py
│   └── tests/
│       ├── test_judge.py
│       └── test_rubrics.py
│
├── safety/                            # Harms, jailbreak, policy, toxicity checks
│   ├── README.md
│   ├── src/
│   │   ├── __init__.py
│   │   ├── evaluator.py               # Main safety evaluation orchestrator
│   │   ├── toxicity.py                # Toxicity and hate speech detection
│   │   ├── bias.py                    # Bias and fairness assessment
│   │   ├── jailbreak.py               # Jailbreak resistance testing
│   │   ├── policy.py                  # Policy compliance checking
│   │   ├── pii.py                     # PII detection and redaction
│   │   ├── red_team.py                # Automated red teaming
│   │   └── classifiers.py             # Safety classification models
│   ├── configs/
│   │   ├── safety_config.yaml
│   │   └── jailbreak_prompts.yaml
│   ├── datasets/
│   │   ├── realtoxicity_samples.jsonl
│   │   ├── bias_test_cases.jsonl
│   │   └── jailbreak_prompts.jsonl
│   ├── scripts/
│   │   ├── run_safety_eval.py
│   │   └── red_team_attack.py
│   └── tests/
│       ├── test_toxicity.py
│       └── test_jailbreak.py
│
├── latency/                           # Runtime and SLA-oriented performance checks
│   ├── README.md
│   ├── src/
│   │   ├── __init__.py
│   │   ├── benchmark.py               # Latency benchmark harness
│   │   ├── metrics.py                 # TTFT, TPOT, throughput calculations
│   │   ├── load_test.py               # Concurrent load testing
│   │   ├── profiling.py               # Detailed performance profiling
│   │   └── sla_checker.py             # SLA compliance verification
│   ├── configs/
│   │   ├── latency_config.yaml
│   │   └── sla_thresholds.yaml
│   ├── scripts/
│   │   ├── run_latency_test.py
│   │   └── generate_report.py
│   └── tests/
│       ├── test_metrics.py
│       └── test_sla_checker.py
│
├── regression/                        # CI-compatible quality regression suites
│   ├── README.md
│   ├── src/
│   │   ├── __init__.py
│   │   ├── runner.py                  # Regression test runner
│   │   ├── golden_suite.py            # Golden dataset evaluation
│   │   ├── diff_report.py             # Score comparison and diffing
│   │   ├── gates.py                   # Quality gates and thresholds
│   │   └── drift_detection.py         # Statistical drift detection
│   ├── configs/
│   │   ├── regression_config.yaml
│   │   └── quality_gates.yaml
│   ├── golden_datasets/
│   │   └── sample_golden.jsonl
│   ├── scripts/
│   │   ├── run_regression.py
│   │   └── check_gates.py
│   └── tests/
│       ├── test_gates.py
│       └── test_drift_detection.py
│
└── src/                               # Shared evaluation utilities
    ├── __init__.py
    ├── base.py                        # Base evaluator class
    ├── metrics.py                     # Common metric calculations
    ├── reporting.py                   # Report generation
    ├── visualization.py               # Chart and graph generation
    └── utils.py                       # Helper utilities
```

## Evaluation Methodology

### Multi-Layered Approach

1. **Automated Metrics** - Fast, deterministic checks (BLEU, ROUGE, Exact Match, format validation)
2. **LLM-as-Judge** - Scalable semantic evaluation with 80-90% human agreement
3. **Human Review** - Targeted assessment for edge cases and calibration
4. **Production Monitoring** - Continuous feedback loops and drift detection

### Best Practices

- **Combine automation and human review** - Let metrics flag obvious issues while people handle nuance
- **Align metrics with product goals** - Different use cases need different evaluation strategies
- **Build evaluation into every sprint** - Make it continuous, not a one-off task
- **Monitor live systems** - Only continuous feedback catches model drift
- **Implement traceability** - Link every score to exact prompt, model, and dataset versions
- **Use component-level evaluation** - Evaluate RAG retrievers, generators, and tools separately

## Supported Benchmarks (2026)

### Academic & General Knowledge
- MMLU (57 subjects, saturation at 88%+)
- GPQA (Graduate-level, Google-proof Q&A)
- GSM8K (Grade school math)
- MATH (Competition mathematics)

### Code Generation
- HumanEval / HumanEval+ (164 Python problems)
- SWE-bench / SWE-bench Verified / SWE-bench Pro
- APPS (10,000 programming problems)

### Factuality & Reasoning
- SimpleQA / SimpleQA Verified
- TruthfulQA
- HellaSwag
- ARC (AI2 Reasoning Challenge)

### Multilingual
- XGLUE (Cross-lingual understanding)
- Belebele (Multilingual reading comprehension)
- FLORES (Translation)

## LLM-as-Judge Rubrics

Pre-built evaluation rubrics with structured scoring:

| Rubric | Scale | Dimensions |
|--------|-------|------------|
| Answer Relevance | 1-5 | Directness, completeness, focus |
| Task Completion | 1-5 | Requirement fulfillment, format matching |
| Faithfulness | 1-5 | Factual accuracy, hallucination detection |
| Coherence | 1-5 | Logical flow, fluency, structure |
| Helpfulness | 1-5 | Actionability, clarity, user value |
| Prompt Adhesion | 1-5 | Instruction following, constraint compliance |

## Safety Evaluation Categories

- **Toxicity** - Hate speech, harassment, explicit content
- **Bias** - Stereotyping, fairness across demographics
- **Jailbreak Resistance** - Prompt injection, role-play attacks
- **Policy Compliance** - Regulatory requirements (EU AI Act, state laws)
- **PII Detection** - Personal information leakage prevention

## Integration

### CI/CD Pipeline

```yaml
# .github/workflows/eval.yml
jobs:
  evaluation:
    steps:
      - name: Run regression tests
        run: python evaluation/regression/scripts/run_regression.py --ci-mode
      - name: Run safety checks
        run: python evaluation/safety/scripts/run_safety_eval.py --fail-on-violation
      - name: Check quality gates
        run: python evaluation/regression/scripts/check_gates.py --block-on-failure
```

### Experiment Tracking

```python
import mlflow
from evaluation.src.reporting import EvaluationReport

report = EvaluationReport(results)
mlflow.log_metrics(report.metrics)
mlflow.log_artifact(report.to_json())
```

## References

- [LLM Evaluation: Frameworks, Metrics, and Best Practices (2026)](https://futureagi.substack.com/p/llm-evaluation-frameworks-metrics)
- [LLM-As-Judge: 7 Best Practices & Evaluation Templates](https://www.montecarlodata.com/blog-llm-as-judge/)
- [LLM Evaluation and Benchmarking 2026 - Zylos Research](https://zylos.ai/research/2026-01-16-llm-evaluation-benchmarking)
- [G-Eval: Neural Evaluator for Text Generation (Liu et al., EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.153.pdf)
- [A Survey on LLM-as-a-Judge (Gu et al., 2024)](https://arxiv.org/abs/2412.12509)
- [OpenAI Evaluation Best Practices](https://platform.openai.com/docs/guides/evaluation-best-practices)
- [Eleuther AI Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
