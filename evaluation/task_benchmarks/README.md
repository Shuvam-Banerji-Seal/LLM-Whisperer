# Task Benchmarks

Standard evaluation benchmarks for LLM systems covering core capabilities.

## Overview

Task benchmarks evaluate fundamental LLM abilities through established datasets:

| Benchmark | Problems | Category | Metric | Scoring |
|-----------|----------|----------|--------|---------|
| **MMLU** | 14,042 | Knowledge | Accuracy | Exact match (A-D) |
| **GSM8K** | 8,500 | Reasoning | Accuracy | Numeric answer match |
| **HumanEval** | 164 | Coding | Pass@k | Code execution in sandbox |
| **SWE-bench** | 2,294 | Engineering | Resolution | Patch applies & tests pass |

## Quick Start

### MMLU (Knowledge Assessment)

```python
from evaluation.task_benchmarks import load_mmlu, BenchmarkRunner

# Load dataset
dataset = load_mmlu('data/mmlu.json')

# Run model and collect predictions (A, B, C, D)
predictions = ["A", "B", "C", "D", ...]  # 14,042 answers

# Evaluate
runner = BenchmarkRunner()
result = runner.run_benchmark("MMLU", dataset, predictions)
print(f"Overall Accuracy: {result['metrics']['accuracy']:.4f}")
print(f"By Subject: {result['metrics']['by_subject']}")
```

### GSM8K (Math Reasoning)

```python
from evaluation.task_benchmarks import load_gsm8k, BenchmarkRunner

# Load dataset
dataset = load_gsm8k('data/gsm8k.json')

# Run model with chain-of-thought
predictions = [
    "Let's think step by step...\n#### 42",
    "First, we calculate...\n#### 100",
    ...
]

# Evaluate (automatically extracts numeric answers)
runner = BenchmarkRunner()
result = runner.run_benchmark("GSM8K", dataset, predictions)
print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
```

### HumanEval (Code Generation)

```python
from evaluation.task_benchmarks import load_humaneval, BenchmarkRunner

# Load dataset
dataset = load_humaneval('data/humaneval.json')

# Run model k times per problem and check test results
results = [True, False, True, ...]  # Pass/fail for each execution

# Evaluate (pass@k metric)
pass_at_1 = dataset.compute_pass_at_k(results, k=1)
print(f"Pass@1: {pass_at_1:.4f}")
```

### SWE-bench (Software Engineering)

```python
from evaluation.task_benchmarks import load_swe_bench, BenchmarkRunner

# Load dataset with GitHub issues
dataset = load_swe_bench('data/swe_bench.json')

# Run model to generate patches, apply and test
results = [True, False, True, ...]  # Issue resolved yes/no

# Evaluate
resolution_rate = dataset.resolution_rate(results)
print(f"Resolution Rate: {resolution_rate:.4f}")
```

## Datasets

### MMLU Dataset

**Source:** https://github.com/hendrycks/MMLU

14,042 multiple-choice questions across 57 subjects:
- 32 humanities (history, philosophy, etc.)
- 14 social sciences (economics, psychology, etc.)  
- 8 STEM (physics, chemistry, biology, etc.)
- 3 technical (computer science, engineering, law)

**Format:**
```json
[
  {
    "subject": "abstract_algebra",
    "question": "What is the result of...",
    "choices": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "answer": "C"
  }
]
```

### GSM8K Dataset

**Source:** https://github.com/openai/grade-school-math

8,500 grade school math word problems with solutions:
- Difficulty: Elementary school arithmetic to middle school algebra
- Format: Natural language problem + step-by-step solution

**Format:**
```json
[
  {
    "question": "John has 10 apples...",
    "answer": "Let's break this down step by step:\n1. John starts with 10 apples\n2. He gives away 3...\n#### 7"
  }
]
```

### HumanEval Dataset

**Source:** https://github.com/openai/human-eval

164 Python coding problems with unit tests:
- Range: Simple string manipulation to complex algorithms
- Each problem includes docstring, signature, and test cases
- Evaluation: Code execution in isolated sandbox

**Format:**
```json
[
  {
    "task_id": "HumanEval/0",
    "prompt": "def solution(x):\n    \"\"\"...\"\"\"\n",
    "canonical_solution": "def solution(x):\n    return x * 2",
    "test_list": ["assert solution(3) == 6", "assert solution(0) == 0"],
    "entry_point": "solution"
  }
]
```

### SWE-bench Dataset

**Source:** https://github.com/princeton-nlp/SWE-bench

2,294 real GitHub issues with repository context:
- Real OSS projects: django, requests, sympy, flask, etc.
- Issues: Bugs, feature requests, failing tests
- Evaluation: Generated patch applies and tests pass

**Format:**
```json
[
  {
    "instance_id": "django__django-11039",
    "repo": "django/django",
    "problem_statement": "Bug: ...",
    "base_commit": "abc123...",
    "patch": "diff --git a/...",
    "test_patch": "diff --git a/..."
  }
]
```

## Metrics

### MMLU
- **Accuracy**: Fraction of correct answers (exact match)
- **By Subject**: Per-subject breakdown to identify strengths/weaknesses

### GSM8K
- **Accuracy**: Numeric answer correctness after chain-of-thought
- Uses regex extraction: `#### <number>` or `answer is <number>`

### HumanEval
- **Pass@k**: Probability problem is solved within k attempts
  - Formula: Pass@k = 1 - C(n-c, k) / C(n, k)
  - where n=total problems, c=correct solutions
- Common thresholds: Pass@1, Pass@5, Pass@10

### SWE-bench
- **Resolution Rate**: Fraction of issues successfully fixed
- **Test Pass Rate**: Percentage of test cases passing
- **Build Success Rate**: Whether patch compiles/installs cleanly

## Configuration

Create `configs/benchmark_config.yaml`:

```yaml
mmlu:
  dataset_path: "data/mmlu.json"
  expected_accuracy: 0.85
  by_subject_threshold: 0.75

gsm8k:
  dataset_path: "data/gsm8k.json"
  expected_accuracy: 0.92
  
humaneval:
  dataset_path: "data/humaneval.json"
  expected_pass_at_1: 0.70
  timeout_per_test: 5
  
swe_bench:
  dataset_path: "data/swe_bench.json"
  expected_resolution_rate: 0.35
  timeout_per_issue: 300
```

## CI/CD Integration

Add to `.github/workflows/benchmarks.yml`:

```yaml
name: Benchmark Evaluation

on: [push, pull_request]

jobs:
  benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r evaluation/requirements.txt
      
      - name: Run benchmarks
        run: |
          python -m pytest evaluation/task_benchmarks/tests/ -v
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        if: always()
        with:
          name: benchmark-results
          path: evaluation/results/
```

## References

- MMLU: Hendrycks et al. (2020) - https://arxiv.org/abs/2009.03300
- GSM8K: Cobbe et al. (2021) - https://arxiv.org/abs/2110.14168
- HumanEval: Chen et al. (2021) - https://arxiv.org/abs/2107.03374
- SWE-bench: Jimenez et al. (2024) - https://arxiv.org/abs/2310.06770
