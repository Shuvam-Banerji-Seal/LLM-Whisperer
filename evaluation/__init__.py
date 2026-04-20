"""Evaluation framework for LLM-Whisperer.

Comprehensive evaluation system supporting five evaluation categories:

1. task_benchmarks: Standard benchmark evaluations (MMLU, GSM8K, HumanEval, SWE-bench)
2. llm_as_judge: LLM-based evaluation with rubrics and scoring
3. safety: Safety evaluation including toxicity, bias, jailbreak detection, and PII detection
4. latency: Performance and latency benchmarking with SLA monitoring
5. regression: Golden dataset-based regression testing and quality gates

Each category provides specialized runners, metrics, and aggregation functionality.
The base module provides shared utilities and interfaces for all evaluation types.
"""

# Base classes and utilities will be imported from src when available
# from .src import (...)

__version__ = "1.0.0"

__all__ = [
    "task_benchmarks",
    "llm_as_judge",
    "safety",
    "latency",
    "regression",
    "src",
]
