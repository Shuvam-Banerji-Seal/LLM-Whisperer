"""Task Benchmarks Evaluation Module

Standard benchmarking for LLM systems:
- MMLU: Massive Multitask Language Understanding (14K questions, 57 subjects)
- GSM8K: Grade School Math (8.5K problems with chain-of-thought)
- HumanEval: Code generation (164 problems with execution-based scoring)
- SWE-bench: Software Engineering (2.3K GitHub issues)
"""

from .benchmarks import (
    BenchmarkQuestion,
    BenchmarkDataset,
    MMLUDataset,
    GSM8KDataset,
    HumanEvalDataset,
    SWEBenchDataset,
    load_mmlu,
    load_gsm8k,
    load_humaneval,
    load_swe_bench,
)

from .runner import BenchmarkRunner, pytest_benchmark_wrapper

__all__ = [
    "BenchmarkQuestion",
    "BenchmarkDataset",
    "MMLUDataset",
    "GSM8KDataset",
    "HumanEvalDataset",
    "SWEBenchDataset",
    "BenchmarkRunner",
    "load_mmlu",
    "load_gsm8k",
    "load_humaneval",
    "load_swe_bench",
    "pytest_benchmark_wrapper",
]
