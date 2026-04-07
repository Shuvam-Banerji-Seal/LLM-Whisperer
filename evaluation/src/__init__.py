"""
Evaluation Framework - Shared Utilities

This module provides base classes and utilities for all evaluation categories:
- task_benchmarks: Standard benchmarking (MMLU, GSM8K, HumanEval, SWE-bench)
- llm_as_judge: LLM-based evaluation with rubrics and scoring
- safety: Toxicity, bias, jailbreak, and PII detection
- latency: Performance benchmarking and SLA monitoring
- regression: Golden dataset testing and quality gates
"""

__version__ = "1.0.0"
__author__ = "LLM-Whisperer Team"
