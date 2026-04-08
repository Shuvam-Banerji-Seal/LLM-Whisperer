"""Datasets module for LLM-Whisperer.

Manages dataset organization, processing, and storage across multiple categories:
- raw/: Original unprocessed datasets
- interim/: Intermediate processing results
- processed/: Final processed datasets ready for training/evaluation
- synthetic/: Synthetically generated datasets
- eval_sets/: Evaluation and test datasets
- prompt_sets/: Prompt templates and prompt-based datasets

This module provides utilities for dataset loading, validation, splitting,
and management across the entire LLM-Whisperer pipeline.
"""

__version__ = "0.1.0"

__all__ = [
    "raw",
    "interim",
    "processed",
    "synthetic",
    "eval_sets",
    "prompt_sets",
]
