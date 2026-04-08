"""Experiments module for LLM-Whisperer.

Manages experiment tracking, design, and results for the LLM-Whisperer framework.

Submodules:
- ablations/: Ablation study configurations and results
- tracking/: Experiment tracking and metrics monitoring
- reports/: Experiment reports, analysis, and visualization

This module provides utilities for designing, executing, and analyzing experiments
to understand model behavior, fine-tuning effects, and system performance.
"""

__version__ = "0.1.0"

__all__ = [
    "ablations",
    "tracking",
    "reports",
]
