"""Models module for LLM-Whisperer.

Comprehensive model management, registry, and interface framework for handling
various large language models and their variants across the LLM-Whisperer system.

Submodules:
- base/: Base model classes, interfaces, and metadata management
- registry/: Model registry for centralized model management
- adapters/: Model adapter implementations for different frameworks and architectures
- exported/: Exported and deployable model versions
- merged/: Merged and consolidated model configurations

This module provides utilities for model loading, configuration, adapter management,
deployment, and multi-framework compatibility.
"""

__version__ = "0.1.0"

__all__ = [
    "base",
    "registry",
    "adapters",
    "exported",
    "merged",
]
