"""Configuration management module for LLM-Whisperer.

Provides centralized configuration management for different environments, datasets,
models, and runtime settings across the LLM-Whisperer framework.

Submodules:
- datasets/: Dataset configuration, registry, and quality gates
- environments/: Environment-specific configurations (dev, staging, prod)
- models/: Model registry, fine-tuning profiles, and inference profiles
- runtime/: Runtime configurations for agents, inference, RAG, and observability
"""

__version__ = "0.1.0"

__all__ = [
    "datasets",
    "environments",
    "models",
    "runtime",
]
