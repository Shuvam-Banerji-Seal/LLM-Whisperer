"""Fine-tuning module for LLM-Whisperer.

Comprehensive fine-tuning framework supporting multiple fine-tuning techniques
and methodologies for large language models.

Submodules:
- base/: Base classes, configurations, and utilities for all fine-tuning approaches
- lora/: Low-Rank Adaptation (LoRA) fine-tuning implementation
- qlora/: Quantized LoRA (QLoRA) for memory-efficient fine-tuning
- multimodal/: Fine-tuning for multimodal models
- rag_tuning/: Fine-tuning with RAG integration
- behavior_tuning/: Behavior-specific fine-tuning
- agentic_tuning/: Agentic system fine-tuning
- reward_modeling/: Reward model training for RLHF
- templates/: Template configurations for various fine-tuning approaches
- configs/: Configuration management for fine-tuning

Each submodule provides specialized implementations, configurations, and utilities
for its respective fine-tuning approach.
"""

__version__ = "0.1.0"

__all__ = [
    "base",
    "lora",
    "qlora",
    "multimodal",
    "rag_tuning",
    "behavior_tuning",
    "agentic_tuning",
    "reward_modeling",
    "templates",
    "configs",
]
