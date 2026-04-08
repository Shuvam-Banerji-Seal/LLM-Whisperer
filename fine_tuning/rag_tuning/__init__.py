"""RAG system fine-tuning module for LLM-Whisperer.

Specialized fine-tuning for retrieval-augmented generation systems.
"""

from .core import RAGFinetuner
from .config import RAGTuningConfig

__version__ = "0.1.0"
__all__ = ["RAGFinetuner", "RAGTuningConfig"]
