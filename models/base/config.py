"""Configuration classes for base models."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""

    LANGUAGE_MODEL = "language_model"
    EMBEDDING_MODEL = "embedding_model"
    CLASSIFICATION_MODEL = "classification_model"
    GENERATIVE_MODEL = "generative_model"
    CUSTOM_MODEL = "custom_model"


@dataclass
class ModelConfig:
    """Configuration for a model.

    This dataclass defines the standard configuration structure for all models
    in the LLM-Whisperer framework.
    """

    name: str
    model_type: ModelType
    version: str = "1.0.0"
    description: Optional[str] = None

    # Model parameters
    model_size: Optional[str] = None  # e.g., "small", "base", "large"
    num_parameters: Optional[int] = None
    vocab_size: Optional[int] = None
    max_sequence_length: Optional[int] = None

    # Inference parameters
    batch_size: int = 32
    precision: str = "float32"  # float32, float16, int8
    device: str = "cpu"

    # Framework info
    framework: Optional[str] = None  # e.g., "pytorch", "transformers", "onnx"

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "version": self.version,
            "description": self.description,
            "model_size": self.model_size,
            "num_parameters": self.num_parameters,
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size,
            "precision": self.precision,
            "device": self.device,
            "framework": self.framework,
            "metadata": self.metadata,
        }


@dataclass
class ModelMetadata:
    """Metadata for a model instance.

    Tracks creation, modification, and runtime information about a model.
    """

    model_id: str
    name: str
    version: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    author: Optional[str] = None
    organization: Optional[str] = None
    license: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # Model information
    framework: Optional[str] = None
    model_type: Optional[str] = None
    num_parameters: Optional[int] = None

    # Paths and references
    source_url: Optional[str] = None
    local_path: Optional[Path] = None

    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary.

        Returns:
            Dictionary representation of the metadata
        """
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
            "organization": self.organization,
            "license": self.license,
            "tags": self.tags,
            "framework": self.framework,
            "model_type": self.model_type,
            "num_parameters": self.num_parameters,
            "source_url": self.source_url,
            "local_path": str(self.local_path) if self.local_path else None,
            "performance_metrics": self.performance_metrics,
            "custom_metadata": self.custom_metadata,
        }
