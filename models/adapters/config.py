"""Configuration for model adapters."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AdapterType(Enum):
    """Types of model adapters."""

    PREFIX_TUNING = "prefix_tuning"
    LORA = "lora"
    BOTTLENECK = "bottleneck"
    ADAPTER_LAYER = "adapter_layer"
    HYPERNETWORK = "hypernetwork"
    BITFIT = "bitfit"
    COMPACTER = "compacter"
    CUSTOM = "custom"


@dataclass
class AdapterConfig:
    """Configuration for a model adapter.

    Defines the structure and parameters for model adapters that can be
    attached to base models for efficient fine-tuning.
    """

    name: str
    adapter_type: AdapterType
    base_model_name: str
    version: str = "1.0.0"
    description: Optional[str] = None

    # Adapter-specific parameters
    hidden_size: Optional[int] = None
    adapter_size: Optional[int] = None
    reduction_factor: Optional[int] = None
    dropout_rate: float = 0.1

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # Framework info
    framework: Optional[str] = None

    # Composition settings
    is_composable: bool = True

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return {
            "name": self.name,
            "adapter_type": self.adapter_type.value,
            "base_model_name": self.base_model_name,
            "version": self.version,
            "description": self.description,
            "hidden_size": self.hidden_size,
            "adapter_size": self.adapter_size,
            "reduction_factor": self.reduction_factor,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "framework": self.framework,
            "is_composable": self.is_composable,
            "metadata": self.metadata,
        }
