"""Configuration for model merging."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Strategies for merging models."""

    LINEAR_INTERPOLATION = "linear_interpolation"
    WEIGHTED_AVERAGE = "weighted_average"
    LAYER_WISE = "layer_wise"
    TASK_ARITHMETIC = "task_arithmetic"
    TIES_MERGING = "ties_merging"
    DARE_TIES = "dare_ties"
    SLERP = "slerp"
    CONSENSUS = "consensus"


@dataclass
class MergeConfig:
    """Configuration for model merging.

    Defines how multiple models should be merged into a single combined model.
    """

    merge_strategy: MergeStrategy
    base_model_name: str
    model_names: List[str]
    output_model_name: str
    version: str = "1.0.0"

    # Merge parameters
    weights: Optional[List[float]] = None
    alpha: float = 0.5
    beta: float = 0.0

    # Interpolation settings
    interpolation_method: str = "linear"

    # Layer-wise settings
    layer_weights: Optional[Dict[int, float]] = None
    exclude_layers: Optional[List[str]] = None

    # Task arithmetic settings
    task_vector_scaling: float = 1.0

    # Consensus settings
    consensus_threshold: float = 0.5

    # Output settings
    output_dtype: Optional[str] = None
    preserve_config: bool = True

    # Custom metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return {
            "merge_strategy": self.merge_strategy.value,
            "base_model_name": self.base_model_name,
            "model_names": self.model_names,
            "output_model_name": self.output_model_name,
            "version": self.version,
            "weights": self.weights,
            "alpha": self.alpha,
            "beta": self.beta,
            "interpolation_method": self.interpolation_method,
            "layer_weights": self.layer_weights,
            "exclude_layers": self.exclude_layers,
            "task_vector_scaling": self.task_vector_scaling,
            "consensus_threshold": self.consensus_threshold,
            "output_dtype": self.output_dtype,
            "preserve_config": self.preserve_config,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
        }
