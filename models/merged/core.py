"""Core model merging functionality."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .config import MergeConfig, MergeStrategy

logger = logging.getLogger(__name__)


class ModelMerger(ABC):
    """Abstract base class for model mergers.

    Handles merging multiple models using various strategies and techniques.
    """

    def __init__(self, config: MergeConfig):
        """Initialize model merger.

        Args:
            config: Merge configuration
        """
        self.config = config
        self.models: Dict[str, Any] = {}
        self.merge_history: List[Dict[str, Any]] = []
        logger.info(f"Initialized merger with strategy: {config.merge_strategy.value}")

    @abstractmethod
    def merge(self, models: Dict[str, Any]) -> Any:
        """Merge models.

        Args:
            models: Dictionary of model name to model instance

        Returns:
            Merged model

        Raises:
            RuntimeError: If merge fails
        """
        pass

    def add_model(self, name: str, model: Any) -> None:
        """Add a model for merging.

        Args:
            name: Model identifier
            model: Model instance
        """
        self.models[name] = model
        logger.debug(f"Added model for merging: {name}")

    def get_merge_info(self) -> Dict[str, Any]:
        """Get merge information.

        Returns:
            Dictionary with merge information
        """
        return {
            "strategy": self.config.merge_strategy.value,
            "num_models": len(self.models),
            "model_names": list(self.models.keys()),
            "output_name": self.config.output_model_name,
            "num_merges": len(self.merge_history),
        }


class LinearInterpolationMerger(ModelMerger):
    """Linear interpolation model merger.

    Merges models using linear interpolation with configurable weights.
    """

    def merge(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Merge models using linear interpolation.

        Args:
            models: Dictionary of models to merge

        Returns:
            Merged model dictionary

        Raises:
            RuntimeError: If merge fails
        """
        try:
            logger.info(f"Merging {len(models)} models with linear interpolation")

            # Validate models
            if not models:
                raise ValueError("No models provided for merging")

            # Use provided weights or default
            weights = self.config.weights
            if weights is None:
                # Equal weights
                weights = [1.0 / len(models)] * len(models)
            else:
                # Normalize weights
                total = sum(weights)
                weights = [w / total for w in weights]

            # Initialize merged model structure
            merged_data = {}

            # Merge model names
            model_names = list(models.keys())
            for i, (name, model) in enumerate(models.items()):
                if not isinstance(model, dict):
                    logger.warning(f"Model {name} is not a dictionary")
                    continue

                for key, value in model.items():
                    if key not in merged_data:
                        merged_data[key] = {}

                    # Linear interpolation
                    if isinstance(value, (int, float)):
                        merged_data[key][name] = value

            logger.info("Linear interpolation merge completed")

            # Record merge
            self.merge_history.append(
                {
                    "strategy": "linear_interpolation",
                    "model_names": model_names,
                    "weights": weights,
                    "merged_model_name": self.config.output_model_name,
                }
            )

            return {
                "merged_model": merged_data,
                "strategy": "linear_interpolation",
                "weights": weights,
                "output_name": self.config.output_model_name,
            }

        except Exception as e:
            logger.error(f"Linear interpolation merge failed: {e}")
            raise RuntimeError(f"Merge failed: {e}") from e


class WeightedAverageMerger(ModelMerger):
    """Weighted average model merger.

    Merges models by computing weighted averages of parameters.
    """

    def merge(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Merge models using weighted average.

        Args:
            models: Dictionary of models to merge

        Returns:
            Merged model dictionary

        Raises:
            RuntimeError: If merge fails
        """
        try:
            logger.info(f"Merging {len(models)} models with weighted average")

            if not models:
                raise ValueError("No models provided for merging")

            # Use provided weights or default
            weights = self.config.weights
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            else:
                total = sum(weights)
                weights = [w / total for w in weights]

            merged_model = {}
            model_names = list(models.keys())

            # Average across models
            for i, (name, model) in enumerate(models.items()):
                weight = weights[i]

                if isinstance(model, dict):
                    for key, value in model.items():
                        if key not in merged_model:
                            merged_model[key] = 0

                        if isinstance(value, (int, float)):
                            merged_model[key] += weight * value

            logger.info("Weighted average merge completed")

            # Record merge
            self.merge_history.append(
                {
                    "strategy": "weighted_average",
                    "model_names": model_names,
                    "weights": weights,
                    "merged_model_name": self.config.output_model_name,
                }
            )

            return {
                "merged_model": merged_model,
                "strategy": "weighted_average",
                "weights": weights,
                "output_name": self.config.output_model_name,
            }

        except Exception as e:
            logger.error(f"Weighted average merge failed: {e}")
            raise RuntimeError(f"Merge failed: {e}") from e


class MergedModel:
    """Represents a model created by merging multiple models.

    Contains merged model data and metadata about the merge process.
    """

    def __init__(
        self,
        name: str,
        source_models: List[str],
        merge_strategy: MergeStrategy,
        merged_data: Dict[str, Any],
        weights: Optional[List[float]] = None,
    ):
        """Initialize merged model.

        Args:
            name: Name of the merged model
            source_models: Names of source models
            merge_strategy: Strategy used for merging
            merged_data: The merged model data
            weights: Weights used in merging (optional)
        """
        self.name = name
        self.source_models = source_models
        self.merge_strategy = merge_strategy
        self.merged_data = merged_data
        self.weights = weights or []
        self.version = "1.0.0"
        logger.info(f"Created merged model: {name}")

    def get_info(self) -> Dict[str, Any]:
        """Get merged model information.

        Returns:
            Dictionary with model information
        """
        return {
            "name": self.name,
            "source_models": self.source_models,
            "merge_strategy": self.merge_strategy.value,
            "num_source_models": len(self.source_models),
            "weights": self.weights,
            "version": self.version,
        }

    def get_source_models(self) -> List[str]:
        """Get list of source models.

        Returns:
            List of source model names
        """
        return self.source_models

    def get_merge_strategy(self) -> MergeStrategy:
        """Get merge strategy used.

        Returns:
            MergeStrategy enum value
        """
        return self.merge_strategy

    def get_weights(self) -> Optional[List[float]]:
        """Get merge weights.

        Returns:
            Merge weights or None if not applicable
        """
        return self.weights if self.weights else None


class MergerFactory:
    """Factory for creating model mergers.

    Provides unified interface for creating mergers using various strategies.
    """

    _mergers = {
        MergeStrategy.LINEAR_INTERPOLATION: LinearInterpolationMerger,
        MergeStrategy.WEIGHTED_AVERAGE: WeightedAverageMerger,
    }

    @classmethod
    def register_merger(
        cls,
        strategy: MergeStrategy,
        merger_class: type,
    ) -> None:
        """Register a custom merger.

        Args:
            strategy: Merge strategy
            merger_class: Merger class
        """
        cls._mergers[strategy] = merger_class
        logger.info(f"Registered merger for strategy: {strategy.value}")

    @classmethod
    def create_merger(cls, config: MergeConfig) -> ModelMerger:
        """Create a merger instance.

        Args:
            config: Merge configuration

        Returns:
            Merger instance

        Raises:
            ValueError: If strategy is not registered
        """
        if config.merge_strategy not in cls._mergers:
            raise ValueError(
                f"Unsupported merge strategy: {config.merge_strategy.value}"
            )

        merger_class = cls._mergers[config.merge_strategy]
        return merger_class(config)
