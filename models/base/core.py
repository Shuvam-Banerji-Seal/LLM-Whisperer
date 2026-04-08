"""Core base classes and interfaces for models."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from .config import ModelConfig, ModelMetadata, ModelType

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a model instance."""

    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    UNLOADED = "unloaded"


class ModelInterface(ABC):
    """Abstract interface for all models.

    Defines the standard interface that all model implementations must follow.
    This ensures consistency across different model types and frameworks.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory.

        Raises:
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Generate predictions on inputs.

        Args:
            inputs: Model input(s)
            **kwargs: Additional model-specific arguments

        Returns:
            Model predictions

        Raises:
            RuntimeError: If model is not loaded
        """
        pass

    @abstractmethod
    def get_config(self) -> ModelConfig:
        """Get the model configuration.

        Returns:
            ModelConfig instance
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Get the model metadata.

        Returns:
            ModelMetadata instance
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        pass


class BaseModel(ModelInterface):
    """Base implementation of ModelInterface.

    Provides common functionality for all model types, including lifecycle
    management, configuration, and metadata handling.
    """

    def __init__(
        self,
        config: ModelConfig,
        metadata: Optional[ModelMetadata] = None,
    ):
        """Initialize base model.

        Args:
            config: Model configuration
            metadata: Model metadata (optional)
        """
        self.config = config
        self.metadata = metadata or self._create_default_metadata()
        self.status = ModelStatus.UNINITIALIZED
        self._loaded = False
        logger.info(f"Initialized model: {config.name}")

    def _create_default_metadata(self) -> ModelMetadata:
        """Create default metadata from config.

        Returns:
            ModelMetadata instance
        """
        return ModelMetadata(
            model_id=f"{self.config.name}-{self.config.version}",
            name=self.config.name,
            version=self.config.version,
            framework=self.config.framework,
            model_type=self.config.model_type.value,
            num_parameters=self.config.num_parameters,
        )

    def load(self) -> None:
        """Load the model into memory.

        Raises:
            RuntimeError: If model loading fails
        """
        if self._loaded:
            logger.warning(f"Model {self.config.name} is already loaded")
            return

        try:
            self.status = ModelStatus.LOADING
            logger.info(f"Loading model: {self.config.name}")

            # Subclasses should override _load_impl for actual loading
            self._load_impl()

            self._loaded = True
            self.status = ModelStatus.READY
            logger.info(f"Model {self.config.name} loaded successfully")
        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"Failed to load model {self.config.name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def unload(self) -> None:
        """Unload the model from memory."""
        if not self._loaded:
            logger.warning(f"Model {self.config.name} is not loaded")
            return

        try:
            logger.info(f"Unloading model: {self.config.name}")
            self._unload_impl()
            self._loaded = False
            self.status = ModelStatus.UNLOADED
            logger.info(f"Model {self.config.name} unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading model {self.config.name}: {e}")
            raise

    def predict(self, inputs: Any, **kwargs) -> Any:
        """Generate predictions on inputs.

        Args:
            inputs: Model input(s)
            **kwargs: Additional model-specific arguments

        Returns:
            Model predictions

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded:
            raise RuntimeError(f"Model {self.config.name} is not loaded")

        try:
            self.status = ModelStatus.RUNNING
            results = self._predict_impl(inputs, **kwargs)
            self.status = ModelStatus.READY
            return results
        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def get_config(self) -> ModelConfig:
        """Get the model configuration.

        Returns:
            ModelConfig instance
        """
        return self.config

    def get_metadata(self) -> ModelMetadata:
        """Get the model metadata.

        Returns:
            ModelMetadata instance
        """
        return self.metadata

    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._loaded

    def get_status(self) -> ModelStatus:
        """Get the current status of the model.

        Returns:
            ModelStatus enum value
        """
        return self.status

    def get_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information
        """
        return {
            "name": self.config.name,
            "version": self.config.version,
            "type": self.config.model_type.value,
            "status": self.status.value,
            "loaded": self._loaded,
            "framework": self.config.framework,
            "num_parameters": self.config.num_parameters,
        }

    # Protected methods for subclasses to override
    def _load_impl(self) -> None:
        """Implement model loading logic.

        Subclasses should override this method to implement framework-specific
        loading logic.
        """
        pass

    def _unload_impl(self) -> None:
        """Implement model unloading logic.

        Subclasses should override this method to implement framework-specific
        unloading logic.
        """
        pass

    def _predict_impl(self, inputs: Any, **kwargs) -> Any:
        """Implement prediction logic.

        Subclasses should override this method to implement framework-specific
        prediction logic.

        Args:
            inputs: Model input(s)
            **kwargs: Additional model-specific arguments

        Returns:
            Model predictions
        """
        raise NotImplementedError(
            f"Prediction not implemented for {self.__class__.__name__}"
        )


class ModelFactory:
    """Factory for creating model instances.

    Provides a unified interface for creating models from various sources
    and configurations.
    """

    _model_builders: Dict[str, callable] = {}

    @classmethod
    def register_builder(cls, model_type: str, builder: callable) -> None:
        """Register a model builder.

        Args:
            model_type: Type identifier for the model
            builder: Callable that creates a model instance
        """
        cls._model_builders[model_type] = builder
        logger.info(f"Registered model builder for type: {model_type}")

    @classmethod
    def create(cls, model_type: str, **kwargs) -> BaseModel:
        """Create a model instance.

        Args:
            model_type: Type identifier for the model
            **kwargs: Arguments to pass to the builder

        Returns:
            Model instance

        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in cls._model_builders:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Registered types: {list(cls._model_builders.keys())}"
            )

        builder = cls._model_builders[model_type]
        return builder(**kwargs)

    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get list of registered model types.

        Returns:
            List of registered model type identifiers
        """
        return list(cls._model_builders.keys())
