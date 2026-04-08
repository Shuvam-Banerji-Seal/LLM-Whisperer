"""Core adapter classes for model adaptation."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .config import AdapterConfig, AdapterType

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    Adapters enable efficient model customization through parameter-efficient
    fine-tuning methods like LoRA, prefix tuning, etc.
    """

    def __init__(self, config: AdapterConfig):
        """Initialize model adapter.

        Args:
            config: Adapter configuration
        """
        self.config = config
        self.is_active = False
        self._parameters: Dict[str, Any] = {}
        logger.info(f"Initialized adapter: {config.name}")

    @abstractmethod
    def attach(self, model: Any) -> None:
        """Attach adapter to a model.

        Args:
            model: Base model to attach to

        Raises:
            RuntimeError: If attachment fails
        """
        pass

    @abstractmethod
    def detach(self) -> None:
        """Detach adapter from model.

        Raises:
            RuntimeError: If detachment fails
        """
        pass

    @abstractmethod
    def enable(self) -> None:
        """Enable the adapter for inference/training."""
        pass

    @abstractmethod
    def disable(self) -> None:
        """Disable the adapter."""
        pass

    @abstractmethod
    def get_trainable_params(self) -> Dict[str, Any]:
        """Get trainable parameters.

        Returns:
            Dictionary of trainable parameters
        """
        pass

    def get_config(self) -> AdapterConfig:
        """Get adapter configuration.

        Returns:
            AdapterConfig instance
        """
        return self.config

    def get_info(self) -> Dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter info
        """
        return {
            "name": self.config.name,
            "type": self.config.adapter_type.value,
            "version": self.config.version,
            "active": self.is_active,
            "composable": self.config.is_composable,
            "base_model": self.config.base_model_name,
        }


class LoRAAdapter(ModelAdapter):
    """LoRA (Low-Rank Adaptation) adapter implementation.

    Implements the LoRA technique for parameter-efficient fine-tuning.
    """

    def __init__(
        self,
        config: AdapterConfig,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        """Initialize LoRA adapter.

        Args:
            config: Adapter configuration
            rank: LoRA rank
            alpha: LoRA alpha scaling parameter
        """
        super().__init__(config)
        self.rank = rank
        self.alpha = alpha
        self.lora_scales = {}
        logger.info(f"LoRA adapter initialized with rank={rank}, alpha={alpha}")

    def attach(self, model: Any) -> None:
        """Attach LoRA adapter to model.

        Args:
            model: Base model to attach to
        """
        if not hasattr(model, "config"):
            raise RuntimeError("Model must have a config attribute")

        try:
            self._model_ref = model
            logger.info(f"Attached LoRA adapter {self.config.name} to model")
        except Exception as e:
            logger.error(f"Failed to attach LoRA adapter: {e}")
            raise RuntimeError(f"Attachment failed: {e}") from e

    def detach(self) -> None:
        """Detach LoRA adapter from model."""
        if hasattr(self, "_model_ref"):
            delattr(self, "_model_ref")
            logger.info(f"Detached LoRA adapter {self.config.name}")

    def enable(self) -> None:
        """Enable the LoRA adapter."""
        self.is_active = True
        logger.info(f"Enabled LoRA adapter {self.config.name}")

    def disable(self) -> None:
        """Disable the LoRA adapter."""
        self.is_active = False
        logger.info(f"Disabled LoRA adapter {self.config.name}")

    def get_trainable_params(self) -> Dict[str, Any]:
        """Get LoRA trainable parameters.

        Returns:
            Dictionary of LoRA parameters
        """
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "num_layers": len(self.lora_scales),
            "total_params": sum(r * c for r, c in self.lora_scales.values())
            if self.lora_scales
            else 0,
        }

    def add_layer(self, layer_name: str, input_dim: int, output_dim: int) -> None:
        """Add LoRA to a layer.

        Args:
            layer_name: Name of the layer
            input_dim: Input dimension
            output_dim: Output dimension
        """
        params = input_dim * self.rank + self.rank * output_dim
        self.lora_scales[layer_name] = (input_dim, output_dim)
        logger.debug(f"Added LoRA to layer {layer_name} with {params} params")


class PrefixTuningAdapter(ModelAdapter):
    """Prefix Tuning adapter implementation.

    Implements prefix tuning for parameter-efficient fine-tuning.
    """

    def __init__(
        self,
        config: AdapterConfig,
        prefix_length: int = 10,
    ):
        """Initialize Prefix Tuning adapter.

        Args:
            config: Adapter configuration
            prefix_length: Length of prefix to add
        """
        super().__init__(config)
        self.prefix_length = prefix_length
        logger.info(f"Prefix Tuning adapter initialized with length={prefix_length}")

    def attach(self, model: Any) -> None:
        """Attach prefix tuning adapter to model.

        Args:
            model: Base model to attach to
        """
        try:
            self._model_ref = model
            logger.info(f"Attached Prefix Tuning adapter {self.config.name}")
        except Exception as e:
            logger.error(f"Failed to attach Prefix Tuning adapter: {e}")
            raise RuntimeError(f"Attachment failed: {e}") from e

    def detach(self) -> None:
        """Detach prefix tuning adapter."""
        if hasattr(self, "_model_ref"):
            delattr(self, "_model_ref")
            logger.info(f"Detached Prefix Tuning adapter {self.config.name}")

    def enable(self) -> None:
        """Enable the adapter."""
        self.is_active = True
        logger.info(f"Enabled Prefix Tuning adapter {self.config.name}")

    def disable(self) -> None:
        """Disable the adapter."""
        self.is_active = False
        logger.info(f"Disabled Prefix Tuning adapter {self.config.name}")

    def get_trainable_params(self) -> Dict[str, Any]:
        """Get trainable parameters.

        Returns:
            Dictionary of trainable parameters
        """
        return {
            "prefix_length": self.prefix_length,
        }


class AdapterRegistry:
    """Registry for managing model adapters.

    Provides centralized management of adapters including registration,
    composition, and lifecycle management.
    """

    def __init__(self):
        """Initialize adapter registry."""
        self._adapters: Dict[str, ModelAdapter] = {}
        self._adapter_types: Dict[str, type] = {}
        self._composed_adapters: Dict[str, List[str]] = {}
        logger.info("Initialized AdapterRegistry")

    def register_adapter_type(
        self,
        adapter_type: str,
        adapter_class: type,
    ) -> None:
        """Register a custom adapter type.

        Args:
            adapter_type: Type identifier
            adapter_class: Adapter class
        """
        self._adapter_types[adapter_type] = adapter_class
        logger.info(f"Registered adapter type: {adapter_type}")

    def add_adapter(self, adapter: ModelAdapter) -> None:
        """Add an adapter to the registry.

        Args:
            adapter: Adapter instance

        Raises:
            ValueError: If adapter with same name already exists
        """
        name = adapter.config.name
        if name in self._adapters:
            raise ValueError(f"Adapter {name} already registered")

        self._adapters[name] = adapter
        logger.info(f"Added adapter to registry: {name}")

    def get_adapter(self, name: str) -> Optional[ModelAdapter]:
        """Get an adapter by name.

        Args:
            name: Adapter name

        Returns:
            Adapter instance or None if not found
        """
        return self._adapters.get(name)

    def remove_adapter(self, name: str) -> bool:
        """Remove an adapter from the registry.

        Args:
            name: Adapter name

        Returns:
            True if removed, False if not found
        """
        if name in self._adapters:
            del self._adapters[name]
            logger.info(f"Removed adapter from registry: {name}")
            return True
        return False

    def compose_adapters(
        self,
        composition_name: str,
        adapter_names: List[str],
    ) -> None:
        """Compose multiple adapters.

        Args:
            composition_name: Name for the composition
            adapter_names: List of adapter names to compose

        Raises:
            ValueError: If any adapter is not found or not composable
        """
        for name in adapter_names:
            if name not in self._adapters:
                raise ValueError(f"Adapter {name} not found")

            adapter = self._adapters[name]
            if not adapter.config.is_composable:
                raise ValueError(f"Adapter {name} is not composable")

        self._composed_adapters[composition_name] = adapter_names
        logger.info(f"Created adapter composition: {composition_name}")

    def get_composed_adapters(self, composition_name: str) -> List[ModelAdapter]:
        """Get adapters in a composition.

        Args:
            composition_name: Name of the composition

        Returns:
            List of adapter instances
        """
        if composition_name not in self._composed_adapters:
            return []

        return [
            self._adapters[name] for name in self._composed_adapters[composition_name]
        ]

    def list_adapters(self) -> List[str]:
        """List all registered adapters.

        Returns:
            List of adapter names
        """
        return list(self._adapters.keys())

    def get_adapter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about an adapter.

        Args:
            name: Adapter name

        Returns:
            Dictionary with adapter info or None if not found
        """
        adapter = self._adapters.get(name)
        return adapter.get_info() if adapter else None

    def list_compositions(self) -> List[str]:
        """List all adapter compositions.

        Returns:
            List of composition names
        """
        return list(self._composed_adapters.keys())
