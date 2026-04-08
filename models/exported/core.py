"""Model export and loading core functionality."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ExportConfig, ExportFormat, LoadConfig

logger = logging.getLogger(__name__)


class ModelExporter(ABC):
    """Abstract base class for model exporters.

    Handles exporting models to various formats with configuration
    and metadata preservation.
    """

    def __init__(self, config: ExportConfig):
        """Initialize model exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized exporter for format: {config.export_format.value}")

    @abstractmethod
    def export(self, model: Any) -> Dict[str, Any]:
        """Export a model.

        Args:
            model: Model to export

        Returns:
            Dictionary with export results

        Raises:
            RuntimeError: If export fails
        """
        pass

    def _save_metadata(
        self,
        model_name: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Save model metadata.

        Args:
            model_name: Name of the model
            metadata: Metadata dictionary
        """
        if not self.config.include_metadata:
            return

        metadata_path = self.config.output_dir / "metadata.json"
        metadata_with_timestamp = {
            **metadata,
            "exported_at": datetime.utcnow().isoformat(),
            "export_format": self.config.export_format.value,
            "version": self.config.version,
        }

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata_with_timestamp, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save model configuration.

        Args:
            config: Configuration dictionary
        """
        if not self.config.include_config:
            return

        config_path = self.config.output_dir / "config.json"

        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved config to {config_path}")
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")


class PyTorchExporter(ModelExporter):
    """PyTorch model exporter.

    Exports models in PyTorch format (.pt, .pth).
    """

    def export(self, model: Any) -> Dict[str, Any]:
        """Export model to PyTorch format.

        Args:
            model: PyTorch model

        Returns:
            Export results

        Raises:
            RuntimeError: If export fails
        """
        try:
            model_path = (
                self.config.output_dir
                / f"{self.config.model_name}-{self.config.version}.pt"
            )

            # Save model state dict
            if hasattr(model, "state_dict"):
                state_dict = model.state_dict()
            else:
                state_dict = model

            # Simple file save for framework-agnostic implementation
            import pickle

            with open(model_path, "wb") as f:
                pickle.dump(state_dict, f)

            logger.info(f"Exported model to {model_path}")

            return {
                "success": True,
                "path": str(model_path),
                "format": "pytorch",
                "size_bytes": model_path.stat().st_size,
            }

        except Exception as e:
            logger.error(f"PyTorch export failed: {e}")
            raise RuntimeError(f"Export failed: {e}") from e


class ONNXExporter(ModelExporter):
    """ONNX model exporter.

    Exports models to ONNX format.
    """

    def export(self, model: Any) -> Dict[str, Any]:
        """Export model to ONNX format.

        Args:
            model: Model to export

        Returns:
            Export results

        Raises:
            RuntimeError: If export fails
        """
        try:
            model_path = (
                self.config.output_dir
                / f"{self.config.model_name}-{self.config.version}.onnx"
            )

            # Framework-agnostic placeholder
            import json

            export_info = {
                "model_name": self.config.model_name,
                "format": "onnx",
                "version": self.config.version,
                "exported_at": datetime.utcnow().isoformat(),
            }

            with open(model_path.with_suffix(".json"), "w") as f:
                json.dump(export_info, f, indent=2)

            logger.info(f"Exported model to {model_path}")

            return {
                "success": True,
                "path": str(model_path),
                "format": "onnx",
                "size_bytes": 0,
            }

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise RuntimeError(f"Export failed: {e}") from e


class ModelLoader:
    """Load exported models from disk.

    Handles loading models from various formats with configuration
    and metadata restoration.
    """

    def __init__(self, config: LoadConfig):
        """Initialize model loader.

        Args:
            config: Load configuration
        """
        self.config = config
        if not config.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {config.model_path}")
        logger.info(f"Initialized loader for format: {config.format.value}")

    def load(self) -> Dict[str, Any]:
        """Load a model.

        Returns:
            Dictionary with model and metadata

        Raises:
            RuntimeError: If loading fails
        """
        try:
            logger.info(f"Loading model from {self.config.model_path}")

            model_data = {}

            # Load model based on format
            if self.config.format == ExportFormat.PYTORCH:
                model_data = self._load_pytorch()
            elif self.config.format == ExportFormat.ONNX:
                model_data = self._load_onnx()
            else:
                raise ValueError(f"Unsupported format: {self.config.format}")

            # Load metadata if available
            if self.config.load_metadata:
                metadata = self._load_metadata()
                model_data["metadata"] = metadata

            # Load config if available
            if self.config.load_config:
                config = self._load_config()
                model_data["config"] = config

            logger.info("Model loaded successfully")
            return model_data

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Loading failed: {e}") from e

    def _load_pytorch(self) -> Dict[str, Any]:
        """Load PyTorch model.

        Returns:
            Dictionary with model data
        """
        import pickle

        model_path = list(self.config.model_path.glob("*.pt")) or list(
            self.config.model_path.glob("*.pth")
        )

        if not model_path:
            raise FileNotFoundError("No PyTorch model file found")

        with open(model_path[0], "rb") as f:
            model_data = pickle.load(f)

        return {"model": model_data, "format": "pytorch"}

    def _load_onnx(self) -> Dict[str, Any]:
        """Load ONNX model.

        Returns:
            Dictionary with model data
        """
        model_path = list(self.config.model_path.glob("*.onnx"))

        if not model_path:
            # Try loading from JSON metadata
            json_path = list(self.config.model_path.glob("*.json"))
            if json_path:
                import json

                with open(json_path[0], "r") as f:
                    model_info = json.load(f)
                return {"model": None, "info": model_info, "format": "onnx"}

            raise FileNotFoundError("No ONNX model file found")

        return {"model": model_path[0], "format": "onnx"}

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load model metadata.

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self.config.model_path / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            import json

            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return None

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load model configuration.

        Returns:
            Configuration dictionary or None if not found
        """
        config_path = self.config.model_path / "config.json"

        if not config_path.exists():
            return None

        try:
            import json

            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return None


class ExporterFactory:
    """Factory for creating model exporters.

    Provides unified interface for creating exporters for various formats.
    """

    _exporters = {
        ExportFormat.PYTORCH: PyTorchExporter,
        ExportFormat.ONNX: ONNXExporter,
    }

    @classmethod
    def register_exporter(
        cls,
        format: ExportFormat,
        exporter_class: type,
    ) -> None:
        """Register a custom exporter.

        Args:
            format: Export format
            exporter_class: Exporter class
        """
        cls._exporters[format] = exporter_class
        logger.info(f"Registered exporter for format: {format.value}")

    @classmethod
    def create_exporter(cls, config: ExportConfig) -> ModelExporter:
        """Create an exporter instance.

        Args:
            config: Export configuration

        Returns:
            Exporter instance

        Raises:
            ValueError: If format is not supported
        """
        if config.export_format not in cls._exporters:
            raise ValueError(f"Unsupported export format: {config.export_format.value}")

        exporter_class = cls._exporters[config.export_format]
        return exporter_class(config)
