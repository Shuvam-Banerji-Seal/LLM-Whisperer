"""Configuration for model export and loading."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported model export formats."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    TORCHSCRIPT = "torchscript"
    SAFETENSORS = "safetensors"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class ExportConfig:
    """Configuration for model export.

    Defines how a model should be exported to a specific format.
    """

    export_format: ExportFormat
    output_dir: Path
    model_name: str
    version: str = "1.0.0"

    # Export options
    include_config: bool = True
    include_metadata: bool = True
    include_optimizer: bool = False
    quantize: bool = False
    quantization_bits: Optional[int] = None

    # Compression
    compress: bool = False
    compression_level: int = 6

    # Framework-specific options
    optimize_for_inference: bool = True

    # Metadata
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # Custom options
    custom_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return {
            "export_format": self.export_format.value,
            "output_dir": str(self.output_dir),
            "model_name": self.model_name,
            "version": self.version,
            "include_config": self.include_config,
            "include_metadata": self.include_metadata,
            "include_optimizer": self.include_optimizer,
            "quantize": self.quantize,
            "quantization_bits": self.quantization_bits,
            "compress": self.compress,
            "compression_level": self.compression_level,
            "optimize_for_inference": self.optimize_for_inference,
            "description": self.description,
            "tags": self.tags,
            "custom_options": self.custom_options,
        }


@dataclass
class LoadConfig:
    """Configuration for model loading.

    Defines how a model should be loaded from a specific format.
    """

    model_path: Path
    format: ExportFormat
    model_name: str
    version: str = "1.0.0"

    # Loading options
    load_config: bool = True
    load_metadata: bool = True
    strict: bool = True

    # Device settings
    device: str = "cpu"
    dtype: Optional[str] = None

    # Performance options
    enable_caching: bool = True
    optimize_memory: bool = False

    # Custom options
    custom_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return {
            "model_path": str(self.model_path),
            "format": self.format.value,
            "model_name": self.model_name,
            "version": self.version,
            "load_config": self.load_config,
            "load_metadata": self.load_metadata,
            "strict": self.strict,
            "device": self.device,
            "dtype": self.dtype,
            "enable_caching": self.enable_caching,
            "optimize_memory": self.optimize_memory,
            "custom_options": self.custom_options,
        }
