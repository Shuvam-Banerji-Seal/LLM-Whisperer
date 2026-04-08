"""Multimodal tuning configuration."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from fine_tuning.base.config import BaseFinetuningConfig

logger = logging.getLogger(__name__)


@dataclass
class MultimodalTuningConfig(BaseFinetuningConfig):
    """Configuration for multimodal model fine-tuning.

    Specialized for vision-language models, video-language models,
    and other multimodal architectures.
    """

    # Multimodal-specific parameters
    modality_types: List[str] = None  # ["vision", "language", "audio", "video"]
    vision_encoder: Optional[str] = "openai/clip-vit-base-patch32"
    video_encoder: Optional[str] = None
    align_loss_weight: float = 0.5  # Weight for alignment loss
    contrastive_loss_weight: float = 0.5  # Weight for contrastive loss
    fusion_method: str = "concat"  # "concat", "cross_attention", "transformer"
    freeze_vision_encoder: bool = False
    freeze_language_encoder: bool = False
    image_size: int = 224
    max_video_frames: int = 8

    def __post_init__(self):
        """Validate multimodal configuration."""
        super().__post_init__()

        if self.modality_types is None:
            self.modality_types = ["vision", "language"]

        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")

        logger.info(
            f"Multimodal tuning: {', '.join(self.modality_types)} fusion={self.fusion_method}"
        )
