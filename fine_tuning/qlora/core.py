"""QLoRA core implementation."""

import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

from fine_tuning.lora.core import LoRAFinetuner
from fine_tuning.base.core import FinetuningMetrics
from .config import QLoRAConfig

logger = logging.getLogger(__name__)


class QLoRAFinetuner(LoRAFinetuner):
    """QLoRA fine-tuning implementation.

    Combines LoRA with 4-bit quantization for extreme memory efficiency.
    """

    def __init__(self, config: QLoRAConfig):
        """Initialize QLoRA fine-tuner.

        Args:
            config: QLoRA configuration
        """
        if not isinstance(config, QLoRAConfig):
            raise TypeError(f"Expected QLoRAConfig, got {type(config)}")

        super().__init__(config)
        self.qlora_config = config

    def setup_model(self) -> None:
        """Setup model with 4-bit quantization and LoRA.

        Loads the model with 4-bit quantization and applies LoRA layers.
        """
        logger.info(f"Loading model with 4-bit quantization: {self.config.model_name}")

        try:
            from transformers import (
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )
        except ImportError:
            logger.error("transformers library not installed")
            raise

        try:
            import bitsandbytes
        except ImportError:
            logger.error(
                "bitsandbytes library not installed. Install with: pip install bitsandbytes"
            )
            raise

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.qlora_config.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16
            if self.qlora_config.bnb_4bit_compute_dtype == "float16"
            else torch.bfloat16,
            bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant,
        )

        logger.info(f"BitsAndBytes Config: {bnb_config}")

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            cache_dir=self.config.cache_dir,
            use_auth_token=self.config.use_auth_token,
            trust_remote_code=self.config.trust_remote_code,
            device_map="auto",
        )

        # Freeze all original parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA to target modules
        self._apply_lora()

        logger.info("Model setup complete with 4-bit quantization and LoRA layers")

    def get_trainable_params_info(self) -> dict:
        """Get information about trainable parameters.

        Returns:
            Dictionary with parameter statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params
            if total_params > 0
            else 0,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> dict:
        """Train the model with QLoRA.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader

        Returns:
            Training results
        """
        # Log parameter information
        param_info = self.get_trainable_params_info()
        logger.info(f"Trainable Parameters Info: {param_info}")

        return super().train(train_dataloader, eval_dataloader)

    def get_memory_footprint(self) -> dict:
        """Get model memory footprint.

        Returns:
            Dictionary with memory statistics
        """
        # Model memory (quantized)
        model_memory_mb = sum(
            p.element_size() * p.nelement() / 1024 / 1024
            for p in self.model.parameters()
        )

        # Optimizer memory (approximate: 2x model size for AdamW)
        optimizer_memory_mb = model_memory_mb * 2 if self.optimizer else 0

        # Gradient memory (same as model)
        gradient_memory_mb = model_memory_mb

        total_memory_mb = model_memory_mb + optimizer_memory_mb + gradient_memory_mb

        return {
            "model_memory_mb": model_memory_mb,
            "optimizer_memory_mb": optimizer_memory_mb,
            "gradient_memory_mb": gradient_memory_mb,
            "total_memory_mb": total_memory_mb,
        }
