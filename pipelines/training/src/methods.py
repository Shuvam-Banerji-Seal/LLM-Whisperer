"""Training methods for different fine-tuning approaches."""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingMethodConfig:
    """Configuration for training method."""

    method: str  # full_finetune, lora, qlora, dpo
    gradient_checkpointing: bool = True
    mixed_precision: bool = True


class FullFineTune:
    """Full fine-tuning approach."""

    @staticmethod
    def apply(model, config: Dict[str, Any]):
        """Apply full fine-tuning (all parameters trainable).

        Args:
            model: Model to fine-tune
            config: Configuration dictionary
        """
        # All parameters are trainable by default
        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        logger.info(f"Full Fine-Tuning:")
        logger.info(f"  Trainable parameters: {trainable:,}")
        logger.info(f"  Total parameters: {total:,}")
        logger.info(f"  Trainable%: {trainable / total * 100:.2f}%")

        return model


class LoRA:
    """Low-Rank Adaptation (LoRA) approach."""

    @staticmethod
    def apply(model, config: Dict[str, Any]):
        """Apply LoRA to model.

        Args:
            model: Model to apply LoRA
            config: Configuration with lora_rank, lora_alpha, lora_dropout
        """
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("peft library required for LoRA")

        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 16)
        lora_dropout = config.get("lora_dropout", 0.05)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],  # For most models
        )

        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        logger.info(f"LoRA Fine-Tuning:")
        logger.info(f"  Rank: {lora_rank}")
        logger.info(f"  Alpha: {lora_alpha}")
        logger.info(f"  Dropout: {lora_dropout}")
        logger.info(f"  Trainable parameters: {trainable:,}")
        logger.info(f"  Total parameters: {total:,}")
        logger.info(f"  Trainable%: {trainable / total * 100:.2f}%")

        return model


class QLoRA:
    """Quantized Low-Rank Adaptation (QLoRA) approach."""

    @staticmethod
    def apply(model, config: Dict[str, Any]):
        """Apply QLoRA to model.

        Args:
            model: Quantized model to apply QLoRA
            config: Configuration with lora parameters
        """
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("peft library required for QLoRA")

        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 16)
        lora_dropout = config.get("lora_dropout", 0.05)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        logger.info(f"QLoRA Fine-Tuning (Quantized):")
        logger.info(f"  Rank: {lora_rank}")
        logger.info(f"  Alpha: {lora_alpha}")
        logger.info(f"  Dropout: {lora_dropout}")
        logger.info(f"  Trainable parameters: {trainable:,}")
        logger.info(f"  Total parameters: {total:,}")
        logger.info(f"  Trainable%: {trainable / total * 100:.2f}%")

        return model


class DPO:
    """Direct Preference Optimization (DPO) approach."""

    @staticmethod
    def apply(model, config: Dict[str, Any]):
        """Prepare model for DPO training.

        DPO requires both reference and target models.
        This sets up the target model.

        Args:
            model: Model to prepare for DPO
            config: Configuration dictionary
        """
        # For DPO, we typically need to set specific parameters
        logger.info("DPO Fine-Tuning Setup:")
        logger.info("  Note: DPO requires paired preference data")
        logger.info("  Both chosen and rejected samples needed")

        # Make model trainable
        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        logger.info(f"  Trainable parameters: {trainable:,}")
        logger.info(f"  Total parameters: {total:,}")

        return model


class TrainingMethodFactory:
    """Factory for creating training methods."""

    methods = {
        "full_finetune": FullFineTune,
        "lora": LoRA,
        "qlora": QLoRA,
        "dpo": DPO,
    }

    @classmethod
    def get_method(cls, method_name: str):
        """Get training method by name.

        Args:
            method_name: Name of training method

        Returns:
            Training method class
        """
        if method_name not in cls.methods:
            raise ValueError(
                f"Unknown method: {method_name}. Available: {list(cls.methods.keys())}"
            )

        return cls.methods[method_name]

    @classmethod
    def apply_method(cls, model, method_name: str, config: Dict[str, Any]):
        """Apply training method to model.

        Args:
            model: Model to apply method
            method_name: Name of training method
            config: Configuration dictionary

        Returns:
            Modified model
        """
        method = cls.get_method(method_name)
        return method.apply(model, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: show available methods
    print("Available training methods:")
    for method in TrainingMethodFactory.methods.keys():
        print(f"  - {method}")
