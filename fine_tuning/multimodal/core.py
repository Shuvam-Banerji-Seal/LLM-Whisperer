"""Multimodal tuning core implementation."""

import logging
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader

from fine_tuning.base.core import BaseFinetuner, FinetuningMetrics
from .config import MultimodalTuningConfig

logger = logging.getLogger(__name__)


class MultimodalFinetuner(BaseFinetuner):
    """Fine-tuner for multimodal models.

    Enables fine-tuning of vision-language, video-language, and other
    multimodal architectures.
    """

    def __init__(self, config: MultimodalTuningConfig):
        """Initialize multimodal fine-tuner.

        Args:
            config: Multimodal tuning configuration
        """
        if not isinstance(config, MultimodalTuningConfig):
            raise TypeError(f"Expected MultimodalTuningConfig, got {type(config)}")

        super().__init__(config)
        self.multimodal_config = config
        self.vision_encoder = None
        self.fusion_module = None

    def setup_model(self) -> None:
        """Setup multimodal model."""
        logger.info(f"Loading multimodal model: {self.config.model_name}")

        try:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float32,
                use_auth_token=self.config.use_auth_token,
                trust_remote_code=self.config.trust_remote_code,
            )
        except ImportError:
            logger.error("transformers library not installed")
            raise

        self.model.to(self.device)
        logger.info("Multimodal model setup complete")

    def setup_optimizer(self) -> None:
        """Setup optimizer."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        optimizer_type = self.config.optimizer.type.value
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        logger.info("Optimizer setup complete")

    def setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")

        from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.scheduler.num_warmup_steps,
            num_training_steps=self.config.scheduler.num_training_steps,
        )

        logger.info("Scheduler setup complete")

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Train the multimodal model.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader

        Returns:
            Training results
        """
        self.state.start_time = __import__("datetime").datetime.now()
        logger.info("Starting multimodal fine-tuning")

        results = {
            "training_loss": [],
            "eval_loss": [],
            "align_loss": [],
            "contrastive_loss": [],
        }

        self.model.train()

        for epoch in range(self.config.training.num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0

            for step, batch in enumerate(train_dataloader):
                self.state.step += 1

                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                except Exception as e:
                    logger.error(f"Error in forward pass: {e}")
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()

                if step % self.config.training.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    logger.info(f"Epoch {epoch}, Step {step}: Loss = {avg_loss:.4f}")

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            results["training_loss"].append(avg_epoch_loss)

            logger.info(f"Epoch {epoch} completed: Loss = {avg_epoch_loss:.4f}")

            if eval_dataloader:
                eval_metrics = self.evaluate(eval_dataloader)
                results["eval_loss"].append(eval_metrics.loss)

            self.save_checkpoint()

        self.state.end_time = __import__("datetime").datetime.now()
        logger.info("Multimodal training complete")

        return results

    def evaluate(self, eval_dataloader: DataLoader) -> FinetuningMetrics:
        """Evaluate the model.

        Args:
            eval_dataloader: Evaluation data loader

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    total_loss += loss.item()
                except Exception as e:
                    logger.error(f"Error in evaluation: {e}")
                    continue

        avg_loss = total_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0

        return FinetuningMetrics(
            loss=avg_loss,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )
