"""Reward model training core implementation."""

import logging
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fine_tuning.base.core import BaseFinetuner, FinetuningMetrics
from .config import RewardModelingConfig

logger = logging.getLogger(__name__)


class RewardModelFinetuner(BaseFinetuner):
    \"\"\"Fine-tuner for reward models.\"\"\"

    def __init__(self, config: RewardModelingConfig):
        if not isinstance(config, RewardModelingConfig):
            raise TypeError(f"Expected RewardModelingConfig, got {type(config)}")
        super().__init__(config)
        self.reward_config = config

    def setup_model(self) -> None:
        logger.info(f"Loading reward model: {self.config.model_name}")
        try:
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=1,
                cache_dir=self.config.cache_dir,
            )
        except ImportError:
            logger.error("transformers library not installed")
            raise
        self.model.to(self.device)
        logger.info("Reward model setup complete")

    def setup_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
        )

    def setup_scheduler(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")
        from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.scheduler.num_warmup_steps,
            num_training_steps=self.config.scheduler.num_training_steps,
        )

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        self.state.start_time = __import__("datetime").datetime.now()
        logger.info("Starting reward model training")
        results = {"training_loss": [], "eval_loss": []}
        self.model.train()

        for epoch in range(self.config.training.num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0

            for step, batch in enumerate(train_dataloader):
                self.state.step += 1
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                except Exception as e:
                    logger.error(f"Error: {e}")
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()

            results["training_loss"].append(epoch_loss / len(train_dataloader))
            if eval_dataloader:
                eval_metrics = self.evaluate(eval_dataloader)
                results["eval_loss"].append(eval_metrics.loss)
            self.save_checkpoint()

        self.state.end_time = __import__("datetime").datetime.now()
        logger.info("Reward model training complete")
        return results

    def evaluate(self, eval_dataloader: DataLoader) -> FinetuningMetrics:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                except Exception as e:
                    logger.error(f"Error: {e}")
        return FinetuningMetrics(
            loss=total_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )
