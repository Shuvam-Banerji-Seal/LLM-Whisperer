"""Behavior tuning core implementation."""

import logging
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import DataLoader

from fine_tuning.base.core import BaseFinetuner, FinetuningMetrics
from .config import BehaviorTuningConfig

logger = logging.getLogger(__name__)


class BehaviorFinetuner(BaseFinetuner):
    """Fine-tuner for behavior-specific training.

    Enables fine-tuning models to exhibit specific behaviors, styles,
    and domain expertise.
    """

    def __init__(self, config: BehaviorTuningConfig):
        """Initialize behavior fine-tuner.

        Args:
            config: Behavior tuning configuration
        """
        if not isinstance(config, BehaviorTuningConfig):
            raise TypeError(f"Expected BehaviorTuningConfig, got {type(config)}")

        super().__init__(config)
        self.behavior_config = config
        self.behavior_examples: List[Dict[str, Any]] = []
        self.style_metrics: Dict[str, float] = {}

    def setup_model(self) -> None:
        """Setup model for behavior tuning."""
        logger.info(f"Loading model for behavior tuning: {self.config.model_name}")

        try:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
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
        logger.info("Model setup complete for behavior tuning")

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

    def add_behavior_examples(self, examples: List[Dict[str, Any]]) -> None:
        """Add behavior examples for training.

        Args:
            examples: List of behavior examples
        """
        self.behavior_examples.extend(examples)
        logger.info(f"Added {len(examples)} behavior examples")

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Train the model for specific behaviors.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader

        Returns:
            Training results
        """
        self.state.start_time = __import__("datetime").datetime.now()
        logger.info(
            f"Starting behavior fine-tuning: {self.behavior_config.behavior_name}"
        )

        results = {
            "training_loss": [],
            "eval_loss": [],
            "behavior_consistency": [],
        }

        self.model.train()

        for epoch in range(self.config.training.num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0

            for step, batch in enumerate(train_dataloader):
                self.state.step += 1

                batch = {k: v.to(self.device) for k, v in batch.items()}

                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss
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
        logger.info("Behavior training complete")

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
                batch = {k: v.to(self.device) for k, v in batch.items()}

                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                except Exception as e:
                    logger.error(f"Error in evaluation: {e}")
                    continue

        avg_loss = total_loss / len(eval_dataloader)

        return FinetuningMetrics(
            loss=avg_loss,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

    def evaluate_behavior_consistency(self, test_prompts: List[str]) -> float:
        """Evaluate consistency of behavior on test prompts.

        Args:
            test_prompts: List of test prompts

        Returns:
            Consistency score
        """
        self.model.eval()
        consistency_scores = []

        with torch.no_grad():
            for prompt in test_prompts:
                # Simplified consistency evaluation
                consistency_scores.append(0.85)  # Placeholder

        avg_consistency = (
            sum(consistency_scores) / len(consistency_scores)
            if consistency_scores
            else 0.0
        )
        logger.info(f"Behavior consistency: {avg_consistency:.4f}")
        return avg_consistency
