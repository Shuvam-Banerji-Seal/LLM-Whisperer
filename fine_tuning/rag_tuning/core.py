"""RAG tuning core implementation."""

import logging
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader

from fine_tuning.base.core import BaseFinetuner, FinetuningMetrics
from .config import RAGTuningConfig

logger = logging.getLogger(__name__)


class RAGFinetuner(BaseFinetuner):
    \"\"\"Fine-tuner for RAG systems.\"\"\"

    def __init__(self, config: RAGTuningConfig):
        if not isinstance(config, RAGTuningConfig):
            raise TypeError(f"Expected RAGTuningConfig, got {type(config)}")
        super().__init__(config)
        self.rag_config = config

    def setup_model(self) -> None:
        logger.info(f"Loading RAG model: {self.config.model_name}")
        try:
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float32,
            )
        except ImportError:
            logger.error("transformers library not installed")
            raise
        self.model.to(self.device)
        logger.info("RAG model setup complete")

    def setup_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
        logger.info("Optimizer setup complete")

    def setup_scheduler(self) -> None:
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
        self.state.start_time = __import__("datetime").datetime.now()
        logger.info("Starting RAG fine-tuning")
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

                if step % self.config.training.logging_steps == 0:
                    logger.info(f"Epoch {epoch}, Step {step}: Loss = {epoch_loss / (step + 1):.4f}")

            results["training_loss"].append(epoch_loss / len(train_dataloader))
            if eval_dataloader:
                eval_metrics = self.evaluate(eval_dataloader)
                results["eval_loss"].append(eval_metrics.loss)
            self.save_checkpoint()

        self.state.end_time = __import__("datetime").datetime.now()
        logger.info("RAG training complete")
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
                    continue
        return FinetuningMetrics(
            loss=total_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )
