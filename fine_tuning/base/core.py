"""Core base classes for fine-tuning."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .config import BaseFinetuningConfig

logger = logging.getLogger(__name__)


@dataclass
class FinetuningMetrics:
    """Container for fine-tuning metrics."""

    loss: float
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[int] = None
    step: Optional[int] = None
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary.

        Returns:
            Dictionary representation of metrics
        """
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "perplexity": self.perplexity,
            "learning_rate": self.learning_rate,
            "epoch": self.epoch,
            "step": self.step,
            "timestamp": self.timestamp,
            **self.custom_metrics,
        }


class FinetuningState:
    """Manages the state of fine-tuning process."""

    def __init__(self):
        """Initialize fine-tuning state."""
        self.epoch: int = 0
        self.step: int = 0
        self.best_metric: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_step: Optional[int] = None
        self.best_checkpoint: Optional[str] = None
        self.metrics_history: List[FinetuningMetrics] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def update(self, metrics: FinetuningMetrics, is_best: bool = False) -> None:
        """Update state with new metrics.

        Args:
            metrics: New metrics
            is_best: Whether these are the best metrics so far
        """
        metrics.epoch = self.epoch
        metrics.step = self.step
        self.metrics_history.append(metrics)

        if is_best:
            self.best_metric = metrics.loss
            self.best_epoch = self.epoch
            self.best_step = self.step
            logger.info(
                f"New best metric: {self.best_metric} at epoch {self.best_epoch}, step {self.best_step}"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of fine-tuning state.

        Returns:
            Summary dictionary
        """
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "epochs": self.epoch,
            "steps": self.step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "best_checkpoint": self.best_checkpoint,
            "total_metrics_recorded": len(self.metrics_history),
            "duration_seconds": duration,
        }


class FinetuningCallback(ABC):
    """Base class for fine-tuning callbacks."""

    @abstractmethod
    def on_init(self, trainer: "BaseFinetuner") -> None:
        """Called when trainer is initialized.

        Args:
            trainer: The fine-tuner instance
        """
        pass

    @abstractmethod
    def on_epoch_start(self, trainer: "BaseFinetuner") -> None:
        """Called at the start of each epoch.

        Args:
            trainer: The fine-tuner instance
        """
        pass

    @abstractmethod
    def on_epoch_end(
        self, trainer: "BaseFinetuner", metrics: FinetuningMetrics
    ) -> None:
        """Called at the end of each epoch.

        Args:
            trainer: The fine-tuner instance
            metrics: Epoch metrics
        """
        pass

    @abstractmethod
    def on_step_end(self, trainer: "BaseFinetuner", metrics: FinetuningMetrics) -> None:
        """Called at the end of each training step.

        Args:
            trainer: The fine-tuner instance
            metrics: Step metrics
        """
        pass

    @abstractmethod
    def on_train_end(self, trainer: "BaseFinetuner") -> None:
        """Called at the end of training.

        Args:
            trainer: The fine-tuner instance
        """
        pass


class BaseFinetuner(ABC):
    """Abstract base class for all fine-tuning approaches."""

    def __init__(self, config: BaseFinetuningConfig):
        """Initialize base fine-tuner.

        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.device = torch.device(
            "cpu"
            if config.training.use_cpu
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.state = FinetuningState()
        self.callbacks: List[FinetuningCallback] = []
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Using device: {self.device}")

    @abstractmethod
    def setup_model(self) -> None:
        """Setup the model for fine-tuning.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def setup_optimizer(self) -> None:
        """Setup the optimizer.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def setup_scheduler(self) -> None:
        """Setup the learning rate scheduler.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader

        Returns:
            Training results dictionary
        """
        pass

    @abstractmethod
    def evaluate(self, eval_dataloader: DataLoader) -> FinetuningMetrics:
        """Evaluate the model.

        Args:
            eval_dataloader: Evaluation data loader

        Returns:
            Evaluation metrics
        """
        pass

    def add_callback(self, callback: FinetuningCallback) -> None:
        """Add a callback.

        Args:
            callback: Callback to add
        """
        self.callbacks.append(callback)
        logger.info(f"Added callback: {callback.__class__.__name__}")

    def remove_callback(self, callback: FinetuningCallback) -> None:
        """Remove a callback.

        Args:
            callback: Callback to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"Removed callback: {callback.__class__.__name__}")

    def save_checkpoint(self, checkpoint_dir: Optional[str] = None) -> str:
        """Save model checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint. If None, uses config.output_dir

        Returns:
            Path to saved checkpoint
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.config.output_dir

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{self.state.epoch}"
        logger.info(f"Saving checkpoint to {checkpoint_path}")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config.to_dict(),
                "state": self.state.get_summary(),
            },
            checkpoint_path,
        )

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info("Checkpoint loaded successfully")

    def get_model(self) -> torch.nn.Module:
        """Get the model.

        Returns:
            The fine-tuned model
        """
        if self.model is None:
            raise RuntimeError("Model not set up. Call setup_model() first.")
        return self.model

    def get_optimizer(self) -> Optimizer:
        """Get the optimizer.

        Returns:
            The optimizer
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set up. Call setup_optimizer() first.")
        return self.optimizer

    def get_scheduler(self) -> LRScheduler:
        """Get the scheduler.

        Returns:
            The learning rate scheduler
        """
        if self.scheduler is None:
            raise RuntimeError("Scheduler not set up. Call setup_scheduler() first.")
        return self.scheduler
