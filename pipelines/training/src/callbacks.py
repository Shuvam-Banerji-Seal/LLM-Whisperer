"""Training callbacks for monitoring and custom logic."""

import logging
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class CallbackConfig:
    """Configuration for callbacks."""

    log_interval: int = 100
    eval_interval: int = 500
    checkpoint_interval: int = 500
    use_tensorboard: bool = False
    tensorboard_dir: Optional[str] = None


class TrainingCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_training_start(self, config: Dict[str, Any]):
        """Called at training start."""
        pass

    @abstractmethod
    def on_training_end(self, metrics: Dict[str, Any]):
        """Called at training end."""
        pass

    @abstractmethod
    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        """Called at end of each training step."""
        pass

    @abstractmethod
    def on_eval(self, step: int, eval_metrics: Dict[str, Any]):
        """Called after evaluation."""
        pass


class LoggingCallback(TrainingCallback):
    """Callback for logging training metrics."""

    def __init__(self, log_interval: int = 100):
        """Initialize logging callback.

        Args:
            log_interval: Steps between logs
        """
        self.log_interval = log_interval

    def on_training_start(self, config: Dict[str, Any]):
        """Log training configuration."""
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        for key, value in config.items():
            logger.info(f"{key}: {value}")

    def on_training_end(self, metrics: Dict[str, Any]):
        """Log final metrics."""
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete")
        logger.info("=" * 80)
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")

    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        """Log step metrics."""
        if step % self.log_interval == 0:
            logger.info(f"Step {step}: {metrics}")

    def on_eval(self, step: int, eval_metrics: Dict[str, Any]):
        """Log evaluation metrics."""
        logger.info(f"Evaluation at step {step}: {eval_metrics}")


class WandbCallback(TrainingCallback):
    """Callback for Weights & Biases logging."""

    def __init__(self):
        """Initialize W&B callback."""
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            logger.warning("wandb not installed")
            self.wandb = None

    def on_training_start(self, config: Dict[str, Any]):
        """Log config to W&B."""
        if self.wandb:
            self.wandb.config.update(config)

    def on_training_end(self, metrics: Dict[str, Any]):
        """Log final metrics to W&B."""
        if self.wandb:
            self.wandb.log({"final_metrics": metrics})

    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        """Log step metrics to W&B."""
        if self.wandb:
            self.wandb.log({"step": step, **metrics})

    def on_eval(self, step: int, eval_metrics: Dict[str, Any]):
        """Log evaluation metrics to W&B."""
        if self.wandb:
            self.wandb.log(
                {"eval_step": step, **{f"eval_{k}": v for k, v in eval_metrics.items()}}
            )


class MLflowCallback(TrainingCallback):
    """Callback for MLflow logging."""

    def __init__(self):
        """Initialize MLflow callback."""
        try:
            import mlflow

            self.mlflow = mlflow
        except ImportError:
            logger.warning("mlflow not installed")
            self.mlflow = None

    def on_training_start(self, config: Dict[str, Any]):
        """Log config to MLflow."""
        if self.mlflow:
            self.mlflow.log_params(config)

    def on_training_end(self, metrics: Dict[str, Any]):
        """Log final metrics to MLflow."""
        if self.mlflow:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(f"final_{key}", value)

    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        """Log step metrics to MLflow."""
        if self.mlflow:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(key, value, step=step)

    def on_eval(self, step: int, eval_metrics: Dict[str, Any]):
        """Log evaluation metrics to MLflow."""
        if self.mlflow:
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(f"eval_{key}", value, step=step)


class CallbackManager:
    """Manager for training callbacks."""

    def __init__(self):
        """Initialize callback manager."""
        self.callbacks = []

    def add_callback(self, callback: TrainingCallback):
        """Add callback.

        Args:
            callback: Callback to add
        """
        self.callbacks.append(callback)
        logger.info(f"Added callback: {callback.__class__.__name__}")

    def remove_callback(self, callback_class):
        """Remove callback by class.

        Args:
            callback_class: Callback class to remove
        """
        self.callbacks = [
            cb for cb in self.callbacks if not isinstance(cb, callback_class)
        ]

    def on_training_start(self, config: Dict[str, Any]):
        """Call on_training_start on all callbacks."""
        for callback in self.callbacks:
            callback.on_training_start(config)

    def on_training_end(self, metrics: Dict[str, Any]):
        """Call on_training_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_training_end(metrics)

    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        """Call on_step_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_step_end(step, metrics)

    def on_eval(self, step: int, eval_metrics: Dict[str, Any]):
        """Call on_eval on all callbacks."""
        for callback in self.callbacks:
            callback.on_eval(step, eval_metrics)

    def get_callback(self, callback_class) -> Optional[TrainingCallback]:
        """Get specific callback by class.

        Args:
            callback_class: Callback class to retrieve

        Returns:
            Callback instance or None
        """
        for callback in self.callbacks:
            if isinstance(callback, callback_class):
                return callback
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    manager = CallbackManager()
    manager.add_callback(LoggingCallback())
    manager.add_callback(WandbCallback())

    manager.on_training_start({"model": "gpt2", "epochs": 3})
    manager.on_step_end(100, {"loss": 2.5, "lr": 5e-4})
    manager.on_eval(100, {"eval_loss": 2.3})
    manager.on_training_end({"final_loss": 1.8, "accuracy": 0.95})
