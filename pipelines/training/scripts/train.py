"""Training pipeline entry point."""

import logging
import yaml
import argparse
from pathlib import Path

from src.orchestrator import TrainingOrchestrator, TrainingConfig
from src.callbacks import CallbackManager, LoggingCallback, WandbCallback
from src.checkpointing import CheckpointManager
from src.methods import TrainingMethodFactory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_training(config: dict):
    """Run training pipeline."""
    logger.info("=" * 80)
    logger.info("LLM-Whisperer Training Pipeline")
    logger.info("=" * 80)

    # Create training configuration
    training_config = TrainingConfig(**config.get("training", {}))

    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(training_config)
    logger.info("Training orchestrator initialized")

    # Setup callbacks
    callback_manager = CallbackManager()
    callback_manager.add_callback(
        LoggingCallback(log_interval=training_config.log_steps)
    )

    if training_config.use_wandb:
        callback_manager.add_callback(WandbCallback())

    # Notify callbacks
    callback_manager.on_training_start(
        {
            "model": training_config.model_name,
            "method": training_config.training_method,
            "epochs": training_config.num_epochs,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
        }
    )

    # Load model
    logger.info("\n[STEP 1] Loading Model")
    logger.info("-" * 40)
    orchestrator.load_model()
    logger.info(f"Model loaded: {training_config.model_name}")

    # Apply training method
    logger.info("\n[STEP 2] Applying Training Method")
    logger.info("-" * 40)
    method_config = {
        "lora_rank": training_config.lora_rank,
        "lora_alpha": training_config.lora_alpha,
        "lora_dropout": training_config.lora_dropout,
    }
    TrainingMethodFactory.apply_method(
        orchestrator.model, training_config.training_method, method_config
    )

    # Setup training components
    logger.info("\n[STEP 3] Setting Up Training")
    logger.info("-" * 40)
    orchestrator.setup_training()

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        f"{training_config.output_dir}/checkpoints", max_checkpoints=5
    )

    # Training statistics
    logger.info("\n[STEP 4] Training Statistics")
    logger.info("-" * 40)
    stats = orchestrator.get_training_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    logger.info("\n" + "=" * 80)
    logger.info("Training Ready to Start")
    logger.info("=" * 80)

    # Final metrics
    callback_manager.on_training_end(
        {
            "status": "ready",
            "model": training_config.model_name,
            "method": training_config.training_method,
            "total_parameters": stats["total_parameters"],
            "trainable_parameters": stats["trainable_parameters"],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_training(config)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
