"""Tests for training pipeline modules."""

import pytest
from dataclasses import dataclass

from src.orchestrator import TrainingConfig, TrainingOrchestrator
from src.callbacks import CallbackManager, LoggingCallback
from src.checkpointing import CheckpointManager, CheckpointMetadata
from src.methods import TrainingMethodFactory


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig(
            model_name="gpt2", dataset_path="data/processed", output_dir="./outputs"
        )

        assert config.model_name == "gpt2"
        assert config.num_epochs == 3
        assert config.batch_size == 16
        assert config.training_method == "full_finetune"

    def test_lora_config(self):
        """Test LoRA configuration."""
        config = TrainingConfig(
            model_name="mistral-7b",
            dataset_path="data/processed",
            output_dir="./outputs",
            training_method="lora",
            lora_rank=64,
            lora_alpha=16,
        )

        assert config.training_method == "lora"
        assert config.lora_rank == 64


class TestCallbacks:
    """Tests for callback system."""

    def test_add_callback(self):
        """Test adding callbacks."""
        manager = CallbackManager()
        manager.add_callback(LoggingCallback())

        assert len(manager.callbacks) == 1

    def test_callback_events(self):
        """Test callback event triggering."""
        manager = CallbackManager()
        manager.add_callback(LoggingCallback())

        # These should not raise
        manager.on_training_start({"model": "gpt2"})
        manager.on_step_end(100, {"loss": 2.5})
        manager.on_eval(100, {"eval_loss": 2.3})
        manager.on_training_end({"final_loss": 1.8})


class TestCheckpointManager:
    """Tests for checkpoint management."""

    def test_metadata_creation(self):
        """Test checkpoint metadata creation."""
        metadata = CheckpointMetadata(
            step=100, epoch=1, global_step=100, loss=2.5, eval_loss=2.3
        )

        assert metadata.step == 100
        assert metadata.loss == 2.5
        assert metadata.eval_loss == 2.3

    def test_checkpoint_manager_init(self, tmp_path):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(str(tmp_path))

        assert manager.checkpoint_dir.exists()
        assert manager.max_checkpoints == 5


class TestTrainingMethods:
    """Tests for training methods."""

    def test_factory_methods(self):
        """Test training method factory."""
        methods = ["full_finetune", "lora", "qlora", "dpo"]

        for method in methods:
            method_class = TrainingMethodFactory.get_method(method)
            assert method_class is not None

    def test_invalid_method(self):
        """Test invalid training method."""
        with pytest.raises(ValueError):
            TrainingMethodFactory.get_method("invalid_method")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
