"""Checkpoint management for training pipelines."""

import logging
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    step: int
    epoch: int
    global_step: int
    loss: float
    eval_loss: Optional[float] = None
    timestamp: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class CheckpointManager:
    """Manager for model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum checkpoints to keep
            save_best_only: Only save best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = float("inf")

        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self):
        """Load metadata of existing checkpoints."""
        for checkpoint_dir in sorted(self.checkpoint_dir.glob("checkpoint-*")):
            metadata_file = checkpoint_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata_dict = json.load(f)
                    self.checkpoints.append(checkpoint_dir)

    def save_checkpoint(
        self,
        model,
        step: int,
        epoch: int,
        loss: float,
        eval_loss: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ):
        """Save model checkpoint.

        Args:
            model: Model to save
            step: Current training step
            epoch: Current epoch
            loss: Training loss
            eval_loss: Evaluation loss
            config: Training configuration
            is_best: Whether this is best checkpoint
        """
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save model
            model.save_pretrained(checkpoint_path)

            # Save metadata
            metadata = CheckpointMetadata(
                step=step,
                epoch=epoch,
                global_step=step,
                loss=loss,
                eval_loss=eval_loss,
                config=config,
            )

            metadata_file = checkpoint_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(metadata), f, indent=2, default=str)

            self.checkpoints.append(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Track best checkpoint
            metric = eval_loss if eval_loss is not None else loss
            if metric < self.best_metric:
                self.best_metric = metric
                self.best_checkpoint = checkpoint_path
                self._save_best_link()

            # Remove old checkpoints if needed
            if len(self.checkpoints) > self.max_checkpoints:
                self._remove_old_checkpoint()

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def _save_best_link(self):
        """Create symlink to best checkpoint."""
        best_link = self.checkpoint_dir / "best"

        # Remove old link if exists
        if best_link.is_symlink():
            best_link.unlink()

        # Create new link
        try:
            best_link.symlink_to(self.best_checkpoint.name)
            logger.info(f"Updated best checkpoint: {self.best_checkpoint}")
        except OSError:
            # Symlinks might not be supported on all platforms
            pass

    def _remove_old_checkpoint(self):
        """Remove oldest checkpoint."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by step number
        sorted_checkpoints = sorted(
            self.checkpoints, key=lambda x: int(x.name.split("-")[1])
        )

        oldest = sorted_checkpoints[0]

        try:
            import shutil

            shutil.rmtree(oldest)
            self.checkpoints.remove(oldest)
            logger.info(f"Removed old checkpoint: {oldest}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {oldest}: {e}")

    def load_checkpoint(self, model, checkpoint_path: Optional[str] = None):
        """Load model checkpoint.

        Args:
            model: Model to load into
            checkpoint_path: Path to checkpoint (None for latest)

        Returns:
            Checkpoint metadata
        """
        if checkpoint_path is None:
            # Load best checkpoint
            if self.best_checkpoint is None:
                raise ValueError("No checkpoints available")
            checkpoint_path = self.best_checkpoint
        else:
            checkpoint_path = Path(checkpoint_path)

        # Load model
        from transformers import AutoModelForCausalLM

        loaded_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        metadata = None
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return loaded_model, metadata

    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get information about all checkpoints."""
        info = []

        for checkpoint_path in sorted(self.checkpoints):
            metadata_file = checkpoint_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    info.append(
                        {
                            "path": str(checkpoint_path),
                            "name": checkpoint_path.name,
                            **metadata,
                        }
                    )

        return info

    def cleanup(self):
        """Clean up all checkpoints."""
        try:
            import shutil

            for checkpoint_path in self.checkpoints:
                shutil.rmtree(checkpoint_path)
            logger.info("Cleaned up all checkpoints")
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    manager = CheckpointManager("checkpoints/", max_checkpoints=3)

    # Print checkpoint info
    info = manager.get_checkpoint_info()
    for checkpoint in info:
        print(checkpoint)
