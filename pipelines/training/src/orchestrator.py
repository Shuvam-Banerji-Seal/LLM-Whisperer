"""Training orchestrator for managing training pipelines."""

import logging
import torch
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training orchestrator."""

    model_name: str
    dataset_path: str
    output_dir: str

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Model parameters
    lora_rank: Optional[int] = None
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    quantization_enabled: bool = False

    # Training method
    training_method: str = "full_finetune"  # full_finetune, lora, qlora, dpo

    # Optimization
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1

    # Evaluation and monitoring
    eval_steps: int = 500
    save_steps: int = 500
    log_steps: int = 100

    # Monitoring
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    use_mlflow: bool = False
    mlflow_experiment: Optional[str] = None


class TrainingOrchestrator:
    """Main training orchestrator for managing training workflows."""

    def __init__(self, config: TrainingConfig):
        """Initialize training orchestrator.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.training_state = {}

        self._setup_directories()
        self._setup_monitoring()

    def _setup_directories(self):
        """Create necessary directories."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        (output_path / "checkpoints").mkdir(exist_ok=True)
        (output_path / "logs").mkdir(exist_ok=True)
        (output_path / "eval_results").mkdir(exist_ok=True)

    def _setup_monitoring(self):
        """Setup monitoring tools."""
        if self.config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=self.config.wandb_project or "llm-training",
                    config=self.config.__dict__,
                )
                logger.info("Weights & Biases monitoring initialized")
            except ImportError:
                logger.warning("wandb not installed, skipping W&B monitoring")

        if self.config.use_mlflow:
            try:
                import mlflow

                mlflow.set_experiment(self.config.mlflow_experiment or "training")
                mlflow.log_params(self.config.__dict__)
                logger.info("MLflow monitoring initialized")
            except ImportError:
                logger.warning("mlflow not installed, skipping MLflow monitoring")

    def load_model(self):
        """Load model based on configuration."""
        logger.info(f"Loading model: {self.config.model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required for model loading")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model based on quantization setting
        if self.config.quantization_enabled:
            import torch
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        # Apply training method
        if self.config.training_method == "lora":
            self._apply_lora()
        elif self.config.training_method == "qlora":
            self._apply_qlora()

        logger.info(f"Model loaded successfully")
        return self.model

    def _apply_lora(self):
        """Apply LoRA to model."""
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("peft library required for LoRA")

        lora_config = LoraConfig(
            r=self.config.lora_rank or 8,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA applied to model")

    def _apply_qlora(self):
        """Apply QLoRA to model."""
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("peft library required for QLoRA")

        lora_config = LoraConfig(
            r=self.config.lora_rank or 8,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        logger.info("QLoRA applied to model (with quantization)")

    def setup_training(self):
        """Setup training components (optimizer, scheduler, etc.)."""
        logger.info("Setting up training components")

        from transformers import AdamW, get_linear_schedule_with_warmup

        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Setup optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup learning rate scheduler
        total_steps = self._estimate_total_steps()
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        logger.info(
            f"Scheduler: LinearWarmup (warmup_steps={self.config.warmup_steps})"
        )

    def _estimate_total_steps(self) -> int:
        """Estimate total training steps."""
        # This would need access to dataset size
        # For now, return a placeholder
        return 10000

    def save_checkpoint(self, checkpoint_dir: Optional[str] = None):
        """Save model checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
        """
        if checkpoint_dir is None:
            checkpoint_dir = f"{self.config.output_dir}/checkpoints/latest"

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        logger.info(f"Checkpoint loaded from {checkpoint_dir}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            "model_name": self.config.model_name,
            "training_method": self.config.training_method,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "num_epochs": self.config.num_epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
        }

        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = TrainingConfig(
        model_name="gpt2",
        dataset_path="data/processed",
        output_dir="./training_outputs",
        training_method="full_finetune",
    )

    orchestrator = TrainingOrchestrator(config)
    print(orchestrator.get_training_stats())
