"""LoRA core implementation."""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fine_tuning.base.core import BaseFinetuner, FinetuningMetrics
from fine_tuning.base.config import BaseFinetuningConfig
from .config import LoRAConfig

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """LoRA layer implementation.

    Adds trainable low-rank decomposition matrices to an existing linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
    ):
        """Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            r: Rank of low-rank decomposition
            lora_alpha: Scaling factor
            lora_dropout: Dropout probability for LoRA layers
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Low-rank decomposition matrices
        self.lora_a = nn.Linear(in_features, r, bias=False)
        self.lora_b = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(lora_dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize LoRA weights."""
        # Initialize A with normal distribution
        nn.init.normal_(self.lora_a.weight, std=1 / math.sqrt(self.in_features))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation.

        Args:
            x: Input tensor

        Returns:
            LoRA contribution
        """
        return self.lora_b(self.lora_a(self.lora_dropout(x))) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
    ):
        """Initialize LoRA linear layer.

        Args:
            original_layer: Original linear layer to adapt
            r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for LoRA
        """
        super().__init__()
        self.original_layer = original_layer
        self.lora_layer = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            r,
            lora_alpha,
            lora_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA.

        Args:
            x: Input tensor

        Returns:
            Output with LoRA adaptation
        """
        original_out = self.original_layer(x)
        lora_out = self.lora_layer(x)
        return original_out + lora_out


class LoRAFinetuner(BaseFinetuner):
    """LoRA fine-tuning implementation.

    Provides parameter-efficient fine-tuning using Low-Rank Adaptation.
    """

    def __init__(self, config: LoRAConfig):
        """Initialize LoRA fine-tuner.

        Args:
            config: LoRA configuration
        """
        if not isinstance(config, LoRAConfig):
            raise TypeError(f"Expected LoRAConfig, got {type(config)}")

        super().__init__(config)
        self.lora_config = config
        self.lora_layers = {}

    def setup_model(self) -> None:
        """Setup model with LoRA layers.

        Loads the model and replaces specified modules with LoRA-adapted versions.
        """
        logger.info(f"Loading model: {self.config.model_name}")

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

        # Freeze all original parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA to target modules
        self._apply_lora()
        self.model.to(self.device)

        logger.info("Model setup complete with LoRA layers")

    def _apply_lora(self) -> None:
        """Apply LoRA to target modules."""
        for name, module in self.model.named_modules():
            if self._should_apply_lora(name, module):
                self._replace_with_lora(name, module)

    def _should_apply_lora(self, name: str, module: nn.Module) -> bool:
        """Check if LoRA should be applied to a module.

        Args:
            name: Module name
            module: Module instance

        Returns:
            True if LoRA should be applied
        """
        if not isinstance(module, nn.Linear):
            return False

        for target in self.lora_config.target_modules:
            if target in name:
                return True

        return False

    def _replace_with_lora(self, name: str, module: nn.Linear) -> None:
        """Replace a module with LoRA version.

        Args:
            name: Module name
            module: Original module
        """
        lora_linear = LoRALinear(
            module,
            self.lora_config.r,
            self.lora_config.lora_alpha,
            self.lora_config.lora_dropout,
        )

        # Replace in model
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_linear)
        self.lora_layers[name] = lora_linear

        logger.debug(f"Applied LoRA to {name}")

    def setup_optimizer(self) -> None:
        """Setup optimizer for LoRA parameters only."""
        # Get only trainable parameters (LoRA parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            logger.warning("No trainable parameters found")
            return

        logger.info(
            f"Setting up optimizer with {len(trainable_params)} trainable parameters"
        )

        optimizer_type = self.config.optimizer.type.value
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                eps=self.config.optimizer.epsilon,
            )
        elif optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.config.optimizer.learning_rate,
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                eps=self.config.optimizer.epsilon,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        logger.info(f"Optimizer setup complete: {self.optimizer.__class__.__name__}")

    def setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        if self.optimizer is None:
            raise RuntimeError(
                "Optimizer not initialized. Call setup_optimizer() first."
            )

        scheduler_type = self.config.scheduler.type.value

        if scheduler_type == "linear":
            from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.scheduler.num_warmup_steps,
                num_training_steps=self.config.scheduler.num_training_steps,
            )
        elif scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler.num_training_steps,
            )
        else:
            from torch.optim.lr_scheduler import LambdaLR

            self.scheduler = LambdaLR(self.optimizer, lambda x: 1)

        logger.info(f"Scheduler setup complete: {self.scheduler.__class__.__name__}")

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> dict:
        """Train the model with LoRA.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader

        Returns:
            Training results
        """
        self.state.start_time = __import__("datetime").datetime.now()
        logger.info("Starting LoRA training")

        results = {
            "training_loss": [],
            "eval_loss": [],
        }

        self.model.train()

        for epoch in range(self.config.training.num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0

            for step, batch in enumerate(train_dataloader):
                self.state.step += 1

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                except Exception as e:
                    logger.error(f"Error in forward pass: {e}")
                    continue

                # Backward pass
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

            # Evaluation
            if eval_dataloader:
                eval_metrics = self.evaluate(eval_dataloader)
                results["eval_loss"].append(eval_metrics.loss)
                logger.info(f"Eval Loss: {eval_metrics.loss:.4f}")

            # Save checkpoint
            self.save_checkpoint()

        self.state.end_time = __import__("datetime").datetime.now()
        logger.info("Training complete")

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

    def get_lora_weights(self) -> dict:
        """Get LoRA weights.

        Returns:
            Dictionary of LoRA weights
        """
        weights = {}
        for name, layer in self.lora_layers.items():
            weights[f"{name}_lora_a"] = layer.lora_layer.lora_a.weight
            weights[f"{name}_lora_b"] = layer.lora_layer.lora_b.weight
        return weights

    def merge_lora_weights(self) -> None:
        """Merge LoRA weights into original model weights.

        This reduces the model size by combining LoRA weights back into the original
        layers. Note: This is a one-way operation.
        """
        logger.info("Merging LoRA weights into base model")

        for name, layer in self.lora_layers.items():
            # Get original and LoRA weights
            original_weight = layer.original_layer.weight
            lora_a_weight = layer.lora_layer.lora_a.weight
            lora_b_weight = layer.lora_layer.lora_b.weight
            scaling = layer.lora_layer.scaling

            # Merge: W = W_original + scaling * (W_b @ W_a)
            lora_weight = (lora_b_weight @ lora_a_weight) * scaling
            merged_weight = original_weight + lora_weight

            # Update original layer
            with torch.no_grad():
                layer.original_layer.weight.copy_(merged_weight)

        logger.info("LoRA weights merged successfully")
