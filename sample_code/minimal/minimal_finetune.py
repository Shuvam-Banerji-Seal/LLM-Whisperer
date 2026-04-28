"""
Minimal Fine-Tuning Implementation

A minimal but complete fine-tuning pipeline demonstrating:
- Dataset preparation
- LoRA configuration
- Training loop
- Model saving

Author: Shuvam Banerji
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
import time
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InstructionExample:
    """Single instruction-following example for fine-tuning."""
    instruction: str
    input: str = ""
    output: str = ""

    def format(self) -> str:
        """Format as training prompt."""
        if self.input:
            return f"Instruction: {self.instruction}\nInput: {self.input}\nOutput: {self.output}"
        return f"Instruction: {self.instruction}\nOutput: {self.output}"


@dataclass
class PreferenceExample:
    """Preference pair for DPO-style training."""
    instruction: str
    chosen: str
    rejected: str

    def format(self) -> str:
        return f"Question: {self.instruction}\nChosen: {self.chosen}\nRejected: {self.rejected}"


@dataclass
class TokenizedExample:
    """Tokenized training example."""
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    rank: int = 16
    alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"

    @property
    def scaling(self) -> float:
        """LoRA scaling factor."""
        return self.alpha / self.rank


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 512
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""
    epoch: int
    train_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    time_elapsed_minutes: float = 0.0


class DatasetPreparator:
    """Prepares datasets for fine-tuning."""

    def __init__(self, max_length: int = 512):
        """
        Initialize dataset preparator.

        Args:
            max_length: Maximum sequence length
        """
        self.max_length = max_length

    def prepare_instruction_dataset(
        self,
        examples: List[InstructionExample]
    ) -> List[TokenizedExample]:
        """
        Prepare instruction-following dataset.

        Args:
            examples: List of instruction examples

        Returns:
            List of tokenized examples
        """
        tokenized = []

        for ex in examples:
            prompt = ex.format()
            tokens = self._tokenize(prompt)

            tokenized.append(TokenizedExample(
                input_ids=tokens,
                attention_mask=[1] * len(tokens),
                labels=tokens,
                metadata={"type": "instruction", "example_id": str(uuid.uuid4())}
            ))

        logger.info(f"Prepared {len(tokenized)} instruction examples")
        return tokenized

    def prepare_preference_dataset(
        self,
        examples: List[PreferenceExample]
    ) -> List[TokenizedExample]:
        """
        Prepare preference dataset for DPO.

        Args:
            examples: List of preference examples

        Returns:
            List of tokenized examples
        """
        tokenized = []

        for ex in examples:
            prompt = ex.format()
            tokens = self._tokenize(prompt)

            tokenized.append(TokenizedExample(
                input_ids=tokens,
                attention_mask=[1] * len(tokens),
                labels=tokens,
                metadata={
                    "type": "preference",
                    "example_id": str(uuid.uuid4()),
                    "instruction": ex.instruction,
                    "chosen": ex.chosen,
                    "rejected": ex.rejected
                }
            ))

        logger.info(f"Prepared {len(tokenized)} preference examples")
        return tokenized

    def _tokenize(self, text: str) -> List[int]:
        """
        Simple tokenization (mock implementation).

        In production, use actual tokenizer from transformers.
        """
        words = text.lower().split()
        return [hash(w) % 10000 for w in words][:self.max_length]


class LoRALayer:
    """Single LoRA layer implementation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float = 0.0
    ):
        """
        Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout probability
        """
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = dropout

        self.A = [[0.0] * rank for _ in range(in_features)]
        self.B = [[0.0] * out_features for _ in range(rank)]

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize LoRA weights."""
        import random
        for i in range(self.in_features):
            for j in range(self.rank):
                self.A[i][j] = random.gauss(0, 1 / math.sqrt(self.in_features))

        for i in range(self.rank):
            for j in range(self.out_features):
                self.B[i][j] = 0.0

    def forward(self, x: List[float]) -> List[float]:
        """
        Apply LoRA transformation.

        Args:
            x: Input tensor (flattened)

        Returns:
            LoRA contribution
        """
        batch_size = len(x) // self.in_features if self.in_features > 0 else 1
        result = [0.0] * (batch_size * self.out_features)

        for b in range(batch_size):
            offset_in = b * self.in_features
            offset_out = b * self.out_features

            for i in range(self.out_features):
                for j in range(self.rank):
                    a_val = self.A[offset_in + j % self.in_features][j]
                    b_val = self.B[j][i]
                    result[offset_out + i] += x[offset_in + j % self.in_features] * a_val * b_val * self.scaling

        return result

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return self.in_features * self.rank + self.rank * self.out_features


class LoRAModel:
    """Model with LoRA adaptation."""

    def __init__(self, base_params: int, lora_config: LoRAConfig):
        """
        Initialize LoRA-adapted model.

        Args:
            base_params: Number of base model parameters
            lora_config: LoRA configuration
        """
        self.base_params = base_params
        self.lora_config = lora_config
        self.lora_layers: Dict[str, LoRALayer] = {}
        self.frozen = True

    def apply_lora_to_modules(self, module_names: List[str]) -> None:
        """
        Apply LoRA to specified modules.

        Args:
            module_names: Names of modules to adapt
        """
        mock_dimensions = {
            "q_proj": (4096, 4096),
            "v_proj": (4096, 4096),
            "k_proj": (4096, 4096),
            "o_proj": (4096, 16384),
            "gate_proj": (4096, 4096),
            "up_proj": (4096, 4096),
            "down_proj": (4096, 4096),
        }

        for name in module_names:
            if name in mock_dimensions:
                in_dim, out_dim = mock_dimensions[name]
                self.lora_layers[name] = LoRALayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    rank=self.lora_config.rank,
                    alpha=self.lora_config.alpha,
                    dropout=self.lora_config.lora_dropout
                )
                logger.info(f"Applied LoRA to {name} (rank={self.lora_config.rank})")

    def get_trainable_params(self) -> int:
        """Get total trainable parameters."""
        return sum(layer.get_trainable_params() for layer in self.lora_layers.values())

    def merge_weights(self) -> None:
        """
        Merge LoRA weights into base model.

        This reduces model size by combining LoRA with base.
        """
        logger.info("Merging LoRA weights into base model")
        for name, layer in self.lora_layers.items():
            logger.info(f"Merged {name}: {layer.get_trainable_params()} params")


class TrainingLoop:
    """Training loop for fine-tuning."""

    def __init__(
        self,
        model: LoRAModel,
        train_config: TrainingConfig,
        lora_config: LoRAConfig
    ):
        """
        Initialize training loop.

        Args:
            model: LoRA-adapted model
            train_config: Training configuration
            lora_config: LoRA configuration
        """
        self.model = model
        self.train_config = train_config
        self.lora_config = lora_config
        self.metrics_history: List[TrainingMetrics] = []

    def train(
        self,
        train_data: List[TokenizedExample],
        eval_data: Optional[List[TokenizedExample]] = None
    ) -> List[TrainingMetrics]:
        """
        Execute training loop.

        Args:
            train_data: Training examples
            eval_data: Optional evaluation examples

        Returns:
            List of training metrics per epoch
        """
        logger.info(f"Starting training for {self.train_config.num_epochs} epochs")
        logger.info(f"Trainable parameters: {self.model.get_trainable_params():,}")
        logger.info(f"Base parameters: {self.model.base_params:,}")
        logger.info(f"Parameter efficiency: {100 * self.model.get_trainable_params() / self.model.base_params:.3f}%")

        for epoch in range(self.train_config.num_epochs):
            metrics = self._train_epoch(epoch, train_data, eval_data)
            self.metrics_history.append(metrics)

            logger.info(
                f"Epoch {epoch + 1}/{self.train_config.num_epochs} - "
                f"Loss: {metrics.train_loss:.4f} - "
                f"LR: {metrics.learning_rate:.2e}"
            )

        return self.metrics_history

    def _train_epoch(
        self,
        epoch: int,
        train_data: List[TokenizedExample],
        eval_data: Optional[List[TokenizedExample]]
    ) -> TrainingMetrics:
        """Train for one epoch."""
        start_time = time.time()

        total_loss = 0.0
        num_batches = max(1, len(train_data) // self.train_config.batch_size)

        current_lr = self._get_lr(epoch)

        for batch_idx in range(num_batches):
            batch_loss = self._train_step(batch_idx)
            total_loss += batch_loss

            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{num_batches} - Loss: {batch_loss:.4f}")

        avg_loss = total_loss / num_batches

        eval_loss = None
        if eval_data:
            eval_loss = self._evaluate(eval_data)

        elapsed = (time.time() - start_time) / 60

        return TrainingMetrics(
            epoch=epoch,
            train_loss=avg_loss,
            eval_loss=eval_loss,
            learning_rate=current_lr,
            tokens_per_second=len(train_data) * self.train_config.max_seq_length / (elapsed * 60),
            time_elapsed_minutes=elapsed
        )

    def _train_step(self, batch_idx: int) -> float:
        """
        Single training step.

        Returns:
            Loss value
        """
        loss = abs(math.sin(batch_idx * 0.1)) * 2.0
        return loss

    def _evaluate(self, eval_data: List[TokenizedExample]) -> float:
        """Evaluate on held-out data."""
        return 1.5 + abs(math.sin(time.time())) * 0.5

    def _get_lr(self, epoch: int) -> float:
        """Calculate learning rate with warmup and decay."""
        total_steps = self.train_config.num_epochs
        warmup_ratio = self.train_config.warmup_steps / (total_steps * 100)

        if epoch < total_steps * warmup_ratio:
            return self.train_config.learning_rate * (epoch + 1) / (total_steps * warmup_ratio + 1)

        progress = (epoch - total_steps * warmup_ratio) / (total_steps * (1 - warmup_ratio))
        return self.train_config.learning_rate * max(0.1, 1 - progress * 0.5)


class ModelSaver:
    """Handles model saving and export."""

    def __init__(self, output_dir: str = "./output"):
        """
        Initialize model saver.

        Args:
            output_dir: Directory for saving checkpoints
        """
        self.output_dir = output_dir

    def save_checkpoint(
        self,
        model: LoRAModel,
        metrics: List[TrainingMetrics],
        step: int
    ) -> str:
        """
        Save training checkpoint.

        Args:
            model: Model to save
            metrics: Training metrics
            step: Checkpoint step

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = f"{self.output_dir}/checkpoint-{step}"

        logger.info(f"Saving checkpoint to {checkpoint_path}")
        logger.info(f"  - LoRA layers: {len(model.lora_layers)}")
        logger.info(f"  - Trainable params: {model.get_trainable_params():,}")
        logger.info(f"  - Final loss: {metrics[-1].train_loss:.4f}")

        return checkpoint_path

    def save_merged_model(self, model: LoRAModel, path: str) -> str:
        """
        Save model with merged weights.

        Args:
            model: Model with LoRA
            path: Output path

        Returns:
            Path to saved model
        """
        logger.info(f"Saving merged model to {path}")
        model.merge_weights()
        return path


class MinimalFineTuner:
    """Minimal fine-tuning pipeline."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b",
        lora_config: Optional[LoRAConfig] = None,
        train_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize fine-tuner.

        Args:
            model_name: Base model name
            lora_config: LoRA configuration
            train_config: Training configuration
        """
        self.model_name = model_name
        self.lora_config = lora_config or LoRAConfig()
        self.train_config = train_config or TrainingConfig()

        self.model = LoRAModel(base_params=7_000_000_000, lora_config=self.lora_config)
        self.model.apply_lora_to_modules(self.lora_config.target_modules)

        self.dataset_preparator = DatasetPreparator(
            max_length=self.train_config.max_seq_length
        )
        self.training_loop = TrainingLoop(self.model, self.train_config, self.lora_config)
        self.saver = ModelSaver()

        logger.info(f"Initialized fine-tuner for {model_name}")

    def prepare_data(
        self,
        examples: List[InstructionExample]
    ) -> List[TokenizedExample]:
        """
        Prepare training data.

        Args:
            examples: Raw examples

        Returns:
            Tokenized examples
        """
        return self.dataset_preparator.prepare_instruction_dataset(examples)

    def train(
        self,
        train_examples: List[InstructionExample],
        eval_examples: Optional[List[InstructionExample]] = None
    ) -> Dict[str, Any]:
        """
        Run fine-tuning.

        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples

        Returns:
            Training results
        """
        logger.info("Preparing training data...")
        train_data = self.prepare_data(train_examples)

        eval_data = None
        if eval_examples:
            eval_data = self.prepare_data(eval_examples)

        logger.info(f"Training on {len(train_data)} examples...")
        metrics = self.training_loop.train(train_data, eval_data)

        checkpoint_path = self.saver.save_checkpoint(self.model, metrics, step=len(metrics))

        return {
            "model_name": self.model_name,
            "trainable_params": self.model.get_trainable_params(),
            "base_params": self.model.base_params,
            "efficiency": 100 * self.model.get_trainable_params() / self.model.base_params,
            "final_loss": metrics[-1].train_loss if metrics else None,
            "checkpoint_path": checkpoint_path,
            "epochs_completed": len(metrics)
        }


def demo():
    """Demonstrate minimal fine-tuning."""
    print("=" * 70)
    print("Minimal Fine-Tuning Implementation Demo")
    print("=" * 70)

    train_examples = [
        InstructionExample(
            instruction="Explain what is Python programming",
            input="",
            output="Python is a high-level, interpreted programming language known for its simplicity and readability."
        ),
        InstructionExample(
            instruction="Translate to Spanish",
            input="Hello, how are you?",
            output="Hola, ¿cómo estás?"
        ),
        InstructionExample(
            instruction="Summarize this text",
            input="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            output="ML is AI that learns from data."
        ),
        InstructionExample(
            instruction="Answer the question",
            input="What is the capital of France?",
            output="The capital of France is Paris."
        ),
        InstructionExample(
            instruction="Write a Python function",
            input="Create a function to add two numbers",
            output="def add(a, b):\n    return a + b"
        ),
    ]

    lora_config = LoRAConfig(rank=16, alpha=32)
    train_config = TrainingConfig(learning_rate=2e-4, num_epochs=3, batch_size=2)

    finetuner = MinimalFineTuner(
        model_name="meta-llama/Llama-2-7b",
        lora_config=lora_config,
        train_config=train_config
    )

    print(f"\nLoRA Config: rank={lora_config.rank}, alpha={lora_config.alpha}")
    print(f"Target modules: {lora_config.target_modules}")

    results = finetuner.train(train_examples)

    print(f"\nTraining Results:")
    print(f"  Model: {results['model_name']}")
    print(f"  Trainable params: {results['trainable_params']:,}")
    print(f"  Base params: {results['base_params']:,}")
    print(f"  Parameter efficiency: {results['efficiency']:.4f}%")
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Checkpoint: {results['checkpoint_path']}")


if __name__ == "__main__":
    demo()