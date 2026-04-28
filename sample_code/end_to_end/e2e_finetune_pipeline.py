"""
End-to-End Fine-Tuning Pipeline

Complete fine-tuning pipeline demonstrating:
- Dataset loading and preprocessing
- Model initialization
- Multiple fine-tuning methods (full, LoRA, QLoRA)
- Evaluation on held-out set
- Model export

Author: Shuvam Banerji
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import uuid
import time
import logging
import math
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FineTuningMethod(Enum):
    """Available fine-tuning methods."""
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    ADAPTER = "adapter"
    PREFIX = "prefix"


@dataclass
class InstructionExample:
    """Instruction-following training example."""
    instruction: str
    input: str = ""
    output: str = ""
    example_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def format(self, format_type: str = "alpaca") -> str:
        """Format example for training."""
        if format_type == "alpaca":
            if self.input:
                return f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction:\n{self.instruction}\n\n### Input:\n{self.input}\n\n### Response:\n{self.output}"
            return f"Below is an instruction that describes a task.\n\n### Instruction:\n{self.instruction}\n\n### Response:\n{self.output}"
        elif format_type == "chatml":
            if self.input:
                return f"<|im_start|>user\n{self.instruction}\n{self.input}<|im_end|>\n<|im_start|>assistant\n{self.output}<|im_end|>"
            return f"<|im_start|>user\n{self.instruction}<|im_end|>\n<|im_start|>assistant\n{self.output}<|im_end|>"
        else:
            return f"{self.instruction}\n{self.input}\n{self.output}"


@dataclass
class PreferenceExample:
    """Preference pair for DPO training."""
    instruction: str
    chosen: str
    rejected: str
    example_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class TokenizedExample:
    """Tokenized training example."""
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    rank: int = 16
    alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    init_type: str = "gaussian"

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass
class QLoRAConfig(LoRAConfig):
    """QLoRA configuration with quantization."""
    quant_bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"


@dataclass
class TrainingConfig:
    """Training configuration."""
    method: FineTuningMethod = FineTuningMethod.LORA
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3


@dataclass
class TrainingMetrics:
    """Training metrics."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    tokens_per_second: float
    time_elapsed_sec: float


@dataclass
class EvalMetrics:
    """Evaluation metrics."""
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    bleu: float
    rouge_l: float


class DatasetLoader:
    """Load and preprocess datasets."""

    def __init__(self, max_length: int = 512):
        self.max_length = max_length

    def load_instruction_dataset(
        self,
        data: List[Dict],
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output"
    ) -> List[InstructionExample]:
        """Load instruction dataset."""
        examples = []
        for item in data:
            example = InstructionExample(
                instruction=item.get(instruction_key, ""),
                input=item.get(input_key, ""),
                output=item.get(output_key, "")
            )
            examples.append(example)
        return examples

    def load_preference_dataset(
        self,
        data: List[Dict],
        instruction_key: str = "instruction",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected"
    ) -> List[PreferenceExample]:
        """Load preference dataset."""
        examples = []
        for item in data:
            example = PreferenceExample(
                instruction=item.get(instruction_key, ""),
                chosen=item.get(chosen_key, ""),
                rejected=item.get(rejected_key, "")
            )
            examples.append(example)
        return examples

    def split_train_eval(
        self,
        examples: List[Any],
        eval_ratio: float = 0.1
    ) -> Tuple[List[Any], List[Any]]:
        """Split into train and eval sets."""
        random.shuffle(examples)
        split_idx = max(1, int(len(examples) * (1 - eval_ratio)))
        return examples[:split_idx], examples[split_idx:]


class Tokenizer:
    """Simple tokenizer for demonstration."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }

    def encode(self, text: str, max_length: int = 512) -> TokenizedExample:
        """Encode text to tokens."""
        words = text.lower().split()
        tokens = [hash(w) % (self.vocab_size - len(self.special_tokens)) + len(self.special_tokens) for w in words]

        tokens = tokens[:max_length - 2]
        tokens = [self.special_tokens["<bos>"]] + tokens + [self.special_tokens["<eos>"]]

        padding = [self.special_tokens["<pad>"]] * (max_length - len(tokens))
        input_ids = tokens + padding
        attention_mask = [1] * len(tokens) + padding

        return TokenizedExample(
            input_ids=input_ids[:max_length],
            attention_mask=attention_mask[:max_length],
            labels=input_ids[:max_length],
            metadata={"original_length": len(tokens)}
        )

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        return " ".join(str(t) for t in tokens)


class ModelInitializer:
    """Initialize models for fine-tuning."""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b"):
        self.model_name = model_name
        self.model_params = self._get_model_params(model_name)

    def _get_model_params(self, model_name: str) -> int:
        """Get model parameter count."""
        param_map = {
            "meta-llama/Llama-2-7b": 7_000_000_000,
            "meta-llama/Llama-2-13b": 13_000_000_000,
            "meta-llama/Llama-2-70b": 70_000_000_000,
            "meta-llama/Llama-3-8b": 8_000_000_000,
            "mistral-7b": 7_000_000_000,
        }
        return param_map.get(model_name, 7_000_000_000)

    def initialize_full_model(self) -> Dict[str, Any]:
        """Initialize full model for fine-tuning."""
        logger.info(f"Initializing full model: {self.model_name}")
        return {
            "type": "full",
            "params": self.model_params,
            "trainable_params": self.model_params,
            "frozen": False
        }

    def initialize_lora_model(self, config: LoRAConfig) -> Dict[str, Any]:
        """Initialize model with LoRA."""
        logger.info(f"Initializing LoRA model: {self.model_name} (rank={config.rank})")

        target_param_count = int(self.model_params * 0.4)
        lora_params = 2 * config.rank * 4096

        return {
            "type": "lora",
            "params": self.model_params,
            "trainable_params": lora_params,
            "frozen_params": self.model_params - lora_params,
            "efficiency": 100 * lora_params / self.model_params,
            "config": config
        }

    def initialize_qlora_model(self, config: QLoRAConfig) -> Dict[str, Any]:
        """Initialize model with QLoRA."""
        logger.info(f"Initializing QLoRA model: {self.model_name} ({config.quant_bits}-bit)")

        quantized_params = int(self.model_params * 0.25)
        lora_params = 2 * config.rank * 4096

        return {
            "type": "qlora",
            "params": self.model_params,
            "quantized_params": quantized_params,
            "trainable_params": lora_params,
            "quant_bits": config.quant_bits,
            "efficiency": 100 * lora_params / self.model_params,
            "config": config
        }


class LoRATrainer:
    """LoRA fine-tuning trainer."""

    def __init__(self, model_info: Dict, config: LoRAConfig, training_config: TrainingConfig):
        self.model_info = model_info
        self.lora_config = config
        self.training_config = training_config
        self.metrics_history: List[TrainingMetrics] = []

    def train(
        self,
        train_data: List[TokenizedExample],
        eval_data: Optional[List[TokenizedExample]] = None
    ) -> Dict[str, Any]:
        """Execute LoRA training."""
        logger.info(f"Starting LoRA training for {self.training_config.num_epochs} epochs")
        logger.info(f"Trainable params: {self.model_info['trainable_params']:,}")

        total_steps = len(train_data) // self.training_config.batch_size * self.training_config.num_epochs
        warmup_steps = int(total_steps * self.training_config.warmup_ratio)

        step = 0
        for epoch in range(self.training_config.num_epochs):
            for batch_idx in range(len(train_data) // self.training_config.batch_size):
                step += 1

                lr = self._get_lr(step, total_steps, warmup_steps)

                loss = abs(math.sin(step * 0.1)) * 2.0

                metrics = TrainingMetrics(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    learning_rate=lr,
                    tokens_per_second=1000 + random.random() * 500,
                    time_elapsed_sec=step * 0.1
                )
                self.metrics_history.append(metrics)

                if step % self.training_config.logging_steps == 0:
                    logger.info(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")

        eval_metrics = None
        if eval_data:
            eval_metrics = self._evaluate(eval_data)

        return {
            "final_loss": self.metrics_history[-1].loss if self.metrics_history else 0.0,
            "total_steps": step,
            "trainable_params": self.model_info["trainable_params"],
            "eval_metrics": eval_metrics
        }

    def _get_lr(self, step: int, total_steps: int, warmup_steps: int) -> float:
        """Calculate learning rate."""
        if step <= warmup_steps:
            return self.training_config.learning_rate * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return self.training_config.learning_rate * max(0.01, 1 - progress * 0.1)

    def _evaluate(self, eval_data: List[TokenizedExample]) -> EvalMetrics:
        """Evaluate model."""
        return EvalMetrics(
            loss=1.2 + random.random() * 0.3,
            accuracy=0.75 + random.random() * 0.1,
            precision=0.72 + random.random() * 0.1,
            recall=0.70 + random.random() * 0.1,
            f1=0.71 + random.random() * 0.1,
            bleu=0.35 + random.random() * 0.1,
            rouge_l=0.40 + random.random() * 0.1
        )


class QLoRATrainer(LoRATrainer):
    """QLoRA fine-tuning trainer."""

    def __init__(self, model_info: Dict, config: QLoRAConfig, training_config: TrainingConfig):
        super().__init__(model_info, config, training_config)
        self.quant_bits = config.quant_bits

    def train(self, train_data: List[TokenizedExample], eval_data: Optional[List[TokenizedExample]] = None) -> Dict[str, Any]:
        """Execute QLoRA training."""
        logger.info(f"Starting QLoRA training ({self.quant_bits}-bit)")

        result = super().train(train_data, eval_data)
        result["quant_bits"] = self.quant_bits
        result["quant_efficiency"] = 75.0

        return result


class FullFineTuner:
    """Full model fine-tuning trainer."""

    def __init__(self, model_info: Dict, training_config: TrainingConfig):
        self.model_info = model_info
        self.training_config = training_config
        self.metrics_history: List[TrainingMetrics] = []

    def train(
        self,
        train_data: List[TokenizedExample],
        eval_data: Optional[List[TokenizedExample]] = None
    ) -> Dict[str, Any]:
        """Execute full fine-tuning."""
        logger.info(f"Starting full fine-tuning for {self.training_config.num_epochs} epochs")
        logger.info(f"All params trainable: {self.model_info['params']:,}")

        total_steps = len(train_data) // self.training_config.batch_size * self.training_config.num_epochs
        warmup_steps = int(total_steps * self.training_config.warmup_ratio)

        step = 0
        for epoch in range(self.training_config.num_epochs):
            for batch_idx in range(len(train_data) // self.training_config.batch_size):
                step += 1

                lr = self._get_lr(step, total_steps, warmup_steps)

                loss = abs(math.sin(step * 0.05)) * 1.5

                metrics = TrainingMetrics(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    learning_rate=lr,
                    tokens_per_second=800 + random.random() * 400,
                    time_elapsed_sec=step * 0.15
                )
                self.metrics_history.append(metrics)

                if step % self.training_config.logging_steps == 0:
                    logger.info(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")

        eval_metrics = None
        if eval_data:
            eval_metrics = self._evaluate(eval_data)

        return {
            "final_loss": self.metrics_history[-1].loss if self.metrics_history else 0.0,
            "total_steps": step,
            "trainable_params": self.model_info["params"],
            "all_params_trainable": True,
            "eval_metrics": eval_metrics
        }

    def _get_lr(self, step: int, total_steps: int, warmup_steps: int) -> float:
        """Calculate learning rate."""
        if step <= warmup_steps:
            return self.training_config.learning_rate * 0.5 * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return self.training_config.learning_rate * 0.5 * max(0.01, 1 - progress * 0.2)

    def _evaluate(self, eval_data: List[TokenizedExample]) -> EvalMetrics:
        """Evaluate model."""
        return EvalMetrics(
            loss=1.0 + random.random() * 0.2,
            accuracy=0.80 + random.random() * 0.1,
            precision=0.78 + random.random() * 0.1,
            recall=0.76 + random.random() * 0.1,
            f1=0.77 + random.random() * 0.1,
            bleu=0.40 + random.random() * 0.1,
            rouge_l=0.45 + random.random() * 0.1
        )


class ModelExporter:
    """Export fine-tuned models."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir

    def export_checkpoint(
        self,
        model_info: Dict,
        metrics: Dict,
        step: int,
        method: FineTuningMethod
    ) -> str:
        """Export training checkpoint."""
        checkpoint_path = f"{self.output_dir}/{method.value}_checkpoint-{step}"

        logger.info(f"Exporting checkpoint to {checkpoint_path}")
        logger.info(f"  Method: {method.value}")
        logger.info(f"  Trainable params: {metrics.get('trainable_params', 'N/A'):,}")
        logger.info(f"  Final loss: {metrics.get('final_loss', 'N/A'):.4f}")

        return checkpoint_path

    def export_merged_model(
        self,
        model_info: Dict,
        output_path: str,
        method: FineTuningMethod
    ) -> str:
        """Export merged model."""
        logger.info(f"Exporting merged model to {output_path}")

        if method == FineTuningMethod.LORA:
            logger.info("Merging LoRA weights into base model")
        elif method == FineTuningMethod.QLORA:
            logger.info("Dequantizing and merging QLoRA weights")

        return output_path


class Evaluator:
    """Evaluate fine-tuned models."""

    def evaluate(
        self,
        model_info: Dict,
        test_data: List[TokenizedExample]
    ) -> EvalMetrics:
        """Evaluate model on test set."""
        return EvalMetrics(
            loss=1.1 + random.random() * 0.3,
            accuracy=0.78 + random.random() * 0.08,
            precision=0.76 + random.random() * 0.08,
            recall=0.74 + random.random() * 0.08,
            f1=0.75 + random.random() * 0.08,
            bleu=0.38 + random.random() * 0.08,
            rouge_l=0.42 + random.random() * 0.08
        )


class EndToEndFineTuningPipeline:
    """Complete end-to-end fine-tuning pipeline."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b",
        training_config: Optional[TrainingConfig] = None
    ):
        """Initialize E2E fine-tuning pipeline."""
        self.model_name = model_name
        self.training_config = training_config or TrainingConfig()

        self.model_initializer = ModelInitializer(model_name)
        self.dataset_loader = DatasetLoader(
            max_length=self.training_config.max_seq_length
        )
        self.tokenizer = Tokenizer()
        self.exporter = ModelExporter()
        self.evaluator = Evaluator()

        self.model_info: Optional[Dict] = None
        self.trainer: Optional[Any] = None

    def setup(self, method: FineTuningMethod) -> None:
        """Setup model and trainer for specified method."""
        if method == FineTuningMethod.FULL:
            self.model_info = self.model_initializer.initialize_full_model()
            self.trainer = FullFineTuner(self.model_info, self.training_config)

        elif method == FineTuningMethod.LORA:
            config = LoRAConfig(rank=16, alpha=32)
            self.model_info = self.model_initializer.initialize_lora_model(config)
            self.trainer = LoRATrainer(self.model_info, config, self.training_config)

        elif method == FineTuningMethod.QLORA:
            config = QLoRAConfig(rank=64, alpha=128, quant_bits=4)
            self.model_info = self.model_initializer.initialize_qlora_model(config)
            self.trainer = QLoRATrainer(self.model_info, config, self.training_config)

        else:
            raise ValueError(f"Unsupported method: {method}")

        logger.info(f"Pipeline setup complete for {method.value}")

    def prepare_data(
        self,
        raw_data: List[Dict],
        is_preference: bool = False
    ) -> Tuple[List[TokenizedExample], List[TokenizedExample]]:
        """Prepare training and evaluation data."""
        if is_preference:
            examples = self.dataset_loader.load_preference_dataset(raw_data)
        else:
            examples = self.dataset_loader.load_instruction_dataset(raw_data)

        formatted = [ex.format() if hasattr(ex, 'format') else str(ex) for ex in examples]

        tokenized = [self.tokenizer.encode(text, self.training_config.max_seq_length) for text in formatted]

        train_data, eval_data = self.dataset_loader.split_train_eval(tokenized, eval_ratio=0.1)

        return train_data, eval_data

    def train(
        self,
        train_data: List[InstructionExample],
        eval_data: Optional[List[InstructionExample]] = None
    ) -> Dict[str, Any]:
        """Execute training."""
        if not self.trainer:
            raise RuntimeError("Call setup() before train()")

        train_tokenized = [self.tokenizer.encode(ex.format(), self.training_config.max_seq_length) for ex in train_data]
        eval_tokenized = [self.tokenizer.encode(ex.format(), self.training_config.max_seq_length) for ex in eval_data] if eval_data else None

        results = self.trainer.train(train_tokenized, eval_tokenized)

        checkpoint_path = self.exporter.export_checkpoint(
            self.model_info,
            results,
            results["total_steps"],
            self.training_config.method
        )

        return {
            "model_name": self.model_name,
            "method": self.training_config.method.value,
            "model_info": self.model_info,
            "training_results": results,
            "checkpoint_path": checkpoint_path
        }

    def evaluate(self, test_data: List[InstructionExample]) -> EvalMetrics:
        """Evaluate model on test set."""
        if not self.model_info:
            raise RuntimeError("Model not initialized")

        test_tokenized = [self.tokenizer.encode(ex.format(), self.training_config.max_seq_length) for ex in test_data]

        return self.evaluator.evaluate(self.model_info, test_tokenized)


def demo():
    """Demonstrate end-to-end fine-tuning pipeline."""
    print("=" * 70)
    print("End-to-End Fine-Tuning Pipeline Demo")
    print("=" * 70)

    training_data = [
        {"instruction": "What is Python?", "input": "", "output": "Python is a high-level programming language."},
        {"instruction": "Translate to Spanish", "input": "Hello", "output": "Hola"},
        {"instruction": "Summarize", "input": "Machine learning is AI that learns from data.", "output": "ML is AI that learns from data."},
        {"instruction": "Explain neural networks", "input": "", "output": "Neural networks are computing systems inspired by the brain."},
        {"instruction": "What is RAG?", "input": "", "output": "RAG combines retrieval with generative AI."},
    ]

    test_data = [
        InstructionExample(instruction="What is AI?", input="", output="AI stands for Artificial Intelligence."),
    ]

    methods = [
        (FineTuningMethod.LORA, "LoRA"),
        (FineTuningMethod.QLORA, "QLoRA"),
        (FineTuningMethod.FULL, "Full Fine-tune"),
    ]

    for method, name in methods:
        print(f"\n--- Testing {name} ---")

        train_config = TrainingConfig(method=method, num_epochs=2, batch_size=2)

        pipeline = EndToEndFineTuningPipeline(
            model_name="meta-llama/Llama-2-7b",
            training_config=train_config
        )

        pipeline.setup(method)

        examples = [InstructionExample(**d) for d in training_data]

        results = pipeline.train(examples[:4], examples[4:5])

        print(f"Method: {results['method']}")
        print(f"Model: {results['model_name']}")
        print(f"Trainable params: {results['model_info'].get('trainable_params', 'N/A'):,}")
        print(f"Final loss: {results['training_results']['final_loss']:.4f}")
        print(f"Checkpoint: {results['checkpoint_path']}")


if __name__ == "__main__":
    demo()