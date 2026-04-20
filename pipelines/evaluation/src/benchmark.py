"""Evaluation benchmark orchestration."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    model_path: str
    benchmarks: List[str]  # mmlu, alpacaeval, gsm8k, hellaswag
    batch_size: int = 32
    num_shots: int = 0
    max_samples: Optional[int] = None
    output_dir: str = "./eval_results"


class BenchmarkOrchestrator:
    """Orchestrates benchmark evaluation."""

    supported_benchmarks = ["mmlu", "alpacaeval", "gsm8k", "hellaswag"]

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark orchestrator.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results = {}
        self._validate_config()

    def _validate_config(self):
        """Validate benchmark configuration."""
        for benchmark in self.config.benchmarks:
            if benchmark not in self.supported_benchmarks:
                logger.warning(f"Unsupported benchmark: {benchmark}")

    def run_all(self) -> Dict[str, Any]:
        """Run all configured benchmarks.

        Returns:
            Dictionary with benchmark results
        """
        logger.info("=" * 80)
        logger.info("Starting Benchmark Evaluation")
        logger.info("=" * 80)

        for benchmark in self.config.benchmarks:
            logger.info(f"\nRunning {benchmark.upper()} benchmark...")
            try:
                result = self.run_benchmark(benchmark)
                self.results[benchmark] = result
                logger.info(f"{benchmark}: {result}")
            except Exception as e:
                logger.error(f"Failed to run {benchmark}: {e}")
                self.results[benchmark] = {"error": str(e)}

        logger.info("\n" + "=" * 80)
        logger.info("Benchmark Evaluation Complete")
        logger.info("=" * 80)

        return self.results

    def run_benchmark(self, benchmark: str) -> Dict[str, Any]:
        """Run specific benchmark.

        Args:
            benchmark: Benchmark name

        Returns:
            Benchmark results
        """
        if benchmark == "mmlu":
            return self._run_mmlu()
        elif benchmark == "alpacaeval":
            return self._run_alpacaeval()
        elif benchmark == "gsm8k":
            return self._run_gsm8k()
        elif benchmark == "hellaswag":
            return self._run_hellaswag()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer for inference.

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch required for model inference")

        logger.info(f"Loading model from {self.config.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        return model, tokenizer

    def _generate_answer(self, model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate answer from model.

        Args:
            model: The language model
            tokenizer: The tokenizer
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated answer text
        """
        import torch

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the generated text
        answer = generated_text[len(prompt):].strip()
        return answer

    def _extract_answer_mmlu(self, text: str) -> str:
        """Extract answer choice from MMLU output."""
        text = text.strip().upper()
        # Look for A, B, C, or D at the beginning
        for char in text:
            if char in ["A", "B", "C", "D"]:
                return char
        return ""

    def _extract_answer_gsm8k(self, text: str) -> str:
        """Extract numeric answer from GSM8K output."""
        import re
        # Look for numbers in the text
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if numbers:
            return numbers[-1]  # Take the last number (usually the answer)
        return ""

    def _run_mmlu(self) -> Dict[str, Any]:
        """Run MMLU benchmark.

        MMLU (Massive Multitask Language Understanding) tests knowledge across
        57 subjects including mathematics, history, law, etc.
        """
        logger.info("Loading MMLU dataset...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for MMLU benchmark")

        try:
            dataset = load_dataset("cais/mmlu", "all", split="test")
        except Exception:
            # Fallback to alternative naming
            try:
                dataset = load_dataset("lukaemon/mmlu", split="test")
            except Exception:
                logger.warning("Could not load MMLU dataset, using fallback")
                return {
                    "accuracy": 0.0,
                    "score": 0.0,
                    "num_samples": 0,
                    "model": self.config.model_path,
                    "error": "Could not load MMLU dataset",
                }

        model, tokenizer = self._load_model_and_tokenizer()

        correct = 0
        total = 0
        max_samples = self.config.max_samples or len(dataset)

        for i, example in enumerate(dataset):
            if i >= max_samples:
                break

            # Format MMLU prompt
            question = example.get("question", example.get("query", ""))
            choices = example.get("choices", [])
            if not choices and "A" in example:
                # Alternative format
                choices = [example.get("A", ""), example.get("B", ""), example.get("C", ""), example.get("D", "")]

            prompt = f"Question: {question}\n"
            for j, choice in enumerate(choices):
                prompt += f"{chr(65+j)}. {choice}\n"
            prompt += "Answer:"

            answer = self._generate_answer(model, tokenizer, prompt, max_new_tokens=10)
            predicted = self._extract_answer_mmlu(answer)
            actual = example.get("answer", example.get("label", "")).strip().upper()

            if predicted == actual:
                correct += 1
            total += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{max_samples} MMLU samples")

        accuracy = correct / total if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "score": accuracy * 100,
            "num_samples": total,
            "model": self.config.model_path,
        }

    def _run_alpacaeval(self) -> Dict[str, Any]:
        """Run AlpacaEval benchmark.

        AlpacaEval evaluates instruction-following capabilities by comparing
        model outputs against reference outputs.
        """
        logger.info("Running AlpacaEval...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for AlpacaEval benchmark")

        try:
            dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval")
        except Exception:
            logger.warning("Could not load AlpacaEval dataset, using fallback")
            return {
                "win_rate": 0.0,
                "score": 0.0,
                "num_samples": 0,
                "model": self.config.model_path,
                "error": "Could not load AlpacaEval dataset",
            }

        model, tokenizer = self._load_model_and_tokenizer()

        # For simplicity, we compute a proxy score based on generation quality
        # In practice, AlpacaEval uses GPT-4 as a judge
        total_length = 0
        total_samples = 0
        max_samples = self.config.max_samples or min(100, len(dataset))

        for i, example in enumerate(dataset):
            if i >= max_samples:
                break

            instruction = example.get("instruction", "")
            if not instruction:
                continue

            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

            answer = self._generate_answer(model, tokenizer, prompt, max_new_tokens=256)

            # Simple heuristic: longer, non-empty responses score better
            # In practice, this should use GPT-4 evaluation
            if len(answer.strip()) > 20:
                total_length += 1
            total_samples += 1

        # Proxy win rate based on response quality
        win_rate = total_length / total_samples if total_samples > 0 else 0.0
        return {
            "win_rate": win_rate,
            "score": win_rate * 100,
            "num_samples": total_samples,
            "model": self.config.model_path,
            "note": "Using proxy evaluation (full AlpacaEval requires GPT-4 judge)",
        }

    def _run_gsm8k(self) -> Dict[str, Any]:
        """Run GSM8K benchmark.

        GSM8K (Grade School Math 8K) tests mathematical reasoning on grade school
        level math word problems.
        """
        logger.info("Running GSM8K...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for GSM8K benchmark")

        try:
            dataset = load_dataset("gsm8k", "main", split="test")
        except Exception:
            logger.warning("Could not load GSM8K dataset, using fallback")
            return {
                "accuracy": 0.0,
                "score": 0.0,
                "num_samples": 0,
                "model": self.config.model_path,
                "error": "Could not load GSM8K dataset",
            }

        model, tokenizer = self._load_model_and_tokenizer()

        correct = 0
        total = 0
        max_samples = self.config.max_samples or len(dataset)

        for i, example in enumerate(dataset):
            if i >= max_samples:
                break

            question = example.get("question", "")
            answer_text = example.get("answer", "")

            # Extract ground truth (last number in answer)
            import re
            ground_truth_match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_text)
            if ground_truth_match:
                ground_truth = ground_truth_match.group(1).strip()
            else:
                # Try to extract last number
                numbers = re.findall(r"[-+]?\d*\.?\d+", answer_text)
                ground_truth = numbers[-1] if numbers else ""

            prompt = f"Question: {question}\nAnswer:"

            generated = self._generate_answer(model, tokenizer, prompt, max_new_tokens=256)
            predicted = self._extract_answer_gsm8k(generated)

            # Compare numerical values
            try:
                pred_val = float(predicted)
                true_val = float(ground_truth)
                if abs(pred_val - true_val) < 0.01:
                    correct += 1
            except (ValueError, TypeError):
                # String comparison fallback
                if predicted.strip() == ground_truth.strip():
                    correct += 1

            total += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{max_samples} GSM8K samples")

        accuracy = correct / total if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "score": accuracy * 100,
            "num_samples": total,
            "model": self.config.model_path,
        }

    def _run_hellaswag(self) -> Dict[str, Any]:
        """Run HellaSwag benchmark.

        HellaSwag tests commonsense reasoning by asking models to complete
        sentences with the most plausible ending.
        """
        logger.info("Running HellaSwag...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for HellaSwag benchmark")

        try:
            dataset = load_dataset("hellaswag", split="validation")
        except Exception:
            logger.warning("Could not load HellaSwag dataset, using fallback")
            return {
                "accuracy": 0.0,
                "score": 0.0,
                "num_samples": 0,
                "model": self.config.model_path,
                "error": "Could not load HellaSwag dataset",
            }

        model, tokenizer = self._load_model_and_tokenizer()

        correct = 0
        total = 0
        max_samples = self.config.max_samples or len(dataset)

        for i, example in enumerate(dataset):
            if i >= max_samples:
                break

            context = example.get("ctx", example.get("context", ""))
            endings = example.get("endings", example.get("choices", []))
            label = int(example.get("label", 0))

            # Score each ending using perplexity
            import torch
            import torch.nn.functional as F

            best_score = float('-inf')
            best_idx = 0

            for idx, ending in enumerate(endings):
                full_text = context + " " + ending
                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    # Negative loss as score (lower loss = higher score)
                    score = -outputs.loss.item()

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == label:
                correct += 1
            total += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{max_samples} HellaSwag samples")

        accuracy = correct / total if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "score": accuracy * 100,
            "num_samples": total,
            "model": self.config.model_path,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results.

        Returns:
            Summary statistics
        """
        scores = []
        for benchmark, result in self.results.items():
            if "error" not in result and "score" in result:
                scores.append(result["score"])

        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "model": self.config.model_path,
            "num_benchmarks": len(self.config.benchmarks),
            "completed_benchmarks": len(scores),
            "average_score": avg_score,
            "details": self.results,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = BenchmarkConfig(
        model_path="./checkpoints/lora", benchmarks=["mmlu", "alpacaeval", "gsm8k"]
    )

    orchestrator = BenchmarkOrchestrator(config)
    results = orchestrator.run_all()
    print(orchestrator.get_summary())
