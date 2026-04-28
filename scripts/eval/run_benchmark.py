#!/usr/bin/env python3
"""
Run standard benchmarks (MMLU, GSM8K, etc.).

This script runs standard LLM benchmarks for evaluating
model performance on various tasks including:
- MMLU (Massive Multitask Language Understanding)
- GSM8K (Grade School Math 8K)
- HumanEval (Code Generation)
- MATH (Mathematical Problem Solving)

Usage:
    python run_benchmark.py --model-id meta-llama/Llama-2-7b --benchmark mmlu --output-dir ./results
    python run_benchmark.py --model-id gpt2 --benchmark gsm8k --output-dir ./results
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    model_id: str
    benchmark: str
    output_dir: str
    num_samples: Optional[int] = None
    batch_size: int = 8
    max_length: int = 2048
    temperature: float = 0.0
    device: str = "auto"
    use_auth_token: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    benchmark: str
    num_samples: int
    num_correct: int
    accuracy: float
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    total_time_seconds: float
    samples_per_second: float
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def validate_benchmark(benchmark: str) -> bool:
    """Validate benchmark name.

    Args:
        benchmark: Benchmark name

    Returns:
        True if valid, False otherwise
    """
    valid_benchmarks = ["mmlu", "gsm8k", "humaneval", "math", "hellaswag", "truthfulqa"]
    return benchmark.lower() in valid_benchmarks


def load_model(model_id: str, device: str, use_auth_token: Optional[str] = None) -> Any:
    """Load model for inference.

    Args:
        model_id: Model identifier
        device: Device to load on
        use_auth_token: Optional auth token

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_id}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Install with: pip install transformers")
        raise

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=use_auth_token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if device != "cpu":
        model_kwargs["device_map"] = device

    if use_auth_token:
        model_kwargs["use_auth_token"] = use_auth_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    return model, tokenizer


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_length: int,
    temperature: float,
) -> str:
    """Generate response from model.

    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    try:
        import torch
    except ImportError:
        logger.error("torch not installed")
        raise

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()

    return response


def load_mmlu_dataset(num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load MMLU dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of MMLU examples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Install with: pip install datasets")
        raise

    dataset = load_dataset("cais/mmlu", "all", split="test")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return [
        {
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
            "subject": item.get("subject", "unknown"),
        }
        for item in dataset
    ]


def load_gsm8k_dataset(num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load GSM8K dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of GSM8K examples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed")
        raise

    dataset = load_dataset("openai/gsm8k", "main", split="test")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return [
        {
            "question": item["question"],
            "answer": item["answer"],
        }
        for item in dataset
    ]


def load_humaneval_dataset(num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load HumanEval dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of HumanEval examples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed")
        raise

    dataset = load_dataset("openai/openai_humaneval", split="test")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return [
        {
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
        }
        for item in dataset
    ]


def run_mmlu_benchmark(
    model: Any,
    tokenizer: Any,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Run MMLU benchmark.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        config: Benchmark configuration

    Returns:
        Benchmark result
    """
    logger.info("Running MMLU benchmark")

    dataset = load_mmlu_dataset(config.num_samples)

    correct = 0
    errors = []
    latencies = []

    for idx, item in enumerate(dataset):
        start_time = time.time()

        try:
            question = item["question"]
            choices = item["choices"]
            choices_text = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
            prompt = f"{question}\n{choices_text}\nAnswer:"

            response = generate_response(
                model, tokenizer, prompt,
                max_length=config.max_length,
                temperature=config.temperature,
            )

            predicted = extract_answer(response)

            if predicted == item["answer"]:
                correct += 1

        except Exception as e:
            errors.append(f"Sample {idx}: {str(e)}")

        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(dataset)} samples")

    return create_benchmark_result("mmlu", correct, len(dataset), latencies, errors)


def run_gsm8k_benchmark(
    model: Any,
    tokenizer: Any,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Run GSM8K benchmark.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        config: Benchmark configuration

    Returns:
        Benchmark result
    """
    logger.info("Running GSM8K benchmark")

    dataset = load_gsm8k_dataset(config.num_samples)

    correct = 0
    errors = []
    latencies = []

    for idx, item in enumerate(dataset):
        start_time = time.time()

        try:
            prompt = f"{item['question']}\nSolve step by step and give the final answer."

            response = generate_response(
                model, tokenizer, prompt,
                max_length=config.max_length,
                temperature=config.temperature,
            )

            answer_str = item["answer"].split("####")[-1].strip()
            predicted_answer = extract_gsm8k_answer(response)

            if answer_matches(answer_str, predicted_answer):
                correct += 1

        except Exception as e:
            errors.append(f"Sample {idx}: {str(e)}")

        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(dataset)} samples")

    return create_benchmark_result("gsm8k", correct, len(dataset), latencies, errors)


def run_humaneval_benchmark(
    model: Any,
    tokenizer: Any,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Run HumanEval benchmark.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        config: Benchmark configuration

    Returns:
        Benchmark result
    """
    logger.info("Running HumanEval benchmark")

    dataset = load_humaneval_dataset(config.num_samples)

    correct = 0
    errors = []
    latencies = []

    for idx, item in enumerate(dataset):
        start_time = time.time()

        try:
            prompt = item["prompt"] + "\n\nComplete the function:"

            response = generate_response(
                model, tokenizer, prompt,
                max_length=config.max_length,
                temperature=config.temperature,
            )

            if response_contains_code(response, item["entry_point"]):
                correct += 1

        except Exception as e:
            errors.append(f"Sample {idx}: {str(e)}")

        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(dataset)} samples")

    return create_benchmark_result("humaneval", correct, len(dataset), latencies, errors)


def extract_answer(response: str) -> Optional[int]:
    """Extract answer from MMLU response.

    Args:
        response: Model response

    Returns:
        Answer index or None
    """
    response = response.strip().upper()

    for char in ["A", "B", "C", "D"]:
        if char in response:
            return ord(char) - ord("A")

    return None


def extract_gsm8k_answer(response: str) -> str:
    """Extract final answer from GSM8K response.

    Args:
        response: Model response

    Returns:
        Extracted answer
    """
    lines = response.strip().split("\n")
    for line in reversed(lines):
        if line.strip() and not line.strip().startswith(("Step", "=", "-")):
            return line.strip()

    return response.strip().split(".")[-1].strip() if "." in response else response.strip()


def answer_matches(expected: str, predicted: str) -> bool:
    """Check if predicted answer matches expected.

    Args:
        expected: Expected answer
        predicted: Predicted answer

    Returns:
        True if matches, False otherwise
    """
    expected_clean = "".join(c for c in expected if c.isdigit() or c in ".-")
    predicted_clean = "".join(c for c in predicted if c.isdigit() or c in ".-")

    return expected_clean == predicted_clean


def response_contains_code(response: str, function_name: str) -> bool:
    """Check if response contains code for function.

    Args:
        response: Model response
        function_name: Function name to look for

    Returns:
        True if contains code, False otherwise
    """
    return f"def {function_name}" in response or f"function {function_name}" in response


def create_benchmark_result(
    benchmark: str,
    num_correct: int,
    num_samples: int,
    latencies: List[float],
    errors: List[str],
) -> BenchmarkResult:
    """Create benchmark result from data.

    Args:
        benchmark: Benchmark name
        num_correct: Number of correct answers
        num_samples: Total number of samples
        latencies: List of latencies in ms
        errors: List of errors

    Returns:
        Benchmark result
    """
    import statistics

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    total_time = sum(latencies) / 1000

    return BenchmarkResult(
        benchmark=benchmark,
        num_samples=num_samples,
        num_correct=num_correct,
        accuracy=accuracy,
        latency_avg_ms=avg_latency,
        latency_p50_ms=sorted_latencies[int(n * 0.5)] if n > 0 else 0.0,
        latency_p95_ms=sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
        latency_p99_ms=sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
        total_time_seconds=total_time,
        samples_per_second=num_samples / total_time if total_time > 0 else 0.0,
        errors=errors,
    )


def save_results(result: BenchmarkResult, output_dir: str) -> None:
    """Save benchmark results to file.

    Args:
        result: Benchmark result
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    result_dict = {
        "benchmark": result.benchmark,
        "num_samples": result.num_samples,
        "num_correct": result.num_correct,
        "accuracy": result.accuracy,
        "latency_avg_ms": result.latency_avg_ms,
        "latency_p50_ms": result.latency_p50_ms,
        "latency_p95_ms": result.latency_p95_ms,
        "latency_p99_ms": result.latency_p99_ms,
        "total_time_seconds": result.total_time_seconds,
        "samples_per_second": result.samples_per_second,
        "errors": result.errors,
    }

    output_path = Path(output_dir) / f"{result.benchmark}_results.json"
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def main() -> int:
    """Main entry point for benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Run standard benchmarks (MMLU, GSM8K, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run MMLU benchmark
    python run_benchmark.py --model-id meta-llama/Llama-2-7b --benchmark mmlu --output-dir ./results

    # Run GSM8K benchmark
    python run_benchmark.py --model-id gpt2 --benchmark gsm8k --output-dir ./results

    # Run with limited samples
    python run_benchmark.py --model-id gpt2 --benchmark humaneval --num-samples 50 --output-dir ./results
        """
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["mmlu", "gsm8k", "humaneval", "math", "hellaswag", "truthfulqa"],
        help="Benchmark to run"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to run (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--use-auth-token",
        help="HuggingFace auth token for private models"
    )

    args = parser.parse_args()

    if not validate_benchmark(args.benchmark):
        logger.error(f"Invalid benchmark: {args.benchmark}")
        return 1

    config = BenchmarkConfig(
        model_id=args.model_id,
        benchmark=args.benchmark,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        device=args.device,
        use_auth_token=args.use_auth_token,
    )

    try:
        model, tokenizer = load_model(args.model_id, args.device, args.use_auth_token)

        if args.benchmark == "mmlu":
            result = run_mmlu_benchmark(model, tokenizer, config)
        elif args.benchmark == "gsm8k":
            result = run_gsm8k_benchmark(model, tokenizer, config)
        elif args.benchmark == "humaneval":
            result = run_humaneval_benchmark(model, tokenizer, config)
        else:
            logger.error(f"Benchmark {args.benchmark} not yet implemented")
            return 1

        save_results(result, args.output_dir)

        print(json.dumps({
            "benchmark": result.benchmark,
            "accuracy": result.accuracy,
            "num_correct": result.num_correct,
            "num_samples": result.num_samples,
            "latency_avg_ms": result.latency_avg_ms,
            "samples_per_second": result.samples_per_second,
        }, indent=2))

        return 0

    except KeyboardInterrupt:
        logger.info("Benchmark cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())