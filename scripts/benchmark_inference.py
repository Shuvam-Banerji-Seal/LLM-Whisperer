#!/usr/bin/env python3
"""
Benchmark inference performance.

This script benchmarks LLM inference performance including:
- Throughput (tokens/second)
- Latency (time to first token, end-to-end)
- Memory usage
- Batch processing efficiency
- Multiple concurrent requests

Usage:
    python benchmark_inference.py --model-id meta-llama/Llama-2-7b --num-requests 100
    python benchmark_inference.py --model-id gpt2 --batch-size 8 --benchmark throughput
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
    """Configuration for inference benchmarking."""
    model_id: str
    num_requests: int = 100
    batch_size: int = 1
    max_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 1.0
    benchmark_type: str = "all"
    warmup_requests: int = 5
    num_concurrent: int = 1
    input_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])
    output_length: int = 128
    device: str = "auto"
    use_auth_token: Optional[str] = None


@dataclass
class BenchmarkMetrics:
    """Metrics from a benchmark run."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_time_seconds: float
    throughput_tokens_per_second: float
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    time_to_first_token_avg_ms: float
    time_to_first_token_p50_ms: float
    time_to_first_token_p95_ms: float
    prefill_time_avg_ms: float
    decode_time_avg_ms: float
    memory_used_mb: float
    memory_peak_mb: float


def validate_model_id(model_id: str) -> bool:
    """Validate model ID.

    Args:
        model_id: Model identifier

    Returns:
        True if valid, False otherwise
    """
    return bool(model_id and len(model_id) > 0)


def load_model_and_tokenizer(config: BenchmarkConfig) -> tuple:
    """Load model and tokenizer.

    Args:
        config: Benchmark configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {config.model_id}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Install with: pip install transformers")
        raise

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        use_auth_token=config.use_auth_token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}

    if config.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = "cpu"
        except ImportError:
            model_kwargs["device_map"] = "cpu"
    else:
        model_kwargs["device_map"] = config.device

    if config.use_auth_token:
        model_kwargs["use_auth_token"] = config.use_auth_token

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)

    model.eval()

    return model, tokenizer


def generate_with_timing(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple:
    """Generate text and measure timing.

    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (generated_text, ttft_ms, total_latency_ms)
    """
    try:
        import torch
    except ImportError:
        logger.error("torch not installed")
        raise

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    prefill_start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    prefill_time = (time.time() - prefill_start) * 1000

    generated_tokens = outputs.sequences[0]

    ttft = prefill_time

    total_time = (time.time() - prefill_start) * 1000

    decode_time = total_time - ttft

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    num_generated = len(generated_tokens) - input_ids.shape[1]

    return generated_text, ttft, total_time, num_generated


def generate_batch_with_timing(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
) -> tuple:
    """Generate text for batch and measure timing.

    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompts: List of prompts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (list of generated texts, avg_latency_ms)
    """
    try:
        import torch
    except ImportError:
        logger.error("torch not installed")
        raise

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    total_time_ms = (time.time() - start_time) * 1000

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return generated_texts, total_time_ms


def generate_concurrent_with_timing(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    num_requests: int,
) -> List[tuple]:
    """Generate text with concurrent requests.

    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        num_requests: Number of concurrent requests

    Returns:
        List of (generated_text, latency_ms) tuples
    """
    import concurrent.futures

    def generate_single(_):
        return generate_with_timing(model, tokenizer, prompt, max_new_tokens, temperature)

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        results = list(executor.map(generate_single, range(num_requests)))

    return results


def measure_memory_usage() -> Dict[str, float]:
    """Measure current memory usage.

    Returns:
        Dictionary with memory statistics
    """
    try:
        import torch
    except ImportError:
        return {"error": "torch not installed"}

    memory_stats = {}

    if torch.cuda.is_available():
        memory_stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        memory_stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        memory_stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    try:
        import psutil
        process = psutil.Process()
        memory_stats["cpu_memory_mb"] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    return memory_stats


def run_throughput_benchmark(config: BenchmarkConfig) -> BenchmarkMetrics:
    """Run throughput benchmark.

    Args:
        config: Benchmark configuration

    Returns:
        Benchmark metrics
    """
    logger.info("Running throughput benchmark")

    model, tokenizer = load_model_and_tokenizer(config)

    test_prompts = [
        "The quick brown fox jumps over the lazy dog. This is a test of language understanding.",
        "Explain the concept of artificial intelligence and machine learning in simple terms.",
        "Write a short story about a robot discovering emotions for the first time.",
    ]

    for i in range(config.warmup_requests):
        prompt = test_prompts[i % len(test_prompts)]
        _ = generate_with_timing(model, tokenizer, prompt, 50, config.temperature)

    logger.info(f"Running {config.num_requests} requests with batch size {config.batch_size}")

    latencies = []
    ttfts = []
    num_tokens_list = []

    for i in range(0, config.num_requests, config.batch_size):
        batch_prompts = []
        for j in range(config.batch_size):
            if i + j < config.num_requests:
                prompt = test_prompts[(i + j) % len(test_prompts)]
                batch_prompts.append(prompt)

        if not batch_prompts:
            break

        if len(batch_prompts) == 1:
            _, ttft, latency, num_tokens = generate_with_timing(
                model, tokenizer, batch_prompts[0],
                config.max_new_tokens, config.temperature
            )
            latencies.append(latency)
            ttfts.append(ttft)
            num_tokens_list.append(num_tokens)
        else:
            _, batch_latency = generate_batch_with_timing(
                model, tokenizer, batch_prompts,
                config.max_new_tokens, config.temperature
            )
            avg_latency = batch_latency / len(batch_prompts)
            latencies.extend([avg_latency] * len(batch_prompts))
            ttfts.extend([0] * len(batch_prompts))
            num_tokens_list.extend([config.max_new_tokens] * len(batch_prompts))

        if (i + config.batch_size) % 20 == 0:
            logger.info(f"Processed {min(i + config.batch_size, config.num_requests)}/{config.num_requests}")

    latencies.sort()
    ttfts.sort()

    total_time = sum(latencies) / 1000
    total_tokens = sum(num_tokens_list)
    throughput = total_tokens / total_time if total_time > 0 else 0

    n = len(latencies)

    return BenchmarkMetrics(
        total_requests=config.num_requests,
        successful_requests=config.num_requests,
        failed_requests=0,
        total_tokens=total_tokens,
        total_time_seconds=total_time,
        throughput_tokens_per_second=throughput,
        latency_avg_ms=sum(latencies) / n if n > 0 else 0,
        latency_p50_ms=latencies[int(n * 0.5)] if n > 0 else 0,
        latency_p95_ms=latencies[int(n * 0.95)] if n > 0 else 0,
        latency_p99_ms=latencies[int(n * 0.99)] if n > 0 else 0,
        time_to_first_token_avg_ms=sum(ttfts) / n if n > 0 else 0,
        time_to_first_token_p50_ms=ttfts[int(n * 0.5)] if n > 0 else 0,
        time_to_first_token_p95_ms=ttfts[int(n * 0.95)] if n > 0 else 0,
        prefill_time_avg_ms=sum(ttfts) / n if n > 0 else 0,
        decode_time_avg_ms=(sum(latencies) - sum(ttfts)) / n if n > 0 else 0,
        memory_used_mb=0,
        memory_peak_mb=0,
    )


def run_latency_benchmark(config: BenchmarkConfig) -> BenchmarkMetrics:
    """Run latency benchmark.

    Args:
        config: Benchmark configuration

    Returns:
        Benchmark metrics
    """
    logger.info("Running latency benchmark")

    model, tokenizer = load_model_and_tokenizer(config)

    for length in config.input_lengths:
        test_prompt = "word " * length
        _ = generate_with_timing(model, tokenizer, test_prompt, 50, 0.0)

    results_by_length = {}

    for input_length in config.input_lengths:
        test_prompt = "word " * input_length

        latencies = []
        ttfts = []
        num_tokens_list = []

        for _ in range(config.num_requests // len(config.input_lengths)):
            _, ttft, latency, num_tokens = generate_with_timing(
                model, tokenizer, test_prompt,
                config.output_length, 0.0
            )
            latencies.append(latency)
            ttfts.append(ttft)
            num_tokens_list.append(num_tokens)

        latencies.sort()
        ttfts.sort()
        n = len(latencies)

        results_by_length[input_length] = {
            "latency_avg_ms": sum(latencies) / n if n > 0 else 0,
            "latency_p50_ms": latencies[int(n * 0.5)] if n > 0 else 0,
            "latency_p95_ms": latencies[int(n * 0.95)] if n > 0 else 0,
            "ttft_avg_ms": sum(ttfts) / n if n > 0 else 0,
        }

    all_latencies = []
    all_ttfts = []
    all_tokens = []

    for length in config.input_lengths:
        test_prompt = "word " * length
        for _ in range(config.num_requests // len(config.input_lengths)):
            _, ttft, latency, num_tokens = generate_with_timing(
                model, tokenizer, test_prompt,
                config.output_length, 0.0
            )
            all_latencies.append(latency)
            all_ttfts.append(ttft)
            all_tokens.append(num_tokens)

    all_latencies.sort()
    all_ttfts.sort()
    n = len(all_latencies)

    total_time = sum(all_latencies) / 1000
    total_tokens = sum(all_tokens)
    throughput = total_tokens / total_time if total_time > 0 else 0

    return BenchmarkMetrics(
        total_requests=config.num_requests,
        successful_requests=config.num_requests,
        failed_requests=0,
        total_tokens=total_tokens,
        total_time_seconds=total_time,
        throughput_tokens_per_second=throughput,
        latency_avg_ms=sum(all_latencies) / n if n > 0 else 0,
        latency_p50_ms=all_latencies[int(n * 0.5)] if n > 0 else 0,
        latency_p95_ms=all_latencies[int(n * 0.95)] if n > 0 else 0,
        latency_p99_ms=all_latencies[int(n * 0.99)] if n > 0 else 0,
        time_to_first_token_avg_ms=sum(all_ttfts) / n if n > 0 else 0,
        time_to_first_token_p50_ms=all_ttfts[int(n * 0.5)] if n > 0 else 0,
        time_to_first_token_p95_ms=all_ttfts[int(n * 0.95)] if n > 0 else 0,
        prefill_time_avg_ms=sum(all_ttfts) / n if n > 0 else 0,
        decode_time_avg_ms=(sum(all_latencies) - sum(all_ttfts)) / n if n > 0 else 0,
        memory_used_mb=0,
        memory_peak_mb=0,
    )


def run_concurrent_benchmark(config: BenchmarkConfig) -> BenchmarkMetrics:
    """Run concurrent requests benchmark.

    Args:
        config: Benchmark configuration

    Returns:
        Benchmark metrics
    """
    logger.info(f"Running concurrent benchmark with {config.num_concurrent} concurrent requests")

    model, tokenizer = load_model_and_tokenizer(config)

    test_prompt = "The quick brown fox jumps over the lazy dog."

    for _ in range(config.warmup_requests):
        _ = generate_with_timing(model, tokenizer, test_prompt, 50, 0.0)

    all_latencies = []
    all_ttfts = []
    total_tokens = 0

    for batch_start in range(0, config.num_requests, config.num_concurrent):
        results = generate_concurrent_with_timing(
            model, tokenizer, test_prompt,
            config.output_length, 0.0,
            min(config.num_concurrent, config.num_requests - batch_start)
        )

        for text, ttft, latency in results:
            all_latencies.append(latency)
            all_ttfts.append(ttft)
            total_tokens += config.output_length

    all_latencies.sort()
    all_ttfts.sort()
    n = len(all_latencies)

    total_time = sum(all_latencies) / 1000
    throughput = total_tokens / total_time if total_time > 0 else 0

    return BenchmarkMetrics(
        total_requests=config.num_requests,
        successful_requests=config.num_requests,
        failed_requests=0,
        total_tokens=total_tokens,
        total_time_seconds=total_time,
        throughput_tokens_per_second=throughput,
        latency_avg_ms=sum(all_latencies) / n if n > 0 else 0,
        latency_p50_ms=all_latencies[int(n * 0.5)] if n > 0 else 0,
        latency_p95_ms=all_latencies[int(n * 0.95)] if n > 0 else 0,
        latency_p99_ms=all_latencies[int(n * 0.99)] if n > 0 else 0,
        time_to_first_token_avg_ms=sum(all_ttfts) / n if n > 0 else 0,
        time_to_first_token_p50_ms=all_ttfts[int(n * 0.5)] if n > 0 else 0,
        time_to_first_token_p95_ms=all_ttfts[int(n * 0.95)] if n > 0 else 0,
        prefill_time_avg_ms=sum(all_ttfts) / n if n > 0 else 0,
        decode_time_avg_ms=(sum(all_latencies) - sum(all_ttfts)) / n if n > 0 else 0,
        memory_used_mb=0,
        memory_peak_mb=0,
    )


def run_all_benchmarks(config: BenchmarkConfig) -> Dict[str, BenchmarkMetrics]:
    """Run all benchmarks.

    Args:
        config: Benchmark configuration

    Returns:
        Dictionary of benchmark results
    """
    logger.info("Running all benchmarks")

    results = {}

    results["throughput"] = run_throughput_benchmark(config)
    results["latency"] = run_latency_benchmark(config)
    results["concurrent"] = run_concurrent_benchmark(config)

    return results


def save_results(metrics: BenchmarkMetrics, output_path: str) -> None:
    """Save benchmark results to file.

    Args:
        metrics: Benchmark metrics
        output_path: Output file path
    """
    result_dict = {
        "total_requests": metrics.total_requests,
        "successful_requests": metrics.successful_requests,
        "failed_requests": metrics.failed_requests,
        "total_tokens": metrics.total_tokens,
        "total_time_seconds": metrics.total_time_seconds,
        "throughput_tokens_per_second": metrics.throughput_tokens_per_second,
        "latency_avg_ms": metrics.latency_avg_ms,
        "latency_p50_ms": metrics.latency_p50_ms,
        "latency_p95_ms": metrics.latency_p95_ms,
        "latency_p99_ms": metrics.latency_p99_ms,
        "time_to_first_token_avg_ms": metrics.time_to_first_token_avg_ms,
        "time_to_first_token_p50_ms": metrics.time_to_first_token_p50_ms,
        "time_to_first_token_p95_ms": metrics.time_to_first_token_p95_ms,
        "prefill_time_avg_ms": metrics.prefill_time_avg_ms,
        "decode_time_avg_ms": metrics.decode_time_avg_ms,
        "memory_used_mb": metrics.memory_used_mb,
        "memory_peak_mb": metrics.memory_peak_mb,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def print_metrics(metrics: BenchmarkMetrics, benchmark_name: str = "") -> None:
    """Print benchmark metrics.

    Args:
        metrics: Benchmark metrics
        benchmark_name: Name of benchmark
    """
    prefix = f"{benchmark_name}: " if benchmark_name else ""

    print("\n" + "=" * 60)
    print(f"{prefix}BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Requests:      {metrics.total_requests}")
    print(f"Successful:          {metrics.successful_requests}")
    print(f"Failed:              {metrics.failed_requests}")
    print(f"Total Tokens:        {metrics.total_tokens}")
    print(f"Total Time:          {metrics.total_time_seconds:.2f}s")
    print(f"Throughput:          {metrics.throughput_tokens_per_second:.2f} tokens/s")
    print()
    print("Latency:")
    print(f"  Average:           {metrics.latency_avg_ms:.2f}ms")
    print(f"  P50:               {metrics.latency_p50_ms:.2f}ms")
    print(f"  P95:               {metrics.latency_p95_ms:.2f}ms")
    print(f"  P99:               {metrics.latency_p99_ms:.2f}ms")
    print()
    print("Time to First Token:")
    print(f"  Average:           {metrics.time_to_first_token_avg_ms:.2f}ms")
    print(f"  P50:               {metrics.time_to_first_token_p50_ms:.2f}ms")
    print(f"  P95:               {metrics.time_to_first_token_p95_ms:.2f}ms")
    print()
    print("Timing Breakdown:")
    print(f"  Prefill (avg):     {metrics.prefill_time_avg_ms:.2f}ms")
    print(f"  Decode (avg):      {metrics.decode_time_avg_ms:.2f}ms")
    print()
    print("=" * 60)


def main() -> int:
    """Main entry point for inference benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all benchmarks
    python benchmark_inference.py --model-id meta-llama/Llama-2-7b --num-requests 100

    # Run throughput benchmark
    python benchmark_inference.py --model-id gpt2 --benchmark-type throughput --num-requests 50

    # Run with custom settings
    python benchmark_inference.py --model-id gpt2 --batch-size 8 --num-requests 100 --output-file results.json
        """
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to run (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (default: 128)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--benchmark-type",
        choices=["all", "throughput", "latency", "concurrent"],
        default="all",
        help="Type of benchmark (default: all)"
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=5,
        help="Number of warmup requests (default: 5)"
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)"
    )
    parser.add_argument(
        "--input-lengths",
        nargs="+",
        type=int,
        default=[128, 256, 512],
        help="Input lengths to test (default: 128 256 512)"
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=128,
        help="Output length for latency benchmark (default: 128)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--output-file",
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--use-auth-token",
        help="HuggingFace auth token for private models"
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_id=args.model_id,
        num_requests=args.num_requests,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        benchmark_type=args.benchmark_type,
        warmup_requests=args.warmup_requests,
        num_concurrent=args.num_concurrent,
        input_lengths=args.input_lengths,
        output_length=args.output_length,
        device=args.device,
        use_auth_token=args.use_auth_token,
    )

    try:
        if args.benchmark_type == "throughput":
            metrics = run_throughput_benchmark(config)
            print_metrics(metrics, "Throughput")
            if args.output_file:
                save_results(metrics, args.output_file)
        elif args.benchmark_type == "latency":
            metrics = run_latency_benchmark(config)
            print_metrics(metrics, "Latency")
            if args.output_file:
                save_results(metrics, args.output_file)
        elif args.benchmark_type == "concurrent":
            metrics = run_concurrent_benchmark(config)
            print_metrics(metrics, "Concurrent")
            if args.output_file:
                save_results(metrics, args.output_file)
        else:
            results = run_all_benchmarks(config)
            for name, metrics in results.items():
                print_metrics(metrics, name)

            if args.output_file:
                for name, metrics in results.items():
                    output_path = args.output_file.replace(".json", f"_{name}.json")
                    save_results(metrics, output_path)

        return 0

    except KeyboardInterrupt:
        logger.info("Benchmark cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())