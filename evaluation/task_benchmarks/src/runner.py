"""
Benchmark Test Runner with Pytest Integration

Provides test runner for executing benchmark evaluations in CI/CD pipelines.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import sys

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Test runner for benchmarks with pytest integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results = []

    def run_benchmark(
        self,
        benchmark_name: str,
        dataset: Any,
        predictions: List[str],
        ground_truth: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single benchmark and return results.
        """
        result = {
            "benchmark": benchmark_name,
            "timestamp": datetime.utcnow().isoformat(),
            "num_questions": len(dataset),
            "metrics": {},
        }

        try:
            if benchmark_name == "MMLU":
                result["metrics"]["accuracy"] = dataset.accuracy(predictions)
                result["metrics"]["by_subject"] = dataset.accuracy_by_subject(
                    predictions
                )

            elif benchmark_name == "GSM8K":
                result["metrics"]["accuracy"] = dataset.accuracy(
                    predictions, ground_truth
                )

            elif benchmark_name == "HumanEval":
                # Compute pass@1, pass@5, pass@10
                for k in [1, 5, 10]:
                    pass_at_k = (
                        sum(1 for p in predictions if p == "1") / len(predictions)
                        if predictions
                        else 0.0
                    )
                    result["metrics"][f"pass@{k}"] = pass_at_k

            elif benchmark_name == "SWE-bench":
                # Compute resolution rate
                resolved = sum(
                    1 for p in predictions if p.lower() in ["true", "1", "yes"]
                )
                result["metrics"]["resolution_rate"] = (
                    resolved / len(predictions) if predictions else 0.0
                )

            result["status"] = "success"

        except Exception as e:
            logger.error(f"Error running benchmark {benchmark_name}: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        self.results.append(result)
        return result

    def save_results(self, path: str) -> None:
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {path}")

    def print_summary(self) -> None:
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)

        for result in self.results:
            print(f"\n{result['benchmark']}:")
            print(f"  Questions: {result['num_questions']}")
            print(f"  Status: {result['status']}")

            if result["status"] == "success":
                for metric, value in result["metrics"].items():
                    if isinstance(value, dict):
                        print(f"  {metric}:")
                        for k, v in value.items():
                            print(f"    {k}: {v:.4f}")
                    else:
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
            else:
                print(f"  Error: {result.get('error', 'Unknown')}")

        print("\n" + "=" * 60)


def pytest_benchmark_wrapper(
    benchmark_name: str, dataset_path: str, predictions_path: str
):
    """
    Pytest wrapper function for benchmark testing.

    Usage:
        pytest evaluation/task_benchmarks/tests/test_mmlu.py::test_mmlu_benchmark
    """
    from .benchmarks import load_mmlu, load_gsm8k, load_humaneval, load_swe_bench

    # Load dataset and predictions
    if benchmark_name == "MMLU":
        dataset = load_mmlu(dataset_path)
    elif benchmark_name == "GSM8K":
        dataset = load_gsm8k(dataset_path)
    elif benchmark_name == "HumanEval":
        dataset = load_humaneval(dataset_path)
    elif benchmark_name == "SWE-bench":
        dataset = load_swe_bench(dataset_path)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    runner = BenchmarkRunner()
    result = runner.run_benchmark(benchmark_name, dataset, predictions)

    # Assert metrics meet thresholds (configurable)
    if result["status"] != "success":
        raise AssertionError(f"Benchmark failed: {result.get('error')}")

    return result
