"""Evaluation pipeline entry point."""

import logging
import argparse
from pathlib import Path

from src.benchmark import BenchmarkOrchestrator, BenchmarkConfig
from src.metrics import MetricsComputer, MetricsConfig, RegressionDetector
from src.reporting import ReportGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_evaluation(
    model_path: str,
    benchmarks: list,
    output_dir: str = "./eval_results",
    save_html: bool = False,
):
    """Run complete evaluation pipeline."""
    logger.info("=" * 80)
    logger.info("LLM-Whisperer Evaluation Pipeline")
    logger.info("=" * 80)

    # Configure and run benchmarks
    benchmark_config = BenchmarkConfig(
        model_path=model_path, benchmarks=benchmarks, output_dir=output_dir
    )

    orchestrator = BenchmarkOrchestrator(benchmark_config)
    benchmark_results = orchestrator.run_all()

    # Compute metrics
    metrics_config = MetricsConfig(task_benchmarks=True, latency_analysis=True)

    computer = MetricsComputer(metrics_config)
    metrics = computer.compute_metrics(benchmark_results)

    # Generate report
    generator = ReportGenerator(output_dir=output_dir)
    report = generator.generate_report(benchmark_results, metrics, model_path)

    generator.save_report(report)
    logger.info(f"\nReport saved to {output_dir}/evaluation_report.json")

    if save_html:
        generator.generate_html_report(report)
        logger.info(f"HTML report saved to {output_dir}/evaluation_report.html")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Summary")
    logger.info("=" * 80)
    summary = orchestrator.get_summary()
    for key, value in summary.items():
        if key != "details":
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mmlu", "alpacaeval", "gsm8k", "hellaswag"],
        help="Benchmarks to run",
    )
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--save-html", action="store_true")

    args = parser.parse_args()

    run_evaluation(args.model, args.benchmarks, args.output_dir, args.save_html)
