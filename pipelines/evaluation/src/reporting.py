"""Evaluation results reporting."""

import logging
import json
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates evaluation reports."""

    def __init__(self, output_dir: str = "./eval_results"):
        """Initialize report generator.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        benchmark_results: Dict[str, Any],
        metrics: Dict[str, Any],
        model_name: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.

        Args:
            benchmark_results: Results from benchmarks
            metrics: Computed metrics
            model_name: Model name/path

        Returns:
            Complete report dictionary
        """
        logger.info("Generating evaluation report...")

        report = {
            "metadata": {
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "report_version": "1.0",
            },
            "benchmarks": benchmark_results,
            "metrics": metrics,
            "summary": self._generate_summary(benchmark_results, metrics),
        }

        return report

    def _generate_summary(
        self, benchmarks: Dict[str, Any], metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "num_benchmarks": len(benchmarks),
            "benchmarks_passed": sum(
                1 for b in benchmarks.values() if "error" not in b
            ),
            "average_score": 0.0,
        }

        # Calculate average score
        scores = []
        for benchmark, result in benchmarks.items():
            if "score" in result:
                scores.append(result["score"])

        if scores:
            summary["average_score"] = sum(scores) / len(scores)

        return summary

    def save_report(
        self, report: Dict[str, Any], filename: str = "evaluation_report.json"
    ) -> str:
        """Save report to file.

        Args:
            report: Report dictionary
            filename: Output filename

        Returns:
            Path to saved report
        """
        report_path = self.output_dir / filename

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {report_path}")
        return str(report_path)

    def generate_html_report(
        self, report: Dict[str, Any], filename: str = "evaluation_report.html"
    ) -> str:
        """Generate HTML report.

        Args:
            report: Report dictionary
            filename: Output filename

        Returns:
            Path to saved HTML report
        """
        html_content = self._create_html(report)

        report_path = self.output_dir / filename
        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {report_path}")
        return str(report_path)

    def _create_html(self, report: Dict[str, Any]) -> str:
        """Create HTML report content."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; }
        .score { font-weight: bold; color: #2196F3; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h1>Model Evaluation Report</h1>
    <div class="metric">
        <p><strong>Model:</strong> {model}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
    </div>
    
    <h2>Summary</h2>
    <div class="metric">
        <p>Benchmarks Completed: {num_benchmarks}/{benchmarks_passed}</p>
        <p class="score">Average Score: {avg_score:.2f}</p>
    </div>
    
    <h2>Benchmark Results</h2>
    <table>
        <tr><th>Benchmark</th><th>Score</th><th>Status</th></tr>
        {benchmark_rows}
    </table>
</body>
</html>
"""

        metadata = report.get("metadata", {})
        summary = report.get("summary", {})
        benchmarks = report.get("benchmarks", {})

        # Generate benchmark rows
        benchmark_rows = ""
        for name, result in benchmarks.items():
            status = "✓ Pass" if "error" not in result else "✗ Fail"
            score = result.get("score", "N/A")
            benchmark_rows += (
                f"<tr><td>{name}</td><td>{score}</td><td>{status}</td></tr>\n"
            )

        html = html.format(
            model=metadata.get("model", "Unknown"),
            timestamp=metadata.get("timestamp", ""),
            num_benchmarks=summary.get("num_benchmarks", 0),
            benchmarks_passed=summary.get("benchmarks_passed", 0),
            avg_score=summary.get("average_score", 0),
            benchmark_rows=benchmark_rows,
        )

        return html


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    generator = ReportGenerator()

    report = generator.generate_report(
        {"mmlu": {"score": 45.0}}, {"average_score": 45.0}, "test-model"
    )

    generator.save_report(report)
    generator.generate_html_report(report)
