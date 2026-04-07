"""
Latency Evaluation Runner
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class LatencyRunner:
    """Runner for latency evaluations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results = []

    def run_benchmark(
        self, traces: List[Any], sla_checker: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Run latency benchmark on request traces."""
        from .metrics import LatencyMetricsComputer

        metrics = LatencyMetricsComputer.compute_metrics(traces)

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics.to_dict(),
        }

        if sla_checker:
            sla_results = sla_checker.check(metrics)
            result["sla_check"] = sla_results
            result["sla_passed"] = sla_checker.all_passed(metrics)

        self.results.append(result)
        return result

    def save_results(self, path: str) -> None:
        """Save results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {path}")

    def print_summary(self, result: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        metrics = result["metrics"]

        print("\n" + "=" * 70)
        print("LATENCY BENCHMARK RESULTS")
        print("=" * 70)

        print(f"\nTotal Requests: {metrics['num_requests']}")
        print(f"Success Rate: {metrics['success_rate'] * 100:.1f}%")

        print("\nTime to First Token (TTFT):")
        print(f"  Mean:   {metrics['ttft']['mean']:.2f} ms")
        print(f"  P50:    {metrics['ttft']['p50']:.2f} ms")
        print(f"  P95:    {metrics['ttft']['p95']:.2f} ms")
        print(f"  P99:    {metrics['ttft']['p99']:.2f} ms")

        print("\nTime Per Output Token (TPOT):")
        print(f"  Mean:   {metrics['tpot']['mean']:.2f} ms")
        print(f"  P50:    {metrics['tpot']['p50']:.2f} ms")
        print(f"  P95:    {metrics['tpot']['p95']:.2f} ms")
        print(f"  P99:    {metrics['tpot']['p99']:.2f} ms")

        print("\nThroughput:")
        print(f"  Requests/sec: {metrics['throughput']['req_per_sec']:.2f}")
        print(f"  Tokens/sec:   {metrics['throughput']['tokens_per_sec']:.2f}")
        print(
            f"  Goodput (tokens/sec): {metrics['throughput']['goodput_tokens_per_sec']:.2f}"
        )

        if "sla_passed" in result:
            print(f"\nSLA Status: {'PASSED' if result['sla_passed'] else 'FAILED'}")
            if result.get("sla_check"):
                for check, passed in result["sla_check"].items():
                    print(f"  {check}: {'✓' if passed else '✗'}")

        print("\n" + "=" * 70)
