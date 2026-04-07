"""
LLM-as-Judge Evaluation Runner

Batch evaluation of LLM outputs using judge models.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class JudgeEvaluationRunner:
    """Runner for LLM-as-judge batch evaluations."""

    def __init__(self, judge, rubric, config: Optional[Dict[str, Any]] = None):
        self.judge = judge
        self.rubric = rubric
        self.config = config or {}
        self.results = []

    def evaluate_batch(
        self, items: List[Dict[str, str]], criteria: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run batch evaluation across multiple criteria.

        Args:
            items: List of {"query": str, "response": str} dicts
            criteria: List of criteria to evaluate (uses all by default)

        Returns:
            List of evaluation results
        """
        if criteria is None:
            criteria = list(self.rubric.criteria.keys())

        batch_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "judge": self.judge.name,
            "rubric": self.rubric.name,
            "items_count": len(items),
            "criteria": criteria,
            "evaluations": [],
        }

        for idx, item in enumerate(items):
            item_result = {
                "item_id": f"item_{idx}",
                "query": item.get("query", ""),
                "response": item.get("response", ""),
                "scores": {},
            }

            for criterion in criteria:
                try:
                    judgment = self.judge.judge(
                        query=item["query"],
                        response=item["response"],
                        rubric=self.rubric,
                        criterion=criterion,
                    )
                    item_result["scores"][criterion] = {
                        "score": judgment.score,
                        "rationale": judgment.rationale,
                    }
                except Exception as e:
                    logger.error(f"Error evaluating item {idx} on {criterion}: {e}")
                    item_result["scores"][criterion] = {"score": 0, "error": str(e)}

            batch_results["evaluations"].append(item_result)

        self.results.append(batch_results)
        return batch_results

    def compute_summary(self, batch_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics for batch evaluation."""
        summary = {
            "judge": batch_result["judge"],
            "rubric": batch_result["rubric"],
            "total_items": len(batch_result["evaluations"]),
            "criterion_stats": {},
        }

        # Compute stats per criterion
        for criterion in batch_result["criteria"]:
            scores = [
                eval_item["scores"][criterion]["score"]
                for eval_item in batch_result["evaluations"]
                if criterion in eval_item["scores"]
                and eval_item["scores"][criterion]["score"] > 0
            ]

            if scores:
                import statistics

                summary["criterion_stats"][criterion] = {
                    "mean": statistics.mean(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores),
                    "median": statistics.median(scores),
                }

        return summary

    def save_results(self, path: str) -> None:
        """Save evaluation results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {path}")

    def print_summary(self, batch_result: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        summary = self.compute_summary(batch_result)

        print("\n" + "=" * 60)
        print(f"JUDGE EVALUATION SUMMARY")
        print(f"Judge: {summary['judge']} | Rubric: {summary['rubric']}")
        print("=" * 60)

        print(f"\nTotal Items Evaluated: {summary['total_items']}")
        print("\nCriterion Statistics:")

        for criterion, stats in summary["criterion_stats"].items():
            print(f"\n  {criterion}:")
            print(f"    Mean:   {stats['mean']:.2f}")
            print(f"    Std Dev: {stats['std_dev']:.2f}")
            print(f"    Range:   {stats['min']} - {stats['max']}")
            print(f"    Median:  {stats['median']}")

        print("\n" + "=" * 60)
