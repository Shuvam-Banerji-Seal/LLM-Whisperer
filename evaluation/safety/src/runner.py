"""
Safety Evaluation Runner
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class SafetyRunner:
    """Runner for comprehensive safety evaluations."""

    def __init__(self, evaluator, config: Optional[Dict[str, Any]] = None):
        self.evaluator = evaluator
        self.config = config or {}
        self.results = []

    def evaluate_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Batch evaluation of multiple texts."""
        batch_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "evaluator": self.evaluator.name,
            "total_texts": len(texts),
            "evaluations": [],
        }

        for idx, text in enumerate(texts):
            result = self.evaluator.evaluate(text, f"text_{idx}")
            batch_result["evaluations"].append(
                {
                    "text_id": result.text_id,
                    "is_safe": result.is_safe,
                    "overall_score": result.overall_score,
                    "findings": [
                        {
                            "category": f.category,
                            "severity": f.severity,
                            "text_span": f.text_span,
                            "score": f.score,
                            "metadata": f.metadata,
                        }
                        for f in result.findings
                    ],
                }
            )

        self.results.append(batch_result)
        return batch_result

    def compute_summary(self, batch_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics."""
        total = len(batch_result["evaluations"])
        safe = sum(1 for e in batch_result["evaluations"] if e["is_safe"])

        # Count findings by category and severity
        findings_by_category = {}
        findings_by_severity = {}

        for eval_item in batch_result["evaluations"]:
            for finding in eval_item["findings"]:
                cat = finding["category"]
                sev = finding["severity"]
                findings_by_category[cat] = findings_by_category.get(cat, 0) + 1
                findings_by_severity[sev] = findings_by_severity.get(sev, 0) + 1

        return {
            "evaluator": batch_result["evaluator"],
            "total_texts": total,
            "safe_texts": safe,
            "safety_rate": safe / total if total > 0 else 0.0,
            "findings_by_category": findings_by_category,
            "findings_by_severity": findings_by_severity,
        }

    def save_results(self, path: str) -> None:
        """Save results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {path}")

    def print_summary(self, batch_result: Dict[str, Any]) -> None:
        """Print summary."""
        summary = self.compute_summary(batch_result)

        print("\n" + "=" * 60)
        print("SAFETY EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Evaluator: {summary['evaluator']}")
        print(f"Total Texts: {summary['total_texts']}")
        print(
            f"Safe Texts: {summary['safe_texts']} ({summary['safety_rate'] * 100:.1f}%)"
        )

        if summary["findings_by_category"]:
            print("\nFindings by Category:")
            for category, count in summary["findings_by_category"].items():
                print(f"  {category}: {count}")

        if summary["findings_by_severity"]:
            print("\nFindings by Severity:")
            for severity, count in summary["findings_by_severity"].items():
                print(f"  {severity}: {count}")

        print("\n" + "=" * 60)
