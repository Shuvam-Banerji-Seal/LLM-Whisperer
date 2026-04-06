from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyyaml. Install with `pip install pyyaml` to use eval_runner.py."
    ) from exc


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return content


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    missing = [k for k in weights if k not in scores]
    if missing:
        raise ValueError(f"Missing score dimensions: {', '.join(missing)}")
    return sum(scores[k] * weights[k] for k in weights)


def evaluate(
    thresholds: dict[str, Any],
    score_value: float,
    safety_score: float,
    critical_findings: int,
    env: str,
) -> dict[str, Any]:
    env_cfg = thresholds.get(env)
    if not isinstance(env_cfg, dict):
        raise ValueError(f"Unknown environment threshold profile: {env}")

    overall_min = float(env_cfg["overall_min"])
    safety_min = float(env_cfg["safety_min"])
    allow_critical = bool(env_cfg["allow_critical_findings"])

    pass_overall = score_value >= overall_min
    pass_safety = safety_score >= safety_min
    pass_critical = allow_critical or critical_findings == 0

    decision = "pass" if pass_overall and pass_safety and pass_critical else "fail"

    return {
        "decision": decision,
        "checks": {
            "overall": pass_overall,
            "safety": pass_safety,
            "critical_findings": pass_critical,
        },
        "thresholds": {
            "overall_min": overall_min,
            "safety_min": safety_min,
            "allow_critical_findings": allow_critical,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weighted score against profile thresholds.")
    parser.add_argument(
        "--weights",
        default="agents/evaluation/score_weights.yaml",
        help="Path to score_weights.yaml",
    )
    parser.add_argument(
        "--scores-json",
        required=True,
        help=(
            "JSON map of rubric scores, for example "
            "'{\"task_completion\":4.2,\"correctness\":4.4,\"safety_policy\":4.8,"
            "\"grounding_evidence\":4.1,\"efficiency\":3.9,\"robustness_recovery\":4.0}'"
        ),
    )
    parser.add_argument("--environment", default="dev", choices=["dev", "prod"])
    parser.add_argument("--critical-findings", type=int, default=0)
    args = parser.parse_args()

    weights_cfg = load_yaml(Path(args.weights))
    weights = weights_cfg.get("weights")
    thresholds = weights_cfg.get("thresholds")
    if not isinstance(weights, dict) or not isinstance(thresholds, dict):
        raise ValueError("score_weights.yaml must contain `weights` and `thresholds` mappings")

    scores = json.loads(args.scores_json)
    if not isinstance(scores, dict):
        raise ValueError("--scores-json must decode to an object")

    scores_cast = {k: float(v) for k, v in scores.items()}
    score_value = weighted_score(scores_cast, {k: float(v) for k, v in weights.items()})

    safety_score = float(scores_cast.get("safety_policy", 0.0))
    result = evaluate(thresholds, score_value, safety_score, args.critical_findings, args.environment)

    output = {
        "environment": args.environment,
        "weighted_score": round(score_value, 4),
        "safety_score": round(safety_score, 4),
        "critical_findings": args.critical_findings,
        **result,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
