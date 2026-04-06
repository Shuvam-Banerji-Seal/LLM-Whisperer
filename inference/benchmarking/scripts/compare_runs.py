import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark summary files")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    baseline = load_json(Path(args.baseline))
    candidate = load_json(Path(args.candidate))

    report = {
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "delta": {
            "latency_p95_ms_pct": round(
                pct_change(
                    candidate.get("latency_ms", {}).get("p95", 0.0),
                    baseline.get("latency_ms", {}).get("p95", 0.0),
                ),
                4,
            ),
            "ttft_p95_ms_pct": round(
                pct_change(
                    candidate.get("ttft_ms", {}).get("p95", 0.0),
                    baseline.get("ttft_ms", {}).get("p95", 0.0),
                ),
                4,
            ),
            "throughput_rps_pct": round(
                pct_change(
                    candidate.get("throughput_rps", 0.0),
                    baseline.get("throughput_rps", 0.0),
                ),
                4,
            ),
            "token_throughput_tps_pct": round(
                pct_change(
                    candidate.get("token_throughput_tps", 0.0),
                    baseline.get("token_throughput_tps", 0.0),
                ),
                4,
            ),
            "success_rate_pct": round(
                pct_change(
                    candidate.get("success_rate", 0.0),
                    baseline.get("success_rate", 0.0),
                ),
                4,
            ),
        },
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
