import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    success_rows = [r for r in rows if r.get("success")]
    latencies = [float(r.get("latency_ms", 0.0)) for r in success_rows]
    ttfts = [float(r.get("ttft_ms", 0.0)) for r in success_rows]
    tpots = [float(r.get("tpot_ms", 0.0)) for r in success_rows]

    start_ms = min((float(r.get("started_unix_ms", 0.0)) for r in rows), default=0.0)
    end_ms = max((float(r.get("finished_unix_ms", 0.0)) for r in rows), default=0.0)
    wall_s = max((end_ms - start_ms) / 1000.0, 1e-9)

    completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in success_rows)

    return {
        "run_id": rows[0].get("run_id", "unknown") if rows else "unknown",
        "requests": total,
        "success": len(success_rows),
        "failure": total - len(success_rows),
        "success_rate": round((len(success_rows) / total) if total else 0.0, 6),
        "wall_time_s": round(wall_s, 4),
        "throughput_rps": round((len(success_rows) / wall_s), 4),
        "token_throughput_tps": round((completion_tokens / wall_s), 4),
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 4),
            "p95": round(percentile(latencies, 0.95), 4),
            "p99": round(percentile(latencies, 0.99), 4),
        },
        "ttft_ms": {
            "p50": round(percentile(ttfts, 0.50), 4),
            "p95": round(percentile(ttfts, 0.95), 4),
            "p99": round(percentile(ttfts, 0.99), 4),
        },
        "tpot_ms": {
            "p50": round(percentile(tpots, 0.50), 4),
            "p95": round(percentile(tpots, 0.95), 4),
            "p99": round(percentile(tpots, 0.99), 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark NDJSON results")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    summary = summarize(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
