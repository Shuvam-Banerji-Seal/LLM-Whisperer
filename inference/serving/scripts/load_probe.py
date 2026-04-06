import argparse
import concurrent.futures
import json
import statistics
import time
from typing import Dict, List

import requests


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def request_once(endpoint: str, headers: Dict[str, str], payload: Dict[str, object], timeout: int) -> Dict[str, object]:
    start = time.perf_counter()
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": False, "status_code": 0, "latency_ms": latency_ms}


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple concurrent load probe for OpenAI-compatible endpoint")
    parser.add_argument("--base-url", required=True, help="Example: http://localhost:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 16,
    }

    start = time.perf_counter()
    results: List[Dict[str, object]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
        futures = [
            pool.submit(request_once, endpoint, headers, payload, args.timeout)
            for _ in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    wall_s = time.perf_counter() - start
    success = [r for r in results if r["ok"]]
    latencies = [float(r["latency_ms"]) for r in success]

    report = {
        "endpoint": endpoint,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "success": len(success),
        "failure": len(results) - len(success),
        "wall_time_s": round(wall_s, 4),
        "throughput_rps": round(len(success) / max(wall_s, 1e-9), 4),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 4) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 4),
            "p99": round(percentile(latencies, 0.99), 4),
        },
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
