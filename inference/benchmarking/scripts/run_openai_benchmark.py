import argparse
import concurrent.futures
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


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


def count_words(text: str) -> int:
    return len([w for w in text.split() if w])


def build_endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def do_request(
    endpoint: str,
    api_key: str,
    model: str,
    item: Dict[str, Any],
    timeout_s: int,
    system_prompt: str,
    run_id: str,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_id = item.get("request_id") or f"req-{uuid.uuid4().hex[:8]}"
    prompt = str(item.get("prompt", ""))
    max_tokens = int(item.get("max_tokens", 128))
    temperature = float(item.get("temperature", 0.0))

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    started_unix_ms = time.time() * 1000.0
    perf_start = time.perf_counter()

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
        elapsed_ms = (time.perf_counter() - perf_start) * 1000.0
        finished_unix_ms = time.time() * 1000.0

        status_code = response.status_code
        data: Dict[str, Any] = {}
        output_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        success = False
        error: Optional[str] = None

        if 200 <= status_code < 300:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                output_text = str(message.get("content", ""))
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            if completion_tokens <= 0:
                completion_tokens = max(count_words(output_text), 1)
            success = True
        else:
            error = response.text[:500]

        ttft_ms = elapsed_ms  # Approximation for non-streaming benchmarks.
        tpot_ms = elapsed_ms / max(completion_tokens, 1)

        return {
            "run_id": run_id,
            "request_id": request_id,
            "started_unix_ms": round(started_unix_ms, 3),
            "finished_unix_ms": round(finished_unix_ms, 3),
            "latency_ms": round(elapsed_ms, 4),
            "ttft_ms": round(ttft_ms, 4),
            "tpot_ms": round(tpot_ms, 4),
            "success": success,
            "status_code": status_code,
            "prompt_chars": len(prompt),
            "output_chars": len(output_text),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error": error,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - perf_start) * 1000.0
        finished_unix_ms = time.time() * 1000.0
        return {
            "run_id": run_id,
            "request_id": request_id,
            "started_unix_ms": round(started_unix_ms, 3),
            "finished_unix_ms": round(finished_unix_ms, 3),
            "latency_ms": round(elapsed_ms, 4),
            "ttft_ms": round(elapsed_ms, 4),
            "tpot_ms": round(elapsed_ms, 4),
            "success": False,
            "status_code": 0,
            "prompt_chars": len(prompt),
            "output_chars": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark against OpenAI-compatible endpoint")
    parser.add_argument("--base-url", required=True, help="Example: http://localhost:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--system-prompt", default="")
    args = parser.parse_args()

    run_id = f"run-{uuid.uuid4().hex[:12]}"
    endpoint = build_endpoint(args.base_url)

    workload = read_jsonl(Path(args.workload))
    if args.max_requests > 0:
        workload = workload[: args.max_requests]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
        futures = [
            pool.submit(
                do_request,
                endpoint,
                args.api_key,
                args.model,
                item,
                args.timeout,
                args.system_prompt,
                run_id,
            )
            for item in workload
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    with output_path.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    latencies = [r["latency_ms"] for r in results if r["success"]]
    success_count = sum(1 for r in results if r["success"])
    report = {
        "run_id": run_id,
        "endpoint": endpoint,
        "model": args.model,
        "requests": len(results),
        "success": success_count,
        "failure": len(results) - success_count,
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 4),
            "p95": round(percentile(latencies, 0.95), 4),
            "p99": round(percentile(latencies, 0.99), 4),
        },
        "output_file": str(output_path),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
