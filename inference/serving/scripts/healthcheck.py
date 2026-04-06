import argparse
import json
import sys
from typing import List

import requests


def probe(urls: List[str], timeout: int) -> dict:
    last_error = "no healthy endpoint found"
    for url in urls:
        try:
            response = requests.get(url, timeout=timeout)
            if 200 <= response.status_code < 300:
                return {"ok": True, "url": url, "status_code": response.status_code}
            if response.status_code == 404:
                continue
            return {
                "ok": False,
                "url": url,
                "status_code": response.status_code,
                "body": response.text[:300],
            }
        except Exception as exc:
            last_error = str(exc)
    return {"ok": False, "url": None, "status_code": 0, "error": last_error}


def main() -> int:
    parser = argparse.ArgumentParser(description="Health probe for OpenAI-compatible inference service")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    candidates = [
        f"{base}/health",
        f"{base}/v1/models",
        f"{base}/metrics",
    ]

    result = probe(candidates, args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
