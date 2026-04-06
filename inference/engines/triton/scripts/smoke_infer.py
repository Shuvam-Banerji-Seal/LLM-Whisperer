import argparse
import json
import sys

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Triton infer endpoint")
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--model", default="echo")
    parser.add_argument("--text", default="hello triton")
    args = parser.parse_args()

    endpoint = f"{args.url}/v2/models/{args.model}/infer"
    payload = {
        "inputs": [
            {
                "name": "TEXT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [args.text],
            }
        ],
        "outputs": [{"name": "ECHO_TEXT"}],
    }

    response = requests.post(endpoint, json=payload, timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
