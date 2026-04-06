import argparse
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import time
from typing import Dict


class OpenAIMockHandler(BaseHTTPRequestHandler):
    server_version = "OpenAIMock/1.0"

    def _write_json(self, status: int, payload: Dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
            return

        if self.path == "/v1/models":
            model = self.server.model_name  # type: ignore[attr-defined]
            payload = {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "llm-whisperer-mock",
                    }
                ],
            }
            self._write_json(200, payload)
            return

        self._write_json(404, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": {"message": "Not found"}})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        try:
            request_payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._write_json(400, {"error": {"message": "Invalid JSON body"}})
            return

        model = request_payload.get("model") or self.server.model_name  # type: ignore[attr-defined]
        messages = request_payload.get("messages") or []
        prompt_text = ""
        if messages and isinstance(messages, list):
            last = messages[-1]
            if isinstance(last, dict):
                prompt_text = str(last.get("content", ""))

        completion = (
            "MOCK_RESPONSE: "
            + (prompt_text[:120] if prompt_text else "No prompt provided.")
        )

        # A tiny deterministic token approximation for benchmarking scripts.
        prompt_tokens = max(len(prompt_text.split()), 1)
        completion_tokens = max(len(completion.split()), 1)

        now = int(time())
        response_payload = {
            "id": f"chatcmpl-mock-{now}",
            "object": "chat.completion",
            "created": now,
            "model": str(model),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        self._write_json(200, response_payload)

    def log_message(self, fmt: str, *args) -> None:
        # Keep benchmark output clean.
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible chat endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--model", default="mock-local-model")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), OpenAIMockHandler)
    server.model_name = args.model  # type: ignore[attr-defined]

    def _shutdown(_sig: int, _frame) -> None:
        server.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Mock server listening on http://{args.host}:{args.port}")
    print("Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions")
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())