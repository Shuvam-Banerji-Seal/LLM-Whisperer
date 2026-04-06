import argparse
import importlib
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test TensorRT-LLM OpenAI-compatible endpoint")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="local-dev-key")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Reply with OK.")
    args = parser.parse_args()

    openai_module = importlib.import_module("openai")
    OpenAI = getattr(openai_module, "OpenAI")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.0,
        max_tokens=24,
    )

    out = {
        "id": response.id,
        "model": response.model,
        "text": response.choices[0].message.content,
        "usage": response.usage.model_dump() if response.usage else None,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
