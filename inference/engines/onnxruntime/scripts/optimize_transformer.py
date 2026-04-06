import argparse
import json
import sys
import time

import onnxruntime as ort


def parse_providers(value: str):
    return [p.strip() for p in value.split(",") if p.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize ONNX model with ONNX Runtime graph passes")
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--providers", default="CPUExecutionProvider")
    args = parser.parse_args()

    providers = parse_providers(args.providers)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = args.output_model

    start = time.perf_counter()
    session = ort.InferenceSession(args.input_model, sess_options=opts, providers=providers)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    report = {
        "input_model": args.input_model,
        "output_model": args.output_model,
        "providers": session.get_providers(),
        "optimization_level": "ORT_ENABLE_ALL",
        "session_build_ms": round(elapsed_ms, 3),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
