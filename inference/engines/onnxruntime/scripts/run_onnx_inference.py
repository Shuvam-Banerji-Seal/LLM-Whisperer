import argparse
import json
import statistics
import sys
import time
from typing import Dict, List

import numpy as np
import onnxruntime as ort


DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def normalize_shape(shape: List[object]) -> List[int]:
    dims = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            dims.append(1)
    return dims


def build_input(node: ort.NodeArg) -> np.ndarray:
    if node.type not in DTYPE_MAP:
        raise ValueError(f"Unsupported ONNX input type: {node.type}")
    dtype = DTYPE_MAP[node.type]
    shape = normalize_shape(node.shape)

    if np.issubdtype(dtype, np.floating):
        return np.random.random(size=shape).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 10, size=shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.randint(0, 2, size=shape).astype(np.bool_)
    raise ValueError(f"Unsupported numpy dtype: {dtype}")


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime inference latency")
    parser.add_argument("--model", required=True)
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    sess = ort.InferenceSession(args.model, providers=providers)
    model_inputs = {node.name: build_input(node) for node in sess.get_inputs()}

    for _ in range(args.warmup):
        sess.run(None, model_inputs)

    latencies_ms: List[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        sess.run(None, model_inputs)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(latencies_ms)
    report: Dict[str, object] = {
        "model": args.model,
        "providers": sess.get_providers(),
        "runs": args.runs,
        "warmup": args.warmup,
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 4),
            "median": round(statistics.median(latencies_ms), 4),
            "p95": round(percentile(ordered, 0.95), 4),
            "p99": round(percentile(ordered, 0.99), 4),
            "min": round(min(latencies_ms), 4),
            "max": round(max(latencies_ms), 4),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
