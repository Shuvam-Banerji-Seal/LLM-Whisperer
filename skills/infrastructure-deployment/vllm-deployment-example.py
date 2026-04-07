"""
Complete vLLM Deployment Example with Production Patterns
=========================================================

This script demonstrates:
- Loading and serving LLM models with vLLM
- Async request handling
- Request batching and optimization
- Error handling and retry logic
- Monitoring and metrics
- Multi-model serving with hot swap
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json

from vllm import AsyncLLM, LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import torch


# ============================================================================
# Configuration & Data Structures
# ============================================================================


@dataclass
class InferenceConfig:
    """Configuration for inference server"""

    model_name: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    dtype: str = "auto"
    quantization: Optional[str] = None  # "awq", "gptq", or None
    enable_prefix_caching: bool = True
    max_num_batched_tokens: int = 20000
    batch_timeout_ms: int = 100


@dataclass
class InferenceRequest:
    """Single inference request"""

    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class InferenceResult:
    """Inference result with metadata"""

    request_id: str
    output_text: str
    tokens_generated: int
    latency_ms: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# ============================================================================
# Metrics Collection
# ============================================================================


class MetricsCollector:
    """Collect and aggregate inference metrics"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.token_counts = deque(maxlen=window_size)
        self.request_times = deque(maxlen=window_size)
        self.errors = deque(maxlen=100)
        self.start_time = datetime.now()

    def record_inference(self, result: InferenceResult):
        """Record successful inference"""
        self.latencies.append(result.latency_ms)
        self.token_counts.append(result.tokens_generated)
        self.request_times.append(result.timestamp)

    def record_error(self, request_id: str, error: str):
        """Record error"""
        self.errors.append(
            {"request_id": request_id, "error": error, "timestamp": datetime.now()}
        )

    def get_stats(self) -> Dict:
        """Get current statistics"""
        if not self.latencies:
            return {}

        latencies = list(self.latencies)
        latencies.sort()

        return {
            "total_requests": len(self.request_times),
            "total_tokens": sum(self.token_counts),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": latencies[len(latencies) // 2],
            "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "total_errors": len(self.errors),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }


# ============================================================================
# Async Inference Server
# ============================================================================


class AsyncInferenceServer:
    """High-performance async inference server"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.llm: Optional[AsyncLLM] = None
        self.metrics = MetricsCollector()
        self.logger = self._setup_logger()
        self.request_queue = asyncio.Queue()
        self.batch_timeout = timedelta(milliseconds=config.batch_timeout_ms)
        self.batch_accumulator = []
        self.batch_start_time = None

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the LLM engine"""
        try:
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                pipeline_parallel_size=self.config.pipeline_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                dtype=self.config.dtype,
                quantization=self.config.quantization,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
            )

            self.llm = AsyncLLM(**engine_args.__dict__)
            self.logger.info(f"Initialized LLM: {self.config.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise

    async def process_batch(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """Process a batch of requests"""
        if not requests:
            return []

        try:
            prompts = [req.prompt for req in requests]
            sampling_params = [
                SamplingParams(
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                )
                for req in requests
            ]

            start_time = datetime.now()
            outputs = await self.llm.generate(prompts, sampling_params)
            latency = (datetime.now() - start_time).total_seconds() * 1000

            results = []
            for req, output in zip(requests, outputs):
                text = output.outputs[0].text
                tokens = len(output.outputs[0].token_ids)

                result = InferenceResult(
                    request_id=req.request_id,
                    output_text=text,
                    tokens_generated=tokens,
                    latency_ms=latency / len(requests),  # Per-request latency
                )
                results.append(result)
                self.metrics.record_inference(result)

            return results

        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            for req in requests:
                self.metrics.record_error(req.request_id, str(e))
            raise

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Single inference with batching"""
        self.batch_accumulator.append(request)

        if self.batch_start_time is None:
            self.batch_start_time = datetime.now()

        # Check if should trigger batch processing
        should_process = (
            len(self.batch_accumulator) >= 32  # Batch size threshold
            or (datetime.now() - self.batch_start_time > self.batch_timeout)
        )

        if should_process:
            batch = self.batch_accumulator[:]
            self.batch_accumulator = []
            self.batch_start_time = None

            results = await self.process_batch(batch)

            # Find and return matching result
            for result in results:
                if result.request_id == request.request_id:
                    return result

        # Wait for batch to process (for demonstration)
        await asyncio.sleep(0.01)
        return await self.infer(request) if self.batch_accumulator else None

    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return self.metrics.get_stats()


# ============================================================================
# Multi-Model Server with Hot Swap
# ============================================================================


class MultiModelServer:
    """Manage multiple models with hot-swapping"""

    def __init__(self, gpu_memory_gb: int = 80):
        self.gpu_memory_gb = gpu_memory_gb
        self.servers: Dict[str, AsyncInferenceServer] = {}
        self.current_model: Optional[str] = None
        self.logger = logging.getLogger(__name__)

    async def add_model(self, config: InferenceConfig):
        """Add a new model"""
        if config.model_name not in self.servers:
            server = AsyncInferenceServer(config)
            await server.initialize()
            self.servers[config.model_name] = server
            self.logger.info(f"Added model: {config.model_name}")

    async def infer(
        self, model_name: str, request: InferenceRequest
    ) -> InferenceResult:
        """Run inference on specified model"""
        if model_name not in self.servers:
            raise ValueError(f"Model {model_name} not found")

        return await self.servers[model_name].infer(request)

    async def switch_model(self, model_name: str):
        """Switch active model (with optional unloading)"""
        if model_name not in self.servers:
            raise ValueError(f"Model {model_name} not found")

        self.current_model = model_name
        self.logger.info(f"Switched to model: {model_name}")


# ============================================================================
# Example Usage & Testing
# ============================================================================


async def main():
    """Example usage"""

    # Configuration
    config = InferenceConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enable_prefix_caching=True,
    )

    # Initialize server
    server = AsyncInferenceServer(config)
    await server.initialize()

    # Example requests
    requests = [
        InferenceRequest(
            request_id="req-1", prompt="What is machine learning?", max_tokens=256
        ),
        InferenceRequest(
            request_id="req-2",
            prompt="Explain neural networks in simple terms",
            max_tokens=256,
        ),
        InferenceRequest(
            request_id="req-3", prompt="What are transformers?", max_tokens=256
        ),
    ]

    # Process requests with batching
    tasks = [server.infer(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Print results
    for result in results:
        if isinstance(result, InferenceResult):
            print(f"\nRequest {result.request_id}:")
            print(f"Output: {result.output_text[:200]}...")
            print(f"Tokens: {result.tokens_generated}")
            print(f"Latency: {result.latency_ms:.2f}ms")

    # Print metrics
    print("\n" + "=" * 50)
    print("SERVER METRICS")
    print("=" * 50)
    stats = server.get_metrics()
    print(json.dumps(stats, indent=2, default=str))


# ============================================================================
# FastAPI Integration Example
# ============================================================================

"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
server = AsyncInferenceServer(config)

class InferenceRequestSchema(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/v1/completions")
async def completions(request: InferenceRequestSchema):
    try:
        inf_request = InferenceRequest(
            request_id=str(datetime.now().timestamp()),
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )
        result = await server.infer(inf_request)
        return {
            "output": result.output_text,
            "tokens": result.tokens_generated,
            "latency_ms": result.latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return server.get_metrics()

# Run: uvicorn script:app --reload
"""


if __name__ == "__main__":
    asyncio.run(main())
