"""
Utility Functions for LLM Engineering
Comprehensive helpers for common LLM tasks: tokenization, batching, evaluation, monitoring.

Key Utilities:
- TokenizationUtils: Handle tokenization edge cases
- BatchingUtils: Efficient batch preparation
- PromptUtils: Format prompts consistently
- EvaluationUtils: Common metrics (BLEU, ROUGE, BERTScore)
- MonitoringUtils: Track performance metrics
- CacheUtils: Smart caching strategies
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import hashlib
import json


# ============================================================================
# TOKENIZATION UTILITIES
# ============================================================================


class TokenizationUtils:
    """Handle tokenization edge cases and optimization."""

    @staticmethod
    def estimate_tokens(text: str, model: str = "gpt-3.5") -> int:
        """
        Estimate token count without loading tokenizer.

        Approximation:
        - GPT-2: ~4 chars per token
        - GPT-3: ~4 chars per token
        - Claude: ~3-4 chars per token
        - Spacing: add 1 token per space

        More accurate: use actual tokenizer
        """
        if model.startswith("gpt"):
            # GPT: ~1 token per 4 chars, plus 1 for spacing
            return len(text) // 4 + len(text.split())
        elif model in ["claude-3", "claude"]:
            # Claude: ~1 token per 3.5 chars
            return len(text) // 3.5
        else:
            # Generic estimate
            return len(text) // 4

    @staticmethod
    def truncate_to_max_tokens(
        text: str, max_tokens: int, model: str = "gpt-3.5"
    ) -> str:
        """Truncate text to fit within token limit."""
        estimated_chars = max_tokens * 4  # Conservative estimate
        if len(text) <= estimated_chars:
            return text
        return text[:estimated_chars]

    @staticmethod
    def split_into_chunks(
        text: str, chunk_tokens: int = 512, overlap_tokens: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks for processing."""
        # Estimate character count
        chars_per_token = 4
        chunk_chars = chunk_tokens * chars_per_token
        overlap_chars = overlap_tokens * chars_per_token

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunks.append(text[start:end])
            start = end - overlap_chars

        return chunks


# ============================================================================
# BATCHING UTILITIES
# ============================================================================


class BatchingUtils:
    """Efficient batch preparation and packing."""

    @staticmethod
    def pack_sequences(
        sequences: List[str],
        batch_size: int,
        pad_token: str = "</s>",
        strategy: str = "optimal",  # "optimal", "max_length", "simple"
    ) -> List[List[str]]:
        """
        Pack sequences into batches efficiently.

        Strategies:
        - simple: Fixed batch size (may waste space)
        - max_length: Pad all to max length (wastes computation)
        - optimal: Group by length, minimize padding
        """
        if strategy == "simple":
            return [
                sequences[i : i + batch_size]
                for i in range(0, len(sequences), batch_size)
            ]

        elif strategy == "max_length":
            batches = []
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i : i + batch_size]
                batches.append(batch)
            return batches

        elif strategy == "optimal":
            # Sort by length for optimal packing
            indexed = [(i, seq) for i, seq in enumerate(sequences)]
            indexed.sort(key=lambda x: len(x[1]))

            batches = []
            for i in range(0, len(indexed), batch_size):
                batch_items = indexed[i : i + batch_size]
                batch = [seq for _, seq in batch_items]
                batches.append(batch)

            return batches

    @staticmethod
    def calculate_batch_efficiency(batch: List[str]) -> float:
        """Calculate efficiency of batch (non-padding ratio)."""
        if not batch:
            return 0.0

        max_len = max(len(seq) for seq in batch)
        total_len = sum(len(seq) for seq in batch)

        if max_len == 0:
            return 1.0

        # Efficiency: actual_tokens / (max_len * batch_size)
        efficiency = total_len / (max_len * len(batch))
        return efficiency


# ============================================================================
# PROMPT UTILITIES
# ============================================================================


class PromptUtils:
    """Format and manage prompts consistently."""

    # Common prompt templates
    TEMPLATES = {
        "chat": "User: {instruction}\nAssistant: ",
        "instruction": "Instruction: {instruction}\nResponse: ",
        "qa": "Question: {instruction}\nAnswer: ",
        "cot": "Question: {instruction}\nLet's think step by step.\n",
    }

    @staticmethod
    def format_prompt(
        instruction: str,
        template: str = "chat",
        context: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Format prompt with template."""
        prompt = PromptUtils.TEMPLATES.get(template, template)
        prompt = prompt.format(instruction=instruction)

        # Add context if provided
        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        # Add examples if provided (few-shot)
        if examples:
            example_text = ""
            for example_input, example_output in examples:
                example_text += f"Q: {example_input}\nA: {example_output}\n\n"
            prompt = example_text + prompt

        return prompt

    @staticmethod
    def create_fewshot_prompt(
        instruction: str, examples: List[Tuple[str, str]], template: str = "qa"
    ) -> str:
        """Create few-shot prompt with examples."""
        prompt = ""
        for example_input, example_output in examples:
            example_prompt = PromptUtils.TEMPLATES[template].format(
                instruction=example_input
            )
            prompt += f"{example_prompt}{example_output}\n\n"

        # Add the actual query
        query_prompt = PromptUtils.TEMPLATES[template].format(instruction=instruction)
        prompt += query_prompt

        return prompt


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================


class EvaluationUtils:
    """Common evaluation metrics for LLM outputs."""

    @staticmethod
    def calculate_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation).

        ROUGE-N: N-gram overlap
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest common subsequence
        """
        # Simple implementation (real ROUGE is more complex)
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()

        # ROUGE-1 (unigram)
        hyp_set = set(hyp_tokens)
        ref_set = set(ref_tokens)
        overlap = len(hyp_set & ref_set)
        rouge1 = overlap / len(ref_set) if ref_set else 0.0

        # ROUGE-2 (bigram)
        hyp_bigrams = set(zip(hyp_tokens[:-1], hyp_tokens[1:]))
        ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
        bigram_overlap = len(hyp_bigrams & ref_bigrams)
        rouge2 = bigram_overlap / len(ref_bigrams) if ref_bigrams else 0.0

        return {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rouge1,  # Simplified
        }

    @staticmethod
    def calculate_bleu(hypothesis: str, reference: str, n: int = 4) -> float:
        """
        Calculate BLEU score (Bilingual Evaluation Understudy).

        Precision of N-grams weighted by length penalty.
        """
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()

        # Calculate N-gram precision
        score = 0.0
        for gram_size in range(1, min(n + 1, len(hyp_tokens))):
            hyp_ngrams = set(zip(*[hyp_tokens[i:] for i in range(gram_size)]))
            ref_ngrams = set(zip(*[ref_tokens[i:] for i in range(gram_size)]))

            if not hyp_ngrams:
                continue

            overlap = len(hyp_ngrams & ref_ngrams)
            precision = overlap / len(hyp_ngrams)
            score += precision * (1.0 / gram_size)

        # Length penalty
        if len(hyp_tokens) == 0:
            return 0.0

        length_ratio = len(hyp_tokens) / len(ref_tokens)
        penalty = np.exp(1 - 1.0 / length_ratio) if length_ratio < 1 else 1.0

        return score / n * penalty

    @staticmethod
    def calculate_exact_match(prediction: str, reference: str) -> bool:
        """Exact match accuracy."""
        return prediction.strip().lower() == reference.strip().lower()

    @staticmethod
    def calculate_f1(prediction: str, reference: str) -> float:
        """F1 score for token overlap."""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())

        overlap = len(pred_tokens & ref_tokens)

        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return f1


# ============================================================================
# MONITORING UTILITIES
# ============================================================================


@dataclass
class PerformanceMetrics:
    """Track performance metrics over time."""

    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    tokens_generated: int = 0
    batch_size: int = 1
    memory_mb: float = 0.0
    gpu_utilization: float = 0.0


class MonitoringUtils:
    """Monitor and track LLM performance."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = deque(maxlen=window_size)

    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        self.metrics.append(metric)

    def get_average_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if not self.metrics:
            return 0.0
        return np.mean([m.latency_ms for m in self.metrics])

    def get_p95_latency_ms(self) -> float:
        """Get 95th percentile latency."""
        if not self.metrics:
            return 0.0
        latencies = [m.latency_ms for m in self.metrics]
        return np.percentile(latencies, 95)

    def get_p99_latency_ms(self) -> float:
        """Get 99th percentile latency."""
        if not self.metrics:
            return 0.0
        latencies = [m.latency_ms for m in self.metrics]
        return np.percentile(latencies, 99)

    def get_average_throughput(self) -> float:
        """Get average throughput in tokens/second."""
        if not self.metrics:
            return 0.0
        return np.mean([m.tokens_per_second for m in self.metrics])

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "avg_latency_ms": self.get_average_latency_ms(),
            "p95_latency_ms": self.get_p95_latency_ms(),
            "p99_latency_ms": self.get_p99_latency_ms(),
            "avg_throughput_tok_per_sec": self.get_average_throughput(),
            "avg_memory_mb": np.mean([m.memory_mb for m in self.metrics])
            if self.metrics
            else 0.0,
        }


# ============================================================================
# CACHING UTILITIES
# ============================================================================


class CacheUtils:
    """Smart caching strategies for LLM inference."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}

    @staticmethod
    def hash_input(text: str, params: Optional[Dict] = None) -> str:
        """Create hash key for input."""
        key = text
        if params:
            key += json.dumps(params, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache, evicting LRU if needed."""
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]

        self.cache[key] = value
        self.access_times[key] = time.time()

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()


# ============================================================================
# RATE LIMITING UTILITIES
# ============================================================================


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, tokens_per_second: float, bucket_size: Optional[int] = None):
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size or int(tokens_per_second * 10)
        self.tokens = self.bucket_size
        self.last_update = time.time()

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, returning wait time in seconds.

        Returns:
            Time to wait (0 if immediate)
        """
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens from elapsed time
        self.tokens = min(
            self.bucket_size, self.tokens + elapsed * self.tokens_per_second
        )
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0

        # Need to wait
        deficit = tokens - self.tokens
        wait_time = deficit / self.tokens_per_second
        self.tokens = 0
        return wait_time


# Example usage
if __name__ == "__main__":
    print("=== Tokenization Utils ===")
    text = "The quick brown fox jumps over the lazy dog"
    tokens = TokenizationUtils.estimate_tokens(text)
    print(f"Text: '{text}'")
    print(f"Estimated tokens: {tokens}")

    print("\n=== Batching Utils ===")
    sequences = ["short", "medium length text", "this is a longer sequence of text"]
    batches = BatchingUtils.pack_sequences(sequences, batch_size=2, strategy="optimal")
    print(f"Packed into {len(batches)} batches")
    for i, batch in enumerate(batches):
        efficiency = BatchingUtils.calculate_batch_efficiency(batch)
        print(f"  Batch {i + 1}: {len(batch)} sequences, {efficiency:.1%} efficiency")

    print("\n=== Prompt Utils ===")
    prompt = PromptUtils.format_prompt("What is AI?", template="chat")
    print(f"Formatted prompt: {prompt}")

    print("\n=== Evaluation Utils ===")
    hyp = "The cat sat on the mat"
    ref = "A cat was sitting on the mat"
    rouge = EvaluationUtils.calculate_rouge(hyp, ref)
    bleu = EvaluationUtils.calculate_bleu(hyp, ref)
    f1 = EvaluationUtils.calculate_f1(hyp, ref)
    print(f"ROUGE-1: {rouge['rouge1']:.3f}")
    print(f"BLEU: {bleu:.3f}")
    print(f"F1: {f1:.3f}")

    print("\n=== Monitoring Utils ===")
    monitor = MonitoringUtils()
    for i in range(10):
        metric = PerformanceMetrics(
            latency_ms=50 + np.random.randn() * 10,
            tokens_per_second=30 + np.random.randn() * 5,
            tokens_generated=256,
            batch_size=32,
        )
        monitor.record_metric(metric)

    summary = monitor.get_metrics_summary()
    print(f"Average latency: {summary['avg_latency_ms']:.1f}ms")
    print(f"P95 latency: {summary['p95_latency_ms']:.1f}ms")
    print(f"P99 latency: {summary['p99_latency_ms']:.1f}ms")
    print(f"Average throughput: {summary['avg_throughput_tok_per_sec']:.1f} tok/sec")
