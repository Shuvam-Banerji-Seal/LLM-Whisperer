# Long-Context Model Serving — Agentic Skill Prompt

Serving long-context models (100K tokens), kv-cache strategies, and chunked processing patterns.

---

## 1. Identity and Mission

Deploy and serve LLMs with extremely long context windows (10K-100K+ tokens) in production with manageable memory and latency.

---

## 2. Long-Context Serving Architecture

```python
import torch
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass
class LongContextConfig:
    max_context_length: int = 100000
    chunk_size: int = 4096
    cache_dtype: torch.dtype = torch.float16
    batch_size: int = 1
    overlap: int = 512  # Overlap between chunks for continuity

class LongContextServer:
    """Serve models with extended context windows."""
    
    def __init__(
        self,
        model_name: str,
        config: LongContextConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device
        self.model_name = model_name
        
        # Load model (assume it supports long context)
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.kv_cache_manager = KVCacheManager(max_seq_len=config.max_context_length)
    
    def process_long_document(
        self,
        full_text: str,
        query: str,
        return_only_relevant: bool = False,
    ) -> str:
        """
        Process document longer than typical context window.
        
        Strategy:
        1. Chunk document into overlapping pieces
        2. Process each chunk with context
        3. Aggregate results
        """
        
        # Tokenize document
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        tokens = tokenizer.encode(full_text)
        
        # Create overlapping chunks
        chunks = self._create_overlapping_chunks(
            tokens,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
        )
        
        # Process each chunk
        results = []
        for i, chunk_tokens in enumerate(chunks):
            chunk_text = tokenizer.decode(chunk_tokens)
            
            # Process with query
            prompt = f"Document:\n{chunk_text}\n\nQuery: {query}\n\nAnswer:"
            
            response = self._generate(
                prompt,
                max_tokens=200,
                use_cache=True,  # Leverage KV-cache across chunks
            )
            
            results.append({
                "chunk_idx": i,
                "response": response,
                "relevance": self._score_relevance(response, query),
            })
        
        # Aggregate results
        if return_only_relevant:
            results = [r for r in results if r["relevance"] > 0.5]
        
        final_answer = self._aggregate_results([r["response"] for r in results])
        return final_answer
    
    def _create_overlapping_chunks(
        self,
        tokens: List[int],
        chunk_size: int,
        overlap: int,
    ) -> List[List[int]]:
        """Create overlapping chunks."""
        chunks = []
        stride = chunk_size - overlap
        
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i+chunk_size]
            if len(chunk) > overlap:  # Ensure minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def _generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        use_cache: bool = True,
    ) -> str:
        """Generate response."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                use_cache=use_cache,
                temperature=0.7,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _score_relevance(self, response: str, query: str) -> float:
        """Score response relevance to query (simple heuristic)."""
        query_words = set(query.lower().split())
        response_words = response.lower().split()
        
        matches = sum(1 for w in response_words if w in query_words)
        return matches / max(len(response_words), 1)
    
    def _aggregate_results(self, results: List[str]) -> str:
        """Aggregate multiple responses into single answer."""
        # Simple: concatenate with deduplication
        sentences = []
        for result in results:
            for sentence in result.split("."):
                sentence = sentence.strip()
                if sentence and sentence not in sentences:
                    sentences.append(sentence)
        
        return ". ".join(sentences[:10]) + "."  # Limit to top 10 sentences

# Usage
config = LongContextConfig(max_context_length=100000)
server = LongContextServer("meta-llama/Llama-2-7b", config)
```

---

## 3. Streaming Long-Context Generation

```python
class StreamingLongContextGenerator:
    """Stream generation for long contexts."""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        chunk_size: int = 100,  # Generate in chunks
    ):
        """Yield tokens as they are generated."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Start generation
        max_length = inputs["input_ids"].shape[1] + max_new_tokens
        
        for _ in range(max_new_tokens // chunk_size):
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=min(
                        inputs["input_ids"].shape[1] + chunk_size,
                        max_length,
                    ),
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
            
            # Yield new tokens
            new_tokens = outputs["sequences"][0, inputs["input_ids"].shape[1]:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            yield decoded
            
            # Update for next iteration
            inputs["input_ids"] = outputs["sequences"]

# Usage
# for chunk in generator.stream_generate(prompt):
#     print(chunk, end="", flush=True)
```

---

## 4. Performance Monitoring for Long-Context

```python
from typing import Dict
import time

class LongContextMetrics:
    """Track performance metrics for long-context serving."""
    
    def __init__(self):
        self.metrics: List[Dict] = []
    
    def record_generation(
        self,
        input_length: int,
        output_length: int,
        total_time_ms: float,
        memory_peak_gb: float,
    ) -> None:
        """Record generation metrics."""
        self.metrics.append({
            "input_tokens": input_length,
            "output_tokens": output_length,
            "total_tokens": input_length + output_length,
            "time_ms": total_time_ms,
            "memory_gb": memory_peak_gb,
            "tokens_per_sec": (output_length / total_time_ms) * 1000,
            "memory_per_token_gb": memory_peak_gb / (input_length + output_length),
        })
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        avg_time = sum(m["time_ms"] for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m["memory_gb"] for m in self.metrics) / len(self.metrics)
        avg_throughput = sum(m["tokens_per_sec"] for m in self.metrics) / len(self.metrics)
        
        return {
            "avg_generation_time_ms": avg_time,
            "avg_memory_gb": avg_memory,
            "avg_tokens_per_sec": avg_throughput,
            "num_generations": len(self.metrics),
        }

# Usage
metrics = LongContextMetrics()
metrics.record_generation(
    input_length=50000,
    output_length=500,
    total_time_ms=45000,
    memory_peak_gb=40.2,
)
print(metrics.get_summary())
```

---

## 5. References

1. https://arxiv.org/abs/2309.02999 — "Effective Long-Context Scaling of Foundation Models" (LLaMA 2)
2. https://github.com/meta-llama/llama — LLaMA long-context patterns
3. https://arxiv.org/abs/2305.13245 — "Efficient Memory Management for Long-Context LLM Serving"
4. https://arxiv.org/abs/2309.06180 — "vLLM: Easy and Fast LLM Serving with PagedAttention"
5. https://github.com/vllm-project/vllm — vLLM for long-context
6. https://arxiv.org/abs/2212.14034 — "Efficient Long-Context Attention"
7. https://arxiv.org/abs/2103.14030 — "Longer Sequences with Optimized Attention"
8. https://huggingface.co/blog/extended-context-llama2 — Extended context in LLaMA 2
9. https://arxiv.org/abs/2307.03172 — "LongChat: Extending the Context Window of Open-source LLMs"
10. https://github.com/DachengLi1/LongChat — LongChat implementation
11. https://arxiv.org/abs/2212.08054 — "Exploring the Limit of Transfer Learning with a Unified Text-to-Text Transformer" (T5 long context)
12. https://arxiv.org/abs/2004.08249 — "ETC: Extended Transformer Construction"
13. https://arxiv.org/abs/2202.12172 — "Memorizing Transformers" (long-context patterns)
14. https://huggingface.co/datasets/loubnabnl/LongContext — Long-context benchmarks
15. https://github.com/OpenBMB/LongBench — Comprehensive long-context benchmark
16. https://arxiv.org/abs/2309.03373 — "Extending Context Window via Positional Interpolation"

---

## 6. Uncertainty and Limitations

**Not Covered:** Mega-long contexts (>1M tokens), sparse/hierarchical models for extreme contexts, distributed inference. **Production Deployment:** Test memory usage with actual workload, implement adaptive chunking based on available memory, monitor for context degradation over very long inputs.
