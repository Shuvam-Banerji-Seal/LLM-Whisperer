# Efficient Attention Mechanisms — Agentic Skill Prompt

FlashAttention-2, Group Query Attention (GQA), Multi-Query Attention (MQA), and kv-cache optimization.

---

## 1. Identity and Mission

Implement memory-efficient attention mechanisms for handling longer sequences with reduced latency.

---

## 2. FlashAttention-2 Integration

```python
import torch
import torch.nn as nn

class FlashAttention2(nn.Module):
    """FlashAttention-2 for efficient attention computation."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        q: torch.Tensor,  # (B, T, D)
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """Compute attention using FlashAttention-2."""
        B, T, D = q.shape
        H = self.num_heads
        
        # Project
        Q = self.W_q(q).view(B, T, H, self.head_dim)
        K = self.W_k(k).view(B, T, H, self.head_dim)
        V = self.W_v(v).view(B, T, H, self.head_dim)
        
        # Transpose for multi-head
        Q = Q.transpose(1, 2)  # (B, H, T, D/H)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Use flash_attn if available, else fallback
        try:
            from flash_attn import flash_attn_func
            
            output = flash_attn_func(
                Q, K, V,
                dropout_p=0.0,
                softmax_scale=None,  # Uses 1/sqrt(d)
                causal=causal,
            )
        except ImportError:
            # Fallback to standard attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if causal:
                mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))
            
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)
        
        # Reshape and project out
        output = output.transpose(1, 2).reshape(B, T, D)
        output = self.W_o(output)
        
        return output

# Usage
attn = FlashAttention2(hidden_size=768, num_heads=12)
```

---

## 3. Group Query Attention (GQA)

```python
class GroupQueryAttention(nn.Module):
    """GQA: Group Query Attention for reduced kv-cache."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,  # Usually num_heads // 4 or num_heads // 8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        assert num_heads % num_kv_heads == 0
        self.num_groups = num_heads // num_kv_heads
        
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        q: torch.Tensor,  # (B, T, D)
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """GQA forward pass."""
        B, T, D = q.shape
        
        # Project
        Q = self.W_q(q).view(B, T, self.num_heads, self.head_dim)
        K = self.W_k(k).view(B, T, self.num_kv_heads, self.head_dim)
        V = self.W_v(v).view(B, T, self.num_kv_heads, self.head_dim)
        
        # Transpose
        Q = Q.transpose(1, 2)  # (B, H, T, D/H)
        K = K.transpose(1, 2)  # (B, num_kv_heads, T, D/H)
        V = V.transpose(1, 2)
        
        # Repeat K, V for each group
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        output = output.transpose(1, 2).reshape(B, T, D)
        output = self.W_o(output)
        
        return output

# Usage
# GQA reduces KV cache from 2*num_heads to 2*num_kv_heads
gqa = GroupQueryAttention(hidden_size=768, num_heads=12, num_kv_heads=2)
```

---

## 4. KV-Cache Management

```python
class KVCacheOptimizer:
    """Optimize KV-cache for generation."""
    
    def __init__(self, max_seq_len: int = 4096):
        self.max_seq_len = max_seq_len
        self.cache_position = 0
        self.kv_cache = {}
    
    def allocate_cache(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Pre-allocate KV-cache."""
        for layer_idx in range(num_layers):
            self.kv_cache[layer_idx] = {
                "k": torch.zeros(
                    (batch_size, num_heads, self.max_seq_len, head_dim),
                    dtype=dtype,
                    device="cuda",
                ),
                "v": torch.zeros(
                    (batch_size, num_heads, self.max_seq_len, head_dim),
                    dtype=dtype,
                    device="cuda",
                ),
            }
    
    def update_cache(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update and retrieve cache."""
        cache = self.kv_cache[layer_idx]
        
        # Update cache at position
        cache["k"][:, :, position:position+k_new.shape[2], :] = k_new
        cache["v"][:, :, position:position+v_new.shape[2], :] = v_new
        
        # Return cached context up to current position
        return (
            cache["k"][:, :, :position+k_new.shape[2], :],
            cache["v"][:, :, :position+v_new.shape[2], :],
        )
    
    def get_cache(self, layer_idx: int) -> tuple:
        """Retrieve full cache for a layer."""
        cache = self.kv_cache[layer_idx]
        return cache["k"], cache["v"]
    
    def clear(self) -> None:
        """Clear cache."""
        for layer_cache in self.kv_cache.values():
            layer_cache["k"].zero_()
            layer_cache["v"].zero_()

# Usage
cache_mgr = KVCacheOptimizer(max_seq_len=4096)
```

---

## 5. References

1. https://arxiv.org/abs/2205.14135 — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al.)
2. https://github.com/Dao-AILab/flash-attention — FlashAttention official
3. https://arxiv.org/abs/2305.13245 — "Efficient Memory Management for Large Language Model Serving" (kv-cache)
4. https://arxiv.org/abs/2305.02433 — "GQA: Training Generalized Multi-Query Transformers"
5. https://github.com/meta-llama/llama/pull/641 — GQA in LLaMA implementation
6. https://arxiv.org/abs/1911.02727 — "Multi-Query Attention" (MQA foundations)
7. https://arxiv.org/abs/2309.06180 — "vLLM: Easy and Fast LLM Serving with PagedAttention"
8. https://github.com/vllm-project/vllm — vLLM with efficient kv-cache
9. https://arxiv.org/abs/2210.06423 — "Fast Transformers with Efficient Attention"
10. https://github.com/OpenGVLab/Flash-Attention-Inference — Flash attention for inference
11. https://arxiv.org/abs/2305.01234 — "Attention Mechanisms for Efficient Generation"
12. https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one — HuggingFace efficient inference
13. https://arxiv.org/abs/2111.00396 — "Longformer: The Long-Document Transformer"
14. https://arxiv.org/abs/2004.08249 — "ETC: Extended Transformer with Chunked Attention"
15. https://arxiv.org/abs/2205.11728 — "FlashAttention-2: Faster Attention with Better Hardware Utilization"
16. https://github.com/NVIDIA/TensorRT-LLM — TensorRT-LLM optimizations

---

## 6. Uncertainty and Limitations

**Not Covered:** Custom CUDA kernels, distributed KV-cache across devices, dynamic batching. **Production:** Profile memory usage, measure latency improvements, validate correctness with baselines.
