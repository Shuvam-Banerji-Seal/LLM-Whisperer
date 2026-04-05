# Position Encodings for Long Context — Agentic Skill Prompt

RoPE, ALiBi, Linear scaling, YaRN, and position extrapolation for 10K-100K context windows.

---

## 1. Identity and Mission

Enable LLMs to handle extended contexts (10K-100K+ tokens) through efficient positional encoding schemes.

---

## 2. RoPE (Rotary Position Embeddings)

```python
import torch
import math

class RotaryPositionalEmbedding:
    """RoPE (Rotary Position Embedding) for efficient long-context."""
    
    def __init__(self, dim: int, base: float = 10000.0, device: str = "cuda"):
        self.dim = dim
        self.base = base
        self.device = device
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """Apply RoPE to tensor."""
        seq_len = seq_len or x.shape[1]
        t = torch.arange(seq_len, device=self.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Compute cos and sin
        cos_emb = torch.cos(freqs).to(x.dtype)
        sin_emb = torch.sin(freqs).to(x.dtype)
        
        # Apply rotation
        def rotate_half(x):
            return torch.cat(
                (-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]),
                dim=-1,
            )
        
        rotated = x * cos_emb + rotate_half(x) * sin_emb
        return rotated
    
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Register as buffer (not learned parameter)."""
        setattr(self, name, tensor)

# Usage
rope = RotaryPositionalEmbedding(dim=128)
# In attention computation: apply rope to Q and K before attention
```

---

## 3. Position Scaling Methods

```python
class PositionScaling:
    """Scale position embeddings for longer contexts."""
    
    @staticmethod
    def linear_scaling(position: int, scale_factor: float) -> int:
        """Linear scaling: pos' = pos / scale_factor."""
        return int(position / scale_factor)
    
    @staticmethod
    def yarn_scaling(
        position: int,
        dim: int,
        scale: float = 1.0,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """YaRN (Yet another RoPE extension method)."""
        # Dimension-dependent scaling
        # Lower dimensions scale more, higher dimensions less
        
        freqs = 10000 ** (-torch.arange(0, dim, 2).float() / dim)
        
        # Compute scaling factor per dimension
        inv_freq_scale = scale_factor = torch.ones_like(freqs)
        
        # Higher frequency components scale more aggressively
        factor = (position - 1) * scale
        scaled_freqs = freqs / (
            1 + (torch.arange(0, dim, 2).float() / dim) ** alpha * factor
        )
        
        return scaled_freqs
    
    @staticmethod
    def nope_scaling(position: int, max_position: int) -> float:
        """No-op scaling: keep original (baseline)."""
        return 1.0

# Usage
# For RoPE with scaling:
# scaled_pos = PositionScaling.linear_scaling(position, scale_factor=2.0)
```

---

## 4. ALiBi (Attention with Linear Biases)

```python
class ALiBiAttention(torch.nn.Module):
    """Attention with Linear Biases for position-aware attention."""
    
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.slopes = self._compute_slopes(num_heads)
    
    def _compute_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes."""
        # Powers of 2: 1/2, 1/4, 1/8, ...
        m = torch.arange(1, num_heads + 1, dtype=torch.float32)
        return 2.0 ** (-5.0 * m / num_heads)
    
    def forward(
        self,
        q: torch.Tensor,  # (B, T, D)
        k: torch.Tensor,  # (B, T, D)
        v: torch.Tensor,  # (B, T, D)
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute ALiBi attention."""
        B, T, D = q.shape
        H = self.num_heads
        
        # Reshape for multi-head attention
        q = q.reshape(B, T, H, -1).transpose(1, 2)  # (B, H, T, D/H)
        k = k.reshape(B, T, H, -1).transpose(1, 2)
        v = v.reshape(B, T, H, -1).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // H)
        
        # Add ALiBi bias (position-dependent)
        bias = torch.arange(T, device=q.device, dtype=q.dtype)
        bias = bias.unsqueeze(0) - bias.unsqueeze(1)  # (T, T)
        
        # Apply slopes per head
        slopes = self.slopes.to(q.device).view(1, H, 1, 1)
        alibi_bias = slopes * bias.unsqueeze(0).unsqueeze(0)
        
        scores = scores + alibi_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Attention
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).reshape(B, T, D)
        
        return output

# Usage
# alibi = ALiBiAttention(num_heads=8)
# output = alibi(q, k, v)
```

---

## 5. References

1. https://arxiv.org/abs/2104.09864 — "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE paper)
2. https://github.com/ZhuiyiTechnology/roformer — RoPE implementation
3. https://arxiv.org/abs/2108.12409 — "Attention with Linear Biases Enables Input Length Extrapolation" (ALiBi)
4. https://github.com/ofirpress/attention_with_linear_biases — ALiBi implementation
5. https://arxiv.org/abs/2309.16039 — "YaRN: Efficient Context Window Extension of Large Language Models" (YaRN)
6. https://github.com/jquesnelle/yarn — YaRN implementation
7. https://arxiv.org/abs/2307.17934 — "Extending Context Windows of Large Language Models"
8. https://arxiv.org/abs/2309.02999 — "Effective Long-Context Scaling of Foundation Models" (LLaMA 2 long context)
9. https://github.com/meta-llama/llama/blob/main/llama/model.py — LLaMA 2 long context patterns
10. https://arxiv.org/abs/2006.16236 — "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
11. https://arxiv.org/abs/2012.14905 — "Compressive Transformers for Long-Range Sequence Modelling"
12. https://arxiv.org/abs/2212.14034 — "Efficient Long-Context Attention"
13. https://github.com/OpenBMB/LongBench — Long-context benchmark
14. https://huggingface.co/datasets/chinwong/LongContext100K — 100K context dataset
15. https://arxiv.org/abs/2309.03373 — "Position Interpolation for Long-Context Scaling"
16. https://github.com/jquesnelle/transformers/blob/yarn/src/transformers/models/llama/modeling_llama.py — Practical implementation

---

## 6. Uncertainty and Limitations

**Not Covered:** Sparse attention for mega-long contexts (>1M tokens), chunk-based processing. **Production:** Benchmark memory usage with actual model, test on diverse long-document tasks.
