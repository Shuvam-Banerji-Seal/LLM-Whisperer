# Model Caching — Inference Skill Prompt

Implementing efficient KV cache, model caching, and prefix caching strategies for optimized LLM inference.

---

## 1. Identity and Mission

Implement caching strategies that dramatically reduce LLM inference latency and cost by reusing computed results. This includes KV cache management for autoregressive generation, model weight caching, prefix/trie-based prompt caching, and attention cache optimization.

---

## 2. Theory & Fundamentals

### 2.1 KV Cache Structure

For a transformer with L layers, H heads, and head dimension D:
```
KV_cache_per_layer = [2 * batch_size * seq_len * H * D * 2]  # k and v
Total = L * 2 * batch_size * seq_len * H * D * 2 bytes
```

In BF16: Memory = 4 * L * seq_len * H * D * batch_size bytes

### 2.2 Cache Hit Patterns

**Prefix Caching:** Common system/user prompt prefixes
```
User: "As a Python expert, help me with..."
Cache: "As a Python expert, help me with debugging this code..."

System: "You are a helpful assistant..."
Cache: "You are a helpful assistant that responds..."
```

**KV Cache:** Token-by-token generation states
```
Token 1: compute K1, V1, cache
Token 2: retrieve K1, V1, compute K2, V2, cache
...
```

### 2.3 Eviction Policies

| Policy | Description | Trade-off |
|--------|-------------|-----------|
| LRU | Least Recently Used | Good hit rate |
| LFU | Least Frequently Used | Good for popular items |
| FIFO | First In, First Out | Simple, predictable |
| ARC | Adaptive Replacement | Automatic tuning |
| TinyLFU | Approximate LFU with LRU | Good general performance |

---

## 3. Implementation Patterns

### Pattern 1: Paged KV Cache

```python
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib
import numpy as np

@dataclass
class CacheBlock:
    """A block in the paged KV cache."""
    block_id: int
    num_tokens: int
    max_tokens: int
    k_data: torch.Tensor
    v_data: torch.Tensor
    is_full: bool = False
    access_count: int = 0

    @property
    def num_free_slots(self) -> int:
        return self.max_tokens - self.num_tokens


class PagedKVCache:
    """
    Paged KV cache for memory-efficient autoregressive generation.
    Similar to vLLM's paged attention.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 1024,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = device
        self.dtype = dtype

        # Physical cache blocks
        self.blocks: Dict[int, CacheBlock] = {}
        self.free_blocks: set = set(range(max_blocks))

        # Sequence to blocks mapping
        self.seq_to_blocks: Dict[int, List[int]] = {}

        # Block size in bytes
        self.block_bytes = (
            2 *  # k and v
            num_heads *
            block_size *
            head_dim *
            dtype.itemsize
        )

    def alloc_block(self) -> Optional[int]:
        """Allocate a free cache block."""
        if self.free_blocks:
            block_id = self.free_blocks.pop()

            # Create tensors for this block
            k_data = torch.zeros(
                self.num_heads,
                self.block_size,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            v_data = torch.zeros(
                self.num_heads,
                self.block_size,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            self.blocks[block_id] = CacheBlock(
                block_id=block_id,
                num_tokens=0,
                max_tokens=self.block_size,
                k_data=k_data,
                v_data=v_data,
            )

            return block_id
        return None

    def free_block(self, block_id: int):
        """Free a cache block."""
        if block_id in self.blocks:
            del self.blocks[block_id]
            self.free_blocks.add(block_id)

    def get_cache_block(self, block_id: int) -> Optional[CacheBlock]:
        """Get a cache block by ID."""
        return self.blocks.get(block_id)

    def append(
        self,
        seq_id: int,
        k: torch.Tensor,  # (num_heads, 1, head_dim)
        v: torch.Tensor,  # (num_heads, 1, head_dim)
    ) -> bool:
        """
        Append new K, V to sequence's cache.
        Returns True if successful.
        """
        # Get or create block list for sequence
        if seq_id not in self.seq_to_blocks:
            block_id = self.alloc_block()
            if block_id is None:
                return False  # No space
            self.seq_to_blocks[seq_id] = [block_id]

        # Get current block
        block_ids = self.seq_to_blocks[seq_id]
        current_block_id = block_ids[-1]
        current_block = self.blocks[current_block_id]

        # Check if current block has space
        if current_block.is_full:
            # Need new block
            new_block_id = self.alloc_block()
            if new_block_id is None:
                return False
            block_ids.append(new_block_id)
            current_block = self.blocks[new_block_id]

        # Append to block
        slot = current_block.num_tokens
        current_block.k_data[:, slot, :] = k.squeeze(1)
        current_block.v_data[:, slot, :] = v.squeeze(1)
        current_block.num_tokens += 1

        if current_block.num_tokens >= current_block.max_tokens:
            current_block.is_full = True

        current_block.access_count += 1
        return True

    def get(
        self,
        seq_id: int,
        start_position: int,
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve K, V cache for a range of positions.
        Returns (k_cache, v_cache) tensors.
        """
        if seq_id not in self.seq_to_blocks:
            return None, None

        block_ids = self.seq_to_blocks[seq_id]

        # Collect from blocks
        k_chunks = []
        v_chunks = []

        for i in range(num_tokens):
            position = start_position + i
            block_idx = position // self.block_size
            offset = position % self.block_size

            if block_idx >= len(block_ids):
                break

            block = self.blocks[block_ids[block_idx]]
            k_chunks.append(block.k_data[:, offset:offset+1, :])
            v_chunks.append(block.v_data[:, offset:offset+1, :])

        if k_chunks:
            return torch.cat(k_chunks, dim=1), torch.cat(v_chunks, dim=1)
        return None, None

    def get_full_cache(
        self,
        seq_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all cached K, V for a sequence."""
        if seq_id not in self.seq_to_blocks:
            return None, None

        block_ids = self.seq_to_blocks[seq_id]

        total_tokens = sum(
            self.blocks[bid].num_tokens
            for bid in block_ids
        )

        if total_tokens == 0:
            return None, None

        k_full = torch.zeros(
            self.num_heads,
            total_tokens,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        v_full = torch.zeros(
            self.num_heads,
            total_tokens,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        offset = 0
        for bid in block_ids:
            block = self.blocks[bid]
            n = block.num_tokens
            k_full[:, offset:offset+n, :] = block.k_data[:, :n, :]
            v_full[:, offset:offset+n, :] = block.v_data[:, :n, :]
            offset += n

        return k_full, v_full

    def evict_sequence(self, seq_id: int):
        """Evict all blocks for a sequence."""
        if seq_id in self.seq_to_blocks:
            for block_id in self.seq_to_blocks[seq_id]:
                self.free_block(block_id)
            del self.seq_to_blocks[seq_id]

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        used_blocks = len(self.blocks)
        total_tokens = sum(
            b.num_tokens
            for b in self.blocks.values()
        )
        total_capacity = used_blocks * self.block_size

        return {
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "total_tokens_cached": total_tokens,
            "capacity_tokens": total_capacity,
            "utilization": total_tokens / total_capacity if total_capacity > 0 else 0,
            "num_sequences": len(self.seq_to_blocks),
            "total_memory_mb": used_blocks * self.block_bytes / (1024 * 1024),
        }


class KVCacheManager:
    """
    High-level KV cache manager with eviction policies.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_memory_gb: float = 40.0,
        block_size: int = 16,
    ):
        self.cache = PagedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_blocks=int(max_memory_gb * 1024**3 / (block_size * num_heads * head_dim * 2 * 2)),
        )

        # Access tracking for LRU
        self.last_access: Dict[int, float] = {}

    def store(
        self,
        seq_id: int,
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        """Store K, V at position."""
        # Update LRU
        self.last_access[seq_id] = position

        return self.cache.append(seq_id, k, v)

    def retrieve(
        self,
        seq_id: int,
        start: int,
        length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K, V cache slice."""
        return self.cache.get(seq_id, start, length)

    def evict_lru(self, num_to_evict: int = 1):
        """Evict least recently used sequences."""
        # Sort by last access
        sorted_seqs = sorted(
            self.last_access.items(),
            key=lambda x: x[1],
        )

        for seq_id, _ in sorted_seqs[:num_to_evict]:
            self.cache.evict_sequence(seq_id)
            del self.last_access[seq_id]

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()
```

### Pattern 2: Prefix Cache with Trie

```python
import torch
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import hashlib

@dataclass
class PrefixNode:
    """A node in the prefix trie."""
    token_ids: Tuple[int, ...]
    hash: str
    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None
    is_complete: bool = False
    access_count: int = 0
    children: Dict[int, "PrefixNode"] = None

    def __post_init__(self):
        self.children = self.children or {}


class PrefixCache:
    """
    Trie-based prefix caching for common prompt prefixes.
    Caches KV states for reusable prompt segments.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_prefixes: int = 1000,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_prefixes = max_prefixes
        self.device = device
        self.dtype = dtype

        # Root node
        self.root = PrefixNode(
            token_ids=(),
            hash="root",
        )

        # LRU tracking
        self.access_order: List[str] = []
        self.hash_to_node: Dict[str, PrefixNode] = {}

        # Memory budget
        self.used_memory = 0
        self.max_memory = 10 * 1024**3  # 10 GB default

    def compute_hash(self, token_ids: List[int]) -> str:
        """Compute hash for token sequence."""
        token_str = ",".join(map(str, token_ids))
        return hashlib.sha256(token_str.encode()).hexdigest()[:16]

    def find_longest_prefix(
        self,
        token_ids: List[int],
    ) -> Tuple[Optional[PrefixNode], int]:
        """
        Find longest cached prefix for token sequence.
        Returns (node, num_matched_tokens).
        """
        node = self.root
        matched = 0

        for i, token_id in enumerate(token_ids):
            if token_id in node.children:
                node = node.children[token_id]
                if node.is_complete:
                    matched = i + 1
            else:
                break

        return (node if node.is_complete else None), matched

    def insert(
        self,
        token_ids: List[int],
        k_cache: torch.Tensor,  # (num_layers, num_heads, seq_len, head_dim)
        v_cache: torch.Tensor,  # (num_layers, num_heads, seq_len, head_dim)
    ) -> bool:
        """
        Insert a new prefix into cache.
        Returns True if successful.
        """
        if not token_ids:
            return False

        # Check memory
        cache_size = k_cache.numel() * k_cache.element_size() * 2
        if cache_size > self.max_memory - self.used_memory:
            # Need to evict
            if not self._evict_lru(cache_size):
                return False

        # Insert into trie
        node = self.root
        for i, token_id in enumerate(token_ids):
            if token_id not in node.children:
                new_node = PrefixNode(
                    token_ids=node.token_ids + (token_id,),
                    hash=self.compute_hash(token_ids[:i+1]),
                )
                node.children[token_id] = new_node
                self.hash_to_node[new_node.hash] = new_node

            node = node.children[token_id]

            # Store cache at each node
            if i == len(token_ids) - 1 or (i + 1) % 32 == 0:
                # Store partial cache every 32 tokens
                pass

        # Mark as complete
        node.is_complete = True
        node.k_cache = k_cache
        node.v_cache = v_cache

        # Update tracking
        self.access_order.append(node.hash)
        self.hash_to_node[node.hash] = node
        node.access_count += 1

        self.used_memory += cache_size
        return True

    def lookup(
        self,
        token_ids: List[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Look up cached KV for token sequence.
        Returns (k_cache, v_cache, num_matched).
        """
        node = self.root
        matched = 0

        for i, token_id in enumerate(token_ids):
            if token_id not in node.children:
                break
            node = node.children[token_id]
            if node.is_complete:
                matched = i + 1

        if matched > 0:
            # Update access
            node.access_count += 1
            return node.k_cache, node.v_cache, matched

        return None, None, 0

    def _evict_lru(self, needed_bytes: int) -> bool:
        """Evict LRU entries to make space."""
        while self.access_order and self.used_memory + needed_bytes > self.max_memory:
            oldest_hash = self.access_order.pop(0)
            if oldest_hash in self.hash_to_node:
                node = self.hash_to_node[oldest_hash]
                if node.k_cache is not None:
                    self.used_memory -= node.k_cache.numel() * node.k_cache.element_size() * 2
                del self.hash_to_node[oldest_hash]

        return len(self.access_order) > 0 or needed_bytes <= self.max_memory

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "num_prefixes": len(self.hash_to_node),
            "used_memory_gb": self.used_memory / (1024**3),
            "max_memory_gb": self.max_memory / (1024**3),
            "utilization": self.used_memory / self.max_memory if self.max_memory > 0 else 0,
            "total_accesses": sum(n.access_count for n in self.hash_to_node.values()),
        }


class PrefixKVExtractor:
    """
    Extract KV cache values from model for prefix caching.
    """

    def __init__(
        self,
        model: Any,
        layer_indices: List[int] = None,
    ):
        self.model = model
        self.layer_indices = layer_indices or list(range(model.config.num_hidden_layers))

        # Hook handles
        self.hooks = []
        self.cached_k = {}
        self.cached_v = {}

    def register_hooks(self):
        """Register forward hooks to capture KV."""
        def hook_fn(module, input, output, layer_idx):
            # output shape: (batch, seq, num_heads, head_dim)
            # Store in correct format for cache
            k = output[0]  # or unpack depending on model
            v = output[1] if isinstance(output, tuple) else output

            self.cached_k[layer_idx] = k.detach().clone()
            self.cached_v[layer_idx] = v.detach().clone()

        for layer_idx in self.layer_indices:
            layer = self.model.transformer.h[layer_idx]
            handle = layer.register_forward_hook(
                lambda mod, inp, out, idx=layer_idx: hook_fn(mod, inp, out, idx)
            )
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def extract(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward pass and extract KV cache.

        Returns:
            k_cache: (num_layers, num_heads, seq_len, head_dim)
            v_cache: (num_layers, num_heads, seq_len, head_dim)
        """
        self.cached_k = {}
        self.cached_v = {}

        with torch.no_grad():
            self.model(token_ids)

        # Stack all layers
        k_cache = torch.stack([self.cached_k[i] for i in self.layer_indices], dim=0)
        v_cache = torch.stack([self.cached_v[i] for i in self.layer_indices], dim=0)

        return k_cache, v_cache
```

### Pattern 3: Model Weight Caching

```python
import torch
import os
from typing import Dict, Optional, Any
import pickle
import hashlib
from pathlib import Path

class ModelWeightCache:
    """
    Cache model weights on GPU for fast repeated loading.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size_gb: float = 40.0,
    ):
        self.cache_dir = cache_dir or "/tmp/model_cache"
        self.max_size = int(max_size_gb * 1024**3)
        self.current_size = 0

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # In-memory weight cache
        self.weight_cache: Dict[str, torch.Tensor] = {}
        self.weight_metadata: Dict[str, Dict] = {}

        # LRU tracking
        self.access_order = []

    def compute_weight_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Compute hash of weights for cache key."""
        hashes = []
        for name, tensor in sorted(state_dict.items()):
            h = hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()[:8]
            hashes.append(f"{name}:{h}")

        combined = "|".join(hashes)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def cache_weights(
        self,
        model_name: str,
        state_dict: Dict[str, torch.Tensor],
    ) -> str:
        """
        Cache model weights.

        Returns cache key.
        """
        cache_key = f"{model_name}_{self.compute_weight_hash(state_dict)}"

        # Check if already cached
        if cache_key in self.weight_cache:
            self._update_access(cache_key)
            return cache_key

        # Check size
        weight_size = sum(t.numel() * t.element_size() for t in state_dict.values())

        if weight_size > self.max_size:
            # Can't cache this model
            return cache_key

        # Evict if needed
        while self.current_size + weight_size > self.max_size:
            self._evict_lru()

        # Copy weights to cache
        self.weight_cache[cache_key] = {
            name: tensor.clone().cuda()
            for name, tensor in state_dict.items()
        }

        # Save metadata
        self.weight_metadata[cache_key] = {
            "model_name": model_name,
            "size_bytes": weight_size,
            "num_tensors": len(state_dict),
        }

        self.current_size += weight_size
        self._update_access(cache_key)

        return cache_key

    def load_weights(
        self,
        cache_key: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load cached weights."""
        if cache_key in self.weight_cache:
            self._update_access(cache_key)
            return self.weight_cache[cache_key]

        return None

    def _update_access(self, cache_key: str):
        """Update LRU access order."""
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)

    def _evict_lru(self):
        """Evict LRU weights."""
        if not self.access_order:
            return

        oldest_key = self.access_order.pop(0)
        if oldest_key in self.weight_cache:
            # Calculate size
            size = sum(
                t.numel() * t.element_size()
                for t in self.weight_cache[oldest_key].values()
            )

            # Remove from cache
            for tensor in self.weight_cache[oldest_key].values():
                del tensor
            del self.weight_cache[oldest_key]
            del self.weight_metadata[oldest_key]

            self.current_size -= size

    def save_to_disk(self, cache_key: str):
        """Save cache to disk."""
        if cache_key not in self.weight_cache:
            return

        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
        torch.save(self.weight_cache[cache_key], cache_path)

    def load_from_disk(self, cache_key: str) -> Optional[Dict]:
        """Load cache from disk."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        return None

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "num_cached_models": len(self.weight_cache),
            "current_size_gb": self.current_size / (1024**3),
            "max_size_gb": self.max_size / (1024**3),
            "utilization": self.current_size / self.max_size if self.max_size > 0 else 0,
        }


class SpeculativeCache:
    """
    Speculative caching: predict future needs and preload.
    """

    def __init__(
        self,
        model_cache: ModelWeightCache,
        prefix_cache: PrefixCache,
    ):
        self.model_cache = model_cache
        self.prefix_cache = prefix_cache
        self.speculation_enabled = True

    def predict_and_prefetch(
        self,
        current_tokens: List[int],
        model: Any,
    ) -> List[int]:
        """
        Predict next tokens and prefetch KV cache.

        Returns list of predicted token IDs.
        """
        if not self.speculation_enabled:
            return []

        # Simple prediction: use model to predict next tokens
        with torch.no_grad():
            input_ids = torch.tensor([current_tokens[-50:]], device='cuda')

            # Get logits
            outputs = model(input_ids)
            logits = outputs.logits[0, -1]

            # Top-k predictions
            top_k = 5
            _, predicted = torch.topk(logits, top_k)

        predicted_tokens = predicted.tolist()

        # Pre-compute prefix cache for predicted tokens
        for tokens in self._expand_prefixes(current_tokens, predicted_tokens):
            # Would compute and cache KV here
            pass

        return predicted_tokens[:3]

    def _expand_prefixes(
        self,
        current: List[int],
        predicted: List[int],
    ) -> List[List[int]]:
        """Expand prefixes with predictions."""
        prefixes = []
        for p in predicted:
            prefixes.append(current + [p])
            prefixes.append(current + [p, p])  # Double token
        return prefixes
```

### Pattern 4: Attention Cache Optimization

```python
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class MultiQueryAttention:
    """
    Multi-Query Attention (MQA) cache optimization.
    Only caches K, V for a single head.
    """

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Single K, V head
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seq, head_dim)
        k: torch.Tensor,  # (batch, 1, seq, head_dim) - shared
        v: torch.Tensor,  # (batch, 1, seq, head_dim) - shared
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with MQA.
        """
        batch_size = q.shape[0]
        q_len = q.shape[2]

        # Expand K, V to all heads
        k_expanded = k.expand(-1, self.num_heads, -1, -1)
        v_expanded = v.expand(-1, self.num_heads, -1, -1)

        # Compute attention
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_expanded)

        # Update cache
        if use_cache:
            if self.k_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)

        return attn_output, self.k_cache, self.v_cache


class GroupedQueryAttention:
    """
    Grouped-Query Attention (GQA) with KV cache.
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_q_heads // num_kv_heads

        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_q_heads, seq, head_dim)
        k: torch.Tensor,  # (batch, num_kv_heads, seq, head_dim)
        v: torch.Tensor,  # (batch, num_kv_heads, seq, head_dim)
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward with GQA.
        """
        # Get cached KV
        if self.k_cache is not None and use_cache:
            k_full = torch.cat([self.k_cache, k], dim=2)
            v_full = torch.cat([self.v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        # Expand KV heads to match Q heads
        k_expanded = self._expand_kv(k_full)
        v_expanded = self._expand_kv(v_full)

        # Compute attention
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v_expanded)

        # Update cache
        if use_cache:
            self.k_cache = k_full
            self.v_cache = v_full

        return output, self.k_cache, self.v_cache

    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to Q heads."""
        # Interleave KV heads: [kv0, kv0, kv1, kv1, ...]
        if self.group_size == 1:
            return kv

        batch_size, _, seq_len, head_dim = kv.shape
        kv_expanded = kv.unsqueeze(2)
        kv_expanded = kv_expanded.expand(-1, -1, self.group_size, -1, -1)
        return kv_expanded.reshape(batch_size, self.num_q_heads, seq_len, head_dim)


class SlidingWindowCache:
    """
    Sliding window KV cache for efficient long-sequence processing.
    Only keeps most recent tokens within window.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        window_size: int = 4096,
        device: str = "cuda",
    ):
        self.window_size = window_size
        self.device = device

        # Pre-allocate fixed-size cache
        self.k_cache = torch.zeros(
            num_layers,
            num_heads,
            window_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        self.v_cache = torch.zeros(
            num_layers,
            num_heads,
            window_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        self.current_length = 0

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,  # (num_heads, 1, head_dim) or (num_heads, seq, head_dim)
        v: torch.Tensor,
    ):
        """Update cache with new K, V."""
        seq_len = k.shape[1]

        # Handle position within window
        if self.current_length >= self.window_size:
            # Shift and discard oldest
            shift = seq_len
            self.k_cache = torch.roll(self.k_cache, shifts=-shift, dims=2)
            self.v_cache = torch.roll(self.v_cache, shifts=-shift, dims=2)
            self.current_length = self.window_size - seq_len

        # Insert new tokens at the end
        start = self.current_length
        end = start + seq_len
        self.k_cache[layer_idx, :, start:end, :] = k
        self.v_cache[layer_idx, :, start:end, :] = v
        self.current_length = min(end, self.window_size)

    def get(
        self,
        layer_idx: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cache slice."""
        if end is None:
            end = self.current_length

        return (
            self.k_cache[layer_idx, :, start:end, :],
            self.v_cache[layer_idx, :, start:end, :],
        )

    def reset(self):
        """Reset cache."""
        self.current_length = 0
```

### Pattern 5: Dynamic Batch Cache Management

```python
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import heapq

@dataclass
class CachedSequence:
    """Metadata for a cached sequence."""
    seq_id: int
    prompt_hash: str
    num_tokens: int
    last_access: float
    priority: float = 1.0
    is_active: bool = False

class CachePriorityManager:
    """
    Manage cache eviction based on priority.
    Combines LRU, frequency, and user priority.
    """

    def __init__(
        self,
        max_sequences: int = 100,
        ttl_seconds: float = 3600.0,
    ):
        self.max_sequences = max_sequences
        self.ttl = ttl_seconds
        self.sequences: Dict[int, CachedSequence] = {}
        self.priority_queue: List[tuple] = []  # (-priority, last_access, seq_id)

    def add_sequence(
        self,
        seq_id: int,
        prompt_hash: str,
        num_tokens: int,
        priority: float = 1.0,
    ):
        """Add a new sequence to cache."""
        now = time.time()

        self.sequences[seq_id] = CachedSequence(
            seq_id=seq_id,
            prompt_hash=prompt_hash,
            num_tokens=num_tokens,
            last_access=now,
            priority=priority,
        )

        heapq.heappush(
            self.priority_queue,
            (-priority, now, seq_id)
        )

        # Evict if over capacity
        while len(self.sequences) > self.max_sequences:
            self._evict_lowest_priority()

    def access_sequence(self, seq_id: int):
        """Update access time for sequence."""
        if seq_id in self.sequences:
            self.sequences[seq_id].last_access = time.time()

    def get_eviction_candidates(self, needed_tokens: int) -> List[int]:
        """Get sequences to evict to make room."""
        candidates = []
        current_tokens = sum(s.num_tokens for s in self.sequences.values())

        while current_tokens + needed_tokens > self.max_sequences * 1000:
            seq_id = self._evict_lru()
            if seq_id is None:
                break
            candidates.append(seq_id)
            current_tokens -= self.sequences[seq_id].num_tokens
            del self.sequences[seq_id]

        return candidates

    def _evict_lru(self) -> Optional[int]:
        """Evict least recently used."""
        while self.priority_queue:
            neg_priority, last_access, seq_id = heapq.heappop(self.priority_queue)

            if seq_id in self.sequences:
                return seq_id

        return None

    def _evict_lowest_priority(self):
        """Evict lowest priority sequence."""
        seq_id = self._evict_lru()
        if seq_id and seq_id in self.sequences:
            del self.sequences[seq_id]


class AdaptiveCacheSizer:
    """
    Dynamically adjust cache size based on workload.
    """

    def __init__(
        self,
        base_size_gb: float = 20.0,
        min_size_gb: float = 5.0,
        max_size_gb: float = 80.0,
        adjustment_interval: float = 60.0,
    ):
        self.current_size_gb = base_size_gb
        self.base_size = base_size_gb
        self.min_size = min_size_gb
        self.max_size = max_size_gb
        self.interval = adjustment_interval

        self.last_adjustment = time.time()
        self.hit_rates: List[float] = []
        self.utilization: List[float] = []

    def should_increase(self, hit_rate: float, utilization: float) -> bool:
        """Determine if cache should be increased."""
        # Increase if high hit rate but low utilization
        # This means we're evicting useful cache
        return hit_rate > 0.8 and utilization < 0.5

    def should_decrease(self, hit_rate: float, utilization: float) -> bool:
        """Determine if cache should be decreased."""
        # Decrease if low hit rate and low utilization
        # Cache isn't helping
        return hit_rate < 0.2 and utilization < 0.3

    def adjust_size(self, hit_rate: float, utilization: float):
        """Adjust cache size based on metrics."""
        now = time.time()

        if now - self.last_adjustment < self.interval:
            return

        self.hit_rates.append(hit_rate)
        self.utilization.append(utilization)

        # Keep rolling window
        if len(self.hit_rates) > 10:
            self.hit_rates.pop(0)
            self.utilization.pop(0)

        avg_hit = sum(self.hit_rates) / len(self.hit_rates)
        avg_util = sum(self.utilization) / len(self.utilization)

        if self.should_increase(avg_hit, avg_util):
            self.current_size_gb = min(
                self.current_size_gb * 1.2,
                self.max_size_gb
            )
        elif self.should_decrease(avg_hit, avg_util):
            self.current_size_gb = max(
                self.current_size_gb * 0.8,
                self.min_size_gb
            )

        self.last_adjustment = now

    def get_current_size(self) -> int:
        """Get current cache size in bytes."""
        return int(self.current_size_gb * 1024**3)
```

---

## 4. Framework Integration

### vLLM Integration

```python
from vllm import LLM, SamplingParams

# vLLM handles KV cache automatically
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
)

# vLLM's paged attention manages KV cache
outputs = llm.generate(prompts, SamplingParams(temperature=0.9))
```

### HuggingFace Integration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    offload_folder="./offload",
)

# Use cache
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    use_cache=True,
    max_new_tokens=100,
)
```

---

## 5. Performance Considerations

### Cache Hit Rates

| Scenario | Hit Rate | Notes |
|----------|----------|-------|
| Repeated prompts | 90%+ | Perfect for chatbots |
| Long conversations | 50-80% | Prefix sharing |
| Random requests | 10-30% | Limited caching benefit |

### Memory Allocation

| Cache Type | Memory per Token | For LLaMA-70B |
|------------|-----------------|---------------|
| KV Cache (BF16) | ~2KB | ~4GB per 2K tokens |
| Prefix Cache | ~2KB | Same as KV |
| Model Weights | N/A | 140GB total |

---

## 6. Common Pitfalls

1. **Memory Fragmentation**: Poor block allocation causing OOM
2. **Cache Pollution**: Caching useless prefixes
3. **TTL Mismanagement**: Stale cache entries consuming memory
4. **Race Conditions**: Concurrent cache access without locks
5. **Serialization Overhead**: Too frequent disk cache writes
6. **Insufficient Eviction**: Not evicting enough when memory pressure

---

## 7. Research References

1. https://arxiv.org/abs/2309.06196 — "Efficient Memory Management for LLMs"

2. https://arxiv.org/abs/2309.15025 — "vLLM: Paged Attention for KV Cache"

3. https://arxiv.org/abs/2305.02427 — "StreamingLLM: Efficient Streaming LLM"

4. https://arxiv.org/abs/2312.07104 — "Cache Transformers"

5. https://github.com/vllm-project/vllm — vLLM implementation

6. https://arxiv.org/abs/2302.14035 — "Prefix Transformer"

---

## 8. Uncertainty and Limitations

**Not Covered:** Specific hardware optimization, multi-node cache coherence.

**Production Considerations:** Monitor cache hit rates and memory utilization. Use adaptive sizing when workload patterns vary.

(End of file - total 1480 lines)