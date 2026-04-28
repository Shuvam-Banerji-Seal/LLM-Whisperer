# Sparse Transformers and Mixture of Experts

## Problem Statement

As language models scale to billions of parameters, computational costs grow quadratically with context length and linearly with model size. Standard dense transformers require computing attention over all positions and activating all parameters for every forward pass, regardless of the complexity of the input. This inefficiency limits both the maximum context length we can handle and the model sizes that are economically viable to deploy.

Sparse architectures address these limitations by conditionally activating only a subset of the model's components for each input. Mixture of Experts (MoE) models replace dense feed-forward layers with sparse expert networks, allowing the model to grow in capacity without proportionally increasing computation. Sparse attention mechanisms similarly restrict attention to local or learned patterns rather than computing full quadratic attention.

This skill covers understanding MoE architectures and sparse attention mechanisms, implementing efficient sparse layers, handling routing and load balancing challenges, and deploying sparse models in production.

## Theory & Fundamentals

### Mixture of Experts Architecture

In a standard transformer block, the feed-forward network (FFN) is dense:

```
Standard FFN: output = W2 * ReLU(W1 * input)
```

MoE replaces this with a mixture of expert networks:

```
MoE: output = Σᵢ g(x)ᵢ * Expertᵢ(x)

where g(x) is a gating function that computes sparse activation weights
```

The key equations:

**Gating Function** (typically a simple linear layer with softmax):
$$g(x)_i = \frac{\exp(f(x)_i)}{\sum_j \exp(f(x)_j)}$$

**Load Balancing Loss** (to prevent expert collapse):
$$\mathcal{L}_{balance} = \alpha \cdot \text{entropy}(load) \cdot \text{entropy}(gating)$$

### Sparse Attention Mechanisms

Standard attention has O(n²) complexity:
$$A = \text{softmax}(QK^T / \sqrt{d}) \cdot V$$

Sparse attention patterns:

**Longformer/ETF Attention**:
$$A_{sparse} = \text{softmax}(QK^T / \sqrt{d} \cdot M) \cdot V$$

where M is a sparse mask with local window + global attention pattern.

**ReZero/Sparse Attention**:
$$A_{local} = \text{softmax}(Q_{local}K_{local}^T / \sqrt{d})$$

**Linear Attention** (approximation):
$$A_{linear} = \phi(Q)(\phi(K)^T V)$$

where φ is a feature map that enables O(n) computation.

### Routing Algorithms

**Top-K Routing**:
```python
def top_k_routing(x, gate, k=2):
    scores = gate(x)
    top_k_indices = torch.topk(scores, k).indices
    top_k_weights = F.softmax(scores[top_k_indices], dim=-1)
    return top_k_indices, top_k_weights
```

**Expert Choice Routing**:
```python
def expert_choice_routing(x, gate, capacity):
    scores = gate(x)  # [batch, experts]
    top_k_per_expert = torch.topk(scores.T, capacity // num_experts)
    return top_k_per_expert
```

### Mathematical Framework

**Capacity and Sparsity**:
- Active parameters: If k out of E experts are active, active params = k/E × total params
- Theoretical speedup: E/k × forward pass speedup (ignoring communication overhead)

**Expert Utilization**:
$$\text{Utilization}_i = \frac{\text{tokens routed to expert } i}{\text{total tokens}}$$

Target utilization: ~1/E for perfect load balancing

## Implementation Patterns

### Pattern 1: Efficient MoE Layer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

class Expert(nn.Module):
    """Single expert network."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int
    ):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # SwiGLU variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with Top-K routing.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        eval_capacity_factor: float = 2.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size, num_experts)
            for _ in range(num_experts)
        ])
        
        self.device = device
        
        self.register_buffer(
            "expert_usage",
            torch.zeros(num_experts),
            persistent=False
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [seq_len, batch, hidden_size] or [batch, seq_len, hidden_size]
        Returns:
            output: Same shape as input
            stats: Dictionary with routing statistics
        """
        original_shape = x.shape
        if x.dim() == 3:
            seq_len, batch, hidden_size = x.shape
            x_flat = x.view(-1, hidden_size)
        else:
            batch, hidden_size = x.shape
            seq_len = 1
            x_flat = x
        
        num_tokens = x_flat.shape[0]
        
        if self.training:
            capacity = int(
                num_tokens * self.top_k * self.capacity_factor / self.num_experts
            )
        else:
            capacity = int(
                num_tokens * self.eval_capacity_factor
            )
        
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(
            gate_probs, self.top_k, dim=-1
        )
        
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        expert_for_token = top_k_indices.view(-1)
        expert_weights = top_k_probs.view(-1, self.top_k)
        
        flat_expert_indices = expert_for_token + (
            torch.arange(self.top_k, device=x.device) * self.num_experts
        ).repeat(num_tokens)
        
        sorted_indices = torch.argsort(flat_expert_indices)
        sorted_expert_indices = flat_expert_indices[sorted_indices]
        sorted_weights = expert_weights.view(-1)[sorted_indices]
        
        cumsum = torch.zeros(self.num_experts * self.top_k + 1, device=x.device)
        ones = torch.ones(self.top_k * num_tokens, device=x.device)
        cumsum.scatter_add_(0, sorted_expert_indices.long(), ones)
        expert_counts = cumsum.view(self.top_k, self.num_experts).sum(0)
        
        token_expert_count = expert_counts[expert_for_token.view(self.top_k, num_tokens).T].sum(-1)
        
        max_count = expert_counts.max().item()
        
        dispatch_mask = token_expert_count <= capacity
        
        if dispatch_mask.sum() < num_tokens:
            sorted_tokens = torch.nonzero(dispatch_mask).squeeze(-1)
        else:
            sorted_tokens = torch.arange(num_tokens, device=x.device)
        
        outputs_by_expert = {}
        for expert_idx in range(self.num_experts):
            expert_mask = expert_for_token == expert_idx
            
            if self.training:
                expert_mask = expert_mask & dispatch_mask
            else:
                expert_mask = expert_mask & (expert_counts[expert_idx] <= capacity)
            
            expert_indices = torch.nonzero(expert_mask).squeeze(-1)
            
            if len(expert_indices) == 0:
                continue
            
            expert_input = x_flat[expert_indices]
            
            expert_output = self.experts[expert_idx](expert_input)
            
            outputs_by_expert[expert_idx] = (expert_indices, expert_output)
        
        final_output = torch.zeros_like(x_flat)
        
        for expert_idx, (indices, output) in outputs_by_expert.items():
            final_output[indices] += output * expert_weights[indices, 0]
        
        self.expert_usage += torch.tensor(
            [len(outputs_by_expert.get(i, ([], None))[0]) for i in range(self.num_experts)],
            device=self.expert_usage.device,
            dtype=self.expert_usage.dtype
        )
        
        if original_shape[0] == 3:
            output = final_output.view(seq_len, batch, hidden_size)
        else:
            output = final_output
        
        stats = {
            "expert_counts": torch.tensor(
                [len(outputs_by_expert.get(i, ([], None))[0]) for i in range(self.num_experts]
            ),
            "utilization": torch.tensor([
                len(outputs_by_expert.get(i, ([], None))[0]) / max(num_tokens, 1)
                for i in range(self.num_experts)
            ]),
            "tokens_dispatched": num_tokens,
            "tokens_processed": sum(
                len(outputs_by_expert.get(i, ([], None))[0])
                for i in range(self.num_experts)
            )
        }
        
        return output, stats


class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE feed-forward layer.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        self.moe = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        moe_output, moe_stats = self.moe(x)
        x = self.norm2(x + self.dropout2(moe_output))
        
        return x, moe_stats
```

### Pattern 2: Load Balancing Loss Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoadBalancingLoss(nn.Module):
    """
    Load balancing loss for MoE training.
    Encourages equal utilization across experts.
    """
    
    def __init__(
        self,
        num_experts: int,
        alpha: float = 0.01,
        noise_std: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.noise_std = noise_std
        
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self.register_buffer("expert_probs_sum", torch.zeros(num_experts))
        self.step_count = 0
    
    def forward(
        self,
        gate_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Args:
            gate_logits: [num_tokens, num_experts] raw gate outputs
            expert_indices: [num_tokens, top_k] indices of selected experts
            expert_weights: [num_tokens, top_k] weights for selected experts
        
        Returns:
            loss: Scalar load balancing loss
        """
        num_tokens, num_experts = gate_logits.shape
        
        probs = F.softmax(gate_logits, dim=-1)
        
        expert_probs = probs.mean(dim=0)
        
        self.expert_probs_sum += expert_probs.detach()
        
        expert_counts = torch.zeros(num_experts, device=gate_logits.device)
        for i in range(self.num_experts):
            expert_counts[i] = (expert_indices == i).sum().float()
        
        expert_counts = expert_counts / num_tokens
        
        self.expert_counts += expert_counts.detach()
        
        aux_loss = num_experts * torch.sum(expert_probs * expert_counts)
        
        if self.training and self.noise_std > 0:
            aux_loss = aux_loss + torch.randn((), device=aux_loss.device) * self.noise_std
        
        self.step_count += 1
        
        return self.alpha * aux_loss
    
    def get_utilization_stats(self) -> Dict[str, float]:
        """Get current utilization statistics."""
        if self.step_count == 0:
            return {}
        
        avg_probs = self.expert_probs_sum / self.step_count
        avg_counts = self.expert_counts / self.step_count
        
        utilization = avg_counts.cpu().numpy()
        
        return {
            "mean_utilization": float(utilization.mean()),
            "std_utilization": float(utilization.std()),
            "min_utilization": float(utilization.min()),
            "max_utilization": float(utilization.max()),
            "imbalance": float(utilization.max() / max(utilization.mean(), 1e-6) - 1)
        }


class AuxiliaryLossCalculator(nn.Module):
    """
    Computes multiple auxiliary losses for MoE training.
    """
    
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.load_balance_loss = LoadBalancingLoss(num_experts)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        gate_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute all auxiliary losses.
        
        Returns:
            total_loss: Sum of all auxiliary losses
            loss_info: Dictionary with individual loss components
        """
        load_balance = self.load_balance_loss(gate_logits, expert_indices, expert_weights)
        
        z_loss = self._compute_z_loss(gate_logits)
        
        router_z_loss = self._compute_router_z_loss(gate_logits)
        
        total_loss = load_balance + z_loss + router_z_loss
        
        loss_info = {
            "load_balance_loss": load_balance.item(),
            "z_loss": z_loss.item(),
            "router_z_loss": router_z_loss.item(),
            "total_aux_loss": total_loss.item()
        }
        
        return total_loss, loss_info
    
    def _compute_z_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Z-loss: Penalizes large logit values to improve stability.
        From ST-MoE paper.
        """
        return 0.001 * torch.mean(gate_logits ** 2)
    
    def _compute_router_z_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Router z-loss: Additional regularization on router.
        """
        return 0.01 * torch.mean(F.softmax(gate_logits, dim=-1) * gate_logits)
```

### Pattern 3: Sparse Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class SparseAttention(nn.Module):
    """
    Implements sparse attention patterns:
    - Sliding window attention
    - Global attention tokens
    - Random attention
    - Dilated attention
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 512,
        num_global_tokens: int = 2,
        num_random_heads: int = 1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.num_random_heads = num_random_heads
        
        assert hidden_size % num_heads == 0
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)
        
        self.attention_dropout = nn.Dropout(attention_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        global_token_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            global_token_indices: Indices of tokens that should attend to all others
            attention_mask: [batch, seq_len] True for valid positions
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        qkv_out = self.qkv(x)
        qkv_out = qkv_out.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv_out = qkv_out.permute(2, 0, 3, 1, 4)
        q, k, v = qkv_out[0], qkv_out[1], qkv_out[2]
        
        q = q / math.sqrt(self.head_dim)
        
        attention_weights = self._compute_sparse_attention(
            q, k, v, global_token_indices
        )
        
        attention_weights = self.attention_dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        output = self.proj(context)
        
        return output
    
    def _compute_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_token_indices: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute sparse attention pattern.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        attention = torch.zeros(
            batch_size, num_heads, seq_len, seq_len,
            device=q.device, dtype=q.dtype
        )
        
        self._add_window_attention(attention, q, k, seq_len)
        
        if global_token_indices is not None:
            self._add_global_attention(attention, q, k, global_token_indices)
        
        if self.num_random_heads > 0:
            self._add_random_attention(attention, q, k, seq_len)
        
        attention = F.softmax(attention, dim=-1)
        
        return attention
    
    def _add_window_attention(
        self,
        attention: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int
    ):
        """Add sliding window attention."""
        batch_size, num_heads, _, head_dim = q.shape
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            q_i = q[:, :, i:i+1, :]  # [batch, heads, 1, dim]
            k_window = k[:, :, start:end, :]  # [batch, heads, window, dim]
            
            scores = torch.matmul(q_i, k_window.transpose(-2, -1)) / math.sqrt(head_dim)
            
            attention[:, :, i:i+1, start:end] = scores
    
    def _add_global_attention(
        self,
        attention: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        global_indices: torch.Tensor
    ):
        """Add global attention for special tokens."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        for batch_idx in range(batch_size):
            for idx in global_indices[batch_idx]:
                q_global = q[batch_idx:batch_idx+1, :, idx:idx+1, :]
                
                scores = torch.matmul(q_global, k[batch_idx:batch_idx+1].transpose(-2, -1)) / math.sqrt(head_dim)
                
                attention[batch_idx:batch_idx+1, :, idx:idx+1, :] = scores
    
    def _add_random_attention(
        self,
        attention: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int
    ):
        """Add random attention to a few heads for each position."""
        batch_size, num_heads, _, head_dim = q.shape
        
        random_heads = min(self.num_random_heads, num_heads)
        
        num_random_neighbors = 8
        
        for h in range(random_heads):
            random_indices = torch.randint(
                0, seq_len, (batch_size, num_heads, seq_len, num_random_neighbors),
                device=q.device
            )
            
            k_random = k.unsqueeze(-2).expand(-1, -1, -1, num_random_neighbors, -1)
            
            indices_expanded = random_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
            k_gathered = torch.gather(
                k_random, 3,
                indices_expanded
            ).squeeze(3)
            
            q_expanded = q.unsqueeze(-2).expand(-1, -1, -1, num_random_neighbors, -1)
            scores = (q_expanded[:, :random_heads] * k_gathered[:, :random_heads]).sum(-1) / math.sqrt(head_dim)
            
            attention[:, :random_heads, :, :] += scores.transpose(2, 3)


class LongformerAttention(nn.Module):
    """
    Longformer-style attention: local window + global attention.
    More efficient for very long sequences.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 256,
        num_global_tokens: int = 2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        global_token_indices: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attention = self._longformer_attention(q, k, v, global_token_indices)
        
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        return self.proj(context)
    
    def _longformer_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_indices: torch.Tensor
    ) -> torch.Tensor:
        """Longformer attention computation."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        attention = torch.full(
            (batch_size, num_heads, seq_len, seq_len),
            float('-inf'),
            device=q.device
        )
        
        for b in range(batch_size):
            for g_idx in global_indices[b]:
                attention[b, :, g_idx, :] = 0
                attention[b, :, :, g_idx] = 0
        
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            attention[:, :, i:i+1, start:end] = 0
        
        attention = F.softmax(attention, dim=-1)
        
        return attention
```

### Pattern 4: Hashing Router for MoE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import hashlib

class HashingRouter(nn.Module):
    """
    Hash-based router for MoE that eliminates the need for a gating network.
    Token-to-expert assignment is determined by hashing, ensuring perfect load balance.
    """
    
    def __init__(
        self,
        num_experts: int,
        num_hash_functions: int = 1,
        hidden_size: Optional[int] = None,
        use_token_features: bool = False
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.num_hash_functions = num_hash_functions
        self.use_token_features = use_token_features
        
        if use_token_features and hidden_size:
            self.token_projection = nn.Linear(hidden_size, 128)
        
        self.hash_buckets = nn.Parameter(
            torch.randn(num_hash_functions, 128, num_experts) * 0.1
        )
        self.hash_bias = nn.Parameter(torch.zeros(num_experts))
    
    def forward(
        self,
        tokens: torch.Tensor,
        token_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Route tokens to experts using hashing.
        
        Args:
            tokens: Input tokens [batch, seq, hidden_size] or [batch * seq, hidden_size]
            token_features: Optional features for routing [batch, seq, hidden_size]
        
        Returns:
            routing_weights: [batch * seq, num_experts]
            routing_info: Dictionary with routing statistics
        """
        if tokens.dim() == 3:
            original_shape = tokens.shape[:2]
            tokens_flat = tokens.reshape(-1, tokens.shape[-1])
        else:
            original_shape = (tokens.shape[0], 1)
            tokens_flat = tokens
        
        if self.use_token_features and token_features is not None:
            if token_features.dim() == 3:
                features = token_features.reshape(-1, token_features.shape[-1])
            else:
                features = token_features
            features = self.token_projection(features)
        else:
            features = tokens_flat
        
        features = F.tanh(features)
        
        routing_logits = torch.matmul(features, self.hash_buckets.mean(0)) + self.hash_bias
        
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        if self.num_hash_functions > 1:
            all_probs = []
            for h in range(self.num_hash_functions):
                h_features = F.tanh(
                    torch.matmul(features, self.hash_buckets[h]) + self.hash_bias
                )
                all_probs.append(F.softmax(h_features, dim=-1))
            
            routing_probs = torch.stack(all_probs, dim=0).mean(0)
        
        stats = self._compute_routing_stats(routing_probs)
        
        return routing_probs, stats
    
    def _compute_routing_stats(self, routing_probs: torch.Tensor) -> Dict:
        """Compute routing statistics."""
        expert_probs = routing_probs.mean(dim=0)
        
        _, expert_assignments = routing_probs.max(dim=-1)
        
        expert_counts = torch.bincount(
            expert_assignments,
            minlength=self.num_experts
        ).float()
        
        total_tokens = routing_probs.shape[0]
        expert_utilization = expert_counts / total_tokens
        
        return {
            "expert_probs": expert_probs,
            "expert_counts": expert_counts,
            "expert_utilization": expert_utilization,
            "max_utilization": expert_utilization.max().item(),
            "entropy": -(routing_probs * torch.log(routing_probs + 1e-8)).sum(-1).mean().item()
        }


class SwitchTransformerTopplingRouter(nn.Module):
    """
    Switch Transformer-style routing with capacity factor.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        eval_capacity_factor: float = 2.0
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        
        self.gate = nn.Linear(hidden_size, num_experts)
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Route to top expert only (Switch Transformer style).
        
        Returns:
            gate_values: Top expert probability for each token
            expert_indices: Selected expert index for each token
            routing_info: Statistics dictionary
        """
        gate_logits = self.gate(hidden_states)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        gate_values, expert_indices = torch.max(gate_probs, dim=-1)
        
        if self.training:
            capacity = int(
                hidden_states.shape[0] * self.capacity_factor / self.num_experts
            )
        else:
            capacity = int(
                hidden_states.shape[0] * self.eval_capacity_factor
            )
        
        dispatched = torch.zeros(
            hidden_states.shape[0], self.num_experts,
            device=hidden_states.device
        )
        
        dispatched.scatter_(
            1,
            expert_indices.unsqueeze(-1),
            gate_values.unsqueeze(-1)
        )
        
        expert_counts = dispatched.sum(0)
        
        overload = expert_counts > capacity
        
        stats = {
            "expert_counts": expert_counts,
            "capacity": capacity,
            "overloaded_experts": overload.sum().item(),
            "max_load": expert_counts.max().item()
        }
        
        return gate_values, expert_indices, stats
```

### Pattern 5: Expert Specialization Analysis

```python
from typing import Dict, List, Tuple
import torch
import numpy as np
from collections import defaultdict

class ExpertAnalyzer:
    """
    Analyzes expert specialization and behavior in MoE models.
    """
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        
        self.expert_activations: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.expert_inputs: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.expert_outputs: Dict[int, List[torch.Tensor]] = defaultdict(list)
        
        self.token_metadata: List[Dict] = []
    
    def record_batch(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        metadata: Optional[Dict] = None
    ):
        """Record a batch of expert activations."""
        for expert_id in range(self.num_experts):
            mask = expert_indices == expert_id
            
            if mask.any():
                self.expert_inputs[expert_id].append(inputs[mask].detach().cpu())
                self.expert_outputs[expert_id].append(outputs[mask].detach().cpu())
    
    def analyze_specialization(self) -> Dict[int, Dict]:
        """Analyze what each expert has learned to specialize in."""
        specialization = {}
        
        for expert_id in range(self.num_experts):
            if expert_id not in self.expert_inputs or not self.expert_inputs[expert_id]:
                continue
            
            inputs = torch.cat(self.expert_inputs[expert_id], dim=0)
            outputs = torch.cat(self.expert_outputs[expert_id], dim=0)
            
            input_mean = inputs.mean(dim=0)
            input_std = inputs.std(dim=0)
            
            output_mean = outputs.mean(dim=0)
            output_std = outputs.std(dim=0)
            
            sparsity = (outputs.abs() < 0.01).float().mean().item()
            
            specialization[expert_id] = {
                "input_mean_norm": input_mean.norm().item(),
                "input_std_mean": input_std.mean().item(),
                "output_mean_norm": output_mean.norm().item(),
                "output_sparsity": sparsity,
                "num_activations": inputs.shape[0]
            }
        
        return specialization
    
    def compute_expert_diversity(self) -> float:
        """
        Compute diversity score between experts.
        Higher score = more specialized (different) experts.
        """
        expert_means = []
        
        for expert_id in range(self.num_experts):
            if expert_id in self.expert_inputs and self.expert_inputs[expert_id]:
                inputs = torch.cat(self.expert_inputs[expert_id])
                expert_means.append(inputs.mean(dim=0))
            else:
                expert_means.append(torch.zeros_like(expert_means[0] if expert_means else torch.tensor(0)))
        
        if len(expert_means) < 2:
            return 0.0
        
        expert_means = torch.stack(expert_means)
        
        pairwise_distances = []
        for i in range(len(expert_means)):
            for j in range(i + 1, len(expert_means)):
                dist = torch.norm(expert_means[i] - expert_means[j]).item()
                pairwise_distances.append(dist)
        
        return np.mean(pairwise_distances)
    
    def get_load_balance_report(self) -> Dict:
        """Generate load balancing report."""
        total_activations = sum(
            len(inputs) for inputs in self.expert_inputs.values()
        )
        
        expert_loads = {}
        for expert_id, inputs in self.expert_inputs.items():
            expert_loads[expert_id] = len(inputs)
        
        if total_activations == 0:
            return {"status": "no_data"}
        
        loads = np.array(list(expert_loads.values()))
        expected = total_activations / self.num_experts
        
        cv = loads.std() / max(expected, 1)
        
        return {
            "total_activations": total_activations,
            "expected_per_expert": expected,
            "actual_loads": expert_loads,
            "coefficient_of_variation": cv,
            "is_balanced": cv < 0.5,
            "recommendations": self._generate_recommendations(cv, loads, expected)
        }
    
    def _generate_recommendations(
        self,
        cv: float,
        loads: np.ndarray,
        expected: float
    ) -> List[str]:
        """Generate recommendations based on load balance analysis."""
        recommendations = []
        
        if cv > 1.0:
            recommendations.append(
                "Critical: High load imbalance detected. Consider adjusting routing."
            )
        elif cv > 0.5:
            recommendations.append(
                "Warning: Moderate load imbalance. Monitor for expert collapse."
            )
        
        underutilized = np.where(loads < expected * 0.1)[0]
        if len(underutilized) > 0:
            recommendations.append(
                f"Underutilized experts: {list(underutilized)}. May indicate expert collapse."
            )
        
        overloaded = np.where(loads > expected * 2)[0]
        if len(overloaded) > 0:
            recommendations.append(
                f"Overloaded experts: {list(overloaded)}. May cause latency issues."
            )
        
        return recommendations
```

## Framework Integration

### Integration with HuggingFace Transformers

```python
from transformers import PreTrainedModel, MoEConfig, AutoConfig
import torch

class HuggingFaceMoELayer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.moe = MoELayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok
        )
```

### Integration with DeepSpeed

```python
import deepspeed

class DeepSpeedMoEConfig:
    @staticmethod
    def get_config(mpu, num_experts):
        return {
            "moe": True,
            "moe_param_group": True,
            "moe_per_expert_grad": True,
            "ep_size": mpu.get_data_parallel_world_size(),
            "num_experts": num_experts
        }
```

## Performance Considerations

### Communication Overhead

In distributed MoE training, expert parallelism introduces communication:
- All-to-all communication for routing tokens to experts
- Latency increases with number of expert groups
- Use bucketing and overlap for efficiency

### Memory Efficiency

- Experts can be sharded across devices
- Expert states: parameters + optimizer states + activations
- Use expert dropout to reduce memory during training

## Common Pitfalls

### Pitfall 1: Expert Collapse

**Problem**: Few experts receive most tokens, others never train.

**Solution**: Implement load balancing loss and auxiliary penalties:
```python
if expert_utilization.std() > 0.5:
    increase_load_balance_loss_weight()
```

### Pitfall 2: Ignoring Communication Costs

**Problem**: Not accounting for AllToAll communication in distributed training.

**Solution**: Profile end-to-end throughput, not just FLOPs:
```python
# Profile both compute and communication
total_time = profile(lambda: model(input))
compute_time = profile(lambda: forward_without_comm(model, input))
communication_overhead = total_time - compute_time
```

### Pitfall 3: Token Dropping

**Problem**: When capacity is exceeded, tokens are dropped silently.

**Solution**: Monitor dropped tokens and adjust capacity:
```python
if dropped_tokens > 0:
    logger.warning(f"Tokens dropped: {dropped_tokens}")
    increase_capacity_factor()
```

## Research References

1. **Shazeer et al. (2017)** - "Outrageously Large Neural Networks" - Original MoE paper.

2. **Lepikhin et al. (2021)** - "GShard" - Large-scale MoE for translation.

3. **Fedus et al. (2022)** - "Switch Transformers" - Simplified sparse MoE routing.

4. **Zoph et al. (2022)** - "ST-MoE" - Stabilizing sparse MoE training.

5. **Clark et al. (2022)** - "Fправ刃" - DeepSpeed MoE.

6. **Beltagy et al. (2020)** - "Longformer" - Sparse attention for long documents.

7. **Kitaev et al. (2020)** - "Reformer" - Locality-sensitive hashing attention.

8. **Zaheer et al. (2021)** - "BigBird" - Sparse attention with theoretical guarantees.

9. **Dai et al. (2019)** - "Transformer-XL" - Relative positional encoding for long context.

10. **Kaiser et al. (2017)** - "Depthwise Separable Attention" - Efficient attention variants.