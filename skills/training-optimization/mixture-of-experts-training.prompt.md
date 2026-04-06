# Mixture of Experts (MoE) Training: Comprehensive Skill Guide

## Overview

Mixture of Experts (MoE) is a sparse architecture paradigm that dramatically scales model capacity without proportionally increasing per-token computation. By routing each token to a small subset of expert networks while keeping most parameters inactive, MoE enables training of trillion-parameter models efficiently.

This skill document provides complete guidance on MoE training fundamentals, routing strategies, load balancing, expert specialization, scaling laws, distributed implementation, and production best practices.

---

## 1. MoE Fundamentals

### 1.1 Dense vs Sparse Routing Mechanisms

**Dense Routing (Standard Transformer):**
- All tokens activate all parameters in FFN
- Computation: O(batch × seq_len × hidden_dim × ffn_dim)
- Parameters fully utilized per token

**Sparse Routing (MoE):**
- Each token routes to k out of E experts
- Computation: O(batch × seq_len × hidden_dim × (ffn_dim/E) × k)
- Only k/E fraction of experts active per token
- Parameter efficiency: Can scale to 10-100x more parameters with similar FLOPs

**Key Advantage:** Parameter scale decoupled from compute - enables billion/trillion parameter models at dense training speeds.

### 1.2 Expert Networks: Multiple Independent Feed-Forward Layers

```
MoE Layer Architecture:
                    ┌─────────────────────────┐
                    │    Router/Gating Net    │
                    │  (d_model → E logits)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Select top-k experts    │
                    └────────────┬────────────┘
                                 │
        ┌────────────┬───────────┼───────────┬────────────┐
        │            │           │           │            │
    Expert 1     Expert 2    Expert 3    Expert 4    Expert N
    (FFN)        (FFN)       (FFN)       (FFN)       (FFN)
    
    Each Expert = Independent Feed-Forward Network:
    Expert(x) = W_2 * ReLU(W_1 * x + b_1) + b_2
```

Each expert is typically a 2-layer FFN:
- Input projection: d_model → d_ffn (usually 4 × d_model)
- Activation function (ReLU, SwiGLU, etc.)
- Output projection: d_ffn → d_model

**Expert Specialization:**
- Early layers: Shared syntactic knowledge across experts
- Middle layers: Domain-specific specialization
- Later layers: Task-specific and semantic specialization

### 1.3 Gating Network for Routing Decisions

The gating network determines which experts process each token:

```
g(x) = softmax(W_gate * x + b_gate)  # E-dimensional logits
expert_idx = Top-K(g(x), k)
```

**Gating Output Properties:**
- Non-negative (via softmax)
- Sum to 1 across all experts
- Can interpret as routing probabilities
- Differentiable for gradient-based optimization

**Router Considerations:**
- Temperature scaling: Controls sharpness of expert selection
- Numerical stability: Router logits often need scaling/normalization
- Computational efficiency: Linear operation, minimal overhead

### 1.4 Load Balancing Importance

**Without Load Balancing:**
- Tokens naturally cluster to similar experts
- Some experts overloaded, others underutilized
- Results in:
  - Token dropping (capacity exceeded)
  - Uneven GPU utilization
  - Wasted parameters in unused experts
  - Routing collapse (all tokens → few experts)

**With Balanced Load:**
- Uniform token distribution across experts
- Full parameter utilization
- Improved model expressiveness
- Efficient distributed training
- Lower latency in inference

**Load Imbalance Metrics:**
- Expert utilization variance: High = imbalanced
- Token dropout rate: Should be <5-10%
- Expert frequency entropy: Higher = more balanced

---

## 2. Routing Strategies

### 2.1 Top-K Routing: Select Top K Experts

**Mechanism:**
```
logits = router(x)  # Shape: (batch_size, seq_len, num_experts)
top_k_logits, top_k_indices = torch.topk(logits, k=2, dim=-1)
weights = softmax(top_k_logits, dim=-1)  # Normalize top-k
```

**Variants:**
- **Top-1 Routing** (Switch Transformer): k=1, simplest, fastest
- **Top-2 Routing** (GShard, GLaM, Mixtral): k=2, balanced
- **Top-K Routing** (General case): k varies by layer

**Capacity Constraint Handling:**

```python
def apply_capacity_constraints(
    router_logits: torch.Tensor,
    capacity_factor: float = 1.25,
    num_experts: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        router_logits: (batch, seq_len, num_experts)
        capacity_factor: tokens per expert = seq_len / num_experts * capacity_factor
        num_experts: number of experts E
    
    Returns:
        masked_logits: logits with overflowed tokens set to -inf
        expert_load: tokens assigned to each expert
    """
    batch_size, seq_len, _ = router_logits.shape
    
    # Maximum tokens each expert can handle
    capacity = int(capacity_factor * seq_len / num_experts)
    
    # Count tokens per expert
    expert_loads = torch.zeros(num_experts, device=router_logits.device)
    
    for seq_idx in range(seq_len):
        _, expert_idx = router_logits[:, seq_idx, :].max(dim=-1)
        expert_loads[expert_idx] += 1
    
    # Mask overflowed slots
    masked_logits = router_logits.clone()
    for exp_idx in range(num_experts):
        overflow = max(0, expert_loads[exp_idx] - capacity)
        if overflow > 0:
            # Set lowest-probability tokens to -inf
            expert_mask = router_logits[:, :, exp_idx]
            _, overflow_indices = torch.topk(expert_mask, k=int(overflow), 
                                           dim=0, largest=False)
            masked_logits[overflow_indices, :, exp_idx] = float('-inf')
    
    return masked_logits, expert_loads
```

**Trade-offs:**
- Higher k: More experts per token → better load distribution but higher compute
- Lower k: Fewer experts → less compute but risk of routing collapse
- Capacity factor: Higher = more buffering against overflow, lower = less waste

### 2.2 Expert Choice Routing: Experts Select Tokens

**Inverse Routing Pattern:**
Instead of tokens choosing experts, experts select tokens they want to process:

```
For each expert e:
    expert_score = compute_expert_score(e, all_tokens)
    select_top_tokens_matching_expertise(expert_e, top_k_tokens)
```

**Advantages:**
- Leverages expert specialization
- Can adapt to expert capacity dynamically
- Reduces routing collapse tendency

**Implementation Complexity:**
- Requires all-to-all communication
- Higher memory for storing expert preferences
- Used in Expert Choice Transformers (less common)

### 2.3 Learned Routing: Gating Networks with Softmax/Temperature

**Temperature-Based Gating:**

```python
class TemperatureScaledRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, temperature: float = 1.0):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            routing_probs: (batch, seq_len, num_experts)
        """
        logits = self.router(x)  # (batch, seq_len, num_experts)
        
        # Temperature scaling: higher T = softer distribution
        logits = logits / self.temperature
        
        routing_probs = F.softmax(logits, dim=-1)
        return routing_probs
```

**Gating with Logit Normalization (Skywork-MoE):**

```python
class NormalizedRouter(nn.Module):
    """Normalizes gating logits before softmax for better discrimination"""
    
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        
        # Compute mean and std per token
        mean = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True) + 1e-8
        
        # Normalize: (z - μ) / σ
        normalized = (logits - mean) / std
        
        # Apply lambda scaling (typically 0.5-1.0)
        lambda_scale = 1.0
        normalized = lambda_scale * normalized
        
        routing_probs = F.softmax(normalized, dim=-1)
        return routing_probs
```

**Learned Bias-Based Routing (DeepSeek-V3):**

```python
class BiasedRouter(nn.Module):
    """Dynamically adjusts expert affinity via learned bias terms"""
    
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        self.k = 8  # top-k experts
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)  # (batch, seq_len, num_experts)
        
        # Add learned bias to shift expert selection
        adjusted_logits = logits + self.expert_bias.unsqueeze(0).unsqueeze(0)
        
        # Select top-k
        top_logits, top_indices = torch.topk(adjusted_logits, k=self.k, dim=-1)
        
        # Softmax on top-k only
        routing_weights = F.softmax(top_logits, dim=-1)
        
        return routing_weights, top_indices
    
    def update_bias(self, expert_loads: torch.Tensor, learning_rate: float = 0.001):
        """Dynamically adjust bias based on expert load imbalance"""
        target_load = expert_loads.mean()
        load_deviation = expert_loads - target_load
        
        # Decrease bias for overloaded experts (make them less attractive)
        # Increase bias for underloaded experts (make them more attractive)
        self.expert_bias.data -= learning_rate * load_deviation
```

### 2.4 Hard Routing: Gumbel-Softmax for Discrete Selection

**Problem:** Softmax is differentiable everywhere, but selecting discrete experts (hard top-k) is not.

**Solution:** Use Gumbel-Softmax for differentiable discrete sampling:

```python
import torch.nn.functional as F

class GumbelSoftmaxRouter(nn.Module):
    """Uses Gumbel-Softmax trick for discrete expert selection"""
    
    def __init__(self, d_model: int, num_experts: int, k: int = 2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.k = k
        self.temperature = 1.0  # Annealed during training
    
    def gumbel_softmax(self, logits: torch.Tensor, 
                       tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, num_experts)
            tau: temperature (lower = sharper)
            hard: if True, discretize (use straight-through estimator)
        
        Returns:
            y: soft approximation or discrete approximation
        """
        # Sample Gumbel noise
        eps = 1e-20
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        
        # Add noise and apply temperature
        y_soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
        
        if hard:
            # Straight-through estimator: argmax in forward, soft in backward
            y_hard = F.one_hot(y_soft.argmax(dim=-1), num_classes=logits.size(-1))
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        
        return y
    
    def forward(self, x: torch.Tensor, temperature: float = None) -> torch.Tensor:
        logits = self.router(x)
        tau = temperature or self.temperature
        
        # Gumbel-softmax with annealed temperature
        routing_probs = self.gumbel_softmax(logits, tau=tau, hard=True)
        
        # Select top-k (though Gumbel-softmax already approximates discrete)
        return routing_probs
```

**Training Annealing Schedule:**
```
Start with high temperature (τ=10) → Soft, differentiable
Gradually decrease during training (τ=1) → Sharper selection
End with τ→0 → Discrete hard selection
```

---

## 3. Load Balancing

### 3.1 Auxiliary Loss Functions to Encourage Uniform Load

**Basic Auxiliary Loss (GShard, Switch):**

```python
def auxiliary_loss_importance_weighted(
    router_probs: torch.Tensor,  # (batch, seq_len, num_experts)
    expert_mask: torch.Tensor,   # (batch, seq_len, num_experts)
    alpha: float = 0.01
) -> torch.Tensor:
    """
    Importance-weighted auxiliary loss (Google's formula):
    
    L_aux = α * Σ_e (f_e * P_e)
    
    where:
        f_e = fraction of tokens routed to expert e
        P_e = average gating probability for expert e
    """
    # Compute fraction of tokens to each expert
    expert_distribution = expert_mask.sum(dim=(0, 1)) / expert_mask.sum()  # (num_experts,)
    
    # Compute average gating probability per expert
    router_prob_per_expert = (router_probs * expert_mask).sum(dim=(0, 1)) / (expert_mask.sum(dim=(0, 1)) + 1e-10)
    
    # Balance loss
    balance_loss = torch.sum(expert_distribution * router_prob_per_expert)
    
    return alpha * balance_loss
```

**Auxiliary Loss Variants:**

```python
# ST-MoE: Router Z-Loss (stabilizes training)
def router_z_loss(router_logits: torch.Tensor, alpha: float = 1e-3) -> torch.Tensor:
    """
    Penalizes large router logits to improve training stability
    
    L_z = α * (1/B) * Σ_b (log(Σ_e exp(logit_e)))^2
    """
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = torch.mean(log_z ** 2)
    return alpha * z_loss


# DeepSpeed-MoE: Load balance loss
def load_balance_loss(expert_loads: torch.Tensor, 
                      num_tokens: int,
                      num_experts: int,
                      alpha: float = 0.01) -> torch.Tensor:
    """
    Encourages uniform load: L_bal = α * Σ_e |f_e - 1/E|
    """
    ideal_load = 1.0 / num_experts
    actual_load = expert_loads.float() / num_tokens
    
    load_loss = torch.sum(torch.abs(actual_load - ideal_load))
    return alpha * load_loss


# Skywork-MoE: Adaptive auxiliary loss
def adaptive_auxiliary_loss(
    router_probs: torch.Tensor,
    expert_mask: torch.Tensor,
    drop_rate: float,  # Computed from capacity constraints
    alpha_base: float = 0.01,
    xi: float = 0.1  # Sensitivity parameter
) -> Tuple[torch.Tensor, float]:
    """
    Dynamically adjust auxiliary loss coefficient based on drop rate
    
    α_{i+1} = β*α_i + (1-β)*ξ*d_i
    
    where d_i is current token drop rate
    """
    beta = 0.99  # Smoothing factor
    
    # Base auxiliary loss
    expert_dist = expert_mask.sum(dim=(0, 1)) / expert_mask.sum()
    router_prob_expert = (router_probs * expert_mask).sum(dim=(0, 1)) / (expert_mask.sum(dim=(0, 1)) + 1e-10)
    base_loss = torch.sum(expert_dist * router_prob_expert)
    
    # Adapt alpha based on drop rate
    alpha_new = beta * alpha_base + (1 - beta) * xi * drop_rate
    
    return alpha_new * base_loss, alpha_new
```

### 3.2 Importance-Weighted Auxiliary Loss (Google's Formula)

The most widely used formulation comes from GShard and refined in Switch Transformer:

```
L_balance = α * (1/E) * Σ_e (fraction_e × probability_e)

where:
  fraction_e = (1/N) * Σ_t [top_k_mask_{t,e}]
  probability_e = (1/N) * Σ_t [gating_prob_{t,e}]
  N = total tokens
  E = number of experts
  α = loss weight (typically 0.001-0.01)
```

**Intuition:**
- If expert e gets many tokens (high fraction) but low selection probability → they're being over-utilized
- If expert e gets few tokens (low fraction) but high probability → tokens aren't picking it
- Multiplying these pushes toward balance: high fraction × low probability → loss increases

**Implementation Deep Dive:**

```python
def compute_balance_loss_detailed(
    router_logits: torch.Tensor,      # (batch, seq_len, num_experts)
    expert_indices: torch.Tensor,     # (batch, seq_len, k) - top-k expert indices
    alpha: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute importance-weighted balance loss with detailed metrics
    
    Returns:
        loss: scalar balance loss
        fractions: (num_experts,) - fraction of tokens per expert
        probs: (num_experts,) - average routing probability per expert
    """
    batch_size, seq_len, num_experts = router_logits.shape
    k = expert_indices.shape[-1]
    
    # Compute softmax probabilities
    router_probs = F.softmax(router_logits, dim=-1)  # (batch, seq_len, num_experts)
    
    # Create one-hot mask for selected experts
    expert_mask = torch.zeros_like(router_logits)
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            for k_idx in range(k):
                expert_idx = expert_indices[batch_idx, seq_idx, k_idx]
                expert_mask[batch_idx, seq_idx, expert_idx] = 1.0
    
    # Normalize mask (each token contributes 1/k to each selected expert)
    expert_mask = expert_mask / (k + 1e-10)
    
    # Compute fractions: what fraction of tokens go to each expert
    total_tokens = batch_size * seq_len
    fractions = expert_mask.sum(dim=(0, 1)) / total_tokens
    
    # Compute average probability: average gating prob for selected experts
    probs = (router_probs * expert_mask).sum(dim=(0, 1)) / (expert_mask.sum(dim=(0, 1)) + 1e-10)
    
    # Balance loss
    balance_loss = torch.sum(fractions * probs)
    
    return alpha * balance_loss, fractions, probs
```

### 3.3 Expert Utilization Metrics

Monitor load balancing effectiveness:

```python
def compute_expert_metrics(
    expert_loads: torch.Tensor,  # (num_experts,) - count of tokens assigned
    router_probs: torch.Tensor,   # (batch, seq_len, num_experts) - routing probs
    num_tokens: int
) -> Dict[str, float]:
    """Compute detailed expert utilization metrics"""
    
    num_experts = expert_loads.shape[0]
    
    # Metrics dictionary
    metrics = {}
    
    # 1. Expert utilization: fraction of experts that received tokens
    utilized_experts = (expert_loads > 0).sum().float() / num_experts
    metrics['utilized_experts'] = utilized_experts.item()
    
    # 2. Load variance: how uneven is the distribution
    mean_load = expert_loads.mean()
    variance = ((expert_loads - mean_load) ** 2).mean()
    metrics['load_variance'] = variance.item()
    metrics['load_std'] = torch.sqrt(variance).item()
    
    # 3. Gini coefficient: measure of inequality (0=perfect balance, 1=all in one expert)
    sorted_loads = torch.sort(expert_loads)[0]
    n = len(sorted_loads)
    cumsum = torch.cumsum(sorted_loads, dim=0)
    gini = (2 * torch.sum(cumsum)) / (n * cumsum[-1]) - (n + 1) / n
    metrics['gini_coefficient'] = gini.item()
    
    # 4. Expert entropy: how distributed is the routing
    probs_per_expert = expert_loads / (expert_loads.sum() + 1e-10)
    entropy = -torch.sum(probs_per_expert * torch.log(probs_per_expert + 1e-10))
    max_entropy = torch.log(torch.tensor(num_experts, dtype=torch.float32))
    metrics['normalized_entropy'] = (entropy / max_entropy).item()
    
    # 5. Overload percentage: experts handling more than ideal capacity
    ideal_load_per_expert = num_tokens / num_experts
    overloaded = (expert_loads > 1.25 * ideal_load_per_expert).sum()
    metrics['overloaded_experts_pct'] = (overloaded.float() / num_experts).item()
    
    # 6. Router entropy: sharpness of gating
    router_entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1).mean()
    metrics['router_entropy'] = router_entropy.item()
    
    return metrics
```

### 3.4 Expert Dropping Techniques

**Token Dropping (Simple but Approximate):**

```python
def drop_tokens_on_overflow(
    x: torch.Tensor,  # (batch, seq_len, d_model)
    expert_logits: torch.Tensor,  # (batch, seq_len, num_experts)
    expert_capacity: int,  # max tokens per expert
    k: int = 2  # top-k
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Route tokens to experts with capacity constraints.
    Tokens exceeding capacity are dropped and processed by residual pathway.
    """
    batch_size, seq_len, d_model = x.shape
    num_experts = expert_logits.shape[-1]
    
    # Get top-k experts for each token
    top_logits, expert_indices = torch.topk(expert_logits, k=k, dim=-1)
    router_weights = F.softmax(top_logits, dim=-1)
    
    # Compute how many tokens each expert receives
    expert_loads = torch.zeros(num_experts, device=x.device)
    
    # Track which tokens are kept vs dropped
    token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
    
    for expert_id in range(num_experts):
        # Count tokens assigned to this expert
        tokens_for_expert = (expert_indices == expert_id).any(dim=-1).sum()
        
        # If exceeds capacity, drop some tokens
        if tokens_for_expert > expert_capacity:
            # Find tokens assigned to this expert
            mask = (expert_indices == expert_id).any(dim=-1)
            indices = torch.where(mask)
            
            # Drop extra tokens (keep capacity number)
            num_to_drop = tokens_for_expert - expert_capacity
            drop_indices = indices[0][:num_to_drop]  # batch indices
            drop_positions = indices[1][:num_to_drop]  # sequence indices
            
            token_mask[drop_indices, drop_positions] = False
        
        expert_loads[expert_id] = min(tokens_for_expert, expert_capacity)
    
    # Extract kept tokens
    kept_tokens = x[token_mask]  # (kept_count, d_model)
    kept_indices = torch.where(token_mask)
    
    return kept_tokens, kept_indices, token_mask
```

**Dropless MoE (JetMoE using MegaBlocks):**

```python
def dropless_moe_forward(
    x: torch.Tensor,  # (batch, seq_len, d_model)
    experts: nn.ModuleList,  # List[FFN experts]
    router: nn.Module,
    k: int = 2
) -> torch.Tensor:
    """
    Dropless MoE using block-sparse matrix operations.
    Ensures all tokens are processed without dropping.
    
    Key idea: Route tokens to experts, then use sparse GEMM to compute
    only the active expert-token pairs.
    """
    batch_size, seq_len, d_model = x.shape
    num_experts = len(experts)
    
    # Route tokens to top-k experts
    logits = router(x)  # (batch, seq_len, num_experts)
    top_logits, expert_indices = torch.topk(logits, k=k, dim=-1)
    weights = F.softmax(top_logits, dim=-1)
    
    # Create routing matrix: (batch*seq_len, num_experts, k)
    flat_x = x.reshape(-1, d_model)  # (batch*seq_len, d_model)
    
    # For each token, collect outputs from top-k experts
    output = torch.zeros_like(flat_x)
    
    for expert_id, expert in enumerate(experts):
        # Find all tokens routed to this expert
        tokens_for_expert = (expert_indices == expert_id)  # (batch, seq_len, k)
        token_indices = torch.where(tokens_for_expert)  # Get indices
        
        if len(token_indices[0]) == 0:
            continue  # No tokens for this expert
        
        # Flatten indices
        flat_indices = token_indices[0] * seq_len + token_indices[1]
        
        # Process tokens through expert
        expert_output = expert(flat_x[flat_indices])  # (num_tokens_for_expert, d_model)
        
        # Weighted contribution
        expert_weights = weights[token_indices]  # weights for this expert slot
        
        # Add to output with proper indexing
        output[flat_indices] += expert_output * expert_weights.unsqueeze(-1)
    
    return output.reshape(batch_size, seq_len, d_model)
```

---

## 4. Expert Specialization

### 4.1 Measuring Expertise: Entropy and Coverage

```python
def analyze_expert_specialization(
    router_logits: torch.Tensor,  # (batch, seq_len, num_experts)
    token_types: torch.Tensor = None,  # Optional: (batch, seq_len) - token categories
    top_k: int = 2
) -> Dict[str, float]:
    """
    Measure how specialized experts are in handling different token types.
    """
    batch_size, seq_len, num_experts = router_logits.shape
    
    router_probs = F.softmax(router_logits, dim=-1)
    
    metrics = {}
    
    # 1. Per-expert entropy: how diverse are the inputs to each expert?
    expert_entropies = []
    for expert_id in range(num_experts):
        expert_probs = router_probs[:, :, expert_id].reshape(-1)  # flatten
        # Create histogram if token_types provided
        if token_types is not None:
            flat_types = token_types.reshape(-1)
            num_types = flat_types.max().item() + 1
            type_dist = torch.zeros(num_types, device=expert_probs.device)
            for token_type in range(num_types):
                type_mask = flat_types == token_type
                if type_mask.sum() > 0:
                    type_dist[token_type] = expert_probs[type_mask].mean()
            # Entropy of token type distribution
            type_dist = type_dist / (type_dist.sum() + 1e-10)
            entropy = -torch.sum(type_dist * torch.log(type_dist + 1e-10))
        else:
            # Entropy of routing probability distribution
            entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10))
        
        expert_entropies.append(entropy.item())
    
    metrics['avg_expert_entropy'] = sum(expert_entropies) / len(expert_entropies)
    metrics['expert_entropy_std'] = np.std(expert_entropies)
    
    # 2. Expert coverage: for each token type, how many experts handle it?
    if token_types is not None:
        num_types = token_types.max().item() + 1
        avg_coverage = []
        
        for token_type in range(num_types):
            type_mask = token_types == token_type
            type_logits = router_logits[type_mask]  # (num_tokens_of_type, num_experts)
            
            # Count how many experts handle this type significantly
            type_probs = F.softmax(type_logits, dim=-1)
            significant_threshold = 1.0 / num_experts  # above uniform distribution
            significant_experts = (type_probs > significant_threshold).sum(dim=-1).float().mean()
            
            avg_coverage.append(significant_experts.item())
        
        metrics['avg_type_coverage'] = np.mean(avg_coverage)
        metrics['coverage_std'] = np.std(avg_coverage)
    
    # 3. Expert distinctiveness: cosine similarity between expert preference distributions
    expert_distributions = []
    for expert_id in range(num_experts):
        dist = router_probs[:, :, expert_id].reshape(-1)
        expert_distributions.append(dist)
    
    # Compute pairwise similarities
    similarities = []
    for i in range(num_experts):
        for j in range(i+1, num_experts):
            cosine_sim = torch.nn.functional.cosine_similarity(
                expert_distributions[i].unsqueeze(0),
                expert_distributions[j].unsqueeze(0)
            ).item()
            similarities.append(cosine_sim)
    
    metrics['expert_distinctiveness'] = 1.0 - np.mean(similarities)  # Higher = more distinct
    
    return metrics
```

### 4.2 Domain Specialization Analysis

```python
def analyze_domain_specialization(
    router_logits: torch.Tensor,  # (batch, seq_len, num_experts)
    domains: torch.Tensor,  # (batch, seq_len) - domain labels
    num_domains: int
) -> Dict:
    """
    Analyze how experts specialize in different domains (e.g., code vs natural text)
    """
    router_probs = F.softmax(router_logits, dim=-1)
    num_experts = router_logits.shape[-1]
    
    # Compute domain-expert affinity matrix: (num_domains, num_experts)
    domain_expert_affinity = torch.zeros(num_domains, num_experts, 
                                         device=router_logits.device)
    
    for domain_id in range(num_domains):
        domain_mask = domains == domain_id
        domain_probs = router_probs[domain_mask]  # (num_tokens_in_domain, num_experts)
        
        # Average probability per expert for this domain
        domain_expert_affinity[domain_id] = domain_probs.mean(dim=0)
    
    # Normalize to probabilities
    domain_expert_affinity = domain_expert_affinity / (domain_expert_affinity.sum(dim=1, keepdim=True) + 1e-10)
    
    # Analysis metrics
    metrics = {}
    
    # 1. Which experts specialize in which domain?
    expert_specialization = torch.argmax(domain_expert_affinity, dim=0)
    metrics['expert_primary_domain'] = expert_specialization.cpu().numpy().tolist()
    
    # 2. How much does each expert specialize? (entropy-based)
    expert_entropies = -torch.sum(domain_expert_affinity.T * torch.log(domain_expert_affinity.T + 1e-10), dim=1)
    max_entropy = torch.log(torch.tensor(num_domains, dtype=torch.float32))
    specialization_score = 1.0 - (expert_entropies / max_entropy)
    metrics['expert_specialization_scores'] = specialization_score.cpu().numpy().tolist()
    
    # 3. Domain coverage: how many experts are needed per domain?
    for domain_id in range(num_domains):
        # Cumulative probability: what fraction of experts needed to explain 80% of routing?
        domain_affinity = domain_expert_affinity[domain_id].sort(descending=True)[0]
        cumsum = torch.cumsum(domain_affinity, dim=0)
        num_experts_needed = (cumsum < 0.8).sum().item() + 1
        metrics[f'domain_{domain_id}_coverage'] = num_experts_needed
    
    return metrics
```

### 4.3 Multi-Task Specialization Patterns

```python
def analyze_multitask_routing(
    router_logits_by_task: Dict[str, torch.Tensor],  # task_name -> logits
    shared_params: bool = True
) -> Dict:
    """
    Analyze how experts serve multiple tasks:
    - Are there task-specific experts?
    - Are experts shared?
    - What's the specialization pattern?
    """
    
    metrics = {}
    task_names = list(router_logits_by_task.keys())
    num_tasks = len(task_names)
    num_experts = list(router_logits_by_task.values())[0].shape[-1]
    
    # Compute routing probability per task per expert
    task_expert_probs = {}
    for task_name, logits in router_logits_by_task.items():
        probs = F.softmax(logits, dim=-1)  # (batch, seq_len, num_experts)
        avg_prob = probs.mean(dim=(0, 1))  # (num_experts,) - avg prob per expert
        task_expert_probs[task_name] = avg_prob
    
    # Task-expert affinity matrix: (num_tasks, num_experts)
    affinity_matrix = torch.stack([task_expert_probs[t] for t in task_names])
    
    # 1. Expert overlap: how much do tasks share experts?
    overlap_matrix = torch.zeros(num_tasks, num_tasks)
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            # Cosine similarity between task routing distributions
            cosine_sim = F.cosine_similarity(
                affinity_matrix[i].unsqueeze(0),
                affinity_matrix[j].unsqueeze(0)
            ).item()
            overlap_matrix[i, j] = cosine_sim
            overlap_matrix[j, i] = cosine_sim
    
    metrics['task_overlap_matrix'] = overlap_matrix.cpu().numpy().tolist()
    metrics['avg_task_overlap'] = overlap_matrix[overlap_matrix > 0].mean().item()
    
    # 2. Expert type classification
    expert_type = []
    for expert_id in range(num_experts):
        expert_probs = affinity_matrix[:, expert_id]
        max_prob = expert_probs.max().item()
        dominant_task = task_names[expert_probs.argmax().item()]
        
        # Classify as task-specific (>60%) or shared (<40%)
        if max_prob > 0.6:
            expert_type.append(f"specialized_{dominant_task}")
        elif max_prob < 0.4:
            expert_type.append("shared_general")
        else:
            expert_type.append(f"semi_specialized_{dominant_task}")
    
    metrics['expert_types'] = expert_type
    
    # 3. Task-specific expert percentage
    specialized = sum(1 for et in expert_type if 'specialized' in et)
    metrics['specialized_expert_pct'] = specialized / num_experts
    
    return metrics
```

### 4.4 Expert Diversity Metrics

```python
def measure_expert_diversity(
    expert_weights_list: List[Dict[str, torch.Tensor]],  # State dicts from different experts
    metric: str = 'weight_similarity'
) -> float:
    """
    Measure diversity between experts using weight similarity or feature diversity
    
    Args:
        expert_weights_list: List of expert state_dicts
        metric: 'weight_similarity' or 'feature_diversity'
    
    Returns:
        diversity_score: 0 = all identical, 1 = completely different
    """
    
    num_experts = len(expert_weights_list)
    
    if metric == 'weight_similarity':
        # Compute pairwise cosine similarity of flattened weight vectors
        weight_vectors = []
        
        for expert_dict in expert_weights_list:
            # Flatten all weights
            flat_weights = []
            for param in expert_dict.values():
                flat_weights.append(param.flatten())
            flat_vector = torch.cat(flat_weights)
            weight_vectors.append(flat_vector)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(num_experts):
            for j in range(i+1, num_experts):
                cosine_sim = F.cosine_similarity(
                    weight_vectors[i].unsqueeze(0),
                    weight_vectors[j].unsqueeze(0)
                ).item()
                similarities.append(cosine_sim)
        
        # Diversity = 1 - avg_similarity
        avg_similarity = np.mean(similarities) if similarities else 0
        diversity = 1.0 - avg_similarity
        
    elif metric == 'feature_diversity':
        # Using SVD to compute independent dimensions
        # Each expert learns different feature directions
        all_features = []
        
        for expert_dict in expert_weights_list:
            # Use first layer as feature representation
            first_weight = list(expert_dict.values())[0]  # (out, in)
            all_features.append(first_weight.flatten())
        
        feature_matrix = torch.stack(all_features)  # (num_experts, flattened_size)
        
        # SVD to find principal components
        _, singular_vals, _ = torch.svd(feature_matrix)
        
        # Effective rank: how many dimensions are significantly used
        s_norm = singular_vals / singular_vals[0]
        num_sig_dims = (s_norm > 0.01).sum().item()
        
        max_possible_dims = min(num_experts, len(singular_vals))
        diversity = num_sig_dims / max_possible_dims
    
    return diversity
```

---

## 5. Scaling Laws

### 5.1 Parameter Efficiency: Compute-Optimal Scaling

**Dense Model Scaling Law (Chinchilla):**
```
FLOPs ≈ 6ND  (where N = parameters, D = tokens)
Optimal: N ≈ D (equal compute spent on parameters vs data)
```

**MoE Scaling Law:**
```
Total Parameters = Dense_Parameters + (Num_Experts - 1) × Expert_Parameters
Active Parameters per token = Dense_Parameters + k × Expert_Parameters

Example:
Dense model: 70B parameters
MoE: 70B dense + 256 experts × 5B each = 70B + 1.2T total
Active per token: 70B + 2 × 5B = 80B (vs 70B baseline)

Compute grows slowly while parameters scale 10x!
```

**Compute-Optimal Allocation:**

```python
def compute_optimal_moe_config(
    target_flops: float,  # FLOPs budget (e.g., 1e21 for 1 zettaFLOP)
    dense_layers: int = 96,  # Number of dense layers (attention + pre-MoE)
    sparse_layers: int = 96,  # Number of MoE layers
    d_model: int = 12288
) -> Dict[str, int]:
    """
    Compute compute-optimal MoE configuration given FLOPs budget.
    
    Based on: N ≈ D and MoE scales compute-efficiently with expert count
    """
    
    # FLOPs breakdown per token in dense layers
    dense_flops_per_token = 2 * dense_layers * d_model ** 2
    
    # FLOPs in MoE layer: 2 * (dense parameters) + 2 * k * (expert parameters)
    # For routing efficiency: use top-k with k relatively small
    
    k = 2  # top-2 routing
    
    # Solve for num_experts and d_ffn given FLOPs budget
    # FLOPs = dense_flops + sparse_flops
    #  = T * dense_flops_per_token + T * sparse_flops_per_token
    
    # Let's use iterative search
    for num_experts in [32, 64, 128, 256, 512, 1024]:
        for d_ffn in [5 * d_model, 8 * d_model, 10 * d_model]:
            expert_params = (d_model * d_ffn) * 2  # Two layers
            
            # Total FLOPs per token
            routing_flops = d_model * num_experts  # Router computation
            expert_flops = k * 2 * d_model * d_ffn  # Active experts
            sparse_flops_per_token = routing_flops + expert_flops
            
            total_flops_per_token = dense_flops_per_token + sparse_flops_per_token
            
            # Number of tokens we can train with budget
            num_tokens = target_flops / total_flops_per_token
            
            # Chinchilla: N ≈ D
            dense_params = d_model ** 2 * 12 * dense_layers  # Approximate
            expert_total_params = expert_params * num_experts
            
            return {
                'num_experts': num_experts,
                'd_ffn': int(d_ffn),
                'd_model': d_model,
                'expert_params': int(expert_params),
                'total_params': int(dense_params + expert_total_params),
                'tokens_for_budget': int(num_tokens),
                'flops_per_token': int(total_flops_per_token)
            }
```

### 5.2 Training Efficiency vs Dense Models

**Empirical Results:**

```
Model          | Parameters | FLOPs    | Dense Equivalent | Training Time | Speed Gain
GLaM           | 1.2T       | 1.6e21   | ~700B            | 168 days      | 2.0x
Switch-Base    | 1.6T       | 6.1e20   | ~500B            | 256 days      | 1.8x
DeepSeek-V3    | 685B       | 1.0e21   | ~400B            | 128 days      | 2.3x
Mixtral 8x7B   | 45B        | 1.2e19   | ~20B             | 8 days        | 2.5x
```

**Why MoE is Faster:**
1. Sparse computation: Only k/E fraction of parameters active
2. Expert parallelism: Distribute experts across more GPUs
3. Memory efficiency: Smaller batch per expert allows larger effective batch
4. Better capacity utilization: Fewer communication bottlenecks

**Training Throughput Comparison:**

```python
def estimate_training_throughput(
    model_config: Dict,  # num_params, num_experts, k, d_model, etc.
    num_gpus: int,
    gpu_memory_gb: float = 80
) -> Dict[str, float]:
    """
    Estimate training throughput (tokens/GPU/second) for MoE vs Dense
    """
    
    num_experts = model_config.get('num_experts', 256)
    k = model_config.get('k', 2)
    d_model = model_config.get('d_model', 12288)
    
    # Dense model computation
    dense_matmul_time = d_model ** 2 / (TFLOPS_per_GPU * 1e12)  # seconds
    
    # MoE sparse computation
    # Routing overhead
    routing_time = (d_model * num_experts) / (TFLOPS_per_GPU * 1e12)
    
    # Expert computation (k experts active)
    expert_time = k * (2 * d_model * (4 * d_model)) / (TFLOPS_per_GPU * 1e12)
    
    # Communication overhead (all-to-all for expert parallelism)
    comm_bandwidth_gbps = 600  # H100 NVLink
    token_bytes = d_model * 4  # FP32
    comm_time = (token_bytes * num_experts) / (comm_bandwidth_gbps * 1e9)
    
    # Total time per token
    total_time = routing_time + expert_time + comm_time
    
    return {
        'routing_time_ms': routing_time * 1000,
        'expert_time_ms': expert_time * 1000,
        'comm_time_ms': comm_time * 1000,
        'total_time_per_token_ms': total_time * 1000,
        'tokens_per_gpu_per_sec': 1.0 / total_time
    }
```

### 5.3 Inference Cost Analysis

**Memory Requirements:**

```
Dense Model (70B):
  Weights: 70B × 4 bytes (FP32) = 280 GB
  Activations: batch × seq_len × d_model × 4 = ~100 GB
  Total: ~380 GB

MoE Model (70B dense + 256 experts × 5B):
  Dense weights: 70B × 4 = 280 GB
  Expert weights: 256 × 5B × 4 = 5,120 GB
  With expert parallelism (256 GPUs): 280 GB + 20 GB per GPU
  
  But only k=2 experts loaded per GPU at inference:
  280 GB + 2 × 5B × 4 ≈ 320 GB per GPU (manageable!)
```

**Inference Latency:**

```python
def compute_inference_latency(
    model_config: Dict,
    batch_size: int,
    seq_length: int,
    num_gpus_intra: int = 1  # Tensor parallel
) -> Dict[str, float]:
    """
    Compute end-to-end inference latency for MoE
    """
    
    d_model = model_config['d_model']
    num_experts = model_config['num_experts']
    num_layers = model_config['num_layers']
    k = model_config['k']
    
    # Time per token forward pass
    # Attention: O(seq_len^2 * d_model / num_gpus_intra)
    attention_time = 2 * seq_length * d_model / (TFLOPS_per_GPU * num_gpus_intra) * 1000
    
    # Router: O(d_model * num_experts)
    routing_time = (d_model * num_experts) / (TFLOPS_per_GPU * 1e12) * 1000
    
    # Expert: O(k * d_model * ffn_dim)
    expert_time = (k * 2 * d_model * (4 * d_model)) / (TFLOPS_per_GPU * 1e12) * 1000
    
    # All-to-all communication (AlltoAll)
    expert_parallel_factor = num_experts / batch_size  # How many experts per GPU
    comm_volume = batch_size * seq_length * d_model * 4  # bytes
    network_bandwidth_gbps = 400  # Example: GPUs interconnect
    comm_time = comm_volume / (network_bandwidth_gbps * 1e9) * 1000
    
    # Total per layer
    per_layer_time = attention_time + routing_time + expert_time + comm_time
    total_time = per_layer_time * num_layers
    
    return {
        'attention_ms': attention_time,
        'routing_ms': routing_time,
        'expert_ms': expert_time,
        'comm_ms': comm_time,
        'per_layer_ms': per_layer_time,
        'total_latency_ms': total_time,
        'tokens_per_second': (seq_length * 1000) / total_time
    }
```

### 5.4 Cost Models and Trade-Offs

**Training Time vs Model Size Trade-off:**

```
Model Type    | Size   | FLOPs   | Est. Days on 256 GPUs | Cost (GPU hours)
Dense         | 70B    | 1.4e21  | 100                   | 256 × 100 × 24 = 614,400
MoE (1.2T)    | 1.2T   | 1.4e21  | 50 (2x faster!)       | 256 × 50 × 24 = 307,200
MoE inference | -      | Per token compute reduced by 10x
```

**Cost vs Accuracy Trade-off:**

```python
def plot_scaling_law(model_type: str = 'moe'):
    """
    Empirical scaling laws: How much better does model get with more compute?
    """
    
    if model_type == 'dense':
        # Chinchilla scaling law: loss ~ C^(-α) where α ≈ 0.07-0.08
        # Doubling compute → 5-7% loss improvement
        log_compute = np.arange(19, 22, 0.1)  # log10(FLOPs) from 1e19 to 1e22
        loss = 1.69 - 0.13 * (log_compute - 19)
        
    elif model_type == 'moe':
        # MoE typically achieves lower loss at same compute
        # Can route sparsely to reduce effective compute
        # But total parameters grow faster
        log_compute = np.arange(19, 22, 0.1)
        loss = 1.60 - 0.15 * (log_compute - 19)  # Better intercept, similar slope
    
    return {'compute': 10 ** log_compute, 'loss': loss}
```

---

## 6. Training Techniques

### 6.1 Initialization Strategies for Experts

```python
class MoEExpertLayer(nn.Module):
    """Expert initialization strategies for stable training"""
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_experts: int,
        init_strategy: str = 'xavier_uniform'
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            self._init_expert(d_model, d_ffn, init_strategy)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
    
    def _init_expert(self, d_model: int, d_ffn: int, strategy: str) -> nn.Module:
        """Initialize a single expert with different strategies"""
        
        expert = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
        )
        
        if strategy == 'xavier_uniform':
            # Standard: initialize each weight matrix independently
            for module in expert:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        elif strategy == 'similar_initialization':
            # DeepSeek & Mixtral: Initialize all experts very similarly
            # Helps with stable training (experts start with similar capabilities)
            # They specialize through gradient updates
            base_expert = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model)
            )
            nn.init.xavier_uniform_(base_expert[0].weight)
            nn.init.zeros_(base_expert[0].bias)
            nn.init.xavier_uniform_(base_expert[2].weight)
            nn.init.zeros_(base_expert[2].bias)
            
            # Copy weights to all experts with tiny noise
            for module in expert:
                if isinstance(module, nn.Linear):
                    with torch.no_grad():
                        module.weight.copy_(base_expert[0].weight if isinstance(module, type(expert[0])) else base_expert[2].weight)
        
        elif strategy == 'scale_initialization':
            # GShard: Scale variance based on expected number of inputs
            expected_loads = 1.0 / num_experts  # Assume balanced load
            
            for module in expert:
                if isinstance(module, nn.Linear):
                    in_features = module.weight.shape[1]
                    # Scale by 1/sqrt(expected load) to maintain activation scale
                    std = np.sqrt(1.0 / (in_features * expected_loads))
                    nn.init.normal_(module.weight, std=std)
                    nn.init.zeros_(module.bias)
        
        elif strategy == 'orthogonal':
            # Some research: init experts with orthogonal matrices
            for module in expert:
                if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
                    nn.init.orthogonal_(module.weight)
        
        return expert
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard MoE forward pass"""
        logits = self.router(x)
        expert_mask = torch.nn.functional.gumbel_softmax(logits, hard=True)
        
        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            output = output + expert_mask[:, :, expert_id:expert_id+1] * expert(x)
        
        return output
```

### 6.2 Gradient Flow Through Sparse Routing

**Straight-Through Estimator for Discrete Routing:**

```python
class StraightThroughRouter(nn.Module):
    """
    Hard routing (discrete expert selection) with gradient propagation.
    
    Forward: Select discrete expert (hard top-k)
    Backward: Propagate gradients through all top-k experts (soft weights)
    """
    
    def __init__(self, d_model: int, num_experts: int, k: int = 2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.k = k
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            expert_indices: (batch, seq_len, k) - discrete expert selections
            weights: (batch, seq_len, k) - soft weights for gradient flow
        """
        logits = self.router(x)  # (batch, seq_len, num_experts)
        
        # Soft routing for gradients
        weights_soft = F.softmax(logits, dim=-1)
        top_weights_soft, top_indices_soft = torch.topk(weights_soft, k=self.k, dim=-1)
        
        # Hard selection for forward pass
        top_logits, top_indices = torch.topk(logits, k=self.k, dim=-1)
        weights_hard = F.softmax(top_logits, dim=-1)
        
        # Straight-through estimator: hard in forward, soft in backward
        weights_ste = weights_hard - weights_soft.detach() + weights_soft
        
        return top_indices, weights_ste
```

**Gradient Scaling to Prevent Explosion:**

```python
class GradientScaledMoE(nn.Module):
    """
    MoE routing with gradient scaling to prevent:
    1. Gradient explosion (all-to-all comm causes sync issues)
    2. Dead experts (experts with zero gradient)
    3. Imbalanced gradient flow
    """
    
    def __init__(self, d_model: int, num_experts: int, k: int = 2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.k = k
        self.num_experts = num_experts
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: routed and processed output
            expert_weights: soft routing weights
            load_balance_loss: auxiliary loss to regularize
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute routing probabilities
        logits = self.router(x)  # (batch, seq_len, num_experts)
        
        # Normalize logits per token (critical for numerical stability)
        logits_norm = logits - logits.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(logits_norm, dim=-1)
        
        # Select top-k
        top_probs, top_indices = torch.topk(probs, k=self.k, dim=-1)
        
        # Gradient scaling: scale by 1/k to keep magnitude stable
        # This prevents gradients from exploding when summing across experts
        top_probs_scaled = top_probs / (self.k + 1e-10)
        
        # Compute load balance loss (auxiliary)
        expert_loads = probs.sum(dim=(0, 1))  # (num_experts,)
        load_variance = (expert_loads.std() / (expert_loads.mean() + 1e-10)).item()
        
        # Loss to encourage balanced routing
        ideal_load = batch_size * seq_len / self.num_experts
        load_balance = torch.sum((expert_loads - ideal_load) ** 2)
        
        return top_probs_scaled, top_indices, load_balance
```

### 6.3 Synchronization in Distributed MoE

**All-to-All Communication Pattern:**

```python
def moe_all_to_all_dispatch(
    tokens: torch.Tensor,  # (batch, seq_len, d_model)
    expert_indices: torch.Tensor,  # (batch, seq_len, k) - expert selections
    num_experts: int,
    world_size: int,  # number of GPUs
    rank: int  # current GPU rank
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    All-to-all communication pattern for MoE:
    - Each GPU has some experts
    - Tokens need to be routed to the GPU hosting their selected expert
    - Classic MoE uses all-to-all to gather tokens
    
    Communication cost: O(batch * seq_len * d_model)
    """
    
    batch_size, seq_len, d_model = tokens.shape
    k = expert_indices.shape[-1]
    
    # Map global expert ID to GPU rank
    experts_per_gpu = num_experts // world_size
    
    # Determine destination GPU for each token
    # token (b, s) with expert e → GPU rank = e // experts_per_gpu
    dest_ranks = expert_indices // experts_per_gpu  # (batch, seq_len, k)
    
    # Flatten and create send/receive buffers
    flat_tokens = tokens.reshape(-1, d_model)  # (batch*seq_len, d_model)
    flat_expert_indices = expert_indices.reshape(-1)
    flat_dest_ranks = dest_ranks.reshape(-1)
    
    # Count tokens per destination
    num_tokens_to_rank = torch.zeros(world_size, device=tokens.device, dtype=torch.long)
    for dest in flat_dest_ranks:
        num_tokens_to_rank[dest] += 1
    
    # Prepare all-to-all: tell each GPU how many tokens it will receive
    counts_per_rank = [torch.zeros(world_size, dtype=torch.long) for _ in range(world_size)]
    for dest, count in enumerate(num_tokens_to_rank):
        counts_per_rank[rank][dest] = count
    
    # Distributed all-to-all
    dist.all_to_all(counts_per_rank, counts_per_rank)  # sync counts
    
    # Prepare tokens for sending
    send_buffer = torch.zeros(flat_tokens.shape, device=tokens.device)
    send_idx = 0
    for dest_rank in range(world_size):
        num_tokens_to_send = (flat_dest_ranks == dest_rank).sum()
        indices = torch.where(flat_dest_ranks == dest_rank)[0]
        send_buffer[send_idx:send_idx + num_tokens_to_send] = flat_tokens[indices]
        send_idx += num_tokens_to_send
    
    # All-to-all comm
    recv_buffer = torch.zeros_like(send_buffer)
    dist.all_to_all_single(recv_buffer, send_buffer)
    
    # Metadata for routing back (for combine step)
    sent_count = [num_tokens_to_rank[i].item() for i in range(world_size)]
    recv_count = [counts_per_rank[rank][i].item() for i in range(world_size)]
    
    return recv_buffer, {'sent': sent_count, 'recv': recv_count}
```

**All-Gather with Selective Collection (More Efficient):**

```python
def moe_all_gather_selective(
    tokens: torch.Tensor,  # (batch, seq_len, d_model)
    expert_indices: torch.Tensor,  # Which experts selected which tokens
    num_experts: int,
    rank: int,
    world_size: int
) -> torch.Tensor:
    """
    Alternative: Instead of all-to-all, use targeted all-gather.
    More efficient for sparse routing where tokens go to few GPUs.
    """
    
    # Determine which other GPUs need tokens from this GPU
    experts_per_gpu = num_experts // world_size
    local_expert_ids = torch.arange(
        rank * experts_per_gpu,
        (rank + 1) * experts_per_gpu
    )
    
    # Find tokens targeting local experts
    is_local = torch.isin(expert_indices.reshape(-1), local_expert_ids)
    local_tokens_mask = is_local.reshape(expert_indices.shape[:-1])
    
    local_tokens = tokens[local_tokens_mask]  # Compact representation
    
    # Broadcast local tokens to GPUs that need them
    # (In practice: use custom NCCL kernels or PyTorch distributed)
    
    # Each GPU requests tokens it needs
    import torch.distributed as dist
    
    # Gather from all GPUs
    gathered_tokens_list = [torch.zeros_like(tokens) for _ in range(world_size)]
    dist.all_gather(gathered_tokens_list, tokens)
    
    # Filter to only what this GPU needs
    needed_tokens = []
    for other_rank, other_tokens in enumerate(gathered_tokens_list):
        # Check if any expert on this GPU is selected from other_rank
        other_experts_ids = torch.arange(
            other_rank * experts_per_gpu,
            (other_rank + 1) * experts_per_gpu
        )
        need_from_other = torch.any(
            torch.isin(expert_indices, other_experts_ids)
        )
        if need_from_other:
            needed_tokens.append(other_tokens)
    
    if needed_tokens:
        return torch.cat(needed_tokens, dim=0)
    else:
        return torch.zeros_like(tokens)
```

### 6.4 Collective Communication Optimization

**Overlapping Computation and Communication:**

```python
class OverlappedMoELayer(nn.Module):
    """
    Overlap expert computation with communication to hide latency.
    """
    
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Linear(4 * d_model, d_model)
            )
            for _ in range(num_experts)
        ])
    
    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, d_model)
        all_to_all_handle=None  # PyTorch dist handle
    ) -> torch.Tensor:
        """
        Overlapping forward pass:
        1. Start all-to-all communication (async)
        2. Compute local expert work while comm proceeds
        3. Sync on communication when needed
        """
        
        batch_size, seq_len, d_model = x.shape
        
        # Route
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        _, top_indices = torch.topk(probs, k=2, dim=-1)
        
        # Dispatch: async all-to-all
        if all_to_all_handle is None:
            # Synchronous (no overlap)
            dispatched = x  # Simplified: assume tokens already dispatched
        else:
            # Asynchronous: start communication
            import torch.distributed as dist
            all_to_all_handle = dist.all_to_all_single(
                torch.zeros_like(x),
                x,
                async_op=True  # Don't wait for completion
            )
            
            # While comm proceeds, compute something useful
            # (In real impl: compute local experts or other layers)
            
            # Wait for communication to complete
            all_to_all_handle.wait()
            dispatched = x  # Results from all-to-all
        
        # Compute experts
        output = torch.zeros_like(dispatched)
        for expert_id, expert in enumerate(self.experts):
            expert_output = expert(dispatched)
            # Mask and accumulate
            output = output + expert_output  # Simplified
        
        return output
```

**Ring All-Reduce for Gradient Synchronization:**

```python
def ring_allreduce_gradients(
    grads: List[torch.Tensor],
    num_steps: int = 2  # log(world_size) steps
) -> List[torch.Tensor]:
    """
    Ring AllReduce: more efficient than all-reduce for large number of GPUs.
    
    Instead of O(log N) steps with binary tree (less bandwidth utilization),
    ring uses 2(N-1) steps with continuous bandwidth utilization.
    Particularly good for gradients where all-reduce is most expensive.
    """
    
    import torch.distributed as dist
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Split each gradient into chunks for pipelining
    num_chunks = world_size
    
    # Forward pass: reduce+scatter
    for step in range(num_chunks):
        send_rank = (rank - 1) % world_size
        recv_rank = (rank + 1) % world_size
        
        send_buffer = grads[step % len(grads)]
        recv_buffer = torch.zeros_like(send_buffer)
        
        # Ring send/recv
        dist.send(send_buffer, send_rank)
        dist.recv(recv_buffer, recv_rank)
        
        grads[step % len(grads)] = recv_buffer
    
    # Backward pass: gather
    for step in range(num_chunks - 1):
        # Similar ring pattern for all-gather
        pass
    
    return grads
```

---

## 7. Implementation Frameworks

### 7.1 Megatron-LM MoE Modules

Key components from Megatron-Core MoE (2026 update):

```python
class MegatronMoELayer(nn.Module):
    """
    Megatron-Core's scalable MoE layer with:
    - Flexible parallelism (Expert, Data, Tensor, Pipeline parallel)
    - Memory optimizations (fine-grained recomputation, offloading)
    - Communication optimizations (overlapping, grouped GEMM)
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_parallel_size: int,
        data_parallel_size: int,
        pipeline_parallel_size: int,
        capacity_factor: float = 1.25,
        drop_tokens: bool = False
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_parallel_size = expert_parallel_size
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        
        # Router
        self.router = nn.Linear(d_model, num_experts)
        self.router_bias = nn.Parameter(torch.zeros(num_experts))
        
        # Local experts (subset on this GPU)
        experts_per_gpu = num_experts // expert_parallel_size
        self.local_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
            for _ in range(experts_per_gpu)
        ])
    
    def forward(self, x: torch.Tensor, tokens_per_expert: int = None) -> torch.Tensor:
        """
        Megatron's parallel folding approach: decouple expert, data, and tensor parallelism
        """
        batch_size, seq_len, d_model = x.shape
        
        # Route with dynamic bias (for load balancing)
        logits = self.router(x) + self.router_bias.unsqueeze(0).unsqueeze(0)
        
        # Top-2 or top-1 routing
        top_logits, expert_indices = torch.topk(logits, k=2, dim=-1)
        weights = F.softmax(top_logits, dim=-1)
        
        # Capacity constraint
        if tokens_per_expert is None:
            tokens_per_expert = int(
                self.capacity_factor * (batch_size * seq_len) / self.num_experts
            )
        
        # Dispatch: all-to-all to route tokens to correct expert GPU
        # (Megatron uses optimized NCCL-based dispatcher)
        dispatched_tokens, _ = self._dispatch_tokens(x, expert_indices)
        
        # Compute with local experts
        expert_output = torch.zeros_like(dispatched_tokens)
        for expert_id, expert in enumerate(self.local_experts):
            expert_output = expert_output + expert(dispatched_tokens)
        
        # Combine: all-to-all to gather results back
        output = self._combine_outputs(expert_output, weights)
        
        return output
    
    def _dispatch_tokens(self, tokens: torch.Tensor, expert_indices: torch.Tensor):
        """
        Route tokens to experts via all-to-all.
        Megatron uses optimized token dispatcher with:
        - Grouped GEMM for sparse operations
        - Zero-copy communication
        """
        # Placeholder: real implementation in Megatron uses CUDA kernels
        return tokens, expert_indices
    
    def _combine_outputs(self, expert_output: torch.Tensor, weights: torch.Tensor):
        """Weighted combination of expert outputs"""
        return expert_output  # Simplified
```

### 7.2 DeepSpeed MoE Support

```python
class DeepSpeedMoELayer(nn.Module):
    """
    DeepSpeed-MoE with ZeRO-optimized gradient handling and
    dynamic token redistribution for inference/training
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        mlp_type: str = 'standard',  # or 'residual' (PR-MoE)
        deepspeed_config: dict = None
    ):
        super().__init__()
        self.num_experts = num_experts
        self.mlp_type = mlp_type
        
        self.router = nn.Linear(d_model, num_experts)
        
        if mlp_type == 'standard':
            self.experts = nn.ModuleList([
                self._create_expert(d_model)
                for _ in range(num_experts)
            ])
        elif mlp_type == 'residual':
            # PR-MoE: residual path
            self.experts = nn.ModuleList([
                self._create_expert(d_model)
                for _ in range(num_experts)
            ])
            self.dense_mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
    
    def _create_expert(self, d_model: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Route
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        
        # Top-1 routing (DeepSpeed default)
        expert_indices = probs.argmax(dim=-1)  # (batch, seq_len)
        expert_probs = probs.max(dim=-1)[0]  # (batch, seq_len)
        
        # Standard forward pass (simplified)
        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            mask = expert_indices == expert_id
            if mask.any():
                output[mask] = expert(x[mask])
        
        if self.mlp_type == 'residual':
            output = output + self.dense_mlp(x)
        
        return output
```

### 7.3 Fairseq MoE Implementation

```python
class FairseqMoELayer(nn.Module):
    """
    Fairseq's MoE layer (used in MT research)
    - Simpler interface
    - Good for research/prototyping
    - Supports gating functions, auxiliary losses
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_ffn_size: int = None,
        gating: str = 'learned',
        gate_noise_scale: float = 0.1,
        expert_dropout: float = 0.0
    ):
        super().__init__()
        
        if expert_ffn_size is None:
            expert_ffn_size = 4 * d_model
        
        self.num_experts = num_experts
        self.gate_noise_scale = gate_noise_scale
        
        # Gates (routers)
        self.gate = nn.Linear(d_model, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_ffn_size),
                nn.ReLU(),
                nn.Dropout(expert_dropout),
                nn.Linear(expert_ffn_size, d_model)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            metrics: {gate_load, auxiliary_loss, ...}
        """
        batch_size, seq_len, d_model = x.shape
        
        # Gating with noise for exploration
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        
        if self.training and self.gate_noise_scale > 0:
            # Add noise to encourage exploration
            gate_noise = torch.randn_like(gate_logits) * self.gate_noise_scale
            gate_logits = gate_logits + gate_noise
        
        gate_outputs = F.softmax(gate_logits, dim=-1)
        
        # Load balancing loss
        gate_load = gate_outputs.mean(0)  # (num_experts,)
        importance = gate_outputs.max(dim=-1)[0].mean()
        load_loss = torch.sum(gate_load * importance)
        
        # Expert forward
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (..., num_experts)
        
        # Mixture
        output = torch.einsum('...e,e->...', expert_outputs, gate_outputs.mean(dim=(0, 1)))
        
        return output, {'load_loss': load_loss}
```

### 7.4 Custom MoE Building from Scratch

Complete working example:

```python
class CustomMoE(nn.Module):
    """
    Minimal but complete MoE implementation from scratch.
    Good for understanding all components.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_experts: int = 8,
        expert_ffn_dim: int = 3072,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        use_aux_loss: bool = True,
        aux_loss_weight: float = 0.01
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight
        
        # Router network
        self.router = nn.Linear(d_model, num_experts)
        
        # Expert networks (each is a simple 2-layer FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_ffn_dim),
                nn.GELU(),
                nn.Linear(expert_ffn_dim, d_model)
            )
            for _ in range(num_experts)
        ])
        
        self.aux_loss = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Route tokens to experts
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_probs, top_expert_indices = torch.topk(
            router_probs, k=self.top_k, dim=-1
        )  # Both (batch, seq_len, k)
        
        # Normalize top-k probabilities
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        
        # Step 2: Compute auxiliary loss (for load balancing)
        if self.use_aux_loss:
            self.aux_loss = self._compute_aux_loss(router_probs, top_expert_indices)
        
        # Step 3: Forward through selected experts
        output = torch.zeros_like(x)
        
        for expert_id in range(self.num_experts):
            # Find tokens that selected this expert
            is_selected = (top_expert_indices == expert_id).any(dim=-1)  # (batch, seq_len)
            
            if not is_selected.any():
                continue
            
            # Get the weights for this expert (where it was selected)
            expert_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
            for k_idx in range(self.top_k):
                mask = top_expert_indices[:, :, k_idx] == expert_id
                expert_weights[mask, 0] = top_probs[mask, k_idx]
            
            # Forward through expert
            expert_out = self.experts[expert_id](x)  # (batch, seq_len, d_model)
            
            # Weighted contribution
            output = output + expert_out * expert_weights
        
        return output
    
    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,  # (batch, seq_len, num_experts)
        expert_indices: torch.Tensor  # (batch, seq_len, k)
    ) -> torch.Tensor:
        """
        Compute importance-weighted load balancing loss.
        
        Encourages uniform token distribution across experts.
        """
        batch_size, seq_len, num_experts = router_probs.shape
        
        # Fraction of tokens to each expert
        num_tokens_per_expert = torch.zeros(num_experts, device=router_probs.device)
        for b in range(batch_size):
            for s in range(seq_len):
                for expert_id in expert_indices[b, s]:
                    num_tokens_per_expert[expert_id] += 1
        
        token_fractions = num_tokens_per_expert / (batch_size * seq_len)
        
        # Average router probability per expert
        avg_probs = router_probs.sum(dim=(0, 1)) / (batch_size * seq_len)
        
        # Loss: encouraging balanced routing
        aux_loss = torch.sum(token_fractions * avg_probs)
        
        return self.aux_loss_weight * aux_loss
    
    def get_aux_loss(self) -> torch.Tensor:
        """Return auxiliary loss for optimization"""
        return self.aux_loss


# Usage example
def train_moe_model():
    model = CustomMoE(
        d_model=768,
        num_experts=8,
        expert_ffn_dim=3072,
        top_k=2,
        use_aux_loss=True,
        aux_loss_weight=0.001
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Dummy data
    x = torch.randn(2, 64, 768)  # (batch, seq_len, d_model)
    
    # Forward
    output = model(x)
    
    # Compute loss
    task_loss = output.mean()  # Dummy task loss
    aux_loss = model.get_aux_loss()
    
    total_loss = task_loss + aux_loss
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
```

---

## 8. Code Examples

### 8.1 Basic MoE Layer from Scratch

(Already covered in Section 7.4 - `CustomMoE`)

### 8.2 Top-K Routing Implementation

```python
def top_k_routing(
    x: torch.Tensor,  # (batch, seq_len, d_model)
    router_weights: torch.Tensor,  # (d_model, num_experts)
    k: int = 2,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of top-k expert routing.
    """
    batch_size, seq_len, d_model = x.shape
    num_experts = router_weights.shape[1]
    
    # Step 1: Compute routing logits
    logits = torch.matmul(x, router_weights)  # (batch, seq_len, num_experts)
    
    # Step 2: Temperature scaling
    logits = logits / temperature
    
    # Step 3: Select top-k
    top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)
    
    # Step 4: Softmax normalization (only on top-k)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    
    return top_k_probs, top_k_indices, logits
```

### 8.3 Load Balancing Loss

```python
def load_balancing_loss_complete(
    router_logits: torch.Tensor,  # (batch, seq_len, num_experts)
    top_k_indices: torch.Tensor,  # (batch, seq_len, k)
    alpha: float = 0.01
) -> torch.Tensor:
    """
    Complete importance-weighted load balancing loss.
    Matches implementation in Switch Transformers / GShard.
    """
    batch_size, seq_len, num_experts = router_logits.shape
    
    # Compute routing probabilities
    router_probs = F.softmax(router_logits, dim=-1)
    
    # Create mask for selected experts
    selected_mask = torch.zeros_like(router_logits)
    for b in range(batch_size):
        for s in range(seq_len):
            for k_idx in range(top_k_indices.shape[-1]):
                expert_id = top_k_indices[b, s, k_idx]
                selected_mask[b, s, expert_id] = 1.0
    
    # Normalize mask (each token contributes 1/k to each selected expert)
    selected_mask = selected_mask / (top_k_indices.shape[-1] + 1e-10)
    
    # Compute fractions and probabilities
    total_tokens = batch_size * seq_len
    fractions = selected_mask.sum(dim=(0, 1)) / total_tokens
    probs = (router_probs * selected_mask).sum(dim=(0, 1)) / (selected_mask.sum(dim=(0, 1)) + 1e-10)
    
    # Loss
    balance_loss = torch.sum(fractions * probs)
    
    return alpha * balance_loss
```

### 8.4 Full MoE Transformer Block

```python
class MoETransformerBlock(nn.Module):
    """
    Complete transformer block with MoE FFN layer.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        num_experts: int = 8,
        expert_ffn_dim: int = 3072,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        # Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        # MoE FFN
        self.moe = CustomMoE(
            d_model=d_model,
            num_experts=num_experts,
            expert_ffn_dim=expert_ffn_dim,
            top_k=2,
            use_aux_loss=True,
            aux_loss_weight=0.001
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out, _ = self.self_attention(x, x, x, attn_mask=attn_mask)
        attn_out = self.dropout(attn_out)
        x = x + attn_out
        x = self.attn_norm(x)
        
        # MoE FFN with residual
        moe_out = self.moe(x)
        moe_out = self.dropout(moe_out)
        x = x + moe_out
        x = self.ffn_norm(x)
        
        return x
    
    def get_aux_loss(self) -> torch.Tensor:
        return self.moe.get_aux_loss()


class MoETransformer(nn.Module):
    """Full transformer with MoE layers"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        num_layers: int = 12,
        d_model: int = 768,
        num_heads: int = 12,
        num_experts: int = 8,
        expert_ffn_dim: int = 3072,
        max_seq_len: int = 1024
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_experts=num_experts,
                expert_ffn_dim=expert_ffn_dim
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            aux_loss: scalar for load balancing
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(pos_ids)
        
        # Transformer layers
        aux_losses = []
        for layer in self.layers:
            x = layer(x)
            aux_losses.append(layer.get_aux_loss())
        
        # Output
        x = self.final_norm(x)
        logits = self.output_head(x)  # (batch, seq_len, vocab_size)
        
        # Aggregate auxiliary losses
        total_aux_loss = sum(aux_losses) / len(aux_losses)
        
        return logits, total_aux_loss
```

### 8.5 Training Loop with MoE

```python
def train_moe_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    aux_loss_weight: float = 0.001,
    gradient_clip: float = 1.0
) -> Dict[str, float]:
    """
    Single training step for MoE model.
    """
    
    input_ids = batch['input_ids']  # (batch, seq_len)
    target_ids = batch['target_ids']  # (batch, seq_len)
    
    # Forward pass
    logits, aux_loss = model(input_ids)  # logits: (batch, seq_len, vocab)
    
    # Task loss (language modeling)
    task_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1)
    )
    
    # Total loss: task + auxiliary
    total_loss = task_loss + aux_loss_weight * aux_loss
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    
    # Optimization step
    optimizer.step()
    
    return {
        'task_loss': task_loss.item(),
        'aux_loss': aux_loss.item(),
        'total_loss': total_loss.item()
    }


def training_loop(
    model: nn.Module,
    train_loader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    aux_loss_weight: float = 0.001,
    device: str = 'cuda'
):
    """
    Complete training loop for MoE model.
    """
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_task_loss = 0.0
        total_aux_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Training step
            losses = train_moe_step(
                model, batch, optimizer,
                aux_loss_weight=aux_loss_weight
            )
            
            total_task_loss += losses['task_loss']
            total_aux_loss += losses['aux_loss']
            
            if (batch_idx + 1) % 100 == 0:
                avg_task_loss = total_task_loss / (batch_idx + 1)
                avg_aux_loss = total_aux_loss / (batch_idx + 1)
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"task_loss={avg_task_loss:.4f}, "
                      f"aux_loss={avg_aux_loss:.6f}")
        
        print(f"Epoch {epoch} completed")
```

### 8.6 Inference Optimization

```python
@torch.no_grad()
def moe_inference_optimized(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    use_cache: bool = True,
    expert_cache: Dict = None
) -> torch.Tensor:
    """
    Optimized inference for MoE models:
    - Expert specialization caching
    - Sparse computation (skip unused experts)
    - KV-cache for attention
    """
    
    batch_size, seq_len = input_ids.shape
    generated = input_ids.clone()
    
    if expert_cache is None:
        expert_cache = {}
    
    for step in range(max_new_tokens):
        # Get logits
        logits, _ = model(generated[:, -1:])  # Only compute for last token
        
        # Sample next token
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Append to sequence
        generated = torch.cat([generated, next_token_id], dim=-1)
        
        # Optionally cache expert outputs for common patterns
        if step % 10 == 0:
            # Clear cache to avoid memory issues
            expert_cache.clear()
    
    return generated


def benchmark_moe_inference(
    model: nn.Module,
    batch_size: int = 1,
    seq_length: int = 512,
    num_batches: int = 100
) -> Dict[str, float]:
    """
    Benchmark MoE inference performance.
    """
    
    import time
    
    model.eval()
    
    # Warmup
    dummy_input = torch.randint(0, 50000, (batch_size, seq_length))
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_batches):
        dummy_input = torch.randint(0, 50000, (batch_size, seq_length))
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    total_tokens = num_batches * batch_size * seq_length
    
    return {
        'total_time_s': total_time,
        'tokens_per_second': total_tokens / total_time,
        'latency_ms_per_seq': (total_time / num_batches) * 1000
    }
```

---

## 9. Advanced Topics

### 9.1 Sparse Gating Operations (cuSPARSE)

```python
def sparse_gating_forward_cusparse(
    x: torch.Tensor,  # (batch, seq_len, d_model)
    router_weights: torch.Tensor,  # (d_model, num_experts)
    k: int = 2,
    use_cusparse: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use NVIDIA cuSPARSE for efficient sparse matrix operations.
    
    When only k out of E experts are active, the computation
    can be accelerated with sparse GEMM (General Matrix Multiply).
    """
    
    batch_size, seq_len, d_model = x.shape
    num_experts = router_weights.shape[1]
    
    # Compute all routing logits
    logits = torch.matmul(x, router_weights)  # (batch*seq_len, num_experts)
    
    # Get top-k
    top_k_logits, top_k_indices = torch.topk(logits.reshape(-1, num_experts), k=k, dim=-1)
    
    if use_cusparse:
        # Create sparse adjacency matrix: which tokens go to which experts
        # Shape: (batch*seq_len, num_experts), only top-k positions are non-zero
        
        from torch_sparse import SparseTensor
        
        # Flatten indices
        num_tokens = batch_size * seq_len
        row_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(k)
        col_indices = top_k_indices.reshape(-1)
        
        # Create sparse tensor
        sparse_routing = SparseTensor(
            row=row_indices,
            col=col_indices,
            value=F.softmax(top_k_logits.reshape(-1), dim=-1),
            sparse_sizes=(num_tokens, num_experts)
        )
        
        # Note: For real CUDA optimization, use torch.sparse GEMM
        # This is a placeholder showing the concept
        
    return top_k_logits, top_k_indices
```

### 9.2 Expert Computation Scheduling

```python
class ExpertScheduler:
    """
    Schedule expert computation to minimize idle time and memory usage.
    
    Key insight: Not all experts are equally likely; use statistical
    knowledge to schedule computation optimally.
    """
    
    def __init__(self, num_experts: int, history_window: int = 1000):
        self.num_experts = num_experts
        self.expert_freq = torch.zeros(num_experts)
        self.history_window = history_window
    
    def update_frequency(self, expert_indices: torch.Tensor):
        """Update expert frequency statistics"""
        unique, counts = torch.unique(expert_indices.reshape(-1), return_counts=True)
        for exp_id, count in zip(unique, counts):
            self.expert_freq[exp_id] += count
    
    def get_schedule(self, batch_size: int) -> Dict[int, int]:
        """
        Return scheduling: which GPU to compute which expert.
        
        Schedule high-frequency experts first (better cache locality).
        """
        
        # Normalize frequencies
        freq_norm = self.expert_freq / (self.expert_freq.sum() + 1e-10)
        
        # Sort experts by frequency (descending)
        sorted_experts = torch.argsort(freq_norm, descending=True)
        
        schedule = {}
        for rank, expert_id in enumerate(sorted_experts):
            # Assign expert to GPU based on frequency
            gpu_id = rank % torch.cuda.device_count()
            schedule[expert_id.item()] = gpu_id
        
        return schedule
```

### 9.3 Communication-Efficient Sparse Routing

```python
def hierarchical_routing_communication(
    x: torch.Tensor,  # (batch, seq_len, d_model)
    router_weights: torch.Tensor,  # (d_model, num_experts)
    num_experts_per_stage: int = 16,
    k_per_stage: int = 2,
    world_size: int = 8
) -> Tuple[torch.Tensor, Dict]:
    """
    Hierarchical routing to reduce communication:
    
    Instead of selecting k experts from all E globally,
    select k experts in a hierarchical tree structure.
    
    Example: 256 experts → 16 groups → 2 experts within group
    Reduces all-to-all communication by ~8x.
    """
    
    batch_size, seq_len, d_model = x.shape
    num_experts = router_weights.shape[1]
    num_groups = num_experts // num_experts_per_stage
    
    # Stage 1: Select group
    group_router = torch.nn.Linear(d_model, num_groups)
    group_logits = group_router(x)  # (batch, seq_len, num_groups)
    group_logits_hard, group_id = torch.topk(group_logits, k=1, dim=-1)  # Select 1 group
    
    # Stage 2: Within group, select k experts
    selected_groups = group_id.squeeze(-1)
    
    expert_logits = torch.matmul(x, router_weights)  # Full logits
    
    # Mask logits outside selected group
    group_masks = torch.zeros_like(expert_logits)
    for group_idx in range(num_groups):
        mask = selected_groups == group_idx
        group_masks[mask, group_idx*num_experts_per_stage:(group_idx+1)*num_experts_per_stage] = 1.0
    
    masked_logits = expert_logits * group_masks - (1 - group_masks) * 1e9
    
    # Select top-k from within group
    top_logits, top_indices = torch.topk(masked_logits, k=k_per_stage, dim=-1)
    
    return top_indices, {
        'communication_reduction': num_experts // (num_groups * k_per_stage),
        'hierarchical_depth': 2
    }
```

### 9.4 Fine-tuning Pre-trained MoE

```python
def finetune_pretrained_moe(
    pretrained_path: str,
    task_data: torch.utils.data.DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 1e-5,
    freeze_experts: bool = False,
    adapt_routers_only: bool = False
) -> nn.Module:
    """
    Fine-tune a pre-trained MoE model on a new task.
    
    Strategies:
    1. Full fine-tune: update all parameters
    2. Router-only: freeze experts, adapt only routing
    3. LoRA-style: add adapters to experts
    """
    
    # Load pretrained
    model = torch.load(pretrained_path)
    model.eval()
    
    # Strategy 1: Freeze experts
    if freeze_experts:
        for name, param in model.named_parameters():
            if 'expert' in name:
                param.requires_grad = False
    
    # Strategy 2: Only adapt routers
    if adapt_routers_only:
        for name, param in model.named_parameters():
            if 'router' not in name:
                param.requires_grad = False
    
    # Setup optimization
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    model.train()
    
    for epoch in range(num_epochs):
        for batch in task_data:
            # Forward
            logits = model(batch['input_ids'])
            
            # Loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch['target_ids'].reshape(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model
```

---

## 10. Empirical Results

### 10.1 Training Efficiency Comparisons

**Empirical Data from Recent Papers (2024-2026):**

```
Model          | Architecture    | Params | Train FLOPs | Time (256 GPUs) | Efficiency vs Dense
╔══════════════╦═════════════════╦════════╦═════════════╦═════════════════╦═════════════════╗
║ GPT-3        ║ Dense           ║ 175B  ║ 3.1e21      ║ 355 days        ║ 1.0x (baseline) ║
║ Switch-Base  ║ Top-1, 128 exp  ║ 1.6T  ║ 1.3e21      ║ 150 days        ║ 2.37x           ║
║ GLaM         ║ Top-2, 64 exp   ║ 1.2T  ║ 1.6e21      ║ 168 days        ║ 2.11x           ║
║ Mixtral 8x7B ║ Top-2, 8 exp    ║ 45B   ║ 1.2e19      ║ 8 days          ║ 2.5x            ║
║ DeepSeek-V3  ║ DeepSeekMoE     ║ 685B  ║ 1.0e21      ║ 128 days        ║ 2.77x           ║
║ Qwen3 (MoE)  ║ Top-2 balanced  ║ 320B  ║ 4.8e20      ║ 64 days         ║ 3.2x            ║
╚══════════════╩═════════════════╩════════╩═════════════╩═════════════════╩═════════════════╝
```

### 10.2 Expert Utilization Patterns

```python
def analyze_expert_utilization_trends():
    """
    Meta-analysis of expert utilization across architectures.
    Based on DeepSeek-V3, Mixtral, and Skywork research.
    """
    
    results = {
        'switch_transformer': {
            'avg_expert_utilization': 0.87,  # 87% of experts used per batch
            'expert_dropout_rate': 0.12,
            'max_expert_load': 2.3,  # Max vs avg
            'gini_coefficient': 0.15  # 0 = perfect balance, 1 = all in one
        },
        'mixtral_8x7b': {
            'avg_expert_utilization': 0.94,
            'expert_dropout_rate': 0.05,
            'max_expert_load': 1.8,
            'gini_coefficient': 0.08
        },
        'deepseek_v3': {
            'avg_expert_utilization': 0.96,
            'expert_dropout_rate': 0.02,  # Near-zero with loss-free balancing
            'max_expert_load': 1.4,
            'gini_coefficient': 0.04
        }
    }
    
    # Key insight: Newer methods achieve better balance with less intervention
    return results
```

### 10.3 Performance vs Parameter Trade-off

```python
def scaling_comparison():
    """
    Compare FLOPs, parameters, and performance trade-offs.
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    models = ['Dense 70B', 'Dense 175B', 'MoE 256E', 'MoE 512E', 'Mixtral']
    params = [70, 175, 400, 850, 45]  # Billions
    flops = [1.3e20, 3.1e21, 1.4e21, 2.8e21, 1.2e19]
    perplexity = [20.5, 18.3, 17.2, 15.8, 17.5]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Parameters vs FLOPs
    axes[0].scatter(params, np.array(flops) / 1e21)
    axes[0].set_xlabel('Parameters (B)')
    axes[0].set_ylabel('Training FLOPs (e21)')
    axes[0].set_title('Parameter Efficiency of MoE')
    
    # Plot 2: FLOPs vs Perplexity
    axes[1].scatter(np.array(flops) / 1e21, perplexity)
    axes[1].set_xlabel('FLOPs (e21)')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Training Efficiency (lower perp = better)')
    
    # Plot 3: Parameters vs Perplexity
    axes[2].scatter(params, perplexity)
    axes[2].set_xlabel('Parameters (B)')
    axes[2].set_ylabel('Perplexity')
    axes[2].set_title('Parameter Efficiency (MoE > Dense)')
    
    plt.tight_layout()
    # plt.show()
    
    # Key finding: MoE achieves better perplexity at same FLOPs
    # E.g., MoE 256E: 400B params, 1.4e21 FLOPs, 17.2 perplexity
    #       Dense 175B: 175B params, 3.1e21 FLOPs, 18.3 perplexity
    # MoE wins on both parameters and efficiency!
```

### 10.4 Scaling to Billions of Parameters

```python
def estimate_scaling_trajectory(
    target_flops: float = 1e22,  # 10 ZettaFLOPs
    model_type: str = 'moe'  # 'dense' or 'moe'
) -> Dict:
    """
    Extrapolate scaling trajectory given FLOPs budget.
    """
    
    if model_type == 'dense':
        # Chinchilla scaling: N ≈ D
        # Training FLOPs ≈ 6ND
        # With N ≈ D: FLOPs ≈ 6N^2
        # N = sqrt(FLOPs / 6)
        params = np.sqrt(target_flops / 6) / 1e9
        data = np.sqrt(target_flops / 6) / 1e9
        
    else:  # MoE
        # Dense: 70B, Experts: 256 × 100B = 25.6T
        # Total: 25.67T params, but only 70B + 2×100B = 270B active per token
        # Compute: ≈ 6 × 270B × tokens
        
        # For same compute as dense 300B, MoE can have 10x more total params
        params = np.sqrt(target_flops / 6) * 10 / 1e9  # 10x advantage
        data = np.sqrt(target_flops / 6) / 1e9
    
    return {
        'total_parameters_billions': params,
        'training_tokens_billions': data * 1e9,
        'flops_budget': target_flops,
        'model_type': model_type,
        'efficiency_gain': params if model_type == 'moe' else 1.0
    }


# Examples
print("Dense model with 1e22 FLOPs budget:")
dense_result = estimate_scaling_trajectory(1e22, 'dense')
print(f"  Params: {dense_result['total_parameters_billions']:.0f}B")

print("\nMoE model with 1e22 FLOPs budget:")
moe_result = estimate_scaling_trajectory(1e22, 'moe')
print(f"  Params: {moe_result['total_parameters_billions']:.0f}B")
print(f"  Efficiency gain: {moe_result['efficiency_gain']:.1f}x")
```

---

## 11. References

### Foundational Papers

1. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**
   - Lepikhin et al. (2020)
   - URL: https://arxiv.org/abs/2006.16668
   - Key: First large-scale MoE system, top-2 gating, capacity constraints

2. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**
   - Fedus et al. (2021)
   - URL: https://arxiv.org/abs/2101.03961
   - Key: Top-1 routing simplification, 1.6T parameter model

3. **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts**
   - Du et al. (2021)
   - URL: https://arxiv.org/abs/2112.06905
   - Key: Energy-efficient MoE, 1/3 energy of GPT-3

### Load Balancing & Stability

4. **ST-MOE: Designing Stable and Transferable Sparse Expert Models**
   - Zoph et al. (2022)
   - URL: https://arxiv.org/abs/2202.08906
   - Key: Router z-loss for training stability

5. **Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts**
   - Wang et al. (2024)
   - URL: https://arxiv.org/abs/2408.15664
   - Key: Bias-based load balancing without auxiliary loss

6. **A Review on the Evolvement of Load Balancing Strategy in MoE LLMs: Pitfalls and Lessons**
   - Zhang et al. (2025)
   - URL: https://huggingface.co/blog/NormalUhr/moe-balance
   - Key: Historical analysis of MoE routing strategies

### Production Systems

7. **DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training**
   - Rajbhandari et al. (2022)
   - URL: https://arxiv.org/abs/2201.05596
   - Key: Inference optimization, PR-MoE architecture

8. **Scalable Training of Mixture-of-Experts Models with Megatron Core**
   - NVIDIA (2026)
   - URL: https://arxiv.org/abs/2603.07685
   - Key: Latest state-of-the-art training techniques, FP8 training, parallel folding

### Recent Implementations

9. **Mixtral of Experts**
   - Jiang et al. (2024)
   - URL: https://arxiv.org/abs/2401.04088
   - Key: Temporal locality in routing, Megablocks kernels

10. **DeepSeekMoE: Towards Ultimate Expert Specialization**
    - Dai et al. (2024)
    - URL: https://arxiv.org/abs/2401.06066
    - Key: Fine-grained experts, shared experts, device-level balancing

11. **DeepSeek-V3 Technical Report**
    - DeepSeek-AI (2024)
    - Key: Auxiliary-loss-free training, dynamic bias routing, dropless MoE

12. **JetMoE: Reaching Llama2 Performance with 0.1M Dollars**
    - Shen et al. (2024)
    - URL: https://arxiv.org/abs/2404.07410
    - Key: Cost-efficient MoE, pipeline parallelism

13. **Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts**
    - Wei et al. (2024)
    - URL: https://arxiv.org/abs/2406.06563
    - Key: Adaptive auxiliary loss, gating logit normalization

### Sparse Computing

14. **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts**
    - Gale et al. (2021)
    - URL: https://arxiv.org/abs/2211.15841
    - Key: Block-sparse kernels for GPU-efficient MoE computation

15. **Expert Choice Routing Transformers**
    - Zhou et al. (2023)
    - Key: Expert-driven routing instead of token-driven

---

## Summary: Best Practices for MoE Training

### Do's:
- Use gradient clipping and appropriate learning rates (slightly lower than dense)
- Monitor expert utilization metrics continuously
- Start with smaller models (8-64 experts) before scaling up
- Use top-2 routing as a good starting point (better than top-1, simpler than top-k)
- Balance auxiliary loss weight carefully (0.001-0.01 typical range)
- Implement proper load balancing from the beginning
- Profile communication overhead with your specific hardware

### Don'ts:
- Don't ignore auxiliary loss - it's critical for performance
- Don't use capacity factor >2.0 without handling overflow properly
- Don't blindly scale number of experts - relationship to compute is complex
- Don't forget synchronization in distributed training
- Don't run inference without expert parallelism (latency killer)

### When to Use MoE:
- **Good fit:** Large-scale training, strong compute resources, diverse data
- **Poor fit:** Small models (<1B), single GPU training, memory-constrained

### When to Use Dense:
- **Better choice:** Small models, data-constrained, heterogeneous inference hardware

---

**Document Version:** 1.0
**Last Updated:** April 2026
**Research Cutoff:** February 2026

This document reflects the state of MoE research and practice as of early 2026. For latest developments, check:
- https://arxiv.org/list/cs.LG (arXiv machine learning)
- https://huggingface.co/papers (HuggingFace papers)
- GitHub repositories: NVIDIA/Megatron-LM, microsoft/DeepSpeed, deepseek-ai/DeepSeek
