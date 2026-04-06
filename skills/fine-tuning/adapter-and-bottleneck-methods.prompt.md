# Adapter-Based Fine-Tuning: Bottleneck Methods and Composition Strategies

## Overview

Adapter-based fine-tuning represents a paradigm shift in efficient model adaptation. Instead of fine-tuning all model parameters, adapters insert small, trainable bottleneck modules into transformer layers while keeping pretrained weights frozen. This approach achieves remarkable parameter efficiency—3-5% of full fine-tuning parameters while maintaining comparable performance.

**Key Innovation**: Adapters convert the problem from "modify all weights" to "learn small residual adjustments," enabling a single pretrained model to serve dozens of tasks simultaneously with minimal storage overhead.

---

## 1. Adapter Fundamentals

### 1.1 Bottleneck Architecture

The core of adapter design is the bottleneck module—a compressed pathway through which task-specific information flows.

#### Architecture Components

```
Input (d-dim) → Down-Project (d→r) → Activation → Up-Project (r→d) → Output
                                                                    +
                                                        Residual Connection
```

**Three-Component Pipeline**:

1. **Down-Projection**: Linear layer W_down ∈ ℝ^(r×d)
   - Compresses hidden dimension d to bottleneck dimension r
   - Acts as learned feature selector
   - Enables information filtering for task-specific features

2. **Activation Function**: f(·) ∈ {GELU, ReLU, Tanh}
   - Introduces nonlinearity
   - Enables complex feature interactions in compressed space
   - Without this, composition of two linear layers would be linear (undercomplete)

3. **Up-Projection**: Linear layer W_up ∈ ℝ^(d×r)
   - Expands from bottleneck r back to original dimension d
   - Reconstructs full hidden dimension for downstream layers
   - Must be compatible with original layer dimensions

#### Mathematical Formulation

```
Adapter(h) = W_up · f(W_down · h + b_down) + b_up

With Residual Connection:
h' = h + Adapter(h)
```

Where:
- h: input hidden state (d-dimensional)
- W_down ∈ ℝ^(r×d): down-projection matrix
- b_down ∈ ℝ^r: down-projection bias
- f(·): nonlinear activation
- W_up ∈ ℝ^(d×r): up-projection matrix
- b_up ∈ ℝ^d: up-projection bias
- h': output hidden state

### 1.2 Parameter Reduction Analysis

#### Single Adapter Module Parameters

```
Total Parameters = 2rd + r + d
                 ≈ 2rd (when r ≪ d)
```

**Example: BERT-base (d=768, r=64)**
- W_down: 64 × 768 = 49,152 params
- b_down: 64 params
- W_up: 768 × 64 = 49,152 params
- b_up: 768 params
- **Total per adapter: ~98,304 params (0.089% of 110M model)**

#### Comparison: Full Fine-Tuning vs. Adapters

```
Full Fine-Tuning:
- Trainable params: 110,000,000 (100% of BERT-base)
- Storage per task: 110M params ≈ 440 MB (fp32)

Single Adapter (r=64):
- Trainable params: 2,400,000 (2.2% for 12 layers dual placement)
- Storage per task: 2.4M params ≈ 9.6 MB (fp32)

Efficiency Gain: 45-50x parameter reduction
```

#### Scaling Across Layers

For N-layer model with placement strategy:

```
- Dual placement (Houlsby): 2N × (2rd + r + d) parameters
- Single FFN placement (Pfeiffer): N × (2rd + r + d) parameters
- Parallel placement: N × (2rd + r + d) parameters
```

### 1.3 Typical Bottleneck Dimensions

| Dimension | Use Case | Compression Ratio (d=768) | Storage (12 layers dual) |
|-----------|----------|---------------------------|-------------------------|
| 8 | Ultra-low resource, task similar to pretraining | 96:1 | 288 KB |
| 16 | Low-data regime (<1K examples) | 48:1 | 576 KB |
| 32 | Small, related tasks | 24:1 | 1.2 MB |
| 64 | Default for most NLP tasks | 12:1 | 2.4 MB |
| 128 | Complex tasks, large datasets | 6:1 | 4.8 MB |
| 256 | Very complex tasks, approaching full FT cost | 3:1 | 9.6 MB |

**Guideline**: Start with r=64, validate, adjust based on task complexity.

### 1.4 Adapter Insertion Points in Transformers

#### Strategy 1: Dual Placement (Houlsby et al., 2019)

```
Standard Transformer Layer:
├─ Input x
├─ Multi-Head Attention
│  └─ → [Adapter 1] → residual add
├─ Layer Norm
├─ Feed-Forward Network
│  └─ → [Adapter 2] → residual add
├─ Layer Norm
└─ Output h'
```

**Advantages**:
- Maximum expressiveness (adapts both attention and FFN)
- Can modify contextual mixing and feature transformation
- Better performance on diverse tasks (0.5-1.5% improvement)

**Disadvantages**:
- 2x parameter overhead
- Slower inference (more sequential operations)
- 24 adapters for 12-layer model

#### Strategy 2: Efficient Single Placement (Pfeiffer et al., 2021)

```
Standard Transformer Layer:
├─ Input x
├─ Multi-Head Attention
├─ Layer Norm
├─ Feed-Forward Network
│  └─ → [Adapter] → residual add
├─ Layer Norm
└─ Output h'
```

**Advantages**:
- 50% parameter reduction
- FFN acts as key-value memory (better adaptation target)
- Faster inference
- Recommended default for most applications

**Disadvantages**:
- Slightly lower performance on some diverse tasks
- Less flexibility for attention-specific adaptation

#### Strategy 3: Parallel Placement

```
Standard Transformer Layer:
├─ Input x
├─ ┌─ Multi-Head Attention
├─ │  └─ output
├─ └─ [Adapter] parallel
│      └─ output
│  → combine with scaling factor s
├─ Layer Norm
├─ Feed-Forward Network
├─ Layer Norm
└─ Output h'

Computation: h' = x + s·MHA(x) + s·Adapter(x)
```

**Advantages**:
- Can be computed in parallel on hardware supporting concurrent ops
- Different gradient flow dynamics
- Potentially more stable training on some tasks

**Disadvantages**:
- Requires careful scaling factor tuning
- Less intuitive integration with standard transformer

---

## 2. Bottleneck Adapters: Advanced Details

### 2.1 Activation Function Choices

#### GELU (Gaussian Error Linear Unit)

```python
f(x) = x · Φ(x)

where Φ(x) is the cumulative distribution function of standard normal

# Approximation
f(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**Characteristics**:
- Smooth, differentiable
- Probabilistic interpretation (gate mechanism)
- Default in most transformers (BERT, GPT-2, T5)
- Slightly more expensive computationally
- Better gradient flow in deep networks

**When to use**: Default choice for most adapters

#### ReLU (Rectified Linear Unit)

```python
f(x) = max(0, x)
```

**Characteristics**:
- Computationally efficient
- Sparse activation (50% inactive in expectation)
- Sharp gradient at 0 (may cause dead units)
- Simple implementation

**When to use**: 
- Inference-latency sensitive applications
- Ultra-low-resource settings
- When training stability is good

#### Tanh (Hyperbolic Tangent)

```python
f(x) = (e^(2x) - 1) / (e^(2x) + 1)
```

**Characteristics**:
- Output bounded to [-1, 1]
- Symmetric around origin
- Vanishing gradient problem in deep networks
- Rarely used in modern transformers

**When to use**:
- Rare; generally inferior to GELU for transformers
- Legacy compatibility

### 2.2 Skip Connections and Layer Normalization

#### Residual Connection Design

The residual connection is critical for adapter effectiveness:

```
h' = h + α · Adapter(h)

where α is typically 1.0 but can be tuned
```

**Why residuals matter**:

1. **Near-Identity Initialization**:
   - Initialize adapter weights to ~zero
   - Adapter outputs near zero initially
   - Model behaves like pretrained model at start
   - Stable training from frozen state

2. **Gradient Flow**:
   - Gradients bypass adapter nonlinearity via skip
   - Information reaches all layers even with small adapter outputs
   - Alleviates vanishing gradient problem

3. **Parameter Efficiency**:
   - Adapter doesn't need to reproduce input
   - Only learns delta (change) needed for task
   - Forces selective, economical use of parameters

#### Layer Normalization Integration

```
Transformer computation:
h₁ = LayerNorm(x + MHA(x))
h₂ = LayerNorm(h₁ + FFN(h₁))

With adapters:
h₁ = LayerNorm(x + Adapter₁(MHA(x)))
h₂ = LayerNorm(h₁ + Adapter₂(FFN(h₁)))
```

**Design considerations**:

1. **Placement of Layer Norm**:
   - Pre-norm: LayerNorm before adapter
   - Post-norm: LayerNorm after adapter
   - AdapterHub default: post-norm (after residual add)

2. **Benefits of Post-Norm with Adapters**:
   - Stabilizes training without layer norm in adapter itself
   - Prevents internal covariate shift in adapter outputs
   - Cleaner gradient flow

### 2.3 Residual Design Patterns

#### Pattern 1: Simple Residual (Default)

```python
def adapter_forward(h):
    # Down projection
    down = self.down_proj(h)  # (d,) → (r,)
    
    # Activation
    activated = self.activation(down)  # (r,) → (r,)
    
    # Up projection
    up = self.up_proj(activated)  # (r,) → (d,)
    
    # Residual addition
    output = h + up
    
    return output
```

**Initialization**:
```python
# Down projection: standard initialization
nn.init.normal_(self.down_proj.weight, std=0.01)

# Up projection: initialize to near-zero (important!)
nn.init.normal_(self.up_proj.weight, std=0.001)
nn.init.constant_(self.up_proj.bias, 0)
```

#### Pattern 2: Scaled Residual

```python
def adapter_forward(h, scale=1.0):
    down = self.down_proj(h)
    activated = self.activation(down)
    up = self.up_proj(activated)
    
    # Scaled residual (can dampen adapter contribution)
    output = h + scale * up
    
    return output
```

**Use case**: Fine-control over adapter influence during training

#### Pattern 3: Gated Residual

```python
def adapter_forward(h):
    down = self.down_proj(h)
    activated = self.activation(down)
    up = self.up_proj(activated)
    
    # Learn a gate controlling adapter contribution
    gate = self.gate(h)  # scalar or (d,) depending on design
    
    output = h + gate * up
    
    return output
```

**Use case**: Adaptive blend between pretrained and task-specific behavior

---

## 3. Advanced Variants

### 3.1 MAD-X: Multi-lingual Adapters

**Paper**: Pfeiffer et al., 2020

**Key Innovation**: Decouple language-specific and task-specific adaptation through stacked adapters.

#### Architecture

```
Embedding Layer
    ↓
[Invertible Adapter] ← Language-specific transformation
    ↓
Layer 1: [Transformer] + [Language Adapter] + [Task Adapter]
    ↓
Layer 2: [Transformer] + [Language Adapter] + [Task Adapter]
    ...
```

#### Components

1. **Language Adapters**:
   - Trained on language modeling for each language
   - Learns language-specific transformations
   - Stacked first in composition

2. **Invertible Adapters**:
   - Placed at embedding layer
   - Forward pass: transform embeddings before first layer
   - Inverse pass: reverse transformation after last layer
   - Enables proper information flow and invertibility

3. **Task Adapters**:
   - Stacked after language adapters
   - Trained on downstream task for specific language

#### Composition

```
Stack(lang_adapter, task_adapter)

Inference flow:
input → embed → [Invertible: forward] → 
  Layer1: [Transformer] + [Lang] + [Task] →
  Layer2: [Transformer] + [Lang] + [Task] → 
  ... → [Invertible: inverse] → output
```

#### Zero-Shot Cross-Lingual Transfer

```python
# Trained on: English + MNLI
# Test on: German (unseen language)

# Switch language adapter, keep task adapter
config.active_adapters = Stack("de_lang", "mnli_task")

# Model transfers knowledge to German without German training!
predictions = model(german_text)
```

**Results**:
- Near full fine-tuning performance on target language
- 2-5% gap compared to language-specific training (task-dependent)
- Storage: English model + adapter for each (language, task) pair

### 3.2 IA³: Infused Adapters by Inhibiting and Amplifying

**Paper**: Liu et al., 2022 (T-Few method)

**Key Innovation**: Ultra-lightweight adapters using element-wise scaling instead of bottleneck layers.

#### Architecture

```
Standard layer computation: h = W x

IA³ modification: h = l_W ⊙ W x

where ⊙ is element-wise multiplication
and l_W is trainable rescaling vector
```

#### Components

1. **Rescaling Vectors**:
   - Trainable parameters applied element-wise
   - Much smaller than bottleneck adapters
   - Only 1-dimensional vectors or projections

2. **Insertion Points**:
   - Self-attention key (K) and value (V) matrices
   - Final feed-forward output projection
   - Not applied to query (Q) matrix

3. **Mathematical Formulation**:
   ```
   For self-attention:
   - K' = l_k ⊙ K       (rescale key projection)
   - V' = l_v ⊙ V       (rescale value projection)
   
   For FFN:
   - output' = l_ffn ⊙ FFN(x)
   ```

#### Parameter Count

For BERT-base (768-dim, 12 layers):
- IA³ parameters: 3 × 12 × 768 = 27,648 params (0.025% of model!)
- vs. Bottleneck (r=64): 2.4M params (2.2% of model)
- **90x more parameter efficient than bottleneck adapters**

#### Training Dynamics

```python
class IA3Module(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x):
        # Element-wise rescaling (multiplicative)
        return x * self.scale
```

**Key difference from bottleneck**:
- Multiplicative composition: h = l_W ⊙ W x
- vs. Additive residual: h' = h + Adapter(h)
- Composition is direct weight modification

#### Performance Characteristics

- **Few-shot learning**: Excellent (designed for 5-10K examples)
- **Full dataset performance**: Slightly below bottleneck adapters
- **Inference speed**: Near-zero overhead (just vector multiplication)
- **Memory**: Minimal (only scaling vectors)

### 3.3 Compacter: Parametric and Efficient Adapters

**Paper**: Mahabadi et al., 2021

**Key Innovation**: Replace linear projections with Parameterized Hypercomplex Multiplication (PHM) layers.

#### Architecture

```
Standard Bottleneck:
Down(d→r) → Activation → Up(r→d)
Weight matrices: W_down ∈ ℝ^(r×d), W_up ∈ ℝ^(d×r)
Parameters: 2rd + (r + d)

Compacter:
Down(PHM) → Activation → Up(PHM)
PHM constructs W from smaller factorized matrices
Parameters: proportional to √(2rd) (square root improvement!)
```

#### PHM Layer Design

```
For down-projection, instead of W_down ∈ ℝ^(r×d):

PHM constructs weight matrix from:
- Two basis matrices B₁, B₂ ∈ ℝ^(√(rd) × √(rd))
- Shared across all layers to reduce parameters
- Can be factorized further: B = L · R (low-rank)

Construction: W = B₁ ⊙ B₂  (Kronecker product)
```

#### Parameter Reduction

```
Standard bottleneck (r=64, d=768):
- Parameters: 2 × 64 × 768 = 98,304

Compacter (r=64, d=768):
- PHM matrices: 2 × √(64×768) × √(64×768) ≈ 6,144
- Factorized further: ≈ 2,000-3,000 parameters
- Reduction: 30-50x fewer parameters!
```

#### Configuration

```python
from adapters import CompacterConfig

config = CompacterConfig(
    phm_dim=4,                    # PHM inner dimension
    shared_phm_rule=True,         # Share PHM across layers
    factorized_phm_rule=True,     # Factorize PHM matrices
    learn_phm=True,               # Learn PHM or keep fixed
)

model.add_adapter("compacter", config=config)
```

**Trade-offs**:
- **Pros**: Extreme parameter efficiency, competitive performance
- **Cons**: More complex implementation, harder to tune, less interpretable

### 3.4 Prefix Adapters: Combining with Prefix Tuning

**Paper**: Li and Liang, 2021

**Key Innovation**: Prepend learnable tokens to attention mechanisms.

#### Architecture

```
Standard attention:
Query, Key, Value ∈ ℝ^(L × d)
Output = Attention(Q, K, V)

Prefix Tuning:
Modified Key/Value:
K' = [P_k | K]  (prepend prefix P_k)
V' = [P_v | V]  (prepend prefix P_v)

New sequence length: L + prefix_length
```

#### Mechanism

```python
def prefix_attention(Q, K, V, P_k, P_v):
    L = Q.shape[0]
    prefix_len = P_k.shape[0]
    
    # Concatenate prefix with keys and values
    K_extended = torch.cat([P_k, K], dim=0)  # (L+p, d)
    V_extended = torch.cat([P_v, V], dim=0)  # (L+p, d)
    
    # Standard attention computation
    scores = Q @ K_extended.T / sqrt(d)
    attn_weights = softmax(scores)
    output = attn_weights @ V_extended
    
    return output
```

#### Reparameterization

Original paper reparameterizes prefix via bottleneck MLP:

```
P_k = MLP_k(small_tensor)
P_v = MLP_v(small_tensor)

where MLP compresses→expands through bottleneck
```

**Benefit**: Reduces prefix parameters further and improves optimization

#### Configurations

```python
from adapters import PrefixTuningConfig

# Direct parameterization (flat=True)
config = PrefixTuningConfig(
    flat=True,
    prefix_length=30
)

# Reparameterized (flat=False, default)
config = PrefixTuningConfig(
    flat=False,
    prefix_length=30,
    bottleneck_dim=64  # MLP bottleneck dimension
)

model.add_adapter("prefix", config=config)
```

#### Combining Adapters + Prefix Tuning

```python
# Stack bottleneck adapter with prefix tuning
config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16)
model.add_adapter("bn_adapter", config=config)

prefix_config = PrefixTuningConfig(flat=False, prefix_length=20)
model.add_adapter("prefix", config=prefix_config)

# Activate both
model.active_adapters = Stack("bn_adapter", "prefix")
```

**Performance**:
- Prefix alone: ~equivalent to bottleneck
- Combined (bottleneck + prefix): Often better than either alone
- Complementary: Bottleneck adjusts layer outputs, prefix reshapes attention

---

## 4. Hypernetwork-Based Adapters

### 4.1 Generating Adapter Weights Dynamically

**Concept**: Instead of using fixed adapter weights, use a hypernetwork to generate them conditioned on inputs.

#### Architecture

```
Input → [Hypernetwork] → Generated Adapter Weights → Apply to Hidden State

Generative Process:
1. Extract conditioning information from input
2. Pass through small network (hypernetwork)
3. Generate task-specific adapter weights
4. Use generated weights to transform hidden states
```

#### Mathematical Formulation

```
For standard adapter:
h' = h + W_up · f(W_down · h)

For hypernetwork adapter:
W_down = HyperNet_down(c)  # Generated from conditioning vector c
W_up = HyperNet_up(c)      # Generated from conditioning vector c
h' = h + W_up · f(W_down · h)
```

### 4.2 Conditioning Mechanisms

#### 1. Task Embedding Conditioning

```python
class TaskConditionedAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, num_tasks):
        super().__init__()
        
        # Embed task IDs
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)
        
        # Hypernetwork: task embedding → adapter weights
        self.hyper_down = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, bottleneck_dim * hidden_dim)
        )
        
        self.hyper_up = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim * bottleneck_dim)
        )
    
    def forward(self, h, task_id):
        # Generate task-specific weights
        task_emb = self.task_embedding(task_id)
        
        w_down_flat = self.hyper_down(task_emb)
        w_down = w_down_flat.reshape(self.bottleneck_dim, self.hidden_dim)
        
        w_up_flat = self.hyper_up(task_emb)
        w_up = w_up_flat.reshape(self.hidden_dim, self.bottleneck_dim)
        
        # Apply generated adapter
        down = F.linear(h, w_down)
        activated = F.relu(down)
        up = F.linear(activated, w_up)
        
        return h + up
```

#### 2. Input-Dependent Conditioning

```python
class InputConditionedAdapter(nn.Module):
    def forward(self, h):
        # Extract global information from input
        global_context = h.mean(dim=0)  # Average pooling
        
        # Generate task-specific weights from context
        w_down = self.hyper_down(global_context)
        w_up = self.hyper_up(global_context)
        
        # Apply dynamic adapter
        down = F.linear(h, w_down)
        activated = F.relu(down)
        up = F.linear(activated, w_up)
        
        return h + up
```

#### 3. Mixture of Adapters

```python
class MixtureOfAdapters(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, num_experts=4):
        super().__init__()
        
        # Multiple expert adapters
        self.experts = nn.ModuleList([
            BottleneckAdapter(hidden_dim, bottleneck_dim)
            for _ in range(num_experts)
        ])
        
        # Gating network: select expert weights
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, h):
        # Compute expert outputs
        expert_outputs = [expert(h) for expert in self.experts]
        
        # Compute gating weights
        gate_weights = self.gate(h.mean(dim=0))  # Shape: (num_experts,)
        
        # Mixture: weighted sum of expert outputs
        output = sum(w * o for w, o in zip(gate_weights, expert_outputs))
        
        return output
```

### 4.3 Task-Specific Weight Generation

#### Multi-Task Learning with Hypernetworks

```python
class MultiTaskHyperAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim):
        super().__init__()
        
        # Shared hypernetwork
        self.hyper_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
    
    def register_task(self, task_name):
        """Register new task with its own adapter generation head"""
        self.task_heads[task_name] = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.bottleneck_dim * self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * self.bottleneck_dim)
        )
    
    def forward(self, h, task_name):
        # Shared processing
        shared = self.hyper_base(h.mean(dim=0))
        
        # Task-specific generation
        if task_name not in self.task_heads:
            self.register_task(task_name)
        
        task_weights = self.task_heads[task_name](shared)
        w_down = task_weights[:self.bottleneck_dim * self.hidden_dim]
        w_up = task_weights[self.bottleneck_dim * self.hidden_dim:]
        
        # Apply generated adapter
        down = F.linear(h, w_down.reshape(self.bottleneck_dim, self.hidden_dim))
        activated = F.relu(down)
        up = F.linear(activated, w_up.reshape(self.hidden_dim, self.bottleneck_dim))
        
        return h + up
```

---

## 5. Composition Methods

### 5.1 Stacking Multiple Adapters

**Use Case**: Combine language-specific and task-specific knowledge (MAD-X pattern).

```python
from adapters import Stack

# Add adapters
model.add_adapter("en_lang", config=lang_config)
model.add_adapter("sentiment_task", config=task_config)

# Stack: input → lang_adapter → task_adapter → output
model.active_adapters = Stack("en_lang", "sentiment_task")

# Forward pass flows through both sequentially
output = model(input_ids=ids)
```

**Mathematical Composition**:
```
h₁ = h₀ + Adapter_lang(h₀)
h₂ = h₁ + Adapter_task(h₁)
    = h₀ + Adapter_lang(h₀) + Adapter_task(h₀ + Adapter_lang(h₀))
```

**Performance**:
- Combines knowledge from both adapters
- Each adapter sees output of previous one
- Useful for hierarchical adaptation

### 5.2 Parallel Composition with Gating

**Use Case**: Use multiple adapters in parallel and combine outputs intelligently.

```python
from adapters import Parallel

# Add multiple task adapters
model.add_adapter("sentiment", config=config1)
model.add_adapter("topic", config=config2)
model.add_adapter("ner", config=config3)

# Parallel: input → [all adapters] → combine outputs
model.active_adapters = Parallel("sentiment", "topic", "ner")

# Multiple prediction heads
sentiment_output, topic_output, ner_output = model(input_ids=ids)
```

**Gating Mechanism**:

```python
class ParallelAdapterGating(nn.Module):
    def __init__(self, hidden_dim, num_adapters):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_adapters)
    
    def forward(self, adapter_outputs):
        # adapter_outputs: list of tensors
        # Compute gating weights
        combined = torch.stack(adapter_outputs)  # (num_adapters, B, L, d)
        mean_output = combined.mean(dim=0)      # (B, L, d)
        
        weights = self.gate(mean_output).softmax(dim=-1)  # (B, L, num_adapters)
        
        # Weighted combination
        output = torch.einsum('bld,bla->bld', weights, combined)
        
        return output
```

### 5.3 Sequential Routing

**Use Case**: Route inputs to different adapters based on content/task signals.

```python
class SequentialRouter(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Router predicts which adapter to use
        self.router = nn.Linear(hidden_dim, num_tasks)
    
    def forward(self, h, adapters_dict):
        # Predict adapter index
        logits = self.router(h.mean(dim=0))
        adapter_idx = logits.argmax(dim=-1)
        
        # Route to selected adapter
        adapter_name = self.idx_to_name[adapter_idx]
        adapter = adapters_dict[adapter_name]
        
        return adapter(h)
```

**Example: Task-Aware Routing**

```python
# Train router to predict task from input
sentiment_adapter = BnConfig(...)
topic_adapter = BnConfig(...)

# Router learns to discriminate tasks
router = TaskRouter(hidden_dim=768, num_tasks=2)

# Inference
for batch in dataloader:
    hidden = model.encoder(batch['input_ids'])
    task_logits = router(hidden)
    task_id = task_logits.argmax()
    
    if task_id == 0:
        model.active_adapters = "sentiment"
    else:
        model.active_adapters = "topic"
    
    output = model(hidden)
```

### 5.4 Merging and Distillation

#### Merging Adapters

**Concept**: Combine multiple trained adapters into one.

```python
from adapters import merge_adapters

# Train adapters separately
model.add_adapter("adapter1", config=config)
model.add_adapter("adapter2", config=config)

# Train both on different tasks...

# Merge into single adapter
merged_adapter = merge_adapters(
    ["adapter1", "adapter2"],
    weights=[0.6, 0.4]  # Weighted combination
)

model.add_adapter("merged", config=merged_adapter)
```

**Mathematical Merging**:

```
For bottleneck adapters:
W_down_merged = w₁·W_down_1 + w₂·W_down_2
W_up_merged = w₁·W_up_1 + w₂·W_up_2

Merged behavior: h' = h + W_up_merged · f(W_down_merged · h)
```

#### Distillation: Compress Many Adapters

```python
class AdapterDistillation:
    def __init__(self, teacher_adapters, student_config):
        self.teachers = teacher_adapters
        self.student_config = student_config
    
    def train_student(self, train_data, num_epochs=10):
        student = create_adapter(self.student_config)
        optimizer = AdamW(student.parameters(), lr=1e-4)
        
        for epoch in range(num_epochs):
            for batch in train_data:
                # Teacher ensemble
                teacher_outputs = [
                    teacher(batch['hidden_states']) 
                    for teacher in self.teachers
                ]
                teacher_output = torch.stack(teacher_outputs).mean(dim=0)
                
                # Student output
                student_output = student(batch['hidden_states'])
                
                # Distillation loss
                loss = F.mse_loss(student_output, teacher_output.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return student
```

---

## 6. Mathematical Analysis

### 6.1 Parameter Efficiency Formulas

#### Reduction Factor

```
Reduction % = (1 - Params_adapter / Params_full) × 100%

For BERT-base:
- Full FT: 110M params
- Single adapter (r=64, 12 layers, dual): 2.4M params
- Reduction: (1 - 2.4M/110M) × 100% = 97.8% ✓
```

#### Speedup Analysis

```
Training FLOPs:

Full FT per layer:
- Attention: 2Ld² forward + backward
- FFN: 4Ld² forward + backward
- Total: ~6Ld²

Adapter FT per layer:
- Frozen sublayers: unchanged
- Adapter: 2×(2Lrd) forward + backward = 4Lrd
- Total overhead: 4Lrd

Reduction factor: 4Lrd / (6Ld²) = 2r/(3d)

For r=64, d=768: 2×64/(3×768) ≈ 5.6% overhead ✓
```

### 6.2 Capacity-Complexity Trade-Off

#### Information Bottleneck Theory

```
I(X;Z) - β·I(Z;Y) minimized

where:
X: input hidden state (768-dim)
Z: bottleneck state (r-dim)
Y: task label
I(·,·): mutual information
β: Lagrange multiplier

Higher β → more compression → better generalization, worse task fit
Lower β → less compression → task fit, potential overfitting
```

#### Effective Capacity

```
Effective parameters ≈ min(2rd, d·log(d/r))

Intuition: bottleneck can represent at most r independent features,
but actual capacity depends on nonlinearity and training dynamics.

For r << d:
- Capacity grows linearly with r (2rd ≈ Capacity)
- Each dimension of bottleneck can contribute meaningfully

For r → d:
- Capacity approaches full dimension representation
- Diminishing returns set in
```

### 6.3 Convergence Analysis with Bottlenecks

#### Training Dynamics

```
Gradient flow in early training:

dL/dW_up ∝ residual gradient + activation gradient
dL/dW_down ∝ residual gradient + upstream gradient × output weights

Residual enables direct gradient flow:
- Gradients don't only flow through adapter nonlinearity
- "Highway" of gradient flow through skip connection
- Stabilizes deep networks

Convergence rate:
- With residual: O(1/√T) to ε-optimal (standard SGD rate)
- Without residual: May be significantly slower
```

#### Bottleneck Effect on Convergence

```
Narrower bottleneck (smaller r):
- Fewer parameters → faster per-step updates
- But: Limited model capacity → may need more steps
- Trade-off: Often faster wall-clock time with r=32-64

Wider bottleneck (larger r):
- More parameters → slower per-step updates
- More expressiveness → may need fewer steps
- Trade-off: Better for complex tasks
```

### 6.4 Expressiveness Bounds

#### Universal Approximation

```
Theorem: Adapter with sufficient bottleneck dimension can approximate
any smooth transformation of the hidden state.

Proof sketch:
1. Dense bottleneck layer (r >> log(d)) can approximate any
   continuous function of input in L^2 sense
2. Up-projection then expands back to dimension d
3. Residual + activation composition gives expressiveness

Practical implication:
- No fundamental limitation on task complexity for large r
- But regularization via small r prevents overfitting
```

#### Approximation Error

```
For task requiring transformation f: R^d → R^d

Adapter approximation: h' = h + W_up · σ(W_down · h)

Approximation error ≤ O(1/r + λ||W||²)

where λ is regularization strength

Implications:
- Error decreases with r (capacity)
- But increases with weight magnitude (regularization needed)
- Optimal r balances both terms
```

---

## 7. Implementation Guide

### 7.1 Using AdapterHub Library

#### Installation

```bash
pip install adapters
# Compatible with: BERT, RoBERTa, ALBERT, DistilBERT, DeBERTa, T5, LLaMA, Mistral, etc.
```

#### Quick Start: Adding Bottleneck Adapters

```python
from adapters import AutoAdapterModel, BnConfig

# Load model with adapter support
model = AutoAdapterModel.from_pretrained("bert-base-uncased")

# Create bottleneck adapter config
config = BnConfig(
    mh_adapter=True,           # After multi-head attention
    output_adapter=True,       # After FFN
    reduction_factor=16,       # r = d/16 (for d=768: r=48)
    non_linearity="gelu",      # Activation function
)

# Add adapter to model
model.add_adapter("sentiment_adapter", config=config)

# Set as active
model.active_adapters = "sentiment_adapter"

# Now ready to train!
# Frozen params: 110M, Trainable params: ~4.8M (4.4%)
```

#### Predefined Configurations

```python
from adapters import (
    BnConfig,
    DoubleSeqBnConfig,  # Houlsby et al. (dual placement)
    SeqBnConfig,        # Pfeiffer et al. (FFN-only)
    ParBnConfig,        # Parallel placement
    PrefixTuningConfig,
    CompacterConfig,
    LoRAConfig,
    IA3Config,
)

# Houlsby: dual adapters (default)
houlsby_config = DoubleSeqBnConfig(reduction_factor=16)

# Pfeiffer: efficient single adapter
pfeiffer_config = SeqBnConfig(reduction_factor=16)

# Parallel adapters
parallel_config = ParBnConfig(reduction_factor=16)

# Prefix tuning
prefix_config = PrefixTuningConfig(
    flat=False,        # Reparameterized via bottleneck
    prefix_length=30,
)

# Compacter
compacter_config = CompacterConfig()

model.add_adapter("config_choice", config=compacter_config)
```

### 7.2 Custom Adapter Implementation

#### Minimal Bottleneck Adapter

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckAdapter(nn.Module):
    """Minimal bottleneck adapter implementation"""
    
    def __init__(
        self,
        hidden_dim,
        bottleneck_dim,
        activation="gelu",
        init_scale=0.001,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Down-projection
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Up-projection
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        
        # Initialize weights
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        
        # Important: small initialization for up-proj for near-identity
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, hidden_states):
        """Apply bottleneck adapter with residual connection"""
        
        # Down-projection
        down = self.down_proj(hidden_states)
        
        # Activation
        activated = self.activation(down)
        
        # Up-projection
        up = self.up_proj(activated)
        
        # Residual connection
        output = hidden_states + up
        
        return output


# Usage
adapter = BottleneckAdapter(
    hidden_dim=768,
    bottleneck_dim=64,
    activation="gelu"
)

hidden = torch.randn(4, 128, 768)  # (batch, seq_len, hidden_dim)
output = adapter(hidden)
assert output.shape == hidden.shape
```

#### Inserting Adapters into Transformer

```python
class TransformerLayerWithAdapter(nn.Module):
    """Standard transformer layer with bottleneck adapter"""
    
    def __init__(self, hidden_dim, num_heads, ffn_dim, adapter_dim):
        super().__init__()
        
        # Standard components
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Adapters
        self.adapter1 = BottleneckAdapter(hidden_dim, adapter_dim)
        self.adapter2 = BottleneckAdapter(hidden_dim, adapter_dim)
    
    def forward(self, hidden_states, attention_mask=None):
        # Multi-head attention + adapter (Houlsby placement)
        attn_out = self.attention(hidden_states, attention_mask)
        attn_out = self.adapter1(attn_out)  # ← Adapter 1
        hidden_states = self.ln1(hidden_states + attn_out)
        
        # Feed-forward + adapter
        ffn_out = self.ffn(hidden_states)
        ffn_out = self.adapter2(ffn_out)    # ← Adapter 2
        output = self.ln2(hidden_states + ffn_out)
        
        return output


# Efficient single-adapter version
class TransformerLayerEfficient(nn.Module):
    """Pfeiffer et al. (2021) - FFN-only adapter"""
    
    def forward(self, hidden_states, attention_mask=None):
        # Multi-head attention (no adapter)
        attn_out = self.attention(hidden_states, attention_mask)
        hidden_states = self.ln1(hidden_states + attn_out)
        
        # Feed-forward + single adapter
        ffn_out = self.ffn(hidden_states)
        ffn_out = self.adapter(ffn_out)     # ← Single adapter
        output = self.ln2(hidden_states + ffn_out)
        
        return output
```

### 7.3 Integration with Transformers

#### Using Hugging Face Integration

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from adapters import AutoAdapterModel
import torch.optim as optim

# Load model with adapter support
model = AutoAdapterModel.from_pretrained(
    "bert-base-uncased",
    num_labels=2  # Binary classification
)

# Add task adapter
from adapters import BnConfig
config = BnConfig(reduction_factor=16)
model.add_adapter("sentiment", config=config)
model.active_adapters = "sentiment"

# Add prediction head
model.add_classification_head("sentiment", num_labels=2)

# Prepare data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
texts = ["Great movie!", "Terrible film."]
tokens = tokenizer(texts, padding=True, return_tensors="pt")

# Training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

labels = torch.tensor([1, 0])
outputs = model(**tokens)
loss = criterion(outputs.logits, labels)

loss.backward()
optimizer.step()

print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### 7.4 Distributed Training with Adapters

#### Data Parallel

```python
import torch.nn as nn
import torch.distributed as dist

# Wrap model for data parallel
model = AutoAdapterModel.from_pretrained("bert-base-uncased")
model.add_adapter("task", config=BnConfig(reduction_factor=16))

# Data parallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Standard distributed training
dist.init_process_group("nccl")
model = nn.parallel.DistributedDataParallel(model)

# Training proceeds normally
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, batch['labels'])
    loss.backward()
    optimizer.step()
```

#### Distributed Adapter Training

```python
# Multi-GPU with gradient accumulation
from torch.nn.parallel import DistributedDataParallel as DDP

model = AutoAdapterModel.from_pretrained("bert-base-uncased")
model.add_adapter("task", config=BnConfig(reduction_factor=16))

# Move to GPU
device = torch.device(f"cuda:{local_rank}")
model = model.to(device)

# DDP wrapper
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# Training with adapter-specific optimization
adapter_params = []
for name, param in model.named_parameters():
    if 'adapter' in name:
        adapter_params.append(param)

optimizer = optim.Adam(adapter_params, lr=1e-4)

# Distributed training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'])
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
```

---

## 8. Code Examples

### 8.1 Building Bottleneck Adapter from Scratch

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# Define adapter
class TaskAdapter(nn.Module):
    def __init__(self, hidden_dim, reduction_factor=16):
        super().__init__()
        bottleneck_dim = hidden_dim // reduction_factor
        
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        
        # Near-identity initialization
        nn.init.normal_(self.up.weight, std=0.001)
    
    def forward(self, h):
        return h + self.up(self.activation(self.down(h)))


# Integrate into model
class AdaptedBertLayer(nn.Module):
    def __init__(self, bert_layer, hidden_dim=768):
        super().__init__()
        self.bert_layer = bert_layer
        self.adapter = TaskAdapter(hidden_dim)
    
    def forward(self, hidden_states, attention_mask=None):
        # Original bert layer output
        layer_output = self.bert_layer(hidden_states, attention_mask)
        
        # Apply adapter
        adapted_output = self.adapter(layer_output[0])
        
        return (adapted_output,) + layer_output[1:]


# Training
def train_with_adapter(model, train_loader, num_epochs=3):
    # Freeze all parameters except adapter
    for param in model.parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True
    
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
```

### 8.2 Using AdapterHub for Quick Setup

```python
from adapters import AutoAdapterModel, BnConfig, SeqBnConfig, BnInvConfig
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load pretrained model
model = AutoAdapterModel.from_pretrained("bert-base-uncased")

# Add language adapter (MAD-X style)
lang_config = BnInvConfig()  # Invertible for language
model.add_adapter("en_lang", config=lang_config)

# Add task adapter
task_config = SeqBnConfig(reduction_factor=16)  # FFN-only
model.add_adapter("sentiment", config=task_config)

# Stack language + task
model.active_adapters = f"en_lang,sentiment"

# Add prediction head
model.add_classification_head("sentiment", num_labels=2)

# Prepare dataset
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length')

dataset = dataset.map(preprocess, batched=True)

# Train with Trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
)

trainer.train()
```

### 8.3 Multi-Task Adapter Composition

```python
from adapters import Stack, Parallel
import adapters.composition as ac

# Add multiple task adapters
model.add_adapter("sentiment", config=SeqBnConfig(reduction_factor=16))
model.add_adapter("topic", config=SeqBnConfig(reduction_factor=16))
model.add_adapter("ner", config=SeqBnConfig(reduction_factor=16))

# Stack composition (language + sentiment task)
model.active_adapters = Stack("en_lang", "sentiment")
sentiment_output = model(input_ids)

# Switch to different task
model.active_adapters = Stack("en_lang", "topic")
topic_output = model(input_ids)

# Parallel composition (multi-task output)
model.add_classification_head("sentiment", num_labels=2)
model.add_classification_head("topic", num_labels=10)

model.active_adapters = Parallel("sentiment", "topic")
sentiment_logits, topic_logits = model(input_ids)

# Adapter fusion (non-destructive combination)
model.add_adapter_fusion(["sentiment", "topic"])
model.active_adapters = ac.Fuse("sentiment", "topic")

# Train fusion layer to optimally combine both adapters
# ...
fused_output = model(input_ids)
```

### 8.4 Adapter Merging

```python
from adapters import AdapterSetup, merge_adapters

# Train two adapters separately
model.add_adapter("en_sentiment", config=config)
model.add_adapter("de_sentiment", config=config)

# After training both...

# Merge with weights
merged = merge_adapters(
    model,
    adapters=["en_sentiment", "de_sentiment"],
    weights=[0.6, 0.4],
    output_adapter_name="merged_sentiment"
)

# Use merged adapter
model.active_adapters = "merged_sentiment"
output = model(input_ids)

# Or merge by averaging parameters
merged_params = {
    'down_proj.weight': (
        model.adapters['en_sentiment']['down_proj.weight'] +
        model.adapters['de_sentiment']['down_proj.weight']
    ) / 2,
    # ... other parameters
}
```

### 8.5 Inference Optimization

```python
import onnx
import onnxruntime as ort

# Method 1: Merge adapters for inference
model.add_adapter("task", config=config)
# ... training ...

# Merge adapter weights into base model
model.merge_adapter("task")

# Now forward pass has no adapter overhead!
with torch.no_grad():
    output = model(input_ids)

# Export to ONNX for even faster inference
torch.onnx.export(
    model,
    (input_ids,),
    "model.onnx",
    input_names=['input_ids'],
    output_names=['output'],
    opset_version=12
)

# ONNX inference
session = ort.InferenceSession("model.onnx")
output = session.run(None, {"input_ids": input_ids.numpy()})

# Method 2: Use LoRA or IA³ for faster inference
from adapters import LoRAConfig, IA3Config

# LoRA can be merged
lora_config = LoRAConfig(r=8)
model.add_adapter("task", config=lora_config)
model.merge_adapter("task")  # Zero-overhead inference!

# Method 3: Load only active adapter weights
model.load_adapter("task")
model.active_adapters = "task"

# Batch inference with different adapters
for adapter_name in ["adapter1", "adapter2", "adapter3"]:
    model.active_adapters = adapter_name
    output = model(batch)
    process_output(output)
```

---

## 9. Empirical Analysis

### 9.1 Parameter Count Reduction

**Bottleneck Adapters (BERT-base, 12 layers)**

| Placement | r=16 | r=32 | r=64 | r=128 | r=256 |
|-----------|------|------|------|-------|-------|
| Dual | 0.3% | 1.1% | 2.2% | 4.4% | 8.6% |
| Single (FFN) | 0.15% | 0.55% | 1.1% | 2.2% | 4.3% |
| Parallel | 0.15% | 0.55% | 1.1% | 2.2% | 4.3% |

**Comparison Across Methods**

| Method | BERT-base Params | % of Model | Trainable |
|--------|------------------|-----------|-----------|
| Full FT | 110M | 100% | Yes |
| Bottleneck (r=64, dual) | 108.4M + 2.4M | 100.2% | 2.2% |
| LoRA (r=8) | 110M + 0.3M | 100.3% | 0.3% |
| Prefix (len=30) | 110M + 0.2M | 100.2% | 0.2% |
| IA³ | 110M + 0.027M | 100.02% | 0.025% |
| Compacter | 110M + 0.04M | 100.04% | 0.035% |

### 9.2 Memory Efficiency vs LoRA

```
Training Memory (per GPU, batch_size=8, seq_len=128):

Full Fine-Tuning:
- Model: 440 MB
- Optimizer states (Adam): 880 MB (2x model for momentum + variance)
- Gradients: 440 MB
- Activations: ~500 MB
- Total: ~2.26 GB

Bottleneck Adapter (r=64, dual):
- Model: 440 MB (frozen, minimal gradient)
- Optimizer states: 20 MB (only adapter params)
- Gradients: 20 MB
- Activations: ~500 MB (must keep full)
- Total: ~980 MB (57% of full FT)

LoRA (r=8):
- Model: 440 MB (frozen)
- Optimizer states: 3 MB
- Gradients: 3 MB
- Activations: ~500 MB
- Total: ~946 MB (55% of full FT)

Note: Adapter and LoRA similar memory, bottleneck still better than full FT
```

### 9.3 Training Speed Comparisons

**GLUE Benchmark (BERT-base, 8 GPUs)**

| Method | SST-2 | MNLI | QQP | QNLI | Time/Epoch |
|--------|-------|------|-----|------|-----------|
| Full FT | 94.5 | 86.3 | 91.2 | 92.8 | 180s |
| Adapter (r=64) | 94.2 | 86.1 | 91.0 | 92.6 | 95s |
| LoRA (r=8) | 94.3 | 86.2 | 91.1 | 92.7 | 50s |
| Prefix | 93.8 | 85.9 | 90.8 | 92.4 | 55s |

**Inference Speed (tokens/sec)**

```
BERT-base, batch_size=32, seq_len=128:

Full FT (baseline): 1000 tokens/sec

Merged Adapter: 998 tokens/sec (-0.2%)
- Near-zero overhead

Active Adapter: 890 tokens/sec (-11%)
- Additional forward pass through adapter

Merged LoRA: 1001 tokens/sec (-0.1%)
- Can be fused into weights

Active LoRA: 850 tokens/sec (-15%)
- Slightly more overhead than bottleneck
```

### 9.4 Accuracy Preservation

**GLUE Benchmark Comparison**

```
Metric: Percentage point gap from full fine-tuning

                Full FT   Adapter(r=64)   LoRA(r=8)   Prefix
SST-2           95.0      -0.3            -0.4        -0.8
MNLI            87.0      -0.5            -0.2        -0.9
MRPC            89.5      -0.8            -0.3        -1.2
CoLA            63.0      -1.2            -0.5        -2.1
Average Gap:              -0.7            -0.35       -1.25

Conclusion: Adapters (r=64) match full FT within 0.7%, 
            LoRA better, but 100x larger reduction factors
```

### 9.5 Generalization to New Tasks

**Transfer Learning: Train on Task A, Test on Task B**

```
BERT + Adapter Generalization:

Seen Domain (MNLI): 86.1% acc
Unseen Domain (Unseen class of MNLI): 81.5% acc
Generalization Gap: 4.6%

Comparison:
Full FT Gap: 5.2%
Adapter Gap: 4.6% (better generalization!)
LoRA Gap: 4.9%

Interpretation:
- Smaller bottleneck acts as regularizer
- Prevents overfitting to seen domain
- Better zero-shot transfer
```

---

## 10. Comparison with LoRA

### 10.1 Architectural Differences

| Aspect | Bottleneck Adapter | LoRA |
|--------|------------------|------|
| **Insertion Point** | After sublayer output | Within weight matrices |
| **Composition** | Additive residual: h' = h + Adapter(h) | Additive weight: W = W₀ + BA |
| **Parameterization** | 2 linear layers + activation | 2 low-rank matrices (no activation) |
| **Bottleneck** | Explicit (r dimension) | Implicit (rank r) |
| **Nonlinearity** | Yes (GELU/ReLU) | No (applied after weight) |

### 10.2 When to Use Each Method

#### Use Bottleneck Adapters When:

1. **Multi-task serving**: Multiple adapters in one model
   - Storage: 45+ adapters fit in model size
   - Clean separation of task-specific logic

2. **Modular composition**: Stack adapters (MAD-X pattern)
   - Language + task hierarchy
   - Invertible adapters for cross-lingual

3. **Controlled inference overhead**: Can merge for zero-cost
   - Merge learned weights into base model
   - Original performance without adapter

4. **Theoretical interpretability**: Clear delta from frozen base
   - Adapter weights = task-specific correction
   - Easier to analyze what changes

#### Use LoRA When:

1. **Parameter efficiency is critical**: 
   - 10-50x smaller than bottleneck
   - For r=8: only 0.3% of BERT-base

2. **Training speed matters**:
   - Slightly faster convergence
   - Less memory overhead

3. **Simple weight modification**:
   - Cleaner conceptually (low-rank update to weights)
   - No activation function complexities

4. **Merging for inference**:
   - Merge BA matrices into weights
   - Zero inference overhead

### 10.3 Combination Approaches

#### Stacking Bottleneck + LoRA

```python
# Combine both for maximum flexibility
model.add_adapter("bottleneck", config=SeqBnConfig(reduction_factor=16))
model.add_adapter("lora", config=LoRAConfig(r=8))

# Bottleneck for major adaptation, LoRA for fine-grained adjustment
model.active_adapters = Stack("bottleneck", "lora")

# Training flow:
# input → bottleneck adapter → LoRA adjustment → output
```

#### Prefix Tuning + Bottleneck

```python
# Prefix for attention, bottleneck for FFN
model.add_adapter("prefix", config=PrefixTuningConfig(prefix_length=30))
model.add_adapter("bottleneck", config=SeqBnConfig(reduction_factor=16))

model.active_adapters = Stack("prefix", "bottleneck")

# Complementary: Prefix shapes attention context, bottleneck adjusts features
```

### 10.4 Trade-Off Analysis

**Table: Parameter/Performance Trade-Off**

| Method | Params | GLUE Avg | Latency | Rank |
|--------|--------|----------|---------|------|
| Full FT | 110M | 86.2 | 100% | - |
| LoRA (r=16) | 0.7M | 86.0 | 98% | 1 (best efficiency) |
| LoRA (r=8) | 0.3M | 85.8 | 99% | 2 (ultra-efficient) |
| Adapter (r=64) | 2.4M | 85.9 | 89% | 3 (good compromise) |
| Prefix (len=30) | 0.2M | 85.4 | 97% | 4 (fastest) |
| Bottleneck (r=256) | 9.4M | 86.1 | 80% | 5 (best accuracy) |

**Decision Heuristic**:

```python
def choose_method(constraints):
    if constraints['num_tasks'] > 10:
        return "BottleneckAdapter"  # Multi-task friendly
    elif constraints['inference_latency_critical']:
        return "LoRA"  # Fast inference
    elif constraints['parameter_budget'] < 0.5:
        return "IA3"  # Ultra-efficient
    elif constraints['model_size'] < 1B:
        return "LoRA"  # Scales well with smaller models
    else:
        return "Adapter"  # Good general-purpose choice
```

---

## 11. Advanced Topics

### 11.1 Adapter Pruning

Remove unimportant connections in adapters post-training:

```python
def prune_adapter(adapter, sparsity=0.5):
    """Prune adapter weights to sparsity level"""
    
    for name, param in adapter.named_parameters():
        if len(param.shape) == 2:  # Weight matrices
            # Get magnitude of weights
            magnitude = torch.abs(param.data)
            
            # Find threshold for sparsity
            threshold = torch.quantile(magnitude.flatten(), sparsity)
            
            # Apply mask
            mask = magnitude > threshold
            param.data = param.data * mask.float()
```

### 11.2 Adapter Distillation

Compress ensemble of adapters:

```python
def distill_adapters(teacher_adapters, student_adapter, train_data):
    """Distill multiple adapters into single student adapter"""
    
    optimizer = AdamW(student_adapter.parameters(), lr=1e-4)
    
    for batch in train_data:
        # Get teacher ensemble outputs
        teacher_outputs = []
        for teacher in teacher_adapters:
            out = teacher(batch['hidden_states'])
            teacher_outputs.append(out)
        
        teacher_ensemble = torch.stack(teacher_outputs).mean(dim=0)
        
        # Student output
        student_out = student_adapter(batch['hidden_states'])
        
        # Distillation loss
        loss = F.mse_loss(student_out, teacher_ensemble.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return student_adapter
```

### 11.3 Task-aware Adapter Routing

Dynamic selection of adapters based on input:

```python
class DynamicAdapterRouter(nn.Module):
    def __init__(self, hidden_dim, num_tasks):
        super().__init__()
        
        # Router predicts which adapter to activate
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tasks),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, hidden_states, adapters_dict):
        # Global representation
        global_repr = hidden_states.mean(dim=0)
        
        # Predict adapter probabilities
        adapter_logits = self.router(global_repr)
        selected_adapter = torch.argmax(adapter_logits)
        
        # Route through selected adapter
        adapter_name = list(adapters_dict.keys())[selected_adapter]
        adapter = adapters_dict[adapter_name]
        
        return adapter(hidden_states)
```

---

## 11. References

### Foundation Papers

1. **Bottleneck Adapters** (Houlsby et al., 2019)
   - "Parameter-Efficient Transfer Learning for NLP"
   - Introduced bottleneck adapter architecture

2. **Efficient Placement** (Pfeiffer et al., 2020)
   - "AdapterHub: A Framework for Adapting Transformers"
   - Showed FFN-only adapters are efficient

3. **MAD-X** (Pfeiffer et al., 2020)
   - "MAD-X: An Adapter-based Framework for Multi-task Cross-lingual Transfer"
   - Stacked language + task adapters

4. **AdapterFusion** (Pfeiffer et al., 2021)
   - "AdapterFusion: Non-Destructive Task Composition for Transfer Learning"
   - Non-destructive composition of adapters

5. **IA³** (Liu et al., 2022)
   - "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"
   - Ultra-lightweight rescaling adapters

6. **Compacter** (Mahabadi et al., 2021)
   - "COMPACTER: Efficient Low-Rank Hypercomplex Adapter Layers"
   - Parametric efficient adapters with PHM

7. **LoRA** (Hu et al., 2021)
   - "LoRA: Low-Rank Adaptation of Large Language Models"
   - Compare with bottleneck approaches

### Implementation Resources

- **AdapterHub**: https://adapterhub.ml
  - Unified library for adapters
  - Pre-trained adapter collection
  - Composition examples

- **PEFT Library**: https://github.com/huggingface/peft
  - Hugging Face parameter-efficient fine-tuning
  - Support for multiple PEFT methods
  - Integration with transformers

- **Adapter Transformers**: https://github.com/adapter-hub/adapter-transformers
  - Original adapter implementation
  - Detailed documentation

### Related Papers

- "Adapters Strike Back" (Steitz & Roth, 2024)
  - Modern adapter variants and improvements
  
- "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., 2024)
  - Combining decomposition with adapters

- "VeRA: Vector-based Random Matrix Adaptation" (Kopiczko et al., 2024)
  - Efficient variants with shared matrices

### Benchmark Datasets

- GLUE (General Language Understanding Evaluation)
- SuperGLUE (challenging tasks)
- MTOP (multilingual)
- mGLUE (multilingual comprehensive)

---

## Quick Reference: Common Configurations

```python
# Quick setup for common scenarios

# 1. Single task, small data
config = BnConfig(reduction_factor=32)  # r = 24

# 2. Multiple tasks in single model
config = SeqBnConfig(reduction_factor=16)  # r = 48, FFN-only

# 3. Cross-lingual transfer
lang_config = SeqBnInvConfig()  # With invertible
task_config = SeqBnConfig(reduction_factor=16)

# 4. Maximum efficiency
config = IA3Config()  # 0.025% params

# 5. Balanced accuracy/efficiency
config = LoRAConfig(r=8, alpha=16)  # 0.3% params

# 6. Composition
model.active_adapters = Stack("lang", "task")
model.active_adapters = Parallel("task1", "task2")
model.active_adapters = ac.Fuse("task1", "task2")
```

This comprehensive skill document covers the full spectrum of adapter-based fine-tuning, from fundamentals to advanced applications.
