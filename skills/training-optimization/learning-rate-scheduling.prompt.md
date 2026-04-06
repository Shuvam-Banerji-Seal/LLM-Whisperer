# Learning Rate Scheduling for LLM Training

## Table of Contents
1. [Introduction](#introduction)
2. [Why Learning Rate Scheduling Matters](#why-learning-rate-scheduling-matters)
3. [Schedule Types with Mathematical Formulations](#schedule-types)
4. [Warmup Strategies](#warmup-strategies)
5. [Advanced Methods](#advanced-methods)
6. [Implementation Guide](#implementation-guide)
7. [Empirical Analysis](#empirical-analysis)
8. [Best Practices](#best-practices)
9. [References](#references)

---

## Introduction

Learning rate scheduling is one of the most critical yet under-appreciated aspects of training large language models. The learning rate fundamentally determines how much a model's weights change in response to the computed gradients during training. Unlike training from scratch where exploration is valued, fine-tuning and continued pre-training require careful calibration to balance task adaptation with preservation of pre-trained knowledge.

### The Core Challenge

Training large models involves navigating a complex optimization landscape:
- **Early training**: Requires higher learning rates to escape poor local minima and make substantial progress
- **Mid training**: Needs moderate rates to explore the loss landscape effectively
- **Late training**: Requires lower rates for fine convergence to a good minimum

A fixed learning rate throughout training cannot satisfy these competing needs. Too high and the model overshoots optimal solutions; too low and it converges slowly or gets stuck in suboptimal regions.

### Historical Context

- **SGDR (2016)**: Introduced warm restarts and cosine annealing schedules
- **BERT (2018)**: Popularized linear warmup + linear decay for pre-training
- **Transformer Paper (2017)**: Introduced inverse square-root schedules
- **ULMFiT (2018)**: Pioneered discriminative learning rates via layer-wise decay
- **Modern Practice (2024-2026)**: Layer-wise learning rate decay + cosine annealing with minimum learning rate has become the gold standard for LLM training

---

## Why Learning Rate Scheduling Matters

### Convergence Behavior

Well-designed schedules dramatically impact:
1. **Training stability**: Prevents divergence and loss spikes
2. **Convergence speed**: Reduces total iterations needed to reach target performance
3. **Final model quality**: Achieves better generalization through proper late-stage refinement
4. **Gradient flow**: Maintains healthy gradient magnitudes through backward passes

### Pre-training vs Fine-tuning

**Pre-training** (training from scratch):
- Typical learning rates: 10⁻⁴ to 10⁻³
- Can tolerate higher early learning rates
- Benefits from aggressive exploration

**Fine-tuning** (adapting pre-trained models):
- Typical learning rates: 10⁻⁵ to 5×10⁻⁵
- Much more sensitive to learning rate magnitude
- Risk of catastrophic forgetting with poor schedules

The fundamental difference: pre-trained weights represent a high-quality local minimum encoding billions of tokens of linguistic knowledge. Large learning rate changes can destroy this knowledge irreversibly.

---

## Schedule Types

### 1. Linear Warmup + Linear Decay

**Best for**: Fine-tuning, standard baseline approach

**Mathematical Formulation**:

During warmup (step t < T_warmup):
```
η_t = (t / T_warmup) × η_base
```

During decay (step t ≥ T_warmup):
```
η_t = η_base × (T_total - t) / (T_total - T_warmup)
```

**Why it works**:
- Protects pre-trained weights during early, volatile phase
- Linear decay provides steady, predictable convergence
- Simple to implement and understand
- Default in BERT, RoBERTa training recipes

**Typical parameters**:
- Warmup fraction: 5-10% of total steps
- Example: 10,000 total steps → 500-1000 warmup steps

**Advantages**:
- Straightforward to reason about
- Stable behavior across different tasks
- No additional hyperparameters

**Disadvantages**:
- May decay too aggressively in middle training
- Reaches zero learning rate at end (can prevent final refinement)

---

### 2. Exponential Decay

**Best for**: Some optimization scenarios, less common for LLMs

**Mathematical Formulation**:
```
η_t = η_initial × exp(-t / τ)
```

Where τ (tau) is the time constant controlling decay speed.

**Alternative with base**: 
```
η_t = η_initial × decay_factor^(t / decay_steps)
```

**Characteristics**:
- Decays quickly early, slows down later
- Never quite reaches zero
- Useful when maintaining some learning rate is important
- Better for oscillatory optimization problems

**Typical parameters**:
- decay_factor: 0.9-0.99 per epoch
- tau: 10,000-50,000 steps for LLMs

**Advantages**:
- Natural exponential form matches many physical processes
- Maintains non-zero learning rate throughout
- Can fine-tune decay speed with exponent

**Disadvantages**:
- Less popular for LLM training
- Less intuitive than linear or cosine schedules

---

### 3. Polynomial Decay

**Best for**: Gradual, controlled convergence

**Mathematical Formulation**:
```
η_t = η_base × ((1 - t/T_total)^p)
```

Where p is the polynomial degree (commonly p=1.0 for linear, p=2.0 for quadratic).

**Used in BERT implementation**:
```
η_t = η_base × max(0, (1 - t/T_total)^p)
```

**Characteristics**:
- p=1: Linear decay (covered separately)
- p=2: Quadratic decay (faster at the end)
- p > 1: Becomes increasingly aggressive late in training

**Typical parameters**:
- p: 1.0-2.0
- Works well with 10-20% warmup fraction

**Advantages**:
- Smooth and mathematically elegant
- Degree parameter provides fine-tuning flexibility
- Used in original BERT paper successfully

**Disadvantages**:
- Reaches zero learning rate (can be problematic)
- Less studied than cosine annealing for recent models

---

### 4. Cosine Annealing (SGDR)

**Best for**: Most modern LLM training, preferred in contemporary practice

**Mathematical Formulation** (after warmup):
```
η_t = η_min + (η_base - η_min) × 0.5 × (1 + cos(π × (t - T_warmup)/(T_total - T_warmup)))
```

**Breaking down the formula**:
- Progress fraction: p = (t - T_warmup) / (T_total - T_warmup)
- Cosine term: cos(π × p) ranges from 1 → -1 as p goes 0 → 1
- Scaled learning rate: Oscillates smoothly from η_base down to η_min

**Characteristics**:
- Slow decay early, faster in middle, slower at end (S-shaped)
- Creates smooth, continuous schedule with no jumps
- Maintains moderate learning rates through training
- Prevents premature convergence

**Typical parameters**:
- η_min (min_lr): 0 to 0.1 × η_base
- η_base: 1e-5 to 5e-5 for fine-tuning
- Warmup fraction: 5-10% of total steps
- num_cycles: 0.5 (default) to 1.0

**Why cosine annealing works**:
1. **Slow early decay**: Allows optimizer to explore
2. **Fast mid-training decay**: Forces convergence to better minima
3. **Slow late decay**: Enables fine-tuning near solution
4. **No abrupt changes**: Smooth gradient throughout

**Cosine with Hard Restarts**:
Periodically resets to high learning rate with multiple cycles:
```
For num_cycles=3, the learning rate undergoes 3 complete cosine cycles
```

**Advantages**:
- Empirically superior performance for LLMs
- Avoids sharp transitions
- Proven by SGDR paper and modern practice
- Works well across diverse tasks

**Disadvantages**:
- More complex than linear schedules
- Introduces additional hyperparameter (min_lr)
- Harder to debug when problems occur

---

### 5. Inverse Square Root (Transformer Schedule)

**Best for**: Large-scale pre-training (original Transformer paper)

**Mathematical Formulation**:
```
η_t = η_base × min(1.0, sqrt(T_warmup / t))  [for t > T_warmup]
```

During warmup:
```
η_t = t / T_warmup × η_base  [linear warmup]
```

**Characteristics**:
- Decays inversely with square root of step count
- Gradual, long-tailed decay curve
- Never reaches zero
- Originally designed for multi-head attention mechanisms

**Mathematical insight**:
- As steps increase, rate decreases as 1/√t
- Slower decay than exponential (which is 1/e^t)
- Maintains higher learning rates longer

**Typical parameters**:
- warmup_steps: 4,000-8,000 for large models
- Works well with large batch sizes
- Base learning rate: 0.5 to 2.0 (normalized differently than others)

**Why it was designed**:
The Transformer paper used this schedule because:
- Compensates for varying gradient magnitudes with scale factor
- Provides steady learning rate decay without premature convergence
- Suitable for complex attention mechanisms

**Advantages**:
- Theoretically motivated for attention mechanisms
- Simple mathematical form
- Proven effective for original Transformer

**Disadvantages**:
- Outperformed by cosine annealing for modern LLMs
- Less commonly used in contemporary practice
- Requires more careful tuning of warmup steps

---

### 6. Step-Based Schedules

**Best for**: Simple heuristic-based training

**Mathematical Formulation**:
```
η_t = η_base × decay_rate^(floor(t / step_size))
```

**Characteristics**:
- Discrete, step-wise reductions
- Learning rate drops by fixed factor every N steps
- No warmup phase (can be added separately)
- Easy to understand and implement

**Example**:
```
Initial LR: 1e-4
Drop by 0.1x every 10,000 steps:
- Steps 0-9,999: LR = 1e-4
- Steps 10,000-19,999: LR = 1e-5
- Steps 20,000+: LR = 1e-6
```

**Typical parameters**:
- decay_factor: 0.1 to 0.5
- step_size: 10,000 to 50,000 steps
- Can use multiple factors: [0.1, 0.1, 0.1]

**Advantages**:
- Very simple to understand and implement
- Useful for quick experiments
- Discrete changes can break plateaus in loss

**Disadvantages**:
- Abrupt learning rate changes can destabilize training
- Not as smooth as continuous schedules
- Less effective for long training runs
- Not recommended for LLM pre-training

---

## Warmup Strategies

Warmup is critical for stable fine-tuning and pre-training. It gradually increases the learning rate from near-zero to the target value over the first few thousand training steps.

### Why Warmup Matters for LLMs

**Initial training is vulnerable**:

1. **Random task head**: Newly added classification/generation layers start with random weights, producing garbage outputs. Large gradients propagate noise back through the network.

2. **Noisy gradient estimates**: Early mini-batches may not be representative. First few batches could contain statistical anomalies that create large, misleading gradients.

3. **Unreliable optimizer statistics**: Adam maintains exponential moving averages of gradient moments:
   - First moment (mean): E[g]
   - Second moment (variance): E[g²]
   - These need ~100 steps to stabilize
   - Bias correction helps but doesn't fully compensate

4. **Catastrophic forgetting risk**: Large updates to pre-trained weights can permanently erase learned knowledge. Unlike training from scratch where this isn't a concern, fine-tuning involves adapting an already-good solution.

### Linear Warmup (BERT Style)

**Most common approach**

**Mathematical Formulation**:
```
During warmup (t < T_warmup):
η_t = t / T_warmup × η_target

After warmup (t ≥ T_warmup):
η_t = η_target (constant or apply decay)
```

**Why linear**:
- Simplest mathematical form
- Provides predictable increase
- No additional hyperparameters
- Empirically effective

**Implementation**:
```python
# In PyTorch
from transformers import get_linear_schedule_with_warmup

num_training_steps = 10000
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# During training loop:
# loss.backward()
# optimizer.step()
# scheduler.step()
```

**Typical parameters**:
- Warmup fraction: 5-10% of total training steps
- For 100,000 step training: 5,000-10,000 warmup steps
- Smaller datasets: Can use shorter warmup (3-5%)
- Larger datasets: Benefit from longer warmup (10-15%)

**Advantages**:
- Simple and robust
- Works across diverse tasks
- Standard baseline in transformers library

**Disadvantages**:
- Can be too fast for very large models
- Doesn't account for task difficulty

---

### Square Root Warmup (LAMB Paper)

**Best for**: Distributed training with large batch sizes

**Mathematical Formulation**:
```
During warmup (t < T_warmup):
η_t = t^0.5 / T_warmup^0.5 × η_target

Interpretation: sqrt of linear
```

**Why square root**:
- Growth rate proportional to √t
- More conservative early, accelerates mid-warmup
- Theoretically motivated for second-order optimization
- Better for layer-wise adaptive learning rates

**When to use**:
- Training with very large batch sizes (>1000)
- Distributed training across many GPUs/TPUs
- When using LAMB optimizer specifically
- Models >10B parameters where instability is common

**Advantages**:
- More conservative early in training
- Better for very large batch sizes
- Prevents early gradient explosions
- Supported in modern optimizers

**Disadvantages**:
- Less commonly used than linear warmup
- Harder to tune than linear
- Additional complexity without clear benefit for small models

---

### Exponential Warmup

**Best for**: Fine-tuning with very sensitive models

**Mathematical Formulation**:
```
During warmup (t < T_warmup):
η_t = (exp(t / T_warmup) - 1) / (e - 1) × η_target

Normalized exponential growth from 0 to η_target
```

**Characteristics**:
- Very conservative at start
- Accelerates toward end of warmup
- Smooth exponential curve
- Stronger protection early

**When to use**:
- Fine-tuning with very small learning rates (<1e-5)
- Models prone to instability
- When linear warmup causes training divergence
- Extreme domain shift scenarios

**Advantages**:
- Maximum early protection
- Natural exponential growth
- Smooth mathematical form

**Disadvantages**:
- Over-cautious for most scenarios
- Longer effective warmup time
- Not widely adopted (hard to compare results)

---

### Warmup Duration Selection

**Based on dataset size**:

| Dataset Size | Typical Warmup Fraction | Example (10k steps) |
|---|---|---|
| < 5k examples | 3-5% | 300-500 steps |
| 5k-50k examples | 5-10% | 500-1000 steps |
| 50k-500k examples | 10-15% | 1000-1500 steps |
| 500k+ examples | 10-20% | 1000-2000 steps |

**Rules of thumb**:
- Minimum: 100 steps (anything less is ineffective)
- Maximum: 20% of training (wastes effective learning time)
- Default: 10% (safe choice across tasks)
- Adjust based on:
  - Task difficulty (harder tasks → longer warmup)
  - Model size (larger models → longer warmup)
  - Pre-training similarity (similar domains → shorter warmup)
  - Batch size (larger batches → longer warmup)

---

## Advanced Methods

### Layer-Wise Learning Rate Decay (LLRD)

**Most important advanced technique for fine-tuning**

**Core Insight**: Not all layers should learn at the same rate. Lower layers encode general linguistic knowledge; upper layers encode task-specific information.

**Mathematical Formulation**:
```
η_l = η_base × ξ^(L - l)

Where:
- η_l: learning rate for layer l
- η_base: base learning rate (for top layer)
- ξ (xi): decay factor (0.9-0.95)
- L: total number of layers
- l: current layer index (L = top, 0 = embeddings)
```

**Example for 12-layer BERT** (η_base=2e-5, ξ=0.95):

| Component | Learning Rate | Ratio to Top |
|---|---|---|
| Layer 12 (top) | 2.0e-5 | 1.00 |
| Layer 11 | 1.9e-5 | 0.95 |
| Layer 10 | 1.81e-5 | 0.90 |
| Layer 6 | 1.47e-5 | 0.74 |
| Layer 1 (bottom) | 1.14e-5 | 0.57 |
| Embeddings | 1.08e-5 | 0.54 |

**Why it works**:

Transformer layers learn hierarchically:
- **Embeddings**: Lexical semantics, word relationships (most general)
- **Layers 1-4**: Syntactic structure, part-of-speech patterns
- **Layers 5-8**: Semantic relationships, coreference
- **Layers 9-12**: Task-specific features (most specialized)

Lower layers have learned structures that ALL downstream tasks benefit from. Disrupting them via large updates hurts performance. Upper layers optimize for the pre-training task (masked LM, next token prediction) and need substantial adaptation.

**Implementation**:

```python
def create_llrd_parameter_groups(model, base_lr=2e-5, decay_factor=0.95):
    """Create parameter groups with layer-wise learning rate decay."""
    
    parameter_groups = []
    num_layers = len(model.transformer.h)  # or model.encoder.layer
    
    # Embedding parameters - lowest learning rate
    embedding_params = []
    for name, param in model.named_parameters():
        if 'embed' in name.lower() and param.requires_grad:
            embedding_params.append(param)
    
    if embedding_params:
        embedding_lr = base_lr * (decay_factor ** num_layers)
        parameter_groups.append({
            'params': embedding_params,
            'lr': embedding_lr,
            'weight_decay': 0.01
        })
    
    # Layer-wise parameters
    for layer_idx, layer_module in enumerate(model.transformer.h):
        layer_lr = base_lr * (decay_factor ** (num_layers - 1 - layer_idx))
        layer_params = [p for p in layer_module.parameters() if p.requires_grad]
        
        parameter_groups.append({
            'params': layer_params,
            'lr': layer_lr,
            'weight_decay': 0.01
        })
    
    # Task head - highest learning rate
    remaining_params = []
    assigned_params = set(id(p) for group in parameter_groups 
                         for p in group['params'])
    
    for param in model.parameters():
        if id(param) not in assigned_params and param.requires_grad:
            remaining_params.append(param)
    
    if remaining_params:
        parameter_groups.append({
            'params': remaining_params,
            'lr': base_lr,  # or higher for task head
            'weight_decay': 0.01
        })
    
    return parameter_groups

# Usage with optimizer
param_groups = create_llrd_parameter_groups(model)
optimizer = torch.optim.AdamW(param_groups)
```

**Choosing decay factor**:
- ξ = 1.0: All layers same rate (no decay)
- ξ = 0.95: Gentle decay (~50% ratio between top and bottom) ← DEFAULT
- ξ = 0.90: Moderate decay (~35% ratio)
- ξ = 0.85: Aggressive decay (~20% ratio)

**Empirical findings**:
- 0.95 works well for most tasks
- 0.90 for very similar domains (domain-specific fine-tuning)
- 1.0 when abundant data available (>100k examples)

**Advantages**:
- Significantly improves fine-tuning performance (2-5% on many benchmarks)
- Prevents catastrophic forgetting
- Respects model architecture
- Simple to implement with modern optimizers

**Disadvantages**:
- Additional hyperparameter to tune
- Requires understanding of model architecture
- More complex than single-rate fine-tuning

---

### Cyclic Learning Rates (CLR)

**Best for**: Escaping local minima, uncertainty estimation

**Core Idea**: Cyclically vary learning rate between min_lr and max_lr, repeating every N steps.

**Mathematical Formulation** (triangular variant):
```
Cycle length: cycle_length = 2 × step_size
Progress in cycle: p = (t mod cycle_length) / cycle_length

If p < 0.5:  # Ascending phase
    η_t = min_lr + (max_lr - min_lr) × 2 × p

If p ≥ 0.5:  # Descending phase
    η_t = max_lr - (max_lr - min_lr) × 2 × (p - 0.5)
```

**Variants**:
- **Triangular**: Linear up, linear down
- **Sawtooth**: Immediate drop, slow rise
- **Sinusoidal**: Smooth cycling

**Characteristics**:
- Learning rate oscillates between min and max
- Complete cycle every 2 × step_size steps
- Allows periodic escaping from local minima
- Creates ensemble-like effect from different LR values

**When to use**:
- Model stuck in loss plateau
- Want to avoid single fixed learning rate
- Uncertainty estimation needed
- Exploratory training phases

**Parameters**:
- min_lr: 1/10 to 1/5 of max_lr
- max_lr: Determined by learning rate range test
- step_size: 1,000-5,000 steps per half-cycle
- Momentum: 0.8-0.99

**Advantages**:
- Escapes local minima
- Simple to implement
- Ensemble learning benefits

**Disadvantages**:
- Not standard for LLM training
- Can cause loss instability
- Requires finding good min/max range
- Less predictable convergence behavior

---

### Warm Restarts (SGDR)

**Best for**: Long training runs, avoiding premature convergence

**Core Idea**: Periodically restart the schedule, returning to high learning rates while resetting optimizer state. Different from CLR by having "hard" resets.

**Mathematical Formulation**:
```
Restart period: T_restart
Current period: p = floor(t / T_restart)
Progress in period: x = (t mod T_restart) / T_restart

Learning rate (cosine schedule in each period):
η_t = η_min + (η_max - η_min) × 0.5 × (1 + cos(π × x))

Optional: Increase T_restart after each period: T_restart → T_restart × T_mult
```

**Characteristics**:
- Multiple complete cosine annealing cycles
- Each cycle goes from high LR down to minimum
- Can optionally increase cycle length over time
- Creates "warm restarts" without random shuffling

**When to use**:
- Very long training (1M+ steps)
- Want multiple exploration phases
- Traditional ML training paradigm
- Model needs escape from local minima

**Parameters**:
- T_0: Initial restart period (e.g., 10,000 steps)
- T_mult: Period multiplier (e.g., 2 = double each time)
- η_min: Minimum learning rate
- η_max: Maximum learning rate

**Example schedule**:
```
Restart 1: Steps 0-10,000 (cosine from max to min)
Restart 2: Steps 10,000-30,000 (cosine, doubled period)
Restart 3: Steps 30,000-70,000 (cosine, doubled again)
```

**Advantages**:
- Theoretically motivated by SGDR paper
- Provides periodic exploration
- Useful for non-convex optimization

**Disadvantages**:
- Less commonly used in modern LLM training
- Abrupt LR jumps can destabilize training
- Harder to debug than smooth schedules
- Not recommended as primary schedule

---

### Learning Rate Range Test (LR Finder)

**Best for**: Finding optimal learning rate bounds**

**Purpose**: Empirically determine what learning rates work for your model on your data.

**Algorithm**:
1. Start with very low learning rate (e.g., 1e-7)
2. Train for 1 epoch while exponentially increasing LR
3. Record loss at each LR
4. Plot loss vs LR (log scale)
5. Find LR where loss starts decreasing sharply (lower bound)
6. Find LR where loss starts diverging (upper bound)

**Implementation**:

```python
from torch_lr_finder import LRFinder

# Create model, criterion, optimizer
model = your_model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

# Create LR finder
lr_finder = LRFinder(model, optimizer, criterion, device='cuda')

# Run range test
lr_finder.range_test(
    train_loader,
    start_lr=1e-7,
    end_lr=10,
    num_iter=200,
    step_mode='exp'
)

# Plot results
lr_finder.plot(log_lrs=True)  # Log scale on x-axis

# Get optimal LR
best_loss = min(lr_finder.history['loss'])
best_lr_idx = lr_finder.history['loss'].index(best_loss)
best_lr = lr_finder.lrs[best_lr_idx]

# Typically, use ~1/10 of best_lr as your training LR
recommended_lr = best_lr / 10
```

**Interpretation guide**:

```
Loss vs LR plot characteristics:

Ideal plot:
- Sharp decrease in loss (good learning)
- Flattens out mid-range (stable)
- Rises sharply at high LRs (divergence)
- Choose LR in steep descent region

Steep throughout:
- Model hasn't converged yet
- Try extended training
- Check data/model architecture

Diverges immediately:
- Learning rate too high even at start
- Reduce by 10x
- Check for NaN gradients

Flat throughout:
- Learning rate too low
- Increase by 10x
- Check optimization setup
```

**Best practices**:
- Run on representative subset of training data (~1-10% of full dataset)
- Use same batch size and optimizer as main training
- 100-200 iterations provides good resolution
- Choose LR roughly 10x lower than "break point" where loss rises

**Advantages**:
- Empirical, data-driven approach
- Avoids guessing learning rates
- Works for any model/data combination
- Useful for new architectures/tasks

**Disadvantages**:
- Takes extra time (1 epoch minimum)
- Creates outliers (doesn't represent normal training)
- Need visualization to interpret correctly
- May not transfer to different batch sizes

---

## Implementation Guide

### PyTorch Native Schedulers

**Linear Warmup + Cosine Decay** (Recommended):

```python
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Option 1: Using transformers library (simplest)
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

num_training_steps = 10000
num_warmup_steps = 1000

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    num_cycles=0.5,  # Half cosine cycle
    last_epoch=-1
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # Update LR each step, not each epoch
```

**With LLRD** (Advanced):

```python
def create_llrd_optimizer(model, base_lr=2e-5, decay_factor=0.95):
    """Create optimizer with layer-wise learning rate decay."""
    
    # Import optimizer class
    from transformers import AdamW
    
    # Separate parameters by layer
    no_decay = ['bias', 'LayerNorm.weight']
    
    param_groups = []
    num_layers = len(model.transformer.h)
    
    # Build layer groups with decay
    layer_lr_dict = {}
    for layer_idx in range(num_layers):
        layer_lr = base_lr * (decay_factor ** (num_layers - 1 - layer_idx))
        layer_lr_dict[layer_idx] = layer_lr
    
    # Embeddings (lowest learning rate)
    embedding_lr = base_lr * (decay_factor ** num_layers)
    
    # Process all parameters
    embedding_params = []
    layer_params = {i: [] for i in range(num_layers)}
    other_params = []
    
    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            embedding_params.append(param)
        elif 'transformer.h' in name:
            # Extract layer number
            layer_num = int(name.split('transformer.h.')[1].split('.')[0])
            layer_params[layer_num].append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': embedding_lr,
            'weight_decay': 0.01
        })
    
    for layer_idx, params in layer_params.items():
        if params:
            param_groups.append({
                'params': params,
                'lr': layer_lr_dict[layer_idx],
                'weight_decay': 0.01
            })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr * 2,  # Task head gets 2x base
            'weight_decay': 0.01
        })
    
    optimizer = AdamW(param_groups, eps=1e-8)
    return optimizer
```

### HuggingFace Trainer Configuration

**Simple configuration**:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    
    # Learning rate scheduling
    learning_rate=2e-5,
    warmup_ratio=0.1,  # 10% of steps for warmup
    warmup_steps=0,  # Ignored if warmup_ratio is set
    lr_scheduler_type='cosine',  # Options: 'linear', 'cosine', 'polynomial'
    
    # Other important settings
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    
    # Logging and saving
    logging_steps=100,
    eval_strategy='steps',
    eval_steps=500,
    save_strategy='steps',
    save_steps=500,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

**Advanced configuration with minimum LR**:

```python
training_args = TrainingArguments(
    ...
    lr_scheduler_type='cosine_with_min_lr',
    learning_rate=5e-5,
    warmup_ratio=0.1,
    min_lr_ratio=0.01,  # Minimum LR is 1% of initial
    ...
)
```

### Custom Scheduler Implementation

**For advanced use cases not covered by built-in schedulers**:

```python
class CustomLRScheduler:
    """Custom learning rate scheduler combining multiple techniques."""
    
    def __init__(self, optimizer, base_lr, num_warmup_steps, num_training_steps,
                 warmup_type='linear', decay_type='cosine', min_lr_ratio=0.0):
        """
        Args:
            optimizer: PyTorch optimizer
            base_lr: Initial learning rate
            num_warmup_steps: Steps for warmup phase
            num_training_steps: Total training steps
            warmup_type: 'linear', 'sqrt', 'exponential'
            decay_type: 'cosine', 'linear', 'polynomial'
            min_lr_ratio: Minimum LR as ratio of base_lr
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.warmup_type = warmup_type
        self.decay_type = decay_type
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
    
    def get_lr(self):
        """Compute learning rate for current step."""
        
        if self.current_step < self.num_warmup_steps:
            return self._get_warmup_lr()
        else:
            return self._get_decay_lr()
    
    def _get_warmup_lr(self):
        """Linear warmup: 0 → base_lr over warmup period."""
        progress = self.current_step / self.num_warmup_steps
        
        if self.warmup_type == 'linear':
            return self.base_lr * progress
        elif self.warmup_type == 'sqrt':
            return self.base_lr * (progress ** 0.5)
        elif self.warmup_type == 'exponential':
            # exp(x) - 1 scaled to go from 0 to 1
            return self.base_lr * (np.exp(progress) - 1) / (np.e - 1)
        else:
            return self.base_lr * progress
    
    def _get_decay_lr(self):
        """Decay LR after warmup phase."""
        progress = (self.current_step - self.num_warmup_steps) / \
                   (self.num_training_steps - self.num_warmup_steps)
        progress = np.clip(progress, 0, 1)
        
        min_lr = self.base_lr * self.min_lr_ratio
        
        if self.decay_type == 'cosine':
            # Cosine annealing
            lr_range = self.base_lr - min_lr
            return min_lr + lr_range * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.decay_type == 'linear':
            # Linear decay
            return self.base_lr * (1 - progress) + min_lr * progress
        elif self.decay_type == 'polynomial':
            # Polynomial decay (p=2 = quadratic)
            return self.base_lr * (1 - progress) ** 2 + min_lr * progress
        else:
            return self.base_lr
    
    def step(self):
        """Update learning rate and move to next step."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr

# Usage
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = CustomLRScheduler(
    optimizer,
    base_lr=5e-5,
    num_warmup_steps=1000,
    num_training_steps=50000,
    warmup_type='linear',
    decay_type='cosine',
    min_lr_ratio=0.01
)

for epoch in range(num_epochs):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

### Integration with Optimizers

**AdamW (Recommended for LLMs)**:

```python
# AdamW with proper configuration
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),  # Default, but can tune
    eps=1e-8,  # Numerical stability
    weight_decay=0.01  # L2 regularization (unlike standard Adam)
)

# With scheduler
scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, 10000)
```

**Other optimizers**:

```python
# SGD with momentum (rarely used for LLMs but possible)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001
)

# Applies to all optimizers
scheduler = get_linear_schedule_with_warmup(optimizer, 1000, 10000)

# Note: Learning rates are MUCH higher for SGD than Adam
# SGD typically uses 0.01-0.1 vs Adam's 1e-5 to 1e-3
```

---

## Empirical Analysis

### Convergence Curves

**Comparison of schedules on typical fine-tuning task**:

Based on empirical studies from 2024-2026 LLM training:

1. **Linear Warmup + Linear Decay** (Baseline)
   - Convergence: Steady, predictable
   - Final Loss: Higher than cosine
   - Stability: Good
   - Recommendation: Default safe choice

2. **Linear Warmup + Cosine Decay** (Recommended)
   - Convergence: Smooth S-curve
   - Final Loss: Best performance
   - Stability: Excellent
   - Recommendation: Use by default

3. **Polynomial Decay** (p=1.5)
   - Convergence: Aggressive mid-training
   - Final Loss: Similar to cosine
   - Stability: Good
   - Recommendation: Alternative to cosine

4. **Step-based Decay**
   - Convergence: Jumpy, with plateaus
   - Final Loss: Often suboptimal
   - Stability: Poor
   - Recommendation: Avoid for important training

5. **Inverse Sqrt**
   - Convergence: Very slow, long tail
   - Final Loss: Good with right setup
   - Stability: Excellent
   - Recommendation: Useful for pre-training

### Effect of Warmup Duration

**Empirical findings from BERT-style fine-tuning**:

| Warmup % | Task Performance | Training Stability | Recommendation |
|---|---|---|---|
| 0% | Low (78-80%) | Poor (loss spikes) | Never use |
| 3% | Medium (82-83%) | Okay | Minimum viable |
| 5% | Good (83-84%) | Good | Acceptable |
| 10% | Better (84-85%) | Excellent | Recommended |
| 15% | Good (84-85%) | Excellent | Also good |
| 20%+ | Same or slightly worse | Excellent | Overkill |

**Key insights**:
- Too little warmup: Erratic early training, worse final performance
- Too much warmup: Wastes time with minimal learning
- Sweet spot: 5-10% for most tasks
- Domain-specific: Increase for novel domains, decrease for similar ones

### Schedule Interaction with Batch Size

**How batch size affects optimal schedules**:

| Batch Size | Warmup Fraction | Decay Type | Typical LR |
|---|---|---|---|
| 8-16 | 5-10% | Linear | 2-5e-5 |
| 32-64 | 10-15% | Cosine | 1-2e-5 |
| 128-256 | 15-20% | Cosine | 0.5-1e-5 |
| 512+ | 20-30% | Inverse sqrt | 0.1-0.5e-5 |

**Explanation**:
- **Larger batches**: More stable gradients, can handle higher LR
- **Smaller batches**: Noisier gradients, need lower LR and longer warmup
- **Warmup relationship**: Larger batches benefit from longer warmup (statistical stabilization)
- **Decay type**: Larger batches work with aggressive decay; smaller batches need conservative schedules

### Model Size and Learning Rate

**Empirical scaling laws**:

| Model Size | Recommended Base LR | Typical Range |
|---|---|---|
| <1B parameters | 1-5e-5 | 0.5-10e-5 |
| 1-10B | 0.5-2e-5 | 0.2-5e-5 |
| 10-100B | 0.1-1e-5 | 0.05-2e-5 |
| 100B+ | 0.05-0.5e-5 | 0.01-1e-5 |

**Why this pattern**:
- Larger models more sensitive to learning rate
- Pre-trained knowledge more fragile in bigger models
- Use smaller LR to prevent catastrophic forgetting
- Learn rate finder critical for new large models

---

## Best Practices

### Choosing a Schedule for Your Use Case

**Pre-training from scratch**:
```
Use: Inverse sqrt warmup + inverse sqrt decay
Reasoning: Original Transformer paper design
LR: 1e-4 to 5e-4 (normalized)
Warmup: 4-10k steps or 0.1-1% of total
```

**Fine-tuning** (primary use case for LLMs):
```
Use: Linear warmup (5-10%) + cosine decay
Reasoning: Proven best for transformer fine-tuning
LR: 1-5e-5 (depends on task and model size)
Min LR: 0.01 × base_lr (allows fine convergence)
Warmup steps: 5-10% of total steps
```

**Domain adaptation** (fine-tuning on very different text):
```
Use: Linear warmup (10-15%) + cosine decay + LLRD
Reasoning: Conservative approach for large domain shift
LR: 0.5-2e-5 (more conservative)
Decay factor: 0.90 (more aggressive layer protection)
Warmup steps: 10-15% of total
```

**Few-shot learning** (< 1000 examples):
```
Use: Linear warmup (5-10%) + linear decay + LLRD + lower LR
Reasoning: Minimize forgetting with limited data
LR: 0.1-1e-5 (very conservative)
Decay factor: 0.85-0.90 (protect lower layers)
Warmup steps: 5-10% (but absolute minimum 100 steps)
```

**Continued pre-training** (on new domain):
```
Use: Warmup (0.1-1%) + cosine decay
Reasoning: Model already well-optimized, needs gentle adjustment
LR: 0.5-5e-5 (intermediate between pre-training and fine-tuning)
Warmup: 0.1-1% of steps (brief, stabilization only)
```

### Interaction with Optimizer Choice

**With AdamW** (most common):
- Works well with all schedule types
- Benefits from warmup (important for this optimizer)
- Learning rates: 1e-5 to 1e-3 range typical
- Default choice for modern LLM training

**With LAMB** (large batch training):
- Use longer warmup (typically sqrt-based)
- Can handle higher learning rates
- Batch size dependent: 1e-3 for batch_size=512, 1e-4 for batch_size=4096
- Good for distributed training

**With SGD** (rare for LLMs):
- Much higher learning rates needed: 0.01-0.1
- Benefits from step-based decay
- Less research on schedules for SGD + transformers
- Not recommended unless specifically required

**With Adafactor** (memory-efficient):
- Has built-in learning rate scheduling
- Often use learning_rate=None for automatic schedule
- Supports different warmup strategies
- Alternative when memory constrained

### Debugging Convergence Issues

**Symptom: Loss diverges (NaN)**
```
Solution 1: Reduce learning rate by 10x
Solution 2: Increase warmup duration
Solution 3: Check for data issues (duplicates, extreme values)
Solution 4: Verify gradient clipping is enabled
Action: Run learning rate range test
```

**Symptom: Loss plateaus, barely improves**
```
Solution 1: Increase learning rate by 2-5x
Solution 2: Use learning rate range test to find optimal value
Solution 3: Reduce batch size (noisier gradients help escape)
Solution 4: Check that scheduler is actually decreasing LR
Action: Print LR at each step to verify
```

**Symptom: Large oscillations in loss**
```
Solution 1: Reduce learning rate
Solution 2: Increase batch size (stable gradients)
Solution 3: Use longer warmup (stabilize early training)
Solution 4: Reduce gradient accumulation steps
Action: Monitor gradient norms during training
```

**Symptom: Good training loss, poor eval performance (overfitting)**
```
Solution: Not a learning rate problem
Solution 1: Increase regularization (weight decay)
Solution 2: Use dropout or other regularization
Solution 3: Increase early stopping patience before critical
Issue: Model memorizing rather than generalizing
```

### Hyperparameter Tuning Strategy

**Step 1: Determine search range**
```python
# Run learning rate range test
lr_finder.range_test(train_loader, 1e-7, 10)
# Identify steep descent region
# Use optimal_lr / 10 as starting point
```

**Step 2: Coarse-grained search**
```python
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
warmup_fractions = [0.05, 0.10, 0.15]

# Try combinations, evaluate on validation set
best_lr = None
best_warmup = None
```

**Step 3: Fine-grained search**
```python
# Around best from step 2
learning_rates = [2e-5, 3e-5, 5e-5, 7e-5, 1e-4]
warmup_fractions = [0.08, 0.10, 0.12]
# Add: decay_factors if using LLRD

# Evaluate on validation set
# Repeat 2-3 times for stability
```

**Step 4: Final validation**
```python
# Use best hyperparameters
# Train multiple random seeds
# Report mean and std of metrics
# Only commit to final model after this
```

### Monitoring and Logging

**Essential metrics to track**:

```python
import wandb

# Initialize wandb
wandb.init(project="llm-training")

# During training
for step, batch in enumerate(train_loader):
    outputs = model(**batch)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    
    # Log current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Log metrics every N steps
    if step % 100 == 0:
        wandb.log({
            'loss': loss.item(),
            'learning_rate': current_lr,
            'step': step,
            'epoch': step // len(train_loader)
        })
        
        # Print for manual monitoring
        print(f"Step {step}: Loss={loss:.4f}, LR={current_lr:.2e}")

wandb.finish()
```

**Visualization of schedule over time**:

```python
import matplotlib.pyplot as plt

# Generate schedule visualization
steps = range(total_steps)
lrs = [scheduler.get_lr_at_step(s) for s in steps]

plt.figure(figsize=(12, 4))
plt.plot(steps, lrs)
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.xscale('linear')
plt.yscale('log')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.axvline(x=num_warmup_steps, color='r', linestyle='--', label='End Warmup')
plt.legend()
plt.tight_layout()
plt.savefig('lr_schedule.png', dpi=150)
```

---

## References

### Key Papers

1. **SGDR: Stochastic Gradient Descent with Warm Restarts** (Loshchilov & Hutter, 2016)
   - URL: https://arxiv.org/abs/1608.03983
   - Key contribution: Cosine annealing and warm restarts
   - Impact: Foundational for modern LLM scheduling

2. **Attention Is All You Need** (Vaswani et al., 2017)
   - URL: https://arxiv.org/abs/1706.03762
   - Introduces: Inverse square root schedule
   - Relevance: Original Transformer training recipe

3. **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
   - URL: https://arxiv.org/abs/1810.04805
   - Introduces: Linear warmup + linear decay standard
   - Impact: Widely adopted for fine-tuning

4. **Universal Language Model Fine-tuning for Text Classification** (Howard & Ruder, 2018)
   - URL: https://arxiv.org/abs/1801.06146
   - Key contribution: Layer-wise learning rate decay (LLRD)
   - Application: ULMFiT fine-tuning recipe

5. **Cyclical Learning Rates for Training Neural Networks** (Smith, 2017)
   - URL: https://arxiv.org/abs/1506.01186
   - Introduces: CLR and learning rate ranges
   - Relevance: Alternative scheduling approach

6. **Large Batch Training of Convolutional Networks** (You et al., 2018)
   - URL: https://arxiv.org/abs/1708.03888
   - Introduces: LARS and LAMB optimizers
   - Relevance: Batch size effects on learning rates

7. **Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training** (Kosson et al., 2024)
   - URL: https://arxiv.org/abs/2410.23922
   - Recent analysis: When warmup is necessary
   - Impact: Contemporary understanding of warmup necessity

8. **Fine-tuning Learning Rates: LLRD, Warmup & Decay Strategies** (Brenndoerfer, 2025)
   - URL: https://mbrenndoerfer.com/writing/fine-tuning-learning-rates-llrd-warmup-decay-transformers
   - Comprehensive modern guide with visualizations
   - Practical examples for transformers

### Implementation Resources

- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
- **PyTorch Documentation**: https://pytorch.org/docs/stable/optim.html
- **PyTorch LR Finder**: https://github.com/davidtvs/pytorch-lr-finder
- **FastAI Learning Rate Finder**: https://github.com/fastai/fastai

### Recommended Reading Order

For beginners:
1. Introduction + Why Learning Rate Scheduling Matters
2. Linear Warmup + Linear Decay
3. Best Practices section
4. BERT paper for concrete example

For intermediate practitioners:
1. All schedule types
2. LLRD section
3. Implementation guide
4. Debugging section
5. SGDR paper for mathematical intuition

For advanced practitioners:
1. All sections
2. References to original papers
3. Custom implementation examples
4. Empirical analysis section
5. Recent papers (2024-2026) for state-of-the-art

---

## Quick Reference

### Cheat Sheet

**For fine-tuning** (most common):
```
Schedule: Linear warmup (10%) + cosine decay
Learning rate: 2e-5 (small), 5e-5 (medium), 1e-4 (large models)
Warmup steps: 10% of total training steps
Min LR: 0.01 × base_lr
Optimizer: AdamW with weight_decay=0.01
LLRD: Yes, decay_factor=0.95
```

**Default PyTorch code**:
```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

for epoch in range(num_epochs):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

### Decision Tree

```
Are you pre-training from scratch?
├─ Yes → Use: Inverse sqrt warmup + decay, LR=1e-4
└─ No → Are you fine-tuning?
    └─ Yes → Use: Linear warmup (10%) + cosine, LR=2e-5
        ├─ Small dataset (< 5k)? → Lower LR, longer warmup
        ├─ Novel domain? → Add LLRD with decay=0.95
        └─ Large model (>10B)? → Lower LR, longer warmup

Convergence problem?
├─ Loss diverges/NaN → Reduce LR by 10x, increase warmup
├─ Loss plateaus → Increase LR by 2-5x, run LR finder
├─ Large oscillations → Reduce LR, increase batch size
└─ Overfitting → Not LR issue, add regularization
```

---

## Conclusion

Learning rate scheduling is a critical component of successful large language model training. Modern practice has settled on linear warmup followed by cosine annealing decay as the default choice, with layer-wise learning rate decay providing significant improvements for fine-tuning tasks.

The key principles to remember:
1. **Warmup is essential**: Protects pre-trained weights during vulnerable early phase
2. **Cosine decay works well**: Smooth, theoretically motivated schedule with good empirical results
3. **Respect model hierarchy**: Use LLRD to preserve general knowledge in lower layers
4. **Monitor everything**: Track learning rates and losses to catch problems early
5. **Empirical validation**: Use learning rate finder for new models or tasks
6. **Task-specific tuning**: Adjust parameters based on dataset size, domain similarity, and model size

With these techniques and best practices, you'll be well-equipped to train and fine-tune large language models effectively.
