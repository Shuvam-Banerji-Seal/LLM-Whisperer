# Mixed-Precision Training: Complete Implementation Guide

## 1. Introduction

### Why Mixed-Precision Training Matters

Mixed-precision training is a computational technique that combines lower-precision (16-bit or 8-bit) floating-point formats with full-precision (32-bit) formats to accelerate deep learning model training while maintaining model quality. This approach has become essential for training billion-parameter language models.

### Key Benefits

- **2x Memory Savings**: Reducing precision from FP32 to FP16/BF16 halves activation memory usage, enabling larger batch sizes or models
- **3x Speedup**: NVIDIA Tensor Cores execute 16-bit operations significantly faster than FP32 operations (up to 3x on newer architectures, 1.59x with FP8)
- **Reduced Power Consumption**: Lower numerical precision requires fewer computational resources and reduced data movement
- **Maintained Model Quality**: With proper techniques (loss scaling, master weights), final model quality is indistinguishable from FP32 training

### Recent Advances (2025-2026)

- **FP8 Training**: 30-40% throughput improvements over BF16, 50% memory savings vs BF16
- **NVFP4**: Up to 1.59x speedup with 4-bit quantization using hierarchical 2-level scaling
- **MXFP8**: Block-level scaling optimized for NVIDIA Blackwell architecture (32 element blocks)
- **Production-Ready**: Major frameworks (Meta, Google, DeepL) now train frontier models with FP8/FP4

---

## 2. Numeric Types and Floating-Point Formats

### 2.1 FP32 (Full Precision / float32)

**Representation**: 1 sign bit + 8 exponent bits + 23 mantissa bits

```
[Sign (1)] [Exponent (8)] [Mantissa (23)]
```

**Properties**:
- Range: ±3.4e-38 to ±3.4e+38
- Precision: ~7 decimal digits
- Default for weight updates and loss computation in mixed precision
- Used as "master weights" for convergence guarantees

**Typical Usage**:
- All gradient accumulation and weight updates
- Loss computation (prevents underflow)
- Optimizer state (momentum, variance in Adam)

### 2.2 FP16 (Half Precision / float16)

**Representation**: 1 sign bit + 5 exponent bits + 10 mantissa bits

```
[Sign (1)] [Exponent (5)] [Mantissa (10)]
```

**Properties**:
- Range: ±6.1e-5 to ±6.55e+4
- Precision: ~4 decimal digits
- Smallest normal positive: 6.1e-5
- Smallest subnormal: 5.96e-8

**Issues with FP16**:
- **Gradient Underflow**: Gradients < 6.1e-5 become zero (critical limitation)
- **Gradient Overflow**: Gradients > 6.55e+4 become infinity
- **Loss of Precision**: Small weight updates (e.g., 1e-6) get rounded away
- **Historical Issues**: Difficult convergence without special handling

**Advantages**:
- Fast on older hardware (V100, older Tensor Cores)
- Wide GPU support

**When to Use**:
- Legacy systems requiring FP16 support
- Cost-sensitive scenarios (memory/bandwidth critical)

### 2.3 BF16 (Brain Float / bfloat16)

**Representation**: 1 sign bit + 8 exponent bits + 7 mantissa bits

```
[Sign (1)] [Exponent (8)] [Mantissa (7)]
```

**Key Innovation**: Matches FP32 exponent range while cutting mantissa precision

**Properties**:
- Range: ±9.18e-41 to ±3.39e+38 (matches FP32 range!)
- Precision: ~3 decimal digits (lower than FP16)
- Smallest normal positive: 1.17e-38
- Directly truncates FP32 (no rounding needed)

**Comparison with FP16**:
```
Property         | BF16          | FP16
Range Exponent   | [8 bits]      | [5 bits]
Mantissa Bits    | [7 bits]      | [10 bits]
Gradient Range   | Wide (±e38)   | Narrow (±e4)
Gradient Underflow | Rare        | Common
Hardware Support | Modern GPUs   | All GPUs
Default Choice   | YES           | Mostly replaced
```

**Advantages**:
- Wider dynamic range prevents gradient underflow
- Simpler than FP16 (direct truncation from FP32)
- Now standard on NVIDIA A100+, Google TPUs, modern CPUs
- Better convergence than FP16 without special tricks

**Disadvantages**:
- Lower precision (7 bits) than FP16 (10 bits)
- Slightly larger loss values during training (usually converges the same)

**When to Use**:
- Default choice for modern hardware (A100, H100, Blackwell)
- Training large language models (Llama, GPT, BERT)
- First choice when mixed precision is available

### 2.4 FP8 (8-bit Floating Point)

**Representation**: 1 sign bit + 4 exponent bits + 3 mantissa bits (E4M3 format)

```
[Sign (1)] [Exponent (4)] [Mantissa (3)]
```

**Variants**:

1. **E4M3F (NVIDIA Native)**:
   - Range: ±2.4e-4 to ±4.48e+4
   - Most common for training
   - NVIDIA GPU native support

2. **E5M2**:
   - Range: ±6.1e-4 to ±5.5e+4
   - Better for dynamic ranges

**Scaling Strategies**:

- **Tensor Scaling (FP8-CS)**: Single scale factor per tensor
  - Simplest approach
  - Suitable for most operations
  - Scale = max(abs(tensor)) / max_representable
  
- **Block Scaling (MXFP8)**: Scale factor per 32 elements
  - Finer granularity for Blackwell optimization
  - Better utilization of reduced range
  - ~1-2% performance improvement over tensor scaling

- **Hierarchical Scaling (NVFP4)**: 2-level scaling structure
  - Global 16×16 block scales
  - Per-row 1×16 block refinement
  - Optimal for low-rank weight matrices

**Challenges**:
- Requires careful scale selection
- Potential for outlier accumulation
- Dynamic scaling overhead
- Need master weights in FP32

**Latest Research (2025)**:
- TWEO (Transformers Without Extreme Outliers): Quantization-aware training to prevent outliers
- NUnit scaling: Simple per-channel scaling for stability
- 30-40% throughput vs BF16, 50% memory savings
- Successfully trains Llama 3 8B, 70B models

**When to Use**:
- Modern Blackwell (H100, GB200) or newer hardware
- Production systems with high throughput requirements
- When memory is critical constraint

### 2.5 FP4 / NVFP4 (4-bit Floating Point)

**Representation**: 1 sign bit + 2 exponent bits + 1 mantissa bit

```
[Sign (1)] [Exponent (2)] [Mantissa (1)]
```

**NVIDIA's Hierarchical Approach**:
- Global scales computed per 16×16 weight block
- Per-row scales for fine-grained control
- Maintains FP32 master weights
- Selective BF16 layers for stability

**Properties**:
- Extreme compression (16x vs FP32)
- Requires selective layer precision
- Works best with final 4 layers in BF16
- 1.59x speedup over BF16 on Blackwell

**Practical Recipe (from NVIDIA/DeepL)**:
```
Optimizer: AdamW with epsilon=1e-8
Learning Rate: 6e-4 → 6e-6 (warmup/decay)
Global Batch Size: 768
Sequence Length: 8192
FP4 Layers: All transformer blocks except last 4
Master Weights: Always FP32
Gradient Accumulation: FP32
```

**When to Use**:
- Next-generation training (Blackwell+)
- Ultra-large models where memory is bottleneck
- Cost-sensitive production training

### 2.6 TF32 (Tensor Float 32)

**Representation**: 1 sign bit + 8 exponent bits + 10 effective mantissa bits

```
[Sign (1)] [Exponent (8)] [Effective Mantissa (10)]
```

**Properties**:
- Hardware native on A100, H100, Blackwell
- Hybrid precision using tensor cores
- Automatically used for matmul operations
- Falls between BF16 and FP32 in precision
- Zero code changes required

**Performance**:
- ~4x speedup vs FP32 on tensor cores
- Minimal accuracy loss
- Enabled by default on NVIDIA GPUs

**Unique Advantage**:
- Works with existing FP32 code
- Automatic precision conversion at hardware level
- No loss scaling or master weights needed

**When to Use**:
- When you don't want to modify code
- Fine-tuning existing models
- Quick performance boost (easy win)

---

## 3. Theoretical Foundation

### 3.1 IEEE 754 Floating-Point Representation

**General Formula**:
```
Value = (-1)^sign × (1 + mantissa/2^p) × 2^(exponent - bias)
```

**Key Concepts**:

1. **Denormalized Numbers (Subnormals)**:
   - When exponent = 0, implicit leading 1 becomes 0
   - Provides gradual underflow instead of cliff
   - Example: FP16 smallest normal is 6.1e-5, smallest subnormal is 5.96e-8

2. **Exponent Bias**:
   - FP32: bias = 127
   - FP16: bias = 15
   - BF16: bias = 127
   - FP8: bias = 7 (E4M3) or 15 (E5M2)

3. **Unit in the Last Place (ULP)**:
   - Distance between consecutive representable numbers
   - Increases with exponent value
   - Critical for understanding rounding

### 3.2 Loss Landscape Analysis in Low Precision

**Challenge**: How do lower-precision formats affect optimization dynamics?

**Key Findings from Recent Research**:

1. **Gradient Discretization**:
   - FP16 gradients: Quantized to ~1e-7 intervals at typical scales
   - BF16 gradients: Quantized to ~1e-5 intervals (50x coarser!)
   - Still sufficient for convergence with proper scaling

2. **Loss Surface Smoothness**:
   ```
   FP32 loss landscape: Smooth, continuous gradients
   FP16 loss landscape: Discrete steps (gradient underflow)
   BF16 loss landscape: Discrete but wider steps (still smooth enough)
   FP8 loss landscape: Requires careful scaling per layer
   ```

3. **Critical Issue - Gradient Underflow**:
   ```
   Normal training gradient: 1e-6 (very small but important)
   FP32 smallest non-zero: ~1.4e-45 ✓ (representable)
   BF16 smallest non-zero: ~1.4e-45 ✓ (representable)
   FP16 smallest normal: 6.1e-5 ✗ (UNDERFLOW to 0!)
   
   Solution: Scale loss up by 2^k before backward pass
   Example: Scale by 1024 (2^10) → gradient becomes 1e-3
   → Representable in any format → Can descale after step
   ```

4. **Precision vs Convergence Trade-off**:
   ```
   More Precision  │ BF16: ~7 bits mantissa, wide range
                   │ FP16: ~10 bits mantissa, narrow range  
                   │ FP32: ~23 bits mantissa, wide range
                   
   BF16 Insight: Wide range (like FP32) often matters more than
                 precision (mantissa bits) for convergence
   ```

### 3.3 Convergence Properties with Lower Precision

**Theorem (Simplified)**:
> For any loss function and optimizer, if gradients are scaled to avoid underflow/overflow, training convergence rate depends on:
> 1. Optimizer choice (SGD, Adam, AdamW)
> 2. Learning rate schedule
> 3. Precision of loss computation
> 
> NOT significantly on forward/backward precision (with proper scaling)

**Empirical Results (NVIDIA 2026 Study - 1 Trillion Token Training)**:

| Metric | BF16 | FP8-CS | MXFP8 | NVFP4 |
|--------|------|--------|-------|-------|
| Training Loss | Baseline | -0.02% | -0.05% | +0.3% |
| Validation Loss | Baseline | -0.01% | -0.02% | +0.2% |
| Downstream Accuracy | 100% | 99.8% | 99.9% | 99.7% |
| Llama 3 8B MMLU | 45.98 | 46.00 | 46.56 | 45.64 |
| HellaSwag Score | 76.44 | 75.25 | 75.46 | 75.59 |

**Key Insight**: Loss values differ slightly, but final model quality is nearly identical!

**Convergence Proof Sketch**:
```
1. Scaled loss ensures gradient ∈ [feasible_range]
2. Optimizer step: w_t+1 = w_t - α * g_t (same regardless of precision)
3. Slight rounding in low precision affects only ULP level
4. Averaged over millions of steps → negligible impact
5. Model convergence determined by optimization trajectory, not digit precision
```

---

## 4. Loss Scaling: The Critical Technique

### 4.1 Why Loss Scaling is Necessary

**The Gradient Underflow Problem**:

```python
# Without loss scaling in FP16
loss = cross_entropy(logits, labels)  # Loss ≈ 2.5 (reasonable)
loss.backward()                        # Gradient ≈ 1e-6 (too small!)
# In FP16, 1e-6 < minimum normal (6.1e-5)
# Result: gradient becomes 0.0 due to underflow
# No learning happens!

# With loss scaling
loss_scaled = loss * 1024             # Scale up to 2560
loss_scaled.backward()                # Gradient ≈ 1e-3 (now representable!)
grad = grad / 1024                    # Scale down after step
# Weights update correctly!
```

**Mathematical Perspective**:
```
Gradient chain rule: ∂L/∂w = ∂L/∂output × ∂output/∂w

Without scaling:  ∂L/∂w ≈ 1e-6 (underflows in FP16)
With scale 2^k:   ∂(k×L)/∂output = k × ∂L/∂output
                  k × ∂L/∂w ≈ 1e-3 (representable)
After descaling:  ∂L/∂w ≈ 1e-6 (mathematically correct)
```

**Why BF16 Needs Less Scaling**:
- BF16 minimum normal: 1.17e-38 (vs FP16: 6.1e-5)
- Gradients rarely underflow in BF16
- Can often use static scaling factor (e.g., 512) or dynamic scaling with large init value

### 4.2 Dynamic Loss Scaling Algorithm

**Why Dynamic Scaling Matters**:
```
Fixed scale (e.g., 1024):
- Early training: gradient_unscaled = 1e-6, scaled = 1e-3 ✓ Good
- Mid training: gradient_unscaled = 1e-3, scaled = 1 ✓ Good
- Late training: gradient_unscaled = 1e-2, scaled = 10 ✗ Overflow!

Dynamic scaling adapts to training phase automatically.
```

**PyTorch GradScaler Algorithm**:

```
Initialize: loss_scale = 2^15 = 32768, scale_growth_interval = 2000

For each step:
  1. Forward pass with autocast (FP16/BF16)
  2. Scale loss: loss_scaled = loss × loss_scale
  3. Backward pass: compute gradients in scaled loss space
  4. Unscale gradients: grad /= loss_scale
  
  5. Check for inf/nan in gradients:
     if contains_inf_or_nan(gradients):
       loss_scale = loss_scale / backoff_factor  # Usually 0.5
       skip weight update
       continue
     
  6. Clip gradients (optional): max_norm = 1.0
  
  7. Optimizer step: w = w - lr * grad
  
  8. Every scale_growth_interval steps without overflow:
     loss_scale = loss_scale × growth_factor  # Usually 2.0
```

**Pseudo-code (PyTorch)**:

```python
from torch.cuda.amp import GradScaler

scaler = GradScaler(
    init_scale=2**15,           # 32768
    growth_factor=2.0,          # Double on success
    backoff_factor=0.5,         # Halve on overflow
    growth_interval=2000        # Steps before growth
)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward with autocast
        with autocast():
            outputs = model(batch)
            loss = criterion(outputs, labels)
        
        # Backward with scaling
        scaler.scale(loss).backward()
        
        # Unscale before clipping/stepping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()  # Adjust loss_scale for next iteration
```

**Key Hyperparameters**:

| Parameter | Default | Typical Range | Impact |
|-----------|---------|---------------|--------|
| init_scale | 2^15 | 2^10 to 2^20 | Higher = more aggressive scaling |
| growth_factor | 2.0 | 1.5-2.0 | Faster growth = quicker scale adaptation |
| backoff_factor | 0.5 | 0.25-0.75 | Smaller = more conservative on overflow |
| growth_interval | 2000 | 1000-5000 | Fewer steps = more frequent scaling |

**Tuning Guide**:
```
If NaN/Inf occurs frequently:
  - Decrease init_scale (start smaller)
  - Increase backoff_factor (more conservative)
  - Decrease growth_interval (check more often)

If training is slow to adapt:
  - Increase growth_factor
  - Increase growth_interval
  - Increase init_scale (if model is stable)
```

### 4.3 Static Loss Scaling

**When to Use Static Scaling**:
- Very stable training (e.g., fine-tuning)
- Distributed training (synchronizing dynamic scale across devices is complex)
- Inference or evaluation (fixed overhead is fine)

**Selection Strategy**:

```
1. Profile your model:
   - Typical gradient magnitude: 1e-4 to 1e-6
   - Largest gradient (outliers): 1e-2 to 1e-1
   - Smallest gradient: 1e-8 to 1e-10

2. For BF16:
   - Minimum safe scale = 2^10 (1024)
   - Recommended scale = 2^11 to 2^15
   - Start with 2^12 (4096) for typical models

3. For FP16:
   - Minimum safe scale = 2^15 (32768)
   - Recommended scale = 2^16 to 2^24
   - Start with 2^16 (65536) for typical models

4. For FP8:
   - Must use dynamic scaling or per-tensor scaling
   - Static scaling rarely sufficient alone
```

**Formula**:
```
loss_scale = 2^k where:
  k = ceil(log2(max_representable / max_expected_gradient))
  
Example: Max gradient expected = 100
  k = ceil(log2(1e6 / 100)) = ceil(13.3) = 14
  loss_scale = 2^14 = 16384
```

### 4.4 Gradient Overflow Handling

**Detection and Recovery**:

```python
# Check for inf/nan
has_inf_nan = torch.isinf(loss).any() or torch.isnan(loss).any()

if has_inf_nan:
    # Option 1: Skip step (most common)
    print(f"Overflow detected at step {step}, skipping")
    scaler.update()  # Reduces scale automatically
    continue
    
# Option 2: Gradient clipping (preventive)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Option 3: Reduce learning rate
if overflow_count > 10:
    learning_rate *= 0.9
```

**Best Practices**:

1. **Always monitor overflow frequency**:
   ```python
   overflow_count = 0
   for step in range(num_steps):
       ...
       if contains_inf_nan(gradients):
           overflow_count += 1
   
   if overflow_count / num_steps > 0.05:  # More than 5%
       print("Warning: High overflow rate, check hyperparams")
   ```

2. **Log scale value over training**:
   ```python
   # Should grow from init_scale, but not explode
   if step % 100 == 0:
       print(f"Step {step}: loss_scale={scaler.get_scale():.0f}")
   ```

3. **Gradient clipping is orthogonal**:
   - Loss scaling handles numerical underflow
   - Gradient clipping handles optimization instability
   - Use both together for robustness

---

## 5. Master Weights: Maintaining High-Precision Copies

### 5.1 The Master Weights Concept

**Problem**: Accumulating rounding errors with repeated updates in low precision

```
FP16 example (exaggerated):
w_t = 1.0000
grad = -0.0001  (small update)
w_{t+1} = 1.0000 - 0.0001 = 1.0000  (rounded back due to FP16 precision)
After many steps: weight doesn't change!

With master weights in FP32:
w_master_t = 1.0000000000
w_compute_t = float16(1.0000000000) = 1.0000
grad = -0.0001
w_master_{t+1} = 1.0000000000 - 0.0001 = 0.9999000000  (accumulates!)
w_compute_{t+1} = float16(0.9999000000)
```

### 5.2 Implementation Architecture

**Typical Mixed Precision Setup**:

```
┌─────────────────────────────────────────┐
│         Training Step                    │
└─────────────────────────────────────────┘
            │
            ├─ Copy FP32 weights → FP16 compute weights
            │  (or use computed weights directly)
            │
            ├─ Forward pass: FP16 activations
            │  (with occasional FP32 for specific ops)
            │
            ├─ Compute loss: FP32 or scaled FP16
            │
            ├─ Backward pass: Gradient computation
            │  All gradients in FP32 (accumulated)
            │
            └─ Update step:
               ├─ Unscale gradients (if using loss scaling)
               ├─ Clip gradients (optional)
               ├─ Compute updates in FP32
               └─ Copy updated weights back to FP32 master
                  (Compute copy updates automatically)
```

**PyTorch Code Structure**:

```python
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        
        # Keep FP32 copy of parameters
        self.master_params = [p.clone().detach().float() 
                              for p in model.parameters()]
    
    def train_step(self, batch):
        # Ensure compute parameters are FP16
        for param, master in zip(self.model.parameters(), 
                                 self.master_params):
            param.data.copy_(master.half())
        
        # Forward + backward in FP16
        with torch.cuda.amp.autocast():
            loss = self.forward_backward(batch)
        
        # Gradients are in FP32 (accumulated from FP16 ops)
        # Update master params in FP32
        with torch.no_grad():
            for param, master in zip(self.model.parameters(),
                                     self.master_params):
                master.sub_(param.grad * learning_rate)
```

### 5.3 Memory Efficiency Trade-offs

**Storage Breakdown** (for 7B parameter model):

```
Single Precision (FP32 only):
  Parameters:    7B × 4 bytes = 28 GB
  Gradients:     7B × 4 bytes = 28 GB
  Optimizer State: 7B × 8 bytes = 56 GB (momentum + variance)
  Activations:   Batch-dependent, ~40-60 GB
  ────────────────────────────────
  Total:         ~160-180 GB

Mixed Precision (BF16 compute + FP32 master):
  Parameters (compute):     7B × 2 bytes = 14 GB
  Parameters (master):      7B × 4 bytes = 28 GB
  Gradients (FP32):         7B × 4 bytes = 28 GB
  Optimizer State:          7B × 8 bytes = 56 GB
  Activations (BF16):       Batch-dependent, ~20-30 GB
  ────────────────────────────────
  Total:         ~150-160 GB  (≈10-20% savings)
  
  vs FP32 only: Not as dramatic when master weights kept!
```

**Key Realization**: Master weights don't add much overhead if stored efficiently:
- Stored in CPU memory or as checkpoint (not during computation)
- Or kept in GPU memory only when needed

**Strategy for Large Models**:

```python
# Option 1: Keep master weights only in GPU during training
# (most common for mixed precision)
model = model.half()  # Compute in FP16
master_weights = [p.clone().float() for p in model.parameters()]

# Option 2: For very large models, use PyTorch FSDP
# (Fully Sharded Data Parallel)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(
    model,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,      # Compute
        reduce_dtype=torch.float32,      # Communication
        buffer_dtype=torch.float32       # Buffers
    )
)

# Option 3: DeepSpeed ZeRO
# Master weights distributed across all GPUs
from deepspeed import initialize
model, optimizer, _, _ = initialize(
    model=model,
    config=ds_config  # Enable ZeRO stages
)
```

### 5.4 Convergence Guarantees with Master Weights

**Theorem** (Informal):
> If weights are updated in FP32 (master weights) and gradients are computed with sufficient precision (FP32 accumulation), then training convergence is theoretically equivalent to full FP32 training.

**Practical Guarantee**:
```
Assumptions:
1. Gradients accumulated in FP32
2. Weight updates in FP32 (master weights)
3. Loss scaling prevents gradient underflow
4. Learning rate ≤ learning_rate_fp32

Then: loss convergence rate ≈ FP32 training
      (with possible minor variation from forward precision)
```

**Empirical Evidence** (From Meta, NVIDIA, DeepL):
- Llama 2 7B-70B: BF16 + master weights achieves identical final accuracy
- GPT-3 training: Reported no convergence issues with mixed precision
- BERT/RoBERTa: Fine-tuning in BF16 matches FP32 exactly

---

## 6. Implementation: Framework-Specific Approaches

### 6.1 PyTorch Native: torch.autocast() and torch.cuda.amp

**Modern Recommended Approach** (PyTorch 1.6+)

#### Basic Usage with autocast

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Create model and optimizer
model = MyTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create gradient scaler
scaler = GradScaler()

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        with autocast():  # Default: BF16 on newer GPUs, FP16 on older
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        
        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step with scaling
        scaler.step(optimizer)
        scaler.update()
        
        if batch_idx % 100 == 0:
            print(f"Loss: {loss.item():.4f}, Scale: {scaler.get_scale():.0f}")

# Save model in FP32 for inference
torch.save(model.float().state_dict(), 'model.pt')
```

#### Controlling Precision Strategy

```python
# Specify target device and dtype
# Available: 'cuda', 'cpu', 'auto'
with autocast(device_type='cuda', dtype=torch.float16):
    # Uses FP16 for supported ops
    pass

with autocast(device_type='cuda', dtype=torch.bfloat16):
    # Uses BF16 for supported ops (recommended for modern GPUs)
    pass

# Disable casting for specific ops
with autocast():
    x = model(inputs)  # Autocast applied
    with autocast(enabled=False):
        y = some_precision_critical_op(x)  # FP32
```

#### Custom Cast Policy

```python
# Fine-grained control over which ops use mixed precision
from torch.cuda.amp import autocast_mode

class CustomAutocast(autocast_mode.autocast):
    def __enter__(self):
        # Custom logic for specific layers
        if isinstance(self.module, TransformerBlock):
            # Cast to FP16 except attention
            torch.amp.autocast_mode.enter(dtype=torch.float16)
        return self

# Or use amp.autocast_mode with custom policies
torch.amp.autocast_mode.set_autocast_gpu_dtype(torch.bfloat16)
torch.amp.autocast_mode.set_autocast_enabled(True)
```

### 6.2 NVIDIA APEX: Advanced Mixed Precision

**When to Use**: Legacy code, specific fine-grained control, distributed training

```bash
pip install apex
# Or build from source for latest features
git clone https://github.com/NVIDIA/apex
cd apex && pip install -v --no-cache-dir ./
```

#### APEX Simplified API (O1)

```python
from apex import amp

# Simple one-liner
model, optimizer = amp.initialize(
    models=model,
    optimizers=optimizer,
    opt_level="O1"  # Mixed precision with dynamic loss scaling
)

# Training loop unchanged, just use with amp.scale_loss
for inputs, targets in train_loader:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
```

#### APEX Optimization Levels

```
O0 (FP32):
  - All operations in FP32
  - Baseline, no speedup

O1 (Mixed):
  - Recommended for most users
  - Automatically selects which ops use FP16
  - Dynamic loss scaling
  - ~3x speedup possible

O2 (Almost FP16):
  - Most ops in FP16, master weights in FP32
  - Requires batch norm in FP32
  - Even higher speedup (4x possible)

O3 (Full FP16):
  - Everything in FP16 (advanced users only)
  - Potential convergence issues
  - Not recommended
```

#### APEX Master Weights Control

```python
model, optimizer = amp.initialize(
    models=model,
    optimizers=optimizer,
    opt_level="O2",
    master_weights=True,  # Explicitly enable master weights
    loss_scale='dynamic'   # Use dynamic scaling
)

# Check master weights
for i, param_group in enumerate(optimizer.param_groups):
    print(f"Param group {i}: {len(param_group['params'])} params")
    for p in param_group['params']:
        print(f"  Compute dtype: {p.dtype}")
        # Master weights stored internally in optimizer
```

### 6.3 HuggingFace Transformers: Integration

**Simplest for LLMs**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

training_args = TrainingArguments(
    output_dir="./results",
    
    # Mixed precision settings
    fp16=True,                          # Use FP16 (or BF16 below)
    bf16=False,                         # Use BF16 (better for modern GPUs)
    fp16_opt_level="O1",                # APEX optimization level
    half_precision_backend="auto",      # Detect automatically
    
    # Loss scaling
    fp16_full_eval=False,               # FP16 in eval? Usually no
    
    # Other training args
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

#### Configuration via Config File

```json
{
  "learning_rate": 2e-5,
  "num_train_epochs": 3,
  
  "fp16": true,
  "fp16_opt_level": "O1",
  "fp16_backend": "cuda_amp",
  
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  
  "warmup_steps": 500,
  "max_steps": 10000,
  
  "logging_steps": 100,
  "save_steps": 1000,
  "eval_steps": 500
}
```

### 6.4 DeepSpeed: Distributed Mixed Precision Training

**For Large-Scale Training** (multi-GPU/multi-node)

```bash
pip install deepspeed
# or from source for latest features
```

#### DeepSpeed Config for Mixed Precision

```json
{
  "train_batch_size": 128,
  "gradient_accumulation_steps": 4,
  
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,           # 0 = dynamic loss scaling
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 16  # 2^16 = 65536
  },
  
  "zero_optimization": {
    "stage": 2,                      # ZeRO-2: Optimizer state sharding
    "offload_optimizer": {
      "device": "cpu",              # Store on CPU when possible
      "pin_memory": true
    }
  }
}
```

#### Training Loop

```python
import deepspeed
from torch.utils.data import DataLoader

# Initialize deepspeed
model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
    args=cmd_args,
    model=model,
    model_parameters=model.parameters(),
    training_data=train_dataset
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        inputs, targets = batch
        
        # Forward
        outputs = model_engine(inputs)
        loss = criterion(outputs, targets)
        
        # Backward (automatically handles loss scaling)
        model_engine.backward(loss)
        
        # Step
        model_engine.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")

# Save checkpoint
model_engine.save_checkpoint(output_dir="checkpoints")
```

#### Loading Checkpoints

```python
# Save
model_engine.save_checkpoint(
    save_dir="./checkpoints",
    tag="epoch3"
)

# Load
model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
    args=cmd_args,
    model=model,
    model_parameters=model.parameters(),
    training_data=train_dataset
)

ckpt_path = model_engine.load_checkpoint(
    load_dir="./checkpoints",
    tag="epoch3"
)

if ckpt_path:
    print(f"Loaded checkpoint from {ckpt_path}")
```

### 6.5 NVIDIA NeMo: Production-Ready LLM Training

**Latest (2026): Supports FP8, MXFP8, NVFP4**

```bash
pip install nvidia-nemo-toolkit
```

#### Low-Precision Training with NeMo

```python
from megatron.bridge.recipes.llama import (
    llama3_8b_low_precision_pretrain_config
)
from megatron.bridge.training.gpt_step import forward_step

# Configure for NVFP4 (latest format)
precision = "bf16_with_nvfp4_mixed"  # or mxfp8, fp8_current_scaling

cfg = llama3_8b_low_precision_pretrain_config(
    mixed_precision_recipe=precision,
    train_iters=100000,
    lr_warmup_iters=1000,
    lr_decay_iters=99000,
    
    # Training config
    global_batch_size=768,
    micro_batch_size=2,  # Per GPU per forward step
    seq_length=8192,
    
    # Learning rate schedule
    lr=6e-4,
    lr_decay_style="cosine",
    min_lr=6e-6,
    
    # Optimizer settings
    optimizer='AdamW',
    adam_eps=1e-8,
)

# Train
pretrain(config=cfg, forward_step_func=forward_step)
```

#### Switching Between Precisions

```python
# Same code, just change precision string!
for precision in [
    "bf16",                           # Baseline
    "bf16_with_fp8_current_scaling_mixed",
    "bf16_with_mxfp8_mixed",
    "bf16_with_nvfp4_mixed"
]:
    cfg = llama3_8b_low_precision_pretrain_config(
        mixed_precision_recipe=precision
    )
    pretrain(config=cfg, forward_step_func=forward_step)
    # Compare results!
```

---

## 7. Advanced Topics

### 7.1 Per-Layer Precision Selection

**Motivation**: Different layers have different numerical requirements

```
Empirical observation:
  - Early layers: Robust to FP16, can use lower precision
  - Middle layers: Standard BF16/FP16 works well
  - Late layers: More sensitive, may need FP32 or BF16
  - Attention layers: Can tolerate FP16
  - Feed-forward: Sensitive, prefer BF16
```

**Implementation in PyTorch**:

```python
class PrecisionControlledTransformer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(layer_idx=i) 
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            # First 2 and last 4 layers in FP32
            if layer_idx < 2 or layer_idx >= len(self.layers) - 4:
                with torch.autocast(enabled=False):
                    x = layer(x.float()).half()
            else:
                # Middle layers in BF16
                with torch.autocast(dtype=torch.bfloat16):
                    x = layer(x)
        return x
```

**NVFP4 Example from DeepL/NVIDIA**:

```python
# Keep final 4 transformer layers in BF16
# Everything else in FP4
for i, block in enumerate(model.transformer.blocks):
    if i >= len(model.transformer.blocks) - 4:
        # Final 4 blocks: BF16
        for param in block.parameters():
            param.data = param.data.bfloat16()
    else:
        # Other blocks: FP4 (with hierarchical scaling)
        apply_fp4_scaling(block)

# Result: 1.59x speedup vs BF16
# Convergence: Identical to BF16 on Llama 3 8B
```

### 7.2 Attention-Specific Precision

**Key Insight**: Attention softmax is sensitive to precision

```python
import torch
import torch.nn.functional as F

class MixedPrecisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Project: typically use input dtype (FP16/BF16)
        Q = self.q_proj(x)  # batch_size, seq_len, hidden_size
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # batch_size, num_heads, seq_len, head_dim
        
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)
        
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)
        
        # Scores: cast to higher precision for numerical stability
        # Key: softmax is sensitive to precision!
        Q_scaled = Q / (self.head_dim ** 0.5)
        
        # Cast to FP32 for softmax computation (crucial!)
        with torch.autocast(enabled=False):
            scores = torch.matmul(Q_scaled.float(), K.float().transpose(-2, -1))
            # scores: batch_size, num_heads, seq_len, seq_len
            
            # Apply mask if needed
            if hasattr(self, 'mask'):
                scores = scores + self.mask
            
            # Softmax in FP32
            attn_weights = F.softmax(scores / (self.head_dim ** 0.5), dim=-1)
        
        # Convert back to compute dtype for matrix multiplication
        attn_weights = attn_weights.to(V.dtype)
        
        # Context: back to compute dtype
        context = torch.matmul(attn_weights, V)  # batch_size, num_heads, seq_len, head_dim
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        
        return output
```

**PyTorch's Native Solution**:

```python
# PyTorch automatically uses higher precision for softmax
# in autocast context (PyTorch 1.10+)
with torch.autocast(device_type='cuda'):
    attn = F.softmax(scores, dim=-1)  # Computed in FP32 automatically
```

### 7.3 Custom Mixed Precision in Optimizers

**Gradient Accumulation with Different Precision**:

```python
class MixedPrecisionOptimizer:
    def __init__(self, optimizer, loss_scaler):
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
    
    def backward_step(self, loss):
        # Scale loss
        scaled_loss = loss * self.loss_scaler.scale_factor
        scaled_loss.backward()
    
    def step(self):
        # Unscale gradients
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    p.grad /= self.loss_scaler.scale_factor
        
        # Check for inf/nan
        has_nan = False
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None and (
                    torch.isnan(p.grad).any() or 
                    torch.isinf(p.grad).any()
                ):
                    has_nan = True
                    break
        
        if has_nan:
            self.loss_scaler.scale_factor *= 0.5
            return False  # Skip update
        
        # Gradient clipping
        total_norm = 0
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    total_norm += p.grad.norm() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > 1.0:
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    if p.grad is not None:
                        p.grad /= total_norm
        
        # Optimizer step
        self.optimizer.step()
        
        # Update scale factor
        self.loss_scaler.scale_factor *= 1.001
        
        return True
```

### 7.4 Debugging NaN/Inf Issues

**Step 1: Identify Where NaN Occurs**

```python
def debug_nan_origin(model, batch):
    # Check forward pass
    with torch.autocast(enabled=False):
        inputs = batch['input_ids'].cuda()
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                hooks_list = []
                
                def make_hook(layer_name):
                    def hook(m, input, output):
                        if isinstance(output, torch.Tensor):
                            if torch.isnan(output).any():
                                print(f"NaN detected in {layer_name}")
                                print(f"  Output shape: {output.shape}")
                                print(f"  Input dtype: {input[0].dtype}")
                                print(f"  Output dtype: {output.dtype}")
                        return output
                    return hook
                
                hooks_list.append(module.register_forward_hook(
                    make_hook(name)
                ))
        
        try:
            outputs = model(inputs)
            loss = outputs.loss
            loss.backward()
        finally:
            for hook in hooks_list:
                hook.remove()
```

**Step 2: Check Loss Scale**

```python
# Verify loss scaling is appropriate
print(f"Loss magnitude: {loss.item():.4f}")
print(f"Gradient scale: {scaler.get_scale():.0f}")
print(f"Scaled loss: {(loss * scaler.get_scale()).item():.2f}")

# Expected: scaled loss in reasonable range (100-10000)
if loss.item() * scaler.get_scale() < 10:
    print("Warning: Scaled loss too small, increase init_scale")
if loss.item() * scaler.get_scale() > 1e6:
    print("Warning: Scaled loss too large, decrease init_scale")
```

**Step 3: Isolate Precision Issue**

```python
# Test with full FP32
model = model.float()
inputs = inputs.float()

with torch.no_grad():
    outputs_fp32 = model(inputs)

print(f"FP32 loss: {outputs_fp32.loss.item()}")
print(f"FP32 output range: [{outputs_fp32.logits.min()}, {outputs_fp32.logits.max()}]")

# Compare with mixed precision
model = model.half()
inputs = inputs.half()

with torch.autocast(enabled=False):  # Force FP16
    with torch.no_grad():
        outputs_fp16 = model(inputs)

print(f"FP16 loss: {outputs_fp16.loss.item()}")
```

**Step 4: Common Causes and Fixes**

```
Cause: Batch norm in FP16 (numerically unstable)
Fix:   Use layer norm instead, or keep batch norm in FP32
       with torch.autocast(enabled=False):
           x = batch_norm_layer(x.float()).half()

Cause: Very small learning rate with large loss scale
Fix:   Reduce learning rate or loss scale init
       optimizer = Adam(lr=1e-5)  # Lower
       scaler = GradScaler(init_scale=2**14)  # Smaller

Cause: Outlier gradients in attention/embedding
Fix:   Use gradient clipping
       torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

Cause: Loss becomes infinity (overflow)
Fix:   Reduce initial loss scale
       scaler = GradScaler(init_scale=2**12)

Cause: Gradients become 0 (underflow)
Fix:   Increase loss scale
       scaler = GradScaler(init_scale=2**16)
```

---

## 8. Code Examples

### 8.1 Complete Basic Training Loop with Autocast

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Set up model, data, and training config
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50000
    hidden_dim = 768
    num_layers = 12
    batch_size = 32
    num_epochs = 3
    learning_rate = 5e-5
    
    # Model, optimizer, scaler
    model = SimpleTransformer(vocab_size, hidden_dim, num_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    # Dummy data
    num_samples = 1000
    seq_length = 512
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_input_ids, batch_labels) in enumerate(dataloader):
            batch_input_ids = batch_input_ids.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                outputs = model(batch_input_ids, batch_labels)
                loss = outputs["loss"]
            
            # Backward with scaling
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                scale = scaler.get_scale()
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {avg_loss:.4f}, Scale: {scale:.0f}")
        
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss/num_batches:.4f}\n")
    
    # Save model in FP32
    torch.save(model.state_dict(), "model_mixed_precision.pt")
    print("Model saved!")

if __name__ == "__main__":
    main()
```

### 8.2 With Gradient Accumulation

```python
def train_with_gradient_accumulation():
    # Configuration
    device = torch.device("cuda")
    model = SimpleTransformer(50000, 768, 12).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()
    
    # Gradient accumulation settings
    accumulation_steps = 4
    effective_batch_size = 32 * accumulation_steps  # 128
    
    dataloader = get_dataloader(batch_size=32)
    
    for epoch in range(3):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward with autocast
            with autocast():
                outputs = model(inputs, labels)
                loss = outputs["loss"]
                
                # Scale loss for accumulation
                loss = loss / accumulation_steps
            
            # Accumulate gradients
            scaler.scale(loss).backward()
            
            # Update after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                print(f"Update at batch {batch_idx+1}, "
                      f"Effective batch size: {effective_batch_size}")
```

### 8.3 Distributed Training with DDP

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def train_distributed():
    rank, world_size = setup_distributed()
    
    # Model on local GPU
    model = SimpleTransformer(50000, 768, 12).cuda()
    model = DDP(model, device_ids=[rank])
    
    # Distributed sampler
    dataset = TensorDataset(input_ids, labels)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Optimizer and scaler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()
    
    for epoch in range(3):
        sampler.set_epoch(epoch)  # For reproducibility
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs, labels)
                loss = outputs["loss"]
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if rank == 0 and (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    # Save only on rank 0
    if rank == 0:
        torch.save(model.module.state_dict(), "model_ddp.pt")
    
    dist.destroy_process_group()

# Run with: torchrun --nproc_per_node=4 script.py
if __name__ == "__main__":
    train_distributed()
```

### 8.4 Optimizer Integration Examples

#### AdamW with Mixed Precision

```python
from torch.optim import AdamW

# Standard setup
model = SimpleTransformer(50000, 768, 12).cuda()
optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
scaler = GradScaler()

# Training loop handles precision automatically
```

#### SGD with Mixed Precision

```python
from torch.optim import SGD

optimizer = SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)
scaler = GradScaler()

# Scaler works with any optimizer
```

#### Custom Optimizer with FP32 Master Weights

```python
class MixedPrecisionSGD(torch.optim.SGD):
    def __init__(self, params, lr=0.01, momentum=0):
        super().__init__(params, lr=lr, momentum=momentum)
        
        # Create FP32 master weights
        self.master_params = [
            p.clone().detach().float() 
            for p in self.param_groups[0]['params']
        ]
    
    def step(self, closure=None):
        # Convert master weights to compute precision (FP16/BF16)
        for fp16_param, fp32_master in zip(
            self.param_groups[0]['params'],
            self.master_params
        ):
            fp16_param.data.copy_(fp32_master.half())
        
        # Standard SGD step
        super().step(closure)
        
        # Update master weights
        with torch.no_grad():
            for fp16_param, fp32_master in zip(
                self.param_groups[0]['params'],
                self.master_params
            ):
                fp32_master.sub_(
                    fp16_param.grad * self.param_groups[0]['lr']
                )
                fp16_param.grad.zero_()
```

### 8.5 Loss Scaling Demonstration

```python
def demonstrate_loss_scaling():
    import numpy as np
    
    # Simulate gradient underflow in FP16
    print("=" * 60)
    print("Gradient Underflow Demonstration")
    print("=" * 60)
    
    # Typical small gradient
    true_gradient = np.float32(1e-6)
    
    # Without scaling - FP16
    fp16_tensor = torch.tensor(true_gradient, dtype=torch.float16)
    print(f"\nWithout scaling (FP16):")
    print(f"  True gradient:   {true_gradient}")
    print(f"  FP16 gradient:   {float(fp16_tensor)}")
    print(f"  Lost:            {true_gradient == float(fp16_tensor)}")
    
    # With scaling - FP16 with scale 1024
    loss_scale = 1024
    scaled_gradient = torch.tensor(true_gradient * loss_scale, dtype=torch.float16)
    unscaled = float(scaled_gradient) / loss_scale
    
    print(f"\nWith scaling (FP16, scale={loss_scale}):")
    print(f"  True gradient:      {true_gradient}")
    print(f"  Scaled (FP16):      {float(scaled_gradient)}")
    print(f"  Unscaled:           {unscaled}")
    print(f"  Preserved:          {abs(unscaled - true_gradient) < 1e-8}")
    
    # BF16 naturally has wider range
    bf16_tensor = torch.tensor(true_gradient, dtype=torch.bfloat16)
    print(f"\nBF16 (no scaling needed):")
    print(f"  True gradient: {true_gradient}")
    print(f"  BF16 gradient: {float(bf16_tensor)}")
    print(f"  Preserved:     {float(bf16_tensor) > 0}")

if __name__ == "__main__":
    demonstrate_loss_scaling()
```

---

## 9. Empirical Analysis: Benchmarks and Trade-offs

### 9.1 Memory Usage Comparison (7B Parameter Model)

**Measurement Conditions**:
- Model: Llama 2 7B
- Batch size: 2 sequences of 8192 tokens
- Optimizer: AdamW (momentum + variance)
- Hardware: NVIDIA H100 (80 GB)

| Component | FP32 | BF16 + Master | FP8 | NVFP4 |
|-----------|------|---------------|-----|-------|
| **Weights (Compute)** | 28 GB | 14 GB | 7 GB | 1.75 GB |
| **Weights (Master)** | — | 28 GB | 28 GB | 28 GB |
| **Gradients** | 28 GB | 28 GB | 28 GB | 28 GB |
| **Optimizer State** | 56 GB | 56 GB | 56 GB | 56 GB |
| **Activations** | ~50 GB | ~25 GB | ~20 GB | ~15 GB |
| **Scaling Factors** | — | — | ~200 MB | ~500 MB |
| **Total** | ~162 GB | ~151 GB | ~140 GB | ~130 GB |
| **Reduction vs FP32** | — | 7% | 14% | 20% |

**Key Finding**: Master weights don't add much overhead if not stored separately from optimizer state!

### 9.2 Training Speed Comparison

**Setup**:
- Model: Llama 3 8B with 1 trillion token training
- Hardware: NVIDIA GB200 NVL72 cluster
- Effective batch size: 128 (gradient accumulation)
- Sequence length: 8192

| Precision | Micro-batch | Throughput (TFLOP/s) | Speedup | Tokens/sec |
|-----------|-------------|----------------------|---------|-----------|
| **BF16** | 2 | 1165 | 1.0x | 45,000 |
| **FP8-CS** | 2 | 1547 | 1.33x | 60,000 |
| **MXFP8** | 2 | 1540 | 1.32x | 59,500 |
| **NVFP4** | 4 | 1850 | 1.59x | 72,000 |

**Time Savings** (to train 1T tokens):
- BF16: ~22,000 GPU hours
- FP8-CS: ~16,500 GPU hours (25% faster)
- NVFP4: ~14,000 GPU hours (36% faster)

### 9.3 Convergence Comparison (Validation Loss)

**Experiment**: Training Llama 3 8B on 1 trillion tokens

```
Step  | BF16  | FP8-CS | MXFP8 | NVFP4
------|-------|--------|-------|------
10K   | 2.87  | 2.88   | 2.87  | 2.89
100K  | 2.34  | 2.34   | 2.34  | 2.35
1M    | 1.89  | 1.89   | 1.89  | 1.92
1B    | 1.45  | 1.45   | 1.45  | 1.47

Final accuracy (MMLU):
BF16:  45.98%
FP8:   46.00%
MXFP8: 46.56%
NVFP4: 45.64%
```

**Observation**: All formats converge nearly identically!

### 9.4 Final Model Quality (Downstream Tasks)

**Benchmark Results** (% accuracy):

| Task | BF16 | FP8-CS | MXFP8 | NVFP4 | Llama 3 Published |
|------|------|--------|-------|-------|------------------|
| **MMLU** | 45.98 | 46.00 | 46.56 | 45.64 | 46.0 |
| **HellaSwag** | 76.44 | 75.25 | 75.46 | 75.59 | 76.4 |
| **WinoGrande** | 70.17 | 70.24 | 71.27 | 69.38 | 70.2 |
| **ARC-C** | 51.28 | 49.91 | 51.11 | 51.28 | 51.3 |

**Key Finding**: Slight differences in some tasks, but all formats produce production-quality models.

### 9.5 Trade-off Analysis

**When to Use Each Format**:

```
Format    | Speedup | Memory Saving | Complexity | Use Case
----------|---------|---------------|------------|------------------
FP32      | 1.0x    | —             | Low        | Baselines, debugging
TF32      | 4.0x    | 0%            | None       | Quick win (free)
BF16      | 2.0x    | 50%           | Low        | Standard choice
FP16      | 2.0x    | 50%           | High       | Legacy hardware
FP8-CS    | 1.33x   | 72%           | Medium     | Modern GPUs, cost-sensitive
MXFP8     | 1.32x   | 72%           | Medium     | Blackwell optimization
NVFP4     | 1.59x   | 75%           | High       | Ultra-large models, cost-critical
```

**Decision Tree**:

```
Do you have modern GPU (A100+)?
├─ YES: Can you modify code?
│   ├─ NO: Use TF32 (automatic)
│   └─ YES: Want simplicity?
│       ├─ YES: Use BF16
│       └─ NO (want max speedup): Use FP8 or NVFP4
│
└─ NO (older GPU): Use FP16 with careful loss scaling
```

---

## 10. Research References and Citations

### Key Papers

1. **Mixed Precision Training (NVIDIA, 2017)**
   - Micikevicius et al., "Mixed Precision Training"
   - https://arxiv.org/abs/1710.03740
   - Foundational work on loss scaling and mixed precision

2. **Automatic Mixed Precision (NVIDIA, 2019)**
   - Carilli & Ruberry, "Automatic Mixed Precision in PyTorch"
   - Developer.nvidia.com/blog (GTC 2019)
   - Introduces torch.cuda.amp and dynamic loss scaling

3. **BF16 and Training Efficiency (Google/NVIDIA, 2021)**
   - Kalamkar et al., "A Study of Bfloat16 for Deep Learning Training"
   - https://arxiv.org/abs/2106.06305
   - BF16 advantages for modern hardware

4. **FP8 Training (NVIDIA, 2024)**
   - Vadimivovskiy et al., "Floating-Point 8: 8-Bit Quantization for Training"
   - https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/
   - FP8-CS and MXFP8 techniques

5. **NVFP4 Training (DeepL/NVIDIA, 2025)**
   - "Using NVFP4 Low-Precision Model Training for Higher Throughput"
   - https://developer.nvidia.com/blog/using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy/
   - Production results on Llama 3 training

6. **FP8 Training at Scale (NeurIPS 2025)**
   - Hernández-Cano et al., "Towards Fully FP8 GEMM LLM Training at Scale"
   - https://openreview.net/forum?id=KYTFXxTJ12
   - Practical FP8 training infrastructure

7. **TWEO: Transformers Without Extreme Outliers**
   - Liang et al., "TWEO: Transformers Without Extreme Outliers"
   - https://arxiv.org/abs/2511.23225
   - Addresses FP8 numerical stability issues

8. **μUnit Scaling (ICML 2025)**
   - Narayan et al., "μnit Scaling: Simple and Scalable FP8 LLM Training"
   - https://proceedings.mlr.press/v267/narayan25b.html
   - Simplified FP8 scaling strategy

### Framework Documentation

- **PyTorch Autocast**: https://pytorch.org/docs/stable/amp.html
- **DeepSpeed FP16**: https://www.deepspeed.ai/training/mixed-precision/
- **NVIDIA APEX**: https://github.com/NVIDIA/apex
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/training
- **NVIDIA NeMo**: https://github.com/NVIDIA-NeMo/Megatron-Bridge

### Practical Guides

- **NVIDIA Mixed Precision Training Guide**: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/
- **PyTorch AMP Examples**: https://pytorch.org/docs/stable/notes/amp_examples.html
- **DeepSpeed Training Guide**: https://www.deepspeed.ai/training/

### Additional Resources

- **NVIDIA GTC Sessions**: https://www.nvidia.com/en-us/on-demand/ (search "mixed precision")
- **Hugging Face Blog**: https://huggingface.co/blog/ (search "mixed precision", "fp8")
- **DeepL Blog**: https://www.deepl.com/en/blog/ (FP8 training posts 2025)

---

## Summary

Mixed-precision training is now standard practice for large-scale model training. Key takeaways:

1. **Use BF16 as default** for modern GPUs (A100, H100, Blackwell)
2. **Loss scaling is critical** for gradients < 1e-5 (automatic with PyTorch autocast)
3. **Master weights** ensure convergence (minimal overhead with modern frameworks)
4. **FP8/NVFP4** for production systems requiring maximum efficiency (30-50% faster than BF16)
5. **Per-layer precision** is emerging technique for ultra-large models

Implementation is straightforward with modern frameworks—just add `with autocast():` and a `GradScaler()` to your training loop!

