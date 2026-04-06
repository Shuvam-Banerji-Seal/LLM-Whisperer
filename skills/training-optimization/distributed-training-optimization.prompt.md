# Distributed Training Optimization for Large Language Models

## 1. Introduction: Challenges of Distributed LLM Training

### Why Distributed Training?

Training modern large language models (LLMs) with billions to trillions of parameters on a single GPU is impossible due to:

1. **Memory Constraints**: A 70B parameter model with Adam optimizer states requires:
   - Model weights: 140 GB (fp32, 70B × 4 bytes)
   - Gradients: 140 GB
   - Optimizer states (Adam): 280 GB (momentum + variance)
   - Activations: varies by batch size
   - **Total: ~560+ GB** without activations

2. **Computation Time**: Single GPU training would take months/years:
   - 70B model: ~500 TFLOPS on H100, ~280 TFLOPS on A100
   - 1 trillion tokens at 2T FLOPs/token = 2 × 10^21 FLOPs
   - Single GPU: ~10,000 years

3. **Bandwidth Bottlenecks**: Communication between devices becomes critical:
   - All-reduce operations: O(log N) rounds in tree-based, O(2N-2) rounds in ring
   - Network bandwidth: 400 Gbps (NVIDIA HGX) vs GPU compute (1,458 TFLOPS)
   - Compute-communication ratio: must balance carefully

### Core Optimization Goals

- **Memory Efficiency**: Reduce per-GPU memory footprint by 4-16x
- **Communication Efficiency**: Minimize bandwidth and latency overhead
- **Compute Utilization**: Maintain high GPU utilization (80%+ target)
- **Scaling Efficiency**: Achieve near-linear scaling with GPU count
- **Code Maintainability**: Minimal code changes for parallelism

---

## 2. Distributed Parallelism Paradigms

### 2.1 Data Parallelism (DP/DDP)

**Concept**: Each GPU has a complete copy of the model. Each processes different data samples in parallel.

**Forward Pass**:
```
GPU 0: forward(batch_0) → gradients_0
GPU 1: forward(batch_1) → gradients_1
GPU N: forward(batch_N) → gradients_N
```

**All-Reduce**: Sum gradients across all GPUs
```
avg_gradient = (gradients_0 + gradients_1 + ... + gradients_N) / N
```

**Backward**: Each GPU updates with averaged gradients

**Memory Per GPU**: M × 1.0 (model) + M × 2.0 (gradients + optim states)
- For 70B model: ~560 GB per GPU (infeasible on single GPU)

**Communication Cost**: O(M) for all-reduce where M = model size

**Pros**:
- Simple implementation (PyTorch DDP native)
- No model code changes required
- Good scaling for small-medium models
- Load balancing is trivial

**Cons**:
- Memory doesn't scale with number of GPUs
- High communication overhead (O(M) per step)
- Limited by single GPU memory for large models

**When to Use**:
- Models ≤ 7B parameters
- Ample GPU memory available
- Communication bandwidth is good

---

### 2.2 Tensor Parallelism (TP)

**Concept**: Split individual tensors/layers across GPUs. Each GPU computes partial forward/backward.

**Matrix Multiplication Parallelism**:
```
For Y = X × W (m×n × n×p)

GPU 0: computes X × W_0 (first p/2 columns)
GPU 1: computes X × W_1 (second p/2 columns)
Result: concatenate [Y_0 | Y_1]
```

**Communication Pattern**:
- **After matrix multiply**: All-reduce across TP group
- **Between layers**: All-gather then distribute
- Communication happens between every layer forward/backward

**Memory Per GPU**: M / TP_size + communication buffers
- For 70B model with TP=8: ~140GB/8 = 17.5GB (feasible!)

**Communication Cost**: Frequent all-reduces between layers

**Pros**:
- Significantly reduces per-GPU memory
- Balanced computation across GPUs
- Works well for transformer layers

**Cons**:
- Requires model code changes
- High synchronization overhead
- More complex to implement
- Communication happens frequently

**Implementation (Megatron-LM style)**:
```python
# Split column-wise
class ColumnParallelLinear(nn.Module):
    def forward(self, x):
        # x: (seq_len, hidden) on each GPU
        # W_local: (hidden, hidden/TP) - only this GPU's columns
        y_local = F.linear(x, self.weight_local, self.bias)
        # All-reduce to sum contributions from all TP GPUs
        y = all_reduce(y_local)  # (seq_len, hidden)
        return y

# Split row-wise
class RowParallelLinear(nn.Module):
    def forward(self, x):
        # x: (seq_len, hidden) on each GPU  
        # W_local: (hidden/TP, hidden) - only this GPU's rows
        y_local = F.linear(x, self.weight_local, self.bias)
        # All-gather to collect output from all TP GPUs
        y = all_gather(y_local)  # (seq_len, hidden)
        return y
```

**When to Use**:
- Models 7B-70B parameters
- Need fine-grained parallelism
- Transformer-based architectures
- Teams with Megatron-LM expertise

---

### 2.3 Pipeline Parallelism (PP)

**Concept**: Split model layers across GPUs sequentially. Different GPUs compute different layers on same sample.

**Forward Pass** (4 GPUs, 12 layers, 3 layers per GPU):
```
Time 0: GPU_0 forward (layers 0-2) on batch_0
Time 1: GPU_0 forward on batch_1 | GPU_1 forward (layers 3-5) on batch_0
Time 2: GPU_0 forward on batch_2 | GPU_1 forward on batch_1 | GPU_2 forward on batch_2
...
```

**Memory Per GPU**: M / PP_size (approximately)
- For 70B model with PP=8: ~70GB/8 = 8.75GB (very efficient!)

**Communication**: Small activations and gradients between stages

**Challenges**:
- **Bubble Overhead**: Early stages idle while later stages compute
- **Activation Storage**: Need to store activations for backward pass

**GPipe Scheduling** (recommended):
```
Micro-batches: M/m where M=batch size, m=micro-batch size
Pipeline bubbles: (P-1)/(m-1) where P=number of stages
For 8 stages, 4 micro-batches: bubble = 7/3 = 2.33 (26% overhead)
```

**Pros**:
- Minimal memory per GPU
- Can use very large batch sizes
- Excellent for very deep models

**Cons**:
- Pipeline bubble overhead
- Complex backward pass (activation recomputation or storage)
- Difficult to balance - some GPUs idle
- Requires careful scheduling

**Optimal Micro-batch Size** (1-F/1):
```
# Where:
# F = forward time per micro-batch
# B = backward time (typically ≈ 2F)
# C = communication time
# P = number of pipeline stages
# Optimal: avoid pipeline bubble by keeping all GPUs busy
```

**When to Use**:
- Very deep models (100+ layers)
- Memory is critical bottleneck
- Have homogeneous GPU cluster
- Willing to tolerate some idle time

---

### 2.4 Hybrid Parallelism Strategies

**Recommended Combinations**:

**DP + TP (e.g., 8 GPUs: 2 DP groups × 4 TP)**:
```
DP Group 0          DP Group 1
[GPU_0, GPU_1,      [GPU_4, GPU_5,
 GPU_2, GPU_3]   ×   GPU_6, GPU_7]
      ↓                    ↓
   TP (4 GPUs)        TP (4 GPUs)
```
- Use case: 30B-100B models on small clusters (8-16 GPUs)
- Benefits: Reduced TP communication, better load balancing
- Typical config: DP=2, TP=4 for 70B on 8xA100

**TP + PP (e.g., 64 GPUs: 4 TP groups × 16 PP stages)**:
```
Stage_0     Stage_1  ...  Stage_15
TP_0,1,2,3 TP_4,5,6,7 ... TP_60-63
```
- Use case: 200B+ models on large clusters
- Benefits: TP reduces per-GPU memory, PP amortizes activation storage
- Typical config: TP=4, PP=16 for 540B on 64xA100

**DP + TP + PP (e.g., 256 GPUs: 4 DP × 8 TP × 8 PP)**:
```
[4 DP groups] × [8 TP within each DP] × [8 PP stages for each TP group]
```
- Use case: 1T+ parameter models, cutting-edge research
- Benefits: Maximum flexibility and scaling
- Typical config: DP=4, TP=8, PP=8 for 1T parameter model

**Selection Heuristic**:
```
Model Size | GPUs Available | Recommended Strategy
≤ 7B      | 1-4           | DDP (DP only)
≤ 7B      | 8-16          | DDP or (DP=2, TP=4)
7B-30B    | 8-32          | DP=2, TP=4
30B-70B   | 16-64         | DP=2, TP=4 or DP=2, TP=8
70B-200B  | 32-128        | TP=4, PP=4-8 or DP=2,TP=4,PP=4
200B+     | 128+          | TP=4-8, PP=8-16, DP=2-4
```

---

## 3. Communication Efficiency

### 3.1 All-Reduce Algorithms

The all-reduce operation is fundamental: each GPU sends its gradient, each receives the average.

**Algorithm 1: Tree All-Reduce**
```
Complexity: O(log P) rounds
Bandwidth per GPU: 2(1 - 1/P) × M

Example (8 GPUs, M = 100MB):
Round 0: [GPU0+GPU1], [GPU2+GPU3], [GPU4+GPU5], [GPU6+GPU7]  (partial sums)
Round 1: [GPU0-1+GPU2-3], [GPU4-5+GPU6-7]                    (larger partial sums)
Round 2: All GPUs have full sum                               (broadcast result)

Latency: 3 × O(log 8) = 9 network hops
Bandwidth: ~200MB per GPU
```

**Algorithm 2: Ring All-Reduce** (More bandwidth efficient)
```
Complexity: O(P) rounds (2P-2 sends to be precise)
Bandwidth per GPU: 2 × M × (P-1)/P ≈ 2M (nearly optimal!)
Latency: 2(P-1) network hops

Ring all-reduce is MORE communication-efficient than tree!

Example (8 GPUs, M = 100MB):
Phase 1 - Reduce-Scatter (P-1 rounds):
Round 1: GPU_i sends chunk_i to GPU_(i+1), receives chunk_(i-1) from GPU_(i-1)
Round 2: GPU_i sends (chunk_i + received) to GPU_(i+1)
...
Round 7: Each GPU holds sum of all its chunks

Phase 2 - All-Gather (P-1 rounds):
Round 1: GPU_i broadcasts its chunk to ring
...
Round 7: All GPUs receive all chunks in correct order

Total: 2 × 7 = 14 rounds, but higher bandwidth utilization
```

**Ring All-Reduce Implementation Concept**:
```python
# For P GPUs numbered 0 to P-1
chunk_size = model_size // num_gpus

for step in range(num_gpus - 1):
    # Reduce-scatter phase
    send_to = (rank + 1) % num_gpus
    recv_from = (rank - 1) % num_gpus
    
    # Each GPU sends/receives one chunk, accumulates
    chunk_idx = (rank - step) % num_gpus
    local_chunk = gradients[chunk_idx * chunk_size:(chunk_idx+1) * chunk_size]
    
    # Concurrent send/recv to hide latency
    send(local_chunk, send_to)
    remote_chunk = recv(recv_from)
    
    # Accumulate in-place
    local_chunk += remote_chunk

for step in range(num_gpus - 1):
    # All-gather phase - similar ring pattern, no accumulation
    ...
```

**Comparison**:

| Algorithm | Latency | Bandwidth | Use Case |
|-----------|---------|-----------|----------|
| Tree | O(log P) | Lower | Small P (≤16), high latency |
| Ring | O(P) | Higher | Large P (≥32), bandwidth-limited |
| Mesh | O(√P) | Better | Specialized topologies |

**When Ring Dominates**: Ring has lower bandwidth requirement (~2M vs ~3M for tree on large clusters). For P≥32, ring is generally preferred.

---

### 3.2 Gradient Compression Techniques

**Problem**: All-reduce transmits full precision (fp32 or fp16) gradients, consuming significant bandwidth.

**Solution**: Reduce transmission size without sacrificing convergence.

### 3.2.1 Quantization-Based Compression

**Uniform Quantization (k-bit)**:
```
Standard: 32-bit float per gradient = 100% bandwidth
k-bit: k bits per gradient = k/32 × 100% bandwidth

1-bit: 3% of original! (-3.33x compression)
2-bit: 6.25% (-16x compression)
4-bit: 12.5% (-8x compression)
8-bit: 25% (-4x compression)

Quantization Formula:
q_i = round(g_i / s)  where s = max(|g_i|) / (2^(k-1) - 1)
g_i_recovered = q_i × s

Error: typically 0.1-1% accuracy loss with proper scaling
```

**Gradient Clipping + Quantization**:
```python
# Example: 4-bit quantization
def quantize_gradient(gradient, bits=4):
    # Find scale
    max_val = torch.abs(gradient).max()
    scale = max_val / (2**(bits-1) - 1)
    
    # Quantize
    quantized = torch.round(gradient / scale).to(torch.int8)
    
    # Communication: send scale (float32) + quantized (int8)
    # Bandwidth: (4 bytes scale + 1 byte × size) vs (4 bytes × size)
    # Compression: 5/4000 = 0.125% for large gradients
    
    return quantized, scale

def dequantize_gradient(quantized, scale):
    return quantized.to(torch.float32) * scale
```

**Convergence Impact**:
- 4-bit quantization: typically <0.5% accuracy loss
- 2-bit quantization: 1-3% accuracy loss (requires careful tuning)
- 1-bit: significant loss unless modified (e.g., gradient averaging)

---

### 3.2.2 Sparsification Methods

**Top-k Sparsification** (transmit only largest |gradient| values):
```
Compression Ratio: k/total (e.g., k=0.01 → 1% sparse = 100x compression!)

Algorithm:
1. Compute gradient norm for each layer
2. Select top-k% gradients by absolute value
3. Send only (index, value) pairs
4. Receiver zero-fills unspecified indices, adds to accumulator

Example (100M parameters, top-1% = 1M):
Original: 400MB (fp32)
Compressed: (1M indices × 4B + 1M values × 4B) ≈ 8MB
Compression: 50x!

Implementation:
```python
def sparse_gradient(gradient, sparsity=0.01):
    # Get top-k
    num_keep = max(1, int(gradient.numel() * (1 - sparsity)))
    values, indices = torch.topk(torch.abs(gradient.flatten()), num_keep)
    
    # Create sparse tensor
    sparse_grad = torch.sparse.FloatTensor(
        indices=indices.unsqueeze(0),
        values=gradient.flatten()[indices],
        size=gradient.shape
    )
    return sparse_grad
```

**Convergence**: With top-1% sparsity, typically <0.5% accuracy loss after tuning.

**Error Feedback Loop** (critical for convergence):
```python
# Without error feedback - divergence risk
for step in range(num_steps):
    gradients = backward(loss)
    sparse_g = top_k_sparsify(gradients, k=0.01)
    averaged_g = all_reduce(sparse_g)
    optimizer.step(averaged_g)

# With error feedback - maintains convergence
residual = 0  # Accumulate dropped gradient errors
for step in range(num_steps):
    gradients = backward(loss)
    
    # Add back residual errors
    gradients_with_feedback = gradients + residual
    
    # Sparsify and communicate
    sparse_g = top_k_sparsify(gradients_with_feedback, k=0.01)
    residual = (gradients_with_feedback - sparse_g)  # Save errors
    
    averaged_g = all_reduce(sparse_g)
    optimizer.step(averaged_g)
```

---

### 3.2.3 Hybrid Approaches

**Layer-wise Adaptive Compression (L-GRECO)**:
```
Observation: Different layers have different gradient distributions
- Early layers: large, dense gradients
- Late layers: smaller, sparser gradients

Adaptive compression per layer:
Layer 0 (embedding):   8-bit quantization (less critical)
Layer 5-20 (middle):   4-bit quantization (most important)
Layer 80+ (output):    2-bit quantization (less data)

Result: Maintains accuracy while reducing bandwidth 4-8x
```

**Accumulation-based Compression**:
```
Use 1-2 bit transmission but accumulate errors locally:

for step in range(num_steps):
    full_gradient = backward(loss)
    accumulated_residual += full_gradient  # Keep full precision locally
    
    # Compress only for communication
    compressed = 1bit_quantize(accumulated_residual)
    all_reduce(compressed)  # Tiny bandwidth
    
    optimizer.step(received_gradient)
    accumulated_residual = 0
```

---

### 3.3 Communication vs. Computation Trade-off

**Hidden Communication** (overlap computation with communication):

```
Traditional (sequential):
Time: [Forward] → [Backward] → [All-Reduce] → [Optimizer]
Total: T_fwd + T_bwd + T_allreduce + T_optim

Overlapped (concurrent):
            [Forward]
                    ↓
            [Backward - Layer N]
                    ↓
            [All-Reduce Layer 1] ← [Backward - Layer N-1] (parallel!)
                    ↓
            [All-Reduce Layer 2] ← [Backward - Layer N-2]
                    ...

Total: T_fwd + T_bwd (hidden all-reduce underneath!)
Savings: ~T_allreduce × (1 - overlap_factor)
```

**Practical Overlap Strategy**:
```python
# With gradient accumulation and bucketing
torch.set_float32_matmul_precision('medium')

def training_step_with_overlap(model, data, optimizer):
    # Forward on full batch
    outputs = model(data)
    loss = compute_loss(outputs)
    
    # Backward with bucketing - group gradients for communication
    loss.backward(retain_graph=False)
    
    # At end of backward, all-reduce happens asynchronously
    # PyTorch DDP: all-reduce in separate thread/GPU stream
    # Overlap: gradient bucket is sent while next backward computes
    
    optimizer.step()
    optimizer.zero_grad()
```

**Bandwidth Saturation Analysis**:
```
Given:
- GPU compute: C TFLOPS
- Network bandwidth: B Gbps = B/8 GB/s
- Model size: M GB

Time for forward+backward: T_compute = 6M / C (rough estimate)
Time for all-reduce: T_comm = 3M / B (rough estimate)

Compute-communication ratio: T_compute / T_comm = (6M/C) / (3M/B) = 2B/C

If ratio >> 1: Communication is hidden (overlappable)
If ratio << 1: Communication is bottleneck (can't hide)

Example:
H100 (1350 TFLOPS) + 400 Gbps network:
Ratio = 2 × (400/8) / 1350 = 2 × 50 / 1350 = 0.074 << 1
→ Communication IS bottleneck! Need compression/selective sync.

A100 (312 TFLOPS) + 200 Gbps network:
Ratio = 2 × (200/8) / 312 = 2 × 25 / 312 = 0.16 << 1
→ Also communication-bottlenecked.
```

**Recommendation**: For modern GPUs and clusters, use:
1. Gradient compression (4-8 bit)
2. Communication overlap (FSDP, DeepSpeed)
3. Ring all-reduce (for large clusters)
4. Periodic (not every-step) all-reduce if training tolerance allows

---

## 4. DeepSpeed ZeRO Optimizer

DeepSpeed ZeRO is the most impactful optimizer for memory efficiency in distributed training.

### 4.1 ZeRO-1: Gradient Sharding

**Memory Breakdown** (70B model, Adam optimizer):
```
Without ZeRO:
- Model weights (fp16): 140 GB
- Gradients (fp16): 140 GB
- Optimizer states (fp32 weights + momentum + variance): 280 GB
- Activations: ~50-100 GB (batch size dependent)
TOTAL PER GPU: ~610 GB ❌ Infeasible on single GPU

With ZeRO-1 (P=8):
- Model weights (fp16): 140 GB (all GPUs have copy)
- Gradients (fp16): 140 GB ÷ 8 = 17.5 GB ✓
- Optimizer states (fp32): 280 GB ÷ 8 = 35 GB ✓
- Activations: ~50-100 GB
TOTAL PER GPU: ~192 GB (still large but much better!)
```

**Implementation**:
```python
# Before ZeRO:
for rank in range(num_gpus):
    # Each GPU computes full gradient
    gradients[rank] = backward(loss)  # Full gradient
    all_reduce(gradients)  # Sum across GPUs
    optimizer_state[rank].step()  # Update using full optimizer state

# With ZeRO-1:
for rank in range(num_gpus):
    gradients[rank] = backward(loss)  # Full gradient computed
    
    # Bucket gradients for all-reduce
    bucket_size = model_size / num_gpus
    for i in range(0, len(gradients), bucket_size):
        # Each rank only updates its partition
        grad_partition = gradients[i:i+bucket_size]
        all_reduce(grad_partition)
        # Only this rank updates its partition of optimizer state
        optimizer_state[rank % num_gpus].step(grad_partition)
```

**Memory Formula**:
```
ZeRO-1 Memory = Model_size + (Grad_size + Optimizer_state_size) / P
Where P = number of GPUs

For 70B model, P=8:
= 140 GB + (140 GB + 280 GB) / 8
= 140 + 52.5 = 192.5 GB per GPU
```

**Trade-off**: Full gradients still replicated (all backward passes see full gradients), but optimizer states partitioned.

---

### 4.2 ZeRO-2: Optimizer + Gradient Sharding

**Key Improvement**: Gradient sharding goes further - don't replicate gradients during backward.

**Memory Breakdown**:
```
With ZeRO-2 (P=8):
- Model weights (fp16): 140 GB
- Gradients (fp16): 140 GB ÷ 8 = 17.5 GB ✓
- Optimizer states (fp32): 280 GB ÷ 8 = 35 GB ✓
- Activations: ~50-100 GB
TOTAL PER GPU: ~192 GB

Wait, looks same as ZeRO-1? The memory difference is in when data is reduced...
```

**Actual Difference**: 
- ZeRO-1: Gradients generated in full, then partitioned for optimizer
- ZeRO-2: Gradients reduced (all-reduced) by partition immediately after backward

**Key Feature - Gradient Partitioning**:
```
ZeRO-2 configuration with contiguous_gradients:
- Reduces memory fragmentation during backward
- All-reduce happens in buckets (not per-parameter)
- Overlaps backward computation with all-reduce

Configuration Example:
{
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,        # 500M elements
        "allgather_bucket_size": 5e8
    }
}
```

**Communication Optimization**:
- `overlap_comm: true` → All-reduce overlaps with backward
- `reduce_scatter: true` → Use reduce-scatter instead of all-reduce (saves 1 step!)

**Memory Savings Formula**:
```
ZeRO-2 Memory = Model_size + (Grad_size + Optimizer_state_size) / P
= 140 GB + (140 GB + 280 GB) / 8 = 192.5 GB

Reduction from baseline (no ZeRO):
4 × (140 + 280) / 2 = 420 GB → 192.5 GB = 2.18x reduction
Or thinking in stages: 4x→2x (per-stage memory reduction)
```

---

### 4.3 ZeRO-3: Full Parameter Sharding

**Game Changer**: Shard model weights themselves across GPUs!

**Memory Breakdown**:
```
With ZeRO-3 (P=8):
- Model weights (fp16): 140 GB ÷ 8 = 17.5 GB ✓✓
- Gradients (fp16): 140 GB ÷ 8 = 17.5 GB ✓✓
- Optimizer states (fp32): 280 GB ÷ 8 = 35 GB ✓✓
- Activations: ~50-100 GB
TOTAL PER GPU: ~120 GB (4x reduction!)

With ZeRO-3 (P=16):
= 8.75 + 8.75 + 17.5 + 50-100
= ~85 GB per GPU (8x reduction!)
```

**How It Works**:

1. **Forward Pass** (all-gather weights):
```python
def forward(x):
    # At start of forward, weights are sharded
    # GPU_i has W_partition[i]
    
    # All-gather to reconstruct full weights for computation
    W_full = all_gather(W_partition)  # O(model_size) communication
    
    # Compute forward
    y = F.linear(x, W_full, bias)
    
    # Free weights after forward to save memory
    del W_full
    
    return y
```

2. **Backward Pass** (all-gather weights again):
```python
def backward(grad_output):
    # All-gather weights for backward computation
    W_full = all_gather(W_partition)
    
    # Compute backward
    grad_input = grad_output @ W_full.T
    grad_weight = grad_output.T @ input
    
    # Free weights
    del W_full
    
    # Gradients are reduced and partitioned by ZeRO-3
    return grad_input
```

**Critical Optimization: Streaming**:
```
Instead of all-gathering full weights at once:

# Traditional (high peak memory):
W_full = all_gather(W_partition)  # 8x peak memory!
compute(W_full)

# Streaming (memory-efficient):
for i in range(P):
    W_partition_i = all_gather_one_partition(i)
    compute_on_partition(W_partition_i)  # Process incrementally
    del W_partition_i
# Memory reduced by 8x!
```

**DeepSpeed ZeRO-3 Configuration**:
```json
{
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        
        "stage3_max_live_parameters": 1e9,      // Max params to keep alive
        "stage3_max_reuse_distance": 1e9,        // Activation reuse threshold
        "stage3_prefetch_bucket_size": 1e7,      // Bucket size for prefetching
        "stage3_param_persistence_threshold": 1e5,  // Don't shard if < threshold
        
        "reduce_bucket_size": 1e7,
        "sub_group_size": 1e9,
        
        "offload_optimizer": {"device": "cpu"},  // Offload optimizer to CPU
        "offload_param": {"device": "cpu"}       // Offload weights to CPU
    }
}
```

**Memory Formula - The 8x Magic**:
```
Without any optimization (4 copies):
- FP32 weights: M
- FP32 gradients: M
- Optimizer: 2M (momentum + variance)
- All replicated across P GPUs
Per-GPU: 4M/P

ZeRO-3 (sharded, can use fp16):
- FP16 weights: M/2
- FP16 gradients: M/2
- FP32 Optimizer states: 2M
- All sharded across P GPUs
Per-GPU: (M/2 + M/2 + 2M) / P = 3M / P

Wait, that's 4M/P vs 3M/P = 1.33x, not 8x!

The 8x comes from multiple factors:
1. Reduced precision (fp16 vs fp32): 2x
2. Weight sharding: P/1 (8x for P=8)
3. Reduced-memory optimizer (e.g., 8-bit): 4x
Combined: 2 × 8 × 1-2 ≈ 8-16x for large P
```

**Realistic ZeRO-3 Example** (70B model on 8xA100):
```
Without ZeRO:
- Per-GPU memory needed: 560GB ❌

With ZeRO-3:
- Per-GPU memory needed: 560GB / 8 = 70GB ✓
- With 8-bit optimizer: 70GB / 2 = 35GB ✓
- With offload to CPU: Can fit on single GPU!
```

---

### 4.4 ZeRO-Infinity: Offloading to CPU/NVMe

**Problem**: Even with ZeRO-3, 70GB per GPU might not fit on V100 (32GB VRAM).

**Solution**: Offload to CPU/NVMe memory temporarily.

**Bandwidth Consideration**:
```
Offload Target | Bandwidth | Latency | Use Case
GPU Memory     | 500 GB/s  | < 1 μs  | Primary compute
GPU ↔ CPU      | 100 GB/s  | 1 μs    | Temporary storage
CPU ↔ NVMe     | 10 GB/s   | 1-10 ms | Large batches
```

**Strategy**:
```python
# ZeRO-Infinity with CPU offload
offload_plan = {
    "device": "cpu",        # Offload to CPU
    "buffer_count": 4,      # 4 concurrent buffers
    "fast_init": False      # Whether to use fast init
}

# During forward:
# 1. All-gather weights from GPU shards to CPU (if offloaded)
# 2. Transfer to GPU in micro-batch pipeline
# 3. Compute forward
# 4. Release weights back to CPU

# Key: All-gather happens asynchronously
```

**ZeRO-Infinity Memory** (training 1T model on 256 GPUs):
```
Without ZeRO-Infinity:
- Per-GPU: (1T weights + 1T gradients + 2T optimizer) / 256 = ~16GB ✓

With ZeRO-Infinity + CPU offload:
- GPU memory: ~5GB (active compute)
- CPU memory: ~11GB (weights + activations)
- Total system memory: 256 × 5GB GPU + 256 × 11GB CPU = 4TB
(Feasible with 16GB CPU RAM per node!)

Without it: Would need full 4T in GPU VRAM (impossible!)
```

---

## 5. Computation-Communication Overlap

### 5.1 Pipeline Parallelism Scheduling

**GPipe Scheduling** (recommended baseline):

```
4 stages (PP=4), 4 micro-batches (MB=4):

Timeline:
F: Forward  B: Backward  W: Weight update

   Stage0  Stage1  Stage2  Stage3
T0   F0
T1   F1      F0
T2   F2      F1      F0
T3   F3      F2      F1      F0          ← All 4 stages busy!
T4   B0      F3      F2      F1
T5   B1      B0      F3      F2
T6   B2      B1      B0      F3
T7   B3      B2      B1      B0          ← All stages busy backward too
T8   W0      B3      B2      B1
T9   W1      W0      B3      B2
T10  W2      W1      W0      B3
T11  W3      W2      W1      W0
T12         W3      W2      W1
T13                W3      W2
T14                       W3

Utilized slots: 4×3 (forward) + 4×3 (backward) + 4×3 (weight) + 3 (tail)
Total slots: 15
Efficiency: 36/15 = 73%
Pipeline bubble: (P-1)/(MB × 3) = 3/12 = 25% (better!)
```

**Micro-batch Sizing**:
```
More micro-batches → Better efficiency:

MB=1: Efficiency = (P + 2) / (3P) = (4+2)/(12) = 50% with P=4
MB=2: Efficiency = (P×2 + 2) / (3×2×P) = 10/24 = 41% (worse!)
      Wait, wrong formula...
      
Correct: Efficiency = (P×MB + 2) / (3×P×MB) as MB→∞ → 1/3 = 33%
         But with proper scheduling: 1 - (P-1)/(P×MB) → 1 as MB increases

MB=4: Efficiency ≈ 1 - 3/16 = 81% ✓
MB=8: Efficiency ≈ 1 - 3/32 = 91% ✓
MB=16: Efficiency ≈ 1 - 3/64 = 95% ✓

Rule of thumb: Use MB = max(4, 2×P)
```

**Activation Memory Trade-off**:
```
Micro-batch stores activations for backward pass:

Total activation memory = Sum of all (layer_activation_size × MB)
                       = avg_activation_size × seq_len × batch_size × MB

For 70B model:
Per-sample: ~500MB activations
With MB=4: 500MB × 4 = 2GB
With MB=16: 500MB × 16 = 8GB (significant but necessary for efficiency)

Optimization: Use activation checkpointing (recompute instead of storing)
With checkpointing: ~500MB total (recompute forward in backward)
Cost: 33% slower backward (acceptable for memory savings)
```

---

### 5.2 Ring All-Reduce with Gradient Computation Overlap

**The Goal**: While GPU computes gradients for layer k, send gradients from layer k-1 over network.

**Implementation Pattern**:
```
Traditional (sequential):
╔═══════════════════╗
║ Backward Layer 0  ║
╚═══════════════════╝
╔═══════════════════╗
║ Backward Layer 1  ║
╚═══════════════════╝
╔═══════════════════╗
║ Backward Layer 2  ║
╚═══════════════════╝
║ All-Reduce (blocked)
╔═══════════════════╗
║ All-Reduce Layer 0║
╚═══════════════════╝
║ (GPU idle here)
Total time: T_backward + T_allreduce

Overlapped (with gradient bucketing):
╔═══════════════════════════════════════════════════╗
║ Backward Layer 0 | Backward Layer 1 | Backward L2 │
║ [send grad_0]   | [send grad_1]     |             │
║   (async)       |   (async)         |             │
║ (GPU computing) │ (GPU computing)    │ (GPU comp)  │
╚═══════════════════════════════════════════════════╝

Total time: T_backward (all-reduce hidden!)
Savings: T_allreduce × (1 - overlap_fraction)
Typical: 80-90% overlap possible
```

**PyTorch DDP/FSDP Implementation**:
```python
class AutoGradAccumulate:
    """Gradient bucketing for overlap"""
    def __init__(self, model, bucket_size=25e6):  # 25M parameters per bucket
        self.buckets = self._create_buckets(model, bucket_size)
        self.backward_hooks = []
        
    def _create_buckets(self, model, bucket_size):
        # Group parameters into buckets by size
        buckets = []
        current = []
        size = 0
        for param in model.parameters():
            if size + param.numel() > bucket_size and current:
                buckets.append(current)
                current, size = [], 0
            current.append(param)
            size += param.numel()
        if current:
            buckets.append(current)
        return buckets
    
    def register_hooks(self):
        # Register backward hooks to reduce gradients per bucket
        for bucket_idx, bucket in enumerate(self.buckets):
            def hook_factory(idx):
                def hook(grad):
                    # Trigger all-reduce for this bucket
                    # while next layers compute gradients
                    all_reduce_async(grad, bucket_idx)
                return hook
            
            for param in bucket:
                param.register_hook(hook_factory(bucket_idx))
```

---

### 5.3 Async SGD and Gradient Buffering

**Asynchronous Optimization**: Update weights with gradients from previous iterations while computing new gradients.

```python
class AsyncOptimizer:
    """Update with delay"""
    def __init__(self, model, optimizer, update_delay=1):
        self.optimizer = optimizer
        self.gradient_buffer = []
        self.update_delay = update_delay
        self.step_count = 0
    
    def step(self):
        # Accumulate gradients
        self.gradient_buffer.append(get_gradients())
        
        # Update with delayed gradients
        if len(self.gradient_buffer) > self.update_delay:
            old_grads = self.gradient_buffer.pop(0)
            
            # Update with stale gradient
            # (network might be sending latest gradient meanwhile)
            update_weights(old_grads)
        
        self.step_count += 1

# Staleness tolerance:
# Delay=0: Synchronous (wait for all-reduce)
# Delay=1: Use gradient from 1 step ago (can hide 1 all-reduce)
# Delay=2: Use gradient from 2 steps ago (can hide 2 all-reduces)
#
# Convergence: Typically tolerates delay up to 2-4 steps with minimal impact
```

---

## 6. Implementation Details

### 6.1 PyTorch DDP Setup

**Basic Data Parallel Distributed Training**:

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch.distributed as dist

def setup(rank, world_size):
    """Initialize distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPUs, gloo for CPUs
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=300)
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    
    # Create model
    model = MyTransformerModel(vocab_size=50000, hidden_size=1024)
    model.to(rank)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,  # Set True if some params not used
        gradient_as_bucket_view=True,   # Memory efficient
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Data loading
    train_dataset = MyDataset()
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=32,  # Per-GPU batch size
        pin_memory=True,
        num_workers=4
    )
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Reshuffle at each epoch
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(rank)
            labels = labels.to(rank)
            
            # Forward pass
            logits = model(input_ids)
            loss = compute_loss(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()  # DDP automatically all-reduces gradients
            
            # Optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}: Loss {loss.item()}")
    
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
```

**Launching DDP**:
```bash
# With torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    train.py

# Or with torchrun (newer)
torchrun --nproc_per_node=8 train.py

# Multi-node
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py
```

---

### 6.2 FSDP (Fully Sharded Data Parallel)

**PyTorch Native ZeRO-3 Alternative**:

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType
)
from torch.distributed.fsdp.wrap import enable_wrap, wrap

def setup_fsdp_model(model):
    """Configure and wrap model with FSDP"""
    
    # Sharding strategy options:
    # FULL_SHARD: Full ZeRO-3 (shard everything)
    # SHARD_GRAD_OP: ZeRO-2 (shard gradients + optimizer)
    # NO_SHARD: DDP (replicate model)
    
    sharding_strategy = ShardingStrategy.FULL_SHARD
    
    # CPU offload for large models
    cpu_offload = CPUOffload(offload_params=True)  # Offload weights to CPU
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        
        # Memory optimization
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch next weights
        forward_prefetch=True,  # Prefetch forward weights
        
        # Checkpointing (save memory)
        use_reentrant=True,  # Use gradient checkpointing
        
        # Limit memory usage
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
    )
    
    return model

def main():
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    # Create and wrap model
    model = MyLargeTransformer()
    model = setup_fsdp_model(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=8)
    
    # Training
    for epoch in range(10):
        sampler.set_epoch(epoch)
        
        for batch in loader:
            # Forward
            output = model(batch)
            loss = loss_fn(output)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()  # FSDP handles gradients
            
            optimizer.step()
    
    # Save model
    state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, 'checkpoint.pt')

if __name__ == '__main__':
    main()
```

**FSDP with Auto-Wrapping** (recommended):

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    # Build model
    model = MyTransformer()
    
    # Auto-wrap configuration: wrap every layer with > 100M params
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=100_000_000  # 100M parameters threshold
    )
    
    # Apply FSDP with auto-wrapping
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )
    
    # ... rest of training
```

---

### 6.3 DeepSpeed Integration

**Full Example with ds_config.json**:

```json
{
  "train_batch_size": 512,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": "auto",
  
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 32,
    "hysteresis": 2
  },
  
  "gradient_clipping": 1.0,
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 1000,
      "total_num_steps": 100000
    }
  },
  
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "synchronize_checkpoint_boundary": true
  },
  
  "communication_data_type": "fp16",
  "reduce_scatter": true,
  
  "flops_profiler": {
    "enabled": true,
    "profile_step": 10,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
  }
}
```

**Python Training Script**:

```python
import deepspeed
import torch
from transformers import AutoModel, AutoTokenizer

def main():
    # Model and tokenizer
    model_name = "meta-llama/Llama-2-70b-hf"
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # DeepSpeed initialization
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config='ds_config.json'
    )
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            # Forward
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            
            # Backward
            model_engine.backward(loss)
            
            # Optimizer step
            model_engine.step()
            
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
    
    # Save checkpoint
    if model_engine.global_rank == 0:
        model_engine.save_checkpoint('/path/to/checkpoint')

if __name__ == '__main__':
    main()
```

**Launching DeepSpeed**:
```bash
deepspeed --num_gpus=8 train.py \
    --deepspeed_config ds_config.json \
    --model_name meta-llama/Llama-2-70b-hf \
    --output_dir ./checkpoints
```

---

### 6.4 Megatron-LM Patterns

**Tensor Parallelism with Megatron**:

```python
from megatron.core.parallel_state import initialize_model_parallel
from megatron.core.model_parallel_config import ModelParallelConfig
import megatron.core.tensor_parallel as tp

def main():
    # Initialize parallel groups
    config = ModelParallelConfig(
        tensor_model_parallel_size=4,      # 4-way tensor parallelism
        pipeline_model_parallel_size=2,    # 2-way pipeline parallelism
        virtual_pipeline_model_parallel_size=1,
        
        # Memory
        use_distributed_optimizer=True,    # Distributed Adam
        
        # Communication
        sequence_parallel=True,            # Sequence parallelism for long seqs
        finalize_model_parallel_in_validation=True
    )
    
    initialize_model_parallel(config)
    
    # Build model with tensor parallel layers
    model = GPTModel(config)
    
    # Wrap forward with loss scaling, gradient accumulation
    # (typically handled by Megatron's training loop)
    
    # Megatron provides custom training loop
    # (rather than standard PyTorch pattern)
    megatron_train(model, config)

# Key Megatron abstractions:
# 1. ColumnParallelLinear / RowParallelLinear
# 2. ParallelMLP / ParallelSelfAttention
# 3. get_tensor_parallel_group() for collective ops
# 4. get_pipeline_parallel_group() for pipeline stages
```

---

## 7. Advanced Optimization Techniques

### 7.1 Batch Size Scaling Laws

**Batch Size Scaling Rule**:

```
When increasing GPU count from N to 8N:
- Increase batch size from B to 8B
- Maintain same batch size per GPU (B/N)
- Learning rate scaling: lr_new = lr_old × √8 (for specific domains)

Why? Gradient variance:
var(gradient) ∝ 1/batch_size
When we accumulate gradients from 8x more samples, variance ↓ 8x
To maintain training dynamics, increase learning rate by √8

Example:
8 GPUs, batch_size=32 per GPU → total=256
16 GPUs, batch_size=32 per GPU → total=512
→ 2x effective batch size, learning rate → lr × √2
```

**Optimal Batch Size**:

```
Theoretical: batch_size = 4 / (learning_rate × variance_per_sample)

Practical heuristic:
- Start with B = 256 for 70B model
- Try B = 512, 1024, 2048
- If loss diverges: reduce B or reduce learning rate
- If loss stable: increase B to reduce training time

Maximum practical batch size:
- Activation memory: ~0.5 GB per sample for 70B model
- 16 × 32GB GPU = 512GB activations max
- Max batch size ≈ 512GB / 0.5GB = 1024 per GPU (with FSDP)
```

---

### 7.2 Learning Rate Scaling

**Linear Scaling Rule** (most important):

```
When batch size increases by k:
lr_new = lr_old × k

Example (Llama 2 training):
Batch size: 256 → 512 (2x increase)
Learning rate: 1.5e-4 → 3e-4

Intuition:
- Larger batch: gradient estimate is more accurate
- Can take larger steps in optimizer
- Linear rule empirically works for batch_size ∈ [256, 4096]

When does it NOT work?
- Very small batch (<32): use smaller multiplier (√k instead of k)
- Very large batch (>4096): diminishing returns
```

**Warmup with Scaling**:

```python
def get_learning_rate_with_scaling(step, args):
    """
    Linear warmup + cosine annealing with batch size scaling
    """
    total_steps = args.num_train_steps
    warmup_steps = int(0.01 * total_steps)  # 1% warmup
    
    if step < warmup_steps:
        # Linear warmup
        return args.lr_base * (step / warmup_steps)
    
    # Cosine annealing
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * args.lr_base * (1 + math.cos(math.pi * progress))

# With batch size scaling:
args.lr_base = 1.5e-4 * math.sqrt(args.batch_size / 256)
```

**Advanced: LAMB Optimizer with Scaling**:

```python
from torch_optimizer import LAMB

optimizer = LAMB(
    model.parameters(),
    lr=1.5e-4 * math.sqrt(effective_batch_size / 256),
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=0.01,
    adam=False,  # Pure LAMB (vs Adam-like)
)

# LAMB automatically scales learning rate with batch size
# Particularly useful for large batch training (>4096)
```

---

### 7.3 Synchronization Patterns

**Bulk Synchronous Parallel (BSP)**:
```
All GPUs wait for all gradients to be ready before optimizer step
- Slowest GPU determines speed
- Stable convergence
- Default for DDP/FSDP

Pattern:
1. All forward+backward (wait for slowest)
2. All-reduce gradients (synchronized)
3. Optimizer step (all do simultaneously)
4. Barrier before next step
```

**Asynchronous Parallel (ASP)**:
```
GPUs proceed without waiting for others
- Faster wall-clock time
- Gradient staleness may affect convergence
- Not typically used for LLM training

Pattern:
1. GPU_i computes forward+backward (independent timeline)
2. All-reduce current gradient
3. Optimizer step
(No synchronization points)

Staleness issue: GPU_0 updates with gradient from GPU_8 computed 3 steps ago
```

**Staleness-Aware Momentum (recommended)**:
```python
class StalenessAwareSGD(torch.optim.Optimizer):
    """Adjust momentum based on gradient staleness"""
    
    def __init__(self, params, lr=1e-3, momentum=0.9, staleness_tau=3):
        super().__init__(params, {})
        self.lr = lr
        self.momentum = momentum
        self.tau = staleness_tau  # Expected staleness in steps
    
    def step(self, staleness):
        """
        staleness: number of steps since gradient was computed
        """
        # Adjust momentum based on staleness
        # Higher staleness → lower momentum (less memory of old gradients)
        adaptive_momentum = self.momentum ** staleness
        
        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                
                d_p = param.grad.data
                
                if 'momentum_buffer' not in param.state:
                    buf = param.state['momentum_buffer'] = torch.clone(d_p)
                else:
                    buf = param.state['momentum_buffer']
                    buf.mul_(adaptive_momentum).add_(d_p)
                
                param.data.add_(buf, alpha=-self.lr)
```

---

### 7.4 Load Balancing

**Problem**: Some GPUs finish computation before others (stragglers).

**Detection**:
```python
def detect_stragglers(gpu_times):
    """Find slowest GPU"""
    mean_time = sum(gpu_times) / len(gpu_times)
    stragglers = [i for i, t in enumerate(gpu_times) if t > 1.2 * mean_time]
    return stragglers

# Example: GPU_3 takes 5s while others take 4s (20% slower)
gpu_times = [4.0, 4.1, 4.0, 5.0]  # GPU_3 is straggler
```

**Solutions**:

1. **Sequence Length Rebalancing** (for transformers):
```python
# Instead of equal-length sequences per GPU:
# GPU_0: padding_token × 2048 (slow because padded)
# GPU_1: actual_tokens × 2048 (faster, less computation)

# Rebalance:
# GPU_0: actual_tokens × 2300 (more actual data to compute)
# GPU_1: actual_tokens × 1800 (less data, more balanced)

def rebalance_sequences(sequences, target_gpu_time):
    """Adjust sequence length to balance computation"""
    # Estimate computation from actual tokens
    actual_tokens = [len(s.strip()) for s in sequences]
    
    # Adjust length to balance
    target_tokens = sum(actual_tokens) / num_gpus
    
    # Pad/truncate to balance
    rebalanced = []
    for tokens in actual_tokens:
        if tokens < target_tokens:
            # Pad to balance
            length = max(tokens, int(target_tokens * 1.1))
        else:
            length = int(target_tokens * 0.9)
        rebalanced.append(length)
    
    return rebalanced
```

2. **Dynamic Batch Size Adjustment**:
```python
def adaptive_batch_size(straggler_ratio, current_batch_size):
    """Reduce batch size if stragglers detected"""
    if straggler_ratio > 1.2:
        # This GPU is 20% slower
        # Reduce its batch size
        return int(current_batch_size * 0.9)
    return current_batch_size
```

---

## 8. Code Examples and Best Practices

### 8.1 Complete DDP Training Example

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50000, hidden_size=1024, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=16,
                dim_feedforward=4096,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        return logits

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.timedelta(minutes=30)
    )
    
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)

def cleanup():
    dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0 and dist.get_rank() == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)

def main(rank, world_size, num_epochs=3):
    """Main training function"""
    setup(rank, world_size)
    
    # Create synthetic dataset
    num_samples = 1000
    seq_length = 512
    vocab_size = 50000
    
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    dataset = TensorDataset(input_ids, labels)
    
    # Distributed sampler ensures different data per GPU
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    train_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=8,
        pin_memory=True
    )
    
    # Model, optimizer, device
    device = rank
    model = TransformerModel(vocab_size=vocab_size, hidden_size=1024, num_layers=12)
    model = model.to(device)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[device],
        output_device=device,
        find_unused_parameters=False
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    for epoch in range(num_epochs):
        # Set epoch for sampler to shuffle properly
        sampler.set_epoch(epoch)
        
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
    
    # Save checkpoint (only rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), 'model_checkpoint.pt')
        print("Checkpoint saved!")
    
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    
    mp.spawn(main, args=(world_size, 3), nprocs=world_size, join=True)
```

**Run with**:
```bash
python train_ddp.py
# Or with torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=8 train_ddp.py
```

---

### 8.2 DeepSpeed Configuration Examples

**Configuration for 70B Model on 8xA100**:

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "steps_per_print": 10,
  
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1.5e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01,
      "adam_w_mode": true
    }
  },
  
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu"
    },
    
    "sub_group_size": 1e9,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 32,
    "hysteresis": 2
  },
  
  "gradient_clipping": 1.0,
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1.5e-4,
      "warmup_num_steps": 1000,
      "total_num_steps": 100000
    }
  },
  
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "synchronize_checkpoint_boundary": true
  },
  
  "data_efficiency": {
    "enabled": false
  }
}
```

---

### 8.3 FSDP Configuration

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
    ShardingStrategy,
    BackwardPrefetch
)
from torch.distributed.fsdp.wrap import enable_wrap, wrap, size_based_auto_wrap_policy
import functools

def create_model_with_fsdp(model_config):
    """Create model and wrap with FSDP"""
    
    # Model creation
    model = create_base_model(model_config)
    
    # FSDP wrapping policy
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=100_000_000  # Wrap layers >100M params
    )
    
    # FSDP configuration
    fsdp_config = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,  # ZeRO-3
        "cpu_offload": CPUOffload(offload_params=True),
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": True,
        "device_id": torch.cuda.current_device(),
        "use_reentrant": True,  # Gradient checkpointing
        "limit_all_gathers": True,
        "sync_module_states": True,
    }
    
    # Apply FSDP
    model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=auto_wrap_policy,
        **fsdp_config
    )
    
    return model
```

---

### 8.4 Debugging Distributed Training

**Common Issues and Solutions**:

```python
import time
import torch.distributed as dist

class DistributedDebugger:
    """Debug distributed training issues"""
    
    @staticmethod
    def check_synchronization(device):
        """Verify all ranks synchronize correctly"""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        print(f"Rank {rank} starting sync check...")
        start = time.time()
        
        # Barrier should block until all ranks reach it
        dist.barrier()
        
        elapsed = time.time() - start
        print(f"Rank {rank} sync time: {elapsed:.4f}s")
        
        # If one rank takes much longer, it's a straggler
        if elapsed > 1.0:
            print(f"WARNING: Rank {rank} is slow ({elapsed:.2f}s)")
    
    @staticmethod
    def check_gradient_flow(model):
        """Verify gradients computed on all ranks"""
        rank = dist.get_rank()
        
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Rank {rank}: No gradient for {name}")
            else:
                grad_norm = torch.norm(param.grad)
                print(f"Rank {rank}: {name} grad norm: {grad_norm:.6f}")
    
    @staticmethod
    def check_all_reduce(tensor):
        """Verify all-reduce behavior"""
        rank = dist.get_rank()
        
        # Send different value from each rank
        tensor = tensor * (rank + 1)
        
        print(f"Before all_reduce - Rank {rank}: {tensor.item()}")
        
        dist.all_reduce(tensor)
        
        print(f"After all_reduce - Rank {rank}: {tensor.item()}")
        
        # All ranks should have same value (sum)
    
    @staticmethod
    def detect_nan_inf(model):
        """Find NaN/Inf in weights and gradients"""
        rank = dist.get_rank()
        
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                print(f"Rank {rank}: NaN in weights {name}")
            if torch.isinf(param.data).any():
                print(f"Rank {rank}: Inf in weights {name}")
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"Rank {rank}: NaN in gradients {name}")

# Usage in training loop
def train_with_debugging(model, train_loader, optimizer, device):
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            # Forward/backward
            loss = compute_loss(model, batch)
            loss.backward()
            
            # Check for NaN/Inf
            DistributedDebugger.detect_nan_inf(model)
            
            # Check gradient flow
            if batch_idx % 100 == 0:
                DistributedDebugger.check_gradient_flow(model)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Check synchronization
            if batch_idx % 1000 == 0:
                DistributedDebugger.check_synchronization(device)
```

---

## 9. Performance Analysis and Benchmarking

### 9.1 Scaling Efficiency Metrics

**Weak Scaling** (increase model/batch size proportionally with GPUs):

```
Definition: Keep compute per GPU constant, increase total capacity

Baseline: 1 GPU, model_size=1B, batch_size=32
         Time per step: 1.0 second

Weak scaling to 8 GPUs:
- Increase model to 8B
- Increase batch size to 256
- Expected time: 1.0 second (same!)

Efficiency: Actual_time / Expected_time
- Perfect: 1.0 (linear scaling)
- Typical: 0.85-0.95 (85-95% of linear)
- With communication overhead: 0.7-0.8

Formula:
Weak_efficiency = 1 / (1 + communication_overhead)
                = 1 / (1 + comm_time / comp_time)
                = comp_time / (comp_time + comm_time)
```

**Strong Scaling** (fix problem size, increase GPU count):

```
Definition: Same total compute, spread across more GPUs

Baseline: 1 GPU, 70B model, time=100 seconds

Strong scaling to 8 GPUs:
- Same 70B model
- Distribute across 8 GPUs (with parallelism)
- Expected speedup: 8x → 12.5 seconds

Actual: 14 seconds (12.5s ideal + 1.5s overhead)
Efficiency: 8 / (100 / 14) = 8 / 7.14 = 1.12 (impossible!)
           
Correctly: speedup = 100 / 14 = 7.14x
          efficiency = 7.14 / 8 = 0.89 (89%)

Amdahl's Law:
speedup = 1 / (f + (1-f)/N)
where f = fraction of non-parallelizable code

For f=0.01 (1% serial), N=8:
speedup = 1 / (0.01 + 0.99/8) = 1 / 0.134 = 7.46x (93% efficiency)
```

**Practical Measurement**:

```python
def measure_scaling_efficiency(models_sizes, gpu_counts):
    """Measure weak and strong scaling"""
    results = {}
    
    for gpu_count in gpu_counts:
        for model_size in model_sizes:
            # Run training for fixed number of iterations
            start = time.time()
            
            # ... run training ...
            
            elapsed = time.time() - start
            
            results[(gpu_count, model_size)] = elapsed
    
    # Calculate efficiency
    baseline_time = results[(1, 100_000_000)]  # 100M params on 1 GPU
    
    print("GPU Count | Model Size | Time (s) | Speedup | Efficiency")
    for gpu_count in gpu_counts:
        for model_size in model_sizes:
            if (gpu_count, model_size) in results:
                time_taken = results[(gpu_count, model_size)]
                speedup = baseline_time / time_taken
                efficiency = speedup / gpu_count
                
                print(f"{gpu_count:3d}      | {model_size/1e9:5.1f}B    | {time_taken:7.2f}    | {speedup:6.2f}x  | {efficiency:6.1%}")
```

---

### 9.2 Communication Overhead Measurement

```python
def measure_communication_overhead(model_size, num_gpus):
    """Measure all-reduce communication cost"""
    import time
    
    # Dummy tensor of model size
    tensor = torch.randn(model_size // 4).cuda()  # Divide by 4 for fp32
    
    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor)
    
    dist.barrier()
    
    # Time all-reduce
    iterations = 100
    start = time.time()
    
    for _ in range(iterations):
        dist.all_reduce(tensor)
    
    dist.barrier()
    elapsed = time.time() - start
    
    # Calculate metrics
    time_per_allreduce = elapsed / iterations
    total_bytes = model_size * 4  # fp32
    bandwidth = total_bytes / time_per_allreduce / 1e9  # GB/s
    
    print(f"Model size: {model_size/1e9:.1f}B parameters")
    print(f"All-reduce time: {time_per_allreduce*1000:.2f} ms")
    print(f"Network bandwidth used: {bandwidth:.1f} GB/s")
    print(f"Measured communication ratio: {time_per_allreduce / compute_time:.2%}")
    
    return time_per_allreduce

# Typical results:
# 70B model, 8 GPUs, A100-GPU
# All-reduce time: ~50ms (at ~200 GB/s network)
# Compute time for forward+backward: ~2-3 seconds
# Overhead: 50ms / 2500ms = 2% (very manageable with overlap)
```

---

### 9.3 Bottleneck Identification

```python
def profile_training_bottleneck(model, train_loader, device):
    """Identify where time is spent"""
    import cProfile
    import pstats
    
    # Profile computation
    prof = cProfile.Profile()
    
    for batch in train_loader:
        prof.enable()
        
        # Forward
        output = model(batch)
        loss = loss_fn(output)
        
        # Backward
        loss.backward()
        
        prof.disable()
    
    # Print stats
    ps = pstats.Stats(prof)
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    
    # Key metrics to look for:
    # 1. If most time in backward → gradient computation is slow
    # 2. If distributed ops take time → all-reduce is bottleneck
    # 3. If data loading takes time → I/O is bottleneck

def measure_distributed_ops(model_size, num_gpus):
    """Measure distributed operation costs"""
    
    # Create dummy gradients
    gradients = torch.randn(model_size // 4).cuda()
    
    import time
    
    # Measure all-reduce
    start = time.time()
    for _ in range(100):
        dist.all_reduce(gradients)
    allreduce_time = (time.time() - start) / 100
    
    # Measure all-gather (for FSDP)
    gathered = [torch.zeros_like(gradients) for _ in range(num_gpus)]
    start = time.time()
    for _ in range(100):
        dist.all_gather(gathered, gradients)
    allgather_time = (time.time() - start) / 100
    
    # Measure broadcast
    start = time.time()
    for _ in range(100):
        dist.broadcast(gradients, src=0)
    broadcast_time = (time.time() - start) / 100
    
    print(f"Model size: {model_size/1e9:.1f}B")
    print(f"All-reduce time: {allreduce_time*1000:.2f} ms")
    print(f"All-gather time: {allgather_time*1000:.2f} ms")
    print(f"Broadcast time: {broadcast_time*1000:.2f} ms")
    
    # Comparison: which is slowest?
    # Helps identify communication bottleneck
```

---

## 10. References and Resources

### Key Papers

1. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (2020)
   - https://arxiv.org/abs/1910.02054
   - Introduces ZeRO-1, ZeRO-2, ZeRO-3 stages
   - Memory reduction: 4x→8x for ZeRO-3

2. **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning** (2021)
   - https://arxiv.org/abs/2104.07857
   - CPU/NVMe offloading for trillion-scale models
   - Infinite memory through smart offloading

3. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** (2019)
   - https://arxiv.org/abs/1909.08053
   - Tensor parallelism and pipeline parallelism
   - Foundation for modern distributed LLM training

4. **Efficient Large-Scale Language Modeling with Mixtures of Experts** (2021)
   - https://arxiv.org/abs/2101.03961
   - Sparsity and conditional computation
   - Relevant for communication efficiency

5. **Communication-Efficient Learning of Deep Networks from Decentralized Data** (2016)
   - https://arxiv.org/abs/1602.05629
   - Gradient compression and federated learning
   - Foundation for communication-efficient training

6. **Gradient Sparsification for Communication-Efficient Distributed Optimization** (2018)
   - https://arxiv.org/abs/1809.08383
   - Top-k sparsification with error feedback
   - Critical for efficient communication

7. **L-GRECO: Layerwise-Adaptive Gradient Compression for Efficient and Accurate Deep Learning** (2022)
   - https://arxiv.org/abs/2210.17357
   - Adaptive compression per layer
   - Better convergence with heterogeneous compression

### Official Documentation

- **PyTorch Distributed**: https://pytorch.org/docs/stable/distributed.html
- **DeepSpeed**: https://www.deepspeed.ai/
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
- **PyTorch FSDP**: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- **Transformers Distributed Training**: https://huggingface.co/docs/transformers/v4.37.0/en/training

### Tools and Frameworks

1. **DeepSpeed**: Complete distributed training framework with ZeRO, optimizations
2. **Megatron-LM**: Tensor/pipeline parallelism reference implementation
3. **PyTorch FSDP**: Native ZeRO-3 style parallelism in PyTorch
4. **Hugging Face Accelerate**: High-level distributed training API
5. **Ray Air**: Distributed ML platform with integrated parallelism strategies

### Benchmarking Resources

- **MLCommons MLPerf**: Industry standard LLM training benchmarks
- **NVIDIA DXStream**: Distributed training simulator
- **Perfetto**: Distributed system performance profiler
- **PyTorch Profiler**: Native profiling with distributed trace support

---

## Key Takeaways

1. **Memory is the primary bottleneck**: Use ZeRO-3 sharding (4-8x reduction) for large models
2. **Communication overhead grows with cluster size**: Use compression + overlap techniques
3. **Batch size scaling is crucial**: Linear scaling rule (lr × √k for batch size increase k)
4. **Hybrid parallelism is standard**: DP + TP or TP + PP for production training
5. **Measurement and profiling are essential**: Profile before and after optimization
6. **Tradeoffs matter**: Accuracy vs speed, communication vs memory, complexity vs stability

---

## Debugging Checklist

- [ ] Verify all ranks initialized correctly (`dist.get_rank()`, `dist.get_world_size()`)
- [ ] Check no NaN/Inf in early steps (gradient clipping issues)
- [ ] Monitor memory usage over time (memory leaks?)
- [ ] Measure computation vs communication breakdown
- [ ] Verify gradient synchronization across ranks
- [ ] Check learning rate scaling with batch size
- [ ] Validate loss decreases smoothly (no spikes from communication issues)
- [ ] Profile forward/backward/all-reduce timing separately
- [ ] Test on small model first, scale gradually
- [ ] Check network bandwidth with `iperf` or similar

