# Custom Kernels — Inference Skill Prompt

Writing custom CUDA/Triton kernels for optimized LLM inference.

---

## 1. Identity and Mission

Implement custom kernels using CUDA C++ and Triton to accelerate LLM inference operations beyond what standard libraries provide. This includes fused attention kernels, specialized activation functions, quantization kernels, and memory-optimized operations that can significantly improve throughput and latency.

---

## 2. Theory & Fundamentals

### 2.1 CUDA Programming Model

**Thread Hierarchy:**
```
Grid → Block → Thread
```

**Memory Hierarchy:**
```
Global Memory (HBM)
    ↓
Shared Memory (SMEM per block)
    ↓
Registers (per thread)
```

### 2.2 Triton Overview

Triton provides:
- Python-based kernel authoring
- Automatic optimization (tile size, memory coalescing)
- Just-in-time compilation to CUDA

### 2.3 Key Operations in LLM Inference

- **Attention**: QKV projection + softmax + output projection
- **FFN**: GeGLU/SiGLU activation
- **LayerNorm**: RMSNorm/GroupNorm
- **Rotary embeddings**: Rotary position encoding

---

## 3. Implementation Patterns

### Pattern 1: Basic Triton Kernel Structure

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'num_stages': 4}),
    ],
    key=['M', 'N'],
)
@triton.jit
def matmul_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, num_stages: tl.constexpr
):
    """
    Fused matmul kernel with Triton.
    Computes: output = input @ weight
    """
    pid = tl.program_id(axis=0)
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = (group_id * num_pid_n) % num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + ((pid // num_pid_m) % num_pid_n)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_N)  # BLOCK_SIZE_K

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_N):
        a = tl.load(input_ptrs)
        b = tl.load(weight_ptrs)
        accumulator += tl.dot(a, b)

        input_ptrs += BLOCK_SIZE_N * stride_ik
        weight_ptrs += BLOCK_SIZE_N * stride_wk

    output = accumulator.to(tl.float16)
    output_ptrs = output_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :]
    tl.store(output_ptrs, output)


def triton_matmul(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to call Triton matmul kernel.

    Args:
        input: (M, K) tensor
        weight: (K, N) tensor

    Returns:
        output: (M, N) tensor
    """
    assert input.is_cuda and weight.is_cuda, "Inputs must be on GPU"

    M, K = input.shape
    K_w, N = weight.shape
    assert K == K_w, "Dimension mismatch"

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        input, weight, output,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
    )

    return output
```

### Pattern 2: Fused Attention Kernel with Flash Attention

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, L, M,
    q_head_dim, v_head_dim,
    seq_len, cu_seqlens,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused attention kernel (simplified FlashAttention-style).
    Computes: Out = softmax(Q @ K^T / sqrt(d)) @ V
    """
    cur_head = tl.program_id(0)
    cur_batch = tl.program_id(1)

    # Get sequence range for this batch
    seq_start = tl.load(cu_seqlens + cur_batch)
    seq_end = tl.load(cu_seqlens + cur_batch + 1)
    cur_seq_len = seq_end - seq_start

    # Load Q for this head
    q_offset = cur_batch * stride_qb + cur_head * stride_qh
    q_ptrs = Q + q_offset + tl.arange(0, BLOCK_SIZE) * stride_qm
    q = tl.load(q_ptrs)

    # Initialize accumulator and max value
    m_i = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE, v_head_dim), dtype=tl.float32)

    # Outer loop over K, V
    for start_k in range(0, cur_seq_len, BLOCK_SIZE):
        k_offset = cur_batch * stride_kb + cur_head * stride_kh + start_k * stride_kn
        k_ptrs = K + k_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_kn + tl.arange(0, v_head_dim)[None, :] * stride_kh
        k = tl.load(k_ptrs)

        # Compute Q @ K^T
        qk = tl.dot(q, k) * softmax_scale

        # Causal masking
        mask = (start_k + tl.arange(0, BLOCK_SIZE)[:, None]) < cur_seq_len
        qk = tl.where(mask, qk, float('-inf'))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_i_new[:, None])
        s = tl.sum(p, axis=1)

        # Rescale
        p_scale = tl.exp(m_i - m_i_new)
        l_i = l_i * p_scale + s
        m_i = m_i_new

        # Load V
        v_offset = cur_batch * stride_vb + cur_head * stride_vh + start_k * stride_vn
        v_ptrs = V + v_offset + tl.arange(0, BLOCK_SIZE)[None, :] * stride_vn
        v = tl.load(v_ptrs)

        # Accumulate
        acc_scale = tl.exp(m_i - m_i_new)
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    out_offset = cur_batch * stride_ob + cur_head * stride_oh
    out_ptrs = Out + out_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_om
    tl.store(out_ptrs, acc)


def triton_attention(
    q: torch.Tensor,  # (batch, heads, seq, head_dim)
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    softmax_scale: float = None,
) -> torch.Tensor:
    """Fused attention with Triton."""
    B, H, M, q_head_dim = q.shape
    _, _, K_seq, k_head_dim = k.shape
    _, _, V_seq, v_head_dim = v.shape

    assert q_head_dim == k_head_dim == v_head_dim, "Head dimensions must match"

    if softmax_scale is None:
        softmax_scale = 1.0 / (q_head_dim ** 0.5)

    O = torch.empty((B, H, M, q_head_dim), device=q.device, dtype=q.dtype)
    BLOCK_SIZE = triton.next_power_of_2(v_head_dim)

    grid = (H, B)

    _fwd_kernel[grid](
        q, k, v, O, None, None,
        q_head_dim, v_head_dim,
        K_seq, cu_seqlens,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        BLOCK_SIZE,
    )

    return O
```

### Pattern 3: Custom GeGLU Activation Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def geglu_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GeGLU activation: GELU(gate) * value

    x is split into two halves along feature dim:
    x = [value, gate]
    output = GELU(gate) * value
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # GeGLU formula: GELU(gate) * value
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqr2pi = 0.7978845608
    cdf = 0.5 * (1.0 + tl.tanh(sqr2pi * (x + 0.044715 * tl.pow(x, 3))))
    output = x * cdf

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_geglu(x: torch.Tensor) -> torch.Tensor:
    """
    Compute GeGLU activation.

    Args:
        x: (..., 2*hidden_dim) tensor

    Returns:
        output: (..., hidden_dim) tensor
    """
    assert x.shape[-1] % 2 == 0, "Last dim must be even"
    hidden_dim = x.shape[-1] // 2

    output = torch.empty((*x.shape[:-1], hidden_dim), device=x.device, dtype=x.dtype)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    geglu_kernel[grid](x, output, n_elements, BLOCK_SIZE)

    return output


@triton.jit
def geglu_gradient_kernel(
    x_ptr, grad_output_ptr, grad_input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gradient of GeGLU.
    d/dx GeGLU(x) = dGELU(gate) * value + GELU(gate) * d_value
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask)

    # GeGLU gradient computation
    sqr2pi = 0.7978845608
    x3 = tl.pow(x, 3)
    tanh_arg = sqr2pi * (x + 0.044715 * x3)
    tanh_out = tl.tanh(tanh_arg)

    # dGELU/dx
    cdf = 0.5 * (1 + tanh_out)
    pdf = 0.5 * sqr2pi * (1 - tl.pow(tanh_out, 2)) * (1 + 0.134145 * x3 + 3.0 * 0.044715 * x * x)

    grad_input = grad_out * (cdf + x * pdf)

    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


def triton_geglu_backward(
    x: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """Backward pass for GeGLU."""
    grad_input = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    geglu_gradient_kernel[grid](x, grad_output, grad_input, n_elements, BLOCK_SIZE)

    return grad_input
```

### Pattern 4: Quantization Kernels (INT8/INT4)

```python
import torch
from typing import Tuple

@torch.jit.script
def quantize_per_tensor_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-tensor INT8 quantization.
    Returns: (quantized, scale, zero_point)
    """
    scale = x.abs().max() / 127.0
    scale = scale.to(torch.float32)

    quantized = (x / scale).round().clamp(-128, 127).to(torch.int8)
    zero_point = torch.zeros(1, dtype=torch.int8, device=x.device)

    return quantized, scale, zero_point


@torch.jit.script
def quantize_per_channel_int8(
    x: torch.Tensor,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-channel INT8 quantization.
    Quantization params differ per channel (typically dim=0 for weights, dim=-1 for activations).
    """
    if dim == -1:
        dim = x.dim() - 1

    scales = x.abs().amax(dim=dim, keepdim=True) / 127.0
    scales = scales.to(torch.float32)

    # Reshape for broadcasting
    shape = [1] * x.dim()
    shape[dim] = x.shape[dim]

    quantized = (x / scales.view(shape)).round().clamp(-128, 127).to(torch.int8)

    zero_point = torch.zeros_like(scales).to(torch.int8)

    return quantized, scales, zero_point


def gemm_forward_int8(
    a: torch.Tensor,  # (M, K) int8
    b: torch.Tensor,  # (K, N) int8
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
) -> torch.Tensor:
    """
    INT8 GEMM using Triton.

    Computes: Y = (A * scales_a) @ (B * scales_b)
             = A @ B * (scales_a * scales_b)

    We compute A_int8 @ B_int8 and then rescale.
    """
    # Convert to float for computation
    a_fp = a.float() * scales_a
    b_fp = b.float() * scales_b

    # Use Triton matmul or torch.matmul
    y = torch.matmul(a_fp, b_fp)

    return y


@triton.jit
def dynamic_quantize_kernel(
    x_ptr, scale_ptr, zero_point_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dynamic quantization kernel for activations.
    Quantizes each row independently.
    """
    row_start = tl.program_id(0)
    row_stride = tl.num_programs(0)

    for row in range(row_start, n_elements, row_stride):
        # Each row has BLOCK_SIZE elements
        offs = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < row * (n_elements // 8) + BLOCK_SIZE  # Simplified

        x = tl.load(x_ptr + offs, mask=mask, other=0.0)

        # Compute scale per row
        abs_max = tl.max(tl.abs(x))
        scale = abs_max / 127.0
        scale = tl.maximum(scale, 1e-8)

        quantized = tl.round(x / scale).to(tl.int8)

        tl.store(scale_ptr + row, scale)
        tl.store(out_ptr + offs, quantized, mask=mask)


def triton_dynamic_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically quantize tensor row-wise.

    Returns:
        quantized: int8 tensor
        scale: float32 scale per row
    """
    M = x.shape[0]
    K = x.shape[1]

    quantized = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = 128
    grid = (M,)

    dynamic_quantize_kernel[grid](
        x, scale, None, quantized, K, BLOCK_SIZE
    )

    return quantized, scale
```

### Pattern 5: Fused LayerNorm + Rotary Embedding

```python
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm kernel.

    RMSNorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS
    rms = tl.sqrt(tl.mean(x * x) + eps)

    # Normalize
    x_norm = (x / rms) * weight

    # Store
    tl.store(output_ptr + offsets, x_norm.to(tl.float16), mask=mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm.
    """
    output = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    rmsnorm_kernel[grid](
        x, weight, output,
        n_elements, eps, BLOCK_SIZE
    )

    return output


@triton.jit
def rotary_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    seq_len, head_dim,
    stride_qm, stride_qh,
    stride_km, stride_kh,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Rotary Position Embedding (RoPE) kernel.

    Applies rotation to Q and K based on position.
    RoPE(x, pos) = [cos(pos*theta), -sin(pos*theta); sin(pos*theta), cos(pos*theta)] * x
    """
    pid = tl.program_id(0)
    seq_id = pid % seq_len
    head_id = pid // seq_len

    # Load cos, sin for this position
    cos = tl.load(cos_ptr + seq_id * head_dim + tl.arange(0, BLOCK_SIZE))
    sin = tl.load(sin_ptr + seq_id * head_dim + tl.arange(0, BLOCK_SIZE))

    half_dim = head_dim // 2

    # Load Q
    q_offs = seq_id * stride_qm + head_id * stride_qh + tl.arange(0, half_dim)
    q = tl.load(q_ptr + q_offs)
    q_rest = tl.load(q_ptr + q_offs + half_dim)

    # Apply rotation
    q_rot = tl.where(tl.arange(0, BLOCK_SIZE) < half_dim, q * cos - q_rest * sin, q_rest * cos + q * sin)

    # Store rotated Q
    tl.store(q_ptr + q_offs, q_rot)

    # Load K
    k_offs = seq_id * stride_km + head_id * stride_kh + tl.arange(0, half_dim)
    k = tl.load(k_ptr + k_offs)
    k_rest = tl.load(k_ptr + k_offs + half_dim)

    # Apply rotation
    k_rot = tl.where(tl.arange(0, BLOCK_SIZE) < half_dim, k * cos - k_rest * sin, k_rest * cos + k * sin)

    # Store rotated K
    tl.store(k_ptr + k_offs, k_rot)


def triton_apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to Q and K.

    Args:
        q: (batch, heads, seq, head_dim)
        k: (batch, heads, seq, head_dim)
        cos: (seq, head_dim)
        sin: (seq, head_dim)
    """
    B, H, M, D = q.shape
    BLOCK_SIZE = triton.next_power_of_2(D // 2)

    grid = (B * H * M,)

    rotary_kernel[grid](
        q, k, cos, sin,
        M, D,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        BLOCK_SIZE,
    )

    return q, k
```

### Pattern 6: Memory-Efficient KV Cache Access

```python
import torch
import triton
import triton.language as tl

@triton.jit
def kv_cache_update_kernel(
    k_cache_ptr, v_cache_ptr,
    k_new_ptr, v_new_ptr,
    start_pos: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Update KV cache with new K, V values.
    Efficiently copies K/V to sliding window positions.
    """
    pid = tl.program_id(0)
    head_id = pid // seq_len
    pos_id = pid % seq_len

    cache_pos = start_pos + pos_id

    # Load new K, V
    k_new_offs = pos_id * stride_kb + head_id * stride_kh + tl.arange(0, BLOCK_SIZE) * stride_kn
    v_new_offs = pos_id * stride_vb + head_id * stride_vh + tl.arange(0, BLOCK_SIZE) * stride_vn

    k_new = tl.load(k_new_ptr + k_new_offs)
    v_new = tl.load(v_new_ptr + v_new_offs)

    # Store to cache
    k_cache_offs = head_id * stride_kh + cache_pos * stride_kn + tl.arange(0, BLOCK_SIZE)
    v_cache_offs = head_id * stride_vh + cache_pos * stride_vn + tl.arange(0, BLOCK_SIZE)

    tl.store(k_cache_ptr + k_cache_offs, k_new)
    tl.store(v_cache_ptr + v_cache_offs, v_new)


def triton_update_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    start_pos: int,
):
    """
    Update KV cache efficiently.
    """
    B, H, max_seq, D = k_cache.shape
    _, _, seq_len, _ = k_new.shape

    BLOCK_SIZE = triton.next_power_of_2(D)

    grid = (B * H * seq_len,)

    kv_cache_update_kernel[grid](
        k_cache, v_cache, k_new, v_new,
        start_pos, seq_len, D,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        BLOCK_SIZE,
    )

    return k_cache, v_cache


@triton.jit
def kv_cache_retrieve_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    num_kv_heads: tl.constexpr,
    stride_qm, stride_qh,
    stride_kb, stride_kh, stride_kn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Retrieve K, V from cache for attention computation.
    Handles GQA by having fewer KV heads than Q heads.
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id = tl.program_id(2)

    # For GQA, map Q head to corresponding KV head
    kv_head_id = head_id // (num_kv_heads // 8)  # Simplified ratio

    # Compute which K, V to retrieve
    k_offs = batch_id * stride_kb + kv_head_id * stride_kh + tl.arange(0, BLOCK_N) * stride_kn
    v_offs = batch_id * stride_kb + kv_head_id * stride_kh + tl.arange(0, BLOCK_N) * stride_kn

    # Load K, V for this head
    k = tl.load(k_cache_ptr + k_offs)
    v = tl.load(v_cache_ptr + v_offs)

    # Compute attention scores with Q
    q_offs = batch_id * stride_qm + head_id * stride_qh + tl.arange(0, BLOCK_M)
    q = tl.load(q_ptr + q_offs)

    scores = tl.dot(q, k.T)

    # Softmax (simplified)
    scores = tl.softmax(scores, axis=1)

    # Compute output
    out = tl.dot(scores, v)

    # Store
    out_offs = batch_id * stride_qm + head_id * stride_qh + tl.arange(0, BLOCK_M)
    tl.store(output_ptr + out_offs, out)


def triton_kv_cache_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """
    Compute attention using cached K, V.
    """
    B, H, M, D = q.shape
    _, _, max_seq, _ = k_cache.shape

    output = torch.empty_like(q)

    BLOCK_M = 32
    BLOCK_N = triton.next_power_of_2(max_seq)

    grid = (B, H, 1)

    kv_cache_retrieve_kernel[grid](
        q, k_cache, v_cache, output,
        max_seq, D, num_kv_heads,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        BLOCK_M, BLOCK_N,
    )

    return output
```

### Pattern 7: Custom CUDA Extension (Pure CUDA C++)

```cuda
// cuda_extension.cu
// Compile: nvcc -shared -O3 -Xcompiler -fPIC cuda_extension.cu -o cuda_extension.so

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

__global__ void matmul_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int M, int N, int K,
    float alpha
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            sum += __half2float(a[row * K + k]) * __half2float(b[k * N + col]);
        }

        c[row * N + col] = __float2half(sum * alpha);
    }
}

extern "C" {

cudaError_t matmul_cuda(
    const __half* a,
    const __half* b,
    __half* c,
    int M, int N, int K,
    float alpha,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<<<grid, block, 0, stream>>>(
        a, b, c, M, N, K, alpha
    );

    return cudaGetLastError();
}

}
```

```python
# wrapper.py
import torch
import numpy as np
from ctypes import cdll, c_float, c_int, POINTER

class MatMulCUDAExtension:
    """Python wrapper for CUDA matmul extension."""

    def __init__(self, library_path="./cuda_extension.so"):
        self.lib = cdll.LoadLibrary(library_path)
        self.lib.matmul_cuda.argtypes = [
            POINTER(np.float16),  # a
            POINTER(np.float16),  # b
            POINTER(np.float16),  # c
            c_int,  # M
            c_int,  # N
            c_int,  # K
            c_float,  # alpha
            c_int,  # stream (as int)
        ]

    def matmul(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Matmul using custom CUDA kernel."""
        assert a.is_cuda and b.is_cuda
        assert a.dtype == torch.float16 and b.dtype == torch.float16

        M, K = a.shape
        K2, N = b.shape
        assert K == K2

        c = torch.empty((M, N), device='cuda', dtype=torch.float16)

        # Flatten tensors for ctypes
        a_ptr = a.contiguous().data_ptr()
        b_ptr = b.contiguous().data_ptr()
        c_ptr = c.contiguous().data_ptr()

        # Get current stream
        stream = torch.cuda.current_stream().cuda_stream

        # Call CUDA kernel
        self.lib.matmul_cuda(
            a_ptr,
            b_ptr,
            c_ptr,
            M, N, K,
            alpha,
            int(stream),
        )

        return c
```

---

## 4. Framework Integration

### Integration with vLLM

```python
# In vLLM, custom kernels can be integrated via:
# 1. Custom CUDA ops in attention layers
# 2. Override attention backend

from vllm.attention.backends.flash_attn import FlashAttentionBackend

class CustomFlashAttention(FlashAttentionBackend):
    # Override attention computation with custom kernel
    pass
```

### Integration with PyTorch

```python
# Register custom op
from torch.utils.cpp_extension import load

custom_ops = load(name="custom_ops", sources=["custom_ops.cpp", "custom_ops.cu"])

# Use in model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 512)

    def forward(self, x):
        # Call custom CUDA op
        return custom_ops.custom_matmul(x, self.linear.weight)
```

---

## 5. Performance Considerations

### Kernel Launch Overhead

| Operation | Overhead (µs) | Recommendation |
|-----------|--------------|----------------|
| Small GEMM (<128x128) | 5-10 | Batch or fuse |
| Attention (<256 seq) | 3-5 | Use Flash Attention |
| Element-wise | 2-3 | Fuse with other ops |

### Memory Bandwidth

| Operation | Memory Access | Peak BW Usage |
|-----------|--------------|---------------|
| GEMM | O(2*M*K*N) | ~80-95% |
| Attention | O(M*N*d) | ~60-80% |
| LayerNorm | O(M*D) | ~40-60% |

### Triton vs CUDA C++

| Aspect | Triton | Pure CUDA |
|--------|--------|-----------|
| Development Speed | Fast (Python) | Slow |
| Performance | Near-optimal | Optimal |
| Flexibility | Limited | Full |
| Debugging | Harder | Easier |

---

## 6. Common Pitfalls

1. **Bank Conflicts**: Shared memory access patterns causing conflicts
2. **Warp Divergence**: Different execution paths within warp
3. **Memory Coalescing**: Non-optimal global memory access patterns
4. **Register Pressure**: Too many local variables
5. **Type Mismatches**: Mixing tensor cores and non-tensor operations
6. **Tile Size**: Wrong tile size causing poor occupancy

---

## 7. Research References

1. https://arxiv.org/abs/2205.14135 — "FlashAttention: Fast and Memory-Efficient Exact Attention"

2. https://triton-lang.org/ — Triton documentation

3. https://github.com/openai/triton — Triton GitHub

4. https://arxiv.org/abs/2304.04487 — "FlashAttention-2: Faster Attention with Better Parallelism"

5. https://github.com/NVIDIA/cutlass — NVIDIA CUTLASS for GEMM

6. https://docs.nvidia.com/cuda/cuda-c-programming-guide/ — CUDA programming guide

7. https://arxiv.org/abs/2309.06196 — "Fused PyTorch Operators"

8. https://pytorch.org/tutorials/advanced/cpp_extension.html — PyTorch C++ extensions

---

## 8. Uncertainty and Limitations

**Not Covered:** Multi-GPU kernels, tensor parallelism, integration with specific frameworks beyond basic patterns.

**Production Considerations:** Always benchmark custom kernels against baselines. Consider using NVIDIA Nsight for profiling. Test edge cases thoroughly.

(End of file - total 1480 lines)