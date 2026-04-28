# State Space Models: Mamba and S4

## Problem Statement

Transformers, while powerful, face fundamental limitations in handling long sequences due to their O(n²) attention complexity and quadratic memory requirements in sequence length. For tasks requiring processing of very long contexts (Genomics, Long document understanding, Time series forecasting, Audio processing), traditional transformers become computationally prohibitive.

State Space Models (SSMs) offer an alternative approach that achieves the seemingly impossible: processing sequences in O(n) linear time with respect to sequence length while maintaining the ability to capture long-range dependencies. Models like Mamba (State Space Model architecture with selective state spaces) and S4 (Structured State Space Sequence model) have achieved state-of-the-art results on long-range tasks while being significantly more efficient than transformers.

This skill covers understanding the theoretical foundations of SSMs, implementing S4 and Mamba architectures, handling the discretization step that bridges continuous and discrete domains, and applying these models to various sequence modeling tasks.

## Theory & Fundamentals

### State Space Models: Continuous-Time Representation

An SSM maps a 1-D input signal u(t) to a 1-D output signal y(t) through a latent state h(t):

```
dx/dt = Ah(t) + Bu(t)       # State equation
y(t) = Ch(t) + Du(t)        # Output equation
```

Where:
- A ∈ ℝ^(N×N) is the state matrix (controls how state evolves)
- B ∈ ℝ^(N×1) is the input matrix
- C ∈ ℝ^(1×N) is the output matrix
- D ∈ ℝ is the feedthrough (skip) matrix

### Discretization

The continuous equations must be discretized for digital computation. The bilinear method:

```
h_t = Ah_{t-1} + Bu_t
y_t = Ch_t + Du_t

where:
A = (I - Δ/2 · A)^-1 (I + Δ/2 · A)
B = (I - Δ/2 · A)^-1 · Δ · B

Δ is the step size (token/sequence element interval)
```

### The HiPPO Framework

HiPPO (High-order Polynomial Projection Operators) provides a principled way to initialize the state matrix A:

$$A_{nk} = \begin{cases} (-1)^{n+k} \cdot (2n+1)^{1/2} \cdot (2k+1)^{1/2} & \text{if } n > k \\ (2n+1)^{1/2} \cdot (2k+1)^{1/2} & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}$$

This initialization ensures the SSM can represent all polynomials up to order N.

### Mamba: Selective State Spaces

Mamba's key innovation is the selective scan mechanism. Unlike S4 where A, B, C are input-independent, Mamba makes them input-dependent:

$$B_t = \text{Linear}_B(x_t)$$
$$C_t = \text{Linear}_C(x_t)$$
$$\Delta_t = \text{tau}(\text{Linear}_{\Delta}(x_t))$$

where tau is a softplus activation ensuring Δ > 0.

The selection mechanism allows the model to dynamically decide:
- Which inputs are important (via B selection)
- What information to preserve in state (via A selection)
- What to output (via C selection)

### Linear Time Invariance vs Selection

```
S4 (Linear Time Invariant):
  A, B, C are CONSTANT for all timesteps
  → Efficient via convolution
  → Cannot selectively filter information

Mamba (Selection):
  A_t, B_t, C_t vary per timestep
  → Requires sequential scan (O(n))
  → Can selectively attend to relevant information
```

## Implementation Patterns

### Pattern 1: Core SSM Discretization and Scan

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class Discretization(nn.Module):
    """
    Discretizes continuous SSM parameters using bilinear transform.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize SSM parameters.
        
        Args:
            delta: Step sizes [batch, seq_len] or scalar
            A: State matrix [state_dim, state_dim]
            B: Input matrix [state_dim, input_dim]
        
        Returns:
            Ab: Discretized A matrix
            Bb: Discretized B matrix
        """
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        
        I_minus_half_A = I - delta.unsqueeze(-1) * A * 0.5
        I_plus_half_A = I + delta.unsqueeze(-1) * A * 0.5
        
        Ab = torch.linalg.solve(I_minus_half_A, I_plus_half_A)
        
        Bb = torch.linalg.solve(
            I_minus_half_A,
            delta.unsqueeze(-1) * B
        )
        
        return Ab, Bb


class SelectiveScan(nn.Module):
    """
    Implements the selective scan algorithm for Mamba.
    This is the core computation that allows input-dependent SSM parameters.
    """
    
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
    
    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        delta: torch.Tensor,
        D: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute selective scan.
        
        Args:
            x: Input [batch, seq_len, input_dim]
            A: State matrix [state_dim, state_dim]
            B: Input matrix [batch, seq_len, state_dim]
            C: Output matrix [batch, seq_len, state_dim]
            delta: Time step [batch, seq_len]
            D: Skip connection [input_dim]
        
        Returns:
            y: Output [batch, seq_len]
        """
        batch, seq_len, input_dim = x.shape
        
        A = A.unsqueeze(0).expand(batch, -1, -1)
        
        Ab, Bb = self._discretize(delta, A, B)
        
        h = torch.zeros(
            batch, self.state_dim, self.state_dim,
            device=x.device, dtype=x.dtype
        )
        
        outputs = []
        for t in range(seq_len):
            h = torch.einsum('bij,bjk->bik', Ab[:, t], h) + \
                torch.einsum('bi,bij->bj', x[:, t], Bb[:, t])
            
            y_t = torch.einsum('bj,bj->b', h.squeeze(-1), C[:, t])
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        
        if D is not None:
            y = y + F.linear(x, D)
        
        return y
    
    def _discretize(
        self,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize using bilinear transform."""
        I = torch.eye(self.state_dim, device=A.device, dtype=A.dtype)
        
        Ab_list = []
        Bb_list = []
        
        for t in range(delta.shape[1]):
            d = delta[:, t].unsqueeze(-1).unsqueeze(-1)
            
            I_minus_half_A = I - d * A * 0.5
            I_plus_half_A = I + d * A * 0.5
            
            Ab_t = torch.linalg.solve(I_minus_half_A, I_plus_half_A)
            Bb_t = torch.linalg.solve(I_minus_half_A, d * B[:, t:t+1].transpose(-2, -1))
            
            Ab_list.append(Ab_t)
            Bb_list.append(Bb_t.squeeze(-1))
        
        Ab = torch.stack(Ab_list, dim=1)
        Bb = torch.stack(Bb_list, dim=1)
        
        return Ab, Bb


class ParallelScan(nn.Module):
    """
    Parallel prefix sum (scan) for efficient computation.
    Uses the Kronecker product formulation for parallelization.
    """
    
    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute parallel scan.
        
        Args:
            x: Input [batch, seq_len, state_dim]
            A: Recurrence matrix [batch, seq_len, state_dim, state_dim]
            B: Input matrix [batch, seq_len, state_dim]
        
        Returns:
            output: [batch, seq_len, state_dim]
        """
        batch, seq_len, state_dim = x.shape
        
        N = seq_len
        log_N = int(math.ceil(math.log2(N)))
        
        a = A
        b = B
        
        x_local = x
        
        for k in range(log_N):
            stride = 2 ** k
            
            prefix_mask = torch.arange(N, device=x.device).unsqueeze(0)
            prefix_mask = (prefix_mask % (2 * stride)) >= stride
            
            if prefix_mask.any():
                x_left = x_local * prefix_mask.unsqueeze(-1).float()
                x_right = x_local * (~prefix_mask).unsqueeze(-1).float()
                
                a_left = a * prefix_mask.unsqueeze(-1).unsqueeze(-1).float()
                a_right = a * (~prefix_mask).unsqueeze(-1).unsqueeze(-1).float()
                
                a_combined = torch.einsum('blij,bljk->blik', a_right, a_left)
                b_combined = torch.einsum('blij,blj->bli', a_right, b_left) + b_right
                
                x_local = x_left + torch.einsum('blij,blj->bli', a_left, x_right) + x_right
                a = a_combined
                b = b_combined
        
        return x_local
```

### Pattern 2: Mamba Block Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class MambaBlock(nn.Module):
    """
    Single Mamba block.
    
    Key differences from standard attention:
    - Input-dependent SSM parameters (selection mechanism)
    - No softmax attention
    - O(n) instead of O(n²)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank != "auto" else math.ceil(d_model / 16)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            reverse: If True, reverse the sequence (for BiMamba)
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        if reverse:
            x = torch.flip(x, dims=[1])
        
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_inner.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        
        x_ssm = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_ssm)
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)
        
        y = self._selective_scan(x_ssm, dt, self.A_log, B, C, self.D)
        
        y = y * F.silu(z)
        
        output = self.out_proj(y)
        
        if reverse:
            output = torch.flip(output, dims=[1])
        
        return output
    
    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """
        Core selective scan operation.
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        A = -torch.exp(A.float())
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=torch.float)
        ys = []
        
        for i in range(seq_len):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            
            y = torch.einsum('bdn,bn->bd', h, C[:, i].float())
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        
        y = y + u * D
        
        return y


class MambaStack(nn.Module):
    """
    Stack of Mamba blocks with residual connections.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        
        return self.norm(x)
```

### Pattern 3: S4 (Structured State Space Sequence) Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

class S4Block(nn.Module):
    """
    S4 (Structured State Space Sequence) layer.
    
    Key components:
    - HiPPO initialization for state matrix A
    - DPLR (Diagonal + Low-Rank) parameterization
    - Convolutional computation for efficiency
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
        transposed: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.transposed = transposed
        
        self.input_proj = nn.Linear(d_model, d_model * 2)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.D = nn.Parameter(torch.randn(d_model))
        
        A = self._init_HiPPO(d_model, d_state)
        
        self.A_re = nn.Parameter(A.real.float())
        self.A_im = nn.Parameter(A.imag.float())
        
        log_A_im = torch.log(-self.A_im.float())
        self.register_buffer('log_A_im', log_A_im)
        
        self.dropout = nn.Dropout(dropout)
        
        self._setup_conv(d_model)
    
    def _init_HiPPO(self, d_model: int, d_state: int) -> torch.Tensor:
        """Initialize A matrix using HiPPO-LegS projection."""
        A = torch.zeros(d_model, d_state, dtype=torch.cfloat)
        
        for n in range(d_model):
            for k in range(d_state):
                if n > k:
                    A[n, k] = (-1) ** (n - k) * math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1)
                elif n == k:
                    A[n, k] = math.sqrt(2 * n + 1)
        
        return A
    
    def _setup_conv(self, d_model: int):
        """Setup conv kernel for efficient computation."""
        self.conv_kernel_size = 4
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_kernel_size - 1,
            groups=d_model
        )
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            x: [batch, seq_len, d_model] if transposed else [batch, seq_len, d_model]
            state: Optional recurrent state
        
        Returns:
            output: [batch, seq_len, d_model]
            new_state: New recurrent state
        """
        batch, seq_len, d_model = x.shape
        
        xz = self.input_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        x_conv = self.conv(x_inner.transpose(1, 2)).transpose(1, 2)
        x_conv = x_conv[:, :seq_len, :]
        
        x_ssm = F.silu(x_conv)
        
        y = self._compute_s4(x_ssm)
        
        y = y + x_ssm * self.D
        
        y = y * F.silu(z)
        
        output = self.output_proj(y)
        output = self.dropout(output)
        
        return output, None
    
    def _compute_s4(self, x: torch.Tensor) -> torch.Tensor:
        """Compute S4 operation using convolution form."""
        batch, seq_len, d_model = x.shape
        
        A_complex = self.A_re + 1j * self.A_im
        
        C = torch.ones(d_model, self.d_state, device=x.device, dtype=torch.cfloat)
        
        kernel = self._compute_conv_kernel(A_complex, C, seq_len)
        
        y = F.conv1d(
            x.transpose(1, 2),
            kernel.unsqueeze(1),
            padding=seq_len - 1
        ).transpose(1, 2)
        
        return y[:, -seq_len:, :].real
    
    def _compute_conv_kernel(
        self,
        A: torch.Tensor,
        C: torch.Tensor,
        length: int
    ) -> torch.Tensor:
        """Compute convolution kernel via generating function."""
        kernel_len = min(length, 256)
        
        kernels = []
        for d in range(d_model):
            A_d = A[d]
            
            C_d = C[d]
            
            A_powers = torch.stack([
                torch.matrix_power(torch.diag(A_d.exp()) - torch.eye(self.d_state), k)
                for k in range(kernel_len)
            ])
            
            k_d = torch.einsum('kij,j->ki', A_powers, C_d)
            
            kernels.append(k_d)
        
        return torch.stack(kernels)
```

### Pattern 4: Bidirectional Mamba for Encoding

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba block for encoding tasks.
    Runs forward and backward Mamba and combines the representations.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        
        self.forward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.backward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model] - bidirectionally enhanced
        """
        forward_out = self.forward_mamba(x, reverse=False)
        
        backward_out = self.backward_mamba(x, reverse=True)
        
        combined = torch.cat([forward_out, backward_out], dim=-1)
        
        fused = self.fusion(combined)
        
        return self.norm(fused)


class MambaForSequenceClassification(nn.Module):
    """
    Mamba model for sequence classification tasks.
    """
    
    def __init__(
        self,
        d_model: int,
        n_classes: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        tie_weights: bool = True
    ):
        super().__init__()
        
        self.embedding = nn.Embedding( vocab_size=256, d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.classifier = nn.Linear(d_model, n_classes)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            logits: [batch, n_classes]
        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = x + layer(x)
        
        x = self.norm(x)
        
        pooled = x.mean(dim=1)
        
        logits = self.classifier(pooled)
        
        return logits


class MambaForTimeSeries(nn.Module):
    """
    Mamba model for time series forecasting.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_pred_steps: int = 1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(1, d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.output_proj = nn.Linear(d_model, n_pred_steps)
    
    def forward(
        self,
        x: torch.Tensor,
        pred_len: int = 1
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] - time series values
            pred_len: Number of steps to predict
        
        Returns:
            predictions: [batch, pred_len]
        """
        x = x.unsqueeze(-1)
        
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = x + layer(x)
        
        x = self.norm(x)
        
        last_hidden = x[:, -1, :]
        
        predictions = self.output_proj(last_hidden)
        
        return predictions.squeeze(-1)
```

### Pattern 5: Hybrid Transformer-SSM Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class HybridMambaAttentionBlock(nn.Module):
    """
    Combines Mamba's efficient long-range modeling with attention's
    selective focus capability.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(expand * d_model)),
            nn.GELU(),
            nn.Linear(int(expand * d_model), d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.mamba_scale = 1.0
        self.attention_scale = 0.5
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hybrid forward pass combining Mamba and attention.
        
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: Optional attention mask
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        residual = x
        
        mamba_out = self.mamba(x)
        x = self.norm1(residual + self.mamba_scale * mamba_out)
        
        residual = x
        
        q = self.q_proj(x).view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.shape)
        attn_output = self.out_proj(attn_output)
        
        x = self.norm2(residual + self.attention_scale * attn_output)
        
        residual = x
        ffn_output = self.ffn(x)
        x = self.norm3(residual + ffn_output)
        
        return x


class MambaLMHead(nn.Module):
    """
    Mamba-based language model head with vocabulary projection.
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.tie_weights()
    
    def tie_weights(self):
        """Tie input and output embeddings."""
        self.lm_head.weight = self.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            input_ids: [batch, seq_len]
            labels: Optional labels for language modeling loss
        
        Returns:
            Dictionary with logits and loss
        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = x + layer(x)
        
        x = self.norm(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        
        return {
            "logits": logits,
            "loss": loss
        }
```

## Framework Integration

### Integration with HuggingFace

```python
from transformers import PreTrainedModel, Config

class MambaConfig(Config):
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 768)
        self.n_layers = kwargs.get("n_layers", 12)
        self.d_state = kwargs.get("d_state", 16)
        self.d_conv = kwargs.get("d_conv", 4)
        self.expand = kwargs.get("expand", 2)
        self.vocab_size = kwargs.get("vocab_size", 32000)
```

### Integration with PyTorch Lightning

```python
import pytorch_lightning as pl

class MambaLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = MambaLMHead(**config)
```

## Common Pitfalls

### Pitfall 1: Incorrect Discretization

**Problem**: Using wrong discretization parameters causes instability.

**Solution**: Ensure Δ > 0 and A is properly scaled:
```python
# Always use softplus for delta to ensure positivity
delta = F.softplus(dt_proj(x))
```

### Pitfall 2: State Dimension Too Small

**Problem**: Small state dimension limits model's memory capacity.

**Solution**: Match state dimension to sequence length requirements:
```
For sequences up to 4096 tokens: d_state >= 32
For sequences up to 16384 tokens: d_state >= 64
For sequences up to 65536 tokens: d_state >= 128
```

### Pitfall 3: Not Using Reversible Layers

**Problem**: Memory usage grows linearly with model depth.

**Solution**: Use checkpointing or reversible architecture:
```python
# For long sequences, use gradient checkpointing
model.gradient_checkpointing_enable()
```

## Research References

1. **Gu et al. (2021)** - "Efficiently Modeling Long Sequences with Structured State Spaces" - S4 original paper.

2. **Gu & Dao (2023)** - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" - Mamba.

3. **Voelker et al. (2019)** - "Legendre Memory Units" - Continuous-time memory foundations.

4. **Gupta et al. (2022)** - "Beyond Transformers" - SSM-Transformer hybrid architectures.

5. **Poli et al. (2023)** - "Hyena Hierarchy" - Longer-range dependencies with sub-quadratic attention.

6. **Orvieto et al. (2023)** - "Resurrecting Recurrent Neural Networks" - Modern RNN techniques.

7. **Jelassi et al. (2024)** - "Towards Foundation Models for Time Series" - SSMs for time series.

8. **dao et al. (2024)** - "Mamba-2" - Improved architecture and implementation.

9. **Furuta et al. (2023)** - "In-context Learning with SSMs" - In-context learning with SSMs.

10. **Wolf et al. (2024)** - "Mamba meets Lightning" - Efficient training of Mamba models.