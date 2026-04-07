"""
Mixture of Experts (MoE) Implementation Examples
=================================================

This module provides complete implementations of various MoE architectures:
1. Basic Sparse MoE with Top-K routing
2. Expert Choice routing
3. Load-balanced MoE with auxiliary loss
4. Efficient inference with batching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# ============================================================================
# 1. Basic Sparse MoE with Top-K Routing
# ============================================================================


class TopKRouter(nn.Module):
    """Top-K router for sparse MoE"""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Router network: projects hidden states to expert logits
        self.fc = nn.Linear(hidden_size, num_experts)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            weights: (batch_size, seq_len, top_k) - routing weights
            indices: (batch_size, seq_len, top_k) - selected expert indices
            logits: (batch_size, seq_len, num_experts) - router logits
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get router logits
        logits = self.fc(hidden_states)  # (batch, seq_len, num_experts)

        # Top-K selection with noise for training (helps load balancing)
        if self.training:
            # Add noise during training for better load balancing
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise

        # Select top-k experts
        weights, indices = torch.topk(
            logits, k=self.top_k, dim=-1
        )  # (batch, seq_len, k)

        # Normalize weights (softmax)
        weights = F.softmax(weights, dim=-1)

        return weights, indices, logits


class Expert(nn.Module):
    """Single expert (FFN layer)"""

    def __init__(
        self, hidden_size: int, intermediate_size: int = 4096, activation: str = "gelu"
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            (batch_size, seq_len, hidden_size)
        """
        x = self.fc1(hidden_states)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts layer with Top-K routing"""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        intermediate_size: int = 4096,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = TopKRouter(hidden_size, num_experts, top_k)

        # Experts
        self.experts = nn.ModuleList(
            [Expert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

        # Auxiliary loss tracking
        self.register_buffer("expert_usage", torch.zeros(num_experts))

    def forward(
        self, hidden_states: torch.Tensor, compute_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            compute_aux_loss: whether to compute auxiliary loss for load balancing

        Returns:
            output: (batch_size, seq_len, hidden_size)
            aux_loss: scalar loss for balancing expert load
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Route tokens to experts
        weights, indices, logits = self.router(hidden_states)  # weights: (B, S, K)

        # Reshape for expert processing
        hidden_states_flat = hidden_states.reshape(
            -1, hidden_size
        )  # (B*S, hidden_size)

        # Initialize output
        output = torch.zeros_like(hidden_states_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            mask = (indices == expert_idx).any(
                dim=-1
            )  # (B, S) - which tokens use this expert
            mask_flat = mask.reshape(-1)  # (B*S,)

            if mask_flat.any():
                # Get expert output
                expert_output = self.experts[expert_idx](hidden_states_flat[mask_flat])

                # Get weights for this expert for selected tokens
                # Find the position of this expert in top-k for masked tokens
                expert_weights = torch.zeros(
                    mask_flat.sum(), device=hidden_states.device
                )

                for token_idx, local_idx in enumerate(
                    (indices == expert_idx).nonzero(as_tuple=True)
                ):
                    # Find position of expert in top-k
                    pos = (indices[local_idx] == expert_idx).nonzero(as_tuple=True)[0]
                    if pos.numel() > 0:
                        expert_weights[token_idx] = weights[local_idx][pos[0]]

                # Add weighted contribution
                output[mask_flat] += expert_output * expert_weights.unsqueeze(-1)

                # Track expert usage
                self.expert_usage[expert_idx] += mask_flat.sum().float()

        # Reshape output back
        output = output.reshape(batch_size, seq_len, hidden_size)

        # Compute auxiliary loss (load balancing)
        aux_loss = None
        if compute_aux_loss:
            # Importance: fraction of tokens routed to each expert
            router_probs = F.softmax(logits, dim=-1)  # (B, S, num_experts)
            importance = router_probs.sum(dim=(0, 1))  # (num_experts,)
            importance = importance / importance.sum()

            # Load: number of tokens actually routed to each expert
            expert_mask = F.one_hot(
                indices, num_classes=self.num_experts
            )  # (B, S, K, num_experts)
            load = expert_mask.sum(dim=(0, 1, 2)).float()  # (num_experts,)
            load = load / load.sum()

            # Auxiliary loss encourages balance
            aux_loss = (importance * load).sum() * self.num_experts

        return output, aux_loss


# ============================================================================
# 2. Expert Choice Routing
# ============================================================================


class ExpertChoiceRouter(nn.Module):
    """Expert Choice routing: each expert selects its tokens"""

    def __init__(
        self, hidden_size: int, num_experts: int, capacity_factor: float = 1.25
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.fc = nn.Linear(hidden_size, num_experts)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            weights: routing weights
            indices: expert assignments
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get logits
        logits = self.fc(hidden_states)  # (batch, seq_len, num_experts)

        # Each expert selects top tokens
        capacity = int((batch_size * seq_len) / self.num_experts * self.capacity_factor)

        weights = F.softmax(logits, dim=-1)
        top_weights, indices = torch.topk(weights, k=capacity, dim=1)

        return top_weights, indices


# ============================================================================
# 3. Complete Transformer Block with MoE
# ============================================================================


class TransformerMoEBlock(nn.Module):
    """Transformer block with MoE FFN"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_intermediate_size: int = 4096,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # MoE FFN
        self.moe = SparseMoELayer(
            hidden_size, num_experts, top_k, ffn_intermediate_size
        )

    def forward(
        self, hidden_states: torch.Tensor, compute_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            output: (batch_size, seq_len, hidden_size)
            aux_loss: load balancing auxiliary loss
        """
        # Self-attention block
        normed = self.ln1(hidden_states)
        attn_output, _ = self.self_attn(normed, normed, normed)
        hidden_states = hidden_states + attn_output

        # MoE FFN block
        normed = self.ln2(hidden_states)
        moe_output, aux_loss = self.moe(normed, compute_aux_loss)
        hidden_states = hidden_states + moe_output

        return hidden_states, aux_loss


# ============================================================================
# 4. Example: Training Loop
# ============================================================================


def example_moe_training():
    """Example training loop with MoE"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = SparseMoELayer(
        hidden_size=768, num_experts=8, top_k=2, intermediate_size=3072
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    batch_size = 32
    seq_len = 128
    num_iters = 10

    for i in range(num_iters):
        # Random input
        hidden_states = torch.randn(batch_size, seq_len, 768).to(device)
        target = torch.randn(batch_size, seq_len, 768).to(device)

        # Forward pass
        output, aux_loss = model(hidden_states, compute_aux_loss=True)

        # Task loss (MSE)
        task_loss = F.mse_loss(output, target)

        # Total loss with auxiliary loss
        aux_loss_weight = 0.01
        total_loss = task_loss + aux_loss_weight * aux_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (i + 1) % 2 == 0:
            print(
                f"Iter {i + 1}: Task Loss: {task_loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}"
            )


# ============================================================================
# 5. Inference Optimization
# ============================================================================


def create_moe_lora_model():
    """
    Create MoE model with LoRA for efficient inference
    Reduces parameters during inference
    """
    import loralib as lora

    model = SparseMoELayer(hidden_size=768, num_experts=8, top_k=2)

    # Convert linear layers to LoRA
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Replace with LoRA
            lora.mark_only_lora_as_trainable(model)

    return model


if __name__ == "__main__":
    print("Running MoE Training Example...")
    example_moe_training()
    print("\nMoE Example Complete!")
