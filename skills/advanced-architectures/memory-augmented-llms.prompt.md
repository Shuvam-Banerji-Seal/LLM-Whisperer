# Memory-Augmented LLMs

## Problem Statement

Large language models suffer from a fundamental limitation: they have fixed internal memory (model weights) and limited context window. This creates challenges for tasks requiring:
- Remembering information across very long conversations
- Accessing specific facts from large knowledge bases
- Maintaining consistent state over extended interactions
- Learning new information after deployment without retraining

Memory-augmented neural networks address these limitations by combining neural networks with external memory systems. Drawing inspiration from the Neural Turing Machine (NTM) and Memory Networks architectures, these systems enable LLMs to read from and write to external storage, significantly expanding their effective memory capacity and enabling persistent knowledge.

This skill covers understanding memory-augmented architectures, implementing reading and writing mechanisms, designing memory addressing schemes, building hybrid memory systems, and applying these to practical LLM applications.

## Theory & Fundamentals

### Memory-Augmented Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Controller Network (LLM)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Read Head   │  │  Write Head  │  │  Controller  │           │
│  │              │  │              │  │   Output     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘           │
│         │                 │                                       │
│         ▼                 ▼                                       │
│  ┌─────────────────────────────────────────┐                     │
│  │           External Memory                │                     │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┐ │                     │
│  │  │ M0  │ M1  │ M2  │ M3  │ ... │ Mn  │ │                     │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┘ │                     │
│  │  [Address] [Address] [Address]          │                     │
│  └─────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Addressing Mechanisms

**Content-Based Addressing**: Read/write based on similarity to content
$$w_i = \frac{\exp(\beta \cdot \text{sim}(k, M_i))}{\sum_j \exp(\beta \cdot \text{sim}(k, M_j))}$$

**Location-Based Addressing**: Direct access by position

**Dynamic Addressing**: Combination of content and location

### Write Operations

**Least Recently Used (LRU)**: Evict least recently accessed memory
**Least Frequently Used (LFU)**: Evict least frequently accessed memory
**Time Decay**: Weaken old memories over time

### Memory Networks vs Neural Turing Machines

```
Neural Turing Machine:
├── Controller: Neural network (often LSTM)
├── Memory: N × M matrix (N locations, M features)
├── Read heads: Content + location addressing
├── Write heads: Erase + add operations
└── Operations: Sequential (step-by-step)

Memory Networks:
├── Controller: Classification/embedding network
├── Memory: Hop-wise attention
├──hop operations: Multiple reasoning hops
└── Operations: Can be parallelized
```

## Implementation Patterns

### Pattern 1: Neural Turing Machine Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

class NTMReadHead(nn.Module):
    """
    Read head for Neural Turing Machine.
    Combines content-based and location-based addressing.
    """
    
    def __init__(
        self,
        memory_size: int,
        memory_vector_dim: int,
        controller_output_dim: int
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        
        self.key_strength = nn.Linear(controller_output_dim, 1)
        self.content_weighting = nn.Linear(memory_vector_dim, memory_vector_dim)
        
        self.shift_weighting = nn.Linear(controller_output_dim, 3)
        self.shift_strength = nn.Parameter(torch.tensor(1.0))
        
        self.gate = nn.Linear(controller_output_dim, 1)
        
        self.sharpening = nn.Linear(controller_output_dim, 1)
    
    def forward(
        self,
        memory: torch.Tensor,
        controller_state: torch.Tensor,
        prev_read_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            memory: [batch, memory_size, memory_vector_dim]
            controller_state: [batch, controller_output_dim]
            prev_read_weight: [batch, memory_size]
        
        Returns:
            read_content: [batch, memory_vector_dim]
            read_weight: [batch, memory_size]
        """
        content_key = self.content_weighting(controller_state)
        content_key = content_key.unsqueeze(1)
        
        similarity = torch.cosine_similarity(
            memory + 1e-8,
            content_key.expand_as(memory),
            dim=-1
        )
        
        beta = torch.sigmoid(self.key_strength(controller_state)) * 10 + 1
        content_weights = F.softmax(similarity * beta, dim=-1)
        
        gamma = torch.sigmoid(self.gate(controller_state)) * 10 + 1
        
        shift_weights = self.shift_weighting(controller_state)
        shift_weights = F.softmax(shift_weights, dim=-1)
        
        interpolated = controller_state[:, :1].squeeze(-1)
        read_weight = interpolated * content_weights + (1 - interpolated) * prev_read_weight
        
        read_weight = self._circular_convolution(read_weight, shift_weights)
        
        read_weight = read_weight ** gamma
        
        read_weight = read_weight / (read_weight.sum(dim=-1, keepdim=True) + 1e-8)
        
        read_content = torch.bmm(
            read_weight.unsqueeze(1),
            memory
        ).squeeze(1)
        
        return read_content, read_weight
    
    def _circular_convolution(
        self,
        weights: torch.Tensor,
        shift: torch.Tensor
    ) -> torch.Tensor:
        """Perform circular convolution for location-based addressing."""
        batch_size, memory_size = weights.shape
        
        max_shift = memory_size // 2
        
        shift_values = torch.arange(-max_shift, max_shift + 1, device=weights.device)
        
        shifted_weights = []
        for shift_prob in shift.chunk(3, dim=-1):
            shift_idx = torch.argmax(shift_prob)
            shift_value = shift_values[shift_idx]
            
            if shift_value == 0:
                shifted_weights.append(weights)
            elif shift_value > 0:
                shifted = torch.cat([weights[:, -shift_value:], weights[:, :-shift_value]], dim=-1)
                shifted_weights.append(shifted)
            else:
                shifted = torch.cat([weights[:, abs(shift_value):], weights[:, :abs(shift_value)]], dim=-1)
                shifted_weights.append(shifted)
        
        result = torch.stack(shifted_weights, dim=-1).sum(dim=-1)
        
        return result


class NTMWriteHead(nn.Module):
    """
    Write head for Neural Turing Machine.
    Implements erase and add operations.
    """
    
    def __init__(
        self,
        memory_size: int,
        memory_vector_dim: int,
        controller_output_dim: int
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        
        self.erase_linear = nn.Linear(controller_output_dim, memory_vector_dim)
        self.add_linear = nn.Linear(controller_output_dim, memory_vector_dim)
        
        self.key_strength = nn.Linear(controller_output_dim, 1)
        self.content_weighting = nn.Linear(memory_vector_dim, memory_vector_dim)
        
        self.gate = nn.Linear(controller_output_dim, 1)
    
    def forward(
        self,
        memory: torch.Tensor,
        controller_state: torch.Tensor,
        prev_write_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            memory: [batch, memory_size, memory_vector_dim]
            controller_state: [batch, controller_output_dim]
            prev_write_weight: [batch, memory_size]
        
        Returns:
            new_memory: [batch, memory_size, memory_vector_dim]
            write_weight: [batch, memory_size]
            erase_vector: [batch, memory_vector_dim]
        """
        content_key = self.content_weighting(controller_state)
        content_key = content_key.unsqueeze(1)
        
        similarity = torch.cosine_similarity(
            memory + 1e-8,
            content_key.expand_as(memory),
            dim=-1
        )
        
        beta = torch.sigmoid(self.key_strength(controller_state)) * 10 + 1
        content_weights = F.softmax(similarity * beta, dim=-1)
        
        gate = torch.sigmoid(self.gate(controller_state))
        write_weight = gate * content_weights + (1 - gate) * prev_write_weight
        
        erase_vector = torch.sigmoid(self.erase_linear(controller_state))
        add_vector = torch.tanh(self.add_linear(controller_state))
        
        erase_signal = write_weight.unsqueeze(-1) * erase_vector.unsqueeze(1)
        erase_signal = torch.clamp(1 - erase_signal, min=0)
        
        new_memory = memory * erase_signal + (write_weight.unsqueeze(-1) * add_vector.unsqueeze(1))
        
        return new_memory, write_weight, erase_vector


class NeuralTuringMachine(nn.Module):
    """
    Complete Neural Turing Machine implementation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        controller_dim: int,
        memory_size: int = 128,
        memory_vector_dim: int = 64,
        num_heads: int = 1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.controller_dim = controller_dim
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        
        self.controller = nn.LSTMCell(input_dim + memory_vector_dim * num_heads, controller_dim)
        
        self.read_heads = nn.ModuleList([
            NTMReadHead(memory_size, memory_vector_dim, controller_dim)
            for _ in range(num_heads)
        ])
        
        self.write_head = NTMWriteHead(memory_size, memory_vector_dim, controller_dim)
        
        self.init_memory = nn.Parameter(
            torch.randn(1, memory_size, memory_vector_dim) * 0.1
        )
        self.init_prev_read_weight = nn.Parameter(torch.zeros(1, memory_size))
        
        self.output_linear = nn.Linear(
            controller_dim + memory_vector_dim * num_heads,
            output_dim
        )
        
        self.num_heads = num_heads
    
    def init_state(self, batch_size: int, device: torch.device):
        """Initialize hidden state and memory."""
        memory = self.init_memory.expand(batch_size, -1, -1).to(device)
        
        h_state = torch.zeros(batch_size, self.controller_dim, device=device)
        c_state = torch.zeros(batch_size, self.controller_dim, device=device)
        
        prev_read_weights = [
            self.init_prev_read_weight.expand(batch_size, -1).to(device)
            for _ in range(self.num_heads)
        ]
        
        prev_write_weight = self.init_prev_read_weight.expand(batch_size, -1).to(device)
        
        return {
            'memory': memory,
            'h': h_state,
            'c': c_state,
            'prev_read_weights': prev_read_weights,
            'prev_write_weight': prev_write_weight
        }
    
    def forward(
        self,
        inputs: torch.Tensor,
        state: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            inputs: [batch, seq_len, input_dim]
            state: Optional previous state
        
        Returns:
            outputs: [batch, seq_len, output_dim]
            final_state: Updated state
        """
        batch_size, seq_len, _ = inputs.shape
        
        if state is None:
            state = self.init_state(batch_size, inputs.device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            
            read_contents = []
            for i, read_head in enumerate(self.read_heads):
                read_content, state['prev_read_weights'][i] = read_head(
                    state['memory'],
                    state['h'],
                    state['prev_read_weights'][i]
                )
                read_contents.append(read_content)
            
            controller_input = torch.cat([x_t] + read_contents, dim=-1)
            
            state['h'], state['c'] = self.controller(
                controller_input,
                (state['h'], state['c'])
            )
            
            state['memory'], state['prev_write_weight'], _ = self.write_head(
                state['memory'],
                state['h'],
                state['prev_write_weight']
            )
            
            output_t = self.output_linear(
                torch.cat([state['h']] + read_contents, dim=-1)
            )
            outputs.append(output_t)
        
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, state
```

### Pattern 2: Memory Network with Hop Operations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class HopOperation(nn.Module):
    """
    Single hop operation in a Memory Network.
    Performs attention over memory based on query.
    """
    
    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.query_transform = nn.Linear(embedding_dim, embedding_dim)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim)
        self.value_transform = nn.Linear(embedding_dim, embedding_dim)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, embedding_dim]
            memory: [batch, memory_size, embedding_dim]
            mask: Optional attention mask
        
        Returns:
            output: [batch, embedding_dim]
            attention_weights: [batch, memory_size]
        """
        Q = self.query_transform(query.unsqueeze(1))
        K = self.key_transform(memory)
        V = self.value_transform(memory)
        
        scores = torch.bmm(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.bmm(attention_weights, V).squeeze(1)
        
        output = self.norm1(query + attended)
        
        output = self.norm2(output + self.ffn(output))
        
        return output, attention_weights.squeeze(1)


class MemoryNetwork(nn.Module):
    """
    Memory Network with multiple hop operations.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_hops: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.hops = nn.ModuleList([
            HopOperation(embedding_dim, dropout)
            for _ in range(num_hops)
        ])
        
        self.output_linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_ids: torch.Tensor,
        question_ids: torch.Tensor,
        num_memory_lines: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input_ids: [batch, seq_len] - input sequence
            memory_ids: [batch, memory_size, memory_seq_len] - memory content
            question_ids: [batch, question_len] - query
            num_memory_lines: Number of actual memory lines
        
        Returns:
            output: [batch, vocab_size]
            attention_weights: List of attention weights from each hop
        """
        batch_size = input_ids.shape[0]
        
        embedded = self.embedding(input_ids)
        embedded = embedded.sum(dim=1) / (embedded.sum(dim=1) != 0).float().clamp(min=1)
        
        memory_mask = torch.arange(
            memory_ids.size(1), device=memory_ids.device
        ).unsqueeze(0).expand(batch_size, -1) >= num_memory_lines.unsqueeze(1)
        
        memory_embedded = self.embedding(memory_ids)
        memory_flat = memory_embedded.view(batch_size, memory_ids.size(1), -1)
        memory_pooled = memory_flat.view(batch_size * memory_ids.size(1), memory_ids.size(2), -1)
        memory_pooled = memory_pooled.sum(dim=1) / (memory_pooled.sum(dim=1) != 0).float().clamp(min=1)
        memory = memory_pooled.view(batch_size, memory_ids.size(1), -1)
        
        attention_weights = []
        x = embedded
        
        for hop in self.hops:
            x, attn_weights = hop(x, memory, memory_mask)
            attention_weights.append(attn_weights)
        
        output = self.output_linear(x)
        
        return output, attention_weights
    
    def predict(self, question_ids: torch.Tensor, memory_ids: torch.Tensor) -> torch.Tensor:
        """Predict answer given question and supporting facts."""
        batch_size = question_ids.shape[0]
        
        question_emb = self.embedding(question_ids).sum(dim=1)
        memory_emb = self.embedding(memory_ids).sum(dim=2)
        
        x = question_emb
        for hop in self.hops:
            x, _ = hop(x, memory_emb)
        
        return self.output_linear(x)
```

### Pattern 3: Differentiable Memory-Augmented Policy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np

class DifferentiableMemory(nn.Module):
    """
    Differentiable memory that supports read, write, and erase operations.
    Based on Differentiable Neural Computer architecture.
    """
    
    def __init__(
        self,
        memory_size: int,
        memory_vector_dim: int,
        num_read_heads: int = 1,
        num_write_heads: int = 1
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        
        self.memory = None
        
        self.usage_vector = None
    
    def init_memory(self, batch_size: int, device: torch.device):
        """Initialize empty memory matrix."""
        self.memory = torch.zeros(
            batch_size, self.memory_size, self.memory_vector_dim,
            device=device
        )
        self.usage_vector = torch.zeros(
            batch_size, self.memory_size,
            device=device
        )
    
    def read(
        self,
        read_keys: torch.Tensor,
        read_strengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using content-based addressing.
        
        Args:
            read_keys: [batch, num_read_heads, memory_vector_dim]
            read_strengths: [batch, num_read_heads]
        
        Returns:
            read_content: [batch, num_read_heads, memory_vector_dim]
            read_weights: [batch, num_read_heads, memory_size]
        """
        read_weights = []
        read_contents = []
        
        for k, beta in zip(read_keys.chunk(self.num_read_heads, dim=1),
                          read_strengths.chunk(self.num_read_heads, dim=1)):
            similarity = torch.cosine_similarity(
                self.memory + 1e-8,
                k.expand(-1, self.memory_size, -1),
                dim=-1
            )
            
            weights = F.softmax(similarity * beta.squeeze(-1), dim=-1)
            
            content = torch.bmm(weights.unsqueeze(1), self.memory).squeeze(1)
            
            read_weights.append(weights)
            read_contents.append(content)
        
        read_weights = torch.stack(read_weights, dim=1)
        read_contents = torch.stack(read_contents, dim=1)
        
        return read_contents, read_weights
    
    def write(
        self,
        write_key: torch.Tensor,
        write_strength: torch.Tensor,
        erase_vector: torch.Tensor,
        add_vector: torch.Tensor
    ):
        """
        Write to memory using content-based addressing.
        
        Args:
            write_key: [batch, memory_vector_dim]
            write_strength: [batch, 1]
            erase_vector: [batch, memory_vector_dim]
            add_vector: [batch, memory_vector_dim]
        """
        similarity = torch.cosine_similarity(
            self.memory + 1e-8,
            write_key.unsqueeze(1).expand(-1, self.memory_size, -1),
            dim=-1
        )
        
        write_weights = F.softmax(similarity * write_strength.squeeze(-1), dim=-1)
        write_weights = write_weights.unsqueeze(-1)
        
        erase_signal = 1 - torch.bmm(
            write_weights,
            erase_vector.unsqueeze(1)
        ).squeeze(-1)
        
        self.memory = self.memory * erase_signal
        
        add_signal = write_weights * add_vector.unsqueeze(1)
        self.memory = self.memory + add_signal
        
        self.usage_vector = self.usage_vector + write_weights.squeeze(-1)
    
    def erase(
        self,
        erase_key: torch.Tensor,
        erase_strength: torch.Tensor
    ):
        """
        Erase memory locations based on key.
        """
        similarity = torch.cosine_similarity(
            self.memory + 1e-8,
            erase_key.unsqueeze(1).expand(-1, self.memory_size, -1),
            dim=-1
        )
        
        erase_weights = F.softmax(similarity * erase_strength.squeeze(-1), dim=-1)
        
        erase_signal = 1 - erase_weights.unsqueeze(-1)
        self.memory = self.memory * erase_signal
        self.usage_vector = self.usage_vector * erase_signal.squeeze(-1)
    
    def free_least_used(self, num_locations: int):
        """Free the least used memory locations."""
        for _ in range(num_locations):
            least_used_idx = torch.argmin(self.usage_vector, dim=-1)
            
            for batch_idx in range(self.memory.size(0)):
                self.memory[batch_idx, least_used_idx[batch_idx]] = 0
                self.usage_vector[batch_idx, least_used_idx[batch_idx]] = 0
    
    def get_usage_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_usage = self.usage_vector.sum().item()
        locations_used = (self.usage_vector > 0).sum().item()
        
        return {
            "total_usage": total_usage,
            "locations_used": locations_used,
            "usage_percentage": locations_used / self.memory_size * 100,
            "avg_usage_per_used": total_usage / max(locations_used, 1)
        }


class MemoryAugmentedLM(nn.Module):
    """
    Language model augmented with external differentiable memory.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        memory_size: int = 256,
        memory_vector_dim: int = 64
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim + memory_vector_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        
        self.memory = DifferentiableMemory(
            memory_size,
            memory_vector_dim
        )
        
        self.key_gen = nn.Linear(hidden_dim, memory_vector_dim)
        self.strength_gen = nn.Linear(hidden_dim, 1)
        self.erase_gen = nn.Linear(hidden_dim, memory_vector_dim)
        self.add_gen = nn.Linear(hidden_dim, memory_vector_dim)
        
        self.read_key_gen = nn.Linear(hidden_dim, memory_vector_dim)
        self.read_strength_gen = nn.Linear(hidden_dim, 1)
        
        self.output = nn.Linear(hidden_dim + memory_vector_dim, vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            input_ids: [batch, seq_len]
            memory_init: Optional initial memory state
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            memory_state: Updated memory tensor
        """
        batch_size, seq_len = input_ids.shape
        
        self.memory.init_memory(batch_size, input_ids.device)
        
        if memory_init is not None:
            self.memory.memory = memory_init
        
        embeddings = self.embedding(input_ids)
        
        outputs = []
        hidden = None
        
        for t in range(seq_len):
            read_keys = torch.sigmoid(self.read_key_gen(
                hidden[0][-1] if hidden else torch.zeros(batch_size, self.lstm.hidden_size, device=input_ids.device)
            )).unsqueeze(1)
            
            read_strengths = torch.sigmoid(self.read_strength_gen(
                hidden[0][-1] if hidden else torch.zeros(batch_size, self.lstm.hidden_size, device=input_ids.device)
            )).unsqueeze(-1) * 10 + 1
            
            read_contents, read_weights = self.memory.read(read_keys, read_strengths)
            
            lstm_input = torch.cat([
                embeddings[:, t, :],
                read_contents.squeeze(1)
            ], dim=-1).unsqueeze(1)
            
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            
            write_key = torch.tanh(self.key_gen(hidden[0][-1]))
            write_strength = torch.sigmoid(self.strength_gen(hidden[0][-1])) * 10 + 1
            erase_vec = torch.sigmoid(self.erase_gen(hidden[0][-1]))
            add_vec = torch.tanh(self.add_gen(hidden[0][-1]))
            
            self.memory.write(write_key, write_strength, erase_vec, add_vec)
            
            output_t = self.output(
                torch.cat([hidden[0][-1], read_contents.squeeze(1)], dim=-1)
            )
            outputs.append(output_t)
        
        logits = torch.stack(outputs, dim=1)
        
        return logits, self.memory.memory.detach()


class EpisodicMemoryModule(nn.Module):
    """
    Episodic memory that stores and retrieves experience tuples.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        memory_size: int = 1000,
        encoding_dim: int = 128
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.encoding_dim = encoding_dim
        
        self.query_encoder = nn.Linear(embedding_dim * 3, encoding_dim)
        self.memory_encoder = nn.Linear(embedding_dim * 3, encoding_dim)
        
        self.attention = nn.MultiheadAttention(encoding_dim, num_heads=4)
        
        self.timestamps = []
        self.episodes = []
        self.importance_scores = []
        
        self.register_buffer('time_encoding',
            self._create_time_encoding(memory_size))
    
    def _create_time_encoding(self, size: int) -> torch.Tensor:
        """Create sinusoidal time encoding."""
        positions = torch.arange(size).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.encoding_dim, 2).float() * 
                             (-np.log(10000.0) / self.encoding_dim))
        pe = torch.zeros(size, self.encoding_dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe
    
    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        importance: float = 1.0
    ):
        """Store an experience tuple."""
        episode = torch.cat([state, action, reward], dim=-1)
        
        self.episodes.append(episode)
        self.timestamps.append(len(self.episodes))
        self.importance_scores.append(importance)
        
        if len(self.episodes) > self.memory_size:
            self.episodes.pop(0)
            self.timestamps.pop(0)
            self.importance_scores.pop(0)
    
    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve most relevant memories.
        
        Args:
            query: [batch, embedding_dim * 3] - encoded state-action-reward
            k: Number of memories to retrieve
        
        Returns:
            retrieved_memories: [batch, k, embedding_dim * 3]
            attention_scores: [batch, k]
        """
        if len(self.episodes) == 0:
            return torch.zeros(query.size(0), k, query.size(-1)), torch.zeros(query.size(0), k)
        
        memory_tensor = torch.stack(self.episodes) if isinstance(self.episodes[0], torch.Tensor) else \
                        torch.stack([e for e in self.episodes if isinstance(e, torch.Tensor)])
        
        if memory_tensor.size(0) < k:
            memory_tensor = torch.cat([
                memory_tensor,
                torch.zeros(k - memory_tensor.size(0), memory_tensor.size(1), memory_tensor.size(2))
            ], dim=0)
        
        query_encoded = self.query_encoder(query)
        memory_encoded = self.memory_encoder(memory_tensor)
        
        scores = torch.matmul(query_encoded, memory_encoded.T)
        
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        
        retrieved = memory_tensor[top_indices]
        
        return retrieved, top_scores
    
    def compute_recency_weight(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute recency weighting for retrieved memories."""
        current_time = len(self.timestamps)
        times = torch.tensor(self.timestamps, device=indices.device)
        recency = torch.exp(-0.01 * (current_time - times))
        return recency[indices]
    
    def compute_importance_weight(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute importance weighting for retrieved memories."""
        importances = torch.tensor(self.importance_scores, device=indices.device)
        return importances[indices]
```

### Pattern 4: RAG-Augmented LLM with Memory

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    content: str
    embedding: torch.Tensor
    access_count: int
    last_accessed: float
    relevance_score: float

class LongTermMemorySystem:
    """
    Long-term memory system with semantic and episodic components.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        memory_size: int = 10000,
        retrieval_top_k: int = 10
    ):
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.retrieval_top_k = retrieval_top_k
        
        self.semantic_memory: List[MemoryEntry] = []
        self.episodic_memory: List[MemoryEntry] = []
        
        self.access_patterns: Dict[int, List[float]] = {}
        
        self.decay_rate = 0.99
        self.consolidation_threshold = 5
    
    def store(
        self,
        content: str,
        embedding: torch.Tensor,
        memory_type: str = "semantic"
    ):
        """Store new memory entry."""
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            access_count=1,
            last_accessed=0.0,
            relevance_score=1.0
        )
        
        if memory_type == "semantic":
            self.semantic_memory.append(entry)
            if len(self.semantic_memory) > self.memory_size:
                self._consolidate_semantic()
        else:
            self.episodic_memory.append(entry)
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        memory_type: str = "semantic",
        top_k: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve most relevant memories.
        
        Returns:
            contents: List of retrieved memory contents
            scores: Relevance scores
        """
        top_k = top_k or self.retrieval_top_k
        
        memory = self.semantic_memory if memory_type == "semantic" else self.episodic_memory
        
        if not memory:
            return [], []
        
        embeddings = torch.stack([e.embedding for e in memory])
        
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            embeddings,
            dim=-1
        )
        
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(memory)))
        
        retrieved_contents = [memory[idx.item()].content for idx in top_indices]
        retrieved_scores = top_scores.tolist()
        
        for idx in top_indices:
            memory[idx.item()].access_count += 1
            memory[idx.item()].last_accessed = 1.0
        
        return retrieved_contents, retrieved_scores
    
    def _consolidate_semantic(self):
        """Consolidate semantic memory when full."""
        self.semantic_memory.sort(key=lambda x: x.relevance_score, reverse=True)
        self.semantic_memory = self.semantic_memory[:int(self.memory_size * 0.8)]
    
    def apply_decay(self):
        """Apply time-based decay to all memory entries."""
        for entry in self.semantic_memory:
            entry.relevance_score *= self.decay_rate
            entry.relevance_score += 0.01 * entry.access_count / 10
        
        for entry in self.episodic_memory:
            entry.relevance_score *= self.decay_rate * 0.95
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "semantic_size": len(self.semantic_memory),
            "episodic_size": len(self.episodic_memory),
            "avg_relevance_semantic": np.mean([e.relevance_score for e in self.semantic_memory]),
            "avg_relevance_episodic": np.mean([e.relevance_score for e in self.episodic_memory]),
            "total_accesses": sum(e.access_count for e in self.semantic_memory + self.episodic_memory)
        }


class MemoryAugmentedRAG(nn.Module):
    """
    RAG model with additional memory components.
    """
    
    def __init__(
        self,
        llm,
        retriever,
        embedding_model,
        memory_system: LongTermMemorySystem
    ):
        super().__init__()
        
        self.llm = llm
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.memory = memory_system
        
        self.fusion_weight = 0.3
    
    def generate_with_memory(
        self,
        query: str,
        context: Optional[str] = None,
        use_memory: bool = True,
        store_interaction: bool = True
    ) -> Dict:
        """
        Generate response with memory augmentation.
        """
        query_embedding = self.embedding_model.encode(query)
        
        retrieved_docs = self.retriever.search(query, top_k=5)
        
        memory_contents = []
        if use_memory:
            memory_contents, memory_scores = self.memory.retrieve(query_embedding)
        
        combined_context = self._build_context(
            query,
            context,
            retrieved_docs,
            memory_contents
        )
        
        response = self.llm.generate(combined_context)
        
        if store_interaction:
            response_embedding = self.embedding_model.encode(response)
            self.memory.store(
                content=f"Q: {query}\nA: {response}",
                embedding=(query_embedding + response_embedding) / 2,
                memory_type="episodic"
            )
        
        return {
            "response": response,
            "retrieved_docs": retrieved_docs,
            "memory_contents": memory_contents,
            "context_used": combined_context
        }
    
    def _build_context(
        self,
        query: str,
        explicit_context: Optional[str],
        retrieved_docs: List[str],
        memory_contents: List[str]
    ) -> str:
        """Build context from multiple sources."""
        sections = []
        
        if explicit_context:
            sections.append(f"Explicit Context:\n{explicit_context}")
        
        if retrieved_docs:
            sections.append(f"Retrieved Knowledge:\n" + "\n".join(f"- {d}" for d in retrieved_docs))
        
        if memory_contents:
            sections.append(f"Relevant Past Interactions:\n" + 
                          "\n".join(f"- {m}" for m in memory_contents[:3]))
        
        sections.append(f"Current Query:\n{query}")
        
        return "\n\n".join(sections)
    
    def consolidate_memories(self):
        """Consolidate episodic memories into semantic memory."""
        if len(self.memory.episodic_memory) > 100:
            self.memory.apply_decay()
            
            high_relevance = [
                e for e in self.memory.episodic_memory
                if e.relevance_score > self.memory.consolidation_threshold
            ]
            
            for entry in high_relevance:
                self.memory.store(
                    content=entry.content,
                    embedding=entry.embedding,
                    memory_type="semantic"
                )
                
                self.memory.episodic_memory.remove(entry)
```

## Framework Integration

### Integration with LangChain

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

class LangChainMemoryIntegration:
    def __init__(self, llm, vectorstore):
        self.memory = ConversationBufferMemory()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
```

### Integration with Haystack

```python
from haystack import Pipeline
from haystack.nodes import Retriever, PromptNode

class HaystackMemoryPipeline:
    def __init__(self, document_store, llm):
        self.pipe = Pipeline()
        self.pipe.add_node(
            component=Retriever(document_store),
            name="Retriever",
            inputs=["Query"]
        )
        self.pipe.add_node(
            component=PromptNode(llm),
            name="LLM",
            inputs=["Retriever"]
        )
```

## Common Pitfalls

### Pitfall 1: Memory Addressing Instability

**Problem**: Attention weights become sharp or flat, making retrieval unstable.

**Solution**: Use temperature scaling and proper initialization:
```python
# Scale key strength to reasonable range
key_strength = torch.sigmoid(gate) * 10 + 1  # Range [1, 11]
```

### Pitfall 2: Catastrophic Forgetting in Memory

**Problem**: New memories overwrite important old ones.

**Solution**: Implement importance-based retention:
```python
if new_memory.relevance > old_memory.relevance * retention_threshold:
    replace_memory(old, new)
```

### Pitfall 3: Not Handling Memory Sparsity

**Problem**: Most memory locations unused while a few are overloaded.

**Solution**: Implement load balancing:
```python
# Track usage distribution
if usage_variance > threshold:
    redistribute_usage()
```

## Research References

1. **Graves et al. (2014)** - "Neural Turing Machines" - Original NTM paper.

2. **Weston et al. (2015)** - "Memory Networks" - Memory Networks architecture.

3. **Graves et al. (2016)** - "Hybrid Computing Using Neural Turing Machines" - DNC paper.

4. **Sukhbaatar et al. (2015)** - "End-to-End Memory Networks" - Memory Networks with backprop.

5. **Santoro et al. (2018)** - "Relational Networks" - Relation Networks with memory.

6. **Menon et al. (2018)** - "Temporal Memory Networks" - Time-series memory.

7. **Borgeaud et al. (2021)** - "Improving Language Models by Retrieving" - RETRO model.

8. **Guu et al. (2020)** - "REALM" - Retrieval-augmented language model pre-training.

9. **Lewis et al. (2020)** - "Retrieval-Augmented Generation for Knowledge-Intensive NLP" - RAG.

10. **Khandelwal et al. (2020)** - "Generalization through Memorization" - Nearest neighbor LM.