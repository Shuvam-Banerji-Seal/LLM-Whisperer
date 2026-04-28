# Gradient-Free Tuning — Fine-tuning Skill Prompt

Implementing PEFT methods beyond LoRA, including prefix tuning, prompt tuning, adapter methods, and other parameter-efficient approaches.

---

## 1. Identity and Mission

Implement parameter-efficient fine-tuning (PEFT) methods that enable model adaptation without updating all parameters. Beyond LoRA, this includes prefix tuning, prompt tuning, adapter layers, IA3, and combinations thereof. These methods reduce computational and memory costs while maintaining performance.

---

## 2. Theory & Fundamentals

### 2.1 PEFT Taxonomy

```
PEFT Methods
├── Additive Methods
│   ├── Prefix Tuning: Prepend learnable tokens
│   ├── Prompt Tuning: Learn soft prompts
│   └── Adapter Methods: Add new layers
│
├── Selection Methods
│   └── BitFit: Fine-tune biases only
│
└── Reparameterization Methods
    ├── LoRA: Low-rank decomposition
    └── Its variants (QLoRA, DoRA, etc.)
```

### 2.2 Prefix Tuning

Learn continuous "prefix" tokens that are prepended to each transformer layer:

```
Original: [x, layer_0, layer_1, ..., layer_n]
Prefix:   [P_0, P_1, ..., P_k, x, layer_0, layer_1, ..., layer_n]
```

Parameters: P_i ∈ R^(k × d) per layer

### 2.3 Prompt Tuning

Learn a "soft prompt" at the input layer only:

```
Original: [x_1, x_2, ..., x_n] → model
Prompt:   [p_1, p_2, ..., p_m, x_1, x_2, ..., x_n] → model
```

Parameters: P ∈ R^(m × d_input)

### 2.4 Adapter Methods

Insert small bottleneck layers between transformer layers:

```
Original: x → LayerNorm → Attention → LayerNorm → FFN
Adapter:  x → LayerNorm → Attention → LayerNorm → [Adapter] → FFN
```

Adapter: Down-project → Nonlin → Up-project

---

## 3. Implementation Patterns

### Pattern 1: Prefix Tuning Implementation

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from transformers import PreTrainedModel

@dataclass
class PrefixConfig:
    """Configuration for prefix tuning."""
    num_prefix_tokens: int = 20
    prefix_dim: int = 512
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    dropout: float = 0.0
    requires_grad: bool = True

class PrefixTuning(nn.Module):
    """
    Prefix tuning implementation.
    Learns prefix tokens for each transformer layer.
    """

    def __init__(self, config: PrefixConfig):
        super().__init__()
        self.config = config

        # Prefix embeddings for each layer
        # Two sets: for key and value
        self.prefix_kv = nn.Parameter(
            torch.randn(
                config.num_layers,
                2 * config.num_prefix_tokens,  # k and v
                config.num_heads * config.head_dim,
            ) * 0.02
        )

        # Prefix for Q (optional)
        self.prefix_q = nn.Parameter(
            torch.randn(
                config.num_layers,
                config.num_prefix_tokens,
                config.num_heads * config.head_dim,
            ) * 0.02
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Get prefix for a specific layer.

        Returns:
            prefix_k: (batch_size, num_prefix_tokens, num_heads * head_dim)
            prefix_v: (batch_size, num_prefix_tokens, num_heads * head_dim)
            prefix_q: (batch_size, num_prefix_tokens, num_heads * head_dim)
        """
        # Get prefix for this layer
        prefix_kv = self.prefix_kv[layer_idx]
        prefix_q = self.prefix_q[layer_idx]

        # Split into k and v
        prefix_k = prefix_kv[:, :self.config.num_prefix_tokens, :]
        prefix_v = prefix_kv[:, self.config.num_prefix_tokens:, :]

        # Expand to batch size
        prefix_k = prefix_k.unsqueeze(0).expand(batch_size, -1, -1)
        prefix_v = prefix_v.unsqueeze(0).expand(batch_size, -1, -1)
        prefix_q = prefix_q.unsqueeze(0).expand(batch_size, -1, -1)

        if self.dropout:
            prefix_k = self.dropout(prefix_k)
            prefix_v = self.dropout(prefix_v)
            prefix_q = self.dropout(prefix_q)

        return {
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
            "prefix_q": prefix_q,
        }

    def get_full_prefix(
        self,
        batch_size: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Get prefix for all layers."""
        return [
            self.forward(batch_size, 0, layer_idx)
            for layer_idx in range(self.config.num_layers)
        ]


class PrefixTuningWrapper(nn.Module):
    """
    Wrapper that applies prefix tuning to a transformer model.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: PrefixConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.prefix_tuning = PrefixTuning(config)

        # Freeze base model
        for p in model.parameters():
            p.requires_grad = False

        # Only prefix parameters are trainable
        for p in self.prefix_tuning.parameters():
            p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward with prefix injection.
        This is simplified - real implementation requires modifying attention.
        """
        batch_size = input_ids.shape[0]

        # Get all layer prefixes
        all_prefixes = self.prefix_tuning.get_full_prefix(batch_size)

        # In real implementation, modify attention compute to use prefixes
        # For now, just return base model output
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)


class PromptEmbedding(nn.Module):
    """
    Prompt embeddings for input-level soft prompts.
    """

    def __init__(
        self,
        num_prompts: int,
        embedding_dim: int,
        vocab_size: int,
        initialize_from_vocab: bool = True,
    ):
        super().__init__()
        self.num_prompts = num_prompts
        self.embedding_dim = embedding_dim

        # Embedding matrix (like regular word embeddings)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Alternatively, learnable prompts
        if not initialize_from_vocab:
            self.embedding = nn.Parameter(
                torch.randn(num_prompts, embedding_dim) * 0.01
            )

    def forward(self, batch_size: int) -> torch.Tensor:
        """Get prompt embeddings for batch."""
        if isinstance(self.embedding, nn.Embedding):
            # Sample random tokens for prompts
            prompt_ids = torch.randint(
                0, self.embedding.num_embeddings,
                (self.num_prompts,),
                device=next(self.parameters()).device,
            )
            prompts = self.embedding(prompt_ids)
        else:
            # Use learnable embeddings
            prompts = self.embedding

        # Expand to batch
        prompts = prompts.unsqueeze(0).expand(batch_size, -1, -1)

        return prompts


class PromptTuningModel(nn.Module):
    """
    Complete model with prompt tuning.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        num_prompts: int = 20,
        prompt_init: str = "random",  # "random" or "text"
        prompt_text: str = "prefix prompt here",
        tokenizer: Any = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_prompts = num_prompts
        self.config = base_model.config

        # Get embedding dimension
        if hasattr(self.config, "hidden_size"):
            embedding_dim = self.config.hidden_size
        elif hasattr(self.config, "d_model"):
            embedding_dim = self.config.d_model
        else:
            embedding_dim = 768

        # Initialize prompt
        if prompt_init == "text" and tokenizer is not None:
            # Initialize from text
            prompt_ids = tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=num_prompts,
            )["input_ids"][0]

            # Embed
            self.prompt_embeddings = nn.Embedding.from_pretrained(
                base_model.get_input_embeddings()(prompt_ids),
                freeze=False,
            )
        else:
            # Random initialization
            self.prompt_embeddings = nn.Embedding(num_prompts, embedding_dim)
            nn.init.normal_(self.prompt_embeddings.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward with prompt prepended."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Get prompt embeddings
        prompt_ids = torch.arange(
            self.num_prompts,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        prompt_embeds = self.prompt_embeddings(prompt_ids)

        # Get input embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Concatenate
        full_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        # Adjust attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.num_prompts,
                device=device,
                dtype=attention_mask.dtype,
            )
            full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            full_attention_mask = None

        # Forward through base model
        return self.base_model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            **kwargs,
        )
```

### Pattern 2: Adapter Methods

```python
import torch
import torch.nn as nn
from typing import Optional, Any
from dataclasses import dataclass

@dataclass
class AdapterConfig:
    """Configuration for adapter layers."""
    hidden_size: int = 768
    adapter_size: int = 64
    adapter_activation: str = "relu"  # or "gelu", "swish"
    adapter_dropout: float = 0.1
    adapter_initializer_range: float = 0.0002

class Adapter(nn.Module):
    """
    Adapter layer: bottleneck with down-project and up-project.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

        # Down projection
        self.down_project = nn.Linear(
            config.hidden_size,
            config.adapter_size,
            bias=True,
        )

        # Activation
        if config.adapter_activation == "relu":
            self.activation = nn.ReLU()
        elif config.adapter_activation == "gelu":
            self.activation = nn.GELU()
        elif config.adapter_activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(config.adapter_dropout)

        # Up projection
        self.up_project = nn.Linear(
            config.adapter_size,
            config.hidden_size,
            bias=True,
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize adapter weights."""
        nn.init.normal_(self.down_project.weight, std=self.config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=self.config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len, hidden_size)

        Returns:
            adapter_output: (batch_size, seq_len, hidden_size)
        """
        # Down project
        h = self.down_project(x)

        # Activate
        h = self.activation(h)

        # Dropout
        h = self.dropout(h)

        # Up project
        h = self.up_project(h)

        # Residual connection (optional, typically not used)
        return h


class AdapterWrapper(nn.Module):
    """
    Wrapper that adds adapters to a transformer layer.
    """

    def __init__(
        self,
        layer: nn.Module,
        adapter_config: AdapterConfig,
        adapter_type: str = "post_attention",  # or "pre_attention", "parallel"
    ):
        super().__init__()
        self.layer = layer
        self.adapter_config = adapter_config
        self.adapter_type = adapter_type

        # Create adapter
        self.adapter = Adapter(adapter_config)

        # Layer norm for adapter input (before adapter)
        self.layer_norm = nn.LayerNorm(adapter_config.hidden_size)

    def forward(self, *args, **kwargs):
        """Forward with adapter injection."""
        if self.adapter_type == "post_attention":
            # Run layer
            output = self.layer(*args, **kwargs)

            # Apply adapter to hidden state
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
                adapter_output = self.adapter(self.layer_norm(hidden_states))
                return (hidden_states + adapter_output,) + rest
            else:
                adapter_output = self.adapter(self.layer_norm(output))
                return output + adapter_output

        elif self.adapter_type == "parallel":
            # Compute both in parallel and add
            output = self.layer(*args, **kwargs)
            if isinstance(output, tuple):
                hidden_states = output[0]
                adapter_output = self.adapter(hidden_states)
                return (hidden_states + adapter_output,) + output[1:]
            else:
                adapter_output = self.adapter(output)
                return output + adapter_output

        elif self.adapter_type == "pre_attention":
            # Apply adapter before attention
            x = args[0]
            adapter_output = self.adapter(self.layer_norm(x))
            x = x + adapter_output

            # Continue with layer
            return self.layer(x, *args[1:], **kwargs)

        return output


class CompacterAdapter(Adapter):
    """
    COMPACTER: Compact Adapter with Kronecker decomposition.
    Uses Kronecker product for more parameter-efficient adapters.
    """

    def __init__(
        self,
        hidden_size: int,
        adapter_size: int,
        rank: int = 1,
    ):
        super().__init__(AdapterConfig(
            hidden_size=hidden_size,
            adapter_size=adapter_size,
        ))

        # Kronecker factors instead of dense matrices
        self.A = nn.Parameter(torch.randn(rank, hidden_size))
        self.B = nn.Parameter(torch.randn(adapter_size, rank))
        self.C = nn.Parameter(torch.randn(rank, adapter_size))
        self.D = nn.Parameter(torch.randn(rank, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with Kronecker decomposition."""
        # Simplified - real implementation uses proper Kronecker
        h = x @ self.A.T @ self.B.T
        h = h @ self.C.T
        h = h @ self.D.T
        return h


class LoRAAdapter(nn.Module):
    """
    LoRA as an adapter variant.
    Low-rank decomposition with A and B matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Scale
        self.scaling = self.alpha / self.rank

        # Initialize
        nn.init.normal_(self.lora_A, std=1 / self.rank)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with LoRA."""
        # x @ (A @ B) = (x @ A) @ B
        h = x @ self.lora_A

        if self.dropout:
            h = self.dropout(h)

        h = h @ self.lora_B

        return h * self.scaling


class AdapterModel(nn.Module):
    """
    Model with adapters inserted into all transformer layers.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_config: AdapterConfig,
        adapter_type: str = "post_attention",
        bottleneck_size: int = 64,
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter_config = adapter_config
        self.adapter_type = adapter_type

        # Insert adapters into transformer layers
        self._insert_adapters()

    def _insert_adapters(self):
        """Insert adapters into transformer layers."""
        for name, module in self.base_model.named_children():
            if self._is_transformer_layer(module):
                # Wrap with adapter
                wrapped = AdapterWrapper(
                    module,
                    self.adapter_config,
                    self.adapter_type,
                )
                setattr(self.base_model, name, wrapped)
            else:
                # Recurse
                self._insert_adapters_in_module(module)

    def _is_transformer_layer(self, module: nn.Module) -> bool:
        """Check if module is a transformer layer."""
        # This depends on the specific model architecture
        return False

    def _insert_adapters_in_module(self, module: nn.Module):
        """Recursively insert adapters."""
        for name, child in module.named_children():
            if self._is_transformer_layer(child):
                wrapped = AdapterWrapper(
                    child,
                    self.adapter_config,
                    self.adapter_type,
                )
                setattr(module, name, wrapped)
            else:
                self._insert_adapters_in_module(child)
```

### Pattern 3: IA3 (Infused Adapter by Inhibiting and Amplifying)

```python
import torch
import torch.nn as nn

class IA3Config:
    """Configuration for IA3 (Infused Adapter by Inhibiting and Amplifying)."""

    def __init__(
        self,
        target_modules: list = None,
        ia3_rank: int = 1,
        ia3_dropout: float = 0.0,
    ):
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "fc1"]
        self.ia3_rank = ia3_rank
        self.ia3_dropout = ia3_dropout

class IA3Linear(nn.Module):
    """
    IA3-modified linear layer.
    Adds learnable vectors that scale or inhibit activations.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 1,
        is_inhibit: bool = False,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.is_inhibit = is_inhibit

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # For IA3, we learn a single vector (rank 1)
        if is_inhibit:
            # Inhibitor: element-wise multiplication before computation
            self.learnable = nn.Parameter(torch.ones(in_features))
        else:
            # Infuser: element-wise multiplication after computation
            self.learnable = nn.Parameter(torch.ones(out_features))

        # Initialize near identity
        nn.init.ones_(self.learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with IA3 modulation."""
        if self.is_inhibit:
            # Apply inhibitor before matrix multiply
            x = x * self.learnable

        output = self.original_layer(x)

        if not self.is_inhibit:
            # Apply infuser after matrix multiply
            output = output * self.learnable

        return output


class IA3Model(nn.Module):
    """
    Model with IA3 adapters.
    IA3 is more parameter-efficient than LoRA (only 1 parameter per weight vector).
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: IA3Config,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # Replace target modules with IA3 versions
        self._apply_ia3()

    def _apply_ia3(self):
        """Apply IA3 to target modules."""
        for name, module in self.base_model.named_modules():
            # Check if this is a target module
            if self._is_target_module(name, module):
                # Create IA3 wrapper
                parent_name, child_name = self._get_parent_and_child(name)

                if parent_name:
                    parent = self.base_model.get_submodule(parent_name)
                else:
                    parent = self.base_model

                # Determine if inhibit or infuser
                target_type = name.split(".")[-1]  # e.g., q_proj, fc1
                is_inhibit = target_type in ["k_proj", "fc1"]

                ia3_layer = IA3Linear(
                    module,
                    rank=self.config.ia3_rank,
                    is_inhibit=is_inhibit,
                )

                setattr(parent, child_name, ia3_layer)

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Check if module should be modified."""
        if not isinstance(module, nn.Linear):
            return False

        for target in self.config.target_modules:
            if target in name:
                return True

        return False

    def _get_parent_and_child(self, full_name: str) -> tuple:
        """Get parent module name and child name."""
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", full_name
```

### Pattern 4: BitFit (Bias-only Fine-tuning)

```python
import torch
import torch.nn as nn
from typing import List, Optional

class BitFitModel(nn.Module):
    """
    BitFit: Fine-tune only bias vectors.
    Very parameter-efficient.
    """

    def __init__(
        self,
        base_model: nn.Module,
        include_bias_grad: bool = True,
        exclude_embedding_biases: bool = False,
    ):
        super().__init__()
        self.base_model = base_model

        # Freeze everything except biases
        for name, param in base_model.named_parameters():
            if "bias" in name:
                if exclude_embedding_biases and "embeddings" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = include_bias_grad
            else:
                param.requires_grad = False

        # Count trainable parameters
        self._count_params()

    def _count_params(self):
        """Count trainable vs total parameters."""
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)

        print(f"BitFit: {trainable:,} / {total:,} parameters trainable ({100*trainable/total:.2f}%)")

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


class BiasOnlyOptimizer:
    """
    Optimizer wrapper that only updates bias parameters.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        self.model = model

        # Get only bias parameters
        bias_params = [p for n, p in model.named_parameters() if "bias" in n and p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            [{"params": bias_params, "lr": learning_rate}],
            weight_decay=0.0,
        )

    def step(self):
        """Single optimization step."""
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### Pattern 5: Combined PEFT Methods

```python
class CombinedPEFTModel(nn.Module):
    """
    Model with multiple PEFT methods combined.
    e.g., LoRA + Adapters + Prefix Tuning
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_config: dict = None,
        adapter_config: dict = None,
        prefix_config: dict = None,
    ):
        super().__init__()
        self.base_model = base_model

        # Apply LoRA if specified
        if lora_config:
            self.lora_wrapper = LoRAWrapper(
                base_model,
                lora_config["rank"],
                lora_config.get("alpha", 1.0),
                lora_config.get("target_modules", ["q_proj", "v_proj"]),
            )

        # Apply adapters if specified
        if adapter_config:
            self.adapter_model = AdapterModel(
                base_model,
                AdapterConfig(**adapter_config),
            )

        # Apply prefix tuning if specified
        if prefix_config:
            self.prefix_tuning = PrefixTuningWrapper(
                base_model,
                PrefixConfig(**prefix_config),
            )

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


class ProgressiveTraining:
    """
    Progressively enable more PEFT components during training.
    """

    def __init__(
        self,
        model: nn.Module,
        peft_components: List[nn.Module],
    ):
        self.model = model
        self.components = peft_components

        # Disable all initially
        for comp in self.components:
            for p in comp.parameters():
                p.requires_grad = False

    def enable_component(self, index: int):
        """Enable a specific component."""
        if index < len(self.components):
            for p in self.components[index].parameters():
                p.requires_grad = True

    def enable_all(self):
        """Enable all components."""
        for comp in self.components:
            for p in comp.parameters():
                p.requires_grad = True

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
```

### Pattern 6: Soft Prompt Library

```python
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any

class PromptPool(nn.Module):
    """
    Pool of soft prompts for multi-task learning.
    """

    def __init__(
        self,
        num_prompts: int,
        prompt_length: int,
        embedding_dim: int,
        num_tasks: int,
    ):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        self.num_tasks = num_tasks

        # Prompt pool: (num_tasks, num_prompts_per_task, prompt_length, embedding_dim)
        self.prompt_pool = nn.Parameter(
            torch.randn(num_tasks, num_prompts, prompt_length, embedding_dim) * 0.01
        )

    def get_prompt(self, task_id: int, prompt_id: int = 0) -> torch.Tensor:
        """Get a specific prompt for a task."""
        return self.prompt_pool[task_id, prompt_id]

    def get_task_prompts(self, task_id: int) -> torch.Tensor:
        """Get all prompts for a task."""
        return self.prompt_pool[task_id]


class PrefixTuningPool(nn.Module):
    """
    Prefix tuning with a pool of prefixes for different tasks.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_prefixes: int,
        prefix_length: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_prefixes = num_prefixes
        self.prefix_length = prefix_length

        # Prefix pool: (num_prefixes, num_layers, 2 * prefix_length, num_heads * head_dim)
        self.prefix_pool = nn.Parameter(
            torch.randn(num_prefixes, num_layers, 2 * prefix_length, num_heads * head_dim) * 0.02
        )

    def get_prefix(self, prefix_id: int) -> nn.Parameter:
        """Get a specific prefix."""
        return self.prefix_pool[prefix_id]
```

---

## 4. Framework Integration

### HuggingFace PEFT Integration

```python
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
)

# LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(base_model, lora_config)

# Prefix Tuning
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    token_dim=768,
    num_layers=12,
)

# Prompt Tuning
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Your task description here",
)
```

### TRL Integration

```python
from trl import AutoModelForCausalLMWithValueHead

# Wrap model with value head for PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "gpt2",
    peft_config=lora_config,
)
```

---

## 5. Performance Considerations

### Parameter Efficiency

| Method | Trainable Params | Memory Savings |
|--------|------------------|----------------|
| Full FT | 100% | 0% |
| LoRA (r=8) | ~0.5% | ~60% |
| Prefix | ~0.1% | ~70% |
| Adapter | ~2-3% | ~50% |
| BitFit | ~0.1% | ~70% |
| IA3 | ~0.1% | ~70% |

### Performance Comparison

| Method | Task Performance | Training Speed |
|--------|------------------|----------------|
| Full FT | Best | Slowest |
| LoRA | ~95-99% | Fast |
| Adapter | ~90-95% | Medium |
| Prefix | ~85-95% | Fast |
| BitFit | ~80-90% | Fastest |

---

## 6. Common Pitfalls

1. **Wrong Target Modules**: Applying adapters to wrong layers
2. **Rank Selection**: Too low rank loses capacity, too high wastes compute
3. **Learning Rate**: PEFT needs different LR than full fine-tuning
4. **Initialization**: Poor initialization leads to slow convergence
5. **Task Interference**: Multi-task prompts can interfere
6. **Prefix Length**: Too short loses expressivity, too long wastes compute

---

## 7. Research References

1. https://arxiv.org/abs/2106.09685 — "Prefix-Tuning: Optimizing Continuous Prompts"

2. https://arxiv.org/abs/2104.08691 — "Prompt Tuning: Learnable Soft Prompts"

3. https://arxiv.org/abs/1909.08478 — "AdapterFusion: Non-Destructive Adapter Composition"

4. https://arxiv.org/abs/2201.03596 — "Compacter: Efficient Low-Rank Adapter"

5. https://arxiv.org/abs/2205.00225 — "IA3: Infused Adapter by Inhibiting and Amplifying"

6. https://arxiv.org/abs/2111.09843 — "BitFit: Simple Parameter-Efficient Fine-Tuning"

7. https://arxiv.org/abs/2304.13088 — "LongLoRA: Efficient Fine-tuning of Long-Context LLMs"

8. https://arxiv.org/abs/2305.13245 — "QLoRA: Efficient Finetuning of Quantized LLMs"

---

## 8. Uncertainty and Limitations

**Not Covered:** Hardware-specific optimizations, advanced reparameterization methods, multi-modal adaptation.

**Production Considerations:** Start with LoRA as baseline. Use PEFT for domain adaptation and task-specific models. Monitor trainable vs total parameter ratio.

(End of file - total 1380 lines)