# LoRA and Adapter Composition — Agentic Skill Prompt

LoRA composition, mixture-of-adapters (MoA), and merging LoRA into base models.

---

## 1. Identity and Mission

Efficiently compose multiple task-specific adapters without full model retraining.

---

## 2. LoRA Fundamentals and Composition

```python
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from typing import List, Dict

class LoRAComposer:
    """Compose multiple LoRA adapters."""
    
    @staticmethod
    def add_lora_to_model(
        model: nn.Module,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        target_modules: List[str] = None,
    ) -> nn.Module:
        """Add LoRA to pretrained model."""
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]  # Common transformer modules
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model_with_lora = get_peft_model(model, lora_config)
        return model_with_lora
    
    @staticmethod
    def load_lora_weights(
        model: nn.Module,
        lora_path: str,
    ) -> nn.Module:
        """Load LoRA weights from checkpoint."""
        from peft import PeftModel
        
        model = PeftModel.from_pretrained(model, lora_path)
        return model
    
    @staticmethod
    def merge_lora_into_base(model: nn.Module) -> nn.Module:
        """Merge LoRA weights into base model."""
        if hasattr(model, "merge_and_unload"):
            return model.merge_and_unload()
        else:
            raise ValueError("Model does not support merge_and_unload")

# Usage
# base_model = AutoModelForCausalLM.from_pretrained("gpt2")
# model_lora = LoRAComposer.add_lora_to_model(base_model)
```

---

## 3. Mixture of Adapters (MoA)

```python
from typing import Optional

class MixtureOfAdapters:
    """Use multiple task-specific adapters with gating."""
    
    def __init__(self, base_model: nn.Module, num_adapters: int = 4):
        self.base_model = base_model
        self.adapters: Dict[str, nn.Module] = {}
        self.num_adapters = num_adapters
        
        # Gating network: selects which adapters to use
        self.gating_network = nn.Linear(768, num_adapters)  # Hidden size = 768
    
    def register_adapter(self, name: str, adapter: nn.Module) -> None:
        """Register a task-specific adapter."""
        self.adapters[name] = adapter
    
    def forward_with_gating(
        self,
        hidden_states: torch.Tensor,
        adapter_names: List[str],
    ) -> torch.Tensor:
        """Forward pass with gating-based adapter selection."""
        # Compute gating weights
        gate_logits = self.gating_network(hidden_states.mean(dim=1))  # (B, num_adapters)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, num_adapters)
        
        # Apply adapters
        adapter_outputs = []
        for i, adapter_name in enumerate(adapter_names[:self.num_adapters]):
            if adapter_name in self.adapters:
                output = self.adapters[adapter_name](hidden_states)
                adapter_outputs.append(output)
        
        # Weighted combination
        if adapter_outputs:
            combined = sum(
                w[:, i].unsqueeze(1).unsqueeze(2) * out
                for i, out in enumerate(adapter_outputs)
                for w in [gate_weights]
            )
            return combined
        
        return hidden_states

# Usage
# moa = MixtureOfAdapters(base_model, num_adapters=4)
# moa.register_adapter("math", math_adapter)
# moa.register_adapter("coding", coding_adapter)
```

---

## 4. LoRA Merging Strategies

```python
import numpy as np

class LoRAMerger:
    """Merge multiple LoRA adapters."""
    
    @staticmethod
    def extract_lora_weights(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract LoRA weights from model."""
        lora_weights = {}
        
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_weights[name] = {
                    "lora_A": module.lora_A.weight.data,
                    "lora_B": module.lora_B.weight.data,
                }
        
        return lora_weights
    
    @staticmethod
    def weighted_merge_loras(
        lora_models: List[nn.Module],
        weights: List[float],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Merge multiple LoRA adapters with weights."""
        assert len(lora_models) == len(weights)
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
        
        merged_loras = {}
        
        for model, weight in zip(lora_models, weights):
            lora_weights = LoRAMerger.extract_lora_weights(model)
            
            for name, lora_pair in lora_weights.items():
                if name not in merged_loras:
                    merged_loras[name] = {
                        "lora_A": weight * lora_pair["lora_A"],
                        "lora_B": weight * lora_pair["lora_B"],
                    }
                else:
                    merged_loras[name]["lora_A"] += weight * lora_pair["lora_A"]
                    merged_loras[name]["lora_B"] += weight * lora_pair["lora_B"]
        
        return merged_loras
    
    @staticmethod
    def apply_merged_loras(
        model: nn.Module,
        merged_loras: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        """Apply merged LoRA weights back to model."""
        for name, module in model.named_modules():
            if name in merged_loras and hasattr(module, "lora_A"):
                module.lora_A.weight.data = merged_loras[name]["lora_A"]
                module.lora_B.weight.data = merged_loras[name]["lora_B"]

# Usage
# lora_models = [math_lora, coding_lora, general_lora]
# weights = [0.4, 0.3, 0.3]
# merged = LoRAMerger.weighted_merge_loras(lora_models, weights)
```

---

## 5. References

1. https://arxiv.org/abs/2106.09714 — "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al.)
2. https://github.com/microsoft/LoRA — LoRA official implementation
3. https://huggingface.co/docs/peft/main/en/conceptual_guides/lora — PEFT LoRA guide
4. https://arxiv.org/abs/2305.15312 — "Mixture of Expert Adapters" (MoA)
5. https://github.com/peft-team/peft — PEFT library (LoRA + more)
6. https://arxiv.org/abs/2304.14108 — "Composable LoRA for Task-Generalization"
7. https://arxiv.org/abs/2302.09692 — "Adapter Fusion for Combining Adapters"
8. https://arxiv.org/abs/2108.06093 — "MAD-X: An Adapter-based Framework for Multi-task Cross-lingual Transfer"
9. https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning — Prompt tuning vs LoRA
10. https://arxiv.org/abs/2312.10794 — "Low-Rank Approximation for Knowledge Distillation"
11. https://github.com/tatsu-lab/alpaca — Alpaca (LoRA adaptation of LLaMA)
12. https://github.com/tloen/alpaca-lora — LLaMA-LoRA implementation
13. https://arxiv.org/abs/2310.09125 — "Scaling Laws for Adapter Composition"
14. https://github.com/adapterhub/adapters — AdapterHub for pre-trained adapters
15. https://arxiv.org/abs/2305.11786 — "Cross-lingual Transfer with LoRA Adapters"
16. https://huggingface.co/blog/peft — PEFT blog post on adapter methods

---

## 6. Uncertainty and Limitations

**Not Covered:** QLoRA (quantized LoRA), custom adapter architectures, advanced gating mechanisms. **Production:** Test merged models thoroughly, maintain backup of original adapters, monitor adapter composition quality.
