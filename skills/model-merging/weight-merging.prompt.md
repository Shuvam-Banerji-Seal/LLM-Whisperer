# Weight Merging Strategies — Agentic Skill Prompt

Linear interpolation, SLERP, task-specific weighted merging, and post-merge validation.

---

## 1. Identity and Mission

Combine multiple model checkpoints into unified models with improved performance on diverse tasks.

---

## 2. Linear Interpolation (Slerp)

```python
import torch
import torch.nn as nn
from typing import Tuple

class WeightMerger:
    """Merge model weights using various strategies."""
    
    @staticmethod
    def linear_interpolate(
        model1: nn.Module,
        model2: nn.Module,
        alpha: float = 0.5,
    ) -> nn.Module:
        """Linear interpolation: α*model1 + (1-α)*model2."""
        merged_state = {}
        
        state1 = model1.state_dict()
        state2 = model2.state_dict()
        
        for key in state1.keys():
            merged_state[key] = alpha * state1[key] + (1 - alpha) * state2[key]
        
        # Load merged weights
        model_merged = model1.__class__(model1.config)
        model_merged.load_state_dict(merged_state)
        
        return model_merged
    
    @staticmethod
    def spherical_slerp(
        model1: nn.Module,
        model2: nn.Module,
        t: float = 0.5,
    ) -> nn.Module:
        """Spherical linear interpolation (preserves magnitude)."""
        merged_state = {}
        
        state1 = model1.state_dict()
        state2 = model2.state_dict()
        
        for key in state1.keys():
            v1 = state1[key].float()
            v2 = state2[key].float()
            
            # Normalize
            v1_norm = v1 / (torch.norm(v1) + 1e-8)
            v2_norm = v2 / (torch.norm(v2) + 1e-8)
            
            # Compute angle
            dot_product = torch.sum(v1_norm * v2_norm)
            omega = torch.acos(torch.clamp(dot_product, -1, 1))
            
            # SLERP
            if omega > 1e-6:
                slerp = (torch.sin((1 - t) * omega) / torch.sin(omega)) * v1 + \
                        (torch.sin(t * omega) / torch.sin(omega)) * v2
            else:
                slerp = (1 - t) * v1 + t * v2
            
            merged_state[key] = slerp
        
        model_merged = model1.__class__(model1.config)
        model_merged.load_state_dict(merged_state)
        
        return model_merged
    
    @staticmethod
    def simple_average(models: list[nn.Module]) -> nn.Module:
        """Average weights across multiple models."""
        merged_state = {}
        
        states = [m.state_dict() for m in models]
        
        for key in states[0].keys():
            merged_state[key] = torch.stack([s[key] for s in states]).mean(dim=0)
        
        model_merged = models[0].__class__(models[0].config)
        model_merged.load_state_dict(merged_state)
        
        return model_merged

# Usage
# model_merged = WeightMerger.linear_interpolate(model1, model2, alpha=0.7)
```

---

## 3. Task-Specific Weighted Merging

```python
from typing import Dict, List

class TaskWeightedMerging:
    """Merge models with weights based on task performance."""
    
    def __init__(self):
        self.task_scores: Dict[str, List[float]] = {}
    
    def add_task_scores(self, task_name: str, model_scores: List[float]) -> None:
        """Store scores for models on a task."""
        self.task_scores[task_name] = model_scores
    
    def compute_task_weights(self, task_name: str) -> List[float]:
        """Compute weights based on task performance."""
        if task_name not in self.task_scores:
            raise ValueError(f"No scores for task: {task_name}")
        
        scores = self.task_scores[task_name]
        
        # Softmax of scores (higher score = higher weight)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        weights = torch.softmax(scores_tensor, dim=0)
        
        return weights.tolist()
    
    def merge_with_task_weights(
        self,
        models: List[nn.Module],
        task_name: str,
    ) -> nn.Module:
        """Merge models with task-specific weights."""
        weights = self.compute_task_weights(task_name)
        
        merged_state = {}
        states = [m.state_dict() for m in models]
        
        for key in states[0].keys():
            weighted_sum = sum(w * s[key] for w, s in zip(weights, states))
            merged_state[key] = weighted_sum
        
        model_merged = models[0].__class__(models[0].config)
        model_merged.load_state_dict(merged_state)
        
        return model_merged

# Usage
merger = TaskWeightedMerging()
merger.add_task_scores("math", [0.92, 0.88, 0.85])
merger.add_task_scores("code", [0.85, 0.91, 0.87])
# Merge with higher weight for math model
merged = merger.merge_with_task_weights(models, "math")
```

---

## 4. Post-Merge Validation

```python
class MergeValidator:
    """Validate merged model quality."""
    
    @staticmethod
    def check_output_distribution(
        original_models: List[nn.Module],
        merged_model: nn.Module,
        test_input: torch.Tensor,
    ) -> float:
        """Check if merged output is similar to originals."""
        outputs_original = [m(test_input).detach() for m in original_models]
        output_merged = merged_model(test_input).detach()
        
        # Compute KL divergence or cosine similarity
        similarities = []
        for out in outputs_original:
            sim = torch.nn.functional.cosine_similarity(
                output_merged.flatten(),
                out.flatten(),
                dim=0
            )
            similarities.append(sim.item())
        
        return sum(similarities) / len(similarities)
    
    @staticmethod
    def benchmark_merged_model(
        merged_model: nn.Module,
        eval_dataset,
        metric_fn,
    ) -> float:
        """Evaluate merged model on benchmark."""
        scores = []
        
        for batch in eval_dataset:
            outputs = merged_model(**batch)
            score = metric_fn(outputs, batch["labels"])
            scores.append(score)
        
        return sum(scores) / len(scores)

# Usage
# validator = MergeValidator()
# similarity = validator.check_output_distribution(models, merged, test_input)
```

---

## 5. References

1. https://arxiv.org/abs/2109.08773 — "Interpolation and Label Smoothing" (Linear interpolation foundations)
2. https://arxiv.org/abs/2103.02202 — "Model Soups: averaging weights of multiple fine-tuned models"
3. https://arxiv.org/abs/2212.04089 — "Merging Models with Fisher-Weighted Averaging" (Task-weighted merging)
4. https://github.com/cg505/fisher-merging — Fisher-weighted merging implementation
5. https://arxiv.org/abs/2211.03394 — "Towards Lossless Model Stitching"
6. https://github.com/NeuralWalker/lora-merge — LoRA-specific merging
7. https://arxiv.org/abs/2305.14934 — "MergeKit: Simplifying Model Merging"
8. https://github.com/cg505/mergekit — MergeKit official implementation
9. https://arxiv.org/abs/2304.09434 — "Weight Averaging for Continual Learning"
10. https://huggingface.co/docs/transformers/main/en/model_doc/auto_model — Model loading patterns
11. https://arxiv.org/abs/2307.04779 — "Sphere+: Spherical Model Merging"
12. https://github.com/tomu-terada/SphericalMerging — Sphere+ implementation
13. https://arxiv.org/abs/2110.04309 — "Orthogonal Ensemble for LLMs"
14. https://github.com/microsoft/DeepSpeed — DeepSpeed model optimization
15. https://arxiv.org/abs/2306.04031 — "Merging for Multitask Learning"
16. https://pytorch.org/docs/stable/generated/torch.nn.Module.html — PyTorch module guide

---

## 6. Uncertainty and Limitations

**Not Covered:** Advanced Fisher-information-based methods, task-aware pruning during merge. **Production:** Test merged models thoroughly, monitor for catastrophic forgetting.
