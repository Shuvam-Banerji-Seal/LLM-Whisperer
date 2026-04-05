# RLHF and Direct Preference Optimization — Agentic Skill Prompt

RLHF fundamentals, reward modeling, DPO (Direct Preference Optimization), and preference data creation.

---

## 1. Identity and Mission

Align LLMs with human preferences through preference learning and reward optimization.

---

## 2. RLHF Pipeline

```python
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str
    score_difference: float = 1.0

class RLHFPipeline:
    """RLHF training setup."""
    
    def __init__(self):
        self.preference_data: List[PreferenceData] = []
    
    def add_preference(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        score_diff: float = 1.0,
    ) -> None:
        """Add preference data point."""
        self.preference_data.append(
            PreferenceData(prompt, chosen, rejected, score_diff)
        )
    
    def prepare_training_batch(self, batch_size: int = 32) -> List[Dict]:
        """Prepare batch for RLHF training."""
        batches = []
        
        for i in range(0, len(self.preference_data), batch_size):
            batch = self.preference_data[i:i+batch_size]
            batches.append({
                "prompts": [d.prompt for d in batch],
                "chosen": [d.chosen_response for d in batch],
                "rejected": [d.rejected_response for d in batch],
                "scores": [d.score_difference for d in batch],
            })
        
        return batches
    
    @staticmethod
    def ranking_loss(
        model_logits_chosen: torch.Tensor,
        model_logits_rejected: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """Ranking loss for RLHF."""
        # Bradley-Terry model loss
        loss = torch.nn.functional.logsigmoid(
            model_logits_chosen - model_logits_rejected - margin
        )
        return -loss.mean()

# Usage
pipeline = RLHFPipeline()
pipeline.add_preference(
    "What is 2+2?",
    "The answer is 4.",
    "The answer is 5.",
)
batches = pipeline.prepare_training_batch()
```

---

## 3. Reward Model Training

```python
import torch.nn as nn

class RewardModel(nn.Module):
    """Scalar reward model for RLHF."""
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        from transformers import AutoModel
        
        self.base_model = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(768, 1)  # Output scalar reward
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute reward for input."""
        outputs = self.base_model(input_ids, attention_mask)
        
        # Use [CLS] representation
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        
        # Compute scalar reward
        reward = self.reward_head(cls_hidden)
        
        return reward.squeeze(-1)
    
    @staticmethod
    def compute_reward_for_pairs(
        model,
        chosen_texts: List[str],
        rejected_texts: List[str],
        tokenizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rewards for preference pairs."""
        chosen_encodings = tokenizer(
            chosen_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=512
        )
        rejected_encodings = tokenizer(
            rejected_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=512
        )
        
        chosen_rewards = model(
            chosen_encodings["input_ids"],
            chosen_encodings["attention_mask"],
        )
        rejected_rewards = model(
            rejected_encodings["input_ids"],
            rejected_encodings["attention_mask"],
        )
        
        return chosen_rewards, rejected_rewards

# Usage
# reward_model = RewardModel(model_name="gpt2")
# optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
```

---

## 4. Direct Preference Optimization (DPO)

```python
class DPOTrainer:
    """Train LLM using Direct Preference Optimization (DPO)."""
    
    def __init__(
        self,
        model,
        reference_model,
        beta: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        self.model = model
        self.reference_model = reference_model
        self.beta = beta
        self.label_smoothing = label_smoothing
    
    def dpo_loss(
        self,
        policy_logps: torch.Tensor,
        reference_logps: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPO loss function.
        
        Directly optimizes for preference satisfaction without reward model.
        """
        # Log probability differences
        policy_logratios = policy_logps - reference_logps
        
        # Bradley-Terry model with label smoothing
        if self.label_smoothing > 0:
            label = label * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # DPO loss
        losses = -torch.nn.functional.logsigmoid(
            self.beta * policy_logratios
        )
        
        # Flip for rejected responses (label=0)
        losses = torch.where(label == 1, losses, -losses)
        
        return losses.mean()
    
    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> float:
        """Single DPO training step."""
        # Forward pass through policy model
        chosen_logits = self.model(chosen_ids).logits
        rejected_logits = self.model(rejected_ids).logits
        
        # Forward pass through reference model
        with torch.no_grad():
            ref_chosen_logits = self.reference_model(chosen_ids).logits
            ref_rejected_logits = self.reference_model(rejected_ids).logits
        
        # Compute log probabilities
        chosen_logps = self._compute_logps(chosen_logits, chosen_ids)
        rejected_logps = self._compute_logps(rejected_logits, rejected_ids)
        ref_chosen_logps = self._compute_logps(ref_chosen_logits, chosen_ids)
        ref_rejected_logps = self._compute_logps(ref_rejected_logits, rejected_ids)
        
        # Compute loss
        loss = self.dpo_loss(
            chosen_logps - rejected_logps,
            ref_chosen_logps - ref_rejected_logps,
            torch.ones_like(chosen_logps),
        )
        
        return loss.item()
    
    def _compute_logps(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities."""
        # Shift logits for causal LM
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = input_ids[..., 1:].contiguous()
        
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        
        # Gather log probs for actual tokens
        selected_log_probs = torch.gather(
            log_probs, -1, shifted_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return selected_log_probs.sum(dim=-1)
```

---

## 5. References

1. https://arxiv.org/abs/2203.02155 — "RLHF: Training Language Models to Follow Instructions with Human Feedback" (Christiano et al.)
2. https://huggingface.co/docs/trl — TRL (Transformers Reinforcement Learning) library
3. https://arxiv.org/abs/2310.12966 — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al.)
4. https://github.com/huggingface/trl — TRL official implementation
5. https://arxiv.org/abs/2306.13649 — "Red Teaming Language Models via Reinforcement Learning"
6. https://arxiv.org/abs/2306.14923 — "Self-Rewarding Language Models"
7. https://github.com/openai/human-eval — Human evaluation datasets
8. https://arxiv.org/abs/2209.07666 — "Training a Helpful and Harmless AI Assistant"
9. https://huggingface.co/datasets/allenai/real_toxicity_prompts — Toxicity datasets
10. https://arxiv.org/abs/2305.18290 — "Constitutional AI: Harmlessness from AI Feedback" (Bai et al.)
11. https://github.com/anthropic-ai/evals — Constitutional AI evaluation
12. https://arxiv.org/abs/2104.08821 — "SaFeRLHF: Safe Reinforcement Learning"
13. https://arxiv.org/abs/2303.04934 — "Beyond Reward: Preference-based Reinforcement Learning"
14. https://github.com/NVIDIA/NeMo-Aligner — NeMo alignment toolkit
15. https://arxiv.org/abs/2302.08582 — "OpenAssistant: A New Large Language Model Trained via RLHF"
16. https://github.com/LAION-AI/Open-Assistant — OpenAssistant RLHF data

---

## 6. Uncertainty and Limitations

**Not Covered:** Reward hacking, distributional shift in RLHF, multi-objective alignment. **Production:** Use human evaluators, implement fallback reward models, monitor for alignment drift.
