# Model Distillation: Compressing LLMs for Efficient Inference
**Author:** Shuvam Banerji Seal  
**Version:** 1.0  
**Date:** April 2026  
**Status:** Production-Ready Skill Documentation

---

## Problem Statement

**Compression Challenge:** 70B model runs on 4 GPUs ($120/hr). Can we get 90-95% quality on 1 GPU ($30/hr)?

**Solution:** Knowledge distillation transfers learned representations from large teacher to smaller student.

---

## Mathematical Foundations

### 1. Distillation Loss

$$L = \alpha \times \text{KL}(p_{\text{teacher}}, p_{\text{student}}) + (1-\alpha) \times \text{CE}(y, p_{\text{student}})$$

Where:
- α = 0.7 (weight KL loss more heavily)
- KL = Kullback-Leibler divergence (probability matching)
- CE = cross-entropy (task accuracy)

### 2. Temperature-Based Softening

$$p_{\text{soft}} = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$

**Effect of Temperature:**
- T=1 (sharp): Focus on correct class
- T=3 (soft): Distribute probability across classes, smoother gradients
- T=10 (very soft): Nearly uniform, emphasizes distillation

**Optimal:** T=3-5 for most tasks

### 3. Knowledge Transfer Effectiveness

$$\text{Quality\_Retention} = 1 - \frac{\text{Error}_{\text{student}}}{\text{Error}_{\text{teacher}}}$$

**Empirical Formula:**
$$\text{Compression\_Ratio} = \frac{\text{Parameters}_{\text{teacher}}}{\text{Parameters}_{\text{student}}}$$

**Typical Results:**
- 2x compression: 98% quality retention
- 5x compression: 93% quality retention
- 10x compression: 85-90% quality retention

---

## Core Concepts

### 1. Standard Distillation

```
Teacher Model (70B): Trained on task
    ↓
Generate soft targets (logits with temperature)
    ↓
Student Model (7B): Learn from teacher's soft targets
    ↓
Loss = α × KL(teacher, student) + (1-α) × CE(labels, student)
```

**Advantages:**
- Best quality retention (95-98%)
- General-purpose (works for any task)

### 2. Layer-Wise Distillation

```
Match intermediate layers:
- Teacher Layer_i → Student Layer_j (different depths)
- MSE loss: ||activations_teacher - activations_student||^2

Result: Student learns internal representations, not just outputs
```

### 3. Attention Head Distillation

```
Transfer attention patterns:
- Teacher Attention_ij → Student Attention_ij
- Loss: ||attention_teacher - attention_student||^2

Useful for: Understanding why model makes decisions
```

### 4. Quantization-Aware Distillation

```
Two approaches:
1. Sequential: Distill → Quantize (FP32→INT8)
2. Joint: Distill while constraining to INT8 values

Result: Extreme compression (175B → 1B possible)
```

---

## Implementation Guide

### Step 1: Setup Teacher and Student

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load teacher
teacher_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    torch_dtype=torch.float16,
    device_map="cuda:0"
)

# Initialize student (smaller architecture)
student_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",  # or custom 7B student
    torch_dtype=torch.float16,
    device_map="cuda:1"
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
)
```

### Step 2: Define Distillation Loss

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, 
                     temperature=3.0, alpha=0.7):
    """
    Compute distillation loss.
    
    Args:
        student_logits: [batch, seq, vocab]
        teacher_logits: [batch, seq, vocab]
        labels: [batch, seq]
        temperature: Softening parameter
        alpha: Weight for KL divergence
    
    Returns:
        Scalar loss value
    """
    
    # Soft targets from teacher
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence loss
    kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
    kl_loss = kl_loss * (temperature ** 2)  # Scale by T^2
    
    # Cross-entropy loss with hard labels
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)),
                              labels.view(-1))
    
    # Combined loss
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
    
    return total_loss
```

### Step 3: Training Loop with HuggingFace Trainer

```python
from transformers import Trainer, TrainingArguments

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=3.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Compute distillation loss
        loss = distillation_loss(
            student_logits, 
            teacher_logits, 
            inputs['labels'],
            temperature=self.temperature,
            alpha=0.7
        )
        
        return (loss, outputs) if return_outputs else loss

# Setup trainer
training_args = TrainingArguments(
    output_dir="./distilled-llama-7b",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=1000,
)

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    teacher_model=teacher_model,
    temperature=3.0
)

trainer.train()
```

### Step 4: Intermediate Layer Distillation

```python
class LayerWiseDistillationTrainer(Trainer):
    def __init__(self, teacher_model, layer_mappings, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        # layer_mappings: {teacher_layer_idx: student_layer_idx}
        self.layer_mappings = layer_mappings
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get full model outputs with hidden states
        student_outputs = model(
            **inputs,
            output_hidden_states=True
        )
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **inputs,
                output_hidden_states=True
            )
        
        # Output logits loss
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        output_loss = distillation_loss(
            student_logits,
            teacher_logits,
            inputs['labels'],
            temperature=3.0,
            alpha=0.5
        )
        
        # Intermediate layer losses
        intermediate_loss = 0.0
        for teacher_idx, student_idx in self.layer_mappings.items():
            teacher_hidden = teacher_outputs.hidden_states[teacher_idx]
            student_hidden = student_outputs.hidden_states[student_idx]
            
            # MSE loss between hidden states
            layer_loss = F.mse_loss(student_hidden, teacher_hidden)
            intermediate_loss += layer_loss
        
        total_loss = output_loss + 0.1 * intermediate_loss
        
        return (total_loss, student_outputs) if return_outputs else total_loss
```

### Step 5: Evaluation and Comparison

```python
def evaluate_distillation(teacher_model, student_model, test_dataset):
    """Compare teacher and student performance."""
    
    teacher_model.eval()
    student_model.eval()
    
    teacher_correct = 0
    student_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_dataset:
            teacher_out = teacher_model(**batch)
            student_out = student_model(**batch)
            
            teacher_preds = teacher_out.logits.argmax(dim=-1)
            student_preds = student_out.logits.argmax(dim=-1)
            
            labels = batch['labels']
            
            teacher_correct += (teacher_preds == labels).sum().item()
            student_correct += (student_preds == labels).sum().item()
            total += labels.numel()
    
    teacher_acc = teacher_correct / total
    student_acc = student_correct / total
    
    print(f"Teacher Accuracy: {teacher_acc:.4f}")
    print(f"Student Accuracy: {student_acc:.4f}")
    print(f"Quality Retention: {student_acc/teacher_acc:.2%}")
    print(f"Speedup: {teacher_model.numel()/student_model.numel():.1f}x")
```

---

## Performance Analysis

### 1. Compression vs Quality Trade-off

| Compression | Student Size | Quality Retention | Speedup | Use Case |
|------------|-------------|------------------|---------|----------|
| **2x** | 35B | 97-98% | 1.8x | Production (minimal loss) |
| **5x** | 14B | 93-95% | 4.5x | Standard (acceptable loss) |
| **10x** | 7B | 85-90% | 8x | Edge/mobile |
| **20x** | 3.5B | 75-85% | 15x | Extreme (significant loss) |

### 2. Training Cost

| Teacher | Student | Distillation Cost | Speedup |
|---------|---------|------------------|---------|
| 70B | 35B | 20-30% of teacher | 1.8x |
| 70B | 14B | 15-25% of teacher | 4.5x |
| 70B | 7B | 10-20% of teacher | 8x |

**Key Finding:** Distillation costs much less than training from scratch.

### 3. Quality Retention Curve

```
Quality Retention (%)
100%  ████████████ Teacher (70B)
       ██████████ Student-2x (35B)
95%    ████████ Student-5x (14B)
       ██████ Student-10x (7B)
85%    ████ Student-20x (3.5B)
       ██
75%
       1x  2x  5x  10x  20x
       Compression Ratio
```

---

## Real-World Examples

### Example 1: Cost-Optimized Production

**Problem:** 70B model on 4 GPUs costs $120/hr, but only using 30% capacity

**Solution:** Distill 70B → 7B
```
Performance:
- Distilled model: 93% quality retention
- Can fit on 1 GPU: $30/hr (75% cost reduction)
- Speedup: 8x per inference
- Total: 60x effective throughput improvement

Result: Process 1000s more requests per day
```

### Example 2: Edge Deployment

**Problem:** Deploy LLM on mobile phones (4GB RAM)

**Solution:** Chain distillation
```
Step 1: 70B → 7B (8x compression)
        Training: 100k tokens
        Quality: 93%

Step 2: 7B → 1B (7x compression)
        Training: 50k tokens
        Quality: 85% relative to 70B (89% relative to 7B)

Result: 1B model on phone
        Latency: 100-200ms per token
        Memory: 2-3GB
```

### Example 3: Multi-Model Distillation

**Problem:** Serve different models for different latency SLAs

**Solution:** Create distillation ladder
```
Models available:
- 70B (full quality, slow) - premium users
- 35B (97% quality, medium) - standard users
- 7B (93% quality, fast) - cost-conscious users
- 1B (85% quality, fastest) - free tier

Cost distribution:
- 5% use 70B: Premium revenue
- 30% use 35B: Main revenue
- 40% use 7B: Volume
- 25% use 1B: Free (attracts users)

Infrastructure: 1x 70B + 2x 35B + 4x 7B + 8x 1B
Cost: $60/hour (vs $1200+ for all full-size)
Coverage: 100% of users
```

---

## Integration Guide

### With HuggingFace Hub

```python
# Upload distilled model
student_model.push_to_hub("my-distilled-llama-7b")

# Later: Load from hub
model = AutoModelForCausalLM.from_pretrained("my-distilled-llama-7b")
```

### With vLLM

```python
from vllm import LLM

# Use distilled model directly
llm = LLM(
    model="my-distilled-llama-7b",
    dtype="float16"
)

outputs = llm.generate(prompts)
```

---

## Sources and Citations

### 1. **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**
- **Authors:** Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf
- **Organization:** Hugging Face
- **ArXiv:** 1910.01108
- **Published:** October 2, 2019
- **Key:** 40% parameter reduction, 60% speedup, 97% BERT performance

### 2. **Everything You Need to Know about Knowledge Distillation**
- **Source:** HuggingFace Blog
- **Author:** Kseniase
- **Date:** March 6, 2025
- **URL:** https://huggingface.co/blog/Kseniase/kd

### 3. **Knowledge Distillation: Teacher-Student Training for LLMs**
- **Author:** Michael Brenndoerfer
- **Date:** February 24, 2026
- **URL:** https://mbrenndoerfer.com/writing/knowledge-distillation-temperature-teacher-student-llm

---

**End of Skill Documentation**

**Integration Status:** Ready for production  
**Recommended Phase:** 3 (Specialized)  
**Estimated Learning Time:** 4-5 hours  
**Code Examples:** 15+ provided
