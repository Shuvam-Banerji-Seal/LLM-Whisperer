# Learning Rate Scheduling Skill Document - Comprehensive Guide

## Overview

A production-ready, comprehensive skill document for learning rate scheduling in LLM training has been created at:

**Location**: `/home/shuvam/codes/LLM-Whisperer/skills/training-optimization/learning-rate-scheduling.prompt.md`

**Size**: 46 KB | **Lines**: 1,566 | **Sections**: 12 main chapters + 86 subsections

---

## What's Included

### 1. **Schedule Types** (6 complete methods)
- ✅ Linear Warmup + Linear Decay
- ✅ Exponential Decay (η = η₀ × exp(-t/τ))
- ✅ Polynomial Decay (adjustable degree)
- ✅ Cosine Annealing (SGDR paper, most popular for LLMs)
- ✅ Inverse Square Root (Transformer paper)
- ✅ Step-Based Schedules (discrete reductions)

Each includes:
- Mathematical formulation with detailed explanations
- Why it works and when to use it
- Typical hyperparameters
- Advantages and disadvantages
- Code examples

### 2. **Warmup Strategies** (5 approaches)
- ✅ Linear Warmup (most common, BERT-style)
- ✅ Square Root Warmup (for large batches)
- ✅ Exponential Warmup (maximum early protection)
- ✅ Theoretical justification (task head init, gradient variance, Adam statistics)
- ✅ Duration selection guidelines (5-20% of training)

### 3. **Advanced Methods** (4 techniques)
- ✅ Layer-Wise Learning Rate Decay (LLRD) - critical for fine-tuning
  - Formula: η_l = η_base × ξ^(L-l)
  - Complete implementation with parameter groups
  - Decay factor selection (0.85-1.0)
  
- ✅ Cyclic Learning Rates (CLR)
  - Oscillating schedules with min/max bounds
  - When to use and typical parameters
  
- ✅ Warm Restarts (SGDR)
  - Periodic resets with optional period growth
  - Multiple exploration phases
  
- ✅ Learning Rate Range Test (LR Finder)
  - Empirical method to find optimal LR bounds
  - Implementation and interpretation guide

### 4. **Implementation Code** (6+ runnable examples)
- ✅ PyTorch native schedulers
- ✅ HuggingFace Trainer configurations (simple + advanced)
- ✅ Custom scheduler from scratch with full class definition
- ✅ LLRD parameter group setup
- ✅ Integration with AdamW, LAMB, SGD, Adafactor
- ✅ Complete training loops with logging

### 5. **Empirical Analysis**
- ✅ Convergence curves for different schedules
- ✅ Warmup duration effects (0-20%, with performance impact)
- ✅ Schedule interaction with batch size (8 to 512+ examples)
- ✅ Model size and learning rate scaling laws
- ✅ Performance comparison tables

### 6. **Best Practices** (5+ use cases)
- Pre-training from scratch
- Fine-tuning (primary LLM use case)
- Domain adaptation with large domain shift
- Few-shot learning with limited data
- Continued pre-training on new domain

For each:
- Recommended schedule
- Typical learning rates
- Warmup duration
- Special considerations

### 7. **Debugging Guide**
Symptom-solution pairs for:
- Loss diverges/NaN
- Loss plateaus
- Large oscillations
- Overfitting

### 8. **References** (8+ papers)
- SGDR: Stochastic Gradient Descent with Warm Restarts (2016)
- Attention Is All You Need (2017)
- BERT: Pre-training of Deep Bidirectional Transformers (2018)
- Universal Language Model Fine-tuning (2018)
- Analyzing & Reducing Need for Warmup in GPT Training (2024)
- Fine-tuning Learning Rates: LLRD Guide (2025)
- Plus implementation resources and tools

---

## Key Features

### Mathematical Rigor
- 10+ detailed mathematical equations
- Step-by-step derivations
- Practical numerical examples
- Clear notation explanations

### Practical Code
- 6+ complete, runnable code examples
- PyTorch implementations
- HuggingFace integration
- Custom implementations from scratch

### Comprehensive Data
- 5+ reference tables
- Performance comparison charts
- Hyperparameter ranges by use case
- Scaling laws for different model sizes

### Actionable Guidance
- Decision tree for selecting schedules
- Quick reference cheat sheet
- Hyperparameter tuning strategy
- Monitoring and logging setup
- Debugging troubleshooting guide

---

## Quick Start

### For Fine-tuning (Most Common)
```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Training loop
for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
```

### For Advanced Fine-tuning with LLRD
See Section 5 "Implementation Guide" for complete LLRD parameter group setup.

---

## Document Statistics

| Category | Count |
|---|---|
| Main Sections | 12 |
| Subsections | 86 |
| Code Examples | 6+ |
| Mathematical Formulas | 10+ |
| Reference Tables | 5+ |
| Papers Referenced | 8+ |
| Lines of Content | 1,566 |
| File Size | 46 KB |

---

## Reading Recommendations

### For Beginners
1. Introduction + Why It Matters
2. Linear Warmup + Linear Decay
3. Best Practices section
4. Quick Reference cheat sheet

### For Intermediate Practitioners
1. All schedule types
2. LLRD section (critical for fine-tuning)
3. Implementation guide
4. Debugging section

### For Advanced Practitioners
1. All sections
2. Original papers (SGDR, BERT, Transformers)
3. Custom implementation examples
4. Recent papers (2024-2026)

---

## Current State of the Art (2026)

Based on latest research integrated:

1. **Linear warmup + cosine decay** is the gold standard
2. **LLRD (layer-wise learning rate decay)** improves fine-tuning by 2-5%
3. **Cosine annealing** outperforms linear decay for LLMs
4. **Warmup necessity** depends on batch size and model size
5. **Learning rate** scales inversely with model size
6. **Batch size** affects both optimal LR and warmup duration

---

## Integration with Existing Skills

This document complements other training optimization skills:
- `mixed-precision-training.prompt.md` - FP16/BF16 training
- `gradient-accumulation-checkpointing.prompt.md` - Memory optimization

Together they form a complete guide to efficient LLM training.

---

## How to Use This Skill

1. **For new projects**: Start with "Quick Start" and "Best Practices"
2. **For troubleshooting**: Use "Debugging Guide"
3. **For deep learning**: Study specific schedule types with math
4. **For implementation**: Copy code examples from "Implementation Guide"
5. **For research**: Check "References" for original papers

---

## Notes

- All code examples are tested patterns from 2024-2026 practice
- Learning rates are typical ranges; always validate with LR finder
- Recommend starting with recommended defaults, then tuning
- Monitor learning rate with loss curves to diagnose problems
- Adjust based on model size, task difficulty, and data domain

---

Created: April 6, 2026
Document Version: 1.0
Status: Production Ready
