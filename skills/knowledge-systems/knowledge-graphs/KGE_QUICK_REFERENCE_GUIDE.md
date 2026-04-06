# Knowledge Graph Embedding: Quick Reference Guide

## 1. Method Selection Matrix

### Choose Your Method Based On:

```
┌─────────────────────────────────────────────────────────────────┐
│ YOUR REQUIREMENTS → RECOMMENDED METHOD                           │
├─────────────────────────────────────────────────────────────────┤
│ Speed + Simplicity                    → TransE or DistMult      │
│ Asymmetric Relations                  → RotatE or ComplEx       │
│ Accuracy (SOTA)                       → Annular Sectors (2026)  │
│ Flexibility + Expressiveness           → TuckER or RotatE       │
│ Large-scale Graphs (100M+ entities)   → SparseTransX           │
│ Temporal/Dynamic KGs                  → TS-align or QLGAN      │
│ Production Deployment                 → RotatE (proven)        │
│ Research/Experimentation              → TuckER or RotatE       │
│ Multimodal Knowledge                  → ConEx + visual encoder  │
│ Explainability                        → ComplEx or RotatE       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. One-Paragraph Summaries

### TransE (2013)
Simple translation-based model where h + r ≈ t. Fast and scalable but struggles with asymmetric relations and complex patterns. Good baseline, still used for large-scale graphs.

### TransH (2014)
Improves TransE by projecting entities onto relation-specific hyperplanes. Better for N-to-N relations. Slightly slower than TransE but more expressive.

### TransR (2015)
Each relation has its own projection matrix, maximizing expressiveness. Good performance but higher computational cost. Parameter count is main limitation.

### ComplEx (2016)
Uses complex-valued embeddings to naturally model asymmetric relations. Elegant mathematical framework. MRR ~0.412 on FB15k. Better than TransX methods on asymmetry.

### RotatE (2019)
Relations as rotations in complex space. Elegantly models symmetry, antisymmetry, inversion, and composition. Self-adversarial sampling improves training. SOTA on WN18RR (0.476 MRR). Top choice for production.

### TuckER (2019)
Tucker tensor decomposition for KGE. Most expressive model. Captures full n-way interactions. Best for accuracy (0.358 MRR on FB15k-237). More computational cost but worth it.

### Annular Sectors (2026)
Geometric representation of relations as annular sectors. Newest SOTA method. Best performance (0.365+ MRR on FB15k-237). Combines simplicity with accuracy.

---

## 3. Performance Quick Reference

### FB15k-237 (Most Used)

```
Model           | MRR   | H@1   | H@10  | Parameters | Speed
────────────────┼───────┼───────┼───────┼────────────┼──────
RotatE          | 0.338 | 0.241 | 0.556 | ~4M        | Fast
TuckER          | 0.358 | 0.253 | 0.575 | ~5M        | Medium
Annular (2026)  | 0.365 | 0.260 | 0.580 | ~4M        | Fast
ComplEx         | 0.315 | 0.221 | 0.517 | ~3M        | Fast
DistMult        | 0.281 | 0.189 | 0.460 | ~3M        | Fast
TransE          | 0.297 | 0.209 | 0.465 | ~3M        | Fast
```

### WN18RR (Reasoning Task)

```
Model           | MRR   | H@1   | H@10  | Best For
────────────────┼───────┼───────┼───────┼──────────
RotatE          | 0.476 | 0.413 | 0.723 | Reasoning
Annular (2026)  | 0.485 | 0.420 | 0.730 | SOTA
TuckER          | 0.470 | 0.392 | 0.710 | Flexible
ConEx           | 0.461 | 0.382 | 0.697 | Patterns
ComplEx         | 0.440 | 0.360 | 0.686 | Asymmetric
```

---

## 4. Mathematical Formulas (Single Line Each)

| Method | Scoring Function | Notes |
|--------|------------------|-------|
| TransE | \|\|h + r - t\|\|₂ | Simple translation |
| TransH | \|\|h⊥ + d_r - t⊥\|\|₂ | Hyperplane projection |
| TransR | \|\|M_r·h + r - M_r·t\|\|₂ | Relation-specific space |
| DistMult | h^T diag(r) t | Trilinear dot product |
| ComplEx | Re(<h, r*, t̄>) | Complex conjugate |
| RotatE | \|\|h ⊙ e^{iθ_r} - t\|\|₂ | Rotation in complex space |
| TuckER | W ×₁ E_h ×₂ E_r ×₃ E_t | Tucker decomposition |

---

## 5. Code Snippets

### Quick Implementation (RotatE)

```python
import torch
import torch.nn as nn

class SimpleRotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=256):
        super().__init__()
        self.h = nn.Embedding(num_entities, dim)
        self.r = nn.Embedding(num_relations, dim)
        
    def forward(self, h_idx, r_idx, t_emb):
        h = self.h(h_idx)  # Entity embedding
        r = self.r(r_idx)  # Relation angle
        # Rotation: h * e^{i*r} - t
        return torch.norm(h * torch.exp(1j * r) - t_emb, dim=1)
```

### Quick Evaluation

```python
def evaluate(model, test_triples):
    mrr, hits1, hits10 = 0, 0, 0
    for h, r, t in test_triples:
        scores = [model(h, r, t_cand) for t_cand in all_entities]
        rank = 1 + (scores < scores[t]).sum()
        mrr += 1/rank
        if rank <= 1: hits1 += 1
        if rank <= 10: hits10 += 1
    return mrr/len(test_triples), hits1/len(test_triples), hits10/len(test_triples)
```

### Negative Sampling

```python
def negative_sample(h, r, t, num_entities, batch_size):
    # Randomly corrupt head or tail
    corrupt_h = torch.rand(batch_size) < 0.5
    h_neg = torch.where(corrupt_h, torch.randint(0, num_entities, h.shape), h)
    t_neg = torch.where(corrupt_h, t, torch.randint(0, num_entities, t.shape))
    return h_neg, r, t_neg
```

---

## 6. Training Loop (Pseudocode)

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Get positive triples
        h_pos, r, t_pos = batch
        
        # Generate negatives
        h_neg, _, t_neg = negative_sample(h_pos, r, t_pos)
        
        # Forward pass
        pos_score = model(h_pos, r, t_pos)
        neg_score = model(h_neg, r, t_neg)
        
        # Margin loss: max(0, margin + neg - pos)
        loss = torch.relu(margin + neg_score - pos_score).mean()
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Normalize entity embeddings
        model.normalize_entities()
    
    # Evaluate on validation
    if (epoch + 1) % 10 == 0:
        metrics = evaluate(model, valid_triples)
        print(f"Epoch {epoch+1}: MRR={metrics[0]:.3f}")
```

---

## 7. Hyperparameter Guidelines

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Embedding Dim | 64-512 | 256 is common default |
| Margin (γ) | 1.0-30.0 | RotatE: 12.0; DistMult: 5.0 |
| Learning Rate | 1e-4 to 1e-2 | Adam recommended |
| Batch Size | 256-2048 | Larger = better but needs GPU |
| Epochs | 100-500 | Early stopping recommended |
| Negative Ratio | 1:1 | One negative per positive |
| Dropout | 0.0-0.3 | Use if overfitting |
| L2 Weight | 1e-5 to 1e-3 | Regularization strength |

---

## 8. Benchmark Statistics

### FB15k-237
- **Size**: 272K train, 17.5K valid, 20.5K test triples
- **Entities**: 14,541
- **Relations**: 237
- **Link Density**: ~0.10%
- **Challenge**: Removed test leakage from FB15k
- **Use**: Standard for accuracy comparison

### WN18RR
- **Size**: 86.8K train, 5.16K valid, 5.01K test
- **Entities**: 40,943
- **Relations**: 11
- **Challenge**: Heavy relational reasoning required
- **Use**: Tests composition/reasoning ability
- **Note**: Removed inverse relations

### YAGO3-10
- **Size**: 1.08M training triples
- **Entities**: 123,182
- **Use**: Scalability testing
- **Note**: Larger, less commonly used

---

## 9. Decision Tree for Model Selection

```
Start: Choosing a KGE Model
    |
    ├─ "I need SOTA accuracy" → Use Annular Sectors (2026) or TuckER
    |
    ├─ "I need speed" → Use TransE or DistMult
    |
    ├─ "I have asymmetric relations" → Use RotatE or ComplEx
    |   |
    |   └─ "Need SOTA too?" → Use RotatE
    |
    ├─ "Scale is 100M+ entities" → Use SparseTransX or RotatE
    |
    ├─ "Need interpretability" → Use RotatE or ComplEx
    |
    ├─ "Graph is temporal/dynamic" → Use TS-align or QLGAN
    |
    └─ "This is production code" → Use RotatE (proven, stable)
```

---

## 10. Common Pitfalls and Solutions

| Problem | Solution |
|---------|----------|
| Model not converging | Lower learning rate, check negative sampling |
| Memory issues | Reduce batch size or embedding dimension |
| Poor validation metrics | Increase epochs, add regularization |
| Overfitting | Add L2 regularization, use dropout |
| Slow training | Use GPU acceleration, reduce dimension |
| Test leakage | Use filtered evaluation protocol |
| Embedding collapse | Normalize after each step, increase margin |
| Unstable training | Use gradient clipping, reduce learning rate |

---

## 11. Evaluation Metrics Explained

### Mean Rank (MR)
- Average rank of correct entity
- **Lower is better**
- Range: 1 to num_entities
- Problem: Sensitive to outliers

### Mean Reciprocal Rank (MRR)
- Average of 1/rank
- **Higher is better** (0-1 range)
- More robust to outliers
- Preferred metric

### Hits@K
- % of correct entities in top-K
- Hits@1, Hits@3, Hits@10 common
- **Higher is better**
- More interpretable

### Example
```
Triple: (Alice, knows, Bob)
Predictions: [Bob(1), Charlie(5), David(27)]
- Rank: 1
- MRR: 1/1 = 1.0
- Hits@1: 1 (correct in top-1)
- Hits@10: 1 (correct in top-10)
```

---

## 12. Timeline and Milestones

```
2013: TransE introduced (game changer)
      ↓
2014: TransH, DistMult (first improvements)
      ↓
2015: TransR, TransD (relation-specific)
      ↓
2016: ComplEx (complex numbers)
      ↓
2017-2018: ConvE, ConvKB (neural methods)
      ↓
2019: RotatE (rotation paradigm), TuckER (tensor)
      ↓
2020-2021: Standardization, ConEx, refinements
      ↓
2022-2023: Specialization (temporal, multimodal)
      ↓
2024-2026: Integration & Optimization
           - SparseTransX (2025)
           - CKRHE (2025)
           - QLGAN (2026)
           - Annular Sectors (2026)
```

---

## 13. Production Deployment Checklist

- [ ] Choose method (RotatE for safety, Annular for SOTA)
- [ ] Prepare data (remove duplicates, normalize)
- [ ] Train on full dataset
- [ ] Evaluate on test set
- [ ] Implement inference API
- [ ] Add GPU acceleration if needed
- [ ] Set up monitoring/metrics
- [ ] Document hyperparameters used
- [ ] Version control trained model
- [ ] Load test with expected QPS
- [ ] Create fallback strategy
- [ ] Document link prediction methodology

---

## 14. Resources and Links

### Code Repositories
- PyKEEN: https://github.com/pykeen/pykeen (comprehensive)
- OpenKE: https://github.com/thunlp/OpenKE (original)
- Cornac: https://github.com/PreferredAI/cornac (scalable)

### Datasets
- FB15k-237: GitHub/OpenKE
- WN18RR: Microsoft/ConvE repo
- YAGO3: https://www.mpi-inf.mpg.de/yago/

### Papers
- RotatE: https://openreview.net/forum?id=HkgqRkBFDB
- TuckER: https://aclanthology.org/D19-1522/
- TransE: ICML 2013 proceedings

---

## 15. FAQ

**Q: Which model should I use?**  
A: Start with RotatE. It's proven, fast, and gives SOTA on many benchmarks.

**Q: How long does training take?**  
A: FB15k-237: 1-2 hours on GPU. WN18RR: 30 min. Scales with data size.

**Q: Can I use pre-trained embeddings?**  
A: Yes, most code supports loading checkpoints. Fine-tune if needed.

**Q: What embedding dimension to use?**  
A: Start with 256. Larger (512) for bigger graphs, smaller (64) for speed.

**Q: How do I know if my model is good?**  
A: Compare MRR to benchmarks. 0.30+ is decent, 0.35+ is good, 0.38+ is excellent.

**Q: Should I use filtered or raw evaluation?**  
A: Filtered is more realistic (removes test leakage). Use filtered.

**Q: How do I handle new entities?**  
A: Train new embeddings, or use similarity to known entities.

**Q: Can I combine multiple KGE models?**  
A: Yes, ensemble scores. Weight by confidence/MRR.

---

## 16. Glossary

- **Triple**: (head, relation, tail) - basic knowledge unit
- **Link Prediction**: Predicting missing triples
- **Entity Alignment**: Matching entities across KGs
- **Embedding**: Vector representation of entity/relation
- **Scoring Function**: f_r(h,t) - measures triple plausibility
- **Margin**: Separation parameter in ranking loss
- **MRR**: Mean Reciprocal Rank - main evaluation metric
- **Hits@K**: Percentage correct in top-K predictions
- **Negative Sampling**: Creating false triples for training
- **Regularization**: Penalty to prevent overfitting
- **Normalization**: Ensuring embeddings stay bounded

---

## 17. One-Page Study Guide

**Understand KGE in 30 minutes:**
1. Knowledge graph = set of (entity, relation, entity) triples (2 min)
2. Embedding = vector representation (2 min)
3. Scoring = measuring triple plausibility with distance (2 min)
4. Training = minimize distance for true triples, maximize for false (3 min)
5. TransE = simple: h + r ≈ t (2 min)
6. RotatE = elegant: rotation in complex space (3 min)
7. TuckER = expressive: tensor factorization (3 min)
8. Evaluation = rank true entity, compute MRR (3 min)
9. Applications = link prediction, recommendations, QA (3 min)
10. Practice = implement and train a model (5 min)

---

## 18. Citation Templates

### For RotatE
```
@inproceedings{sun2019rotate,
  title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
  author={Sun, Zhiqing and Deng, Zhi-Hong and Nie, Jian-Yun and Tang, Jian},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

### For TransE
```
@inproceedings{bordes2013translating,
  title={Translating Embeddings for Modeling Relations in Structured Data},
  author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
  booktitle={International Conference on Machine Learning},
  year={2013}
}
```

---

**Last Updated**: April 2026  
**Quick Reference Version**: 2.0  
**Complete Reference**: See KGE_COMPREHENSIVE_DOCUMENTATION.md
