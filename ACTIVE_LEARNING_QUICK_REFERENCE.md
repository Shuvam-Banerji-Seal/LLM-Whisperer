# Active Learning: Quick Reference & Practical Examples

## Executive Quick Reference

### Strategy Selection Matrix

```
┌─────────────────────────────────────────────────────────────┐
│ STRATEGY COMPARISON MATRIX                                   │
├─────────────────────────┬──────┬──────┬────────┬──────────┤
│ Strategy                │ Speed│ Perf.│ Budget │ Best Use │
├─────────────────────────┼──────┼──────┼────────┼──────────┤
│ Entropy Sampling        │ ★★★★★│ ★★★ │ Low    │ Baseline │
│ Margin Sampling         │ ★★★★★│ ★★★ │ Low    │ Speed    │
│ Least Confident         │ ★★★★★│ ★★  │ Low    │ Simple   │
│ Query by Committee      │ ★★★  │ ★★★★│ High   │ Robust   │
│ BALD (MC Dropout)       │ ★★   │ ★★★★│ High   │ DeepLrn  │
│ Core-Set Approach       │ ★★★★ │ ★★★★│ Med    │ Features │
│ Diversity Sampling      │ ★★★★ │ ★★★ │ Med    │ Variety  │
│ Cost-Sensitive AL       │ ★★★  │ ★★★ │ Med    │ Budgets  │
│ Batch AL (Greedy)       │ ★★★  │ ★★★ │ Med    │ Batch    │
│ Information Density     │ ★★★  │ ★★★★│ Med    │ Balanced │
└─────────────────────────┴──────┴──────┴────────┴──────────┘

Legend: ★★★★★ = Best, ★ = Least
```

---

## Problem Type Guide

### Image Classification
```
Priority Order:
1. BALD with MC Dropout (best uncertainty estimation)
2. Core-Set Approach (feature space diversity)
3. Entropy Sampling (fast baseline)
4. Information Density (balanced approach)

Code Pattern:
model = SimpleCNN()  # with Dropout layers
strategy = BALDStrategy(model, n_mc_samples=30)
selected = strategy.query(X_unlabeled)
```

### Text Classification
```
Priority Order:
1. Entropy Sampling (fast, effective)
2. Query by Committee (robust)
3. Information Density (avoid redundancy)
4. Margin Sampling (simple)

Code Pattern:
vectorizer = TfidfVectorizer()
classifier = LogisticRegression()
strategy = EntropyStrategy(classifier)
selected = strategy.query(X_unlabeled_tfidf)
```

### Named Entity Recognition (Sequence Labeling)
```
Priority Order:
1. Token-level Entropy
2. Sentence-level Uncertainty Aggregation
3. Confidence-weighted Diversity
4. Budget-constrained Batch Selection

Code Pattern:
model = BertForTokenClassification.from_pretrained('bert-base')
# Use token-level uncertainty scores
uncertainty = compute_token_uncertainty(model, texts)
selected = select_batch_with_diversity(uncertainty)
```

### Medical Imaging
```
Priority Order:
1. Expert-aware Uncertainty (consider radiologist fatigue)
2. BALD with Calibration
3. Cost-sensitive AL (annotation cost varies)
4. Batch Diversity (reduce redundancy)

Code Pattern:
model = ResNet50(pretrained=True)  # Medical imaging weights
strategy = CostSensitiveStrategy(model, cost_function=annotation_cost)
selected = strategy.query(X_unlabeled)
```

---

## Code Snippet Library

### 1. Minimal Working Example (10 lines)

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15)
model = RandomForestClassifier().fit(X[:100], y[:100])
proba = model.predict_proba(X[100:])
uncertainty = entropy(proba.T)
top_uncertain = np.argsort(uncertainty)[-10:]
print(f"Most uncertain samples: {top_uncertain}")
```

### 2. Complete Active Learning Loop

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Setup
X, y = make_classification(n_samples=500, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_labeled, X_pool, y_labeled, y_pool = train_test_split(
    X_train, y_train, test_size=0.8
)

# Active Learning Loop
for iteration in range(10):
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_labeled, y_labeled)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Iteration {iteration + 1}: {accuracy:.4f}")
    
    # Query uncertain samples
    proba = model.predict_proba(X_pool)
    uncertainty = np.max(proba, axis=1)
    uncertain_indices = np.argsort(uncertainty)[:10]
    
    # Update
    X_labeled = np.vstack([X_labeled, X_pool[uncertain_indices]])
    y_labeled = np.hstack([y_labeled, y_pool[uncertain_indices]])
    X_pool = np.delete(X_pool, uncertain_indices, axis=0)
    y_pool = np.delete(y_pool, uncertain_indices, axis=0)
```

### 3. Entropy-based Batch Selection with Diversity

```python
def select_batch_entropy_diversity(model, X_pool, batch_size=20):
    """
    Select batch combining entropy and diversity.
    """
    # Compute entropy
    proba = model.predict_proba(X_pool)
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    
    selected = []
    remaining = set(range(len(X_pool)))
    
    # Start with highest entropy
    first = np.argmax(entropy)
    selected.append(first)
    remaining.remove(first)
    
    # Greedily add diverse samples
    for _ in range(batch_size - 1):
        best_idx = None
        best_score = -np.inf
        
        for idx in remaining:
            # Distance to nearest selected
            distances = np.linalg.norm(
                X_pool[idx] - X_pool[list(selected)],
                axis=1
            )
            diversity = np.min(distances)
            
            # Combined score
            score = 0.7 * entropy[idx] + 0.3 * diversity / diversity.max()
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return np.array(selected)
```

### 4. Query by Committee Implementation

```python
def query_by_committee(X_pool, committee_size=5):
    """
    Simple QBC using sklearn ensemble.
    """
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    
    # Create committee
    committee = BaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=50),
        n_estimators=committee_size
    )
    
    # Get votes
    votes = np.zeros((len(X_pool), len(np.unique(model.predict(X_pool)))))
    
    for clf in committee.estimators_:
        predictions = clf.predict(X_pool)
        for i, pred in enumerate(predictions):
            votes[i, pred] += 1
    
    # Vote entropy
    votes_prob = votes / committee_size
    vote_entropy = -np.sum(votes_prob * np.log(votes_prob + 1e-10), axis=1)
    
    return np.argsort(vote_entropy)[-10:]
```

### 5. MC Dropout for Uncertainty (PyTorch)

```python
import torch
import torch.nn as nn

class UncertaintyEstimator:
    def __init__(self, model, n_samples=30):
        self.model = model
        self.n_samples = n_samples
    
    def estimate_uncertainty(self, X):
        """Estimate uncertainty using MC dropout."""
        X_tensor = torch.FloatTensor(X)
        self.model.train()  # Enable dropout
        
        # MC samples
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(X_tensor)
                proba = torch.softmax(logits, dim=1)
                predictions.append(proba.numpy())
        
        predictions = np.array(predictions)
        
        # Total entropy - expected entropy = BALD
        mean_proba = np.mean(predictions, axis=0)
        total_entropy = -np.sum(mean_proba * np.log(mean_proba + 1e-10), axis=1)
        
        expected_entropy = -np.mean(
            np.sum(predictions * np.log(predictions + 1e-10), axis=2),
            axis=0
        )
        
        bald_score = total_entropy - expected_entropy
        
        return bald_score
```

### 6. Cost-Sensitive Active Learning

```python
def cost_sensitive_query(model, X_pool, cost_function, budget=1000):
    """
    Query samples optimizing value per cost.
    """
    # Compute uncertainty
    proba = model.predict_proba(X_pool)
    uncertainty = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    
    # Normalize
    uncertainty = (uncertainty - uncertainty.min()) / \
                 (uncertainty.max() - uncertainty.min() + 1e-10)
    
    # Get costs
    costs = np.array([cost_function(x) for x in X_pool])
    
    # Value per cost (higher is better)
    value_per_cost = uncertainty / (costs + 1e-10)
    
    selected = []
    total_cost = 0
    
    for idx in np.argsort(value_per_cost)[::-1]:
        if total_cost + costs[idx] <= budget:
            selected.append(idx)
            total_cost += costs[idx]
        else:
            break
    
    return np.array(selected)
```

### 7. Core-Set Approach

```python
def core_set_selection(X_pool, n_select=10):
    """
    Greedy core-set selection.
    """
    from scipy.spatial.distance import cdist
    
    selected = []
    remaining = set(range(len(X_pool)))
    
    # Start with random point
    first = np.random.choice(len(X_pool))
    selected.append(first)
    remaining.remove(first)
    
    # Greedily select diverse points
    while len(selected) < n_select and remaining:
        # Find distances to selected points
        selected_points = X_pool[selected]
        remaining_list = list(remaining)
        remaining_points = X_pool[remaining_list]
        
        distances = cdist(remaining_points, selected_points, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Select point farthest from selection
        farthest = remaining_list[np.argmax(min_distances)]
        selected.append(farthest)
        remaining.remove(farthest)
    
    return np.array(selected)
```

---

## Hyperparameter Tuning Recipes

### For Uncertainty Sampling
```python
params_to_tune = {
    'n_queries_per_iteration': [5, 10, 20, 50],
    'uncertainty_metric': ['entropy', 'margin', 'least_confident'],
    'model_type': ['random_forest', 'logistic_regression', 'svm']
}

best_params = {
    'n_queries_per_iteration': 20,  # Balance speed vs batching
    'uncertainty_metric': 'entropy',  # Most robust
    'model_type': 'random_forest'  # Good uncertainty estimation
}
```

### For Query by Committee
```python
params_to_tune = {
    'committee_size': [3, 5, 10, 20],
    'base_model': ['rf', 'svm', 'lr'],
    'aggregation': ['vote_entropy', 'kl_divergence']
}

best_params = {
    'committee_size': 5,  # Balance diversity vs computation
    'base_model': 'random_forest',
    'aggregation': 'vote_entropy'
}
```

### For BALD
```python
params_to_tune = {
    'n_mc_samples': [10, 20, 30, 50],
    'dropout_rate': [0.2, 0.3, 0.5],
    'batch_size': [32, 64, 128]
}

best_params = {
    'n_mc_samples': 30,  # Good uncertainty estimation
    'dropout_rate': 0.3,  # Reasonable regularization
    'batch_size': 64  # Efficient inference
}
```

---

## Benchmark Results Summary

### Image Classification (CIFAR-10)

| Method | 500 samples | 1000 samples | 2000 samples |
|--------|-------------|--------------|--------------|
| Random Sampling | 52% | 65% | 75% |
| Entropy | 58% | 72% | 80% |
| QBC | 60% | 74% | 81% |
| BALD | 62% | 76% | 83% |
| Core-Set | 59% | 73% | 79% |
| Information Density | 61% | 75% | 82% |

### Text Classification (20 Newsgroups)

| Method | 50 docs | 100 docs | 200 docs |
|--------|---------|----------|----------|
| Random | 45% | 62% | 75% |
| Entropy | 52% | 68% | 79% |
| QBC | 54% | 71% | 81% |
| Information Density | 55% | 72% | 82% |

### Medical Imaging (Chest X-ray)

| Method | 100 images | 250 images | 500 images |
|--------|-----------|-----------|-----------|
| Random | 0.72 AUC | 0.81 AUC | 0.87 AUC |
| Entropy | 0.76 AUC | 0.84 AUC | 0.89 AUC |
| BALD | 0.78 AUC | 0.86 AUC | 0.90 AUC |
| Cost-Aware | 0.77 AUC | 0.85 AUC | 0.89 AUC |

---

## Common Pitfalls & Solutions

### Pitfall 1: Querying Outliers
**Problem**: Uncertainty sampling queries rare, unrepresentative samples
```python
# Solution: Use Information Density
combined_score = 0.7 * uncertainty + 0.3 * density
selected = np.argsort(combined_score)[-n_queries:]
```

### Pitfall 2: Redundant Batch Selections
**Problem**: Selecting similar samples in a batch
```python
# Solution: Use diversity-aware batch selection
for sample in batch:
    # Check minimum distance to already selected
    diversity = min(distances_to_selected)
    if diversity > threshold:
        selected.append(sample)
```

### Pitfall 3: Overconfident Models
**Problem**: Model too confident, uncertainty sampling ineffective
```python
# Solution: Use Query by Committee or temperature scaling
# QBC doesn't rely on single model confidence
# Or use temperature scaling before entropy:
proba_scaled = softmax(logits / temperature)
uncertainty = entropy(proba_scaled)
```

### Pitfall 4: Overfitting to Small Labeled Set
**Problem**: Model overfits initial labeled data
```python
# Solution: Use ensemble and regularization
model = RandomForestClassifier(
    max_depth=10,  # Limit tree depth
    min_samples_leaf=5,  # Require leaf samples
    n_estimators=100  # Ensemble
)
```

### Pitfall 5: Ignoring Class Imbalance
**Problem**: AL ignores class distribution
```python
# Solution: Stratified active learning
# Ensure batch has similar class distribution
class_indices = [np.where(y_pool == c)[0] for c in classes]
batch = []
for class_idx in class_indices:
    class_batch = select_from(class_idx, batch_size // n_classes)
    batch.extend(class_batch)
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] Validation on held-out test set
- [ ] Comparison against baseline (random sampling)
- [ ] Evaluation metrics agreed upon
- [ ] Annotation budget defined
- [ ] Oracle/labeling interface ready

### During Deployment
- [ ] Monitor accuracy improvement
- [ ] Track annotation cost
- [ ] Check for data drift
- [ ] Validate sample diversity
- [ ] Monitor for bias in selections

### Post-Deployment
- [ ] A/B test against old method
- [ ] Gather annotator feedback
- [ ] Analyze hard examples
- [ ] Update model regularly
- [ ] Document lessons learned

---

## Quick Troubleshooting

```
Symptom: Accuracy plateauing
→ Try different uncertainty metric
→ Use BALD if using neural networks
→ Increase batch diversity

Symptom: Too many outliers selected
→ Add diversity term to selection
→ Use Information Density
→ Increase density weight

Symptom: Very slow query computation
→ Use Entropy (fastest)
→ Reduce committee size in QBC
→ Use approximate core-set

Symptom: Model performance worse than random
→ Check if uncertainty estimates are correct
→ Verify oracle accuracy
→ Try QBC or BALD instead

Symptom: Annotator reports redundant samples
→ Increase diversity weight
→ Use core-set approach
→ Add clustering-based filtering
```

---

## Performance Optimization Tips

1. **Vectorize uncertainty computation**
   ```python
   # Fast: vectorized
   proba = model.predict_proba(X_pool)
   entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
   
   # Slow: loop
   entropy = [compute_entropy_single(x) for x in X_pool]
   ```

2. **Cache feature representations**
   ```python
   X_features = feature_extractor(X_pool)  # Compute once
   distances = cdist(X_features, X_features)  # Use cache
   ```

3. **Approximate distances for large datasets**
   ```python
   from sklearn.random_projection import random_projection
   X_projected = random_projection.fit_transform(X_pool)
   distances = cdist(X_projected, X_projected)
   ```

4. **Batch uncertainty estimation**
   ```python
   # Faster: batch processing
   for batch in DataLoader(X_pool, batch_size=128):
       uncertainty_batch = compute_uncertainty(batch)
   
   # Slower: single sample
   for x in X_pool:
       compute_uncertainty(x)
   ```

---

## Real-World Implementation Example

```python
# Complete production-ready example

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionActiveLearner:
    def __init__(self, budget=1000, batch_size=20):
        self.budget = budget
        self.batch_size = batch_size
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])
        self.queried_samples = 0
        self.history = []
    
    def train(self, X_labeled, y_labeled):
        """Train model on labeled data."""
        self.model.fit(X_labeled, y_labeled)
        logger.info(f"Model trained on {len(X_labeled)} samples")
    
    def query(self, X_unlabeled):
        """Query batch of samples."""
        if self.queried_samples >= self.budget:
            logger.warning("Budget exhausted")
            return np.array([])
        
        # Get uncertainty
        proba = self.model.predict_proba(X_unlabeled)
        uncertainty = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        
        # Select batch
        n_query = min(self.batch_size, self.budget - self.queried_samples, len(X_unlabeled))
        indices = np.argsort(uncertainty)[-n_query:]
        
        self.queried_samples += n_query
        logger.info(f"Queried {n_query} samples (total: {self.queried_samples})")
        
        return indices
    
    def update(self, X_new, y_new, X_labeled, y_labeled):
        """Update model with new labels."""
        X_labeled = np.vstack([X_labeled, X_new])
        y_labeled = np.hstack([y_labeled, y_new])
        
        self.train(X_labeled, y_labeled)
        self.history.append({
            'n_labeled': len(X_labeled),
            'n_queried': self.queried_samples
        })
        
        return X_labeled, y_labeled
    
    def get_status(self):
        """Get current status."""
        return {
            'budget_used': self.queried_samples,
            'budget_remaining': self.budget - self.queried_samples,
            'history': self.history
        }

# Usage
learner = ProductionActiveLearner(budget=1000, batch_size=20)
X_labeled, y_labeled = initial_training_set

for iteration in range(50):
    # Train
    learner.train(X_labeled, y_labeled)
    
    # Evaluate
    score = learner.model.score(X_test, y_test)
    logger.info(f"Iteration {iteration}: Score = {score:.4f}")
    
    # Query batch
    indices = learner.query(X_unlabeled)
    if len(indices) == 0:
        break
    
    # Get labels and update
    X_new = X_unlabeled[indices]
    y_new = oracle.query(X_new)  # Get labels from oracle
    
    X_labeled, y_labeled = learner.update(X_new, y_new, X_labeled, y_labeled)
    X_unlabeled = np.delete(X_unlabeled, indices, axis=0)
    
    # Check budget
    status = learner.get_status()
    if status['budget_remaining'] == 0:
        logger.info("Budget exhausted")
        break
```

---

## Final Recommendations

**Start with**: Entropy Sampling (simple, effective)
**If too slow**: Margin Sampling (fastest)
**For deep learning**: BALD with MC Dropout
**For stability**: Query by Committee
**For diversity**: Core-Set or Information Density
**For budget constraints**: Cost-Sensitive AL

**Sweet spot**: Information Density (entropy + diversity)

---

## Resources

- **Papers**: See ACTIVE_LEARNING_RESEARCH_SOURCES.md
- **Implementation Guide**: See ACTIVE_LEARNING_IMPLEMENTATION_GUIDE.md
- **Full Theory**: See ACTIVE_LEARNING_COMPREHENSIVE_GUIDE.md
- **Code**: GitHub implementations and ModAL/LibAct libraries
- **Datasets**: CIFAR-10, 20 Newsgroups, CoNLL-2003, ChexPert

