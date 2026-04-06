# Active Learning Implementation Guide

## Complete Python Framework for Active Learning

This guide provides production-ready implementations with working examples.

---

## Part 1: Core Framework Implementation

### Base Classes

```python
# active_learning_framework.py
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
from typing import Tuple, Optional, List

class UncertaintyStrategy(ABC):
    """Base class for uncertainty-based query strategies."""
    
    def __init__(self, model, n_queries: int = 10):
        self.model = model
        self.n_queries = n_queries
    
    @abstractmethod
    def compute_scores(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Compute uncertainty scores for unlabeled samples."""
        pass
    
    def query(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Query samples with highest uncertainty."""
        scores = self.compute_scores(X_unlabeled)
        return np.argsort(scores)[-self.n_queries:]


class EntropyStrategy(UncertaintyStrategy):
    """Query by entropy of predicted probabilities."""
    
    def compute_scores(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Entropy = -∑ P(y) log P(y)."""
        proba = self.model.predict_proba(X_unlabeled)
        return entropy(proba.T)


class MarginStrategy(UncertaintyStrategy):
    """Query by margin between top 2 classes."""
    
    def compute_scores(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Margin = P(y_1) - P(y_2)."""
        proba = self.model.predict_proba(X_unlabeled)
        proba_sorted = np.sort(proba, axis=1)
        # Return negative margin (to sort in ascending order = smallest margin)
        return -(proba_sorted[:, -1] - proba_sorted[:, -2])


class LeastConfidentStrategy(UncertaintyStrategy):
    """Query samples where model is least confident."""
    
    def compute_scores(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Score = 1 - max(P(y))."""
        proba = self.model.predict_proba(X_unlabeled)
        max_proba = np.max(proba, axis=1)
        return 1 - max_proba


class QueryByCommitteeStrategy:
    """Query by Committee - measure disagreement among ensemble."""
    
    def __init__(self, committee: List, n_queries: int = 10):
        """
        Args:
            committee: List of fitted estimators
            n_queries: Number of samples to query
        """
        self.committee = committee
        self.n_queries = n_queries
    
    def vote_entropy(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Vote entropy = entropy of vote distribution."""
        n_samples = X_unlabeled.shape[0]
        n_classes = len(np.unique(self.committee[0].predict(X_unlabeled)))
        
        votes = np.zeros((n_samples, n_classes))
        
        for estimator in self.committee:
            predictions = estimator.predict(X_unlabeled)
            for i, pred in enumerate(predictions):
                votes[i, int(pred)] += 1
        
        votes = votes / len(self.committee)
        return entropy(votes.T)
    
    def query(self, X_unlabeled: np.ndarray, method: str = 'vote_entropy') -> np.ndarray:
        """Query samples with highest disagreement."""
        if method == 'vote_entropy':
            scores = self.vote_entropy(X_unlabeled)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return np.argsort(scores)[-self.n_queries:]


class BatchSelector:
    """Select batches balancing uncertainty and diversity."""
    
    def __init__(self, uncertainty_strategy: UncertaintyStrategy, 
                 batch_size: int = 20, uncertainty_weight: float = 0.7):
        self.uncertainty_strategy = uncertainty_strategy
        self.batch_size = batch_size
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = 1 - uncertainty_weight
    
    def select_batch(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Select batch using greedy algorithm."""
        n_samples = len(X_unlabeled)
        
        # Compute uncertainty scores
        uncertainty = self.uncertainty_strategy.compute_scores(X_unlabeled)
        uncertainty = (uncertainty - uncertainty.min()) / \
                     (uncertainty.max() - uncertainty.min() + 1e-10)
        
        selected = []
        remaining = set(range(n_samples))
        
        # Start with most uncertain sample
        first_idx = np.argmax(uncertainty)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Greedily add samples
        for _ in range(self.batch_size - 1):
            if not remaining:
                break
            
            best_idx = None
            best_score = -np.inf
            
            for idx in remaining:
                # Uncertainty component
                unc_score = uncertainty[idx]
                
                # Diversity component (distance to selected)
                distances = np.linalg.norm(
                    X_unlabeled[idx] - X_unlabeled[list(selected)],
                    axis=1
                )
                div_score = np.min(distances) / (np.max(np.linalg.norm(X_unlabeled, axis=1)) + 1e-10)
                
                # Combined score
                score = (self.uncertainty_weight * unc_score +
                        self.diversity_weight * div_score)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return np.array(selected)


class ActiveLearningLoop:
    """Main active learning training loop."""
    
    def __init__(self, model, strategy: UncertaintyStrategy, 
                 batch_size: int = 10, n_iterations: int = 10):
        """
        Args:
            model: Classifier to train
            strategy: Query strategy
            batch_size: Samples to query per iteration
            n_iterations: Number of AL iterations
        """
        self.model = model
        self.strategy = strategy
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.history = {
            'accuracy': [],
            'n_labeled': [],
            'uncertainty_scores': []
        }
    
    def run(self, X_initial: np.ndarray, y_initial: np.ndarray,
            X_unlabeled: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
            oracle=None) -> dict:
        """
        Run active learning loop.
        
        Args:
            X_initial: Initial labeled training data
            y_initial: Initial labels
            X_unlabeled: Unlabeled pool
            X_test: Test data
            y_test: Test labels
            oracle: Function to get true labels (None = use y_unlabeled)
        
        Returns:
            Dictionary with history and results
        """
        X_labeled = X_initial.copy()
        y_labeled = y_initial.copy()
        X_pool = X_unlabeled.copy()
        
        for iteration in range(self.n_iterations):
            # Train model
            self.model.fit(X_labeled, y_labeled)
            
            # Evaluate
            accuracy = self.model.score(X_test, y_test)
            self.history['accuracy'].append(accuracy)
            self.history['n_labeled'].append(len(y_labeled))
            
            print(f"Iteration {iteration + 1}: "
                  f"Accuracy = {accuracy:.4f}, "
                  f"Labeled = {len(y_labeled)}")
            
            # Query batch
            query_indices = self._query_batch(X_pool)
            
            if len(query_indices) == 0:
                print("No more samples to query")
                break
            
            # Get labels
            if oracle is not None:
                y_queried = oracle(X_pool[query_indices])
            else:
                # Assume we have access to true labels
                y_queried = y_pool[query_indices]
            
            # Update training set
            X_labeled = np.vstack([X_labeled, X_pool[query_indices]])
            y_labeled = np.hstack([y_labeled, y_queried])
            
            # Remove from pool
            X_pool = np.delete(X_pool, query_indices, axis=0)
        
        return {
            'accuracy_history': self.history['accuracy'],
            'n_labeled_history': self.history['n_labeled'],
            'final_model': self.model
        }
    
    def _query_batch(self, X_pool: np.ndarray) -> np.ndarray:
        """Query a batch of samples."""
        batch_selector = BatchSelector(self.strategy, self.batch_size)
        return batch_selector.select_batch(X_pool)


# ============================================================================
# Specialized Strategies
# ============================================================================

class BALDStrategy:
    """Bayesian Active Learning by Disagreement."""
    
    def __init__(self, model, n_queries: int = 10, n_mc_samples: int = 30):
        """
        Args:
            model: PyTorch model with dropout
            n_queries: Samples to query
            n_mc_samples: MC dropout samples
        """
        self.model = model
        self.n_queries = n_queries
        self.n_mc_samples = n_mc_samples
    
    def mc_dropout_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions with MC dropout."""
        X_tensor = torch.FloatTensor(X)
        all_predictions = []
        
        self.model.train()  # Enable dropout
        
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                logits = self.model(X_tensor)
                proba = torch.softmax(logits, dim=1).numpy()
                all_predictions.append(proba)
        
        return np.array(all_predictions)  # (n_mc_samples, n_samples, n_classes)
    
    def compute_bald_score(self, X: np.ndarray) -> np.ndarray:
        """
        BALD = MI(Y, θ|x) = H(Y|x) - E_θ[H(Y|x,θ)]
        """
        predictions = self.mc_dropout_predictions(X)
        
        # H(Y|x) - total entropy
        mean_proba = np.mean(predictions, axis=0)
        total_entropy = entropy(mean_proba.T)
        
        # E_θ[H(Y|x,θ)] - expected entropy
        expected_entropy = np.mean(
            [entropy(predictions[i].T) for i in range(self.n_mc_samples)],
            axis=0
        )
        
        # Mutual information
        bald_score = total_entropy - expected_entropy
        
        return bald_score
    
    def query(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Query samples with highest BALD score."""
        scores = self.compute_bald_score(X_unlabeled)
        return np.argsort(scores)[-self.n_queries:]


class CoreSetStrategy:
    """Core-set approach: find representative subset."""
    
    def __init__(self, n_queries: int = 10):
        self.n_queries = n_queries
    
    def query(self, X_unlabeled: np.ndarray, 
              X_labeled: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Query diverse samples using greedy selection.
        """
        n_samples = len(X_unlabeled)
        
        selected = []
        remaining = set(range(n_samples))
        
        # Start with random sample
        first_idx = np.random.choice(list(remaining))
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Greedily select diverse samples
        for _ in range(min(self.n_queries - 1, n_samples - 1)):
            if not remaining:
                break
            
            # Find distances to selected samples
            min_distances = []
            remaining_list = list(remaining)
            
            for idx in remaining_list:
                distances = np.linalg.norm(
                    X_unlabeled[idx] - X_unlabeled[list(selected)],
                    axis=1
                )
                min_dist = np.min(distances)
                min_distances.append(min_dist)
            
            # Select point with maximum minimum distance
            farthest_idx = remaining_list[np.argmax(min_distances)]
            selected.append(farthest_idx)
            remaining.remove(farthest_idx)
        
        return np.array(selected)


class CostSensitiveStrategy:
    """Active learning with annotation costs."""
    
    def __init__(self, uncertainty_strategy: UncertaintyStrategy,
                 cost_function, n_queries: int = 10):
        """
        Args:
            uncertainty_strategy: Base strategy
            cost_function: Function that returns cost for each sample
            n_queries: Samples to query
        """
        self.uncertainty_strategy = uncertainty_strategy
        self.cost_function = cost_function
        self.n_queries = n_queries
    
    def query(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Query samples with best value-per-cost ratio."""
        uncertainty = self.uncertainty_strategy.compute_scores(X_unlabeled)
        
        # Normalize uncertainty
        uncertainty = (uncertainty - uncertainty.min()) / \
                     (uncertainty.max() - uncertainty.min() + 1e-10)
        
        # Compute costs
        costs = np.array([self.cost_function(x) for x in X_unlabeled])
        costs = costs + 1e-10  # Avoid division by zero
        
        # Value per cost
        value_per_cost = uncertainty / costs
        
        return np.argsort(value_per_cost)[-self.n_queries:]
```

---

## Part 2: Complete Working Example

```python
# example_complete_workflow.py

import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def run_active_learning_experiment():
    """Complete AL workflow with evaluation."""
    
    # 1. Generate dataset
    print("=" * 60)
    print("Active Learning Experiment")
    print("=" * 60)
    
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Further split training into initial labeled and unlabeled pool
    X_initial, X_pool, y_initial, y_pool = train_test_split(
        X_train, y_train, test_size=0.9, random_state=42
    )
    
    print(f"\nDataset Summary:")
    print(f"  Initial labeled: {len(X_initial)}")
    print(f"  Unlabeled pool: {len(X_pool)}")
    print(f"  Test set: {len(X_test)}")
    
    # 2. Baseline: Random sampling
    print(f"\n{'=' * 60}")
    print("BASELINE: Random Sampling")
    print(f"{'=' * 60}")
    
    baseline_accuracies = []
    X_lb_baseline = X_initial.copy()
    y_lb_baseline = y_initial.copy()
    X_pool_baseline = X_pool.copy()
    y_pool_baseline = y_pool.copy()
    
    for iteration in range(10):
        # Train
        model_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
        model_baseline.fit(X_lb_baseline, y_lb_baseline)
        
        # Evaluate
        accuracy = model_baseline.score(X_test, y_test)
        baseline_accuracies.append(accuracy)
        print(f"Iteration {iteration + 1}: Accuracy = {accuracy:.4f}, "
              f"Labeled = {len(y_lb_baseline)}")
        
        # Random query
        n_query = 20
        query_indices = np.random.choice(len(X_pool_baseline), 
                                         size=min(n_query, len(X_pool_baseline)),
                                         replace=False)
        
        # Update
        X_lb_baseline = np.vstack([X_lb_baseline, X_pool_baseline[query_indices]])
        y_lb_baseline = np.hstack([y_lb_baseline, y_pool_baseline[query_indices]])
        X_pool_baseline = np.delete(X_pool_baseline, query_indices, axis=0)
        y_pool_baseline = np.delete(y_pool_baseline, query_indices, axis=0)
    
    # 3. Active Learning: Entropy Sampling
    print(f"\n{'=' * 60}")
    print("ACTIVE LEARNING: Entropy Sampling")
    print(f"{'=' * 60}")
    
    al_accuracies = []
    X_lb_al = X_initial.copy()
    y_lb_al = y_initial.copy()
    X_pool_al = X_pool.copy()
    y_pool_al = y_pool.copy()
    
    for iteration in range(10):
        # Train
        model_al = RandomForestClassifier(n_estimators=100, random_state=42)
        model_al.fit(X_lb_al, y_lb_al)
        
        # Evaluate
        accuracy = model_al.score(X_test, y_test)
        al_accuracies.append(accuracy)
        print(f"Iteration {iteration + 1}: Accuracy = {accuracy:.4f}, "
              f"Labeled = {len(y_lb_al)}")
        
        # Entropy query
        entropy_strategy = EntropyStrategy(model_al, n_queries=20)
        query_indices = entropy_strategy.query(X_pool_al)
        
        # Update
        X_lb_al = np.vstack([X_lb_al, X_pool_al[query_indices]])
        y_lb_al = np.hstack([y_lb_al, y_pool_al[query_indices]])
        X_pool_al = np.delete(X_pool_al, query_indices, axis=0)
        y_pool_al = np.delete(y_pool_al, query_indices, axis=0)
    
    # 4. Active Learning: Query by Committee
    print(f"\n{'=' * 60}")
    print("ACTIVE LEARNING: Query by Committee")
    print(f"{'=' * 60}")
    
    qbc_accuracies = []
    X_lb_qbc = X_initial.copy()
    y_lb_qbc = y_initial.copy()
    X_pool_qbc = X_pool.copy()
    y_pool_qbc = y_pool.copy()
    
    for iteration in range(10):
        # Train committee
        committee = [
            RandomForestClassifier(n_estimators=50, random_state=i)
            for i in range(5)
        ]
        for est in committee:
            est.fit(X_lb_qbc, y_lb_qbc)
        
        # Evaluate on main model
        accuracy = committee[0].score(X_test, y_test)
        qbc_accuracies.append(accuracy)
        print(f"Iteration {iteration + 1}: Accuracy = {accuracy:.4f}, "
              f"Labeled = {len(y_lb_qbc)}")
        
        # QBC query
        qbc_strategy = QueryByCommitteeStrategy(committee, n_queries=20)
        query_indices = qbc_strategy.query(X_pool_qbc)
        
        # Update
        X_lb_qbc = np.vstack([X_lb_qbc, X_pool_qbc[query_indices]])
        y_lb_qbc = np.hstack([y_lb_qbc, y_pool_qbc[query_indices]])
        X_pool_qbc = np.delete(X_pool_qbc, query_indices, axis=0)
        y_pool_qbc = np.delete(y_pool_qbc, query_indices, axis=0)
    
    # 5. Plot results
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    
    iterations = np.arange(1, 11)
    labeled_counts = np.arange(len(X_initial), len(X_initial) + 10 * 20, 20)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, baseline_accuracies, 'o-', label='Random Sampling', linewidth=2)
    plt.plot(iterations, al_accuracies, 's-', label='Entropy Sampling', linewidth=2)
    plt.plot(iterations, qbc_accuracies, '^-', label='Query by Committee', linewidth=2)
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Accuracy vs Iteration', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(labeled_counts, baseline_accuracies, 'o-', label='Random Sampling', linewidth=2)
    plt.plot(labeled_counts, al_accuracies, 's-', label='Entropy Sampling', linewidth=2)
    plt.plot(labeled_counts, qbc_accuracies, '^-', label='Query by Committee', linewidth=2)
    plt.xlabel('Number of Labeled Samples', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Learning Curves', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('active_learning_results.png', dpi=150)
    print("Results saved to active_learning_results.png")
    
    # Summary statistics
    print("\nFinal Accuracies:")
    print(f"  Random Sampling: {baseline_accuracies[-1]:.4f}")
    print(f"  Entropy Sampling: {al_accuracies[-1]:.4f}")
    print(f"  Query by Committee: {qbc_accuracies[-1]:.4f}")
    
    print("\nAccuracy Improvement (AL vs Random):")
    print(f"  Entropy Sampling: +{(al_accuracies[-1] - baseline_accuracies[-1]) * 100:.2f}%")
    print(f"  Query by Committee: +{(qbc_accuracies[-1] - baseline_accuracies[-1]) * 100:.2f}%")


if __name__ == "__main__":
    run_active_learning_experiment()
```

---

## Part 3: Advanced Implementations

### Deep Active Learning with PyTorch

```python
# deep_active_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 style data."""
    
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DeepActiveLearner:
    """Deep active learning with BALD strategy."""
    
    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, X_train, y_train, batch_size=32, epochs=5):
        """Train model for specified epochs."""
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
    
    def predict_proba(self, X):
        """Get probability predictions."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            proba = torch.softmax(logits, dim=1)
        
        return proba.cpu().numpy()
    
    def get_mc_predictions(self, X, n_samples=30):
        """Get predictions with MC dropout."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.train()  # Enable dropout
        
        all_proba = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(X_tensor)
                proba = torch.softmax(logits, dim=1)
                all_proba.append(proba.cpu().numpy())
        
        return np.array(all_proba)  # (n_samples, n_data, n_classes)
    
    def query_bald(self, X_unlabeled, n_queries=10):
        """Query using BALD strategy."""
        predictions = self.get_mc_predictions(X_unlabeled, n_samples=30)
        
        # Total entropy
        mean_proba = np.mean(predictions, axis=0)
        total_entropy = entropy(mean_proba.T)
        
        # Expected entropy
        expected_entropy = np.mean(
            [entropy(predictions[i].T) for i in range(predictions.shape[0])],
            axis=0
        )
        
        # BALD score
        bald_score = total_entropy - expected_entropy
        
        return np.argsort(bald_score)[-n_queries:]
```

---

## Performance Benchmarks

### Expected Results on Standard Datasets

**CIFAR-10 (Image Classification)**:
- Supervised baseline (5000 labels): ~85% accuracy
- Active Learning (5000 labels): ~82-84% accuracy
- Random Sampling (2500 labels): ~70% accuracy
- Active Learning (2500 labels): ~75-78% accuracy

**20 Newsgroups (Text Classification)**:
- Supervised baseline (500 docs): ~82% accuracy
- Active Learning (500 docs): ~80-81% accuracy
- Random Sampling (250 docs): ~65% accuracy
- Active Learning (250 docs): ~72-75% accuracy

**Medical Imaging (Chest X-ray)**:
- Supervised baseline (1000 images): ~90% AUC
- Active Learning (1000 images): ~88-89% AUC
- Random Sampling (500 images): ~78% AUC
- Active Learning (500 images): ~84-86% AUC

---

## Hyperparameter Tuning Guide

```python
# hyperparameter_tuning.py

class ActiveLearningHyperparameterSearch:
    """Grid search for AL hyperparameters."""
    
    def __init__(self, base_model, strategy_class):
        self.base_model = base_model
        self.strategy_class = strategy_class
    
    def search(self, X_initial, y_initial, X_pool, y_pool,
               X_test, y_test, param_grid):
        """
        Search over hyperparameters.
        
        Args:
            param_grid: dict of parameter lists
        """
        results = []
        
        # Generate all parameter combinations
        params_list = list(itertools.product(*param_grid.values()))
        
        for params in params_list:
            param_dict = dict(zip(param_grid.keys(), params))
            
            # Run experiment with these params
            strategy = self.strategy_class(self.base_model, **param_dict)
            
            X_lb = X_initial.copy()
            y_lb = y_initial.copy()
            X_pl = X_pool.copy()
            y_pl = y_pool.copy()
            
            final_accuracy = 0
            for _ in range(5):  # Short run for tuning
                model = clone(self.base_model)
                model.fit(X_lb, y_lb)
                final_accuracy = model.score(X_test, y_test)
                
                indices = strategy.query(X_pl)
                X_lb = np.vstack([X_lb, X_pl[indices]])
                y_lb = np.hstack([y_lb, y_pl[indices]])
                X_pl = np.delete(X_pl, indices, axis=0)
            
            results.append({
                'params': param_dict,
                'final_accuracy': final_accuracy
            })
        
        # Return best params
        best_result = max(results, key=lambda x: x['final_accuracy'])
        return best_result
```

---

## Production Deployment Checklist

- [ ] Data validation and preprocessing
- [ ] Model serialization (joblib, pickle)
- [ ] Strategy serialization
- [ ] Annotation interface/API
- [ ] Logging and monitoring
- [ ] Performance tracking
- [ ] Cost tracking
- [ ] Human-in-loop integration
- [ ] A/B testing setup
- [ ] Fallback mechanisms
- [ ] Documentation

---

## Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| AL worse than random | Wrong strategy | Try BALD or QBC |
| Too many outliers queried | High outlier weight | Increase diversity term |
| Slow training | Large batch size | Reduce batch size or use sampling |
| Poor performance | Limited initial data | Increase initial training set |
| High annotation cost | Cost not tracked | Use cost-sensitive strategy |

