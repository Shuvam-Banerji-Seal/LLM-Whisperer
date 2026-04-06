# Class Imbalance Handling Skill

**Author:** Shuvam Banerji Seal  
**Version:** 1.0  
**Last Updated:** April 2026

## Overview

Class imbalance is a prevalent problem in machine learning where one class significantly outnumbers others. This creates biased models that fail to learn minority class patterns. This skill covers detection, resampling, and advanced techniques for balanced learning.

### Definition

**Class Imbalance Ratio (IR):**
$$IR = \frac{\text{Number of majority class samples}}{\text{Number of minority class samples}}$$

**Severity Classification:**
- Mild: IR < 2:1
- Moderate: 2:1 ≤ IR < 10:1
- Severe: 10:1 ≤ IR < 100:1
- Extreme: IR ≥ 100:1

## Mathematical Formulations

### 1. Imbalance Metrics

#### Class Distribution
$$P(y=i) = \frac{n_i}{N}$$

Where $n_i$ = samples in class i, $N$ = total samples

#### Imbalance Ratio
$$IR = \max_i(n_i) / \min_i(n_i)$$

### 2. Performance Metrics

#### Balanced Accuracy
$$BA = \frac{1}{K}\sum_{i=1}^{K} \frac{TP_i}{TP_i + FN_i}$$

Where K = number of classes

#### F1-Score (Weighted)
$$F1_{\text{weighted}} = \sum_i w_i \cdot F1_i$$

Where $w_i = n_i / N$

#### G-Mean (Geometric Mean)
$$G\text{-}Mean = \sqrt{\prod_{i=1}^{K} Recall_i}$$

### 3. Oversampling (SMOTE)

**Synthetic Minority Oversampling Technique:**

For each minority sample $x_i$:
$$\hat{x} = x_i + \lambda(x_{k} - x_i), \quad \lambda \sim U(0, 1)$$

Where $x_k$ is a randomly selected k-nearest neighbor

### 4. Undersampling

**Random Undersampling:**
Randomly remove majority class samples

**Stratified Undersampling:**
Maintain class distribution in subsets

### 5. Cost-Sensitive Learning

**Weighted Loss Function:**
$$L = -\sum_i w_i \cdot [\hat{y}_i \log(p_i) + (1-\hat{y}_i) \log(1-p_i)]$$

Where $w_i$ = weight for class i

**Class Weight Calculation:**
$$w_i = \frac{N}{K \cdot n_i}$$

## Implementation

### Python Code Examples

#### 1. Imbalance Detection and Analysis

```python
import pandas as pd
import numpy as np
from collections import Counter

class ImbalanceAnalyzer:
    """Analyze and quantify class imbalance"""
    
    def __init__(self, y: np.ndarray):
        self.y = y
        self.class_counts = Counter(y)
        self.n_samples = len(y)
    
    def get_imbalance_ratio(self) -> float:
        """Calculate imbalance ratio"""
        counts = np.array(list(self.class_counts.values()))
        return counts.max() / counts.min()
    
    def get_class_weights(self, method: str = 'balanced') -> Dict[int, float]:
        """
        Calculate class weights
        
        Args:
            method: 'balanced' or 'effective'
        """
        if method == 'balanced':
            # w_i = N / (K * n_i)
            n_classes = len(self.class_counts)
            weights = {
                cls: self.n_samples / (n_classes * count)
                for cls, count in self.class_counts.items()
            }
        elif method == 'effective':
            # Effective number: w_i = (1 - β) / (1 - β^n_i)
            beta = 0.9999
            weights = {
                cls: (1 - beta) / (1 - beta**count)
                for cls, count in self.class_counts.items()
            }
        
        return weights
    
    def analyze(self) -> Dict:
        """Comprehensive imbalance analysis"""
        counts = np.array(list(self.class_counts.values()))
        
        analysis = {
            'total_samples': self.n_samples,
            'n_classes': len(self.class_counts),
            'class_distribution': dict(self.class_counts),
            'class_percentages': {
                cls: (count / self.n_samples) * 100
                for cls, count in self.class_counts.items()
            },
            'imbalance_ratio': self.get_imbalance_ratio(),
            'severity': self._classify_severity(),
            'class_weights': self.get_class_weights()
        }
        
        return analysis
    
    def _classify_severity(self) -> str:
        """Classify imbalance severity"""
        ir = self.get_imbalance_ratio()
        if ir < 2:
            return 'Mild'
        elif ir < 10:
            return 'Moderate'
        elif ir < 100:
            return 'Severe'
        else:
            return 'Extreme'

# Usage
y = np.array([0]*900 + [1]*100)  # 90:10 imbalance
analyzer = ImbalanceAnalyzer(y)
print(analyzer.analyze())
```

#### 2. Resampling Techniques

```python
from sklearn.utils import resample

class ResamplingPipeline:
    """Resampling strategies for class imbalance"""
    
    @staticmethod
    def random_oversampling(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random Oversampling - duplicate minority samples
        """
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Separate classes
        majority = df[df['target'] == 0]
        minority = df[df['target'] == 1]
        
        # Oversample minority to match majority
        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42
        )
        
        df_balanced = pd.concat([majority, minority_upsampled])
        
        X_balanced = df_balanced.drop('target', axis=1).values
        y_balanced = df_balanced['target'].values
        
        return X_balanced, y_balanced
    
    @staticmethod
    def random_undersampling(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random Undersampling - remove majority samples
        """
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Separate classes
        majority = df[df['target'] == 0]
        minority = df[df['target'] == 1]
        
        # Undersample majority to match minority
        majority_downsampled = resample(
            majority,
            replace=False,
            n_samples=len(minority),
            random_state=42
        )
        
        df_balanced = pd.concat([majority_downsampled, minority])
        
        X_balanced = df_balanced.drop('target', axis=1).values
        y_balanced = df_balanced['target'].values
        
        return X_balanced, y_balanced
    
    @staticmethod
    def smote_oversampling(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE (Synthetic Minority Over-sampling Technique)
        Generates synthetic samples along k-NN paths
        """
        from sklearn.neighbors import NearestNeighbors
        
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Separate classes
        majority = df[df['target'] == 0]
        minority = df[df['target'] == 1]
        
        X_minority = minority.drop('target', axis=1).values
        
        # Find k-nearest neighbors for each minority sample
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X_minority)
        distances, indices = nbrs.kneighbors(X_minority)
        
        # Generate synthetic samples
        n_synthetic = len(majority) - len(minority)
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Randomly select minority sample
            sample_idx = np.random.randint(0, len(X_minority))
            
            # Randomly select one of its neighbors
            neighbor_idx = indices[sample_idx][np.random.randint(1, k_neighbors)]
            neighbor = X_minority[neighbor_idx]
            
            # Generate synthetic sample along interpolation
            sample = X_minority[sample_idx]
            alpha = np.random.rand()
            synthetic = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        # Combine
        X_synthetic = np.array(synthetic_samples)
        synthetic_df = pd.DataFrame(X_synthetic)
        synthetic_df['target'] = 1
        
        df_balanced = pd.concat([
            majority,
            minority,
            synthetic_df
        ], ignore_index=True)
        
        X_balanced = df_balanced.drop('target', axis=1).values
        y_balanced = df_balanced['target'].values
        
        return X_balanced, y_balanced
    
    @staticmethod
    def smote_tomek(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE + Tomek Links
        Combines oversampling with removal of noisy samples
        """
        try:
            from imblearn.combine import SMOTETomek
            
            smt = SMOTETomek(random_state=42)
            X_balanced, y_balanced = smt.fit_resample(X, y)
            
            return X_balanced, y_balanced
        except ImportError:
            print("Install imbalanced-learn: pip install imbalanced-learn")
            return X, y

# Usage
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    weights=[0.9, 0.1],
    random_state=42
)

print("Original distribution:", Counter(y))

X_balanced, y_balanced = ResamplingPipeline.smote_oversampling(X, y)
print("After SMOTE:", Counter(y_balanced))
```

#### 3. Cost-Sensitive Learning

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class CostSensitiveLearning:
    """Cost-sensitive learning for imbalanced data"""
    
    @staticmethod
    def balanced_logistic_regression(X: np.ndarray, y: np.ndarray, **kwargs):
        """Logistic regression with class weight balancing"""
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            **kwargs
        )
        model.fit(X, y)
        return model
    
    @staticmethod
    def weighted_random_forest(X: np.ndarray, y: np.ndarray, **kwargs):
        """Random Forest with automatic class weighting"""
        model = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            random_state=42,
            **kwargs
        )
        model.fit(X, y)
        return model
    
    @staticmethod
    def custom_weighted_loss(y_true, y_pred_proba, class_weights: Dict):
        """Calculate weighted binary cross-entropy"""
        from sklearn.metrics import log_loss
        
        # Apply class weights to samples
        sample_weights = np.array([class_weights[y] for y in y_true])
        
        return log_loss(y_true, y_pred_proba, sample_weight=sample_weights)
    
    @staticmethod
    def threshold_optimization(y_true, y_pred_proba, metric='f1'):
        """
        Optimize prediction threshold for imbalanced data
        Default threshold=0.5 may not be optimal
        """
        from sklearn.metrics import precision_recall_curve, f1_score
        
        if metric == 'f1':
            # Find threshold maximizing F1-score
            precision, recall, thresholds = precision_recall_curve(
                y_true, y_pred_proba
            )
            
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
        elif metric == 'gmean':
            # Find threshold maximizing G-Mean
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            gmeans = np.sqrt(tpr * (1 - fpr))
            best_idx = np.argmax(gmeans)
            optimal_threshold = thresholds[best_idx]
        
        return optimal_threshold

# Usage
analyzer = ImbalanceAnalyzer(y)
weights = analyzer.get_class_weights()
print(f"Class weights: {weights}")

# Train with balanced class weights
model = CostSensitiveLearning.balanced_logistic_regression(X, y)
y_pred_proba = model.predict_proba(X)[:, 1]

# Optimize threshold
optimal_threshold = CostSensitiveLearning.threshold_optimization(y, y_pred_proba)
print(f"Optimal threshold: {optimal_threshold:.3f}")
```

#### 4. Evaluation Framework

```python
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, f1_score
)

class ImbalancedEvaluator:
    """Evaluation metrics for imbalanced classification"""
    
    @staticmethod
    def balanced_metrics(y_true, y_pred):
        """Calculate metrics suitable for imbalanced data"""
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Balanced Accuracy
        ba = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
        
        # Sensitivity (Recall for minority class)
        sensitivity = tp / (tp + fn)
        
        # Specificity (Recall for majority class)
        specificity = tn / (tn + fp)
        
        # G-Mean
        gmean = np.sqrt(sensitivity * specificity)
        
        metrics = {
            'balanced_accuracy': ba,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'g_mean': gmean,
            'f1_score': f1_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        return metrics
    
    @staticmethod
    def probability_metrics(y_true, y_pred_proba):
        """Metrics based on probability predictions"""
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(precision[::-1], recall[::-1])
        
        metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'pr_curve': (precision, recall)
        }
        
        return metrics
    
    @staticmethod
    def generate_evaluation_report(y_true, y_pred, y_pred_proba) -> Dict:
        """Comprehensive evaluation report"""
        
        balanced = ImbalancedEvaluator.balanced_metrics(y_true, y_pred)
        probs = ImbalancedEvaluator.probability_metrics(y_true, y_pred_proba)
        
        report = {
            **balanced,
            **probs,
            'total_samples': len(y_true),
            'positive_samples': (y_true == 1).sum(),
            'negative_samples': (y_true == 0).sum()
        }
        
        return report

# Usage
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

report = ImbalancedEvaluator.generate_evaluation_report(y, y_pred, y_pred_proba)
print("Evaluation Report:")
for metric, value in report.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.4f}")
```

#### 5. Complete Pipeline

```python
class ImbalanceHandlingPipeline:
    """End-to-end pipeline for imbalanced classification"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3):
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    
    def run_full_pipeline(self):
        """Execute complete imbalance handling pipeline"""
        
        # Step 1: Analyze imbalance
        print("1. Analyzing imbalance...")
        analyzer = ImbalanceAnalyzer(self.y_train)
        print(analyzer.analyze())
        
        # Step 2: Resample training data
        print("\n2. Resampling training data (SMOTE)...")
        X_train_balanced, y_train_balanced = ResamplingPipeline.smote_oversampling(
            self.X_train, self.y_train
        )
        
        # Step 3: Train model with cost-sensitive learning
        print("\n3. Training cost-sensitive model...")
        model = CostSensitiveLearning.balanced_logistic_regression(
            X_train_balanced, y_train_balanced
        )
        
        # Step 4: Optimize threshold
        print("\n4. Optimizing prediction threshold...")
        y_train_proba = model.predict_proba(self.X_train)[:, 1]
        optimal_threshold = CostSensitiveLearning.threshold_optimization(
            self.y_train, y_train_proba
        )
        
        # Step 5: Evaluate on test set
        print("\n5. Evaluating on test set...")
        y_test_proba = model.predict_proba(self.X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        report = ImbalancedEvaluator.generate_evaluation_report(
            self.y_test, y_test_pred, y_test_proba
        )
        
        print("\nEvaluation Report:")
        for metric, value in report.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        return model, optimal_threshold, report

# Usage
pipeline = ImbalanceHandlingPipeline(X, y)
model, threshold, report = pipeline.run_full_pipeline()
```

## Authoritative Sources

1. **Chawla et al. (2002)** - "SMOTE: Synthetic Minority Over-sampling Technique"
   - Original SMOTE paper
   - Journal of Artificial Intelligence Research

2. **He & Garcia (2009)** - "Learning from Imbalanced Data"
   - Comprehensive survey of imbalance handling techniques
   - IEEE Transactions on Knowledge and Data Engineering

3. **Scikit-learn: Imbalanced-learn** - https://imbalanced-learn.org/
   - Specialized library for handling imbalanced datasets
   - Multiple resampling strategies

4. **Batista et al. (2004)** - "A Study of the Behavior of Several Methods"
   - Comparison of oversampling, undersampling, and hybrid methods
   - ACM SIGKDD

5. **Elkan (2001)** - "The Foundations of Cost-Sensitive Learning"
   - Theoretical foundations of cost-sensitive methods
   - IJCAI 2001

## Practical Checklist

- [ ] Quantify class imbalance severity
- [ ] Choose appropriate resampling strategy
- [ ] Validate resampling on training set only
- [ ] Use stratified cross-validation
- [ ] Select evaluation metrics appropriate for imbalance
- [ ] Avoid evaluation metric trap (accuracy is misleading)
- [ ] Tune decision threshold for deployment
- [ ] Monitor class distribution in production
- [ ] Document resampling parameters for reproducibility

## Edge Cases & Considerations

### Case 1: Extreme Imbalance (IR > 100:1)
**Challenge:** Standard resampling may generate too many synthetic samples  
**Solution:** Hierarchical approach - first balance to 1:10, then SMOTE

### Case 2: Multi-class Imbalance
**Challenge:** Multiple minority classes  
**Solution:** One-vs-Rest approach or multi-class SMOTE variants

### Case 3: Temporal Data
**Challenge:** Time order matters, can't freely resample  
**Solution:** Time-aware resampling or separate temporal buckets

### Case 4: High-Dimensional Data
**Challenge:** Curse of dimensionality affects k-NN based methods  
**Solution:** Feature selection before SMOTE or ensemble methods

## Strategy Comparison

| Strategy | Data Loss | Bias Risk | Complexity | Computational Cost |
|----------|-----------|-----------|-----------|-------------------|
| **Random Oversample** | No | High | Low | Low |
| **Random Undersample** | Yes | Low | Low | Low |
| **SMOTE** | No | Medium | Medium | Medium |
| **Cost-Sensitive** | No | Low | Low | Low |
| **Ensemble** | No | Very Low | High | High |

## Conclusion

Handling class imbalance requires a multi-faceted approach combining data-level (resampling) and algorithm-level (cost-sensitive) techniques. The optimal strategy depends on data characteristics, computational constraints, and business requirements.

---

**Last Reviewed:** April 2026  
**Skill Status:** Production Ready
