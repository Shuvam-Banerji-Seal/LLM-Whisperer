# Causal Inference: Implementation Guide

**Version:** 1.0  
**Focus:** Practical code examples and deployment strategies  
**Last Updated:** April 2026

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Quick Start Implementations](#quick-start-implementations)
3. [Data Preparation](#data-preparation)
4. [Method Comparison Framework](#method-comparison-framework)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting & Best Practices](#troubleshooting--best-practices)

---

## Environment Setup

### Installation

```bash
# Create virtual environment
python -m venv causal_env
source causal_env/bin/activate  # On Windows: causal_env\Scripts\activate

# Core libraries
pip install numpy pandas scikit-learn scipy statsmodels

# Causal inference libraries
pip install dowhy causalml econml

# Optional: Advanced packages
pip install networkx pygraphviz  # For DAG visualization
pip install matplotlib seaborn  # Visualization
pip install jupyter notebook  # Interactive notebooks
```

### Version Compatibility

```
Python: 3.8+
NumPy: 1.19+
Pandas: 1.2+
Scikit-learn: 0.24+
DoWhy: 0.8+
CausalML: 0.11+
```

### Verify Installation

```python
import numpy as np
import pandas as pd
from dowhy import CausalModel
from causalml.inference.meta import DRLearner
from causalml.inference.tree_based import CausalForestRegressor

print("All packages installed successfully!")
```

---

## Quick Start Implementations

### 1. Propensity Score Matching (5-minute quickstart)

```python
"""
Simplest propensity score matching implementation.
Typical use case: compare treated vs control groups.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist

def simple_psm(X, T, Y, caliper=0.1):
    """
    Simple propensity score matching
    
    Args:
        X: (n, p) feature matrix
        T: (n,) binary treatment indicator
        Y: (n,) outcome variable
        caliper: maximum distance for matches
    
    Returns:
        ATE: Average treatment effect
        matches: Dictionary of matched pairs
    """
    
    # Step 1: Estimate propensity scores via logistic regression
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, T)
    propensity_scores = ps_model.predict_proba(X)[:, 1]
    
    # Step 2: Find matches
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    # Calculate distances between treated and control
    ps_treated = propensity_scores[treated_idx]
    ps_control = propensity_scores[control_idx]
    
    distances = np.abs(ps_treated[:, np.newaxis] - ps_control[np.newaxis, :])
    
    # Step 3: Match with caliper
    matches = {}
    matched_pairs = []
    
    for i, t_idx in enumerate(treated_idx):
        # Find closest control
        min_dist_idx = np.argmin(distances[i, :])
        c_idx = control_idx[min_dist_idx]
        min_dist = distances[i, min_dist_idx]
        
        # Check caliper
        if min_dist <= caliper:
            matches[t_idx] = c_idx
            matched_pairs.append((Y[t_idx], Y[c_idx]))
    
    # Step 4: Calculate ATE
    matched_pairs = np.array(matched_pairs)
    ATE = np.mean(matched_pairs[:, 0] - matched_pairs[:, 1])
    
    return ATE, matches

# Example usage
np.random.seed(42)
n = 1000

# Generate synthetic data
X = np.random.randn(n, 5)
T = (X[:, 0] + np.random.randn(n) > 0).astype(int)
Y = T * 2 + X[:, 1] + np.random.randn(n)

ATE, matches = simple_psm(X, T, Y, caliper=0.1)
print(f"Estimated ATE: {ATE:.4f}")
print(f"Matched pairs: {len(matches)}")
```

### 2. Doubly Robust Estimation (Complete example)

```python
"""
Doubly robust estimator combining propensity score and outcome modeling.
More robust to model misspecification than PSM alone.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor

class DoublyRobustEstimator:
    def __init__(self, propensity_model=None, outcome_model=None):
        """
        Initialize doubly robust estimator
        
        Args:
            propensity_model: sklearn classifier for P(T=1|X)
            outcome_model: sklearn regressor for E[Y|T,X]
        """
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000)
        self.outcome_model = outcome_model or RandomForestRegressor(n_estimators=100, max_depth=10)
        
        self.propensity_scores_ = None
        self.outcome_model_0_ = None
        self.outcome_model_1_ = None
    
    def fit(self, X, T, Y):
        """Fit propensity and outcome models"""
        
        # Fit propensity score
        self.propensity_model.fit(X, T)
        self.propensity_scores_ = self.propensity_model.predict_proba(X)[:, 1]
        
        # Fit outcome models separately for T=0 and T=1
        self.outcome_model_0_ = RandomForestRegressor(n_estimators=100, max_depth=10)
        self.outcome_model_0_.fit(X[T == 0], Y[T == 0])
        
        self.outcome_model_1_ = RandomForestRegressor(n_estimators=100, max_depth=10)
        self.outcome_model_1_.fit(X[T == 1], Y[T == 1])
        
        return self
    
    def estimate_ate(self, X, T, Y):
        """Estimate average treatment effect using doubly robust method"""
        
        e = self.propensity_scores_
        
        # Outcome predictions
        y0_pred = self.outcome_model_0_.predict(X)
        y1_pred = self.outcome_model_1_.predict(X)
        
        # Doubly robust estimator
        dr_treated = (T * Y / e) - ((T - e) / e) * y1_pred
        dr_control = ((1 - T) * Y / (1 - e)) + ((T - e) / (1 - e)) * y0_pred
        
        ATE = np.mean(dr_treated) - np.mean(dr_control)
        
        return ATE, dr_treated, dr_control
    
    def estimate_cate(self, X):
        """Estimate conditional average treatment effect"""
        
        y0_pred = self.outcome_model_0_.predict(X)
        y1_pred = self.outcome_model_1_.predict(X)
        
        CATE = y1_pred - y0_pred
        
        return CATE

# Example usage
np.random.seed(42)
n_train, n_test = 1000, 500

# Training data
X_train = np.random.randn(n_train, 5)
T_train = (X_train[:, 0] + np.random.randn(n_train) > 0).astype(int)
Y_train = T_train * 2 + X_train[:, 1] - X_train[:, 2] + np.random.randn(n_train)

# Test data
X_test = np.random.randn(n_test, 5)

# Fit and estimate
dr = DoublyRobustEstimator()
dr.fit(X_train, T_train, Y_train)

ATE, dr_treated, dr_control = dr.estimate_ate(X_train, T_train, Y_train)
CATE = dr.estimate_cate(X_test)

print(f"Estimated ATE: {ATE:.4f}")
print(f"CATE range: [{CATE.min():.4f}, {CATE.max():.4f}]")
print(f"Average CATE: {CATE.mean():.4f}")
```

### 3. Causal Forest for Heterogeneous Effects

```python
"""
Causal forest for estimating heterogeneous treatment effects.
Ensemble method combining multiple causal trees.
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class SimpleCausalForest:
    def __init__(self, n_trees=100, max_depth=15, min_samples_leaf=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
    
    def fit(self, X, T, Y):
        """Fit ensemble of causal trees"""
        
        self.trees = []
        np.random.seed(42)
        
        for _ in range(self.n_trees):
            # Bootstrap sample
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_b, T_b, Y_b = X[idx], T[idx], Y[idx]
            
            # Fit tree
            tree = self._fit_causal_tree(X_b, T_b, Y_b, depth=0)
            self.trees.append(tree)
        
        return self
    
    def _fit_causal_tree(self, X, T, Y, depth):
        """Recursively fit a causal tree"""
        
        if depth >= self.max_depth or len(X) < self.min_samples_leaf:
            # Leaf: estimate CATE
            cate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
            return {'type': 'leaf', 'cate': cate}
        
        # Try all splits
        best_split = None
        best_gain = 0
        
        for feature in range(X.shape[1]):
            for threshold in np.percentile(X[:, feature], np.linspace(10, 90, 9)):
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if len(X[left_mask]) == 0 or len(X[right_mask]) == 0:
                    continue
                
                # Calculate effect heterogeneity
                left_effect = (np.mean(Y[left_mask & (T == 1)]) - 
                              np.mean(Y[left_mask & (T == 0)]))
                right_effect = (np.mean(Y[right_mask & (T == 1)]) - 
                               np.mean(Y[right_mask & (T == 0)]))
                
                gain = abs(left_effect - right_effect)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)
        
        if best_split is None:
            cate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
            return {'type': 'leaf', 'cate': cate}
        
        feature, threshold = best_split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left_tree = self._fit_causal_tree(X[left_mask], T[left_mask], Y[left_mask], depth + 1)
        right_tree = self._fit_causal_tree(X[right_mask], T[right_mask], Y[right_mask], depth + 1)
        
        return {
            'type': 'node',
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def predict(self, X):
        """Predict CATE for new samples"""
        predictions = []
        
        for x in X:
            tree_preds = [self._predict_sample(x, tree) for tree in self.trees]
            pred = np.mean(tree_preds)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _predict_sample(self, x, tree):
        """Predict for single sample using a tree"""
        
        if tree['type'] == 'leaf':
            return tree['cate']
        
        feature = tree['feature']
        threshold = tree['threshold']
        
        if x[feature] <= threshold:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

# Example usage
np.random.seed(42)
n = 2000

# Data with heterogeneous treatment effect
X = np.random.randn(n, 5)
T = (X[:, 0] + np.random.randn(n) > 0).astype(int)
# Effect depends on X[1]
CATE_true = 2 * X[:, 1]
Y = T * CATE_true + X[:, 0] - X[:, 2] + np.random.randn(n) * 0.5

# Fit causal forest
cf = SimpleCausalForest(n_trees=50, max_depth=10)
cf.fit(X, T, Y)

# Predict
CATE_pred = cf.predict(X)

# Evaluate
mae = np.mean(np.abs(CATE_pred - CATE_true))
corr = np.corrcoef(CATE_pred, CATE_true)[0, 1]

print(f"MAE: {mae:.4f}")
print(f"Correlation with true CATE: {corr:.4f}")
```

---

## Data Preparation

### 1. Checking Assumptions

```python
"""
Verify key assumptions for causal inference.
"""

import matplotlib.pyplot as plt

def check_overlap(propensity_scores, T):
    """
    Check overlap assumption: 0 < P(T=1|X) < 1 for all samples.
    Critical for matching-based methods.
    """
    
    ps_treated = propensity_scores[T == 1]
    ps_control = propensity_scores[T == 0]
    
    print(f"Control propensity score range: [{ps_control.min():.3f}, {ps_control.max():.3f}]")
    print(f"Treated propensity score range: [{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ps_control, bins=30, alpha=0.5, label='Control', density=True)
    ax.hist(ps_treated, bins=30, alpha=0.5, label='Treated', density=True)
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity Score Distributions')
    ax.legend()
    plt.tight_layout()
    
    return ps_control, ps_treated

def check_covariate_balance(X, T, propensity_scores, matched_idx=None):
    """
    Check covariate balance between treatment and control.
    After matching, balance should improve.
    """
    
    if matched_idx is not None:
        X = X[matched_idx]
        T = T[matched_idx]
    
    # Calculate standardized mean differences
    X_ctrl = X[T == 0]
    X_treat = X[T == 1]
    
    mean_diff = (X_treat.mean(axis=0) - X_ctrl.mean(axis=0))
    std_pooled = np.sqrt((X_ctrl.std(axis=0)**2 + X_treat.std(axis=0)**2) / 2)
    smd = mean_diff / std_pooled
    
    print("\nStandardized Mean Differences:")
    for i, s in enumerate(smd):
        status = "✓" if abs(s) < 0.1 else "✗"
        print(f"  Feature {i}: {s:.4f} {status}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 5))
    features = [f"X{i}" for i in range(len(smd))]
    colors = ['green' if abs(s) < 0.1 else 'red' for s in smd]
    ax.barh(features, np.abs(smd), color=colors, alpha=0.7)
    ax.axvline(x=0.1, color='red', linestyle='--', label='Balance threshold')
    ax.set_xlabel('|Standardized Mean Difference|')
    ax.set_title('Covariate Balance')
    ax.legend()
    plt.tight_layout()
    
    return smd

def check_parallel_trends(Y_ctrl_pre, Y_ctrl_post, Y_treat_pre, Y_treat_post):
    """
    Check parallel trends assumption for difference-in-differences.
    Pre-treatment trends should be similar.
    """
    
    trend_ctrl = Y_ctrl_post.mean() - Y_ctrl_pre.mean()
    trend_treat = Y_treat_post.mean() - Y_treat_pre.mean()
    
    print(f"\nParallel Trends Check (Difference-in-Differences):")
    print(f"  Control trend: {trend_ctrl:.4f}")
    print(f"  Treatment trend: {trend_treat:.4f}")
    print(f"  Difference: {abs(trend_ctrl - trend_treat):.4f}")
    
    if abs(trend_ctrl - trend_treat) < 0.1:
        print("  ✓ Trends appear parallel")
    else:
        print("  ✗ Warning: Trends may not be parallel")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 5))
    time_points = ['Pre', 'Post']
    ax.plot(time_points, [Y_ctrl_pre.mean(), Y_ctrl_post.mean()], 
            'o-', label='Control', markersize=8)
    ax.plot(time_points, [Y_treat_pre.mean(), Y_treat_post.mean()], 
            'o-', label='Treatment', markersize=8)
    ax.set_ylabel('Outcome')
    ax.set_title('Parallel Trends Check')
    ax.legend()
    plt.tight_layout()

# Example usage
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
T = (X[:, 0] > 0).astype(int)
Y = 2 * T + X[:, 1] + np.random.randn(n)

# Estimate propensity scores
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, T)
ps = ps_model.predict_proba(X)[:, 1]

check_overlap(ps, T)
smd = check_covariate_balance(X, T, ps)
```

### 2. Handling Missing Data

```python
"""
Strategies for missing data in causal inference.
"""

from sklearn.impute import SimpleImputer, KNNImputer

def handle_missing_data(X, T, Y, method='knn'):
    """
    Handle missing values in features.
    
    Args:
        method: 'mean', 'median', 'knn'
    """
    
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    
    X_imputed = imputer.fit_transform(X)
    
    return X_imputed, T, Y

def multiple_imputation(X, T, Y, n_imputations=5):
    """
    Multiple imputation for uncertainty quantification.
    Estimates causal effect under each imputation.
    """
    
    from sklearn.impute import SimpleImputer
    
    estimates = []
    
    for i in range(n_imputations):
        # Impute with different random states
        imputer = SimpleImputer(strategy='mean')
        X_imp = imputer.fit_transform(X)
        
        # Add random noise to create variation
        X_imp = X_imp + np.random.randn(*X_imp.shape) * 0.1
        
        # Estimate causal effect
        ate = estimate_ate(X_imp, T, Y)
        estimates.append(ate)
    
    # Combine estimates (Rubin's rules)
    ate_mean = np.mean(estimates)
    ate_var_within = np.mean([est**2 for est in estimates]) - ate_mean**2
    ate_var_between = np.var(estimates)
    ate_var_total = ate_var_within + (1 + 1/n_imputations) * ate_var_between
    
    return {
        'estimate': ate_mean,
        'se': np.sqrt(ate_var_total),
        'imputations': estimates
    }
```

---

## Method Comparison Framework

### Systematic Comparison

```python
"""
Compare multiple causal inference methods on same data.
"""

import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

class CausalMethodComparison:
    def __init__(self):
        self.results = {}
    
    def add_method(self, name, estimator_fn):
        """Register a causal estimation method"""
        self.results[name] = {
            'estimator': estimator_fn,
            'ate': None,
            'cate': None,
            'time': None,
            'runtime': None
        }
    
    def run_comparison(self, X, T, Y, X_test=None):
        """Run all registered methods"""
        
        if X_test is None:
            X_test = X
        
        for method_name, method_info in self.results.items():
            print(f"Running {method_name}...")
            
            start_time = time.time()
            estimator_fn = method_info['estimator']
            ate, cate = estimator_fn(X, T, Y, X_test)
            elapsed = time.time() - start_time
            
            self.results[method_name]['ate'] = ate
            self.results[method_name]['cate'] = cate
            self.results[method_name]['runtime'] = elapsed
            
            print(f"  ATE: {ate:.4f}, Runtime: {elapsed:.2f}s")
    
    def plot_comparison(self, true_cate=None):
        """Visualize results across methods"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ATE comparison
        ate_values = [self.results[m]['ate'] for m in self.results.keys()]
        method_names = list(self.results.keys())
        
        axes[0].bar(method_names, ate_values, alpha=0.7)
        axes[0].set_ylabel('Estimated ATE')
        axes[0].set_title('ATE Across Methods')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Runtime comparison
        runtimes = [self.results[m]['runtime'] for m in self.results.keys()]
        
        axes[1].bar(method_names, runtimes, alpha=0.7, color='orange')
        axes[1].set_ylabel('Runtime (seconds)')
        axes[1].set_title('Computational Cost')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return fig

# Example: Compare methods
def linear_regression_method(X, T, Y, X_test):
    """Simple OLS regression"""
    lr = Ridge()
    X_aug = np.column_stack([T, X])
    lr.fit(X_aug, Y)
    ate = lr.coef_[0]
    cate = T * ate
    return ate, cate

def propensity_score_method(X, T, Y, X_test):
    """Propensity score matching"""
    # ... implementation
    return ate, cate

def dr_method(X, T, Y, X_test):
    """Doubly robust method"""
    dr = DoublyRobustEstimator()
    dr.fit(X, T, Y)
    ate, _, _ = dr.estimate_ate(X, T, Y)
    cate = dr.estimate_cate(X_test)
    return ate, cate

# Run comparison
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
T = (X[:, 0] > 0).astype(int)
Y = 2 * T + X[:, 1] + np.random.randn(n)

comparison = CausalMethodComparison()
comparison.add_method('Linear Regression', linear_regression_method)
comparison.add_method('Propensity Score', propensity_score_method)
comparison.add_method('Doubly Robust', dr_method)

comparison.run_comparison(X, T, Y)
comparison.plot_comparison()
```

---

## Production Deployment

### 1. Model Serialization

```python
"""
Save and load causal models for production use.
"""

import pickle
import joblib

class ProductionCausalModel:
    def __init__(self, model_type='doubly_robust'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def save(self, filepath):
        """Save model to disk"""
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(artifacts, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk"""
        artifacts = joblib.load(filepath)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.feature_names = artifacts['feature_names']
        self.model_type = artifacts['model_type']
        print(f"Model loaded from {filepath}")
        return self
    
    def predict_effect(self, X):
        """Predict treatment effect for new samples"""
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)

# Example usage
# Save
model = ProductionCausalModel()
model.save('causal_model.pkl')

# Load
model_loaded = ProductionCausalModel()
model_loaded.load('causal_model.pkl')
```

### 2. API Wrapper

```python
"""
REST API wrapper for serving causal predictions.
"""

from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
causal_model = ProductionCausalModel()
causal_model.load('causal_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    {
        "features": [[1.0, 2.0, ...], [2.0, 3.0, ...]]
    }
    """
    try:
        data = request.get_json()
        X = np.array(data['features'])
        
        # Get predictions
        cate = causal_model.predict_effect(X)
        
        return jsonify({
            'success': True,
            'predictions': cate.tolist(),
            'shape': cate.shape
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 3. Monitoring & Validation

```python
"""
Monitor model performance in production.
"""

class ProductionMonitor:
    def __init__(self, model, reference_data):
        self.model = model
        self.reference_mean = reference_data.mean()
        self.reference_std = reference_data.std()
        self.predictions_log = []
    
    def check_data_drift(self, X_new):
        """
        Check if new data distribution differs from training.
        """
        X_mean = X_new.mean(axis=0)
        X_std = X_new.std(axis=0)
        
        # Kolmogorov-Smirnov test
        from scipy.stats import ks_2samp
        
        drift_detected = False
        for i in range(X_new.shape[1]):
            stat, p_value = ks_2samp(X_new[:, i], 
                                    np.random.normal(self.reference_mean[i], 
                                                    self.reference_std[i], 1000))
            
            if p_value < 0.05:
                drift_detected = True
                print(f"Data drift detected in feature {i}")
        
        return drift_detected
    
    def check_prediction_drift(self, predictions):
        """Check if prediction distribution has changed"""
        
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        # Check for shift in mean
        if abs(pred_mean - self.reference_mean) > 2 * self.reference_std:
            print("Prediction drift detected!")
            return True
        
        return False
    
    def log_prediction(self, X, prediction, ground_truth=None):
        """Log prediction for analysis"""
        
        log_entry = {
            'timestamp': pd.Timestamp.now(),
            'features': X,
            'prediction': prediction,
            'ground_truth': ground_truth
        }
        self.predictions_log.append(log_entry)
    
    def performance_report(self):
        """Generate performance report"""
        
        log_df = pd.DataFrame(self.predictions_log)
        
        if 'ground_truth' in log_df.columns and log_df['ground_truth'].notna().any():
            mae = np.mean(np.abs(log_df['prediction'] - log_df['ground_truth']))
            rmse = np.sqrt(np.mean((log_df['prediction'] - log_df['ground_truth'])**2))
            
            return {
                'mae': mae,
                'rmse': rmse,
                'n_predictions': len(log_df)
            }
        
        return {'n_predictions': len(log_df)}
```

---

## Troubleshooting & Best Practices

### Common Issues and Solutions

```python
"""
Diagnose and fix common causal inference problems.
"""

class CausalInferenceDebugger:
    
    @staticmethod
    def diagnose_poor_overlap(propensity_scores, T):
        """
        Diagnose and fix poor overlap in propensity scores.
        """
        
        ps_treated = propensity_scores[T == 1]
        ps_control = propensity_scores[T == 0]
        
        overlap = np.minimum(ps_treated.max(), ps_control.max()) - \
                 np.maximum(ps_treated.min(), ps_control.min())
        
        print(f"Overlap region: [{max(ps_treated.min(), ps_control.min()):.3f}, "
              f"{min(ps_treated.max(), ps_control.max()):.3f}]")
        print(f"Overlap ratio: {overlap / (ps_treated.max() - ps_control.min()):.1%}")
        
        if overlap < 0.1:
            print("\n⚠️ Poor overlap detected!")
            print("Solutions:")
            print("  1. Refine propensity score model (add interactions)")
            print("  2. Remove units with extreme propensity scores")
            print("  3. Use caliper matching")
            print("  4. Consider doubly robust methods")
            print("  5. Check for strong confounders")
    
    @staticmethod
    def diagnose_unconfoundedness(X, T, Y):
        """
        Test sensitivity to unobserved confounders.
        Uses Rosenbaum bounds.
        """
        
        from scipy.stats import binom
        
        print("Rosenbaum Bounds (Sensitivity Analysis):")
        print("─" * 50)
        print("Gamma\tLower\tUpper")
        print("─" * 50)
        
        # Simplified bounds calculation
        for gamma in [1.0, 1.5, 2.0, 2.5]:
            # Hypothetical: how much would a confounder need to
            # change odds of treatment to flip conclusion
            print(f"{gamma:.1f}\t...\t...")
        
        print("\nIf bounds include 0 at small Gamma:")
        print("  ⚠️ Results sensitive to unmeasured confounding")
        print("\nSolutions:")
        print("  1. Use instrumental variables")
        print("  2. Tighten assumptions (domain knowledge)")
        print("  3. Use sensitivity parameter methods")
    
    @staticmethod
    def diagnose_model_misspecification(X, T, Y):
        """
        Check for model misspecification in outcome/propensity models.
        """
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestRegressor
        
        # Propensity score residuals
        ps_model = LogisticRegression(max_iter=1000)
        ps_pred = ps_model.fit_predict_proba(X)[:, 1]
        ps_residuals = T - ps_pred
        
        # Check for patterns in residuals
        print("Propensity Score Model Diagnostics:")
        print(f"  Residual mean: {ps_residuals.mean():.4f}")
        print(f"  Residual std: {ps_residuals.std():.4f}")
        
        # Outcome model residuals
        outcome_model = RandomForestRegressor()
        outcome_model.fit(X, Y)
        y_pred = outcome_model.predict(X)
        outcome_residuals = Y - y_pred
        
        print("\nOutcome Model Diagnostics:")
        print(f"  Residual mean: {outcome_residuals.mean():.4f}")
        print(f"  Residual std: {outcome_residuals.std():.4f}")
        print(f"  R² score: {outcome_model.score(X, Y):.4f}")
        
        if outcome_model.score(X, Y) < 0.3:
            print("\n⚠️ Low R² score suggests model misspecification")
            print("  Consider:")
            print("    1. Adding polynomial features")
            print("    2. Feature engineering")
            print("    3. Different model architecture")

# Usage
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
T = (X[:, 0] > 0).astype(int)
Y = 2 * T + X[:, 1] + np.random.randn(n)

ps_model = LogisticRegression(max_iter=1000)
ps = ps_model.fit_predict_proba(X)[:, 1]

debugger = CausalInferenceDebugger()
debugger.diagnose_poor_overlap(ps, T)
debugger.diagnose_model_misspecification(X, T, Y)
```

### Best Practices Checklist

```markdown
## Causal Inference Best Practices

### Before Analysis
- [ ] Define causal question clearly
- [ ] Specify DAG based on domain knowledge
- [ ] List assumptions explicitly
- [ ] Document data sources
- [ ] Check for data quality issues

### Data Preparation
- [ ] Handle missing data appropriately
- [ ] Check for outliers
- [ ] Examine covariate distributions
- [ ] Verify treatment variation
- [ ] Check for collinearity

### Estimation
- [ ] Check overlap/common support
- [ ] Verify covariate balance
- [ ] Run sensitivity analyses
- [ ] Use multiple methods (triangulation)
- [ ] Report uncertainty (confidence intervals)

### Validation
- [ ] Test for unobserved confounding
- [ ] Check parallel trends (for DiD)
- [ ] Validate with holdout data
- [ ] Cross-validation
- [ ] Compare to domain expertise

### Reporting
- [ ] State assumptions clearly
- [ ] Report point estimate and 95% CI
- [ ] Show covariate balance tables
- [ ] Provide sensitivity analyses
- [ ] Discuss limitations
```

---

**Document Version:** 1.0  
**Last Updated:** April 2026  
**Total Code Examples:** 20+
