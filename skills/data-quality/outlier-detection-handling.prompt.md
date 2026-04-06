# Outlier Detection and Handling Skill

**Author:** Shuvam Banerji Seal  
**Version:** 1.0  
**Last Updated:** April 2026

## Overview

Outlier detection is critical for data quality and model robustness. Outliers can significantly distort statistical analyses, inflate variance, and reduce model generalization. This skill covers statistical, density-based, and ensemble methods for robust outlier detection.

### Definition

An **outlier** is an observation that deviates substantially from the expected pattern or distribution of the data. Outliers can be:

1. **Univariate** - Extreme in a single dimension
2. **Multivariate** - Unusual in combination of features
3. **Contextual** - Abnormal within specific contexts
4. **Collective** - Subset of points forming unusual pattern

## Mathematical Formulations

### 1. Statistical Methods

#### Z-Score Method
$$Z_i = \frac{x_i - \mu}{\sigma}$$

Where:
- $\mu$ = mean
- $\sigma$ = standard deviation
- Threshold: typically |Z| > 3 (99.7% confidence)

**Detection Rule:** $|Z_i| > 3$ indicates outlier

#### Modified Z-Score (robust to outliers)
$$M_i = 0.6745 \cdot \frac{x_i - \text{median}}{\text{MAD}}$$

Where MAD = Median Absolute Deviation = $\text{median}(|x_i - \text{median}|)$

**Threshold:** |M| > 3.5

#### Interquartile Range (IQR) Method
$$IQR = Q_3 - Q_1$$

**Lower Bound:** $Q_1 - 1.5 \times IQR$  
**Upper Bound:** $Q_3 + 1.5 \times IQR$

Points outside these bounds are outliers.

### 2. Distance-Based Methods

#### Mahalanobis Distance
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

Where:
- $\mu$ = mean vector
- $\Sigma$ = covariance matrix
- Threshold: $D_M > \chi^2_{d,\alpha}$ (d = dimensions)

### 3. Density-Based Methods

#### Local Outlier Factor (LOF)
$$LOF_k(p) = \frac{1}{k} \sum_{o \in N_k(p)} \frac{LRD_k(o)}{LRD_k(p)}$$

Where:
- $N_k(p)$ = k-nearest neighbors
- $LRD_k$ = Local Reachability Density

**Interpretation:**
- LOF ≈ 1: Point is similar to neighbors
- LOF > 1: Point is outlier
- Threshold: LOF > 1.5-2.0

### 4. Isolation Forest

**Isolation Tree Recursion:**
$$h(x) = \text{average path length to isolate point } x$$

**Anomaly Score:**
$$s(x) = 2^{-\frac{h(x)}{c(n)}}$$

Where $c(n)$ = average path length for unsuccessful search

**Range:** [0, 1]
- s(x) → 1: likely outlier
- s(x) → 0.5: normal point

## Implementation

### Python Code Examples

#### 1. Statistical Outlier Detection

```python
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict

class StatisticalOutlierDetector:
    """Statistical methods for outlier detection"""
    
    @staticmethod
    def zscore_detection(data: pd.Series, threshold: float = 3.0) -> Tuple[pd.Series, np.ndarray]:
        """
        Detect outliers using Z-score
        
        Args:
            data: Input series
            threshold: Z-score threshold (default 3.0 = 99.7%)
        
        Returns:
            Tuple of (outlier boolean mask, z-scores)
        """
        z_scores = np.abs(stats.zscore(data.dropna()))
        outliers = z_scores > threshold
        
        return outliers, z_scores
    
    @staticmethod
    def modified_zscore_detection(data: pd.Series, threshold: float = 3.5) -> pd.Series:
        """
        Robust Z-score using median absolute deviation (MAD)
        """
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return pd.Series([False] * len(data), index=data.index)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    @staticmethod
    def iqr_detection(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """
        Interquartile Range method
        
        Args:
            data: Input series
            multiplier: IQR multiplier (1.5 = standard, 1.0 = stricter)
        
        Returns:
            Boolean mask of outliers
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (data < lower_bound) | (data > upper_bound)

# Usage Example
data = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
detector = StatisticalOutlierDetector()

z_outliers, scores = detector.zscore_detection(data)
print(f"Z-score outliers: {data[z_outliers].tolist()}")

iqr_outliers = detector.iqr_detection(data)
print(f"IQR outliers: {data[iqr_outliers].tolist()}")
```

#### 2. Density-Based Outlier Detection (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

class DensityOutlierDetector:
    """Density-based outlier detection methods"""
    
    @staticmethod
    def lof_detection(X: np.ndarray, n_neighbors: int = 20, threshold: float = 1.5) -> Dict:
        """
        Local Outlier Factor detection
        
        Args:
            X: Feature matrix (n_samples, n_features)
            n_neighbors: Number of neighbors to consider
            threshold: LOF threshold for outlier classification
        
        Returns:
            Dictionary with outlier information
        """
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        outlier_labels = lof.fit_predict(X_scaled)
        lof_scores = -lof.negative_outlier_factor_
        
        return {
            'outlier_mask': outlier_labels == -1,
            'lof_scores': lof_scores,
            'outlier_indices': np.where(outlier_labels == -1)[0],
            'n_outliers': (outlier_labels == -1).sum()
        }

# Usage
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
X[0] = [100, 100]  # Inject outlier

results = DensityOutlierDetector.lof_detection(X, n_neighbors=20)
print(f"LOF detected {results['n_outliers']} outliers")
print(f"LOF scores range: [{results['lof_scores'].min():.2f}, {results['lof_scores'].max():.2f}]")
```

#### 3. Isolation Forest

```python
from sklearn.ensemble import IsolationForest

class EnsembleOutlierDetector:
    """Ensemble-based outlier detection methods"""
    
    @staticmethod
    def isolation_forest(X: np.ndarray, contamination: float = 0.1) -> Dict:
        """
        Isolation Forest for outlier detection
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of outliers
        
        Returns:
            Detection results
        """
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        outlier_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.score_samples(X)
        
        return {
            'outlier_mask': outlier_labels == -1,
            'anomaly_scores': anomaly_scores,
            'outlier_indices': np.where(outlier_labels == -1)[0],
            'n_outliers': (outlier_labels == -1).sum()
        }

# Usage
results = EnsembleOutlierDetector.isolation_forest(X, contamination=0.05)
print(f"Isolation Forest detected {results['n_outliers']} outliers")
```

#### 4. Multivariate Outlier Detection

```python
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

class MultivariateOutlierDetector:
    """Multivariate outlier detection using Mahalanobis distance"""
    
    @staticmethod
    def mahalanobis_detection(X: np.ndarray, alpha: float = 0.05) -> Dict:
        """
        Detect outliers using Mahalanobis distance
        
        Args:
            X: Feature matrix
            alpha: Significance level
        
        Returns:
            Detection results
        """
        # Calculate mean and covariance
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        
        # Handle singular covariance matrix
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            inv_cov = np.linalg.pinv(cov)
        
        # Calculate Mahalanobis distance for each point
        distances = np.array([
            np.sqrt(mahalanobis(xi, mean, inv_cov)) for xi in X
        ])
        
        # Chi-square threshold (d = n_features)
        n_features = X.shape[1]
        threshold = chi2.ppf(1 - alpha, df=n_features)
        
        return {
            'mahalanobis_distances': distances,
            'threshold': threshold,
            'outlier_mask': distances > threshold,
            'outlier_indices': np.where(distances > threshold)[0],
            'n_outliers': (distances > threshold).sum()
        }

# Usage
results = MultivariateOutlierDetector.mahalanobis_detection(X)
print(f"Mahalanobis method detected {results['n_outliers']} outliers")
print(f"Threshold: {results['threshold']:.2f}")
```

#### 5. Integrated Outlier Detection Pipeline

```python
class OutlierDetectionPipeline:
    """Ensemble approach combining multiple methods"""
    
    def __init__(self, X: np.ndarray, contamination: float = 0.05):
        self.X = X
        self.contamination = contamination
        self.results = {}
    
    def detect_all_methods(self) -> Dict:
        """Run multiple detection methods"""
        
        # Statistical
        z_scores = np.abs(stats.zscore(self.X, axis=0))
        z_outliers = (z_scores > 3).any(axis=1)
        
        # LOF
        lof_detector = DensityOutlierDetector()
        lof_results = lof_detector.lof_detection(self.X)
        
        # Isolation Forest
        iso_detector = EnsembleOutlierDetector()
        iso_results = iso_detector.isolation_forest(self.X, self.contamination)
        
        # Mahalanobis
        maha_detector = MultivariateOutlierDetector()
        maha_results = maha_detector.mahalanobis_detection(self.X)
        
        self.results = {
            'zscore': z_outliers,
            'lof': lof_results['outlier_mask'],
            'isolation_forest': iso_results['outlier_mask'],
            'mahalanobis': maha_results['outlier_mask']
        }
        
        return self.results
    
    def ensemble_consensus(self, min_methods: int = 2) -> np.ndarray:
        """
        Identify outliers by consensus across multiple methods
        
        Args:
            min_methods: Minimum number of methods to flag as outlier
        
        Returns:
            Boolean array of consensus outliers
        """
        if not self.results:
            self.detect_all_methods()
        
        outlier_count = np.zeros(len(self.X))
        for method, mask in self.results.items():
            outlier_count += mask.astype(int)
        
        return outlier_count >= min_methods
    
    def generate_report(self) -> Dict:
        """Generate comprehensive outlier detection report"""
        if not self.results:
            self.detect_all_methods()
        
        consensus = self.ensemble_consensus()
        
        report = {
            'total_points': len(self.X),
            'method_results': {
                method: {
                    'n_outliers': mask.sum(),
                    'percentage': (mask.sum() / len(self.X)) * 100
                }
                for method, mask in self.results.items()
            },
            'consensus_outliers': consensus.sum(),
            'consensus_percentage': (consensus.sum() / len(self.X)) * 100
        }
        
        return report

# Usage
pipeline = OutlierDetectionPipeline(X, contamination=0.05)
report = pipeline.generate_report()

print("\nOutlier Detection Report:")
print(f"Total points: {report['total_points']}")
print("\nPer-method detection:")
for method, stats in report['method_results'].items():
    print(f"  {method}: {stats['n_outliers']} outliers ({stats['percentage']:.1f}%)")
print(f"\nConsensus outliers: {report['consensus_outliers']} ({report['consensus_percentage']:.1f}%)")
```

## Authoritative Sources

1. **Chandola et al. (2009)** - "Anomaly Detection: A Survey"
   - Comprehensive taxonomy of outlier detection techniques
   - IEEE Computing Surveys, Vol. 41, No. 3

2. **Liu et al. (2008)** - "Isolation Forest"
   - Original paper introducing Isolation Forest
   - ICDM 2008

3. **Breunig et al. (2000)** - "LOF: Identifying Density-Based Local Outliers"
   - Introduces Local Outlier Factor
   - SIGMOD 2000

4. **Scikit-learn Outlier Detection** - https://scikit-learn.org/stable/modules/outlier_detection.html
   - Comprehensive implementations and documentation
   - Multiple algorithms and use cases

5. **Wikipedia: Outlier** - Statistical definitions and properties
   - Mathematical formulations
   - Detection methodologies

## Practical Checklist

- [ ] Understand domain to distinguish outliers from valid extremes
- [ ] Apply multiple detection methods
- [ ] Use consensus approach for robust detection
- [ ] Visualize detected outliers in original data
- [ ] Investigate root cause of outliers
- [ ] Document handling decision (remove, retain, treat)
- [ ] Consider impact on model training and evaluation
- [ ] Monitor for new outlier patterns over time
- [ ] Adjust thresholds based on business requirements

## Edge Cases & Handling Strategies

### Case 1: Legitimate Extreme Values
Some outliers represent valid data (e.g., VIP customers, rare events)
**Strategy:** Domain validation before removal

### Case 2: Multiple Outlier Types
Different outlier patterns in different features
**Strategy:** Use multivariate methods like LOF or Mahalanobis

### Case 3: High-Dimensional Data
Curse of dimensionality affects distance-based methods
**Strategy:** Use Isolation Forest or feature selection first

### Case 4: Clustered Outliers
Groups of anomalous points rather than isolated instances
**Strategy:** Use LOF or DBSCAN

## Handling Strategies

| Strategy | Pros | Cons | When to Use |
|----------|------|------|------------|
| **Remove** | Simple, clean | Loss of information | Measurement errors |
| **Impute** | Retain data | May introduce bias | Missing values |
| **Cap/Floor** | Reduce impact | Loses extreme value | Statistical outliers |
| **Separate Model** | Captures pattern | Complex | Natural sub-groups |
| **Weighted Loss** | Robust training | Hyperparameter tuning | ML training |

## Performance Metrics

```python
def evaluate_outlier_detection(y_true, y_pred):
    """Evaluate outlier detection quality"""
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'true_positive_rate': tp / (tp + fn),
        'false_positive_rate': fp / (fp + tn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1_score': f1_score(y_true, y_pred)
    }
    
    return metrics
```

## Visualization

```python
import matplotlib.pyplot as plt

def visualize_outliers(X, outlier_mask, title="Outlier Detection Results"):
    """Visualize detected outliers in 2D"""
    if X.shape[1] >= 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[~outlier_mask, 0], X[~outlier_mask, 1], 
                   label='Normal', alpha=0.7)
        plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
                   label='Outlier', color='red', marker='x', s=200)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.show()
```

## Conclusion

Robust outlier detection requires combining multiple methods and domain expertise. No single method is optimal for all scenarios—ensemble approaches leveraging statistical, density-based, and isolation methods provide the most reliable detection across diverse data characteristics.

---

**Last Reviewed:** April 2026  
**Skill Status:** Production Ready
