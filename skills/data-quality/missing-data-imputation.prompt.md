# Missing Data Imputation Skill

**Author:** Shuvam Banerji Seal  
**Version:** 1.0  
**Last Updated:** April 2026

## Overview

Missing data is ubiquitous in real-world datasets and can significantly impact model performance if not handled properly. This skill covers missing data mechanisms, imputation methods, multiple imputation strategies, and diagnostic techniques.

### Definition

**Missing Data:** Values that are not available in a dataset due to data collection, processing, or storage issues.

**Three Missing Mechanisms (Rubin, 1976):**

1. **MCAR** (Missing Completely At Random) - P(missing | X, Y) = P(missing)
   - Missingness independent of observed and unobserved data
   - Most benign; can safely ignore or impute

2. **MAR** (Missing At Random) - P(missing | X, Y) = P(missing | X)
   - Missingness depends only on observed data
   - Addressable through imputation using available features

3. **MNAR** (Missing Not At Random) - P(missing | X, Y) depends on unobserved Y
   - Missingness depends on the missing value itself
   - Most problematic; requires domain knowledge

## Mathematical Formulations

### Missing Data Proportion

$$p_j = \frac{n_{missing,j}}{n_{total}}$$

Where $p_j$ = proportion of missing values in feature j

### Missingness Pattern Analysis

For multivariate missing data, define missingness indicator:
$$R_{ij} = \begin{cases} 1 & \text{if } X_{ij} \text{ is missing} \\ 0 & \text{if } X_{ij} \text{ is observed} \end{cases}$$

**Monotone Pattern:** Rows can be ordered so missing values form a triangular pattern  
**Arbitrary Pattern:** No specific structure to missing data

### Multiple Imputation Combining Rules (Rubin, 1987)

For m imputed datasets, estimate parameter θ and variance:

$$\hat{\theta}_{MI} = \frac{1}{m}\sum_{k=1}^{m}\hat{\theta}_k$$

$$Var(\hat{\theta}_{MI}) = \overline{U} + B \cdot \frac{m+1}{m}$$

Where:
- $\overline{U}$ = average within-imputation variance
- $B$ = between-imputation variance

### Mean Squared Error (MSE) for Imputation

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(\hat{x}_i - x_i)^2$$

## Implementation

### Python Code Examples

#### 1. Missing Data Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class MissingDataAnalyzer:
    """Analyze missing data patterns and mechanisms"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.analysis = {}
    
    def analyze_missing_pattern(self) -> Dict:
        """Comprehensive missing data analysis"""
        
        missing_data = {
            'column': [],
            'missing_count': [],
            'missing_percentage': [],
            'data_type': []
        }
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            missing_data['column'].append(col)
            missing_data['missing_count'].append(missing_count)
            missing_data['missing_percentage'].append(missing_pct)
            missing_data['data_type'].append(str(self.df[col].dtype))
        
        self.analysis['missing_summary'] = pd.DataFrame(missing_data).sort_values(
            'missing_percentage', ascending=False
        )
        
        return self.analysis['missing_summary']
    
    def detect_missingness_mechanism(self) -> Dict:
        """
        Attempt to infer missingness mechanism
        Returns indicators suggesting MCAR, MAR, or MNAR
        """
        mechanisms = {}
        
        for col in self.df.columns:
            missing_mask = self.df[col].isna()
            
            if missing_mask.sum() == 0:
                continue
            
            # Check correlation between missingness and other variables
            correlations = []
            for other_col in self.df.columns:
                if other_col == col:
                    continue
                
                if self.df[other_col].dtype in [np.float64, np.int64]:
                    # Point-biserial correlation
                    valid_data = self.df[~missing_mask]
                    if len(valid_data) > 1:
                        corr = valid_data[other_col].corr(
                            pd.Series(missing_mask[~missing_mask].astype(int))
                        )
                        if abs(corr) > 0.3:
                            correlations.append((other_col, corr))
            
            if len(correlations) == 0:
                mechanisms[col] = 'Likely MCAR'
            elif len(correlations) > 0:
                mechanisms[col] = f'Likely MAR (correlated with {correlations[0][0]})'
            
        self.analysis['mechanisms'] = mechanisms
        return mechanisms
    
    def visualize_missing_pattern(self):
        """Visualize missing data pattern"""
        missing_matrix = self.df.isnull().astype(int)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heatmap of missing values
        ax1.imshow(missing_matrix, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Samples')
        ax1.set_title('Missing Data Pattern')
        
        # Bar plot of missing percentages
        missing_pcts = self.df.isnull().sum() / len(self.df) * 100
        missing_pcts = missing_pcts[missing_pcts > 0].sort_values(ascending=False)
        
        missing_pcts.plot(kind='barh', ax=ax2, color='coral')
        ax2.set_xlabel('Missing Percentage (%)')
        ax2.set_title('Missing Data by Column')
        
        plt.tight_layout()
        plt.show()

# Usage
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5, np.nan],
    'B': [10, np.nan, 30, np.nan, 50, 60],
    'C': [100, 200, 300, 400, 500, 600],
    'D': ['a', np.nan, 'c', 'd', np.nan, 'f']
})

analyzer = MissingDataAnalyzer(df)
print(analyzer.analyze_missing_pattern())
print("\nMissingness Mechanisms:")
print(analyzer.detect_missingness_mechanism())
```

#### 2. Simple Imputation Methods

```python
class SimpleImputer:
    """Basic imputation strategies"""
    
    @staticmethod
    def mean_imputation(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with column mean"""
        df_imputed = df.copy()
        
        for col in df_imputed.select_dtypes(include=[np.number]).columns:
            if df_imputed[col].isnull().any():
                df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
        
        return df_imputed
    
    @staticmethod
    def median_imputation(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with column median (robust to outliers)"""
        df_imputed = df.copy()
        
        for col in df_imputed.select_dtypes(include=[np.number]).columns:
            if df_imputed[col].isnull().any():
                df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
        
        return df_imputed
    
    @staticmethod
    def mode_imputation(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing categorical values with mode"""
        df_imputed = df.copy()
        
        for col in df_imputed.select_dtypes(include=['object']).columns:
            if df_imputed[col].isnull().any():
                mode_val = df_imputed[col].mode()
                if len(mode_val) > 0:
                    df_imputed[col].fillna(mode_val[0], inplace=True)
        
        return df_imputed
    
    @staticmethod
    def forward_fill(df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill for time-series data"""
        return df.fillna(method='ffill')
    
    @staticmethod
    def backward_fill(df: pd.DataFrame) -> pd.DataFrame:
        """Backward fill for time-series data"""
        return df.fillna(method='bfill')
    
    @staticmethod
    def constant_imputation(df: pd.DataFrame, constant: float = 0) -> pd.DataFrame:
        """Impute with a constant value"""
        return df.fillna(constant)

# Usage
df_mean = SimpleImputer.mean_imputation(df)
print("After mean imputation:")
print(df_mean)
```

#### 3. K-Nearest Neighbors Imputation

```python
from sklearn.impute import KNNImputer

class KNNImputationMethod:
    """K-Nearest Neighbors based imputation"""
    
    @staticmethod
    def knn_imputation(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Impute missing values using k-nearest neighbors
        
        Args:
            df: Input dataframe with missing values
            n_neighbors: Number of neighbors to consider
        
        Returns:
            Imputed dataframe
        """
        # Separate numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_numeric = df[numeric_cols].copy()
        df_numeric_imputed = pd.DataFrame(
            imputer.fit_transform(df_numeric),
            columns=numeric_cols
        )
        
        # Handle categorical with mode
        df_categorical = df[categorical_cols].copy()
        for col in categorical_cols:
            if df_categorical[col].isnull().any():
                df_categorical[col].fillna(
                    df_categorical[col].mode()[0], 
                    inplace=True
                )
        
        # Combine
        result = pd.concat([df_numeric_imputed, df_categorical], axis=1)
        return result[df.columns]  # Restore original column order

# Usage
df_knn = KNNImputationMethod.knn_imputation(df, n_neighbors=3)
print("After KNN imputation:")
print(df_knn)
```

#### 4. Multiple Imputation by Chained Equations (MICE)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

class MICEImputation:
    """Multiple Imputation by Chained Equations"""
    
    @staticmethod
    def mice_imputation(df: pd.DataFrame, max_iter: int = 10, 
                        random_state: int = 42) -> Tuple[pd.DataFrame, List]:
        """
        MICE imputation - iterative method using regression
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # IterativeImputer uses BayesianRidge by default
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=random_state,
            verbose=0
        )
        
        df_numeric = df[numeric_cols].copy()
        df_numeric_imputed = pd.DataFrame(
            imputer.fit_transform(df_numeric),
            columns=numeric_cols
        )
        
        # Handle categorical
        df_categorical = df.select_dtypes(include=['object']).copy()
        for col in df_categorical.columns:
            if df_categorical[col].isnull().any():
                df_categorical[col].fillna(
                    df_categorical[col].mode()[0], 
                    inplace=True
                )
        
        # Combine
        result = pd.concat([df_numeric_imputed, df_categorical], axis=1)
        return result[df.columns], imputer.n_iter_

# Usage
df_mice, n_iters = MICEImputation.mice_imputation(df)
print(f"MICE imputation converged in {n_iters} iterations")
print(df_mice)
```

#### 5. Multiple Imputation Framework

```python
class MultipleImputationFramework:
    """Create and analyze multiple imputations"""
    
    def __init__(self, df: pd.DataFrame, n_imputations: int = 5):
        self.df = df
        self.n_imputations = n_imputations
        self.imputations = []
    
    def create_multiple_imputations(self, method: str = 'mice') -> List[pd.DataFrame]:
        """
        Create m imputed datasets
        
        Args:
            method: 'mice', 'knn', or 'mean'
        """
        for i in range(self.n_imputations):
            if method == 'mice':
                df_imputed, _ = MICEImputation.mice_imputation(
                    self.df, 
                    random_state=42 + i
                )
            elif method == 'knn':
                df_imputed = KNNImputationMethod.knn_imputation(self.df)
                # Add randomness for multiple imputations
                numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
                noise = np.random.normal(0, 0.01, df_imputed[numeric_cols].shape)
                df_imputed[numeric_cols] += noise
            else:  # mean
                df_imputed = SimpleImputer.mean_imputation(self.df)
            
            self.imputations.append(df_imputed)
        
        return self.imputations
    
    def combine_estimates(self, estimator_func, *args, **kwargs):
        """
        Combine estimates from multiple imputations using Rubin's rules
        
        Args:
            estimator_func: Function that takes dataframe and returns estimate
        """
        estimates = []
        variances = []
        
        for df_imputed in self.imputations:
            estimate = estimator_func(df_imputed, *args, **kwargs)
            estimates.append(estimate)
        
        # Rubin's combination rules
        # Point estimate
        combined_estimate = np.mean(estimates)
        
        # Within-imputation variance
        within_var = np.mean([v for v in variances]) if variances else np.var(estimates)
        
        # Between-imputation variance
        between_var = np.var(estimates)
        
        # Total variance
        total_var = within_var + between_var * (self.n_imputations + 1) / self.n_imputations
        
        return {
            'estimate': combined_estimate,
            'within_imputation_variance': within_var,
            'between_imputation_variance': between_var,
            'total_variance': total_var,
            'confidence_interval': (
                combined_estimate - 1.96 * np.sqrt(total_var),
                combined_estimate + 1.96 * np.sqrt(total_var)
            )
        }

# Usage
mi_framework = MultipleImputationFramework(df, n_imputations=5)
imputations = mi_framework.create_multiple_imputations(method='mice')

print(f"Created {len(imputations)} imputed datasets")
print(f"First imputation:\n{imputations[0]}")
```

#### 6. Imputation Quality Assessment

```python
class ImputationQualityAssessment:
    """Evaluate imputation quality"""
    
    @staticmethod
    def assess_imputation_quality(df_original: pd.DataFrame, 
                                 df_imputed: pd.DataFrame,
                                 missing_indices: List[Tuple]) -> Dict:
        """
        Compare imputed values with original (for simulated missing data)
        
        Args:
            df_original: Original complete dataframe
            df_imputed: Dataframe after imputation
            missing_indices: List of (row, col) indices that were imputed
        """
        mse = 0
        mae = 0
        rmse = 0
        count = 0
        
        for row_idx, col_idx in missing_indices:
            col_name = df_original.columns[col_idx]
            
            original_val = df_original.iloc[row_idx, col_idx]
            imputed_val = df_imputed.iloc[row_idx, col_idx]
            
            if pd.notna(original_val):
                error = original_val - imputed_val
                mse += error ** 2
                mae += abs(error)
                count += 1
        
        if count > 0:
            mse /= count
            mae /= count
            rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'n_imputed': count
        }
        
        return metrics
    
    @staticmethod
    def simulate_and_evaluate(df: pd.DataFrame, 
                            missing_rate: float = 0.2,
                            method: str = 'mean') -> Dict:
        """
        Simulate missing data and evaluate imputation performance
        """
        # Create copy and introduce missing values
        df_with_missing = df.copy()
        missing_mask = np.random.random(df.shape) < missing_rate
        missing_indices = list(zip(*np.where(missing_mask)))
        
        # Only for numeric columns
        for row_idx, col_idx in missing_indices:
            if df.iloc[row_idx, col_idx].dtype in [np.float64, np.int64]:
                df_with_missing.iloc[row_idx, col_idx] = np.nan
        
        # Impute
        if method == 'mean':
            df_imputed = SimpleImputer.mean_imputation(df_with_missing)
        elif method == 'knn':
            df_imputed = KNNImputationMethod.knn_imputation(df_with_missing)
        else:  # mice
            df_imputed, _ = MICEImputation.mice_imputation(df_with_missing)
        
        # Evaluate
        quality = ImputationQualityAssessment.assess_imputation_quality(
            df, df_imputed, missing_indices
        )
        
        return {
            'missing_count': len(missing_indices),
            'quality_metrics': quality,
            'method': method
        }

# Usage
evaluation = ImputationQualityAssessment.simulate_and_evaluate(df, missing_rate=0.3, method='mice')
print(f"Imputation Quality (MICE):")
print(f"  RMSE: {evaluation['quality_metrics']['rmse']:.4f}")
print(f"  MAE:  {evaluation['quality_metrics']['mae']:.4f}")
```

## Authoritative Sources

1. **Rubin (1987)** - "Multiple Imputation for Nonresponse in Surveys"
   - Foundational work on multiple imputation
   - Statistical theory and combining rules

2. **Little & Rubin (2002)** - "Statistical Analysis with Missing Data"
   - Comprehensive treatment of missing data mechanisms
   - MCAR, MAR, MNAR theory and practice

3. **Buuren & Groothuis-Oudshoorn (2011)** - "Mice: Multivariate Imputation by Chained Equations"
   - Original MICE paper
   - Journal of Statistical Software

4. **Scikit-learn Imputation** - https://scikit-learn.org/stable/modules/impute.html
   - SimpleImputer, KNNImputer, IterativeImputer implementations
   - Best practices and examples

5. **Kingma & Welling (2013)** - "Auto-Encoding Variational Bayes"
   - Modern deep learning approaches to imputation
   - Foundational work for neural imputation methods

## Practical Checklist

- [ ] Analyze missing data patterns (MCAR, MAR, MNAR)
- [ ] Calculate missing data proportions per variable
- [ ] Decide on deletion vs. imputation based on pattern
- [ ] Choose appropriate imputation method
- [ ] Handle multiple imputation if uncertainty is critical
- [ ] Validate imputation on hold-out test set
- [ ] Document all imputation decisions and parameters
- [ ] Create indicators for originally missing values
- [ ] Monitor imputation impact on model performance
- [ ] Consider domain expertise in imputation strategy

## Edge Cases & Considerations

### Case 1: High Missing Rate (>50%)
**Problem:** Limited information for imputation  
**Solution:** Consider deletion or separate model for missing subgroup

### Case 2: MNAR Mechanism
**Problem:** Missing values depend on unobserved data  
**Solution:** Sensitivity analysis with different assumptions

### Case 3: Categorical with Many Categories
**Problem:** Mode imputation loses information  
**Solution:** Use multiple imputation or probabilistic approach

### Case 4: Time-Series Data
**Problem:** Sequential dependencies violated by standard imputation  
**Solution:** Time-aware methods (forward fill, interpolation)

### Case 5: Missingness Creates New Information
**Problem:** Fact of missingness is itself predictive  
**Solution:** Create missingness indicator variables

## Imputation Method Comparison

| Method | Speed | Accuracy | Assumptions | Code Complexity |
|--------|-------|----------|-------------|-----------------|
| **Mean** | Very Fast | Poor | MCAR | Low |
| **KNN** | Medium | Good | MAR, local similarity | Medium |
| **MICE** | Slow | Excellent | MAR, no perfect collinearity | High |
| **Deep Learning** | Slow | Excellent | MAR, large sample | Very High |
| **Domain Knowledge** | Variable | Excellent | Domain expert available | Variable |

## Missing Data Indicator Variables

```python
def create_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary indicator variables for originally missing values"""
    result = df.copy()
    
    for col in df.columns:
        if df[col].isnull().any():
            result[f'{col}_missing'] = df[col].isnull().astype(int)
    
    return result

# Usage
df_with_indicators = create_missing_indicators(df)
print(df_with_indicators)
```

## Conclusion

Proper handling of missing data is crucial for model integrity. The choice between deletion, imputation, and advanced techniques depends on the missing mechanism, data characteristics, and business context. Multiple imputation provides the most rigorous statistical treatment when uncertainty quantification is important.

---

**Last Reviewed:** April 2026  
**Skill Status:** Production Ready
