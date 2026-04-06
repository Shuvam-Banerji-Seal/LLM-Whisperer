# Data Quality Skills - Quick Reference Card

## When to Use Which Skill

### 1. Data Quality Assessment
**Use when:**
- Establishing baseline data quality metrics
- Creating quality monitoring dashboards
- Validating data before ML training
- Auditing data sources

**Key Classes:**
```python
DataQualityAssessor(df).generate_report()
DataQualityMonitor().log_assessment(df)
```

**Primary Metrics:**
- Completeness (0-100%)
- Consistency (CV < 0.5)
- Accuracy (Precision/Recall/F1)
- Validity (format conformance)
- Uniqueness (duplicate detection)

**Time Complexity:** O(n*m) where n=rows, m=cols

---

### 2. Outlier Detection & Handling
**Use when:**
- Identifying anomalous data points
- Detecting fraud/unusual patterns
- Quality assurance for sensor data
- Statistical analysis preparation

**Key Classes:**
```python
StatisticalOutlierDetector.zscore_detection(data)
DensityOutlierDetector.lof_detection(X)
EnsembleOutlierDetector.isolation_forest(X)
OutlierDetectionPipeline(X, y).ensemble_consensus()
```

**Best For Different Scenarios:**
| Scenario | Method | Why |
|----------|--------|-----|
| Univariate anomalies | Z-score | Fast, interpretable |
| Local density issues | LOF | Catches local patterns |
| General-purpose | Isolation Forest | Robust, scalable |
| Multivariate | Mahalanobis | Uses correlations |

**Quick Thresholds:**
- Z-score: |Z| > 3 (99.7% confidence)
- IQR: Beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
- LOF: > 1.5 indicates outlier
- Isolation: s(x) > 0.7 indicates outlier

---

### 3. Class Imbalance Handling
**Use when:**
- Training on imbalanced classification data
- Minority class accuracy is critical
- Binary or multi-class classification
- Fraud detection, rare disease prediction

**Imbalance Severity:**
```
Mild:     IR < 2:1       → No special handling needed
Moderate: 2:1 ≤ IR < 10:1  → SMOTE or class weights
Severe:   10:1 ≤ IR < 100:1 → SMOTE + cost-sensitive
Extreme:  IR ≥ 100:1       → Hierarchical approach
```

**Key Classes:**
```python
# Detection
analyzer = ImbalanceAnalyzer(y)
analyzer.analyze()  # Get severity

# Resampling
ResamplingPipeline.smote_oversampling(X, y)

# Cost-sensitive
CostSensitiveLearning.balanced_logistic_regression(X, y)

# Evaluation
ImbalancedEvaluator.balanced_metrics(y_true, y_pred)
ImbalancedEvaluator.probability_metrics(y_true, y_proba)
```

**Quick Metric Selection:**
| Metric | When to Use |
|--------|------------|
| Accuracy | ❌ NEVER for imbalanced data |
| Balanced Accuracy | ✓ Binary classification |
| G-Mean | ✓ Sensitivity/Specificity balance |
| F1-Score | ✓ Binary, precision-recall |
| ROC-AUC | ✓ Probability-based evaluation |
| PR-AUC | ✓ Extreme imbalance |

**Optimal Strategy Selection:**
- Data-level: SMOTE for moderate imbalance
- Algorithm-level: Cost weights for ensemble
- Hybrid: SMOTE + threshold optimization
- Extreme: Hierarchical + ensemble

---

### 4. Missing Data Imputation
**Use when:**
- Dataset has missing values (NaNs)
- Columns have different missing patterns
- Need multiple imputations for uncertainty
- Time-series with gaps

**Missing Mechanism:**
```
MCAR: P(missing|X,Y) = P(missing)
  → Safe to impute or delete
  → Methods: Any (mean, KNN, MICE)

MAR: P(missing|X,Y) = P(missing|X)
  → Use variables predicting missingness
  → Methods: MICE, KNN, regression

MNAR: P(missing|X,Y) depends on Y
  → Most problematic
  → Methods: Multiple imputation, sensitivity analysis
```

**Key Classes:**
```python
# Analysis
analyzer = MissingDataAnalyzer(df)
analyzer.detect_missingness_mechanism()

# Simple methods
SimpleImputer.mean_imputation(df)
SimpleImputer.median_imputation(df)

# Advanced methods
KNNImputationMethod.knn_imputation(df, n_neighbors=5)
MICEImputation.mice_imputation(df)

# Multiple imputation
mi_fw = MultipleImputationFramework(df, n_imputations=5)
mi_fw.create_multiple_imputations(method='mice')
```

**Method Selection by Data Type:**
| Method | Numeric | Categorical | Time-Series | When to Use |
|--------|---------|-------------|-------------|-----------|
| Mean | ✓ | ✗ | ✗ | Simple, fast |
| Median | ✓ | ✗ | ✗ | Robust to outliers |
| Mode | ✓ | ✓ | ✗ | Most frequent |
| KNN | ✓ | ✓ | ~ | Local patterns |
| MICE | ✓ | ✓ | ~ | Complex relationships |
| Forward Fill | ~ | ~ | ✓ | Time-series |
| Multiple | ✓ | ✓ | ~ | Uncertainty estimation |

**Missing Rate Thresholds:**
- 0-5%: Any method is fine
- 5-20%: Use KNN or MICE
- 20-40%: Use MICE with multiple imputations
- 40%+: Consider deletion or separate model

---

## Integration Workflow

```
Raw Data
    ↓
[1] Quality Assessment
    └─→ Completeness, consistency checks
    └─→ Generate baseline report
    
    ↓
[2] Missing Data Imputation
    └─→ Analyze missingness mechanism
    └─→ Apply appropriate imputation
    └─→ Validate imputation quality
    
    ↓
[3] Outlier Detection
    └─→ Run multiple detection methods
    └─→ Apply consensus approach
    └─→ Document handling decision
    
    ↓
[4] Class Imbalance Handling (if classification)
    └─→ Quantify imbalance severity
    └─→ Apply resampling + cost-sensitive learning
    └─→ Optimize decision threshold
    
    ↓
Feature Engineering
```

---

## Common Pitfalls & Solutions

### Data Quality Assessment
❌ Using accuracy as only metric for quality
✓ Use weighted combination of 5 dimensions

❌ Ignoring temporal quality changes
✓ Set up monitoring dashboard

### Outlier Detection
❌ Removing all outliers without investigation
✓ Analyze root causes, consider domain context

❌ Using same threshold for all features
✓ Normalize/standardize before detection

### Class Imbalance
❌ Oversampling on entire dataset before split
✓ Oversample only training set, test on original

❌ Using accuracy for evaluation
✓ Use balanced accuracy or F1-score

### Missing Data Imputation
❌ Imputing without understanding mechanism
✓ Analyze MCAR/MAR/MNAR first

❌ Single imputation for critical analysis
✓ Use multiple imputation with Rubin's rules

---

## Performance Guidelines

### Speed (for 100K rows)
- Z-score detection: < 100ms
- LOF detection: 5-10 seconds
- Isolation Forest: 1-2 seconds
- SMOTE: 2-5 seconds
- MICE: 30-60 seconds

### Memory Requirements
- Statistical methods: O(1) per feature
- LOF: O(n²) in worst case
- Isolation Forest: O(n log n)
- MICE: O(n × features × iterations)

### Scalability Recommendations
- Up to 1M rows: All methods feasible
- 1-100M rows: Use ensemble methods, MICE with sampling
- 100M+ rows: Distributed frameworks needed

---

## Code Snippets for Quick Start

### Full Pipeline Example
```python
from data_quality_assessment import DataQualityAssessor
from missing_data_imputation import MICEImputation
from outlier_detection_handling import OutlierDetectionPipeline
from class_imbalance_handling import ImbalanceHandlingPipeline

# 1. Quality Check
assessor = DataQualityAssessor(df)
report = assessor.generate_report()
print(f"Quality Score: {report['overall_quality_score']:.2%}")

# 2. Impute Missing Data
df_clean, _ = MICEImputation.mice_imputation(df)

# 3. Detect Outliers
pipeline = OutlierDetectionPipeline(X, y, contamination=0.05)
outliers = pipeline.ensemble_consensus(min_methods=2)

# 4. Handle Imbalance (if needed)
if y.dtype == 'object' or y.nunique() < len(y) * 0.5:
    imbalance_pipeline = ImbalanceHandlingPipeline(X, y)
    model, threshold, _ = imbalance_pipeline.run_full_pipeline()
```

---

## When Each Framework Excels

### Great Expectations
- Comprehensive validation rules
- Production monitoring
- Team collaboration

### Apache Griffin
- Big data quality at scale
- Distributed processing
- Batch pipelines

### Scikit-learn
- Standard ML preprocessing
- Well-documented
- Easy integration

### These Skills
- Specialized implementations
- Educational reference
- Custom pipeline building

---

## Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| Quality score too low | Unexpected missingness | Use MissingDataAnalyzer |
| Outliers not detected | Wrong threshold | Adjust z-score or LOF cutoff |
| Imbalance handling failing | Single method ineffective | Use ensemble: SMOTE + cost weights |
| Imputation quality poor | Wrong mechanism assumed | Check MCAR/MAR/MNAR detection |
| Memory explosion | Dataset too large | Use sampling or distributed methods |
| Results not reproducible | Random seed not set | Set random_state parameter |

---

## External Resources

**Documentation:**
- Great Expectations: https://greatexpectations.io
- Apache Griffin: https://griffin.apache.org
- Scikit-learn: https://scikit-learn.org
- Imbalanced-learn: https://imbalanced-learn.org

**Papers:**
- Rubin (1987) - Multiple Imputation
- Chandola et al. (2009) - Anomaly Detection Survey
- Chawla et al. (2002) - SMOTE

---

**Last Updated:** April 2026  
**Author:** Shuvam Banerji Seal  
**Skill Status:** Production Ready
