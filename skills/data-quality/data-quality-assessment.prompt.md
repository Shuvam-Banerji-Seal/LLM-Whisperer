# Data Quality Assessment Skill

**Author:** Shuvam Banerji Seal  
**Version:** 1.0  
**Last Updated:** April 2026

## Overview

Data quality assessment is the systematic evaluation of data characteristics to ensure they meet organizational standards and business requirements. This skill covers the fundamental metrics, dimensions, and frameworks for comprehensive data quality evaluation in machine learning pipelines.

### Definition

According to ISO/IEC 8000:2018, **data quality** is the degree to which a set of inherent characteristics of data fulfills stated and implied needs when used under specified conditions. Quality assessment involves measuring:

1. **Completeness** - Presence of required data
2. **Consistency** - Uniformity across data sources
3. **Accuracy** - Correctness against source of truth
4. **Validity** - Conformance to defined formats
5. **Timeliness** - Currency and availability when needed
6. **Uniqueness** - Absence of duplicates

## Mathematical Formulations

### Completeness Score

$$C = 1 - \frac{\text{Number of missing values}}{\text{Total number of values}}$$

**Range:** [0, 1] where 1 = complete dataset

```
Example: Dataset with 1000 values and 50 nulls
C = 1 - (50/1000) = 0.95 (95% complete)
```

### Consistency Score

For categorical data with n categories where each category has expected frequency p_i:

$$CS = 1 - \frac{1}{n}\sum_{i=1}^{n}|observed_i - expected_i| / expected_i$$

For numerical data using coefficient of variation:

$$CV = \frac{\sigma}{\mu}$$

Where σ is standard deviation and μ is mean.

### Accuracy Metrics

**Precision Rate (for classification):**
$$P = \frac{TP}{TP + FP}$$

**Recall Rate:**
$$R = \frac{TP}{TP + FN}$$

**F1-Score (balanced accuracy):**
$$F1 = 2 \cdot \frac{P \times R}{P + R}$$

### Validity Assessment

**Format Conformance Rate:**
$$V = \frac{\text{Records matching schema}}{\text{Total records}}$$

### Overall Data Quality Score

$$DQ = w_1 \cdot C + w_2 \cdot Cons + w_3 \cdot Acc + w_4 \cdot Val + w_5 \cdot Time$$

Where w_i are business-defined weights summing to 1.

## Implementation

### Python Code Examples

#### 1. Basic Data Quality Assessment

```python
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class DataQualityAssessor:
    """Comprehensive data quality assessment framework"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.metrics = {}
    
    def calculate_completeness(self) -> Dict[str, float]:
        """Calculate completeness for each column"""
        completeness = {}
        for col in self.df.columns:
            missing = self.df[col].isna().sum()
            total = len(self.df)
            completeness[col] = 1 - (missing / total)
        
        self.metrics['completeness'] = completeness
        return completeness
    
    def calculate_consistency(self) -> Dict[str, float]:
        """Assess consistency using statistical measures"""
        consistency = {}
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            try:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                cv = std_val / mean_val if mean_val != 0 else np.inf
                # CV < 0.5 indicates good consistency
                consistency[col] = max(0, 1 - min(cv / 0.5, 1))
            except:
                consistency[col] = 0
        
        self.metrics['consistency'] = consistency
        return consistency
    
    def calculate_uniqueness(self) -> Dict[str, float]:
        """Check for duplicate records and values"""
        total_rows = len(self.df)
        duplicate_rows = self.df.duplicated().sum()
        
        uniqueness = {
            'row_uniqueness': 1 - (duplicate_rows / total_rows),
            'duplicate_count': int(duplicate_rows)
        }
        
        # Column-level uniqueness
        col_uniqueness = {}
        for col in self.df.columns:
            unique_vals = self.df[col].nunique()
            col_uniqueness[col] = unique_vals / len(self.df)
        
        uniqueness['column_uniqueness'] = col_uniqueness
        self.metrics['uniqueness'] = uniqueness
        return uniqueness
    
    def assess_validity(self, schema: Dict[str, str]) -> Dict[str, float]:
        """Validate data against schema"""
        validity_scores = {}
        
        for col, dtype in schema.items():
            if col not in self.df.columns:
                validity_scores[col] = 0.0
                continue
            
            try:
                if dtype == 'numeric':
                    valid = pd.to_numeric(self.df[col], errors='coerce').notna().sum()
                elif dtype == 'integer':
                    valid = self.df[col].apply(
                        lambda x: isinstance(x, (int, np.integer)) or 
                        (isinstance(x, float) and x.is_integer())
                    ).sum()
                elif dtype == 'categorical':
                    # Check if values are within expected categories
                    valid = self.df[col].notna().sum()
                else:
                    valid = self.df[col].notna().sum()
                
                validity_scores[col] = valid / len(self.df)
            except:
                validity_scores[col] = 0
        
        self.metrics['validity'] = validity_scores
        return validity_scores
    
    def detect_anomalies(self) -> Dict[str, list]:
        """Detect common data quality anomalies"""
        anomalies = {
            'missing_values': [],
            'inconsistent_types': [],
            'outliers': [],
            'duplicates': self.df.duplicated(keep=False).sum()
        }
        
        for col in self.df.columns:
            # Missing values
            missing_pct = (self.df[col].isna().sum() / len(self.df)) * 100
            if missing_pct > 5:  # Flag if >5% missing
                anomalies['missing_values'].append({
                    'column': col,
                    'missing_percentage': missing_pct
                })
            
            # Type inconsistencies
            if col in self.df.select_dtypes(include=[object]).columns:
                inferred_type = pd.api.types.infer_dtype(self.df[col], skipna=True)
                anomalies['inconsistent_types'].append({
                    'column': col,
                    'inferred_type': inferred_type
                })
        
        self.metrics['anomalies'] = anomalies
        return anomalies
    
    def generate_report(self, weights: Dict[str, float] = None) -> Dict:
        """Generate comprehensive quality report"""
        if weights is None:
            weights = {
                'completeness': 0.3,
                'consistency': 0.25,
                'uniqueness': 0.15,
                'validity': 0.2,
                'timeliness': 0.1
            }
        
        # Calculate all metrics
        completeness = self.calculate_completeness()
        consistency = self.calculate_consistency()
        uniqueness = self.calculate_uniqueness()
        validity = self.assess_validity({
            col: 'numeric' if self.df[col].dtype in [np.number]
            else 'categorical' 
            for col in self.df.columns
        })
        
        # Overall quality score
        overall_score = (
            weights['completeness'] * np.mean(list(completeness.values())) +
            weights['consistency'] * np.mean(list(consistency.values())) +
            weights['uniqueness'] * uniqueness['row_uniqueness'] +
            weights['validity'] * np.mean(list(validity.values()))
        )
        
        return {
            'overall_quality_score': overall_score,
            'completeness_scores': completeness,
            'consistency_scores': consistency,
            'uniqueness_scores': uniqueness,
            'validity_scores': validity,
            'anomalies': self.detect_anomalies(),
            'timestamp': pd.Timestamp.now()
        }

# Usage Example
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', None, 'David', 'Eve'],
    'salary': [50000, 60000, 55000, np.nan, 65000],
    'date_joined': ['2020-01-15', '2019-03-20', '2021-06-10', '2020-11-05', '2021-09-30']
})

assessor = DataQualityAssessor(df)
report = assessor.generate_report()
print(f"Data Quality Score: {report['overall_quality_score']:.2%}")
```

#### 2. Column-Level Quality Assessment

```python
class ColumnQualityAnalyzer:
    """Detailed column-level quality analysis"""
    
    @staticmethod
    def analyze_column(series: pd.Series, column_name: str) -> Dict:
        """Comprehensive analysis for a single column"""
        analysis = {
            'column_name': column_name,
            'data_type': str(series.dtype),
            'total_values': len(series),
            'null_count': series.isna().sum(),
            'null_percentage': (series.isna().sum() / len(series)) * 100,
            'unique_values': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100,
            'duplicate_count': len(series) - series.nunique()
        }
        
        if series.dtype in [np.float64, np.int64]:
            analysis.update({
                'mean': series.mean(),
                'median': series.median(),
                'std_dev': series.std(),
                'min': series.min(),
                'max': series.max(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75),
                'iqr': series.quantile(0.75) - series.quantile(0.25)
            })
        
        elif series.dtype == object:
            analysis.update({
                'most_common': series.mode()[0] if not series.mode().empty else None,
                'least_common': series.value_counts().iloc[-1] if len(series.value_counts()) > 0 else None
            })
        
        return analysis

# Usage
for col in df.columns:
    analysis = ColumnQualityAnalyzer.analyze_column(df[col], col)
    print(f"\n{col}:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
```

#### 3. Data Quality Monitoring Dashboard

```python
from datetime import datetime
import json

class DataQualityMonitor:
    """Track quality metrics over time"""
    
    def __init__(self):
        self.history = []
    
    def log_assessment(self, df: pd.DataFrame, label: str = None):
        """Log quality metrics with timestamp"""
        assessor = DataQualityAssessor(df)
        report = assessor.generate_report()
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'label': label or f"Assessment_{len(self.history)+1}",
            'quality_score': report['overall_quality_score'],
            'row_count': len(df),
            'column_count': len(df.columns),
            'completeness_avg': np.mean(list(report['completeness_scores'].values())),
            'consistency_avg': np.mean(list(report['consistency_scores'].values()))
        }
        
        self.history.append(entry)
        return entry
    
    def get_trend(self):
        """Analyze quality trend over time"""
        if len(self.history) < 2:
            return None
        
        scores = [h['quality_score'] for h in self.history]
        trend = 'improving' if scores[-1] > scores[0] else 'declining'
        change = ((scores[-1] - scores[0]) / scores[0]) * 100
        
        return {
            'trend': trend,
            'percent_change': change,
            'assessments': len(self.history)
        }

# Usage
monitor = DataQualityMonitor()
monitor.log_assessment(df, "Initial Load")
# ... after some data processing ...
print(monitor.get_trend())
```

## Authoritative Sources

1. **ISO/IEC 8000-1:2018** - Data Quality. Framework and Overview
   - International standard for data quality dimensions
   - Defines completeness, consistency, timeliness, accuracy, validity

2. **Great Expectations** - https://docs.greatexpectations.io/
   - Open-source framework for data quality
   - Provides expectations, validations, and profiling

3. **Apache Griffin** - https://griffin.apache.org/
   - Data quality service for big data
   - Implements DQ metrics: accuracy, completeness, distribution

4. **Wang et al. (1996)** - "A Product Perspective on Total Data Quality Management"
   - Foundational paper on data quality dimensions
   - Introduces quality assessment frameworks

5. **Batini et al. (2009)** - "Methodologies for Data Quality Assessment and Improvement"
   - Comprehensive review of quality assessment techniques
   - Practical quality metrics and indicators

## Practical Checklist

- [ ] Define quality dimensions relevant to business requirements
- [ ] Establish baseline quality metrics for current datasets
- [ ] Set quality thresholds for each dimension
- [ ] Implement automated quality checks in data pipelines
- [ ] Create quality monitoring dashboards
- [ ] Document data quality issues and root causes
- [ ] Establish SLAs for data quality metrics
- [ ] Train teams on quality assessment practices
- [ ] Review and update quality standards quarterly

## Edge Cases & Considerations

### 1. Handling Missing Data Patterns
- **MCAR** (Missing Completely At Random): Can safely ignore
- **MAR** (Missing At Random): May need imputation
- **MNAR** (Missing Not At Random): Investigate root cause

### 2. Type Mismatches
- Numeric columns with string values (e.g., "N/A")
- Date formats varying within same column
- Categorical values with inconsistent casing

### 3. Temporal Quality Issues
- Outdated data in operational systems
- Delayed data arrivals
- Time-zone inconsistencies

### 4. Distributed Data Quality
- Different quality standards across data sources
- Schema evolution and versioning
- Cross-system consistency checks

## Performance Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| Completeness | > 95% | 80-95% | < 80% |
| Uniqueness | > 99% | 95-99% | < 95% |
| Consistency | CV < 0.3 | CV 0.3-0.5 | CV > 0.5 |
| Validity | > 98% | 90-98% | < 90% |

## Integration with ML Pipelines

Data quality assessment should be integrated at:

1. **Data Ingestion** - Validate incoming data
2. **Feature Engineering** - Check transformed features
3. **Training** - Ensure training data quality
4. **Monitoring** - Track production data quality

```python
def quality_gated_pipeline(df, min_quality_score=0.85):
    """Only proceed with pipeline if quality meets threshold"""
    assessor = DataQualityAssessor(df)
    report = assessor.generate_report()
    
    if report['overall_quality_score'] < min_quality_score:
        raise ValueError(
            f"Data quality score {report['overall_quality_score']:.2%} "
            f"below threshold {min_quality_score:.2%}"
        )
    
    return report
```

## Conclusion

Systematic data quality assessment is foundational for ML success. By implementing standardized metrics and monitoring procedures, organizations can ensure reliable, trustworthy datasets that drive better model performance and business outcomes.

---

**Last Reviewed:** April 2026  
**Skill Status:** Production Ready
