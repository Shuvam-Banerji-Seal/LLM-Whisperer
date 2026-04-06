# LLM-Whisperer Data Engineering Skills - Quick Start Guide

**Generated:** April 6, 2026  
**Total Skills:** 12 | **Status:** Production Ready ✅

---

## 🎯 Quick Navigation

### Data Quality Skills (4)
1. **data-quality-assessment** - Completeness, accuracy, consistency, timeliness, validity
2. **outlier-detection-handling** - Statistical, LOF, Isolation Forest methods
3. **missing-data-imputation** - KNN, MICE, multiple imputation strategies
4. **class-imbalance-handling** - SMOTE, cost-sensitive learning, sampling

### Data Preprocessing Skills (4)
5. **text-preprocessing-nlp** - Tokenization, lemmatization, cleaning, NER, POS
6. **numerical-data-scaling** - StandardScaler, MinMaxScaler, RobustScaler
7. **categorical-encoding** - One-hot, label, target, embedding methods
8. **data-augmentation-techniques** - Image, text, numerical, tabular augmentation

### Dataset Engineering Skills (4)
9. **dataset-construction-curation** - Collection, cleaning, filtering, balancing
10. **annotation-and-labeling** - Annotation strategies, quality control, active learning
11. **synthetic-data-generation** - GMM, GANs, VAEs, differential privacy
12. **dataset-versioning-management** - Snapshots, deltas, lineage, catalogs, DVC

---

## 📂 File Locations

```
/skills/
├── data-quality/
│   ├── data-quality-assessment.prompt.md
│   ├── outlier-detection-handling.prompt.md
│   ├── missing-data-imputation.prompt.md
│   └── class-imbalance-handling.prompt.md
├── data-preprocessing/
│   ├── text-preprocessing-nlp.prompt.md
│   ├── numerical-data-scaling.prompt.md
│   ├── categorical-encoding.prompt.md
│   └── data-augmentation-techniques.prompt.md
└── datasets/
    ├── dataset-construction-curation.prompt.md
    ├── annotation-and-labeling.prompt.md
    ├── synthetic-data-generation.prompt.md
    └── dataset-versioning-management.prompt.md
```

---

## 🚀 Getting Started

### For Data Quality Issues
**Start with:** `data-quality-assessment.prompt.md`
- Define quality dimensions
- Measure completeness, accuracy, consistency
- Set up monitoring and alerting

**Then use:**
- `outlier-detection-handling.prompt.md` → for anomalies
- `missing-data-imputation.prompt.md` → for missing values
- `class-imbalance-handling.prompt.md` → for imbalanced classes

### For Data Preprocessing
**Start with:** Identify your data type (text, numerical, categorical)

**Text Data:**
1. `text-preprocessing-nlp.prompt.md` - Clean and tokenize
2. `data-augmentation-techniques.prompt.md` - Augment if needed

**Numerical Data:**
1. `numerical-data-scaling.prompt.md` - Scale features
2. `data-augmentation-techniques.prompt.md` - Augment if needed

**Categorical Data:**
1. `categorical-encoding.prompt.md` - Encode categories
2. `data-augmentation-techniques.prompt.md` - Augment if needed

### For Dataset Management
**Start with:** `dataset-construction-curation.prompt.md`
1. Collect data from sources
2. Clean and filter
3. Balance classes if needed

**Then use:**
- `annotation-and-labeling.prompt.md` → if manual labeling needed
- `synthetic-data-generation.prompt.md` → if limited data
- `dataset-versioning-management.prompt.md` → for versioning and governance

---

## 💡 Quick Decision Trees

### Outlier Detection Method
```
├─ Statistical? (small data, known distribution)
│  └─ Z-score or IQR
├─ Density-based? (local context matters)
│  └─ Local Outlier Factor (LOF)
└─ Isolation? (large data, unknown distribution)
   └─ Isolation Forest
```

### Missing Data Imputation
```
├─ Simple & fast?
│  └─ Mean/median/mode
├─ KNN available?
│  └─ KNN imputation
├─ Multiple variables related?
│  └─ MICE (Multiple Imputation by Chained Equations)
└─ Preserve uncertainty?
   └─ Multiple imputation with Bayesian
```

### Feature Scaling Method
```
├─ Normal distribution?
│  └─ StandardScaler
├─ Bounded range needed?
│  └─ MinMaxScaler
├─ Has outliers?
│  └─ RobustScaler
└─ Right-skewed?
   └─ Log transformation
```

### Categorical Encoding
```
├─ Ordinal variable?
│  └─ Label encoding (preserve order)
├─ Low cardinality (<10)?
│  └─ One-hot encoding
├─ High cardinality (>50)?
│  └─ Target encoding or embedding
└─ Need interpretability?
   └─ Target encoding or ordinal
```

### Class Imbalance Solution
```
├─ Small dataset?
│  └─ SMOTE or ADASYN
├─ Large dataset?
│  └─ Undersampling + oversampling
├─ Need probabilities?
│  └─ Cost-sensitive learning
└─ Tree models?
   └─ Adjust class_weight parameter
```

---

## 📊 Key Metrics by Skill

### Data Quality Assessment
- **Metrics:** Completeness %, Accuracy rate, Consistency score, Timeliness
- **Tools:** Great Expectations, pandas, custom rules

### Outlier Detection
- **Metrics:** Precision, Recall, False Positive Rate
- **Methods:** IQR, Z-score, LOF, Isolation Forest

### Missing Data
- **Metrics:** RMSE, MAE, entropy loss
- **Methods:** Mean, KNN, MICE, Regression

### Class Imbalance
- **Metrics:** F1-score, Precision, Recall, AUC-ROC
- **Methods:** SMOTE, ADASYN, Cost-sensitive

### Text Preprocessing
- **Metrics:** Vocabulary size, token count, compression ratio
- **Techniques:** Tokenization, Lemmatization, NER, POS

### Feature Scaling
- **Metrics:** Mean, std, min, max
- **Methods:** StandardScaler, MinMaxScaler, RobustScaler

### Categorical Encoding
- **Metrics:** Cardinality, entropy, mutual information
- **Methods:** One-hot, Label, Target, Embedding

### Data Augmentation
- **Metrics:** Data diversity, generalization improvement
- **Methods:** Mixup, CutMix, EDA, SMOTE

### Dataset Quality
- **Metrics:** Completeness, consistency, accuracy
- **Process:** Collection → Cleaning → Filtering → Balancing

### Annotation
- **Metrics:** Cohen's Kappa (>0.61), Fleiss' Kappa
- **Process:** Guidelines → Annotation → QC → Aggregation

### Synthetic Data
- **Metrics:** Privacy score, Utility score, Fidelity
- **Methods:** GMM, GAN, VAE, DP-Synthesis

### Dataset Versioning
- **Metrics:** Change rate, lineage completeness
- **Methods:** Snapshots, deltas, hashing, catalogs

---

## 🔗 Integration Examples

### Complete ML Pipeline
```
1. Data Collection → dataset-construction-curation
2. Data Quality Check → data-quality-assessment
3. Outlier Handling → outlier-detection-handling
4. Missing Data → missing-data-imputation
5. Text Preprocessing (if text) → text-preprocessing-nlp
6. Scaling (if numerical) → numerical-data-scaling
7. Encoding (if categorical) → categorical-encoding
8. Augmentation (if limited data) → data-augmentation-techniques
9. Class Balancing (if imbalanced) → class-imbalance-handling
10. Version & Document → dataset-versioning-management
11. Annotation (if needed) → annotation-and-labeling
```

### Data Quality Pipeline
```
1. data-quality-assessment (measure all dimensions)
2. outlier-detection-handling (find anomalies)
3. missing-data-imputation (handle missing)
4. class-imbalance-handling (balance classes)
5. dataset-versioning-management (track changes)
```

### Synthetic Data Pipeline
```
1. dataset-construction-curation (understand data)
2. synthetic-data-generation (create synthetic)
3. data-quality-assessment (validate synthetic)
4. data-augmentation-techniques (enhance if needed)
5. dataset-versioning-management (version and track)
```

---

## 📚 Learning Path by Role

### Data Engineers
1. dataset-construction-curation
2. dataset-versioning-management
3. data-quality-assessment
4. outlier-detection-handling
5. annotation-and-labeling

### Machine Learning Engineers
1. data-quality-assessment
2. missing-data-imputation
3. numerical-data-scaling
4. categorical-encoding
5. data-augmentation-techniques
6. class-imbalance-handling

### NLP Specialists
1. text-preprocessing-nlp
2. data-augmentation-techniques
3. class-imbalance-handling
4. annotation-and-labeling
5. dataset-versioning-management

### Data Scientists
1. All 12 skills for comprehensive understanding
2. Focus on evaluation and metrics
3. Integrate with domain knowledge

---

## ⚡ Common Problems & Solutions

### Problem: Low Model Accuracy
**Check:**
1. data-quality-assessment (quality issues?)
2. missing-data-imputation (missing values?)
3. categorical-encoding (encoding correct?)
4. class-imbalance-handling (imbalanced?)

### Problem: Model Overfitting
**Check:**
1. data-augmentation-techniques (augment data?)
2. numerical-data-scaling (scale features?)
3. outlier-detection-handling (outliers causing?)

### Problem: Slow Annotation
**Check:**
1. annotation-and-labeling (use active learning?)
2. synthetic-data-generation (generate synthetic?)
3. data-augmentation-techniques (augment?)

### Problem: Data Reproducibility
**Check:**
1. dataset-versioning-management (version tracked?)
2. dataset-construction-curation (documented?)
3. data-quality-assessment (quality baseline?)

---

## 🎓 Example Code Snippets

### Quick Start: Data Quality Check
```python
from skills.data_quality import DataQualityFramework

df = pd.read_csv('data.csv')
framework = DataQualityFramework(df)
report = framework.assess_all_dimensions()
score = framework.get_quality_score()
print(f"Quality Score: {score:.1f}/100")
```

### Quick Start: Missing Data
```python
from skills.data_quality import CompletenessAssessment

assessor = CompletenessAssessment()
completeness = assessor.calculate_completeness(df)
critical = assessor.identify_critical_missing_patterns(df, threshold=0.1)
```

### Quick Start: Feature Scaling
```python
from skills.preprocessing import StandardizationScaler

scaler = StandardizationScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Quick Start: Text Preprocessing
```python
from skills.preprocessing import TextCleaner, Tokenizer

cleaner = TextCleaner()
text = cleaner.clean_pipeline(raw_text)
tokens = Tokenizer.nltk_word_tokenization(text)
```

---

## 📖 Reading Tips

1. **Start with Overview:** Every skill starts with importance and context
2. **Theory First:** Mathematical formulations are early in each skill
3. **Code Examples:** Look for "Example Usage" sections
4. **Checklists Last:** Quality checklists at end are for implementation
5. **References:** Check authoritative sources for deep dives

---

## 🔗 Useful Resources

- **Scikit-learn:** https://scikit-learn.org/
- **Pandas:** https://pandas.pydata.org/
- **DVC:** https://dvc.org/
- **Great Expectations:** https://greatexpectations.io/
- **Hugging Face:** https://huggingface.co/
- **NLTK:** https://www.nltk.org/
- **spaCy:** https://spacy.io/

---

## ✨ Best Practices Summary

✅ Always fit preprocessing on training data only  
✅ Use stratified sampling for class imbalance  
✅ Document all data transformations  
✅ Track data versions with hashes  
✅ Measure inter-annotator agreement  
✅ Test on held-out validation data  
✅ Monitor data quality in production  
✅ Maintain audit trails  

---

**Last Updated:** April 6, 2026  
**All Skills:** Production Ready  
**Status:** ✅ Complete
