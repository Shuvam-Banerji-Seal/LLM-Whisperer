# Causal Inference: Quick Reference Guide

**Version:** 1.0  
**Last Updated:** April 2026  
**Format:** Concise reference for practitioners

---

## Quick Reference Index

1. [Key Concepts](#key-concepts)
2. [Method Selection Flowchart](#method-selection-flowchart)
3. [Assumption Checklist](#assumption-checklist)
4. [Code Snippets](#code-snippets)
5. [Common Pitfalls](#common-pitfalls)
6. [Resources](#resources)

---

## Key Concepts

### Fundamental Definitions

| Concept | Definition | Notation |
|---------|-----------|----------|
| **Causal Effect** | Effect of intervention, not just association | E[Y\|do(X=x)] |
| **Treatment** | Variable we manipulate/intervene on | T, X, A |
| **Outcome** | Variable we measure/care about | Y |
| **Confounder** | Variable affecting both T and Y | C, W |
| **Propensity Score** | P(T=1\|X), probability of treatment | e(X) |
| **Potential Outcome** | Outcome under specific treatment value | Y(t) |
| **Backdoor Path** | Confounding path not through treatment | ← |
| **Collider** | Variable with arrows from both sides | X → Z ← Y |
| **DAG** | Directed Acyclic Graph representing causality | Graph structure |
| **CATE** | Treatment effect conditional on X | τ(X) |
| **ATE** | Average Treatment Effect | E[Y(1) - Y(0)] |

### Parameter of Interest

```
E[Y|do(T=1)] - E[Y|do(T=0)]

This is causal effect - the difference in outcome 
if we intervene to set T=1 vs T=0
```

### Key Assumptions

```
1. UNCONFOUNDEDNESS: T ⊥ Y(0), Y(1) | X
   No unmeasured confounders given X

2. POSITIVITY (OVERLAP): 0 < P(T=1|X) < 1
   All units have non-zero probability of treatment

3. SUTVA: No interference, stable treatment
   Your treatment doesn't affect others' outcomes
```

---

## Method Selection Flowchart

```
START
  │
  ├─ Do you have RANDOMIZED assignment?
  │  ├─ YES → Use Simple Difference of Means
  │  │         ATE = E[Y|T=1] - E[Y|T=0]
  │  │
  │  └─ NO → Continue
  │
  ├─ Can you specify a DAG?
  │  ├─ YES, has confounders → Continue
  │  ├─ YES, has instrumental variable → Use IV / 2SLS
  │  └─ NO → Use sensitivity analysis bounds
  │
  ├─ Can you measure all confounders?
  │  ├─ YES, few confounders → Propensity Score Methods
  │  ├─ YES, many confounders (>10) → Use Doubly Robust / Causal Forest
  │  └─ NO → Use Instrumental Variables or bound effects
  │
  ├─ Do you care about HETEROGENEOUS effects?
  │  ├─ YES → Causal Forest or Causal Trees
  │  └─ NO → Continue
  │
  ├─ Panel data with policy change?
  │  ├─ YES → Difference-in-Differences
  │  └─ NO → Continue
  │
  └─ Single unit case study?
     ├─ YES → Synthetic Control
     └─ NO → Done - use selected method
```

---

## Assumption Checklist

### Before Analysis

- [ ] **Define causal question** clearly and specifically
- [ ] **Specify DAG** based on domain knowledge
- [ ] **List all assumptions** explicitly
- [ ] **Check data quality**: Missing values, outliers, coding errors
- [ ] **Verify treatment variation**: Enough treated and control units

### Data Preparation

- [ ] **Assess covariate distributions**: Are controls similar to treated?
- [ ] **Check for collinearity**: High correlations among predictors?
- [ ] **Examine treatment assignment**: Is it truly exogenous/random?
- [ ] **Handle missing data**: Document approach (deletion, imputation)
- [ ] **Normalize/scale features**: For ML-based methods

### Overlap/Common Support

- [ ] **Check propensity score range**: Close to (0, 1)?
- [ ] **Visualize distributions**: Treated vs control PS should overlap
- [ ] **Identify regions of poor overlap**: May need to trim
- [ ] **Report sample size with overlap**: How many units remain?

### Model Validation

- [ ] **Covariate balance**: After adjustment, X independent of T?
- [ ] **Propensity model fit**: ROC AUC > 0.7?
- [ ] **Outcome model fit**: R² reasonable for the domain?
- [ ] **Residual diagnostics**: Any patterns suggesting misspecification?
- [ ] **Sensitivity analysis**: How robust to violations?

### Reporting Standards

- [ ] **Point estimate reported** with 95% CI
- [ ] **Balance table** showing covariate distributions
- [ ] **Causal diagram** included
- [ ] **Assumptions stated** and validated where possible
- [ ] **Limitations discussed** clearly

---

## Code Snippets

### 1. Quick Propensity Score Matching

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Fit propensity score
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, T)
propensity_scores = ps_model.predict_proba(X)[:, 1]

# Match nearest neighbor within caliper
from scipy.spatial.distance import cdist
treated_idx = np.where(T == 1)[0]
control_idx = np.where(T == 0)[0]

matches = {}
for t_idx in treated_idx:
    distances = np.abs(propensity_scores[t_idx] - propensity_scores[control_idx])
    if distances.min() < 0.1:  # Caliper
        c_idx = control_idx[np.argmin(distances)]
        matches[t_idx] = c_idx

# Calculate ATE
ate = np.mean([Y[t] - Y[c] for t, c in matches.items()])
```

### 2. Doubly Robust Estimation

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Propensity score
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, T)
e = ps_model.predict_proba(X)[:, 1]

# Outcome models
m0 = RandomForestRegressor(n_estimators=100)
m0.fit(X[T==0], Y[T==0])

m1 = RandomForestRegressor(n_estimators=100)
m1.fit(X[T==1], Y[T==1])

# DR estimation
y0_pred = m0.predict(X)
y1_pred = m1.predict(X)

dr_treat = (T * Y / e) - ((T - e) / e) * y1_pred
dr_ctrl = ((1-T) * Y / (1-e)) + ((T-e) / (1-e)) * y0_pred

ATE = np.mean(dr_treat) - np.mean(dr_ctrl)
```

### 3. Causal Forest with Inference

```python
from causalml.inference.tree_based import CausalForestRegressor
import numpy as np

# Fit
cf = CausalForestRegressor(n_trees=100, max_depth=25, n_jobs=-1)
cf.fit(X, T, Y)

# Predict CATE
cate = cf.predict(X)

# Get variance for 95% CI
var = cf.predict_variance(X)
se = np.sqrt(var)

ci_lower = cate - 1.96 * se
ci_upper = cate + 1.96 * se
```

### 4. Check Overlap

```python
import matplotlib.pyplot as plt

# Propensity scores
ps_treated = propensity_scores[T == 1]
ps_control = propensity_scores[T == 0]

# Visualize
plt.hist(ps_control, bins=30, alpha=0.5, label='Control', density=True)
plt.hist(ps_treated, bins=30, alpha=0.5, label='Treated', density=True)
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Overlap Check')
plt.legend()
plt.show()

# Check ranges
print(f"Control range: [{ps_control.min():.3f}, {ps_control.max():.3f}]")
print(f"Treated range: [{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
```

### 5. Balance Table

```python
import pandas as pd

# Standardized mean difference
def smd(X, T):
    X_ctrl = X[T == 0]
    X_treat = X[T == 1]
    
    mean_diff = X_treat.mean(axis=0) - X_ctrl.mean(axis=0)
    std_pool = np.sqrt((X_ctrl.std(axis=0)**2 + X_treat.std(axis=0)**2) / 2)
    
    return mean_diff / std_pool

balance = pd.DataFrame({
    'Feature': [f'X{i}' for i in range(X.shape[1])],
    'Std. Mean Diff': smd(X, T),
    'Balanced': np.abs(smd(X, T)) < 0.1
})

print(balance)
```

---

## Common Pitfalls

### 🚫 Pitfall 1: Ignoring Poor Overlap

**Problem:** Extrapolating beyond common support region.

**Solution:**
```python
# Trim samples with extreme propensity scores
trim_lower = 0.05
trim_upper = 0.95
keep = (propensity_scores > trim_lower) & (propensity_scores < trim_upper)
X_trim = X[keep]
Y_trim = Y[keep]
T_trim = T[keep]
```

### 🚫 Pitfall 2: Not Checking Covariate Balance

**Problem:** Confounders may still differ after matching.

**Solution:**
```python
# Always check balance
balance = pd.DataFrame({
    'Control Mean': X[T==0].mean(),
    'Treated Mean': X[T==1].mean(),
    'Std Diff': smd(X, T)
})
print(balance)
assert (np.abs(balance['Std Diff']) < 0.1).all(), "Poor balance!"
```

### 🚫 Pitfall 3: Model Overfitting

**Problem:** Complex models may learn spurious patterns.

**Solution:**
```python
# Use cross-validation for propensity score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(), X, T, cv=5)
print(f"CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")
```

### 🚫 Pitfall 4: Forgetting Uncertainty Quantification

**Problem:** Reporting point estimates only.

**Solution:**
```python
# Use bootstrap for confidence intervals
from sklearn.utils import resample

ate_bootstrap = []
for _ in range(1000):
    idx = resample(np.arange(len(X)), n_samples=len(X))
    ate_b = compute_ate(X[idx], T[idx], Y[idx])
    ate_bootstrap.append(ate_b)

ci_lower = np.percentile(ate_bootstrap, 2.5)
ci_upper = np.percentile(ate_bootstrap, 97.5)
print(f"ATE: {np.mean(ate_bootstrap):.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

### 🚫 Pitfall 5: Ignoring Unmeasured Confounding

**Problem:** Hidden variables can bias all estimates.

**Solution:**
```python
# Sensitivity analysis (Rosenbaum bounds)
# How much would confounding need to change treatment odds to flip conclusion?
# Use rosenbaum_bounds() function

# Conservative estimate: Report under different assumptions
ate_optimistic = estimate_ate(method='doubly_robust')  # Assumes no unmeasured confounding
ate_conservative = estimate_ate_with_sensitivity(gamma=1.5)  # Allows some confounding
```

---

## Resources

### Online Courses

- **Brady Neal's Intro to Causal Inference**
  https://www.bradyneal.com/causal-inference-course
  Duration: ~12 weeks, Free, Comprehensive

- **Udacity: Statistics & A/B Testing**
  https://www.udacity.com/
  Format: Video lectures, Projects

- **MIT OpenCourseWare**
  https://ocw.mit.edu/
  Format: Lectures, Problem Sets

### Key Papers (Start Here)

**Beginner:**
1. Pearl (2009) - Causality book, Chapters 1-3
2. Angrist & Pischke (2008) - Mostly Harmless Econometrics
3. Austin (2011) - Propensity score intro

**Intermediate:**
1. Athey & Wager (2019) - Causal forests
2. Chernozhukov et al. (2018) - DML
3. Kunzel et al. (2019) - Meta-learners

**Advanced:**
1. Peters et al. (2017) - Elements of Causal Inference
2. Kennedy (2020) - Optimal DR
3. Recent arXiv papers on your topic

### Software Documentation

- **DoWhy**: https://www.microsoft.com/en-us/research/project/dowhy/
- **CausalML**: https://causalml.readthedocs.io/
- **EconML**: https://econml.azurewebsites.net/
- **Causal Impact**: https://github.com/jamalsenouci/causalimpact

### Datasets

- **IHDP Dataset**: Infant health development program (RCT validation)
- **NSW LaLonde Data**: Job training program evaluation (classic benchmark)
- **ACIC Data Challenge**: 1000 realistic simulated scenarios

---

## Symbol Reference

```
T           Treatment variable
Y           Outcome variable
X           Covariates/confounders
Y(t)        Potential outcome under treatment t
E[Y|do(X)]  Causal expectation (post-intervention)
U           Unmeasured confounder
e(X)        Propensity score P(T=1|X)
τ(x)        CATE, heterogeneous treatment effect
ATE         Average treatment effect
CATE        Conditional ATE
⊥           Independence
→           Causality
←           Confounding
```

---

## Diagnostic Summary

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| Poor overlap | PS range doesn't cover [0,1] | Trim extreme, use more flexible PS model |
| Unbalanced covariates | Std diff > 0.1 after matching | Refine matching, use regression adjustment |
| Bad PS model fit | AUC < 0.7 | Add interactions, polynomial terms, different model |
| High variance estimates | Large SE, wide CI | Get more data, use regularization |
| Suspected unmeasured confounding | Estimates sensitive to omitted variable | Use IV, bounds analysis, sensitivity parameters |
| Heterogeneity | Effects vary by subgroup | Use causal forest, stratify by key variable |
| Multiple time periods | Trends not parallel | Add time trends, use synthetic control |
| Small sample size | Statistical power low | Combine methods, focus on direction |

---

## 30-Second Decision Guide

```
Question: What causal method should I use?

IF random_assignment:
    THEN: t-test (simple difference of means)
    
ELSE IF have_instrumental_variable:
    THEN: Two-Stage Least Squares
    
ELSE IF care_about_heterogeneity:
    THEN: Causal Forest (best modern choice)
    
ELSE IF panel_data_with_policy:
    THEN: Difference-in-Differences
    
ELSE IF single_unit_case_study:
    THEN: Synthetic Control
    
ELSE:
    # Standard observational study
    IF many_covariates (>10):
        THEN: Doubly Robust Learner
    ELSE:
        THEN: Propensity Score + outcome regression
    
    # Always do sensitivity analysis!
    THEN: Check robustness to unobserved confounding
```

---

## Quick Stats Formulas

### Average Treatment Effect (ATE)

```
ATE = E[Y|do(T=1)] - E[Y|do(T=0)]

Under randomization:
ATE = E[Y|T=1] - E[Y|T=0]

Standard error:
SE(ATE) = √[Var(Y|T=1)/n₁ + Var(Y|T=0)/n₀]

95% Confidence Interval:
ATE ± 1.96 × SE(ATE)
```

### Propensity Score Matching

```
Weight = 1/e(X) for treated
Weight = 1/(1-e(X)) for control

ATE = E_weighted[Y|T=1] - E_weighted[Y|T=0]
```

### Doubly Robust

```
ATE = E[(T×Y)/e - (T-e)×m₁/e] - E[((1-T)×Y)/(1-e) + (T-e)×m₀/(1-e)]

where:
m₁ = E[Y|T=1, X]
m₀ = E[Y|T=0, X]
e = P(T=1|X)
```

### Difference-in-Differences

```
Y = α + β₁×Group + β₂×Time + β₃×(Group × Time) + ε

DID Effect = β₃ = (Y₁₁ - Y₁₀) - (Y₀₁ - Y₀₀)
```

---

**Quick Ref Version:** 1.0  
**Last Updated:** April 2026  
**Print-Friendly:** Yes  
**Est. Reading Time:** 15-30 minutes
