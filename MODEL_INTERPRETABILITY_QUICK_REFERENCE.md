# Model Interpretability and Explainability: Quick Reference Guide

## At a Glance Comparison

### Interpretability Methods

| Method | Speed | Accuracy | Complexity | Best For |
|--------|-------|----------|-----------|----------|
| **SHAP** | Medium | Excellent | High | Feature importance, game-theoretic |
| **LIME** | Medium | Good | Low | Model-agnostic, local explanations |
| **Integrated Gradients** | Fast | Excellent | Medium | Neural networks |
| **Grad-CAM** | Very Fast | Good | Low | Vision models |
| **Attention Visualization** | Fast | Medium | Low | Transformers, NLP |
| **Feature Importance** | Very Fast | Good | Low | Tree-based models |
| **Saliency Maps** | Fast | Medium | Low | Gradient-based |
| **TCAV** | Slow | Medium | High | Concept-based |
| **Influence Functions** | Very Slow | Excellent | Very High | Training data analysis |

---

## Installation Quick Start

```bash
# Install all required libraries
pip install shap lime captum torch torchvision transformers numpy pandas matplotlib seaborn scikit-learn

# Verify installation
python -c "import shap, lime, captum, torch; print('All imports successful')"
```

---

## Code Snippets

### SHAP - 30 Seconds

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```

### LIME - 30 Seconds

```python
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=features)
exp = explainer.explain_instance(X_test[0], model.predict_proba)
exp.show_in_notebook()
```

### Grad-CAM - 30 Seconds

```python
from captum.attr import LayerCAM
layer_cam = LayerCAM(model, model.layer4)
attr = layer_cam.attribute(image_tensor, target=class_idx)
visualize(attr)
```

### Integrated Gradients - 30 Seconds

```python
from captum.attr import IntegratedGradients
ig = IntegratedGradients(model)
attr = ig.attribute(input_tensor, target=class_idx)
visualize(attr)
```

### Attention Visualization - 30 Seconds

```python
outputs = model(input_ids, output_attentions=True)
attention = outputs.attentions[-1][0, 0]  # Last layer, first head
plt.imshow(attention.numpy(), cmap='viridis')
```

---

## SHAP Quick Reference

### Different SHAP Types

```python
# Tree-based models (FAST)
explainer = shap.TreeExplainer(model)

# Kernel SHAP (Model-agnostic)
explainer = shap.KernelExplainer(model.predict, X_background)

# Deep SHAP (Neural networks)
explainer = shap.DeepExplainer(model, X_background)

# Linear SHAP (Linear models)
explainer = shap.LinearExplainer(model, X_train)

# Sampling SHAP (Sampling-based)
explainer = shap.SamplingExplainer(model.predict, X_train)
```

### SHAP Visualizations

```python
# Summary plot (bar chart)
shap.summary_plot(shap_values, X)

# Summary plot (scatter)
shap.summary_plot(shap_values, X, plot_type='scatter')

# Force plot (single prediction)
shap.force_plot(explainer.expected_value, shap_values[0], X[0])

# Dependence plot (feature interaction)
shap.dependence_plot('feature', shap_values, X)

# Waterfall plot (contribution breakdown)
shap.waterfall_plot(shap_values[0])

# Mean absolute SHAP values
shap.mean_abs_shap_values(shap_values, X)
```

---

## Grad-CAM Variants

### Quick Comparison

```python
from captum.attr import LayerCAM, GradientShap, Saliency

# Grad-CAM (gradient-weighted feature maps)
LayerCAM(model, target_layer)

# Gradient SHAP (combines gradients and sampling)
GradientShap(model)

# Saliency (input gradients)
Saliency(model)

# DeepLIFT (layer-wise relevance)
DeepLIFT(model)
```

### Code

```python
from captum.attr import LayerCAM, visualization

# Create explainer
layer_cam = LayerCAM(model, model.layer4)

# Get attribution
attribution = layer_cam.attribute(input_tensor, target=class_idx)

# Visualize
visualization.visualize_image_attr_multiple(
    [attribution],
    [input_tensor],
    methods=['original_image', 'heat_map'],
    signs=['all', 'positive']
)
```

---

## Feature Importance Methods

### Tree-Based Models

```python
# MDI (Mean Decrease in Impurity)
importances = model.feature_importances_

# Permutation Importance
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test)

# Drop-Column Importance
# (manually drop each column and measure performance change)
```

### Neural Networks

```python
# Input Gradients
grad = torch.autograd.grad(output[0, target], input, create_graph=True)[0]
importance = grad.abs().mean(dim=0)

# Layer Activation Variance
activations = []
def hook(module, input, output):
    activations.append(output.detach())
model.layer.register_forward_hook(hook)

# Feature Importance via Occlusion
for i in range(num_features):
    X_occluded = X.copy()
    X_occluded[:, i] = 0
    importance[i] = model.score(X_occluded)
```

---

## Attention Visualization Patterns

### BERT/Transformer

```python
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world", return_tensors='pt')
outputs = model(**inputs)

attention = outputs.attentions  # (layers, batch, heads, seq_len, seq_len)

# Visualize single head
import matplotlib.pyplot as plt
plt.imshow(attention[-1][0, 0].detach().numpy(), cmap='viridis')
```

### Vision Transformer

```python
from torchvision.models import vit_b_16

model = vit_b_16(pretrained=True)
outputs = model(image_tensor)  # Need to patch first

# Get attention rollout
attention = outputs.attentions[-1][0, 0]
```

---

## Common Patterns

### Pattern: Batch Explanation Generation

```python
def batch_explain(model, X, method='shap', batch_size=32):
    """Generate explanations for multiple samples"""
    
    if method == 'shap':
        explainer = shap.TreeExplainer(model)
        explanations = explainer.shap_values(X)
    elif method == 'lime':
        explainer = lime.lime_tabular.LimeTabularExplainer(X)
        explanations = [
            explainer.explain_instance(x, model.predict_proba).as_list()
            for x in X
        ]
    
    return explanations
```

### Pattern: Explanation Caching

```python
import joblib

def explain_with_cache(model, X, cache_path='explanations.pkl'):
    """Cache explanations to avoid recomputation"""
    
    try:
        explanations = joblib.load(cache_path)
    except FileNotFoundError:
        explainer = shap.TreeExplainer(model)
        explanations = explainer.shap_values(X)
        joblib.dump(explanations, cache_path)
    
    return explanations
```

### Pattern: Multi-Method Comparison

```python
def compare_explanations(model, X, y_true):
    """Compare multiple explanation methods"""
    
    results = {}
    
    # SHAP
    shap_exp = shap.TreeExplainer(model)
    results['shap'] = shap_exp.shap_values(X)
    
    # Feature Importance
    results['importance'] = model.feature_importances_
    
    # Permutation Importance
    from sklearn.inspection import permutation_importance
    results['permutation'] = permutation_importance(model, X, y_true).importances_mean
    
    return results
```

---

## Evaluation Metrics

### Faithfulness

```python
def faithfulness_score(model, X, explanation, top_k=5):
    """How much does prediction change when removing top-k features?"""
    
    top_features = np.argsort(np.abs(explanation))[-top_k:]
    X_removed = X.copy()
    X_removed[:, top_features] = 0
    
    return np.mean(np.abs(model.predict(X) - model.predict(X_removed)))
```

### Stability

```python
def stability_score(explanation_fn, sample, num_runs=10):
    """How consistent are explanations across multiple runs?"""
    
    explanations = [explanation_fn(sample) for _ in range(num_runs)]
    
    # Pairwise correlations
    correlations = []
    for i in range(num_runs):
        for j in range(i+1, num_runs):
            corr = np.corrcoef(explanations[i], explanations[j])[0, 1]
            correlations.append(corr)
    
    return np.mean(correlations)
```

### Completeness

```python
def completeness_check(shap_values, model_output, baseline):
    """Do SHAP values sum to model output?"""
    
    sum_shap = np.sum(shap_values)
    expected_output = baseline + sum_shap
    
    return np.allclose(expected_output, model_output)
```

---

## Common Issues & Solutions

### Issue: NaN in Explanations

```python
# Solution 1: Replace NaN with 0
explanations = np.nan_to_num(explanations, nan=0.0)

# Solution 2: Handle before computation
X = np.nan_to_num(X, nan=X.mean(axis=0))
```

### Issue: Slow SHAP Computation

```python
# Use Tree SHAP instead of Kernel SHAP
explainer = shap.TreeExplainer(model)  # FAST
# explainer = shap.KernelExplainer(model.predict, X)  # SLOW

# Use sampling for large datasets
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[:1000])  # Sample
```

### Issue: LIME Explanations Not Intuitive

```python
# Increase num_samples for better local approximation
exp = explainer.explain_instance(
    X[0],
    model.predict_proba,
    num_samples=10000  # Increase this
)

# Use different kernels
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    mode='classification',
    kernel_width=0.5  # Adjust for locality
)
```

### Issue: Memory Error with Large Models

```python
# Use batch processing
from torch.utils.data import DataLoader

data_loader = DataLoader(X, batch_size=32)

for batch in data_loader:
    explanations_batch = explain_batch(batch)
    # Process and save to disk
    save_to_disk(explanations_batch)
```

---

## When to Use Each Method

### SHAP
- ✅ Feature importance ranking
- ✅ Model debugging
- ✅ Feature interaction analysis
- ❌ Real-time explanations (slow)
- ❌ Large feature sets (exponential complexity)

### LIME
- ✅ Model-agnostic explanations
- ✅ Fast local explanations
- ✅ Black-box models
- ❌ Inconsistent across samples
- ❌ Requires careful tuning

### Grad-CAM
- ✅ Vision models
- ✅ Real-time explanations
- ✅ Attention visualization
- ❌ Only for neural networks
- ❌ Less faithful than global methods

### Integrated Gradients
- ✅ Neural networks
- ✅ Mathematically sound
- ✅ Good baseline handling
- ❌ Requires backpropagation
- ❌ Slower than simple gradients

### Attention Visualization
- ✅ Transformers/NLP
- ✅ Understanding information flow
- ✅ Model diagnosis
- ❌ May not reflect actual behavior
- ❌ Requires attention mechanism

---

## Performance Tips

### Speed Optimization

```python
# 1. Use Tree SHAP for tree models
explainer = shap.TreeExplainer(model)  # O(depth)

# 2. Sample data for LIME
explainer.explain_instance(X[0], model.predict_proba, num_samples=100)

# 3. Batch GPU computation
model.to('cuda')
X = X.to('cuda')
explanations = batch_compute(X)

# 4. Cache results
import functools
@functools.lru_cache(maxsize=1000)
def cached_explain(x):
    return explainer.explain_instance(x, model.predict_proba)
```

### Memory Optimization

```python
# 1. Process in batches
for batch in chunks(X, batch_size=100):
    process(batch)

# 2. Use sparse representations
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(X)

# 3. Delete unnecessary objects
del explainer
import gc; gc.collect()

# 4. Use memory-mapped arrays
X = np.memmap('data.dat', dtype='float32')
```

---

## Visualization Cheat Sheet

### SHAP Plots

```python
# All in one summary
shap.summary_plot(shap_values, X)

# By class (for binary classification)
shap.summary_plot(shap_values, X, plot_type='bar')

# Feature interaction
shap.dependence_plot('feature', shap_values, X)

# Waterfall for single prediction
shap.waterfall_plot(shap_values[0])

# Force plot
shap.force_plot(explainer.expected_value, shap_values[0], X[0])
```

### Matplotlib Heatmaps

```python
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP values heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(shap_values, cmap='RdBu_r', center=0)
plt.title('SHAP Values Heatmap')

# Attention heatmap
sns.heatmap(attention_matrix, cmap='viridis')
plt.title('Attention Weights')

# Feature importance
sns.barplot(x=importance_values, y=feature_names)
plt.title('Feature Importance')
```

---

## Reference Commands

### Pandas Integration

```python
import pandas as pd

# Create SHAP explanation dataframe
shap_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

# Feature importance ranking
importance = shap_df.abs().mean().sort_values(ascending=False)
```

### Numpy Operations

```python
import numpy as np

# Get top-k important features
top_k_indices = np.argsort(np.abs(shap_values).mean(axis=0))[-k:]

# Normalize explanations
normalized = shap_values / np.linalg.norm(shap_values, axis=1, keepdims=True)

# Compute explanation agreement
agreement = np.corrcoef(shap_values_method1, shap_values_method2)
```

---

## Resources

### Key Papers
1. **SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
2. **LIME**: Ribeiro et al. (2016) - "Why Should I Trust You?"
3. **Integrated Gradients**: Sundararajan et al. (2017)
4. **Grad-CAM**: Selvaraju et al. (2017)
5. **Transformers**: Vaswani et al. (2017) - "Attention is All You Need"

### Documentation
- SHAP: https://shap.readthedocs.io/
- LIME: https://lime-ml.readthedocs.io/
- Captum: https://captum.ai/
- PyTorch: https://pytorch.org/docs/

### Tools
- SHAP: Tree, Kernel, Deep, Linear explainers
- LIME: Tabular, Text, Image explainers
- Captum: 20+ attribution methods
- Alibi: Counterfactual, Prototype explanations

---

## Checklists

### Pre-Explanation Checklist
- [ ] Model is trained and validated
- [ ] Test data is preprocessed
- [ ] Feature names are available
- [ ] Output format is understood
- [ ] Performance baseline is established

### Explanation Generation Checklist
- [ ] Choose appropriate explainer
- [ ] Set hyperparameters
- [ ] Validate on subset first
- [ ] Monitor memory usage
- [ ] Cache results

### Post-Explanation Checklist
- [ ] Check for NaN/Inf values
- [ ] Verify explanation shapes
- [ ] Compare multiple methods
- [ ] Validate faithfulness
- [ ] Document findings

---

## Quick Troubleshooting

```
Problem: SHAP too slow
→ Use Tree SHAP or sample data

Problem: LIME inconsistent
→ Increase num_samples or adjust kernel_width

Problem: Attention not interpretable
→ Check attention_rollout for information flow

Problem: Memory error
→ Use batch processing or reduce data size

Problem: NaN explanations
→ Check for invalid input values or normalize

Problem: Grad-CAM blurry
→ Try different layers or use Grad-CAM++

Problem: Can't visualize
→ Check tensor shapes and convert to numpy
```

---

*Quick Reference Version: 1.0*
*Last Updated: 2024*
*Status: Ready for Production Use*
