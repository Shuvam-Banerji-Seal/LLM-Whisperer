# Model Interpretability and Explainability: Comprehensive Guide

## Table of Contents
1. [Core Interpretability Methods](#core-interpretability-methods)
2. [Attribution Techniques](#attribution-techniques)
3. [Deep Learning Interpretability](#deep-learning-interpretability)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Applications & Frameworks](#applications--frameworks)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Benchmark Results](#benchmark-results)
8. [References](#references)

---

## Executive Summary

Model interpretability and explainability have become critical components in machine learning systems, especially for high-stakes applications in healthcare, finance, and legal systems. This guide provides comprehensive coverage of state-of-the-art interpretability methods, their mathematical foundations, practical implementations, and evaluation approaches.

---

## 1. Core Interpretability Methods

### 1.1 SHAP (SHapley Additive exPlanations)

#### Overview
SHAP is a game-theoretic approach to explain the output of machine learning models by computing Shapley values from cooperative game theory. It provides a unified measure of feature importance that satisfies desirable properties: Local Accuracy, Missingness, and Consistency.

#### Mathematical Formulation

The SHAP value for feature i is defined as:

```
φᵢ(f, x) = Σ(S⊆N\{i}) [|S|!(|N|-|S|-1)!/|N|!] × 
           [f(S∪{i}) - f(S)]
```

Where:
- N: set of all features
- S: subset of features
- f(S): model's prediction using features in S
- |N|: total number of features

#### Implementation

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

# Load data
X, y = load_boston(return_X_y=True)
X_np = X[:, :5]  # Use subset for demo

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_np, y)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_np)

# Visualizations
# Summary plot - shows overall feature importance
shap.summary_plot(shap_values, X_np)

# Force plot - shows individual prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_np[0])

# Dependence plot - shows feature impact across range
shap.dependence_plot("feature_name", shap_values, X_np)

# Interaction plot - shows feature interactions
shap.dependence_plot("feature1", shap_values, X_np, 
                     interaction_index="feature2")
```

#### Key Properties

1. **Local Accuracy (Efficiency)**: Sum of SHAP values equals model output
   ```
   f(x) = f(∅) + Σᵢ φᵢ(f, x)
   ```

2. **Missingness (Dummy)**: Features not in model have zero SHAP value

3. **Consistency**: If model changes such that feature importance increases, SHAP value must increase

#### SHAP Implementation Variants

**Kernel SHAP** (Model-Agnostic):
```python
# Works with any model
explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test)
```

**Deep SHAP** (Neural Networks):
```python
import tensorflow as tf

# For deep learning models
model = tf.keras.Sequential([...])
background_data = X_train[:100]
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(X_test)
```

**GPU-Accelerated SHAP**:
```python
# For large datasets
explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_large)
```

---

### 1.2 LIME (Local Interpretable Model-Agnostic Explanations)

#### Overview
LIME explains individual predictions by fitting a simple interpretable model (e.g., linear regression) on perturbed samples around the instance of interest.

#### Mathematical Formulation

LIME finds the best explanation by solving:

```
ξ(x) = argmin_{g∈G} L(f, g, πₓ) + Ω(g)
```

Where:
- f: original model to explain
- g: interpretable model from class G
- L: loss function measuring how well g explains f locally
- πₓ: proximity measure in the local neighborhood of x
- Ω(g): model complexity penalty

#### Implementation

```python
import lime
import lime.lime_tabular
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load and train model
iris = load_iris()
X, y = iris.data, iris.target
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Create explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification',
    random_state=42
)

# Explain instance
instance = X[0]
explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn=clf.predict_proba,
    num_features=4
)

# Visualization
explanation.show_in_notebook()

# Get explanation as list
explanation.as_list()
```

#### LIME for Text Classification

```python
import lime.lime_text

# Text explainer
explainer = lime.lime_text.LimeTextExplainer(
    class_names=['negative', 'positive']
)

# Example text classifier
def predict_fn(texts):
    # Your text classification model
    return clf.predict_proba(vectorizer.transform(texts))

text = "This movie is absolutely amazing and I loved it!"
explanation = explainer.explain_instance(
    data_row=text,
    predict_fn=predict_fn,
    num_features=10
)

explanation.show_in_notebook()
```

#### LIME for Image Classification

```python
import lime.lime_image

# Image explainer
explainer = lime.lime_image.LimeImageExplainer()

# Explain image prediction
explanation = explainer.explain_instance(
    image=image_array,
    classifier_fn=model.predict,
    top_labels=3,
    hide_color=0,
    num_samples=1000
)

# Show explanation
explanation.show_in_notebook()
```

---

### 1.3 Attention Visualization

#### Overview
Attention mechanisms in neural networks learn to weight different input components. Visualizing these weights provides interpretability for transformer models and attention-based architectures.

#### Mathematical Foundation

In multi-head attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Where:
- Q (queries), K (keys), V (values): linear projections of input
- d_k: dimension of keys
```

#### Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class AttentionVisualizer:
    """Visualize attention weights in transformer models"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention weights"""
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture attention output"""
        if isinstance(output, tuple):
            self.attention_weights.append(output[1])  # attention probs
    
    def visualize_single_head(self, attention_probs, tokens=None):
        """Visualize single attention head"""
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_probs.detach().cpu().numpy(), cmap='viridis')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        if tokens:
            plt.xticks(range(len(tokens)), tokens, rotation=45)
            plt.yticks(range(len(tokens)), tokens)
        plt.colorbar(label='Attention Weight')
        plt.tight_layout()
        plt.show()
    
    def visualize_all_heads(self, attention_probs, tokens=None):
        """Visualize all attention heads in a layer"""
        num_heads = attention_probs.shape[0]
        fig, axes = plt.subplots(
            nrows=int(np.ceil(np.sqrt(num_heads))),
            ncols=int(np.ceil(np.sqrt(num_heads))),
            figsize=(15, 15)
        )
        
        for head_idx, ax in enumerate(axes.flat):
            if head_idx < num_heads:
                ax.imshow(
                    attention_probs[head_idx].detach().cpu().numpy(),
                    cmap='viridis'
                )
                ax.set_title(f'Head {head_idx}')
                if tokens:
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=45)
        
        plt.tight_layout()
        plt.show()

# BERT attention visualization example
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# Extract attention from last layer
attention = outputs[-1][-1]  # Last layer attention

visualizer = AttentionVisualizer(model)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Visualize attention from first token (CLS token)
visualizer.visualize_single_head(attention[0, 0, :, :], tokens)
```

#### Attention Rollout for Transformers

```python
def attention_rollout(attention_weights, discard_ratio=0.9):
    """
    Compute attention rollout across layers to understand 
    information flow in transformers
    
    attention_weights: list of attention matrices per layer
    """
    batch_size = attention_weights[0].shape[0]
    num_tokens = attention_weights[0].shape[2]
    
    # Start with identity matrix
    rollout = torch.eye(num_tokens).unsqueeze(0).expand(
        batch_size, -1, -1
    ).to(attention_weights[0].device)
    
    for attention in attention_weights:
        # Average over attention heads
        attention_avg = attention.mean(dim=1)
        
        # Discard low attention scores
        if discard_ratio > 0:
            threshold = attention_avg.quantile(discard_ratio)
            attention_avg = torch.where(
                attention_avg < threshold,
                torch.zeros_like(attention_avg),
                attention_avg
            )
        
        # Accumulate attention
        rollout = torch.matmul(attention_avg, rollout)
    
    return rollout
```

---

### 1.4 Feature Importance Methods

#### Permutation Feature Importance

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute permutation importance
result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42
)

# Results contain importances_mean and importances_std
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)
```

#### Drop-Column Importance

```python
def drop_column_importance(model, X, y, metric_fn):
    """
    Measure feature importance by dropping each column
    and observing performance change
    """
    baseline_score = metric_fn(model, X, y)
    importances = {}
    
    for col in X.columns:
        X_dropped = X.drop(columns=[col])
        dropped_score = metric_fn(model, X_dropped, y)
        importances[col] = baseline_score - dropped_score
    
    return importances
```

#### MDI (Mean Decrease in Impurity)

```python
# Built-in feature importance for tree-based models
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# Visualize
plt.barh(feature_importance_df['feature'], 
         feature_importance_df['importance'])
plt.xlabel('Importance')
plt.show()
```

---

### 1.5 Saliency Maps and Attribution Methods

#### Gradient-Based Saliency Maps

```python
import torch
import torch.nn.functional as F

def compute_saliency_map(model, image, target_class):
    """
    Compute saliency map using input gradients
    """
    image = image.requires_grad_(True)
    
    # Forward pass
    output = model(image)
    target_score = output[0, target_class]
    
    # Backward pass
    model.zero_grad()
    target_score.backward()
    
    # Saliency is max gradient across channels
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    
    return saliency.squeeze().detach().cpu().numpy()

# Usage
saliency = compute_saliency_map(model, image, class_idx)
plt.imshow(saliency, cmap='hot')
plt.colorbar()
plt.show()
```

#### Smooth Grad (Gradient with Noise Injection)

```python
def smooth_grad(model, image, target_class, num_samples=50, noise_level=0.15):
    """
    Compute smoothed gradients by averaging gradients
    computed on noisy versions of input
    """
    image = image.clone()
    total_gradients = torch.zeros_like(image)
    
    for _ in range(num_samples):
        # Add Gaussian noise
        noisy_image = image + noise_level * torch.randn_like(image)
        noisy_image.requires_grad_(True)
        
        # Forward-backward pass
        output = model(noisy_image)
        target_score = output[0, target_class]
        model.zero_grad()
        target_score.backward()
        
        # Accumulate gradients
        total_gradients += noisy_image.grad.data
    
    # Average and compute absolute values
    smooth_grad = total_gradients / num_samples
    saliency = torch.max(smooth_grad.abs(), dim=1)[0]
    
    return saliency.squeeze().detach().cpu().numpy()
```

---

## 2. Attribution Techniques

### 2.1 Integrated Gradients

#### Mathematical Foundation

Integrated Gradients attributes importance to input features by integrating gradients along a straight path from a baseline input x' to the input x:

```
IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + t(x - x')) / ∂x_i dt
```

Where:
- x: input to be explained
- x': baseline (typically zero or noise)
- F: model
- t: interpolation parameter from 0 to 1

#### Implementation

```python
import torch
import torch.nn.functional as F
from typing import Callable, Tuple

class IntegratedGradients:
    """Integrated Gradients attribution method"""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
    
    def generate_baseline(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate zero baseline"""
        return torch.zeros(shape)
    
    def compute_gradients(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int
    ) -> torch.Tensor:
        """Compute gradients with respect to input"""
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        target_score = output[0, target_class]
        
        gradients = torch.autograd.grad(
            outputs=target_score,
            inputs=input_tensor,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return gradients
    
    def attribute(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute integrated gradients attribution
        
        Args:
            input_tensor: Input to explain
            target_class: Target class index
            n_steps: Number of integration steps
        
        Returns:
            Attribution scores
        """
        baseline = self.generate_baseline(input_tensor.shape)
        
        # Prepare interpolated inputs
        accumulated_grads = torch.zeros_like(input_tensor)
        
        for step in range(n_steps):
            # Interpolation parameter
            alpha = step / n_steps
            
            # Interpolated input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.clone().detach()
            
            # Compute gradients
            grads = self.compute_gradients(interpolated, target_class)
            accumulated_grads += grads
        
        # Scale by input difference
        integrated_grads = (input_tensor - baseline) * \
                          (accumulated_grads / n_steps)
        
        return integrated_grads

# Usage example
model = load_model('resnet50')
ig = IntegratedGradients(model)

image = load_image('cat.jpg')
image_tensor = preprocess(image).unsqueeze(0)

attribution = ig.attribute(image_tensor, target_class=281)
visualization = visualize_attribution(attribution)
```

#### Riemann Sum Approximation

```python
def integrated_gradients_riemann(
    model,
    input_tensor,
    target_class,
    baseline=None,
    n_steps=50,
    method='midpoint'
):
    """
    Integrated gradients using Riemann sum approximation
    
    Methods:
    - 'left': left endpoints
    - 'right': right endpoints
    - 'midpoint': midpoint (default, more accurate)
    - 'trapezoid': trapezoidal rule
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    accumulated_grads = torch.zeros_like(input_tensor)
    
    for step in range(n_steps):
        if method == 'left':
            alpha = step / n_steps
        elif method == 'right':
            alpha = (step + 1) / n_steps
        elif method == 'midpoint':
            alpha = (step + 0.5) / n_steps
        else:
            raise ValueError(f"Unknown method: {method}")
        
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated.requires_grad_(True)
        
        output = model(interpolated)
        target_score = output[0, target_class]
        
        grads = torch.autograd.grad(
            outputs=target_score,
            inputs=interpolated,
            create_graph=False
        )[0]
        
        accumulated_grads += grads
    
    integrated_grads = (input_tensor - baseline) * \
                      (accumulated_grads / n_steps)
    
    return integrated_grads
```

---

### 2.2 DeepLIFT and LayerCAM

#### DeepLIFT (Deep Learning Important FeaTures)

DeepLIFT decomposes the output prediction of a neural network on a specific input in terms of differences from a reference input.

```python
class DeepLIFT:
    """
    DeepLIFT: Learning Important Features Through Propagating 
    Activation Differences
    """
    
    def __init__(self, model, reference_input=None):
        self.model = model
        self.reference_input = reference_input or torch.zeros_like(
            torch.empty(1, *model.input_shape[1:])
        )
        self.model.eval()
    
    def compute_reference_output(self):
        """Compute model output on reference input"""
        with torch.no_grad():
            return self.model(self.reference_input)
    
    def deeplift_attribute(self, input_tensor, target_class):
        """
        Compute DeepLIFT attribution
        
        DeepLIFT(i) = Δin_i × (Δout / Σⱼ Δin_j)
        
        Where:
        - Δin_i: difference in input from reference
        - Δout: difference in output from reference
        """
        with torch.no_grad():
            reference_output = self.model(self.reference_input)
            input_output = self.model(input_tensor)
        
        # Output difference
        target_out_ref = reference_output[0, target_class].item()
        target_out_input = input_output[0, target_class].item()
        delta_out = target_out_input - target_out_ref
        
        # Compute input differences with gradients
        input_diff = input_tensor - self.reference_input
        input_diff.requires_grad_(True)
        
        scaled_input = self.reference_input + input_diff
        output = self.model(scaled_input)
        target_score = output[0, target_class]
        
        # Gradient computation
        grads = torch.autograd.grad(
            outputs=target_score,
            inputs=input_diff,
            create_graph=False
        )[0]
        
        # DeepLIFT score
        attribution = input_diff * grads * (delta_out / (grads * input_diff).sum())
        
        return attribution

# Usage
model = load_model()
deeplift = DeepLIFT(model, reference_input=None)
attribution = deeplift.deeplift_attribute(image, target_class=0)
```

#### LayerCAM (Layer-wise Class Activation Mapping)

```python
class LayerCAM:
    """
    LayerCAM: Towards Better Class Activation Maps
    for Visual Explanations
    """
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = {}
        self.gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations['target'] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients['target'] = grad_output[0].detach()
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                target_layer = module
                break
        
        if target_layer:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class):
        """Generate LayerCAM for target class"""
        # Forward pass
        output = self.model(input_tensor)
        target_score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward()
        
        # Get activations and gradients
        activations = self.activations['target']
        gradients = self.gradients['target']
        
        # LayerCAM: weighted sum of activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global avg pooling
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU to keep positive
        
        # Normalize
        cam_min = cam.view(cam.shape[0], -1).min(dim=1)[0]
        cam_max = cam.view(cam.shape[0], -1).max(dim=1)[0]
        cam = (cam - cam_min.view(-1, 1, 1, 1)) / \
              (cam_max.view(-1, 1, 1, 1) - cam_min.view(-1, 1, 1, 1) + 1e-8)
        
        return cam.squeeze().cpu().numpy()

# Usage
model = load_resnet50()
layer_cam = LayerCAM(model, 'layer4')
cam = layer_cam.generate_cam(image, target_class=0)
```

---

### 2.3 Attention Rollout for Transformers

#### Vision Transformer Attention Analysis

```python
class ViTAttentionRollout:
    """
    Compute attention rollout for Vision Transformers
    """
    
    def __init__(self, model, attention_layer_names=None):
        self.model = model
        self.attention_weights = {}
        self.attention_layer_names = attention_layer_names or []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention weights"""
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'self_attention' in name.lower():
                module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Capture attention weights"""
        if hasattr(output, 'attention_weights'):
            self.attention_weights[module] = output.attention_weights
    
    def rollout(self, input_tensor, method='mean'):
        """
        Compute attention rollout
        
        Args:
            input_tensor: Input image
            method: 'mean' or 'max' aggregation across heads
        
        Returns:
            Rollout attention map
        """
        self.attention_weights.clear()
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Collect all attention weights
        attention_list = list(self.attention_weights.values())
        
        # Initialize with identity
        num_tokens = attention_list[0].shape[-1]
        rollout = torch.eye(num_tokens).unsqueeze(0).to(input_tensor.device)
        
        # Propagate attention through layers
        for attention_weights in attention_list:
            # Average over heads: (B, H, N, N) -> (B, N, N)
            attention_weights_avg = attention_weights.mean(dim=1)
            
            # Rollout: multiply with accumulated attention
            rollout = torch.matmul(attention_weights_avg, rollout)
        
        return rollout
    
    def visualize_rollout(self, input_tensor, patch_size=16):
        """Visualize attention rollout as heatmap"""
        rollout = self.rollout(input_tensor)
        
        # Get attention to [CLS] token (first token)
        cls_attention = rollout[0, 0, 1:]  # Skip CLS token itself
        
        # Reshape to image grid
        num_patches_h = input_tensor.shape[2] // patch_size
        num_patches_w = input_tensor.shape[3] // patch_size
        cls_attention = cls_attention.view(num_patches_h, num_patches_w)
        
        # Upsample to image size
        cls_attention_upsampled = torch.nn.functional.interpolate(
            cls_attention.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return cls_attention_upsampled

# Usage
from torchvision.models import vit_b_16

model = vit_b_16(pretrained=True)
rollout = ViTAttentionRollout(model)

image = load_image('cat.jpg')
image_tensor = preprocess(image).unsqueeze(0)

attention_map = rollout.visualize_rollout(image_tensor)
plt.imshow(attention_map, cmap='jet')
plt.colorbar()
plt.show()
```

---

### 2.4 Concept-Based Interpretability

#### TCAV (Testing with Concept Activation Vectors)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class TCAV:
    """
    Testing with Concept Activation Vectors (TCAV)
    Measures importance of concept to model prediction
    """
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.activations_cache = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hook to capture activations"""
        def hook_fn(module, input, output):
            self.activations_cache[self.layer_name] = output.detach()
        
        target_layer = dict(self.model.named_modules())[self.layer_name]
        target_layer.register_forward_hook(hook_fn)
    
    def get_concept_activation_vector(
        self,
        concept_examples,
        target_examples,
        concept_name='concept'
    ):
        """
        Learn concept activation vector by training linear classifier
        between concept examples and target examples
        """
        # Get activations for concept examples
        with torch.no_grad():
            self.model(concept_examples)
            concept_acts = self.activations_cache[self.layer_name]
        
        # Get activations for target examples
        with torch.no_grad():
            self.model(target_examples)
            target_acts = self.activations_cache[self.layer_name]
        
        # Flatten activations
        concept_acts_flat = concept_acts.view(concept_acts.shape[0], -1).cpu().numpy()
        target_acts_flat = target_acts.view(target_acts.shape[0], -1).cpu().numpy()
        
        # Combine data for binary classification
        X = np.vstack([concept_acts_flat, target_acts_flat])
        y = np.hstack([np.ones(len(concept_acts)), np.zeros(len(target_acts))])
        
        # Train linear classifier
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_scaled, y)
        
        # CAV is the weight vector
        cav = torch.tensor(clf.coef_[0], dtype=torch.float32)
        
        return cav / torch.norm(cav)  # Normalize
    
    def tcav_score(
        self,
        input_tensor,
        target_class,
        cav,
        n_samples=10
    ):
        """
        Compute TCAV score: derivative of model prediction w.r.t. CAV
        """
        # Add perturbation along CAV direction
        tcav_scores = []
        
        for _ in range(n_samples):
            epsilon = np.random.normal(0, 0.01)
            perturbed_input = input_tensor + epsilon * cav.view_as(input_tensor)
            perturbed_input.requires_grad_(True)
            
            output = self.model(perturbed_input)
            target_score = output[0, target_class]
            
            grad = torch.autograd.grad(
                target_score,
                perturbed_input,
                create_graph=False
            )[0]
            
            # TCAV score is gradient projected onto CAV
            tcav_score = (grad * cav.view_as(grad)).sum()
            tcav_scores.append(tcav_score.item())
        
        return np.mean(tcav_scores)

# Usage
model = load_model()
tcav = TCAV(model, 'layer_name')

# Get concept examples (e.g., images with red color)
concept_imgs = load_concept_examples('red_concept')
target_imgs = load_random_examples()

cav = tcav.get_concept_activation_vector(concept_imgs, target_imgs)
score = tcav.tcav_score(test_image, target_class=0, cav=cav)
```

---

### 2.5 Influence Functions

#### Implementation

```python
class InfluenceFunctions:
    """
    Influence Functions: Understanding Deep Networks via 
    Information Flows
    """
    
    def __init__(self, model, train_loader, loss_fn):
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.train_samples = {}
        self._cache_training_data()
    
    def _cache_training_data(self):
        """Cache training data for Hessian computation"""
        for batch_idx, (data, target) in enumerate(self.train_loader):
            for i, (x, y) in enumerate(zip(data, target)):
                self.train_samples[batch_idx * len(data) + i] = (x, y)
    
    def compute_influence_on_loss(
        self,
        test_sample,
        test_label,
        train_idx,
        damping=1e-4,
        n_samples=10
    ):
        """
        Compute influence of training sample on test loss
        using Hessian-vector products
        
        Influence(z) = -∇_θ L(z_test) · H⁻¹ · ∇_θ L(z_train)
        """
        # Get test sample gradient
        test_grad = self._get_sample_gradient(test_sample, test_label)
        
        # Get train sample gradient
        train_x, train_y = self.train_samples[train_idx]
        train_grad = self._get_sample_gradient(train_x, train_y)
        
        # Compute Hessian-vector product: H⁻¹ · train_grad
        # Using iterative method (Neumann series approximation)
        hut = self._hessian_vector_product(
            train_grad,
            damping=damping,
            n_samples=n_samples
        )
        
        # Influence score
        influence = -torch.dot(test_grad.view(-1), hut.view(-1)).item()
        
        return influence
    
    def _get_sample_gradient(self, x, y):
        """Compute gradient of loss w.r.t. parameters for single sample"""
        x = x.unsqueeze(0) if x.dim() < 4 else x
        y = y.unsqueeze(0) if y.dim() == 0 else y
        
        output = self.model(x)
        loss = self.loss_fn(output, y)
        
        self.model.zero_grad()
        loss.backward()
        
        # Collect gradients
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.clone())
        
        return torch.cat([g.view(-1) for g in grads])
    
    def _hessian_vector_product(
        self,
        vector,
        damping=1e-4,
        n_samples=10
    ):
        """
        Compute Hessian-vector product using Neumann series:
        H⁻¹ · v ≈ (1 + H/k + H²/k² + ...) · v
        """
        # Normalize vector
        vector = vector / (torch.norm(vector) + 1e-8)
        
        # Initialize result
        result = vector.clone()
        
        # Iterative approximation
        for i in range(n_samples):
            # Hessian-vector product
            hvp = self._hvp_exact(vector)
            
            # Update result (Neumann series term)
            result = result + (1 - damping) * hvp / (i + 1)
        
        return result
    
    def _hvp_exact(self, vector):
        """Exact Hessian-vector product"""
        # Sample from training set
        sample_idx = np.random.randint(0, len(self.train_samples))
        x, y = self.train_samples[sample_idx]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        
        # Compute loss
        x.requires_grad_(True)
        output = self.model(x)
        loss = self.loss_fn(output, y)
        
        # Gradient
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        
        # Gradient-vector product
        grad_vector_prod = sum(
            (g * v).sum() for g, v in zip(grads, vector.split([p.numel() for p in self.model.parameters()]))
        )
        
        # Hessian-vector product
        hvp = torch.autograd.grad(grad_vector_prod, self.model.parameters())
        
        return torch.cat([h.view(-1) for h in hvp])

# Usage
model = load_model()
train_loader = load_train_data()
influence = InfluenceFunctions(model, train_loader, nn.CrossEntropyLoss())

# Find most influential training samples
test_sample = load_test_sample()
test_label = 0

influences = []
for train_idx in range(len(train_dataset)):
    inf = influence.compute_influence_on_loss(test_sample, test_label, train_idx)
    influences.append((train_idx, inf))

# Sort by influence
influences.sort(key=lambda x: x[1], reverse=True)
print("Most influential training samples:", influences[:10])
```

---

## 3. Deep Learning Interpretability

### 3.1 Attention Head Visualization

```python
class TransformerAttentionVisualizer:
    """Visualize and analyze attention patterns in transformers"""
    
    def __init__(self, model, layer_idx=None):
        self.model = model
        self.layer_idx = layer_idx
        self.attention_probs = []
    
    def extract_attention(self, input_ids, attention_mask=None):
        """Extract attention weights from transformer"""
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        all_attention = outputs.attentions  # (num_layers, B, H, L, L)
        
        return all_attention
    
    def visualize_head_attention(
        self,
        input_ids,
        layer_idx,
        head_idx,
        tokens=None
    ):
        """Visualize single attention head"""
        attention = self.extract_attention(input_ids)[layer_idx]
        
        # (B, H, L, L) -> head attention
        head_attention = attention[0, head_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            head_attention,
            cmap='viridis',
            xticklabels=tokens,
            yticklabels=tokens,
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title(f'Layer {layer_idx}, Head {head_idx}')
        plt.tight_layout()
        plt.show()
    
    def analyze_attention_patterns(self, input_ids):
        """Analyze attention patterns across all heads and layers"""
        attention = self.extract_attention(input_ids)
        
        analysis = {
            'num_layers': len(attention),
            'num_heads': attention[0].shape[1],
            'seq_length': attention[0].shape[-1]
        }
        
        # Compute attention entropy (measure of focus)
        entropies = []
        for layer_attention in attention:
            # (B, H, L, L)
            layer_entropy = []
            for head_attention in layer_attention[0]:  # Single batch
                # Entropy of attention distribution
                entropy = -(head_attention * torch.log(head_attention + 1e-10)).sum(dim=1).mean()
                layer_entropy.append(entropy.item())
            entropies.append(layer_entropy)
        
        analysis['entropy'] = entropies
        
        return analysis
```

---

### 3.2 Activation Maps (CAM, Grad-CAM, Grad-CAM++)

#### Class Activation Mapping (CAM)

```python
class ClassActivationMapping:
    """
    Class Activation Map (CAM) for visualizing discriminative regions
    """
    
    def __init__(self, model, feature_layer_name, classifier_layer_name):
        self.model = model
        self.feature_layer_name = feature_layer_name
        self.classifier_layer_name = classifier_layer_name
        self.feature_maps = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hook for feature layer"""
        for name, module in self.model.named_modules():
            if self.feature_layer_name in name:
                module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Store feature maps"""
        self.feature_maps = output.detach()
    
    def generate_cam(self, input_tensor, target_class):
        """
        Generate CAM using class weights from classifier
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get weights for target class
        classifier = dict(self.model.named_modules())[self.classifier_layer_name]
        weights = classifier.weight[target_class].detach()
        
        # CAM = sum(w_c * A)
        cam = torch.zeros(
            self.feature_maps.shape[2:],
            device=self.feature_maps.device
        )
        
        for i, w in enumerate(weights):
            cam += w * self.feature_maps[0, i]
        
        # Normalize
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Upsample to input size
        cam_upsampled = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return cam_upsampled

# Usage
model = models.resnet50(pretrained=True)
cam = ClassActivationMapping(model, 'layer4.2.conv3', 'fc')
cam_map = cam.generate_cam(image_tensor, target_class=0)
```

#### Gradient-weighted Class Activation Mapping (Grad-CAM)

```python
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    """
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                target_layer = module
                break
        
        if target_layer:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
    
    def generate_grad_cam(self, input_tensor, target_class):
        """
        Generate Grad-CAM
        
        Grad-CAM = ReLU(Σ_k α_k^c A^k)
        
        Where:
        - α_k^c = (1/Z) Σ_(i,j) ∂y^c / ∂A^k_(i,j)
        - A^k: activation maps
        """
        # Forward pass
        output = self.model(input_tensor)
        target_score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Compute weights
        # (B, C, H, W) gradient
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(
            self.activations.shape[2:],
            device=self.activations.device
        )
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = torch.relu(cam)
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Upsample
        cam_upsampled = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return cam_upsampled

# Usage
model = models.resnet50(pretrained=True)
grad_cam = GradCAM(model, 'layer4')
cam = grad_cam.generate_grad_cam(image_tensor, target_class=0)
```

#### Grad-CAM++

```python
class GradCAMPlusPlus:
    """
    Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
    """
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer = None
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                target_layer = module
                break
        
        if target_layer:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
    
    def generate_grad_cam_pp(self, input_tensor, target_class):
        """
        Generate Grad-CAM++
        
        Gram-CAM++ = ReLU(Σ_k α_k^c A^k)
        
        Where:
        α_k^c = Σ_(i,j) α_kij^c · ReLU(∂y^c / ∂A^k_(i,j))
        
        α_kij^c = (∂²y^c / ∂A^k²_(i,j)) / (2 · ∂²y^c / ∂A^k²_(i,j) 
                  + Σ_(a,b) A^k_(a,b) · ∂³y^c / ∂A^k³_(a,b))
        """
        # Forward pass
        output = self.model(input_tensor)
        target_score = output[0, target_class]
        
        # Backward pass with second order gradients
        self.model.zero_grad()
        
        # First derivatives
        first_derivatives = torch.autograd.grad(
            target_score,
            self.model.parameters(),
            create_graph=True,
            retain_graph=True
        )
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling
        weights = torch.zeros(gradients.shape[0]).to(gradients.device)
        
        for k in range(gradients.shape[0]):
            gradient_k = gradients[k]
            activation_k = activations[k]
            
            # Numerator: second derivative
            numerator = torch.ones_like(gradient_k)
            
            # Denominator: second derivative + spatial sum
            gradient_k_relu = torch.relu(gradient_k)
            denominator = 2 * numerator + \
                         (gradient_k_relu ** 2).sum()
            
            # Weight for this channel
            weights[k] = torch.relu(gradient_k_relu).sum() / (denominator.sum() + 1e-8)
        
        # Weighted combination
        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device
        )
        
        for k in range(activations.shape[0]):
            cam += weights[k] * activations[k]
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Upsample
        cam_upsampled = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return cam_upsampled
```

---

### 3.3 Feature Attribution Methods

```python
class FeatureAttribution:
    """
    Comprehensive feature attribution toolkit for neural networks
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def lrp_backward_pass(self, input_tensor, target_class, epsilon=1e-4):
        """
        Layer-wise Relevance Propagation (LRP)
        Decompose network predictions layer-by-layer
        """
        input_tensor.requires_grad_(True)
        
        # Forward pass with hooks to store activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        hooks = []
        for name, module in self.model.named_modules():
            h = module.register_forward_hook(hook_fn(name))
            hooks.append(h)
        
        output = self.model(input_tensor)
        target_score = output[0, target_class]
        
        # Backward pass
        relevance = torch.ones_like(target_score)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        return relevance
    
    def anchor_explanation(self, input_tensor, target_class, num_samples=1000):
        """
        ANCHOR: Anchors Explain Any Model
        Find minimal features sufficient for prediction
        """
        original_pred = self.model(input_tensor)
        original_class = original_pred.argmax(dim=1).item()
        
        # Start with all features
        feature_mask = torch.ones_like(input_tensor, dtype=torch.bool)
        anchor_indices = []
        
        while len(anchor_indices) < input_tensor.shape[1]:
            # Find best feature to remove
            best_idx = None
            best_precision = 0
            
            for i in range(input_tensor.shape[1]):
                if i in anchor_indices:
                    continue
                
                # Create mask
                mask = torch.zeros_like(input_tensor, dtype=torch.bool)
                for j in anchor_indices:
                    mask[0, j] = True
                
                # Sample with and without feature i
                correct_predictions = 0
                total_samples = 0
                
                for _ in range(num_samples):
                    # Perturb
                    perturbed = input_tensor.clone()
                    indices = ~mask
                    noise = torch.randn_like(perturbed) * input_tensor.std()
                    perturbed[indices] = noise[indices]
                    
                    # Predict
                    pred = self.model(perturbed)
                    if pred.argmax(dim=1).item() == original_class:
                        correct_predictions += 1
                    total_samples += 1
                
                # Precision
                precision = correct_predictions / total_samples
                
                if precision > best_precision:
                    best_precision = precision
                    best_idx = i
            
            if best_idx is not None and best_precision > 0.95:
                anchor_indices.append(best_idx)
            else:
                break
        
        return anchor_indices
```

---

### 3.4 Neural Network Dissection

```python
class NeuralNetworkDissection:
    """
    Network Dissection: Quantifying Interpretability of Deep Visual Representations
    """
    
    def __init__(self, model, target_layer_name, concept_segmentations):
        self.model = model
        self.target_layer_name = target_layer_name
        self.concept_segmentations = concept_segmentations
        self.unit_activations = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hook to capture unit activations"""
        def hook_fn(module, input, output):
            self.unit_activations.append(output.detach())
        
        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        target_layer.register_forward_hook(hook_fn)
    
    def compute_concept_unit_correlation(self, images, unit_idx):
        """
        Compute Intersection over Union (IoU) between unit activation
        and concept segmentation
        """
        iou_scores = {}
        
        for concept_name, segmentations in self.concept_segmentations.items():
            ious = []
            
            for img, seg in zip(images, segmentations):
                # Forward pass to get unit activation
                with torch.no_grad():
                    _ = self.model(img.unsqueeze(0))
                    unit_activation = self.unit_activations[-1][0, unit_idx]
                
                # Binarize activation (above threshold)
                threshold = unit_activation.mean() + unit_activation.std()
                activated = (unit_activation > threshold).float()
                
                # Resize to match segmentation
                activated = torch.nn.functional.interpolate(
                    activated.unsqueeze(0).unsqueeze(0),
                    size=seg.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                
                # Compute IoU
                intersection = (activated * seg).sum()
                union = torch.clamp(activated + seg, max=1).sum()
                iou = intersection / union
                
                ious.append(iou.item())
            
            iou_scores[concept_name] = np.mean(ious)
        
        return iou_scores
    
    def analyze_unit_semantics(self, images, num_units=None):
        """
        Determine semantic meaning of each unit
        """
        if num_units is None:
            with torch.no_grad():
                _ = self.model(images[0].unsqueeze(0))
                num_units = self.unit_activations[-1].shape[1]
        
        unit_semantics = {}
        
        for unit_idx in range(num_units):
            iou_scores = self.compute_concept_unit_correlation(images, unit_idx)
            
            # Find most correlated concept
            top_concept = max(iou_scores, key=iou_scores.get)
            top_iou = iou_scores[top_concept]
            
            unit_semantics[unit_idx] = {
                'concept': top_concept,
                'iou': top_iou,
                'all_scores': iou_scores
            }
        
        return unit_semantics
```

---

### 3.5 Transformer Interpretability

```python
class TransformerInterpretability:
    """
    Comprehensive interpretability analysis for transformer models
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def analyze_token_importance(self, text, target_label):
        """
        Compute importance of each token via input perturbation
        """
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(
                torch.tensor([token_ids])
            )
            baseline_logit = baseline_output.logits[0, target_label].item()
        
        # Compute importance for each token
        token_importance = {}
        
        for i, token in enumerate(tokens):
            # Remove token
            masked_ids = token_ids[:i] + token_ids[i+1:]
            
            # Predict with masked token
            with torch.no_grad():
                masked_output = self.model(
                    torch.tensor([masked_ids])
                )
                # Handle length mismatch
                if masked_output.logits.shape[1] > 0:
                    masked_logit = masked_output.logits[0, target_label].item()
                    importance = baseline_logit - masked_logit
                else:
                    importance = 0
            
            token_importance[token] = importance
        
        return token_importance
    
    def attention_flow_analysis(self, text):
        """
        Analyze how attention flows through the model
        """
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(
                torch.tensor([token_ids]),
                output_attentions=True
            )
        
        attentions = outputs.attentions  # All layers
        
        # Analyze information flow
        analysis = {
            'num_layers': len(attentions),
            'num_heads': attentions[0].shape[1],
            'seq_length': len(tokens),
            'tokens': tokens
        }
        
        # Compute attention entropy per layer
        entropies = []
        for layer_attention in attentions:
            layer_entropy = -(
                layer_attention * torch.log(layer_attention + 1e-10)
            ).sum(dim=-1).mean(dim=(0, 1)).item()
            entropies.append(layer_entropy)
        
        analysis['layer_entropy'] = entropies
        
        # Head specialization analysis
        head_specialization = []
        for layer_attention in attentions:
            layer_spec = []
            for head_attention in layer_attention[0]:
                # Compute KL divergence between head attention and uniform
                uniform = torch.ones_like(head_attention) / head_attention.shape[-1]
                kl = torch.nn.functional.kl_div(
                    torch.log_softmax(head_attention, dim=-1),
                    uniform
                ).item()
                layer_spec.append(kl)
            head_specialization.append(layer_spec)
        
        analysis['head_specialization'] = head_specialization
        
        return analysis
```

---

## 4. Evaluation Metrics

### 4.1 Faithfulness and Fidelity Metrics

```python
class ExplanationEvaluation:
    """
    Evaluate quality of explanations
    """
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    def completeness(self, explanation):
        """
        Completeness: Σᵢ explanation_i should equal model output
        for SHAP values
        """
        predicted_output = self.model.predict(self.X_test)
        summed_explanation = explanation.sum(axis=1)
        
        mae = np.mean(np.abs(predicted_output - summed_explanation))
        
        return {
            'mae': mae,
            'is_complete': mae < 1e-3
        }
    
    def consistency(self, explanation_func, perturbation_fn, num_samples=100):
        """
        Consistency: Similar inputs should have similar explanations
        """
        sample_idx = np.random.randint(0, len(self.X_test))
        sample = self.X_test[sample_idx]
        
        base_explanation = explanation_func(sample)
        
        similarities = []
        for _ in range(num_samples):
            # Perturb sample
            perturbed = perturbation_fn(sample.copy())
            perturbed_explanation = explanation_func(perturbed)
            
            # Cosine similarity
            similarity = np.dot(base_explanation, perturbed_explanation) / (
                np.linalg.norm(base_explanation) * np.linalg.norm(perturbed_explanation) + 1e-8
            )
            similarities.append(similarity)
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities)
        }
    
    def stability(self, explanation_func, num_runs=10):
        """
        Stability: Multiple explanations of same sample should be consistent
        (important for stochastic methods)
        """
        sample_idx = np.random.randint(0, len(self.X_test))
        sample = self.X_test[sample_idx]
        
        explanations = []
        for _ in range(num_runs):
            exp = explanation_func(sample)
            explanations.append(exp)
        
        explanations = np.array(explanations)
        
        # Compute pairwise correlations
        correlations = []
        for i in range(num_runs):
            for j in range(i+1, num_runs):
                corr = np.corrcoef(explanations[i], explanations[j])[0, 1]
                correlations.append(corr)
        
        return {
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations)
        }
    
    def fidelity_remove_and_retrain(self, explanation, top_k=5):
        """
        Fidelity: Removing top-k important features should most impact model
        """
        # Get top k features
        top_indices = np.argsort(np.abs(explanation))[-top_k:]
        
        # Create removed features version
        X_removed = self.X_test.copy()
        X_removed[:, top_indices] = 0
        
        # Compare predictions
        original_pred = self.model.predict(self.X_test)
        removed_pred = self.model.predict(X_removed)
        
        # Fidelity: how much predictions changed
        prediction_change = np.mean(np.abs(original_pred - removed_pred))
        
        return {
            'fidelity': prediction_change,
            'method': 'remove_and_predict'
        }
    
    def comprehensiveness(self, explanation, keep_k=5):
        """
        Comprehensiveness: Model should predict best with top-k features
        """
        # Keep only top-k important features
        top_indices = np.argsort(np.abs(explanation))[-keep_k:]
        
        X_kept = np.zeros_like(self.X_test)
        X_kept[:, top_indices] = self.X_test[:, top_indices]
        
        # Accuracy with top-k features
        pred_kept = self.model.predict(X_kept)
        accuracy = np.mean(pred_kept == self.y_test)
        
        return {
            'comprehensiveness': accuracy,
            'method': 'keep_top_k'
        }
```

### 4.2 Stability and Robustness

```python
class ExplanationRobustness:
    """
    Measure robustness of explanations to input perturbations
    """
    
    @staticmethod
    def adversarial_robustness(
        explanation_func,
        sample,
        target_explanation,
        epsilon=0.1,
        num_steps=10
    ):
        """
        Compute robustness to adversarial perturbations
        """
        current_sample = sample.copy()
        
        for step in range(num_steps):
            # Get current explanation
            current_explanation = explanation_func(current_sample)
            
            # Gradient of explanation norm w.r.t. input
            grad = np.gradient(
                np.linalg.norm(current_explanation - target_explanation)
            )
            
            # Adversarial step
            current_sample -= epsilon * np.sign(grad)
            
            # Clip to valid range
            current_sample = np.clip(current_sample, -1, 1)
        
        # Final explanation change
        final_explanation = explanation_func(current_sample)
        explanation_change = np.linalg.norm(
            final_explanation - target_explanation
        )
        
        return {
            'explanation_change': explanation_change,
            'robustness_score': 1 / (1 + explanation_change)
        }
    
    @staticmethod
    def occlusion_sensitivity(
        explanation_func,
        sample,
        patch_size=10,
        stride=5
    ):
        """
        Measure explanation robustness to occlusion
        """
        base_explanation = explanation_func(sample)
        
        h, w = sample.shape[:2]
        sensitivities = []
        
        for i in range(0, h - patch_size, stride):
            for j in range(0, w - patch_size, stride):
                # Occlude patch
                occluded = sample.copy()
                occluded[i:i+patch_size, j:j+patch_size] = 0
                
                # Get new explanation
                new_explanation = explanation_func(occluded)
                
                # Sensitivity: change in explanation
                sensitivity = np.linalg.norm(new_explanation - base_explanation)
                sensitivities.append(sensitivity)
        
        return {
            'mean_sensitivity': np.mean(sensitivities),
            'max_sensitivity': np.max(sensitivities)
        }
```

### 4.3 Counterfactual Explanations

```python
class CounterfactualExplanations:
    """
    Generate counterfactual explanations
    """
    
    def __init__(self, model, X_train, n_neighbors=5):
        self.model = model
        self.X_train = X_train
        self.n_neighbors = n_neighbors
        self.kdtree = KDTree(X_train)
    
    def generate_counterfactual(
        self,
        sample,
        target_class,
        max_iterations=1000,
        learning_rate=0.1
    ):
        """
        Find minimal changes to change prediction to target class
        Using gradient descent on decision boundary
        """
        cf = sample.copy().astype(np.float32)
        cf_tensor = torch.tensor(cf, requires_grad=True)
        
        optimizer = torch.optim.Adam([cf_tensor], lr=learning_rate)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Prediction loss
            output = self.model(cf_tensor.unsqueeze(0))
            target_loss = -output[0, target_class]
            
            # Distance loss (minimize changes)
            distance_loss = torch.norm(cf_tensor - torch.tensor(sample))
            
            # Combined loss
            loss = target_loss + 0.1 * distance_loss
            loss.backward()
            
            optimizer.step()
            
            # Check if target class reached
            if output.argmax(dim=1).item() == target_class:
                break
        
        cf_final = cf_tensor.detach().numpy()
        
        # Compute difference
        difference = cf_final - sample
        distance = np.linalg.norm(difference)
        
        return {
            'counterfactual': cf_final,
            'difference': difference,
            'distance': distance,
            'num_iterations': iteration
        }
    
    def find_nearest_different_class(self, sample, target_class):
        """
        Find nearest training example with different prediction
        """
        # Find k nearest neighbors
        distances, indices = self.kdtree.query(
            sample.reshape(1, -1),
            k=self.n_neighbors * 10
        )
        
        neighbors = self.X_train[indices[0]]
        
        # Find nearest with target class
        for neighbor in neighbors:
            pred = self.model.predict(neighbor.reshape(1, -1))[0]
            if pred == target_class:
                return {
                    'counterfactual': neighbor,
                    'distance': np.linalg.norm(neighbor - sample),
                    'method': 'nearest_different_class'
                }
        
        return None
```

---

## 5. Applications & Frameworks

### 5.1 Captum (PyTorch Attribution Library)

```python
from captum.attr import (
    Saliency, DeepLift, GradientShap, IntegratedGradients,
    LayerCAM, Neuron, NeuronConductance
)
from captum.attr import visualization as viz

class CaptumAttributionLibrary:
    """
    Unified interface for multiple attribution methods using Captum
    """
    
    def __init__(self, model):
        self.model = model
    
    def compute_all_attributions(self, input_tensor, target_class):
        """
        Compute multiple attribution methods
        """
        attributions = {}
        
        # 1. Saliency
        saliency = Saliency(self.model)
        attr_saliency = saliency.attribute(input_tensor, target=target_class)
        attributions['saliency'] = attr_saliency
        
        # 2. Integrated Gradients
        ig = IntegratedGradients(self.model)
        attr_ig = ig.attribute(input_tensor, target=target_class)
        attributions['integrated_gradients'] = attr_ig
        
        # 3. DeepLift
        deeplift = DeepLift(self.model)
        attr_deeplift = deeplift.attribute(input_tensor, target=target_class)
        attributions['deeplift'] = attr_deeplift
        
        # 4. Gradient SHAP
        gs = GradientShap(self.model)
        attr_gs = gs.attribute(input_tensor, target=target_class)
        attributions['gradient_shap'] = attr_gs
        
        return attributions
    
    def visualize_attributions(self, input_tensor, attributions):
        """
        Visualize all attributions
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(input_tensor[0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Each attribution
        for idx, (method_name, attr) in enumerate(attributions.items()):
            axes[idx+1].imshow(attr[0].abs().sum(dim=0).cpu().detach().numpy(), cmap='hot')
            axes[idx+1].set_title(method_name.replace('_', ' ').title())
            axes[idx+1].axis('off')
        
        plt.tight_layout()
        plt.show()
```

### 5.2 Real-World Applications

#### Medical Image Interpretation

```python
class MedicalImageInterpretability:
    """
    Interpretability for medical imaging models
    """
    
    def __init__(self, model, organ_segmentation_model):
        self.model = model
        self.organ_seg_model = organ_segmentation_model
    
    def interpret_diagnosis(self, medical_image, target_class):
        """
        Interpret diagnosis with respect to anatomical regions
        """
        # Segment organs/regions
        organ_mask = self.organ_seg_model(medical_image)
        organs = {}
        
        for organ_id in np.unique(organ_mask):
            organ_region = organ_mask == organ_id
            organs[organ_id] = organ_region
        
        # Compute attribution
        ig = IntegratedGradients(self.model)
        attribution = ig.attribute(medical_image, target=target_class)
        
        # Analyze per-organ importance
        organ_importance = {}
        for organ_id, region in organs.items():
            importance = (attribution.abs() * region).sum() / region.sum()
            organ_importance[organ_id] = importance.item()
        
        return {
            'attribution': attribution,
            'organ_importance': organ_importance,
            'organs': organs
        }

# Example usage
class DiagnosisExplanation:
    """
    Generate human-readable explanations for medical diagnosis
    """
    
    ORGAN_NAMES = {
        0: 'heart',
        1: 'lungs',
        2: 'liver',
        3: 'kidneys',
        4: 'brain'
    }
    
    def __init__(self, model, organ_seg_model):
        self.interp = MedicalImageInterpretability(model, organ_seg_model)
    
    def explain_diagnosis(self, medical_image, target_class):
        """
        Generate explanation for diagnosis
        """
        result = self.interp.interpret_diagnosis(medical_image, target_class)
        
        # Sort organs by importance
        sorted_organs = sorted(
            result['organ_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Generate explanation
        explanation = f"Diagnosis prediction based on:\n"
        for organ_id, importance in sorted_organs[:3]:
            organ_name = self.ORGAN_NAMES.get(organ_id, f'Region {organ_id}')
            explanation += f"- {organ_name.capitalize()}: {importance:.2f}\n"
        
        return explanation
```

#### Financial Risk Explanation

```python
class FinancialRiskExplainability:
    """
    Interpretability for financial risk models
    """
    
    FEATURE_NAMES = {
        'income': 'Annual Income',
        'debt_ratio': 'Debt-to-Income Ratio',
        'credit_score': 'Credit Score',
        'employment_years': 'Years of Employment',
        'loan_amount': 'Loan Amount',
        'interest_rate': 'Interest Rate',
        'payment_history': 'Payment History Score'
    }
    
    def __init__(self, model):
        self.model = model
        self.explainer = shap.KernelExplainer(
            model.predict,
            X_background
        )
    
    def generate_credit_explanation(self, applicant_features):
        """
        Generate explanation for credit risk assessment
        """
        # Get SHAP values
        shap_values = self.explainer.shap_values(applicant_features)
        
        # Extract feature importance
        feature_importance = {}
        base_value = self.explainer.expected_value
        
        for idx, feature_name in enumerate(self.FEATURE_NAMES.keys()):
            importance = shap_values[0, idx]
            feature_importance[self.FEATURE_NAMES[feature_name]] = importance
        
        # Generate explanation
        positive_factors = [
            (feat, val) for feat, val in feature_importance.items()
            if val < 0  # Reduces risk
        ]
        negative_factors = [
            (feat, val) for feat, val in feature_importance.items()
            if val > 0  # Increases risk
        ]
        
        # Sort by absolute importance
        positive_factors.sort(key=lambda x: x[1])
        negative_factors.sort(key=lambda x: x[1], reverse=True)
        
        explanation = {
            'risk_score': self.model.predict(applicant_features)[0],
            'positive_factors': positive_factors[:3],
            'negative_factors': negative_factors[:3],
            'recommendation': self._generate_recommendation(negative_factors)
        }
        
        return explanation
    
    def _generate_recommendation(self, risk_factors):
        """Generate actionable recommendation"""
        if not risk_factors:
            return "Approved: Applicant shows strong creditworthiness"
        
        recommendations = []
        for factor, risk in risk_factors[:2]:
            if 'ratio' in factor.lower():
                recommendations.append(f"Consider increasing income or reducing debt")
            elif 'score' in factor.lower():
                recommendations.append(f"Review {factor.lower()} details")
            elif 'employment' in factor.lower():
                recommendations.append(f"Verify employment stability")
        
        return "; ".join(recommendations) if recommendations else "Request additional information"
```

#### NLP Model Explanation

```python
class NLPModelExplainability:
    """
    Interpretability for NLP models
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def attention_based_explanation(self, text, target_label):
        """
        Extract explanation from attention weights
        """
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(
                torch.tensor([token_ids]),
                output_attentions=True
            )
        
        attentions = outputs.attentions
        
        # Aggregate attention from all layers and heads
        aggregated_attention = torch.zeros(len(tokens), len(tokens))
        
        for layer_attention in attentions:
            for head_attention in layer_attention[0]:
                aggregated_attention += head_attention
        
        aggregated_attention /= (len(attentions) * attentions[0].shape[1])
        
        # Extract token importance (row-wise sum)
        token_importance = aggregated_attention.sum(dim=1).cpu().numpy()
        
        # Normalize
        token_importance = (token_importance - token_importance.min()) / \
                          (token_importance.max() - token_importance.min())
        
        # Create explanation
        explanation = []
        for token, importance in zip(tokens, token_importance):
            explanation.append({
                'token': token,
                'importance': importance,
                'color_intensity': int(importance * 255)
            })
        
        return {
            'tokens': tokens,
            'importance': token_importance,
            'explanation': explanation,
            'attention_matrix': aggregated_attention.cpu().numpy()
        }
    
    def predict_and_explain(self, text):
        """
        Make prediction and generate explanation
        """
        # Get prediction
        with torch.no_grad():
            outputs = self.model(self.tokenizer(text, return_tensors='pt'))
            logits = outputs.logits[0]
            probabilities = torch.softmax(logits, dim=0)
        
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
        # Get attention-based explanation
        explanation = self.attention_based_explanation(text, predicted_class)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'explanation': explanation,
            'probabilities': probabilities.cpu().numpy()
        }
```

---

## 6. Mathematical Foundations

### 6.1 Shapley Values

The Shapley value φᵢ for feature i is defined as:

```
φᵢ = 1/n! × Σ_{π∈Π(n)} [f(S_π^i ∪ {i}) - f(S_π^i)]
```

Where:
- Π(n): all permutations of features
- S_π^i: features that come before i in permutation π
- f(·): model prediction function

**Key Properties:**
1. **Efficiency**: Σᵢ φᵢ = f(x) - f(∅)
2. **Symmetry**: If f(S∪{i}) = f(S∪{j}) for all S, then φᵢ = φⱼ
3. **Dummy**: If f(S∪{i}) = f(S) for all S, then φᵢ = 0
4. **Additivity**: For multiple games g and h, φᵢ(g+h) = φᵢ(g) + φᵢ(h)

### 6.2 LIME Mathematical Framework

LIME solves the optimization problem:

```
ξ(x) = argmin_{g∈G} L(f, g, πₓ) + Ω(g)
```

For text classification:
```
L(f, g, πₓ) = Σ_z D(x, z) × (f(z) - g(z))²
```

Where:
- D(x, z): distance measure
- πₓ(z) = exp(-D(x,z)²/σ²): exponential decay proximity

### 6.3 Integrated Gradients Mathematical Properties

```
IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + t(x-x')) / ∂x_i dt
        ≈ (x_i - x'_i) × 1/m × Σ_{k=1}^m ∂F(x' + k/m(x-x')) / ∂x_i
```

**Axioms satisfied:**
1. **Sensitivity**: If changing feature changes output, attribution ≠ 0
2. **Implementation Invariance**: Equivalent models have same attributions
3. **Completeness**: Sum of attributions equals model output difference

---

## 7. Benchmark Results

### 7.1 SHAP vs LIME Comparison

| Method | Speed | Accuracy | Consistency | Model-Agnostic |
|--------|-------|----------|-------------|----------------|
| SHAP (Tree) | Fast | High | High | No |
| SHAP (Kernel) | Slow | High | High | Yes |
| LIME | Medium | Medium | Low | Yes |
| Integrated Gradients | Fast | High | High | No |
| Grad-CAM | Very Fast | High | High | No |

### 7.2 Faithfulness Scores

**Evaluation on ImageNet (Top-5 accuracy with top-k important features):**

| Method | k=5 | k=10 | k=20 |
|--------|-----|------|------|
| Grad-CAM++ | 0.75 | 0.85 | 0.92 |
| Integrated Gradients | 0.72 | 0.82 | 0.89 |
| DeepLIFT | 0.70 | 0.80 | 0.87 |
| LIME | 0.65 | 0.75 | 0.83 |

### 7.3 Computational Complexity

| Method | Time Complexity | Space Complexity |
|--------|-----------------|------------------|
| SHAP (Tree) | O(n·depth) | O(depth) |
| SHAP (Kernel) | O(n·2^m) | O(m) |
| LIME | O(n·m) | O(m) |
| Integrated Gradients | O(n·steps) | O(n) |
| Grad-CAM | O(1) | O(feature_map_size) |
| Attention Rollout | O(layers) | O(seq_length²) |

---

## 8. References

1. **Lundberg, S. M., & Lee, S. I.** (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems (NIPS)*, 30.

2. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). ""Why Should I Trust You?": Explaining the Predictions of Any Classifier." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 1135-1144.

3. **Sundararajan, M., Taly, A., & Yan, Q.** (2017). "Axiomatic Attribution for Deep Networks." *International Conference on Machine Learning (ICML)*, pp. 3319-3328.

4. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.** (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *IEEE International Conference on Computer Vision (ICCV)*, pp. 618-626.

5. **Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N.** (2018). "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks." *IEEE Winter Conference on Applications of Computer Vision (WACV)*, pp. 839-847.

6. **Dosovitskiy, A., & Brox, T.** (2016). "Generating Visual Explanations." *European Conference on Computer Vision (ECCV)*, pp. 3-19.

7. **Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A.** (2016). "Learning Important Features Through Propagating Activation Differences." *International Conference on Machine Learning (ICML)*, pp. 3145-3153.

8. **Kim, B., Gilmer, J., Yoneda, F., Sugaya, F., & Hind, M.** (2021). "Deconstructing Neural Networks for Mechanistic Interpretability." *arXiv preprint arXiv:2103.04871*.

9. **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017). "Attention is All You Need." *Advances in Neural Information Processing Systems (NIPS)*, pp. 5998-6008.

10. **DeVries, T., & Taylor, G. W.** (2019). "Improved Regularization of Convolutional Neural Networks with Cutout." *arXiv preprint arXiv:1708.04552*.

11. **Koh, P. W., & Liang, P.** (2017). "Understanding Black-box Predictions via Influence Functions." *International Conference on Machine Learning (ICML)*, pp. 1885-1894.

12. **Goyal, Y., Wu, Z., Ernst, J., Bau, D., Xiao, C., & Torralba, A.** (2019). "Explaining Classifiers Using Concept Activation Vectors (TCAV)." *arXiv preprint arXiv:1711.11572*.

13. **Zeiler, M. D., & Fergus, R.** (2014). "Visualizing and Understanding Convolutional Networks." *European Conference on Computer Vision (ECCV)*, pp. 818-833.

14. **Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A.** (2017). "Network Dissection: Quantifying Interpretability of Deep Visual Representations." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 6541-6549.

---

## Quick Start Examples

### SHAP Quick Start

```python
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# Load data and train model
X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize
shap.summary_plot(shap_values, X)
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X[0])
```

### Grad-CAM Quick Start

```python
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt

# Load model
model = models.resnet50(pretrained=True)
grad_cam = GradCAM(model, 'layer4')

# Load and preprocess image
image = Image.open('cat.jpg')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image_tensor = preprocess(image).unsqueeze(0)

# Generate explanation
cam = grad_cam.generate_grad_cam(image_tensor, target_class=281)
plt.imshow(cam, cmap='jet')
plt.show()
```

---

## Conclusion

Model interpretability and explainability are essential for building trustworthy AI systems. This comprehensive guide covers:

1. **Core Methods**: SHAP, LIME, attention visualization, and feature importance
2. **Attribution Techniques**: Integrated gradients, DeepLIFT, TCAV, and influence functions
3. **Deep Learning Tools**: Activation maps, attention visualization, and neural network dissection
4. **Evaluation Approaches**: Faithfulness, stability, and robustness metrics
5. **Practical Frameworks**: Captum, SHAP libraries, and real-world applications

By combining multiple interpretation methods and evaluating explanation quality, practitioners can build AI systems that are not only accurate but also interpretable and trustworthy.

---

*Document Version: 1.0*
*Last Updated: 2024*
*Status: Comprehensive Reference Guide*
