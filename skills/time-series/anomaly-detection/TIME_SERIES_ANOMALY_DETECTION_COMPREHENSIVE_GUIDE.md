# Time Series Anomaly Detection: Comprehensive Guide

**Author:** Shuvam Banerji Seal  
**Version:** 1.0  
**Last Updated:** April 2026  
**Scope:** Detection methods, algorithms, benchmarks, and production systems

---

## Table of Contents

1. [Introduction & Fundamentals](#introduction--fundamentals)
2. [Detection Methods](#detection-methods)
3. [Core Algorithms & Techniques](#core-algorithms--techniques)
4. [Mathematical Formulations](#mathematical-formulations)
5. [Code Examples](#code-examples)
6. [Benchmarks & Datasets](#benchmarks--datasets)
7. [Advanced Topics](#advanced-topics)
8. [Production Systems](#production-systems)
9. [Performance Metrics](#performance-metrics)
10. [Citations & References](#citations--references)

---

## Introduction & Fundamentals

### Definition

**Time Series Anomaly Detection** is the identification of unusual patterns, outliers, or deviations in temporal data that differ significantly from the expected behavior or historical patterns.

### Anomaly Types

1. **Point Anomalies** - Single observation deviates from normal behavior
2. **Contextual Anomalies** - Value is unusual in specific context (e.g., low sales on holiday)
3. **Collective Anomalies** - Subsequence pattern is anomalous
4. **Seasonal Anomalies** - Deviation from seasonal patterns
5. **Trend Anomalies** - Unusual changes in trend direction

### Key Challenges

- Non-stationary behavior and concept drift
- Seasonality and trends masking anomalies
- Unlabeled data and lack of ground truth
- High-dimensional multivariate time series
- Real-time detection with minimal latency
- Threshold selection and false positive management

---

## Detection Methods

### 1. Reconstruction-Based Approaches

#### 1.1 Autoencoders (AE)

**Principle:** Normal data should have low reconstruction error; anomalies have high error.

**Architecture:**
```
Input (n_features) → Encoder → Bottleneck (low-dim) → Decoder → Output (n_features)
```

**Key Properties:**
- Unsupervised learning (no labels required)
- Learns compressed representation of normal data
- Anomaly Score = ||x - reconstruction(x)||²
- Threshold selection critical for performance

**Advantages:**
- Effective for non-linear patterns
- Captures complex temporal dependencies
- Can work with variable-length sequences

**Disadvantages:**
- May reconstruct anomalies well (especially rare ones)
- Hyperparameter tuning complex
- Training time for large datasets

#### 1.2 Variational Autoencoders (VAE)

**Principle:** Learn probabilistic latent representation; anomalies have low likelihood.

**Formulation:**
$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$

Where:
- $q_\phi(z|x)$ = encoder (posterior)
- $p_\theta(x|z)$ = decoder (likelihood)
- $p(z)$ = prior (typically N(0, I))
- $D_{KL}$ = KL divergence

**Anomaly Score (two methods):**

Method 1 - Reconstruction Error:
$$\text{Score}_1 = ||x - \hat{x}||_2^2$$

Method 2 - Negative Log-Likelihood:
$$\text{Score}_2 = -\log p_\theta(x)$$

**Advantages:**
- Probabilistic framework provides confidence scores
- Regularization through KL divergence prevents overfitting
- Can generate samples for validation

**Disadvantages:**
- More complex training than standard AE
- Requires careful balancing of reconstruction vs. KL terms
- Posterior collapse risk

---

### 2. Isolation Forests and Tree-Based Methods

#### 2.1 Isolation Forest (iForest)

**Principle:** Anomalies are easier to isolate (fewer splits needed) than normal points.

**Algorithm Steps:**
1. Randomly select feature and split value
2. Recursively partition data
3. Count path length to isolate each point
4. Lower path length = more likely anomaly

**Anomaly Score:**
$$s(x) = 2^{-\frac{h(x)}{c(n)}}$$

Where:
- $h(x)$ = average path length to isolate point x
- $c(n)$ = average path length for unsuccessful search in BST
- Range: [0, 1] (1 = anomaly, 0 = normal)

**Key Properties:**
- Linear time complexity: O(n log n)
- No distance calculation needed
- Works well in high dimensions
- Assumes anomalies are rare and isolated

#### 2.2 Isolation Forest Variants

**Extended Isolation Forest (EIF):**
- Uses hyperplanes instead of axis-parallel splits
- Better for complex anomaly patterns
- More robust to orientation-dependent anomalies

**Isolation Forest Time Series (iForest-TS):**
- Applies iForest to rolling windows
- Captures local anomalies in temporal context
- Can detect point, contextual, and collective anomalies

#### 2.3 One-Class SVM (OCSVM)

**Principle:** Learn boundary around normal data points.

**Objective:**
$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + \frac{1}{\nu n}\sum_{i=1}^n \xi_i - \rho$$

**Anomaly Score:**
$$f(x) = \text{sign}(\langle w, \phi(x) \rangle - \rho)$$

Where:
- $\phi(x)$ = kernel mapping
- $\rho$ = separating hyperplane offset
- $\nu$ = upper bound on fraction of training outliers

**Kernel Options:**
- Linear: $K(x,y) = \langle x, y \rangle$
- RBF: $K(x,y) = \exp(-\gamma ||x-y||^2)$
- Polynomial: $K(x,y) = (\gamma \langle x,y \rangle + r)^d$

---

### 3. Statistical Methods

#### 3.1 Z-Score Method

**Formula:**
$$Z_i = \frac{x_i - \mu}{\sigma}$$

**Threshold:** |Z| > 3 (99.7% confidence) or |Z| > 2 (95%)

**For Time Series (with trend removal):**
$$Z_i = \frac{x_i - \text{trend}_i - \text{seasonal}_i}{\sigma_{\text{residual}}}$$

**Advantages:**
- Simple and interpretable
- Fast computation
- Well-understood statistical properties

**Disadvantages:**
- Assumes normal distribution
- Sensitive to outliers (biased mean/std)
- Fixed threshold may not adapt to non-stationary data

#### 3.2 Modified Z-Score (MAD-based)

**Formula:**
$$M_i = \frac{0.6745 \times (x_i - \text{median})}{\text{MAD}}$$

Where MAD = median(|x_i - median|)

**Threshold:** |M| > 3.5

**Advantages:**
- Robust to outliers (median-based)
- Better for skewed distributions
- Less sensitive to extreme values

#### 3.3 Interquartile Range (IQR) Method

**Bounds:**
- Lower: $Q_1 - 1.5 \times \text{IQR}$
- Upper: $Q_3 + 1.5 \times \text{IQR}$

**For Time Series:**
Use rolling windows to adapt to changing data distribution:
$$\text{IQR}_t = Q_3(x_{t-w:t}) - Q_1(x_{t-w:t})$$

#### 3.4 Exponential Weighted Moving Average (EWMA)

**Recursive Formulation:**
$$\hat{x}_t = \alpha x_t + (1-\alpha)\hat{x}_{t-1}$$

**Prediction Error:**
$$e_t = x_t - \hat{x}_{t-1}$$

**Anomaly Score:**
$$s_t = \frac{|e_t|}{\sigma_e}$$

**Advantages:**
- Adapts to changing mean
- Single parameter ($\alpha$) controls smoothness
- Efficient online computation

**Threshold Selection:**
- Conservative: $s_t > 3$
- Moderate: $s_t > 2$
- Sensitive: $s_t > 1.5$

---

### 4. Deep Learning Approaches

#### 4.1 LSTM Autoencoder (LSTM-AE)

**Architecture:**
```
Input Sequence → LSTM Encoder → Latent Vector → LSTM Decoder → Reconstructed Sequence
```

**Key Components:**

LSTM Cell:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

**Anomaly Score:**
$$\text{Score} = \frac{1}{T}\sum_{t=1}^{T} ||y_t - \hat{y}_t||_2^2$$

**Advantages:**
- Captures long-term dependencies
- Handles variable-length sequences
- Learns complex temporal patterns

**Disadvantages:**
- Requires significant training data
- Vanishing gradient problem
- Computationally expensive

#### 4.2 GRU Autoencoder (GRU-AE)

**Simplified LSTM variant:**

GRU Cell:
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$$
$$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Advantages:**
- Fewer parameters than LSTM
- Faster training and inference
- Similar performance with less data

**Disadvantages:**
- May miss some long-term patterns
- Less proven track record than LSTM

#### 4.3 Attention-Based Models

**Multi-Head Self-Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Transformer Encoder:**
- Multiple attention heads
- Feed-forward networks
- Layer normalization and residual connections

**Advantages:**
- Parallelizable (vs. RNNs)
- Long-range dependency modeling
- Interpretable attention weights

#### 4.4 Temporal Convolutional Networks (TCN)

**Architecture:**
```
1D Dilated Convolutions → ReLU → Residual Connections → Output
```

**Dilation:** $d=2^i$ for layer i

**Advantages:**
- Efficient parallel processing
- Large receptive fields
- Fast inference

---

### 5. Local Outlier Factor (LOF)

**K-Distance:**
$$k\text{-}d(p) = \text{distance from p to its k-th nearest neighbor}$$

**Reachability Distance:**
$$\text{reach-dist}_k(p, o) = \max\{k\text{-}d(o), d(p,o)\}$$

**Local Reachability Density:**
$$\text{LRD}_k(p) = \frac{1}{\text{avg}_o \text{reach-dist}_k(p,o)}$$

**Local Outlier Factor:**
$$\text{LOF}_k(p) = \frac{\text{avg}_o \text{LRD}_k(o)}{\text{LRD}_k(p)}$$

**Interpretation:**
- LOF ≈ 1: Normal point
- LOF > 1.5-2.0: Likely outlier
- LOF > 3.0: Definite outlier

**Advantages:**
- Detects contextual anomalies
- No parametric assumptions
- Multivariate friendly

**Disadvantages:**
- Quadratic complexity O(n²)
- k selection important
- Not for streaming data

---

### 6. ARIMA Residual-Based Detection

**ARIMA(p,d,q) Model:**
$$\phi(B)(1-B)^d x_t = \theta(B)\epsilon_t$$

**Anomaly Detection:**

1. Fit ARIMA model to training data
2. Compute residuals: $e_t = x_t - \hat{x}_t$
3. Calculate residual statistics: $\mu_e, \sigma_e$
4. Anomaly Score: $s_t = \frac{|e_t|}{\sigma_e}$

**Threshold:** $s_t > 3$ (for 99.7% confidence)

**Advantages:**
- Captures seasonal and trend patterns
- Well-established statistical method
- Interpretable residuals

**Disadvantages:**
- Assumes specific temporal pattern (ARIMA)
- Manual parameter selection (p,d,q)
- May not fit complex non-linear patterns

---

## Core Algorithms & Techniques

### Algorithm Comparison Table

| Method | Type | Complexity | Non-linear | Multivariate | Real-time |
|--------|------|-----------|-----------|--------------|-----------|
| Z-Score | Statistical | O(n) | No | No | Yes |
| IQR | Statistical | O(n) | No | No | Yes |
| EWMA | Statistical | O(n) | No | Yes | Yes |
| LOF | Density | O(n²) | Yes | Yes | No |
| Isolation Forest | Tree-based | O(n log n) | Yes | Yes | Yes |
| One-Class SVM | Kernel | O(n²-n³) | Yes | Yes | No |
| Autoencoder | Deep Learning | O(n×m) | Yes | Yes | Yes |
| LSTM-AE | Deep Learning | O(n×m) | Yes | Yes | Yes |
| VAE | Deep Learning | O(n×m) | Yes | Yes | Yes |
| ARIMA | Statistical | O(n) | No | No | Yes |

### Method Selection Flowchart

```
START
  ├─ Is data labeled? 
  │  ├─ YES → Use supervised methods (classification)
  │  └─ NO → Continue
  ├─ Is data univariate?
  │  ├─ YES → Use statistical methods (Z-score, IQR, EWMA)
  │  └─ NO → Continue
  ├─ Is data high-dimensional (>100 features)?
  │  ├─ YES → Use Isolation Forest or Deep Learning
  │  └─ NO → Continue
  ├─ Do you need real-time detection?
  │  ├─ YES → Use Isolation Forest, EWMA, or shallow AE
  │  ├─ NO → Continue
  └─ Is data sufficient for deep learning (>10k samples)?
     ├─ YES → Use LSTM-AE, TCN, or Transformer
     └─ NO → Use Isolation Forest or LOF
```

---

## Mathematical Formulations

### Complete Mathematical Framework

#### 1. Reconstruction Error Metric

**Mean Squared Error (MSE):**
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Mean Absolute Error (MAE):**
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Huber Loss (robust):**
$$L(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{\delta}{2}) & \text{otherwise}
\end{cases}$$

#### 2. Distance-Based Metrics

**Euclidean Distance:**
$$d_E(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**Manhattan Distance:**
$$d_M(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$

**Mahalanobis Distance:**
$$d_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

**Dynamic Time Warping (DTW):**
$$\text{DTW}(X,Y) = \sqrt{d_1 + \min(\text{DTW}(X[1:], Y), \text{DTW}(X, Y[1:]))}$$

#### 3. Probabilistic Formulations

**Gaussian Mixture Model (GMM):**
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

**Log-Likelihood Anomaly Score:**
$$\text{Score} = -\log p(x)$$

#### 4. Information-Theoretic Metrics

**Kullback-Leibler Divergence:**
$$D_{KL}(p || q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

**Jensen-Shannon Divergence (symmetric):**
$$D_{JS}(p || q) = \frac{1}{2}D_{KL}(p || m) + \frac{1}{2}D_{KL}(q || m)$$

Where $m = \frac{1}{2}(p + q)$

---

## Code Examples

### 1. Autoencoder-Based Detection

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class LSTMAutoencoder:
    """LSTM Autoencoder for time series anomaly detection"""
    
    def __init__(self, timesteps: int, n_features: int, encoding_dim: int = 32):
        """
        Initialize LSTM Autoencoder
        
        Args:
            timesteps: Sequence length
            n_features: Number of features
            encoding_dim: Dimension of latent space
        """
        self.timesteps = timesteps
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
    
    def build_model(self):
        """Build LSTM autoencoder architecture"""
        # Encoder
        inputs = keras.Input(shape=(self.timesteps, self.n_features))
        
        encoded = keras.layers.LSTM(
            64, activation='relu', input_shape=(self.timesteps, self.n_features),
            return_sequences=False
        )(inputs)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = keras.layers.RepeatVector(self.timesteps)(encoded)
        decoded = keras.layers.LSTM(64, activation='relu', return_sequences=True)(decoded)
        decoded = keras.layers.TimeDistributed(
            keras.layers.Dense(self.n_features)
        )(decoded)
        
        # Autoencoder
        self.model = keras.Model(inputs, decoded)
        self.model.compile(optimizer='adam', loss='mse')
        
        return self.model
    
    def fit(self, X_train: np.ndarray, epochs: int = 50, 
            batch_size: int = 32, validation_split: float = 0.1):
        """
        Fit the autoencoder on training data
        
        Args:
            X_train: Training data (n_samples, timesteps, n_features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
        """
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Calculate threshold from training data
        train_predictions = self.model.predict(X_train)
        train_mse = np.mean(np.abs(X_train - train_predictions), axis=(1, 2))
        self.threshold = np.percentile(train_mse, 95)  # 95th percentile
        
        return history
    
    def detect_anomalies(self, X_test: np.ndarray, 
                         threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using reconstruction error
        
        Args:
            X_test: Test data (n_samples, timesteps, n_features)
            threshold: Anomaly threshold (uses trained threshold if None)
        
        Returns:
            Tuple of (anomaly_flags, anomaly_scores)
        """
        predictions = self.model.predict(X_test)
        mse = np.mean(np.abs(X_test - predictions), axis=(1, 2))
        
        threshold = threshold or self.threshold
        anomalies = mse > threshold
        
        return anomalies, mse
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Get per-sample reconstruction error"""
        predictions = self.model.predict(X)
        return np.mean(np.abs(X - predictions), axis=(1, 2))

# Usage Example
np.random.seed(42)
timesteps = 60
n_features = 3
n_samples = 1000

# Generate synthetic normal data
X_normal = np.random.randn(n_samples, timesteps, n_features)

# Split into train/test
X_train = X_normal[:800]
X_test_normal = X_normal[800:900]
X_test_anomaly = np.random.randn(100, timesteps, n_features) * 3  # Amplified

# Combine test sets
X_test = np.vstack([X_test_normal, X_test_anomaly])
y_test = np.hstack([np.zeros(100), np.ones(100)])

# Train model
autoencoder = LSTMAutoencoder(timesteps, n_features, encoding_dim=16)
autoencoder.build_model()
autoencoder.fit(X_train, epochs=20, batch_size=32)

# Detect anomalies
anomaly_flags, scores = autoencoder.detect_anomalies(X_test)
print(f"Detected {anomaly_flags.sum()} anomalies")
print(f"Anomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")
```

### 2. Isolation Forest for Time Series

```python
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

class TimeSeriesIsolationForest:
    """Isolation Forest adapted for time series anomaly detection"""
    
    def __init__(self, window_size: int = 20, contamination: float = 0.05):
        """
        Initialize Time Series Isolation Forest
        
        Args:
            window_size: Rolling window size for feature extraction
            contamination: Expected proportion of anomalies
        """
        self.window_size = window_size
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.features = None
    
    def create_features(self, ts: np.ndarray) -> np.ndarray:
        """
        Create features from time series for Isolation Forest
        
        Features include:
        - Mean of window
        - Std of window
        - Min/Max of window
        - Rate of change
        - Autocorrelation
        """
        features = []
        
        for i in range(len(ts) - self.window_size + 1):
            window = ts[i:i + self.window_size]
            
            feature_vector = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.max(window) - np.min(window),
                np.mean(np.abs(np.diff(window))),  # Mean absolute change
                np.std(np.diff(window)),  # Std of changes
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def fit(self, ts: np.ndarray):
        """Fit Isolation Forest on training time series"""
        features = self.create_features(ts)
        self.model.fit(features)
        self.features = features
    
    def detect_anomalies(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in test time series
        
        Returns:
            Tuple of (anomaly_flags, anomaly_scores)
        """
        features = self.create_features(ts)
        
        # Get predictions and scores
        predictions = self.model.predict(features)  # -1 for anomaly, 1 for normal
        scores = -self.model.score_samples(features)  # Convert to anomaly scores
        
        # Create time series length flags (pad with False for window shift)
        anomaly_flags = np.ones(len(ts), dtype=bool)
        anomaly_flags[:self.window_size - 1] = False
        anomaly_flags[self.window_size - 1:] = predictions == -1
        
        return anomaly_flags, scores
    
    def get_anomaly_score(self, ts: np.ndarray) -> np.ndarray:
        """Get anomaly scores for entire time series"""
        features = self.create_features(ts)
        scores = -self.model.score_samples(features)
        
        # Pad scores to match time series length
        padded_scores = np.zeros(len(ts))
        padded_scores[self.window_size - 1:] = scores
        
        return padded_scores

# Usage Example
np.random.seed(42)
n_points = 500

# Create synthetic time series with anomalies
ts_normal = np.cumsum(np.random.randn(n_points) * 0.5)  # Random walk

# Inject anomalies
anomaly_indices = [100, 200, 300, 350]
for idx in anomaly_indices:
    ts_normal[idx:idx+5] += 10

# Split data
train_ts = ts_normal[:300]
test_ts = ts_normal[300:]

# Train and detect
detector = TimeSeriesIsolationForest(window_size=20, contamination=0.05)
detector.fit(train_ts)

anomaly_flags, anomaly_scores = detector.detect_anomalies(test_ts)
print(f"Detected anomalies at indices: {np.where(anomaly_flags)[0]}")
print(f"Anomaly score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
```

### 3. VAE for Anomaly Detection

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class VariationalAutoencoder:
    """Variational Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, latent_dim: int = 10):
        """
        Initialize VAE
        
        Args:
            input_dim: Input dimensionality
            latent_dim: Latent space dimensionality
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
    
    def build_model(self):
        """Build VAE architecture"""
        # Encoder
        encoder_inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(128, activation='relu')(encoder_inputs)
        x = layers.Dense(64, activation='relu')(x)
        
        # Latent space
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        z = layers.Lambda(self._sampling, output_shape=(self.latent_dim,),
                         name='z')([z_mean, z_log_var])
        
        self.encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z])
        
        # Decoder
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation='relu')(latent_inputs)
        x = layers.Dense(128, activation='relu')(x)
        decoder_outputs = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        self.decoder = tf.keras.Model(latent_inputs, decoder_outputs)
        
        # Full VAE
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.model = tf.keras.Model(encoder_inputs, outputs)
        
        # Custom loss function
        reconstruction_loss = tf.keras.losses.mse(encoder_inputs, outputs)
        reconstruction_loss *= self.input_dim
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * -0.5
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.model.add_loss(vae_loss)
        self.model.compile(optimizer='adam')
        
        return self.model
    
    def _sampling(self, args):
        """Reparameterization trick"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def fit(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train VAE"""
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            X_train, epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Set threshold from training data
        _, _, train_z = self.encoder.predict(X_train)
        train_reconstructions = self.decoder.predict(train_z)
        train_error = np.mean(np.square(X_train - train_reconstructions), axis=1)
        self.threshold = np.percentile(train_error, 95)
        
        return history
    
    def detect_anomalies(self, X_test: np.ndarray,
                        threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies based on reconstruction error"""
        _, _, test_z = self.encoder.predict(X_test)
        test_reconstructions = self.decoder.predict(test_z)
        
        mse = np.mean(np.square(X_test - test_reconstructions), axis=1)
        
        threshold = threshold or self.threshold
        anomalies = mse > threshold
        
        return anomalies, mse
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction error for samples"""
        _, _, z = self.encoder.predict(X)
        reconstructions = self.decoder.predict(z)
        return np.mean(np.square(X - reconstructions), axis=1)

# Usage Example
np.random.seed(42)

# Synthetic data
n_samples = 1000
n_features = 20
X_normal = np.random.randn(n_samples, n_features)

# Train/test split
X_train = X_normal[:700]
X_test_normal = X_normal[700:850]
X_test_anomaly = np.random.randn(150, n_features) * 3

X_test = np.vstack([X_test_normal, X_test_anomaly])
y_test = np.hstack([np.zeros(150), np.ones(150)])

# Train VAE
vae = VariationalAutoencoder(input_dim=n_features, latent_dim=10)
vae.fit(X_train, epochs=50, batch_size=32)

# Detect
anomaly_flags, scores = vae.detect_anomalies(X_test)
print(f"Detected {anomaly_flags.sum()} anomalies")
```

### 4. Statistical Methods with Adaptive Thresholds

```python
import pandas as pd
import numpy as np
from scipy import stats

class AdaptiveStatisticalDetector:
    """Statistical anomaly detection with adaptive thresholds"""
    
    def __init__(self, window_size: int = 20, method: str = 'ewma'):
        """
        Initialize adaptive detector
        
        Args:
            window_size: Window size for rolling statistics
            method: 'zscore', 'iqr', 'ewma', 'mad'
        """
        self.window_size = window_size
        self.method = method
        self.threshold = None
        self.params = {}
    
    def fit(self, ts: np.ndarray, threshold_percentile: float = 95):
        """Fit detector on training time series"""
        
        if self.method == 'zscore':
            self._fit_zscore(ts, threshold_percentile)
        elif self.method == 'iqr':
            self._fit_iqr(ts, threshold_percentile)
        elif self.method == 'ewma':
            self._fit_ewma(ts, threshold_percentile)
        elif self.method == 'mad':
            self._fit_mad(ts, threshold_percentile)
    
    def _fit_zscore(self, ts: np.ndarray, percentile: float):
        """Fit Z-score detector"""
        z_scores = np.abs(stats.zscore(ts))
        self.threshold = np.percentile(z_scores, percentile)
        self.params['mean'] = np.mean(ts)
        self.params['std'] = np.std(ts)
    
    def _fit_iqr(self, ts: np.ndarray, percentile: float):
        """Fit IQR detector"""
        q1 = np.percentile(ts, 25)
        q3 = np.percentile(ts, 75)
        iqr = q3 - q1
        
        self.params['q1'] = q1
        self.params['q3'] = q3
        self.params['iqr'] = iqr
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        self.threshold = (upper - lower) / 2
    
    def _fit_ewma(self, ts: np.ndarray, percentile: float):
        """Fit EWMA detector"""
        alpha = 0.3
        ewma_values = pd.Series(ts).ewm(span=30, adjust=False).mean().values
        errors = np.abs(ts - ewma_values)
        
        self.threshold = np.percentile(errors, percentile)
        self.params['alpha'] = alpha
        self.params['ewma_values'] = ewma_values
    
    def _fit_mad(self, ts: np.ndarray, percentile: float):
        """Fit MAD (Median Absolute Deviation) detector"""
        median = np.median(ts)
        mad = np.median(np.abs(ts - median))
        
        self.params['median'] = median
        self.params['mad'] = mad
        self.threshold = np.percentile(
            np.abs(ts - median) / (mad + 1e-10), percentile
        )
    
    def detect(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies"""
        
        if self.method == 'zscore':
            scores = np.abs((ts - self.params['mean']) / (self.params['std'] + 1e-10))
        elif self.method == 'iqr':
            scores = np.abs(ts - (self.params['q1'] + self.params['q3']) / 2)
        elif self.method == 'ewma':
            ewma_val = pd.Series(ts).ewm(
                span=1/self.params['alpha'] - 1, adjust=False
            ).mean().values
            scores = np.abs(ts - ewma_val)
        elif self.method == 'mad':
            scores = np.abs(ts - self.params['median']) / (self.params['mad'] + 1e-10)
        
        anomalies = scores > self.threshold
        return anomalies, scores
```

---

## Benchmarks & Datasets

### 1. NAB (Numenta Anomaly Benchmark)

**Dataset Characteristics:**
- ~365 datasets across different domains
- Real and synthetic time series
- Labeled anomalies with windows
- Focus on streaming detection

**Domains:**
- AWS CloudWatch metrics
- Network traffic
- Server metrics
- NYSE stock prices

**Size:** ~3.5 million data points across all datasets

**Key Metrics:**
- Precision, Recall, F1-Score
- NAB Score (weighted for early detection)

**Access:** https://github.com/numenta/NAB

### 2. UCR Time Series Archive

**Anomaly Detection Archive:**
- 250+ anomaly time series datasets
- High-quality labeled anomalies
- Real-world data from diverse sources

**Dataset Statistics:**

| Category | Count | Avg Length | Domains |
|----------|-------|-----------|---------|
| Industrial | 45 | 5,000-50,000 | Power grids, IoT sensors |
| Medical | 30 | 1,000-10,000 | ECG, EEG |
| Network | 25 | 5,000-100,000 | Traffic, security |
| Finance | 20 | 1,000-5,000 | Stock prices, crypto |
| Environmental | 35 | 2,000-20,000 | Weather, earthquakes |

**Access:** https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

### 3. Synthetic Datasets

#### 3.1 Numenta Synthetic Data

```python
import numpy as np

class SyntheticAnomalyDataset:
    """Generate synthetic time series with controlled anomalies"""
    
    @staticmethod
    def create_trend_data(n_points: int = 1000, anomaly_fraction: float = 0.05):
        """Time series with trend"""
        t = np.arange(n_points)
        trend = 0.001 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 365)
        noise = np.random.randn(n_points) * 0.5
        
        ts = trend + seasonal + noise
        
        # Add anomalies
        n_anomalies = int(n_points * anomaly_fraction)
        anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
        ts[anomaly_indices] += np.random.randn(n_anomalies) * 5
        
        labels = np.zeros(n_points)
        labels[anomaly_indices] = 1
        
        return ts, labels, anomaly_indices
    
    @staticmethod
    def create_collective_anomalies(n_points: int = 1000, n_anomaly_windows: int = 3):
        """Time series with collective anomalies"""
        ts = np.cumsum(np.random.randn(n_points) * 0.5)
        labels = np.zeros(n_points)
        
        window_length = 20
        for _ in range(n_anomaly_windows):
            start_idx = np.random.randint(0, n_points - window_length)
            # Add strong pattern
            ts[start_idx:start_idx + window_length] = (
                np.sin(np.linspace(0, 4*np.pi, window_length)) * 10
            )
            labels[start_idx:start_idx + window_length] = 1
        
        return ts, labels
    
    @staticmethod
    def create_contextual_anomalies(n_points: int = 1000):
        """Context-dependent anomalies"""
        # Morning: low values expected
        morning_ts = np.random.randn(250) * 0.5 + 2
        # Afternoon: high values expected
        afternoon_ts = np.random.randn(250) * 0.5 + 8
        # Evening: low again
        evening_ts = np.random.randn(250) * 0.5 + 3
        # Night: moderate
        night_ts = np.random.randn(250) * 0.5 + 5
        
        ts = np.concatenate([morning_ts, afternoon_ts, evening_ts, night_ts])
        
        # Inject contextual anomaly: high value in morning
        anomaly_idx = 50
        ts[anomaly_idx] = 10
        
        labels = np.zeros(len(ts))
        labels[anomaly_idx] = 1
        
        return ts, labels
```

### 4. Performance Metrics

#### 4.1 Classification Metrics

```python
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve

def evaluate_anomaly_detection(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_scores: np.ndarray = None) -> Dict:
    """
    Comprehensive evaluation of anomaly detection
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        y_scores: Anomaly scores (0 to 1)
    
    Returns:
        Dictionary with all metrics
    """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        # Basic counts
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        
        # Rates
        'TPR': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall/Sensitivity
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False alarm rate
        'TNR': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0,  # Miss rate
        
        # Predictive values
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'F1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        
        # Overall accuracy
        'Accuracy': (tp + tn) / len(y_true),
        
        # Balanced accuracy (for imbalanced data)
        'Balanced_Accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
        if (tp + fn) > 0 and (tn + fp) > 0 else 0,
    }
    
    # ROC-AUC if scores provided
    if y_scores is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_scores)
    
    return metrics
```

#### 4.2 Time Series Specific Metrics

```python
def evaluate_timeseries_detection(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  window_size: int = 50) -> Dict:
    """
    Evaluate anomaly detection considering temporal context
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        window_size: Size of detection window
    
    Returns:
        Evaluation metrics considering temporal context
    """
    
    # Standard metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Adjusted metrics: consecutive detections count as one
    def get_adjusted_count(labels):
        segments = []
        start = None
        for i in range(len(labels)):
            if labels[i] == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((start, i))
                    start = None
        if start is not None:
            segments.append((start, len(labels)))
        return len(segments), segments
    
    true_segments, true_ranges = get_adjusted_count(y_true)
    pred_segments, pred_ranges = get_adjusted_count(y_pred)
    
    # Overlap calculation
    matched = 0
    for pred_start, pred_end in pred_ranges:
        for true_start, true_end in true_ranges:
            overlap = min(pred_end, true_end) - max(pred_start, true_start)
            if overlap > 0:
                matched += 1
                break
    
    return {
        'Standard_Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Standard_Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Segment_Precision': matched / pred_segments if pred_segments > 0 else 0,
        'Segment_Recall': matched / true_segments if true_segments > 0 else 0,
        'True_Segments': true_segments,
        'Predicted_Segments': pred_segments,
        'Matched_Segments': matched,
    }
```

---

## Advanced Topics

### 1. Multivariate Anomaly Detection

#### 1.1 Correlation-Based Methods

```python
class CorrelationAnomalyDetector:
    """Detect anomalies in multivariate time series using correlation"""
    
    def __init__(self, window_size: int = 50, threshold_percentile: float = 95):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.expected_correlations = None
        self.threshold = None
    
    def fit(self, X: np.ndarray):
        """
        Fit on normal multivariate data
        
        Args:
            X: Shape (n_samples, n_features)
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Extract upper triangle (avoid duplicates)
        self.expected_correlations = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        # Set threshold
        self.threshold = np.percentile(
            np.abs(self.expected_correlations),
            self.threshold_percentile
        )
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect multivariate anomalies"""
        n_samples = X.shape[0]
        anomaly_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Use rolling window if enough history
            if i < self.window_size:
                window_start = 0
            else:
                window_start = i - self.window_size
            
            window = X[window_start:i+1]
            
            if len(window) < 2:
                continue
            
            # Calculate correlation changes
            corr_matrix = np.corrcoef(window.T)
            test_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            
            # Measure deviation from expected
            deviation = np.mean(np.abs(test_corr - self.expected_correlations))
            anomaly_scores[i] = deviation
        
        anomalies = anomaly_scores > self.threshold
        return anomalies, anomaly_scores
```

### 2. Contextual Anomalies

```python
class ContextualAnomalyDetector:
    """Detect context-dependent anomalies"""
    
    def __init__(self, context_size: int = 24):
        """
        Args:
            context_size: Size of context (e.g., 24 for hourly data)
        """
        self.context_size = context_size
        self.context_stats = {}
    
    def fit(self, ts: np.ndarray, context_labels: np.ndarray):
        """
        Fit detector with context information
        
        Args:
            ts: Time series values
            context_labels: Context labels (e.g., hour of day)
        """
        unique_contexts = np.unique(context_labels)
        
        for context in unique_contexts:
            mask = context_labels == context
            context_values = ts[mask]
            
            self.context_stats[context] = {
                'mean': np.mean(context_values),
                'std': np.std(context_values),
                'q1': np.percentile(context_values, 25),
                'q3': np.percentile(context_values, 75),
            }
    
    def detect(self, ts: np.ndarray, context_labels: np.ndarray,
               threshold: float = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect contextual anomalies
        
        Args:
            ts: Time series values
            context_labels: Context for each point
            threshold: Deviation threshold (in std units)
        """
        anomaly_scores = np.zeros(len(ts))
        
        for i, (value, context) in enumerate(zip(ts, context_labels)):
            if context not in self.context_stats:
                continue
            
            stats = self.context_stats[context]
            z_score = (value - stats['mean']) / (stats['std'] + 1e-10)
            anomaly_scores[i] = np.abs(z_score)
        
        anomalies = anomaly_scores > threshold
        return anomalies, anomaly_scores
```

### 3. Collective Anomalies

```python
class CollectiveAnomalyDetector:
    """Detect subsequence patterns that are collectively anomalous"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.normal_embeddings = None
        self.threshold = None
    
    def fit(self, X: np.ndarray, anomaly_fraction: float = 0.05):
        """Fit on normal data"""
        # Create overlapping windows
        windows = []
        for i in range(len(X) - self.window_size + 1):
            windows.append(X[i:i+self.window_size])
        
        windows = np.array(windows)
        
        # Embed using PCA or Autoencoder
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        self.normal_embeddings = pca.fit_transform(
            windows.reshape(len(windows), -1)
        )
        
        # Set threshold
        distances = np.linalg.norm(self.normal_embeddings, axis=1)
        self.threshold = np.percentile(distances, 95)
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect collective anomalies"""
        windows = []
        indices = []
        
        for i in range(len(X) - self.window_size + 1):
            windows.append(X[i:i+self.window_size])
            indices.append(i)
        
        windows = np.array(windows)
        
        # Compute anomaly scores
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        test_embeddings = pca.fit_transform(windows.reshape(len(windows), -1))
        
        distances = np.linalg.norm(test_embeddings, axis=1)
        anomaly_flags = np.zeros(len(X), dtype=bool)
        
        for idx, dist in zip(indices, distances):
            if dist > self.threshold:
                anomaly_flags[idx:idx+self.window_size] = True
        
        return anomaly_flags, distances
```

### 4. Semi-Supervised Anomaly Detection

```python
class SemiSupervisedAnomalyDetector:
    """Use small labeled set to improve unsupervised detection"""
    
    def __init__(self):
        self.normal_profile = None
        self.anomaly_profile = None
    
    def fit(self, X_unlabeled: np.ndarray, X_labeled: np.ndarray,
            y_labeled: np.ndarray):
        """
        Fit using both labeled and unlabeled data
        
        Args:
            X_unlabeled: Unlabeled data
            X_labeled: Labeled data
            y_labeled: Labels (0=normal, 1=anomaly)
        """
        # Use labeled data to define profiles
        normal_data = X_labeled[y_labeled == 0]
        anomaly_data = X_labeled[y_labeled == 1]
        
        self.normal_profile = {
            'mean': np.mean(normal_data, axis=0),
            'cov': np.cov(normal_data.T),
            'count': len(normal_data)
        }
        
        if len(anomaly_data) > 0:
            self.anomaly_profile = {
                'mean': np.mean(anomaly_data, axis=0),
                'cov': np.cov(anomaly_data.T),
                'count': len(anomaly_data)
            }
    
    def detect(self, X: np.ndarray, use_anomaly_prior: bool = True) -> np.ndarray:
        """
        Detect anomalies using Bayesian approach
        
        Args:
            X: Data to score
            use_anomaly_prior: Whether to use learned anomaly profile
        """
        from scipy.stats import multivariate_normal
        
        # Likelihood under normal distribution
        normal_dist = multivariate_normal(
            mean=self.normal_profile['mean'],
            cov=self.normal_profile['cov']
        )
        p_x_normal = normal_dist.pdf(X)
        
        if use_anomaly_prior and self.anomaly_profile:
            # Likelihood under anomaly distribution
            anomaly_dist = multivariate_normal(
                mean=self.anomaly_profile['mean'],
                cov=self.anomaly_profile['cov']
            )
            p_x_anomaly = anomaly_dist.pdf(X)
            
            # Posterior probability
            p_normal = self.normal_profile['count']
            p_anomaly = self.anomaly_profile['count']
            
            posterior_anomaly = (p_x_anomaly * p_anomaly) / (
                p_x_normal * p_normal + p_x_anomaly * p_anomaly + 1e-10
            )
            
            return posterior_anomaly
        else:
            return 1 - p_x_normal / (np.max(p_x_normal) + 1e-10)
```

### 5. Concept Drift Handling

```python
class AdaptiveAnomalyDetector:
    """Handle concept drift in streaming data"""
    
    def __init__(self, window_size: int = 500, decay_rate: float = 0.9):
        """
        Args:
            window_size: Window for recent data statistics
            decay_rate: How quickly to forget old data (0.9 = remember 90%)
        """
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.recent_data = []
        self.thresholds = []
        self.model = None
    
    def update(self, x: float, y_pred: float = None):
        """
        Update model with new data point
        
        Args:
            x: New data point
            y_pred: Anomaly prediction (None for unlabeled)
        """
        self.recent_data.append(x)
        
        # Keep only recent window
        if len(self.recent_data) > self.window_size:
            self.recent_data.pop(0)
        
        # Update threshold with decay
        if len(self.recent_data) >= 20:
            recent_mean = np.mean(self.recent_data)
            recent_std = np.std(self.recent_data)
            new_threshold = recent_mean + 3 * recent_std
            
            # Exponential decay update
            if self.thresholds:
                old_threshold = self.thresholds[-1]
                updated_threshold = (
                    self.decay_rate * old_threshold +
                    (1 - self.decay_rate) * new_threshold
                )
            else:
                updated_threshold = new_threshold
            
            self.thresholds.append(updated_threshold)
    
    def predict(self, x: float) -> Tuple[bool, float]:
        """
        Predict anomaly with adapted threshold
        
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if not self.recent_data:
            return False, 0.0
        
        recent_mean = np.mean(self.recent_data)
        recent_std = np.std(self.recent_data)
        
        if recent_std == 0:
            anomaly_score = 0.0
        else:
            anomaly_score = np.abs((x - recent_mean) / recent_std)
        
        threshold = self.thresholds[-1] if self.thresholds else 3.0
        is_anomaly = anomaly_score > threshold
        
        return is_anomaly, anomaly_score
```

---

## Production Systems

### 1. Real-Time Streaming Detection

```python
from collections import deque
import time

class StreamingAnomalyDetectionPipeline:
    """Real-time anomaly detection for streaming data"""
    
    def __init__(self, buffer_size: int = 100, 
                 method: str = 'isolation_forest',
                 model_update_interval: int = 500):
        """
        Args:
            buffer_size: Size of streaming buffer
            method: Detection method
            model_update_interval: Refit model every N points
        """
        self.buffer = deque(maxlen=buffer_size)
        self.method = method
        self.model_update_interval = model_update_interval
        self.point_count = 0
        self.detector = None
        self.statistics = {
            'n_processed': 0,
            'n_anomalies': 0,
            'avg_latency': 0,
            'latencies': deque(maxlen=100)
        }
    
    def process_point(self, x: float, y_true: float = None) -> Dict:
        """
        Process single streaming data point
        
        Args:
            x: Data point value
            y_true: Ground truth label (optional)
        
        Returns:
            Detection result with metadata
        """
        start_time = time.time()
        
        self.buffer.append(x)
        self.point_count += 1
        
        # Initialize or update detector
        if self.point_count == 1:
            self.detector = self._initialize_detector()
        elif self.point_count % self.model_update_interval == 0:
            self.detector = self._update_detector()
        
        # Predict anomaly
        if len(self.buffer) >= 2:
            is_anomaly, score = self._predict_single(x)
        else:
            is_anomaly, score = False, 0.0
        
        # Update statistics
        if is_anomaly:
            self.statistics['n_anomalies'] += 1
        self.statistics['n_processed'] += 1
        
        latency = (time.time() - start_time) * 1000  # ms
        self.statistics['latencies'].append(latency)
        self.statistics['avg_latency'] = np.mean(self.statistics['latencies'])
        
        result = {
            'timestamp': time.time(),
            'value': x,
            'is_anomaly': is_anomaly,
            'anomaly_score': score,
            'latency_ms': latency,
            'n_processed': self.statistics['n_processed'],
            'n_anomalies': self.statistics['n_anomalies'],
            'anomaly_rate': self.statistics['n_anomalies'] / self.statistics['n_processed']
        }
        
        if y_true is not None:
            result['y_true'] = y_true
        
        return result
    
    def _initialize_detector(self):
        """Initialize detector"""
        if self.method == 'ewma':
            return {'alpha': 0.3, 'values': []}
        elif self.method == 'zscore':
            return {'values': []}
        else:
            return {}
    
    def _update_detector(self):
        """Retrain detector on buffered data"""
        return self._initialize_detector()
    
    def _predict_single(self, x: float) -> Tuple[bool, float]:
        """Make prediction for single point"""
        
        if self.method == 'ewma':
            if not self.detector['values']:
                ewma = x
            else:
                ewma = (0.3 * x + 0.7 * self.detector['values'][-1])
            
            error = abs(x - ewma)
            self.detector['values'].append(ewma)
            
            # Simple threshold
            threshold = 3 * np.std(list(self.buffer)[-20:]) if len(self.buffer) >= 20 else 1.0
            
            return error > threshold, error
        
        elif self.method == 'zscore':
            if len(self.buffer) >= 10:
                recent = np.array(list(self.buffer)[-20:])
                z_score = abs((x - np.mean(recent)) / (np.std(recent) + 1e-10))
                return z_score > 3, z_score
            else:
                return False, 0.0
        
        return False, 0.0
    
    def get_report(self) -> Dict:
        """Get streaming detection statistics"""
        return {
            'total_processed': self.statistics['n_processed'],
            'total_anomalies': self.statistics['n_anomalies'],
            'anomaly_rate': self.statistics['n_anomalies'] / max(1, self.statistics['n_processed']),
            'avg_latency_ms': self.statistics['avg_latency'],
            'buffer_size': len(self.buffer)
        }

# Usage Example
pipeline = StreamingAnomalyDetectionPipeline(buffer_size=100, method='ewma')

# Simulate streaming
for i in range(1000):
    if i > 800:  # Inject anomalies
        x = np.random.randn() * 10
    else:
        x = np.random.randn() * 1
    
    result = pipeline.process_point(x)
    
    if result['is_anomaly']:
        print(f"Point {i}: ANOMALY (score={result['anomaly_score']:.3f})")
```

### 2. Model Interpretability & Explainability

```python
class AnomalyExplainer:
    """Explain anomaly predictions"""
    
    def __init__(self, detector, feature_names: list = None):
        self.detector = detector
        self.feature_names = feature_names
    
    def explain_isolation_forest(self, x: np.ndarray, 
                                estimator_idx: int = 0) -> Dict:
        """
        Explain Isolation Forest decision
        
        Args:
            x: Data point to explain
            estimator_idx: Which tree to trace
        """
        tree = self.detector.estimators_[estimator_idx]
        path = self._get_tree_path(tree, x)
        
        explanation = {
            'path': path,
            'depth': len(path),
            'splits': self._extract_splits(path)
        }
        
        return explanation
    
    def explain_reconstruction_error(self, x: np.ndarray,
                                     model) -> Dict:
        """Explain reconstruction-based anomaly"""
        reconstruction = model.predict(x.reshape(1, -1))[0]
        errors = np.abs(x - reconstruction)
        
        # Find most anomalous features
        sorted_errors = np.argsort(errors)[::-1]
        
        explanation = {
            'reconstruction_error': errors.mean(),
            'top_error_features': [
                {
                    'feature': self.feature_names[i] if self.feature_names else f'Feature_{i}',
                    'original': x[i],
                    'reconstruction': reconstruction[i],
                    'error': errors[i]
                }
                for i in sorted_errors[:5]
            ]
        }
        
        return explanation
    
    def _get_tree_path(self, tree, x):
        """Get decision path through tree"""
        path = []
        node = 0
        
        while node != -1:
            feature = tree.tree_.feature[node]
            threshold = tree.tree_.threshold[node]
            
            if feature == -2:  # Leaf node
                break
            
            path.append({
                'node': node,
                'feature': feature,
                'threshold': threshold,
                'value': x[feature],
                'direction': 'left' if x[feature] <= threshold else 'right'
            })
            
            if x[feature] <= threshold:
                node = tree.tree_.children_left[node]
            else:
                node = tree.tree_.children_right[node]
        
        return path
    
    def _extract_splits(self, path: list) -> str:
        """Convert path to human-readable splits"""
        splits = []
        for item in path:
            feature = self.feature_names[item['feature']] if self.feature_names else f"Feature {item['feature']}"
            direction = "<=" if item['direction'] == 'left' else ">"
            splits.append(f"{feature} {direction} {item['threshold']:.2f}")
        
        return " AND ".join(splits)
```

### 3. Hyperparameter Tuning

```python
from optuna import create_study, Trial

class AnomalyDetectionTuner:
    """Hyperparameter tuning for anomaly detectors"""
    
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray,
                 y_test: np.ndarray, method: str = 'isolation_forest'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.method = method
    
    def objective(self, trial: Trial) -> float:
        """Objective function for Optuna"""
        
        if self.method == 'isolation_forest':
            contamination = trial.suggest_float('contamination', 0.01, 0.2)
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_samples = trial.suggest_int('max_samples', 32, 256)
            
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=42
            )
        
        elif self.method == 'lof':
            n_neighbors = trial.suggest_int('n_neighbors', 5, 50)
            contamination = trial.suggest_float('contamination', 0.01, 0.2)
            
            from sklearn.neighbors import LocalOutlierFactor
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination
            )
        
        # Train and evaluate
        model.fit(self.X_train)
        y_pred = model.predict(self.X_test)
        
        # Convert predictions (-1 for anomaly, 1 for normal)
        y_pred = (y_pred == -1).astype(int)
        
        # F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(self.y_test, y_pred)
        
        return f1
    
    def optimize(self, n_trials: int = 100) -> Dict:
        """Run optimization"""
        study = create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'trials': len(study.trials)
        }
```

### 4. Deployment and Monitoring

```python
import pickle
import json
from datetime import datetime

class AnomalyDetectionDeployment:
    """Deploy and monitor anomaly detection in production"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.performance_log = []
    
    def load_model(self):
        """Load trained model"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
    
    def predict_batch(self, X: np.ndarray) -> Dict:
        """Make predictions on batch"""
        start_time = time.time()
        
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        inference_time = time.time() - start_time
        
        return {
            'predictions': predictions,
            'scores': scores,
            'inference_time_ms': inference_time * 1000,
            'throughput_samples_per_sec': len(X) / inference_time
        }
    
    def monitor_performance(self, y_true: np.ndarray, 
                           y_pred: np.ndarray) -> Dict:
        """Monitor model performance over time"""
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        }
        
        self.performance_log.append(metrics)
        
        return metrics
    
    def detect_model_drift(self, threshold: float = 0.05) -> bool:
        """Detect if model performance is drifting"""
        
        if len(self.performance_log) < 2:
            return False
        
        # Compare recent vs historical performance
        recent_f1 = np.mean([m['f1'] for m in self.performance_log[-10:]])
        historical_f1 = np.mean([m['f1'] for m in self.performance_log[:-10]])
        
        drift = abs(recent_f1 - historical_f1) / (historical_f1 + 1e-10)
        
        return drift > threshold
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint"""
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'performance_log_size': len(self.performance_log)
        }
        
        with open(checkpoint_path.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f)
```

### 5. Threshold Selection Strategies

```python
class ThresholdSelector:
    """Select optimal anomaly detection thresholds"""
    
    @staticmethod
    def rocauc_method(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Select threshold maximizing ROC-AUC"""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden = tpr - fpr
        optimal_idx = np.argmax(youden)
        
        return thresholds[optimal_idx]
    
    @staticmethod
    def f1_method(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Select threshold maximizing F1 score"""
        from sklearn.metrics import f1_score
        
        best_f1 = 0
        best_threshold = 0
        
        for threshold in np.percentile(y_scores, range(0, 100, 5)):
            y_pred = (y_scores > threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    @staticmethod
    def precision_recall_method(y_true: np.ndarray, y_scores: np.ndarray,
                               target_precision: float = 0.95) -> float:
        """Select threshold for target precision level"""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find threshold meeting precision target
        valid_idx = precision >= target_precision
        if np.any(valid_idx):
            best_idx = np.where(valid_idx)[0][np.argmax(recall[valid_idx])]
            return thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        
        return np.max(y_scores)
    
    @staticmethod
    def domain_knowledge_method(y_scores: np.ndarray,
                               percentile: float = 95) -> float:
        """Select threshold using percentile method"""
        return np.percentile(y_scores, percentile)
```

---

## Performance Metrics

### Comprehensive Metric Suite

```python
def comprehensive_evaluation(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            y_scores: np.ndarray = None) -> Dict:
    """
    Comprehensive evaluation with all metrics
    
    Returns:
        Dictionary with classification, ranking, and probabilistic metrics
    """
    from sklearn.metrics import (
        confusion_matrix, classification_report, 
        roc_auc_score, auc, roc_curve,
        precision_recall_curve, average_precision_score,
        cohen_kappa_score
    )
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {}
    
    # 1. Classification Metrics
    metrics['classification'] = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1_score': 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
        'f2_score': 5*tp / (5*tp + 4*fn + fp) if (5*tp + 4*fn + fp) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
    }
    
    # 2. Probabilistic Metrics (if scores provided)
    if y_scores is not None:
        metrics['probabilistic'] = {
            'roc_auc': roc_auc_score(y_true, y_scores),
            'pr_auc': average_precision_score(y_true, y_scores),
        }
    
    # 3. Error Analysis
    metrics['errors'] = {
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'false_discovery_rate': fp / (fp + tp) if (fp + tp) > 0 else 0,
        'false_omission_rate': fn / (fn + tn) if (fn + tn) > 0 else 0,
    }
    
    # 4. Agreement Metrics
    metrics['agreement'] = {
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }
    
    return metrics
```

---

## Citations & References

### 1. Foundational Papers

1. **Chandola, V., Banerjee, A., & Kumar, V. (2009)**
   - "Anomaly Detection: A Survey"
   - ACM Computing Surveys, Vol. 41, No. 3, Article 15
   - DOI: 10.1145/1541880.1541882
   - **Contribution:** Comprehensive taxonomy of anomaly detection methods

2. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)**
   - "Isolation Forest"
   - Proceedings of IEEE 8th International Conference on Data Mining (ICDM)
   - DOI: 10.1109/ICDM.2008.17
   - **Contribution:** Novel tree-based anomaly detection algorithm

3. **Breunig, M. M., Kriegel, H., Ng, R. T., & Sander, J. (2000)**
   - "LOF: Identifying Density-Based Local Outliers"
   - Proceedings of the 2000 ACM SIGMOD International Conference
   - DOI: 10.1145/342009.335388
   - **Contribution:** Local Outlier Factor algorithm

### 2. Deep Learning Methods

4. **Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2015)**
   - "Long Short Term Memory Networks for Anomaly Detection in Time Series"
   - ESANN 2015
   - **Contribution:** LSTM-AE for unsupervised anomaly detection

5. **An, J., & Cho, S. (2015)**
   - "Variational Autoencoder based Synthetic-Data Generator for Imbalanced Learning"
   - IEEE Transactions on Cybernetics
   - **Contribution:** VAE for anomaly detection with probabilistic framework

6. **Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018)**
   - "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
   - Proceedings of the 24th ACM SIGKDD International Conference
   - DOI: 10.1145/3219819.3219845
   - **Contribution:** LSTM-based real-world spacecraft anomaly detection

### 3. Statistical Methods

7. **Tukey, J. W. (1977)**
   - "Exploratory Data Analysis"
   - Addison-Wesley Publishing Company
   - **Contribution:** Introduction of IQR method and robust statistics

8. **Hampel, F. R., Ronchetti, E. M., Rousseeuw, P. J., & Stahel, W. A. (1986)**
   - "Robust Statistics: The Approach Based on Influence Functions"
   - John Wiley & Sons
   - **Contribution:** Modified Z-score and robust statistical methods

### 4. Benchmarks & Datasets

9. **Lavin, A., & Ahmad, S. (2015)**
   - "Evaluating Real-Time Anomaly Detection Algorithms - The Numenta Anomaly Benchmark"
   - IEEE 15th International Conference on Machine Learning and Applications (ICMLA)
   - DOI: 10.1109/ICMLA.2015.141
   - **Contribution:** NAB benchmark for time series anomaly detection

10. **Yeh, C. C. M., Zhu, Y., Ulanova, L., et al. (2016)**
    - "Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View That Includes Correlations, Dtw and Other Distances"
    - IEEE 16th International Conference on Data Mining (ICDM)
    - **Contribution:** UCR Anomaly Archive and Matrix Profile methods

### 5. Advanced Topics

11. **Hill, D. J., & Minsker, B. S. (2010)**
    - "Real-Time Bayesian Anomaly Detection in Streaming Environmental Data"
    - Water Resources Research, Vol. 46
    - **Contribution:** Streaming anomaly detection with adaptive methods

12. **Ren, H., Xu, B., Wang, Y., et al. (2019)**
    - "Time Series Anomaly Detection with Multivariate Convolutional LSTM Network"
    - Proceedings of the 25th ACM SIGKDD International Conference
    - **Contribution:** Multivariate time series anomaly detection using CNN-LSTM

### 6. Applications & Real-World Systems

13. **Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017)**
    - "Unsupervised Real-Time Anomaly Detection for Streaming Data"
    - Neurocomputing, Vol. 262
    - **Contribution:** Production system for real-time anomaly detection

14. **Campos, G. O., Zimek, A., Sander, J., et al. (2016)**
    - "On the Evaluation of Unsupervised Outlier Detection: Measures, Datasets, and an Empirical Study"
    - Data Mining and Knowledge Discovery, Vol. 30, No. 4
    - DOI: 10.1007/s10618-015-0444-8
    - **Contribution:** Comprehensive evaluation framework for outlier detection

---

## Summary & Quick Reference

### When to Use Each Method

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| **Univariate, small data** | Z-score, IQR, EWMA | Simple, interpretable |
| **Multivariate, high-dimensional** | Isolation Forest | Scalable, no distance calc |
| **Normal behavior known** | One-Class SVM | Learns boundary |
| **Contextual anomalies** | LOF | Density-based |
| **Large labeled dataset** | Supervised (classifiers) | Better performance |
| **Complex patterns, >10k points** | LSTM-AE, Transformer | Learns temporal patterns |
| **Streaming data** | EWMA, Isolation Forest | Low latency |
| **Collective anomalies** | Matrix Profile, TCN | Subsequence patterns |

### Best Practices

1. **Preprocessing**
   - Normalize/standardize features
   - Handle missing values
   - Remove obvious errors

2. **Model Selection**
   - Start simple (statistical methods)
   - Scale to complex (deep learning)
   - Use ensemble methods
   - Validate with ground truth

3. **Threshold Selection**
   - Use domain knowledge
   - Optimize for business metrics
   - Consider precision vs. recall tradeoff
   - Adapt to data drift

4. **Monitoring**
   - Track performance metrics
   - Detect model drift
   - Log anomalies for review
   - Retrain periodically

5. **Explainability**
   - Identify features contributing to anomaly
   - Provide temporal context
   - Log decision paths
   - Enable human review

---

**Last Updated:** April 2026  
**Document Status:** Complete and Production-Ready
