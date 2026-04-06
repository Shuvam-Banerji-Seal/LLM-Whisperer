# Synthetic Data Generation: Privacy-Preserving Techniques and Implementation

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Category:** Dataset Engineering & Privacy

## 1. Overview and Importance

Synthetic data generation creates artificial datasets that mimic the statistical properties of real data while preserving privacy. This enables training models without exposing sensitive information and solving the problem of limited data.

### Key Applications

- **Privacy Protection:** Generate data for public sharing without exposing individuals
- **Data Augmentation:** Create additional training examples
- **Benchmark Creation:** Develop datasets for algorithm testing
- **Imbalanced Data:** Generate examples of rare classes
- **Regulatory Compliance:** GDPR, CCPA compliance without data deletion

### Privacy-Utility Trade-off

The fundamental challenge: balance between privacy protection and synthetic data utility.

## 2. Classical Synthetic Data Methods

### 2.1 Gaussian Mixture Models (GMM)

```python
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd

class GMMSyntheticDataGeneration:
    """Generate synthetic data using Gaussian Mixture Models."""
    
    def __init__(self, n_components=5):
        """
        Initialize GMM-based synthetic data generator.
        
        Mathematical Concept:
        GMM models the data as a mixture of K Gaussian distributions:
        p(x) = Σ π_k * N(x | μ_k, Σ_k)
        where π_k are mixture weights, μ_k are means, Σ_k are covariances
        """
        self.n_components = n_components
        self.model = None
    
    def fit(self, X):
        """Fit GMM to real data."""
        self.model = GaussianMixture(n_components=self.n_components, random_state=42)
        self.model.fit(X)
        return self
    
    def generate(self, n_samples=1000):
        """Generate synthetic samples."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        X_synthetic, _ = self.model.sample(n_samples)
        return X_synthetic
    
    def get_model_parameters(self):
        """Return learned model parameters."""
        return {
            'weights': self.model.weights_,
            'means': self.model.means_,
            'covariances': self.model.covariances_
        }

# Example
X_real = np.random.randn(500, 5)
gmm_gen = GMMSyntheticDataGeneration(n_components=3)
gmm_gen.fit(X_real)
X_synthetic = gmm_gen.generate(n_samples=1000)

print(f"Real data shape: {X_real.shape}")
print(f"Synthetic data shape: {X_synthetic.shape}")
print(f"Real data mean: {X_real.mean(axis=0)}")
print(f"Synthetic data mean: {X_synthetic.mean(axis=0)}")
```

### 2.2 Copula Models

```python
class CopulaBasedSynthetic:
    """Generate synthetic data preserving multivariate dependencies."""
    
    @staticmethod
    def generate_copula_synthetic(X, n_samples=1000):
        """
        Copula-based generation: Separates marginal distributions from dependence.
        
        Steps:
        1. Fit marginal distributions to each column
        2. Fit copula to uniform transforms
        3. Sample from copula and inverse-transform
        """
        from scipy.stats import gaussian_kde
        from scipy.stats import norm
        
        n_features = X.shape[1]
        
        # Fit marginal distributions (using KDE)
        marginals = []
        for i in range(n_features):
            kde = gaussian_kde(X[:, i])
            marginals.append(kde)
        
        # Compute ranks (uniform transforms)
        ranks = np.argsort(np.argsort(X, axis=0), axis=0)
        uniform_X = ranks / len(X)
        
        # Fit multivariate Gaussian to uniform data
        mean_uniform = uniform_X.mean(axis=0)
        cov_uniform = np.cov(uniform_X.T)
        
        # Generate synthetic samples
        n_synthetic = n_samples
        uniform_synthetic = np.random.multivariate_normal(
            mean=mean_uniform,
            cov=cov_uniform,
            size=n_synthetic
        )
        
        # Inverse-transform to original scale
        X_synthetic = np.zeros((n_synthetic, n_features))
        
        for i in range(n_features):
            # Approximate inverse CDF using quantiles
            quantiles = np.quantile(X[:, i], uniform_synthetic[:, i])
            X_synthetic[:, i] = quantiles
        
        return X_synthetic

# Example
X_real = np.random.randn(200, 3)
X_synthetic = CopulaBasedSynthetic.generate_copula_synthetic(X_real, n_samples=500)
print(f"Copula-based synthetic data shape: {X_synthetic.shape}")
```

## 3. Privacy-Preserving Synthetic Data: Differential Privacy

### 3.1 Differential Privacy Concepts

```python
class DifferentialPrivacySynthetic:
    """
    Generate synthetic data with differential privacy guarantees.
    
    Differential Privacy Definition:
    A mechanism M is (ε, δ)-differentially private if for any two neighboring
    datasets D and D' differing in one record:
    
    P(M(D) ∈ S) ≤ e^ε * P(M(D') ∈ S) + δ
    
    Smaller ε = stronger privacy, larger ε = less privacy
    Typical ε values: 0.1 (strong), 1.0 (moderate), 10.0 (weak)
    """
    
    @staticmethod
    def laplace_mechanism(true_value, sensitivity, epsilon):
        """
        Laplace mechanism: Add Laplace noise to statistic.
        
        Mathematical: noisy_value = true_value + Lap(0, sensitivity/ε)
        where Lap(0, b) ~ Laplace distribution with scale b
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise
    
    @staticmethod
    def exponential_mechanism(scores, sensitivity, epsilon):
        """
        Exponential mechanism: Probabilistically choose best item.
        Useful for discrete choices.
        
        Mathematical: P(choosing i) ∝ exp(ε * score_i / (2 * sensitivity))
        """
        scaled_scores = epsilon * scores / (2 * sensitivity)
        # Numerical stability: subtract max
        scaled_scores = scaled_scores - np.max(scaled_scores)
        
        probabilities = np.exp(scaled_scores)
        probabilities /= probabilities.sum()
        
        return np.random.choice(len(scores), p=probabilities)
    
    @staticmethod
    def generate_dp_synthetic(X, epsilon=1.0, delta=1e-5):
        """
        Generate differentially private synthetic data.
        
        Algorithm: DP-Synthesis (10-step)
        1. Compute marginal distributions with noise
        2. Compute conditional distributions with noise
        3. Sample from noisy distributions
        """
        n_features = X.shape[1]
        
        # Sensitivity = 1 for proportions
        sensitivity = 1.0
        
        # Add noise to marginals
        X_synthetic_params = {}
        
        for i in range(n_features):
            column = X[:, i]
            unique_vals = np.unique(column)
            
            true_proportions = np.array([
                (column == val).sum() / len(column) for val in unique_vals
            ])
            
            # Add Laplace noise
            noisy_proportions = np.array([
                DifferentialPrivacySynthetic.laplace_mechanism(p, sensitivity, epsilon)
                for p in true_proportions
            ])
            
            # Ensure valid probabilities
            noisy_proportions = np.clip(noisy_proportions, 0, 1)
            noisy_proportions /= noisy_proportions.sum()
            
            X_synthetic_params[i] = {
                'values': unique_vals,
                'probabilities': noisy_proportions
            }
        
        return X_synthetic_params

# Example
X_real = np.array([[0, 1], [1, 0], [0, 1], [1, 1], [0, 0]] * 100)
params = DifferentialPrivacySynthetic.generate_dp_synthetic(X_real, epsilon=1.0)

print("Differentially private synthetic data parameters:")
for feature, param in params.items():
    print(f"Feature {feature}: {param['probabilities']}")
```

## 4. Generative Adversarial Networks (GANs)

### 4.1 Simple GAN Architecture

```python
class GANSynthetic:
    """
    Generate synthetic data using GANs.
    
    GAN Components:
    1. Generator: G(z) produces fake samples from noise z
    2. Discriminator: D(x) distinguishes real from fake
    
    Objective (Min-Max Game):
    min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
    """
    
    @staticmethod
    def create_simple_gan_tensorflow(input_dim, latent_dim=100):
        """
        Create a simple GAN using TensorFlow.
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Generator
            generator = keras.Sequential([
                layers.Dense(128, activation='relu', input_dim=latent_dim),
                layers.BatchNormalization(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(input_dim, activation='tanh')
            ])
            
            # Discriminator
            discriminator = keras.Sequential([
                layers.Dense(512, activation='relu', input_dim=input_dim),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
            
            return generator, discriminator
        except ImportError:
            print("Install TensorFlow: pip install tensorflow")
            return None, None
    
    @staticmethod
    def generate_from_gan(generator, n_samples, latent_dim=100):
        """Generate synthetic samples from trained generator."""
        noise = np.random.normal(0, 1, (n_samples, latent_dim))
        synthetic = generator.predict(noise, verbose=0)
        return synthetic

# Example (conceptual - requires training loop)
# latent_dim = 100
# input_dim = 10
# generator, discriminator = GANSynthetic.create_simple_gan_tensorflow(
#     input_dim=input_dim,
#     latent_dim=latent_dim
# )
```

## 5. Variational Autoencoders (VAE)

### 5.1 VAE for Synthetic Data

```python
class VAESynthetic:
    """
    Generate synthetic data using Variational Autoencoders.
    
    VAE Components:
    1. Encoder: Maps data x to latent distribution q(z|x)
    2. Decoder: Maps latent z back to data p(x|z)
    
    Objective (ELBO):
    L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    """
    
    @staticmethod
    def create_vae_tensorflow(input_dim, latent_dim=10):
        """Create VAE using TensorFlow."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Encoder
            encoder_input = layers.Input(shape=(input_dim,))
            encoded = layers.Dense(128, activation='relu')(encoder_input)
            encoded = layers.Dense(64, activation='relu')(encoded)
            
            # Latent space
            z_mean = layers.Dense(latent_dim, name='z_mean')(encoded)
            z_log_var = layers.Dense(latent_dim, name='z_log_var')(encoded)
            
            # Sampling layer
            def sampling(args):
                z_mean, z_log_var = args
                epsilon = tf.random.normal(tf.shape(z_mean))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
            
            # Decoder
            decoded = layers.Dense(64, activation='relu')(z)
            decoded = layers.Dense(128, activation='relu')(decoded)
            decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
            
            vae = keras.Model(encoder_input, decoded)
            
            return vae, z_mean, z_log_var, z
        except ImportError:
            print("Install TensorFlow: pip install tensorflow")
            return None, None, None, None
    
    @staticmethod
    def generate_from_vae(latent_dim=10, n_samples=1000):
        """Generate synthetic samples from VAE latent space."""
        # Sample from standard normal
        z = np.random.normal(0, 1, (n_samples, latent_dim))
        return z

# Example (conceptual)
# vae, z_mean, z_log_var, z = VAESynthetic.create_vae_tensorflow(
#     input_dim=20,
#     latent_dim=10
# )
```

## 6. Synthetic Data Quality Assessment

### 6.1 Utility and Privacy Metrics

```python
class SyntheticDataEvaluation:
    """
    Evaluate synthetic data quality.
    
    Framework: SynthEval (Lautrup et al., 2024)
    - Privacy metrics: Protection against membership inference
    - Utility metrics: Statistical similarity and ML utility
    """
    
    @staticmethod
    def calculate_statistical_similarity(X_real, X_synthetic, metrics=['mean', 'std', 'correlation']):
        """
        Measure statistical similarity between real and synthetic data.
        """
        results = {}
        
        if 'mean' in metrics:
            real_mean = X_real.mean(axis=0)
            synth_mean = X_synthetic.mean(axis=0)
            results['mean_mse'] = np.mean((real_mean - synth_mean) ** 2)
        
        if 'std' in metrics:
            real_std = X_real.std(axis=0)
            synth_std = X_synthetic.std(axis=0)
            results['std_mse'] = np.mean((real_std - synth_std) ** 2)
        
        if 'correlation' in metrics:
            real_corr = np.corrcoef(X_real.T)
            synth_corr = np.corrcoef(X_synthetic.T)
            results['correlation_mse'] = np.mean((real_corr - synth_corr) ** 2)
        
        return results
    
    @staticmethod
    def ml_utility_test(X_real, y_real, X_synthetic, y_synthetic):
        """
        Test ML utility: Train on synthetic, test on real.
        Measures how well synthetic data trains models.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        
        # Train on synthetic
        model = LogisticRegression(random_state=42)
        model.fit(X_synthetic, y_synthetic)
        
        # Test on real
        y_pred = model.predict(X_real)
        
        return {
            'accuracy': accuracy_score(y_real, y_pred),
            'f1': f1_score(y_real, y_pred)
        }
    
    @staticmethod
    def membership_inference_attack(X_real, X_synthetic, membership_size=100):
        """
        Assess privacy: Can we infer membership from synthetic data?
        Lower success rate = better privacy.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest synthetic samples for real data
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_synthetic)
        distances, indices = nn.kneighbors(X_real)
        
        # Estimate membership based on distance threshold
        threshold = np.percentile(distances, 75)
        inferred_members = distances.flatten() < threshold
        
        true_members = np.arange(len(X_real)) < membership_size
        
        # Attack success = correct inferences
        accuracy = np.mean(inferred_members == true_members)
        
        return {
            'attack_accuracy': accuracy,
            'threshold': threshold,
            'privacy_score': 1 - accuracy  # Higher = more private
        }

# Example
X_real = np.random.randn(500, 10)
y_real = np.random.randint(0, 2, 500)

# Generate synthetic (placeholder)
X_synthetic = np.random.randn(500, 10)
y_synthetic = np.random.randint(0, 2, 500)

evaluator = SyntheticDataEvaluation()

# Utility assessment
utility = evaluator.ml_utility_test(X_real, y_real, X_synthetic, y_synthetic)
print(f"ML Utility - Accuracy: {utility['accuracy']:.3f}, F1: {utility['f1']:.3f}")

# Statistical similarity
similarity = evaluator.calculate_statistical_similarity(X_real, X_synthetic)
print(f"Statistical Similarity - Mean MSE: {similarity['mean_mse']:.3f}")

# Privacy assessment
privacy = evaluator.membership_inference_attack(X_real, X_synthetic)
print(f"Privacy Score: {privacy['privacy_score']:.3f}")
```

## 7. Quality Checklist

### Synthetic Data Generation Best Practices
- [ ] Define privacy-utility requirements
- [ ] Choose appropriate generation method
- [ ] Evaluate statistical properties
- [ ] Assess ML utility
- [ ] Measure privacy protection
- [ ] Document generation process
- [ ] Validate synthetic data quality
- [ ] Test on downstream tasks
- [ ] Document limitations
- [ ] Create audit trail

## 8. Authoritative Sources

1. Lautrup, A. D., et al. (2024). "SynthEval: A Framework for Utility and Privacy Evaluation." *Data Mining and Knowledge Discovery*
2. Hermsen, F., & Mandal, A. (2024). "Privacy and Utility Evaluation of Synthetic Tabular Data." Fraunhofer FIT
3. Sarkar, A., et al. (2025). "Synthetic Data: Revisiting the Privacy-Utility Trade-off." *arXiv:2407.07926*
4. Padariya, D., & Wagner, I. (2025). "Privacy-Preserving Generative Models: Comprehensive Survey." arXiv
5. "Systematic Review of Generative Modelling Tools" - ACM Computing Surveys (2024)
6. Goodfellow, I., et al. (2014). "Generative Adversarial Nets." *NeurIPS*

---

**Citation Format:**
Banerji Seal, S. (2026). "Synthetic Data Generation: Privacy-Preserving Techniques and Implementation." LLM-Whisperer Skills Library.

**Version:** 1.0  
**Status:** Production Ready
