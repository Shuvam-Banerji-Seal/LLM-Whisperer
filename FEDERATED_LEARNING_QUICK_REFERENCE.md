# Federated Learning: Implementation Quick Reference

## Quick-Start Implementations

### 1. Basic FedAvg in 100 Lines

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleFedAvg:
    def __init__(self, model, num_clients=10, num_rounds=100):
        self.global_model = model
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.client_data = [None] * num_clients
    
    def aggregate(self, client_models, client_sizes):
        """Weighted averaging of client models"""
        total = sum(client_sizes)
        with torch.no_grad():
            for param in self.global_model.parameters():
                param.zero_()
            
            for w, client_model in enumerate(client_models):
                weight = client_sizes[w] / total
                for p, param in enumerate(self.global_model.parameters()):
                    param.data += weight * list(client_model.parameters())[p].data
    
    def federated_train(self, global_epochs=100, local_epochs=1, batch_size=32):
        """Main federated training loop"""
        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for round_num in range(global_epochs):
            client_models = []
            client_sizes = []
            
            # Sample and train clients
            for client_id in range(min(3, self.num_clients)):
                model = self._create_client_model()
                
                # Local training
                for _ in range(local_epochs):
                    for X_batch, y_batch in self.get_client_data(client_id, batch_size):
                        output = model(X_batch)
                        loss = criterion(output, y_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                client_models.append(model)
                client_sizes.append(len(self.client_data[client_id]))
            
            # Aggregation
            self.aggregate(client_models, client_sizes)
            
            if round_num % 10 == 0:
                print(f"Round {round_num}: Training complete")
    
    def _create_client_model(self):
        """Create new client model with global weights"""
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        model.load_state_dict(self.global_model.state_dict())
        return model
    
    def get_client_data(self, client_id, batch_size):
        """Get batches for client"""
        dataset = TensorDataset(
            torch.randn(100, 784),
            torch.randint(0, 10, (100,))
        )
        return DataLoader(dataset, batch_size=batch_size)
```

### 2. Differential Privacy-SGD

```python
import torch
import torch.nn.functional as F

def dp_sgd_step(model, loss, learning_rate, noise_scale, clip_norm=1.0):
    """Single DP-SGD update with gradient clipping and noise"""
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    
    # Add Gaussian noise
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_scale * clip_norm
            param.grad.add_(noise)
    
    # Optimization step
    for param in model.parameters():
        if param.grad is not None:
            param.data -= learning_rate * param.grad
            param.grad.zero_()

# Privacy budget computation
def compute_noise_scale(epsilon, delta, epochs, batch_size, dataset_size):
    """Compute noise scale for target (epsilon, delta) privacy"""
    sampling_rate = batch_size / dataset_size
    steps = epochs * (dataset_size // batch_size)
    noise = 2.0 * torch.sqrt(torch.tensor(2.0 * torch.log(torch.tensor(1.25/delta)) * steps)) / epsilon
    return float(noise / sampling_rate)

# Example
epsilon = 1.0  # Very strong privacy
delta = 1e-5
noise_scale = compute_noise_scale(epsilon, delta, epochs=10, batch_size=32, dataset_size=10000)
print(f"Required noise scale: {noise_scale:.4f}")
```

### 3. Byzantine-Robust Aggregation

```python
import numpy as np
import torch

def krum_aggregation(updates, num_byzantine=1):
    """Krum aggregation: robust to Byzantine clients"""
    K = len(updates)
    f = num_byzantine
    
    # Vectorize updates
    vectors = [torch.cat([v.flatten() for v in u.values()]) for u in updates]
    vectors = torch.stack(vectors)
    
    # Compute distances
    distances = torch.cdist(vectors, vectors)
    
    # Find update with smallest sum of distances to K-f-2 neighbors
    min_score = float('inf')
    selected_idx = 0
    
    for i in range(K):
        # Sum of K-f-2 smallest distances
        score = torch.topk(distances[i], K - f - 1, largest=False)[0].sum()
        if score < min_score:
            min_score = score
            selected_idx = i
    
    return updates[selected_idx]

def coordinate_median_aggregation(updates):
    """Coordinate-wise median: simpler Byzantine-robust method"""
    keys = updates[0].keys()
    result = {}
    
    for key in keys:
        values = torch.stack([u[key] for u in updates])
        result[key] = torch.median(values, dim=0)[0]
    
    return result

# Example usage
updates = [
    {'weight': torch.randn(10, 10) * 0.1},
    {'weight': torch.randn(10, 10) * 0.1},
    {'weight': torch.randn(10, 10) * 100},  # Byzantine
]

krum = krum_aggregation(updates, num_byzantine=1)
median = coordinate_median_aggregation(updates)
```

### 4. Client Sampling Strategies

```python
import numpy as np

class ClientSamplingStrategy:
    def __init__(self, num_clients, sampling_fraction=0.1):
        self.num_clients = num_clients
        self.sampling_fraction = sampling_fraction
        self.num_samples = max(1, int(sampling_fraction * num_clients))
    
    def uniform_sampling(self):
        """Sample clients uniformly at random"""
        return np.random.choice(
            self.num_clients,
            size=self.num_samples,
            replace=False
        )
    
    def biased_sampling(self, data_sizes):
        """Sample proportional to local data size"""
        probabilities = np.array(data_sizes) / sum(data_sizes)
        return np.random.choice(
            self.num_clients,
            size=self.num_samples,
            replace=False,
            p=probabilities
        )
    
    def stragggler_aware(self, device_speeds):
        """Prioritize faster devices"""
        speeds = np.array(device_speeds)
        speeds = speeds / speeds.sum()
        return np.random.choice(
            self.num_clients,
            size=self.num_samples,
            replace=False,
            p=speeds
        )
    
    def power_of_two_choices(self):
        """FedCS: Compare two random clients, pick faster"""
        sampled = set()
        while len(sampled) < self.num_samples:
            candidates = np.random.choice(self.num_clients, 2, replace=True)
            # Pick faster one (simplified)
            sampled.add(candidates[0])
        return list(sampled)

# Example
sampler = ClientSamplingStrategy(num_clients=100, sampling_fraction=0.1)
print("Uniform:", sampler.uniform_sampling())
print("Power of 2:", sampler.power_of_two_choices())
```

### 5. Non-IID Data Distribution

```python
import numpy as np
from scipy.stats import dirichlet

def create_noniid_distribution(num_classes, num_clients, alpha=0.1):
    """
    Create non-IID (Dirichlet) data distribution
    Lower alpha = higher heterogeneity
    """
    # Dirichlet distribution for label probabilities
    class_distribution = dirichlet.rvs([alpha] * num_classes, size=num_clients)
    return class_distribution

def assign_noniid_data(num_samples_per_client, class_distribution):
    """
    Assign samples according to class distribution
    """
    num_clients, num_classes = class_distribution.shape
    client_data = []
    
    for client_id in range(num_clients):
        # Sample labels for this client from Dirichlet
        client_labels = np.random.choice(
            num_classes,
            size=num_samples_per_client,
            p=class_distribution[client_id]
        )
        client_data.append(client_labels)
    
    return client_data

def measure_heterogeneity(client_data):
    """Measure how non-IID the data is"""
    client_data = np.array(client_data)
    
    # Compute label distribution per client
    client_distributions = []
    for samples in client_data:
        dist = np.bincount(samples) / len(samples)
        client_distributions.append(dist)
    
    # KL divergence from uniform distribution
    uniform = np.ones(len(client_distributions[0])) / len(client_distributions[0])
    kl_divs = [
        np.sum(dist * np.log(dist / uniform + 1e-10))
        for dist in client_distributions
    ]
    
    return {
        'mean_kl_divergence': np.mean(kl_divs),
        'max_kl_divergence': np.max(kl_divs),
        'label_distributions': client_distributions
    }

# Example
alpha = 0.1  # High heterogeneity
distributions = create_noniid_distribution(num_classes=10, num_clients=100, alpha=alpha)
data = assign_noniid_data(num_samples_per_client=100, class_distribution=distributions)
het = measure_heterogeneity(data)
print(f"Heterogeneity metrics: {het}")
```

## Performance Benchmarking

### Convergence Metrics

```python
class ConvergenceBenchmark:
    def __init__(self, test_set):
        self.test_set = test_set
        self.history = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'communication_cost': [],
            'time': []
        }
    
    def evaluate_round(self, model, round_num, bytes_sent=0, elapsed_time=0):
        """Evaluate model performance for a round"""
        predictions = model(self.test_set['X'])
        accuracy = (predictions.argmax(1) == self.test_set['y']).float().mean()
        
        loss = self.compute_loss(predictions, self.test_set['y'])
        
        self.history['round'].append(round_num)
        self.history['accuracy'].append(accuracy.item())
        self.history['loss'].append(loss.item())
        self.history['communication_cost'].append(bytes_sent)
        self.history['time'].append(elapsed_time)
        
        return {'accuracy': accuracy.item(), 'loss': loss.item()}
    
    def compute_loss(self, pred, target):
        return torch.nn.functional.cross_entropy(pred, target)
    
    def time_to_accuracy(self, target_accuracy=0.9):
        """Find round and time when target accuracy reached"""
        accs = self.history['accuracy']
        for i, acc in enumerate(accs):
            if acc >= target_accuracy:
                return {
                    'round': self.history['round'][i],
                    'time_seconds': self.history['time'][i],
                    'communication_bytes': self.history['communication_cost'][i]
                }
        return None
    
    def communication_efficiency(self):
        """Accuracy per MB of communication"""
        final_accuracy = self.history['accuracy'][-1]
        total_bytes = self.history['communication_cost'][-1]
        mb = total_bytes / (1024 * 1024)
        return final_accuracy / mb if mb > 0 else 0
```

## Framework Integration Snippets

### TensorFlow Federated Integration

```python
import tensorflow as tf
import tensorflow_federated as tff

# Simple model
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

# Create federated training process
process = tff.learning.build_federated_averaging_process(
    model_fn=create_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0)
)

# Initialize and run
state = process.initialize()
for round in range(10):
    client_data = [create_client_dataset(i) for i in range(10)]
    state, metrics = process.next(state, client_data)
    print(f'Round {round}: {metrics}')
```

### FLOWER Framework Integration

```python
from flwr.client import NumPyClient, start_numpy_client
from flwr.server import start_server

class MyClient(NumPyClient):
    def __init__(self, model):
        self.model = model
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # Set parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
        
        # Train
        for _ in range(5):
            train_epoch(self.model)
        
        return self.get_parameters(config), len(train_data), {}
    
    def evaluate(self, parameters, config):
        # Similar to fit but for evaluation
        return loss, len(test_data), {"accuracy": accuracy}

if __name__ == "__main__":
    start_numpy_client(
        server_address="localhost:8080",
        client=MyClient(model)
    )
```

## Troubleshooting Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| Diverging loss | Learning rate too high | Reduce lr to 0.01-0.001 |
| No convergence | Non-IID data too extreme | Use FedProx with μ>0 |
| Memory OOM | Batch size too large | Use gradient accumulation |
| Stragglers | Unbalanced system heterogeneity | Use asynchronous aggregation |
| Privacy budget exhausted | Too many rounds | Allocate larger epsilon or fewer rounds |
| Byzantine clients | Model corruption | Switch to Krum/Median aggregation |

