# Federated Learning: Comprehensive Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Federated Learning Fundamentals](#federated-learning-fundamentals)
3. [Advanced Techniques](#advanced-techniques)
4. [Privacy & Security](#privacy--security)
5. [Frameworks & Implementation](#frameworks--implementation)
6. [Applications & Benchmarks](#applications--benchmarks)
7. [Mathematical Formulations](#mathematical-formulations)
8. [Code Examples](#code-examples)
9. [References](#references)

---

## Introduction

**Federated Learning (FL)** is a machine learning paradigm that enables training models across decentralized data sources without centralizing data. Instead of sending raw data to a central server, FL sends model updates (gradients) to a central aggregator, which combines them to improve the global model. This approach addresses:

- **Privacy**: Raw data never leaves the device
- **Bandwidth**: Only model updates are transmitted
- **Regulation Compliance**: GDPR, HIPAA, CCPA adherence
- **Latency**: Computation occurs at the edge

### Key Characteristics

| Aspect | Traditional ML | Federated Learning |
|--------|----------------|-------------------|
| Data Location | Centralized | Distributed |
| Privacy | Lower | Higher (with DP) |
| Bandwidth | High | Low |
| Latency | Low (inference) | Potentially higher |
| Communication | Minimal | High (mitigated with compression) |

---

## Federated Learning Fundamentals

### 1. Federated Averaging (FedAvg)

The **FedAvg algorithm**, introduced by McMahan et al. (2017), is the foundational algorithm in federated learning. It enables training on heterogeneous data distributed across multiple devices.

#### Algorithm Overview

**FedAvg** operates in synchronous rounds:

```
Algorithm 1: Federated Averaging (FedAvg)
─────────────────────────────────────────
Input: Number of rounds T, devices K, local epochs E, learning rate η
Initialize: Server model weights w₀

for round t = 1 to T do
    S_t ← Sample fraction C of devices uniformly at random
    
    for each device k ∈ S_t (in parallel) do
        w^k_{t+1} ← ClientUpdate(k, w_t)  // Local training
    
    w_{t+1} ← (1/|S_t|) Σ_{k∈S_t} n_k/n · w^k_{t+1}  // Aggregate

return w_T
```

#### Mathematical Formulation

**Objective Function:**
```
minimize   f(w) = Σ_{k=1}^K (n_k/n) · f_k(w)
where      f_k(w) = (1/n_k) Σ_{i∈D_k} ℓ(w; x_i, y_i)
```

- `K`: Total number of clients
- `n_k`: Number of samples on client k
- `n`: Total samples across all clients
- `D_k`: Local dataset on client k
- `ℓ`: Loss function

**Update Rule (Weighted Averaging):**
```
w_{t+1} = Σ_{k=1}^K (n_k/n) · w_k^{t+1}
```

**Client-side Update:**
```
For each local epoch e = 1 to E:
    For each batch b in client k's data:
        w_k^e ← w_k^{e-1} - η∇ℓ_b(w_k^{e-1})

return w_k^E
```

#### Key Insights

1. **Weighted Aggregation**: Accounts for heterogeneous dataset sizes
2. **Multiple Local Epochs**: Reduces communication rounds (E > 1)
3. **Sampling**: Clients sampled at random each round (typically C = 0.1)
4. **Stale Synchronization**: Tolerates slow clients using local updates

### 2. Communication-Efficient Learning

Communication is the bottleneck in federated learning. Strategies to reduce communication:

#### A. Gradient Compression

**Top-K Sparsification:**
```
compress(g) = {
    top_k ← indices of largest |g_i|
    return g[top_k] with top_k stored separately
}
```

**Reduction Factor:** (1 - k/d) × 100%, where d is gradient dimension

#### B. Quantization

**Uniform Quantization:**
```
quantize(x) = round(x / Δ) · Δ
where Δ = (max - min) / (2^b)
and b = number of bits
```

**Communication Savings:**
- Float32 (32 bits) → Int8 (8 bits) = 4× compression
- With sparsification: up to 1000× possible

#### C. Sketching and Sampling

**Iterative Averaging with Compression (ITAVG):**
```
w_t ← w_{t-1} - η · aggregate(compress(∇f_k(w_{t-1})))
```

### 3. Convergence Analysis and Theory

#### Convergence Rate Results

**Theorem (McMahan et al., 2017):** Under standard assumptions:

```
Convergence Rate: O(1/T) for convex functions
                  O(1/√T) for non-convex functions

Final Error Bound:
E[f(w_T)] - f(w*) ≤ O(1/T) + O(σ²) + O(δ²)

where:
  T: number of rounds
  σ²: variance from data heterogeneity
  δ²: convergence error due to local SGD
```

**Impact of Heterogeneity:**

Data heterogeneity significantly affects convergence. Three types:

1. **Feature Skew**: Different features across clients
2. **Label Skew**: Imbalanced class distribution (non-IID)
3. **Quantity Skew**: Unequal data volume per client

**Non-IID Data Impact:**
```
Convergence degradation factor ≈ 1 + (σ_heterogeneity / σ_iid)²
```

#### Convergence Comparison Table

| Algorithm | IID Data | Non-IID Data | Communication |
|-----------|----------|--------------|----------------|
| FedAvg    | O(1/T)   | O(1/√T) + ϵ  | ~100 rounds   |
| FedProx   | O(1/T)   | O(1/√T)      | ~100 rounds   |
| FedDyn    | O(1/T)   | O(1/T)       | ~50 rounds    |
| FedPAQ    | -        | O(1/√T)      | ~10 rounds    |

### 4. Privacy Guarantees and Differential Privacy

#### Differential Privacy Fundamentals

**ε-Differential Privacy Definition:**
```
An algorithm M is ε-differentially private if for all adjacent datasets
D and D' (differing in one record), and for all possible outputs S:

P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S)
```

**Intuition:** The algorithm's output distribution should be nearly identical
whether a specific record is included or not.

#### Privacy Guarantees in FL

**Privacy Amplification by Sampling:**
```
If each round samples C fraction of clients uniformly:
  ε_total ≤ Σ_{t=1}^T ε_t · √(2 · ln(1.25/δ) · C · T)

where ε_t is the per-round privacy loss
```

**Typical Privacy Budgets:**
- Healthcare: ε ≤ 1 (strict privacy)
- Finance: ε ≤ 5-10 (moderate privacy)
- Research: ε ≤ 100+ (weaker privacy)

### 5. Heterogeneous Data and Systems

#### Data Heterogeneity Challenges

**Non-IID (Non-Independent and Identically Distributed) Data:**

```
IID Assumption Violation Metrics:

1. Gini Coefficient:
   G = 1 - (2/n(n-1)) · Σ_{i<j} |X_i - X_j|

2. Dirichlet Distribution Parameter α:
   Lower α → Higher heterogeneity
   α = 0.1: Highly non-IID
   α = 1.0: Moderately non-IID
   α = 10.0: Near-IID

3. Label Skew Ratio:
   S = max_k(class_count_k) / min_k(class_count_k)
   S > 100: Extreme skew
```

#### System Heterogeneity

**Stragglers Problem:**
```
Synchronization Time = max(T_1, T_2, ..., T_K)
  where T_k = local_training_time_k + network_latency_k

Wasted Time = Σ_{k=1}^K max(0, max_time - T_k)
Efficiency = Total_Useful_Time / (Total_Useful_Time + Wasted_Time)
```

**Solutions:**
1. **Asynchronous Aggregation**: Don't wait for slow clients
2. **Client Deadline**: Set time limits for local training
3. **Adaptive Client Selection**: Prioritize faster clients

---

## Advanced Techniques

### 1. Personalized Federated Learning

Standard FL learns a single global model, but clients may benefit from personalized models.

#### Piecewise Averaging (PA)

Clients can have different parts of the shared model:

```
Algorithm: Personalized Federated Averaging (pFedAvg)
─────────────────────────────────────────────────
Maintain:
  - Global model parameters: w^global (shared)
  - Personal model parameters: w^k (client-specific)

Update rule:
  w^k_{t+1} = (1 - α) · w^k_t + α · w^global_t - η·∇f_k(w^k_t)
  
where α ∈ [0, 1] balances personalization vs. generalization
```

#### Federated Multi-Task Learning (FMTL)

Each client optimizes a slightly different task:

```
Objective: min Σ_{k=1}^K f_k(w_k)
Subject to: w_k ≈ w̄ (regularization toward shared model)

Regularized formulation:
  f_k(w_k) = ℓ_k(w_k) + λ · ||w_k - w̄||²
  
where w̄ = (1/K) Σ_k w_k
```

### 2. Multi-Task Federated Learning

**Per-Client Task Modeling:**

```
Global objectives per task:
  Minimize: F_m(W) = Σ_{k=1}^K (n_{k,m}/n_m) · f_{k,m}(w_{k,m})

Multi-task model:
  w_{k,m} = w_shared + w_{k,m}^personal
  
The shared model captures common patterns across all tasks
The personal models capture task-specific patterns
```

**Convergence with Multi-Task:**
```
Expected Error: O(1/T) + O(√(d/T)) + O(heterogeneity_term)
where d is the dimension of the shared space
```

### 3. Federated Meta-Learning

Meta-learning learns to learn faster with limited data updates.

#### MAML in Federated Settings (FedMAML)

```
Algorithm: Federated Meta-Learning
────────────────────────────────────
Meta-training loop:
  for round t = 1 to T:
      for each sampled client k:
          // Inner loop: adapt to local task
          w_k^{inner} ← w_t - α · ∇f_k(w_t)
          
          // Compute meta-gradient
          meta_grad_k ← ∇f_k(w_k^{inner})
      
      // Outer loop: meta-update (aggregation)
      w_{t+1} ← w_t - β · (1/|batch|) Σ_k meta_grad_k

Benefits:
- Faster adaptation to local data
- Better generalization with few samples
- Reduced communication rounds
```

### 4. Asynchronous Aggregation

Addresses stragglers by not waiting for all clients:

```
Algorithm: Asynchronous Federated Averaging
─────────────────────────────────────────────

Server: 
  w_t ← weighted aggregation of all received updates
  Send w_t to next batch of clients

Client k:
  Receive w_{t_k} (possibly stale)
  Perform local training: E epochs
  Send update back

Staleness-aware update:
  w_{t+1} ← w_t + (1/K) Σ_k α_{t,k} · (w_k^{new} - w_k^{old})
  
  where α_{t,k} accounts for staleness of update k
```

**Staleness Decay Function:**
```
α(τ) = e^(-c·τ)  or  α(τ) = 1/(1 + c·τ)

where τ is number of rounds since update was computed
c is decay constant
```

### 5. Byzantine-Robust Aggregation

Defends against malicious or faulty clients sending corrupted updates.

#### Krum Aggregation

```
Algorithm: Krum (Byzantine-Robust Aggregation)
───────────────────────────────────────────────

for each parameter dimension:
    1. Calculate pairwise distances: d_ij = ||w_i - w_j||
    2. For each w_i, sum its (K-f-2) nearest neighbors
       score_i = Σ_(j in nearest K-f-2) d_ij
    3. Select w_krum with minimum score
    
This selects the update closest to (K-f-2) others,
eliminating outliers from f Byzantine clients
```

#### Median Aggregation

```
Algorithm: Coordinate-wise Median
──────────────────────────────────

For each parameter dimension d:
    values_d = [w_1[d], w_2[d], ..., w_K[d]]
    w_new[d] = median(values_d)
    
Robustness:
  - Tolerates ⌊(K-1)/2⌋ Byzantine clients
  - Slower convergence than averaging
  - More expensive computation
```

#### Multi-Krum

```
Selects multiple (m) trusted updates before averaging:

scores = [score_1, score_2, ..., score_K]
top_m_indices = argsort(scores)[:m]
w_{t+1} = (1/m) Σ_{i in top_m_indices} w_i
```

---

## Privacy & Security

### 1. Differential Privacy in FL

#### Local Differential Privacy

Clients add noise locally before sending updates:

```
Algorithm: Federated Learning with Local DP
─────────────────────────────────────────────

for each client k:
    1. Compute gradient: g_k = ∇f_k(w_t)
    
    2. Clip gradient: ḡ_k = g_k / max(1, ||g_k||/C)
       (prevents large gradients from leaking information)
    
    3. Add Gaussian noise:
       g_k^{noisy} = ḡ_k + noise ~ N(0, σ² · C² · I)
    
    4. Send: g_k^{noisy}

Server aggregates noisy updates: w_{t+1} ← aggregate(g_k^{noisy})

Privacy cost per round:
  Δε ≈ 1 / (σ · √(T · log(1/δ)))
```

#### Central Differential Privacy

Server adds noise after aggregation:

```
Algorithm: Central DP Aggregation
──────────────────────────────────

Aggregate updates:
  ḡ_t = (1/K) Σ_k (ḡ_k / max(1, ||ḡ_k||/C))

Add noise:
  ḡ_t^{noisy} = ḡ_t + noise ~ N(0, σ² · I)

Update model:
  w_{t+1} = w_t - η · ḡ_t^{noisy}

Advantages:
- Weaker privacy assumptions for clients
- Better convergence than local DP
- Requires trusted server
```

**Privacy Budget Composition:**

```
After T rounds with noise scale σ:
  (ε, δ)-DP guarantee:
  
  ε(T) = √(2T · ln(1/δ)) / σ
  
  or with Rényi DP composition:
  
  λ(T) = Σ_{t=1}^T √(2 · λ_t · ln(1/δ_t))
```

### 2. Secure Aggregation Protocols

Prevents the server from seeing individual client updates while still learning the aggregate.

#### Additive Secret Sharing (ASS)

```
Protocol: Secure Summation with Pairwise Masking
─────────────────────────────────────────────────

Setup phase:
  For each pair of clients (i, j):
    s_ij ← random symmetric matrix
    Client i stores S_i = Σ_j s_ij
    Client j stores S_j = Σ_i s_ji^T

Update phase:
  Client k:
    1. Computes local update: u_k
    2. Masks update: m_k = u_k + Σ_j (s_kj - s_jk^T)
    3. Sends: m_k to server

Server:
  aggregation = Σ_k m_k = Σ_k u_k (masks cancel!)
  
Security:
  - Server sees only masked updates
  - No information leak if < n-1 clients collude
  - Dropout tolerant with redundancy
```

#### Homomorphic Encryption Approach

```
Protocol: FL with Homomorphic Encryption
─────────────────────────────────────────

Setup:
  Server generates public-private key pair (pk, sk)
  Distribute pk to all clients

Encryption phase:
  Client k:
    1. Computes update: u_k
    2. Encrypts: E_pk(u_k)
    3. Sends: E_pk(u_k) to server

Aggregation phase:
  Server computes:
    E_pk(aggregate) = Σ_k E_pk(u_k)  [via homomorphic addition]
  
Decryption phase:
  Server decrypts with sk:
    aggregate = D_sk(E_pk(aggregate))

Computation Cost: High (slow encryption/decryption)
Privacy Level: Information-theoretic
```

### 3. Membership Inference Attacks

Attacker tries to determine if a specific record was in training data.

#### Black-box Membership Inference

```
Attack: Membership Inference via Model Queries
───────────────────────────────────────────

1. Train reference models on datasets with/without target record
2. Compute loss on target record for each reference model
3. Measure difference in loss distribution
4. Use loss threshold to infer membership

Defense: Regularization and DP
  - Add regularization λ·||w||² to equalize losses
  - Use differential privacy (adds random noise)
  - Limit model updates (reduce information leakage)

Privacy risk metric:
  Δaccuracy = |accuracy_with_record - accuracy_without|
  Δaccuracy > 10% indicates potential vulnerability
```

#### White-box Membership Inference

```
Attack: Gradient-based Membership Inference
──────────────────────────────────────────

1. Obtain gradient updates or model weights
2. Extract training data from gradients via optimization:
   x^* = argmin_x ||∇f_k(w) - ∇L(w; x)||²
3. Match reconstructed data with candidate records

Defense Strategies:
  1. Gradient clipping: ||g|| ≤ C
     - Limits reconstructed data fidelity
  
  2. Gradient compression: Keep only top-k gradients
     - Reduces attack surface
  
  3. Differential privacy: Add noise to gradients
     - Strongest defense, but costs convergence
  
  4. Update aggregation: Use multiple client updates
     - Blurs individual information
```

### 4. Model Inversion Attacks

Attacker reconstructs training data from model weights/gradients.

#### Property Inference

```
Attack: Inferring Dataset Properties
────────────────────────────────────

Methods:
  1. Train two models with/without target property
  2. Compare model weights/activations
  3. Use SVM/DNN to classify membership of property

Example:
  - Model A: trained on balanced dataset
  - Model B: trained on imbalanced dataset
  - Measure model parameter statistics
  - Infer if deployment model has property
```

#### Gradient-based Reconstruction

```
Attack: Exact Gradient Matching
───────────────────────────────

Problem:
  min_x ||∇L(w; x, y) - g_target||²

where g_target is intercepted gradient

For small batches (batch_size = 1):
  Can reconstruct images with high fidelity
  
For batch_size = 32:
  Reconstruction becomes difficult
  
Defense: Batch aggregation
  - Use larger batch sizes
  - Aggregate gradients across multiple batches
  - Add noise to obscure individual contributions
```

### 5. Privacy Budgets and Guarantees

#### Privacy Budget Tracking

```
Privacy Budget Accounting:
─────────────────────────

Initial budget: (ε_total, δ)
Typical values:
  - ε_total = 8 (moderate privacy)
  - δ = 1e-5 (very small failure probability)

Per-round allocation:
  ε_t = ε_total / T
  δ_t = δ / T
  
  where T is total number of rounds

Remaining budget after t rounds:
  ε_remaining = ε_total - Σ_{i=1}^t ε_i

Once ε_remaining ≈ 0, training must stop
```

#### Rényi Differential Privacy

More general than ε-δ DP, easier to compose:

```
Definition:
  D_λ(P||Q) = (1/(λ-1)) · log(E_x~P[(P(x)/Q(x))^(λ-1)])

A mechanism M satisfies (λ, ρ)-RDP if:
  D_λ(M(D) || M(D')) ≤ ρ

Conversion to (ε, δ)-DP:
  ε = min_{λ>1} (log(1/δ)/λ - ρ)
  
Benefits:
  - Automatic group privacy composition
  - Tighter bounds than basic composition
  - Better for adaptive queries
```

---

## Frameworks & Implementation

### 1. TensorFlow Federated (TFF)

Official Google framework for federated learning research.

#### Architecture

```
TFF Components:
───────────────

1. Federated Core (FC)
   - Low-level building blocks
   - Defines distributed computations
   - Type system for federated values

2. Federated Learning API (FL)
   - High-level constructs
   - Simplifies federated training
   - Pre-built algorithms

3. Simulation Runtime
   - Simulate federated learning locally
   - Testing and prototyping
   - Debugging without distributed setup
```

#### Basic Usage

```python
import tensorflow_federated as tff
import tensorflow as tf

# 1. Create model
def create_keras_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# 2. Build federated learning process
process = tff.learning.build_federated_averaging_process(
    model_fn=create_keras_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=1.0)
)

# 3. Run federated training
state = process.initialize()
for round_num in range(1, 11):
    # Sample clients and their data
    sampled_clients = random.sample(all_clients, 10)
    client_data = [datasets[c] for c in sampled_clients]
    
    # Perform round
    state, metrics = process.next(state, client_data)
    print(f'Round {round_num}: {metrics}')
```

#### Advanced Features

**Differential Privacy Integration:**

```python
# With DP-FedAvg
private_process = tff.learning.build_dp_federated_averaging_process(
    model_fn=create_keras_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
    server_optimizer_fn_for_dp=
        lambda: tf.keras.optimizers.SGD(1.0),
    
    # Privacy parameters
    clients_per_round=10,
    dp_mechanism='gaussian',
    noise_multiplier=0.55,  # Controls privacy-utility tradeoff
    
    # Budget
    dp_target_unclipped_quantile=0.5,
    dp_clip_norm=1.0
)
```

### 2. PySyft Framework

OpenMined's privacy-preserving machine learning framework.

#### Key Concepts

```python
import torch
import syft as sy

# Create virtual workers (simulating devices)
alice = sy.VirtualWorker(hook=hook, id="alice")
bob = sy.VirtualWorker(hook=hook, id="bob")

# Data on different workers
alice_data = torch.tensor([1.0, 2.0, 3.0])
alice_data.send(alice)

bob_data = torch.tensor([4.0, 5.0, 6.0])
bob_data.send(bob)

# Federated operations
result = alice_data.sum() + bob_data.sum()
result.get()  # Pull result back
```

#### Secure Multi-Party Computation (SMPC)

```python
# Secure aggregation
def secure_aggregate(worker_updates):
    """
    Uses secret sharing to aggregate without revealing individual updates
    """
    # Secret shares are distributed
    shared_updates = [
        update.share(workers=[alice, bob, charlie])
        for update in worker_updates
    ]
    
    # Aggregation on secret shares
    aggregated = sum(shared_updates) / len(shared_updates)
    
    # Reveal only final result
    return aggregated.get()

# Training with privacy
model = SimpleNet().send(alice)
optimizer = syft.frameworks.torch.optim.SGD(
    params=model.parameters(),
    lr=0.02
)

for epoch in range(epochs):
    # Local training
    output = model(alice_data)
    loss = loss_fn(output, alice_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 3. FLOWER Framework

Flexible and scalable framework for federated learning.

#### Architecture

```
FLOWER Architecture:
───────────────────

┌─────────────────────────────┐
│     Server (Aggregator)     │
│  - Coordinates round        │
│  - Aggregates updates       │
│  - Manages convergence      │
└──────────────┬──────────────┘
               │
        ┌──────┴──────┬──────────┬──────────┐
        │             │          │          │
    ┌───▼──┐     ┌───▼──┐   ┌──▼───┐   ┌──▼───┐
    │Client│     │Client│   │Client│   │Client│
    │  1   │     │  2   │   │  3   │   │  K   │
    └──────┘     └──────┘   └──────┘   └──────┘

Components:
- Client: Runs local training
- Server: Aggregates updates
- Strategy: Defines aggregation logic
```

#### Basic Implementation

```python
from flwr.client import NumPyClient, start_numpy_client
from flwr.server import start_server
from flwr.server.strategy import FedAvg

# Client implementation
class IrisClient(NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def get_parameters(self, config):
        """Return model parameters as list of numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Update model parameters with received parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Local training"""
        self.set_parameters(parameters)
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(config.get("epochs", 1)):
            for x, y in self.train_dataloader:
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        return self.get_parameters({}), len(self.x_train), {}
    
    def evaluate(self, parameters, config):
        """Local evaluation"""
        self.set_parameters(parameters)
        
        loss, accuracy = 0, 0
        with torch.no_grad():
            for x, y in self.test_dataloader:
                output = self.model(x)
                loss += criterion(output, y).item()
                accuracy += (output.argmax(-1) == y).float().mean().item()
        
        return loss / len(self.test_dataloader), \
               {"accuracy": accuracy / len(self.test_dataloader)}

# Server setup
strategy = FedAvg(
    fraction_fit=0.3,        # Fraction of clients to sample
    fraction_evaluate=0.2,
    min_fit_clients=3,
    min_evaluate_clients=2,
    min_available_clients=5,
    initial_parameters=get_initial_parameters(model)
)

# Start server
start_server(
    server_address="0.0.0.0:8080",
    config=ServerConfig(num_rounds=10),
    strategy=strategy,
)

# Client connection
def client_fn(cid: str) -> IrisClient:
    """Factory function for creating clients"""
    model = Net()
    x_train, y_train = load_train_data(cid)
    x_test, y_test = load_test_data(cid)
    return IrisClient(model, x_train, y_train, x_test, y_test)

start_numpy_client(
    server_address="localhost:8080",
    client=client_fn(client_id)
)
```

#### Custom Strategy

```python
from flwr.server.strategy import Strategy

class CustomFedLearning(Strategy):
    def initialize_parameters(self, client_manager):
        """Initialize global model parameters"""
        pass
    
    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure client training"""
        clients = client_manager.sample(
            num_clients=10,
            min_num_clients=10,
            criterion=None
        )
        
        config = {
            "batch_size": 32,
            "local_epochs": 1,
            "learning_rate": 0.01
        }
        
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client updates"""
        # weights, metrics = zip(*results)
        # Implement custom aggregation here
        return aggregate_weighted(results)
    
    def evaluate(self, server_round: int, parameters):
        """Evaluate on server-side validation set"""
        # Optional server-side evaluation
        pass
```

---

## Applications & Benchmarks

### 1. Healthcare Data Sharing

Federated learning enables hospitals to train models without sharing patient data.

#### Use Case: Federated COVID-19 Prediction

```
Scenario: Multiple hospitals train shared diagnostic model
──────────────────────────────────────────────────────

Hospital A: 5,000 patient records
Hospital B: 3,000 patient records  
Hospital C: 2,000 patient records

Traditional ML: Violates HIPAA (centralized data)
Federated ML: Each hospital trains locally with private data

Architecture:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Hospital A  │  │  Hospital B  │  │  Hospital C  │
│  Private DB  │  │  Private DB  │  │  Private DB  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │ ∇L (gradient)   │                 │
       └─────────┬───────┴─────────┬───────┘
                 │                 │
           ┌─────▼─────────────────▼────┐
           │   Aggregation Server      │
           │   (no patient data)       │
           │   w_new = avg(∇L_A, ...)  │
           └──────────┬────────────────┘
                      │
       ┌──────────────┴─────────────────┬─────────┐
       │              │                 │         │
    ┌──▼──┐        ┌──▼──┐          ┌──▼──┐   ┌─▼──┐
    │HA   │        │HB   │          │HC   │   │... │
```

#### Performance Comparison

```
Metric                    | Centralized | Federated | Federated+DP
─────────────────────────────────────────────────────────────────
Accuracy                  | 91.2%       | 90.8%     | 89.5%
Patient Privacy Breach    | 0% (broken) | 0%        | 0%
Data Transfer             | 500 GB      | 50 MB     | 50 MB
Model Confidence          | High        | High      | Moderate
Training Time (rounds)    | 1           | 100       | 100
Communication Cost        | High        | Low       | Low
Regulatory Compliance     | No          | Yes       | Yes
```

### 2. Mobile Devices Training

On-device learning for smartphones and IoT devices.

#### Use Case: Keyboard Prediction

```
Scenario: Federated learning on smartphones for next-word prediction
────────────────────────────────────────────────────────────────

Each device:
  - Has unique user typing patterns
  - 1,000-5,000 words of typing data
  - Limited computation (ARM processor)
  - Intermittent connectivity

Training loop:
  1. Download latest model (0.5 MB)
  2. Train locally for 1 epoch (2-5 minutes)
  3. Send update when: plugged in + WiFi + idle
  4. Update model from server

Privacy:
  - User text never leaves device
  - Only gradient updates transmitted
  - Server cannot reconstruct text from updates
```

#### Communication Optimization

```python
# Compression for mobile
def compress_update(gradient, compression_ratio=0.01):
    """
    Compress gradient for mobile transmission
    """
    # Top-K sparsification
    k = int(len(gradient) * compression_ratio)
    top_k_indices = np.argsort(np.abs(gradient))[-k:]
    
    compressed = np.zeros_like(gradient)
    compressed[top_k_indices] = gradient[top_k_indices]
    
    # Quantization to int8
    max_val = np.max(np.abs(compressed))
    quantized = (compressed / max_val * 127).astype(np.int8)
    
    return {
        'values': quantized[top_k_indices],
        'indices': top_k_indices,
        'scale': max_val,
        'size_bytes': len(top_k_indices) * 2 + 8  # approximate
    }

def decompress_update(compressed, original_shape):
    """Reconstruct full gradient"""
    reconstructed = np.zeros(original_shape)
    reconstructed[compressed['indices']] = \
        compressed['values'] * compressed['scale'] / 127
    return reconstructed

# Bandwidth savings
original_size = 100_000 * 4  # 100K float32
compressed_size = 100_000 * 0.01 * 2 + 16  # ~2 KB
savings = (1 - compressed_size / original_size) * 100
print(f"Bandwidth savings: {savings:.1f}%")  # ~99.98%
```

### 3. IoT Sensor Networks

Federated learning for distributed sensor networks.

#### Predictive Maintenance Example

```
Scenario: Factory equipment monitoring with IoT sensors
─────────────────────────────────────────────────────

Each machine has:
  - 10-100 IoT sensors (temperature, vibration, noise)
  - Real-time data streaming
  - Embedded processor (limited resources)
  - Goal: Predict equipment failure

Federated approach:
  1. Each machine trains failure prediction model locally
  2. Shares only model updates (not raw sensor data)
  3. Factory learns global failure patterns
  4. Avoids sending sensitive production data to cloud

Benefits:
  ✓ Low latency (local models)
  ✓ Data privacy (no centralization)
  ✓ Fault tolerance (redundant models)
  ✓ Bandwidth efficiency (updates << raw data)
```

#### Resource-Constrained Training

```python
# Lightweight FL for edge devices
import numpy as np

class EdgeFLClient:
    def __init__(self, model, device_id, memory_limit_mb=64):
        self.model = model
        self.device_id = device_id
        self.memory_limit = memory_limit_mb * 1024 * 1024
    
    def train_adaptive(self, data_loader, global_model):
        """
        Adaptive training based on available resources
        """
        # Monitor memory
        available_memory = self.get_available_memory()
        
        if available_memory < self.memory_limit * 0.2:
            # Low memory: reduce batch size, fewer epochs
            batch_size = 8
            epochs = 1
            gradient_accumulation = True
        else:
            batch_size = 32
            epochs = 5
            gradient_accumulation = False
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        cumulative_grad = None
        for epoch in range(epochs):
            for x, y in DataLoader(data_loader, batch_size=batch_size):
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                
                if gradient_accumulation:
                    if cumulative_grad is None:
                        cumulative_grad = [g.clone() for g in self.model.parameters()]
                    else:
                        for i, g in enumerate(self.model.parameters()):
                            cumulative_grad[i] += g.grad
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
        
        if gradient_accumulation:
            # Apply accumulated gradients
            for param, grad in zip(self.model.parameters(), cumulative_grad):
                param.grad = grad / len(data_loader)
            optimizer.step()
            optimizer.zero_grad()
        
        return self.model.state_dict()
```

### 4. LEAF Benchmark

Federated Learning Evaluation of Algorithms (LEAF) benchmark suite.

#### Benchmark Datasets

| Dataset | Clients | Samples/Client | Tasks | Features | Setting |
|---------|---------|----------------|-------|----------|---------|
| FEMNIST | 3,550   | 226 (avg)      | 62    | 784      | Vision  |
| CIFAR-100 | 500   | 600 (avg)      | 100   | 3,072    | Vision  |
| Shakespeare | 715 | 101 (avg)      | -     | 80       | Text    |
| Reddit  | 1.4M   | 5-500          | -     | 10,000   | Text    |
| Synthetic | 1,000 | 600            | 30    | 60       | Generated |
| MNIST   | 1,000  | 600 (avg)      | 10    | 784      | Vision  |

#### Benchmark Metrics

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class FLBenchmark:
    def __init__(self, num_rounds=100, num_clients_per_round=10):
        self.num_rounds = num_rounds
        self.num_clients_per_round = num_clients_per_round
        self.metrics_history = []
    
    def evaluate_round(self, global_model, client_models, test_data):
        """
        Evaluate FL round on standard metrics
        """
        # Global accuracy on IID test set
        global_preds = global_model.predict(test_data['x'])
        global_accuracy = accuracy_score(test_data['y'], global_preds)
        
        # Per-client accuracy (local evaluation)
        client_accuracies = []
        for client_id, client_model in client_models.items():
            if client_id in test_data:
                preds = client_model.predict(test_data[client_id]['x'])
                acc = accuracy_score(test_data[client_id]['y'], preds)
                client_accuracies.append(acc)
        
        local_avg_accuracy = np.mean(client_accuracies)
        local_std_accuracy = np.std(client_accuracies)
        
        # Communication cost
        comm_rounds_so_far = len(self.metrics_history)
        
        metrics = {
            'round': comm_rounds_so_far,
            'global_accuracy': global_accuracy,
            'local_avg_accuracy': local_avg_accuracy,
            'local_std_accuracy': local_std_accuracy,
            'convergence_gap': global_accuracy - local_avg_accuracy,
            'communication_cost': comm_rounds_so_far  # in terms of rounds
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def report(self):
        """Generate benchmark report"""
        metrics_array = np.array([
            [m['round'], m['global_accuracy'], m['local_avg_accuracy']]
            for m in self.metrics_history
        ])
        
        print("FL Benchmark Report")
        print("─" * 60)
        print(f"Total Rounds: {self.num_rounds}")
        print(f"Clients per Round: {self.num_clients_per_round}")
        print(f"\nFinal Global Accuracy: {metrics_array[-1, 1]:.4f}")
        print(f"Final Local Avg Accuracy: {metrics_array[-1, 2]:.4f}")
        print(f"Convergence Gap: {metrics_array[-1, 1] - metrics_array[-1, 2]:.4f}")
        
        # Convergence speed (rounds to reach 80% accuracy)
        target_acc = 0.80
        rounds_to_target = None
        for m in self.metrics_history:
            if m['global_accuracy'] >= target_acc:
                rounds_to_target = m['round']
                break
        
        if rounds_to_target:
            print(f"Rounds to reach {target_acc:.1%} accuracy: {rounds_to_target}")
        
        return metrics_array
```

### 5. Performance Metrics

#### Standard Evaluation Metrics

```python
class PerformanceMetrics:
    @staticmethod
    def convergence_speed(accuracies, target=0.90):
        """Rounds needed to reach target accuracy"""
        for i, acc in enumerate(accuracies):
            if acc >= target:
                return i
        return len(accuracies)
    
    @staticmethod
    def communication_efficiency(bytes_transmitted, accuracy_final):
        """Accuracy per GB of communication"""
        gb_transmitted = bytes_transmitted / (1e9)
        return accuracy_final / gb_transmitted
    
    @staticmethod
    def local_vs_global_gap(local_accuracies, global_accuracy):
        """Measure divergence between local and global models"""
        avg_local = np.mean(local_accuracies)
        return global_accuracy - avg_local
    
    @staticmethod
    def personalization_gain(personalized_acc, global_acc):
        """Improvement from personalization"""
        return (personalized_acc - global_acc) / global_acc * 100
    
    @staticmethod
    def privacy_cost(epsilon, accuracy_dp, accuracy_nodp):
        """Privacy-utility tradeoff"""
        utility_loss = (accuracy_nodp - accuracy_dp) / accuracy_nodp * 100
        return {
            'epsilon': epsilon,
            'utility_loss_percent': utility_loss,
            'privacy_per_unit_utility': epsilon / utility_loss if utility_loss > 0 else float('inf')
        }

# Usage example
metrics = PerformanceMetrics()

# Evaluate convergence
convergence_rounds = metrics.convergence_speed([0.5, 0.65, 0.75, 0.82, 0.85])
print(f"Convergence: {convergence_rounds} rounds")

# Evaluate communication efficiency
comm_eff = metrics.communication_efficiency(
    bytes_transmitted=50e6,  # 50 MB
    accuracy_final=0.87
)
print(f"Communication Efficiency: {comm_eff:.3f} accuracy/GB")

# Privacy-utility analysis
privacy_analysis = metrics.privacy_cost(
    epsilon=8,
    accuracy_dp=0.85,
    accuracy_nodp=0.90
)
print(f"Privacy Cost: {privacy_analysis['utility_loss_percent']:.2f}% utility loss")
```

---

## Mathematical Formulations

### 1. Convergence Rate Analysis

**Theorem 1 (FedAvg Convergence - Convex Case):**

```
Under standard smoothness and convexity assumptions:

E[f(w_T)] - f(w*) ≤ 
  (2ηLσ²) / (CK) · (1 + τ(E-1)) + 2ηLG²

where:
  - η: learning rate
  - L: smoothness constant (∇²f ≤ LI)
  - σ²: variance of local SGD
  - C: client sampling fraction
  - K: total clients
  - τ: local epoch ratio
  - E: local epochs
  - G: gradient bound
```

**Theorem 2 (Non-Convex Convergence):**

```
For non-convex objectives:

E[||∇f(w_T)||²] ≤ O(1/√T) + O(heterogeneity_error)

The heterogeneity error term:
  ρ = max_k ||∇f(w) - ∇f_k(w)||

High heterogeneity increases final error significantly
```

### 2. Privacy-Utility Tradeoff

**Lemma (Composition of DP):**

```
If each round uses (ε_t, δ_t)-DP with client sampling probability q:

Total privacy after T rounds:
  ε_total ≤ √(2T·ln(1/δ)) / σ

where σ is the noise scale:
  σ = q · √T / ε_total

Privacy loss per round:
  ε_per_round = √(2·ln(1/δ)) / σ
```

### 3. Communication Complexity

**Theorem 3 (Communication Complexity):**

```
To reach ε-optimal solution:

Communication rounds needed:
  T = O(1/ε²)  for convex objectives
  T = O(1/ε⁴)  for non-convex objectives

Total communication (bits):
  C = T × K × d × bits_per_value
  
  where d is model dimension

Reduction with compression:
  C_compressed = C × compression_ratio
  
  If compression_ratio = 0.01:
    C_compressed = C / 100
```

---

## Code Examples

### Example 1: From-Scratch FedAvg Implementation

```python
import numpy as np
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.optim import SGD

class FedAvgServer:
    def __init__(self, model: nn.Module, learning_rate: float = 1.0):
        self.global_model = model
        self.learning_rate = learning_rate
        self.round = 0
        self.history = {'loss': [], 'accuracy': []}
    
    def aggregate(self, client_weights: List[Dict], client_data_sizes: List[int]):
        """
        FedAvg aggregation with weighted averaging
        """
        total_samples = sum(client_data_sizes)
        
        # Get parameter names from first client
        param_names = list(client_weights[0].keys())
        
        # Aggregate parameters
        aggregated = {}
        for param_name in param_names:
            aggregated[param_name] = torch.zeros_like(
                client_weights[0][param_name]
            )
            
            # Weighted sum
            for i, client_param in enumerate(client_weights):
                weight = client_data_sizes[i] / total_samples
                aggregated[param_name] += weight * client_param[param_name]
        
        return aggregated
    
    def federated_round(self, clients: List['FedAvgClient'],
                       num_clients_per_round: int,
                       epochs: int = 1):
        """
        Execute one federated learning round
        """
        # Sample clients
        sampled_indices = np.random.choice(
            len(clients),
            size=min(num_clients_per_round, len(clients)),
            replace=False
        )
        sampled_clients = [clients[i] for i in sampled_indices]
        
        # Client training
        client_weights = []
        client_sizes = []
        
        for client in sampled_clients:
            # Send global model to client
            client.set_weights(self.get_weights())
            
            # Local training
            client_loss = client.train(epochs=epochs)
            
            # Collect updated weights
            client_weights.append(client.get_weights())
            client_sizes.append(len(client.train_data))
        
        # Server aggregation
        aggregated_weights = self.aggregate(client_weights, client_sizes)
        self.set_weights(aggregated_weights)
        
        # Evaluate
        val_loss, val_accuracy = self.evaluate()
        self.history['loss'].append(val_loss)
        self.history['accuracy'].append(val_accuracy)
        
        self.round += 1
        print(f"Round {self.round}: Loss={val_loss:.4f}, Acc={val_accuracy:.4f}")
    
    def get_weights(self):
        return self.global_model.state_dict()
    
    def set_weights(self, weights):
        self.global_model.load_state_dict(weights)
    
    def evaluate(self):
        # Placeholder for server-side evaluation
        return 0.0, 0.0


class FedAvgClient:
    def __init__(self, model: nn.Module, train_data, test_data=None):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
    
    def set_weights(self, weights):
        self.model.load_state_dict(weights)
    
    def get_weights(self):
        return self.model.state_dict()
    
    def train(self, epochs: int = 1, batch_size: int = 32, lr: float = 0.01):
        """
        Local SGD training
        """
        self.optimizer = SGD(self.model.parameters(), lr=lr)
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch in self.train_data:
                x, y = batch
                
                # Forward pass
                output = self.model(x)
                loss = self.criterion(output, y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches


# Usage
if __name__ == "__main__":
    # Create model
    model = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Initialize server
    server = FedAvgServer(model, learning_rate=1.0)
    
    # Create 10 clients with heterogeneous data
    clients = []
    for i in range(10):
        client_model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        # Simulate data (in practice, load real data)
        train_data = [(torch.randn(32, 784), torch.randint(0, 10, (32,)))]
        client = FedAvgClient(client_model, train_data)
        clients.append(client)
    
    # Federated training
    for round_num in range(100):
        server.federated_round(clients, num_clients_per_round=5, epochs=1)
```

### Example 2: Differential Privacy in FL

```python
import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD

class DifferentiallyPrivateFLClient:
    def __init__(self, model, train_data, l2_norm_clip=1.0, noise_scale=1.0):
        self.model = model
        self.train_data = train_data
        self.criterion = nn.CrossEntropyLoss()
        self.l2_norm_clip = l2_norm_clip
        self.noise_scale = noise_scale
    
    def clip_gradients(self):
        """
        Clip gradients to L2 norm bound
        """
        total_norm = 0
        for param in self.model.parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2.)
        
        clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.0)
        for param in self.model.parameters():
            param.grad.data.mul_(clip_coef)
    
    def add_noise(self):
        """
        Add Gaussian noise for differential privacy
        """
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_scale
                param.grad.data.add_(noise)
    
    def train_with_dp(self, epochs: int = 1, batch_size: int = 32, lr: float = 0.01):
        """
        Local training with DP-SGD
        """
        optimizer = SGD(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for batch in self.train_data:
                x, y = batch
                
                # Forward pass
                output = self.model(x)
                loss = self.criterion(output, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # DP-SGD steps
                self.clip_gradients()  # Gradient clipping
                self.add_noise()        # Add noise
                
                optimizer.step()
        
        return self.model.state_dict()


class DifferentiallyPrivateFLServer:
    def __init__(self, epsilon_total: float = 8.0, delta: float = 1e-5,
                 total_rounds: int = 100):
        self.epsilon_total = epsilon_total
        self.delta = delta
        self.total_rounds = total_rounds
        self.epsilon_per_round = epsilon_total / np.sqrt(total_rounds)
        self.current_round = 0
    
    def compute_noise_scale(self, gradient_clip_norm: float) -> float:
        """
        Compute noise scale for given epsilon budget and gradient clip norm
        """
        # Using advanced composition
        noise_scale = np.sqrt(2 * np.log(1.25 / self.delta)) * gradient_clip_norm
        noise_scale /= self.epsilon_per_round
        return noise_scale
    
    def get_remaining_epsilon(self) -> float:
        """
        Track privacy budget consumption
        """
        consumed = self.current_round * self.epsilon_per_round
        remaining = self.epsilon_total - consumed
        return remaining
    
    def aggregate_with_dp(self, client_updates, client_sizes):
        """
        Aggregate client updates (already have noise added locally)
        Optionally add server-side noise
        """
        total_samples = sum(client_sizes)
        aggregated = {}
        
        # Weighted averaging (noise already in client updates)
        for key in client_updates[0].keys():
            aggregated[key] = torch.zeros_like(client_updates[0][key])
            for i, update in enumerate(client_updates):
                weight = client_sizes[i] / total_samples
                aggregated[key] += weight * update[key]
        
        self.current_round += 1
        return aggregated


# Usage with privacy accounting
if __name__ == "__main__":
    # Privacy parameters
    epsilon_total = 8.0
    delta = 1e-5
    gradient_clip_norm = 1.0
    
    # Create server
    dp_server = DifferentiallyPrivateFLServer(
        epsilon_total=epsilon_total,
        delta=delta,
        total_rounds=100
    )
    
    # Create clients with DP
    noise_scale = dp_server.compute_noise_scale(gradient_clip_norm)
    print(f"Noise scale: {noise_scale:.4f}")
    
    clients = []
    for i in range(10):
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        train_data = [(torch.randn(32, 784), torch.randint(0, 10, (32,)))]
        client = DifferentiallyPrivateFLClient(
            model, train_data,
            l2_norm_clip=gradient_clip_norm,
            noise_scale=noise_scale
        )
        clients.append(client)
    
    # Training loop with privacy accounting
    for round_num in range(100):
        print(f"\nRound {round_num + 1}")
        print(f"  Epsilon remaining: {dp_server.get_remaining_epsilon():.4f}")
        
        # Sample and train clients
        sampled = np.random.choice(len(clients), size=5, replace=False)
        client_updates = []
        client_sizes = []
        
        for idx in sampled:
            update = clients[idx].train_with_dp(epochs=1)
            client_updates.append(update)
            client_sizes.append(100)  # assume 100 samples per client
        
        # Server aggregation
        aggregated = dp_server.aggregate_with_dp(client_updates, client_sizes)
```

### Example 3: Byzantine-Robust Aggregation

```python
import torch
import numpy as np
from typing import List, Dict

class ByzantineRobustAggregator:
    @staticmethod
    def krum(updates: List[Dict], num_byzantine: int = 1):
        """
        Krum: Byzantine-robust aggregation
        Selects update closest to (K-f-2) others
        """
        K = len(updates)
        f = num_byzantine
        
        # Convert dict updates to vectors for distance computation
        vectors = []
        for update in updates:
            vec = torch.cat([v.flatten() for v in update.values()])
            vectors.append(vec)
        vectors = torch.stack(vectors)
        
        # Compute pairwise distances
        distances = torch.cdist(vectors, vectors)
        
        # Score each update
        scores = []
        for i in range(K):
            # Sum of (K-f-2) nearest neighbors
            nearest_k = torch.topk(distances[i], K - f - 1, largest=False)[0]
            score = torch.sum(nearest_k)
            scores.append(score)
        
        # Select update with minimum score
        selected_idx = np.argmin(scores)
        return updates[selected_idx]
    
    @staticmethod
    def coordinate_wise_median(updates: List[Dict]):
        """
        Coordinate-wise median aggregation
        Robust to ⌊(K-1)/2⌋ Byzantine clients
        """
        # Get all parameters
        param_names = list(updates[0].keys())
        aggregated = {}
        
        for param_name in param_names:
            # Stack parameter values
            param_values = torch.stack([
                update[param_name] for update in updates
            ])
            
            # Take median along client dimension
            aggregated[param_name] = torch.median(param_values, dim=0)[0]
        
        return aggregated
    
    @staticmethod
    def multi_krum(updates: List[Dict], num_byzantine: int = 1, m: int = None):
        """
        Multi-Krum: Select m trusted updates before averaging
        """
        K = len(updates)
        f = num_byzantine
        
        if m is None:
            m = K - 2 * f - 2
        
        # Compute Krum scores for all updates
        vectors = []
        for update in updates:
            vec = torch.cat([v.flatten() for v in update.values()])
            vectors.append(vec)
        vectors = torch.stack(vectors)
        
        distances = torch.cdist(vectors, vectors)
        
        scores = []
        for i in range(K):
            nearest_k = torch.topk(distances[i], K - f - 1, largest=False)[0]
            score = torch.sum(nearest_k)
            scores.append(score)
        
        # Select top-m with lowest scores
        selected_indices = np.argsort(scores)[:m]
        selected_updates = [updates[i] for i in selected_indices]
        
        # Average selected updates
        param_names = list(updates[0].keys())
        aggregated = {}
        
        for param_name in param_names:
            aggregated[param_name] = torch.zeros_like(updates[0][param_name])
            for update in selected_updates:
                aggregated[param_name] += update[param_name]
            aggregated[param_name] /= len(selected_updates)
        
        return aggregated
    
    @staticmethod
    def trimmed_mean(updates: List[Dict], num_byzantine: int = 1):
        """
        Trim extreme values and take mean
        """
        K = len(updates)
        f = num_byzantine
        
        param_names = list(updates[0].keys())
        aggregated = {}
        
        for param_name in param_names:
            # Stack values
            param_values = torch.stack([
                update[param_name] for update in updates
            ])
            
            # Sort along client dimension and trim
            sorted_vals, _ = torch.sort(param_values, dim=0)
            trimmed = sorted_vals[f:-f]  # Remove top-f and bottom-f
            
            # Mean of trimmed values
            aggregated[param_name] = torch.mean(trimmed, dim=0)
        
        return aggregated


# Testing Byzantine robustness
if __name__ == "__main__":
    # Create normal and Byzantine updates
    K = 10
    num_byzantine = 3
    model_dim = 1000
    
    updates = []
    
    # Normal updates
    for i in range(K - num_byzantine):
        update = {'weight': torch.randn(100, 100) * 0.1}
        updates.append(update)
    
    # Byzantine updates (extreme values)
    for i in range(num_byzantine):
        update = {'weight': torch.randn(100, 100) * 100}  # Extreme values
        updates.append(update)
    
    aggregator = ByzantineRobustAggregator()
    
    # Compare aggregation methods
    print("Aggregation Methods Comparison")
    print("─" * 50)
    
    # Standard mean (vulnerable)
    standard_mean = {}
    for key in updates[0].keys():
        standard_mean[key] = torch.stack([
            u[key] for u in updates
        ]).mean(dim=0)
    print(f"Standard Mean Norm: {torch.norm(standard_mean['weight']):.4f}")
    
    # Krum
    krum_result = aggregator.krum(updates, num_byzantine=num_byzantine)
    print(f"Krum Norm: {torch.norm(krum_result['weight']):.4f}")
    
    # Median
    median_result = aggregator.coordinate_wise_median(updates)
    print(f"Median Norm: {torch.norm(median_result['weight']):.4f}")
    
    # Multi-Krum
    multi_krum_result = aggregator.multi_krum(
        updates, num_byzantine=num_byzantine, m=5
    )
    print(f"Multi-Krum Norm: {torch.norm(multi_krum_result['weight']):.4f}")
    
    # Trimmed mean
    trimmed_result = aggregator.trimmed_mean(updates, num_byzantine=num_byzantine)
    print(f"Trimmed Mean Norm: {torch.norm(trimmed_result['weight']):.4f}")
```

---

## References

### Foundational Papers

1. **McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).** "Communication-Efficient Learning of Deep Networks from Decentralized Data." *International Conference on Machine Learning (ICML)*, 2017.
   - Introduces FedAvg algorithm
   - Foundation of federated learning

2. **Bonawitz, K., Eichner, H., Grieskamp, W., Huba, D., Ingerman, A., Ivanov, V., ... & Zhao, T. (2019).** "Towards Federated Learning at Scale: System Design." *Symposium on Operating Systems Design and Implementation (OSDI)*, 2019.
   - TensorFlow Federated architecture
   - Large-scale FL deployment

3. **Yang, Q., Liu, Y., Chen, T., & Suzumura, T. (2019).** "Federated Learning." *Synthesis Lectures on Artificial Intelligence and Machine Learning*, Morgan & Claypool.
   - Comprehensive survey of FL techniques

### Privacy & Security

4. **Dwork, C., & Roth, A. (2014).** "The Algorithmic Foundations of Differential Privacy." *Foundations and Trends in Theoretical Computer Science*.
   - Differential privacy foundations

5. **Wei, K., Li, J., Ding, M., Ma, C., Yang, H. H., Farokhi, F., ... & Poor, H. V. (2020).** "Federated Learning with Differential Privacy: Algorithms and Performance Analysis." *IEEE Transactions on Information Forensics and Security*, 15, 3454-3469.
   - DP in federated learning

6. **Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., ... & Yoneki, E. (2017).** "Practical Secure Aggregation for Privacy-Preserving Machine Learning." *ACM Conference on Computer and Communications Security (CCS)*, 2017.
   - Secure aggregation protocols

### Advanced Techniques

7. **Li, T., Sahu, A. K., Zaheer, M., Savarese, S., & Talwalkar, A. (2020).** "Federated Optimization in Heterogeneous Networks." *Conference on Machine Learning and Systems (MLSys)*, 2020.
   - FedProx for heterogeneous data
   - Convergence analysis

8. **Smith, V., Chiang, C. Y., Sanjabi, M., & Talwalkar, A. S. (2017).** "Federated Multi-Task Learning." *Advances in Neural Information Processing Systems (NIPS)*, 2017.
   - Multi-task federated learning
   - Task-specific personalization

9. **Fallah, A., Mokhtari, A., & Ozdaglar, A. (2020).** "Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach." *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.
   - Federated meta-learning
   - Personalization guarantees

### Byzantine Robustness

10. **Lamport, L., Shostak, R., & Pease, M. (1982).** "The Byzantine Generals Problem." *ACM Transactions on Programming Languages and Systems*, 4(3), 382-401.
    - Classical Byzantine fault tolerance

11. **Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017).** "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent." *Advances in Neural Information Processing Systems (NIPS)*, 2017.
    - Krum aggregation for Byzantine robustness

### Benchmarks & Applications

12. **Caldas, S., Dudley, J. M., Wu, P., Li, T., Konečný, J., McMahan, H. B., ... & Talwalkar, A. (2018).** "LEAF: A Benchmark for Federated Settings." *Advances in Neural Information Processing Systems (NeurIPS) Workshops*, 2018.
    - LEAF benchmark suite
    - Standardized evaluation

13. **Kairouz, P., McMahan, H. B., Avent, B., Belilovsky, E., Bennis, M., Bhagoji, A. N., ... & Zhao, S. (2021).** "Advances and Open Problems in Federated Learning." *Foundations and Trends in Machine Learning*, 14(1–2), 1–210.
    - Comprehensive survey of FL
    - Open research problems

### Implementation & Systems

14. **Beutel, D. J., Topal, T., Mathur, A., Qian, S., Geyer, R. C., & Lane, N. D. (2020).** "FLOWER: A Friendly Federated Learning Research Framework." *arXiv preprint arXiv:2007.14861*.
    - FLOWER framework design
    - Extensible FL system

15. **Ryffel, T., Trask, A., Pasquini, M., Cecchini, G., Oprea, A., Sandholm, T., & Passemiers, A. (2019).** "A Generic Framework for Interesting Subspace Clustering of High-dimensional Data." *Proceedings of the 28th USENIX Security Symposium*, 2019.
    - PySyft framework
    - Privacy-preserving ML

---

## Summary

This comprehensive guide covers:

- **Fundamentals**: FedAvg algorithm, communication efficiency, convergence analysis, privacy, and heterogeneity
- **Advanced Techniques**: Personalization, multi-task learning, meta-learning, asynchronous aggregation, Byzantine robustness
- **Privacy & Security**: Differential privacy, secure aggregation, membership inference, model inversion
- **Frameworks**: TensorFlow Federated, PySyft, FLOWER with implementation examples
- **Applications**: Healthcare, mobile devices, IoT networks
- **Benchmarks**: LEAF datasets and performance metrics
- **Mathematical Formulations**: Convergence bounds, privacy accounting, communication complexity
- **Code Examples**: From-scratch FedAvg, DP-FL, Byzantine-robust aggregation

### Key Takeaways

1. **FedAvg** is the foundational algorithm enabling practical federated learning
2. **Privacy** requires careful implementation: differential privacy + secure aggregation
3. **Communication** is the bottleneck: compression and gradient sparsification critical
4. **Heterogeneity** degrades convergence: requires specialized optimization
5. **Frameworks** (TFF, PySyft, FLOWER) simplify real-world deployment

### Future Directions

- **Vertical Federated Learning**: Features distributed across parties
- **Cross-Device FL**: Billions of heterogeneous devices
- **Federated Transfer Learning**: Leveraging pre-trained models
- **Trustworthy FL**: Certified robustness and privacy guarantees
- **Green FL**: Energy-efficient distributed learning

