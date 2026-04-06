# GNN Implementation Reference - Quick Start Guide

## 1. Installation & Setup

### PyTorch Geometric

```bash
# Basic installation
pip install torch-geometric

# With CUDA support (faster)
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# DGL
pip install dgl

# Utilities
pip install scikit-learn networkx pandas numpy matplotlib
```

## 2. Basic GNN Implementations

### 2.1 Node Classification (Most Common)

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import Cora
from sklearn.metrics import accuracy_score

# Load dataset
dataset = Cora()
data = dataset[0]

# Define model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# Evaluation
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    train_acc = accuracy_score(
        data.y[data.train_mask].cpu(),
        pred[data.train_mask].cpu()
    )
    val_acc = accuracy_score(
        data.y[data.val_mask].cpu(),
        pred[data.val_mask].cpu()
    )
    test_acc = accuracy_score(
        data.y[data.test_mask].cpu(),
        pred[data.test_mask].cpu()
    )
    return train_acc, val_acc, test_acc

# Run training
for epoch in range(100):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss={loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}')
```

### 2.2 Link Prediction

```python
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        # Get node embeddings
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_label_index):
        # Predict edge existence from node embeddings
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

model = GCNLinkPredictor(data.num_node_features, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_link():
    model.train()
    optimizer.zero_grad()
    
    # Get node embeddings
    z = model(data.x, data.edge_index)
    
    # Positive edges
    pos_edge_index = data.edge_index[:, :data.edge_index.size(1)//2]
    
    # Negative edges
    neg_edge_index = negative_sampling(
        data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    
    # Concatenate positive and negative edges
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ])
    
    # Predict
    out = model.decode(z, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, edge_label)
    
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test_link():
    model.eval()
    z = model(data.x, data.edge_index)
    
    # Test edges
    out = model.decode(z, test_edge_index)
    pred = torch.sigmoid(out).cpu()
    
    auc = roc_auc_score(test_edge_label.cpu(), pred)
    return auc

for epoch in range(100):
    loss = train_link()
    if epoch % 10 == 0:
        auc = test_link()
        print(f'Epoch {epoch}: Loss={loss:.4f}, AUC={auc:.4f}')
```

### 2.3 Graph Classification

```python
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Graph convolution
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        
        # Global pooling to get graph representation
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.lin(x)
        return x

# Train on multiple graphs
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = GraphClassifier(dataset.num_node_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch}: Loss={total_loss/len(loader):.4f}')
```

## 3. Advanced Architectures

### 3.1 Graph Attention Networks (GAT)

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GAT(dataset.num_node_features, 8, dataset.num_classes, heads=8)
```

**Advantages**:
- Learns different importance weights for different neighbors
- Multi-head attention improves capacity
- Interpretable via attention weights

### 3.2 GraphSAGE (Inductive Learning)

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# For inductive learning with neighborhood sampling
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=256,
    shuffle=True,
    directed=False,
)

model = GraphSAGE(data.num_node_features, 64, data.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        
        # Only compute loss for batch nodes (not sampled neighbors)
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch}: Loss={total_loss/len(loader):.4f}')
```

**Advantages**:
- Inductive (generalizes to unseen nodes)
- Scalable via neighborhood sampling
- Multiple aggregation functions available

### 3.3 Graph Isomorphism Network (GIN)

```python
from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = hidden_channels
            
            # MLP for message function
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_ch, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        return self.lin(x)

model = GIN(dataset.num_node_features, 64, dataset.num_classes)
```

**Advantages**:
- Theoretically proven expressive power
- MLP aggregation is more powerful than mean
- Good for complex graph patterns

## 4. Scaling to Large Graphs

### 4.1 Mini-Batch Training

```python
from torch_geometric.loader import NeighborLoader

# Create loader with neighborhood sampling
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # 25 in first hop, 10 in second
    batch_size=256,
    shuffle=True,
    num_workers=4,
)

val_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=256,
    shuffle=False,
    num_workers=4,
)

model = GCN(data.num_node_features, 64, data.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        
        # Compute loss only on target nodes
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss)
    
    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
            val_losses.append(float(loss))
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f'Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Val Loss={avg_val_loss:.4f}')
```

### 4.2 Distributed Training (PyG)

```python
from torch_geometric.distributed import DistributedGraphStore
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Initialize distributed training
dist.init_process_group("nccl")

# Create distributed graph store
graph_store = DistributedGraphStore(data)

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Training proceeds normally, but distributed
```

### 4.3 Graph Sampling Strategies

```python
from torch_geometric.sampler import NeighborSampler

# Layer-wise sampling (exponential reduction)
sampler = NeighborSampler(
    data.edge_index,
    sizes=[32, 32, 32],  # Sample 32 neighbors at each layer
    batch_size=256,
    num_workers=4,
    shuffle=True,
)

# Importance-weighted sampling (sample high-degree neighbors more)
# More neighbors from degree-weighted distribution

# Cluster-based sampling (sample within clusters)
from torch_geometric.sampler import ClusterSampler

cluster_sampler = ClusterSampler(data.edge_index, num_clusters=10)
```

## 5. Hyperparameter Tuning

```python
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import AdamW

# Learning rate schedule
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training with schedule
for epoch in range(200):
    train()
    scheduler.step()

# Grid search over hyperparameters
import itertools

param_grid = {
    'hidden_channels': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout': [0.3, 0.5, 0.7],
    'num_layers': [2, 3, 4],
}

best_val_acc = 0
best_params = {}

for params in itertools.product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    
    model = create_model(config)
    val_acc = train_and_evaluate(model, config)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = config

print(f'Best params: {best_params}, Val Acc: {best_val_acc:.4f}')
```

## 6. Debugging & Profiling

```python
import torch.profiler as profiler

# Profile training
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Check for NaN/Inf
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in {name}"

# Memory usage
import tracemalloc
tracemalloc.start()
output = model(data.x, data.edge_index)
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current/1e6:.2f}MB, Peak: {peak/1e6:.2f}MB")
```

## 7. Common Patterns & Best Practices

### 7.1 Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

early_stop = EarlyStopping(patience=30)

for epoch in range(1000):
    train_loss = train()
    val_loss = validate()
    
    if early_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 7.2 Residual Connections for Deep Networks

```python
class DeepGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        # First layer
        h = self.convs[0](x, edge_index)
        h = self.bns[0](h)
        h = F.relu(h)
        h_init = h
        
        # Hidden layers with residual connections
        for conv, bn in zip(self.convs[1:-1], self.bns[1:]):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = h + h_init  # Residual connection
        
        # Output layer
        h = self.convs[-1](h, edge_index)
        return h
```

### 7.3 Class Imbalance Handling

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(data.y),
    y=data.y.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Use in loss function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Or use focal loss for more aggressive handling
def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    p = torch.exp(-ce_loss)
    loss = alpha * (1 - p) ** gamma * ce_loss
    return loss.mean()
```

## 8. Saving and Loading Models

```python
# Save model
torch.save(model.state_dict(), 'gnn_model.pt')

# Load model
model.load_state_dict(torch.load('gnn_model.pt'))

# Save full checkpoint for resuming training
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

**Reference Implementation Guide - April 2026**  
**For latest patterns, check PyTorch Geometric tutorials**
