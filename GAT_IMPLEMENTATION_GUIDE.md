# Graph Attention Networks (GAT): Practical Implementation Guide

## Quick Start Tutorial

### Installation

```bash
# PyTorch
pip install torch==2.1.0

# PyTorch Geometric
pip install torch-geometric

# Additional dependencies
pip install numpy matplotlib networkx scikit-learn
```

### Basic Usage

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

# Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Define model
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, 
                           dropout=0.6, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                           dropout=0.6, concat=False)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATNet(dataset.num_features, 8, dataset.num_classes, heads=8).to(device)
data = data.to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    acc = (out.argmax(dim=1) == data.y)[data.test_mask].float().mean()
    return acc.item()

# Run training
for epoch in range(200):
    train_loss = train()
    test_acc = test()
    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}')
```

---

## Attention Visualization Tutorial

### Extract and Plot Attention Weights

```python
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class GATWithAttention(torch.nn.Module):
    """GAT that exposes attention weights"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, 
                           dropout=0.6, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                           dropout=0.6, concat=False)
        self.attention_weights = None
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        
        # Hook to capture attention
        hook = self.gat1.register_forward_hook(self._save_attention)
        x = F.elu(self.gat1(x, edge_index))
        hook.remove()
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def _save_attention(self, module, input, output):
        # Attention weights from the first GAT layer
        self.attention_weights = module.att.detach()

def visualize_attention_head(model, data, head_idx=0, target_node=0):
    """Visualize attention weights for a specific head"""
    
    # Get attention weights
    model.eval()
    with torch.no_grad():
        _ = model(data.x, data.edge_index)
        attn_weights = model.attention_weights  # (num_edges, num_heads, 1)
    
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]
    
    # Create adjacency matrix for visualization
    attn_matrix = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    
    src, dst = edge_index
    attn_matrix[src, dst] = attn_weights[:, head_idx, 0]
    
    # Create network graph
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges with weights
    for i in range(src.shape[0]):
        s, d = src[i].item(), dst[i].item()
        w = attn_matrix[s, d].item()
        G.add_edge(s, d, weight=w)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Subgraph around target node
    neighbors = list(G.predecessors(target_node)) + [target_node]
    subgraph = G.subgraph(neighbors)
    
    pos = nx.spring_layout(subgraph, k=2, iterations=50)
    
    # Draw nodes
    node_colors = ['red' if n == target_node else 'lightblue' for n in subgraph.nodes()]
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                          node_size=500, ax=ax1)
    
    # Draw edges with color based on attention weight
    edges = subgraph.edges()
    weights = [subgraph[u][v]['weight'] for u, v in edges]
    
    if max(weights) > 0:
        norm = Normalize(vmin=0, vmax=max(weights))
        cmap = plt.cm.Reds
    else:
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.Greys
    
    for (u, v), w in zip(edges, weights):
        ax1.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=cmap(norm(w)), width=2*w+0.5, alpha=0.8,
                arrowsize=15, arrowstyle='->')
    
    nx.draw_networkx_labels(subgraph, pos, font_size=10, ax=ax1)
    ax1.set_title(f'Attention Head {head_idx}, Target Node {target_node}')
    ax1.axis('off')
    
    # Full graph heatmap
    attn_np = attn_matrix.cpu().numpy()
    im = ax2.imshow(attn_np, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Target Node')
    ax2.set_ylabel('Source Node')
    ax2.set_title('Attention Heatmap (Full Graph)')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    return fig

# Usage
model = GATWithAttention(dataset.num_features, 8, dataset.num_classes, heads=8)
fig = visualize_attention_head(model, data, head_idx=0, target_node=10)
plt.show()
```

---

## Batch Training on Large Graphs

### Mini-Batch Sampling

```python
from torch_geometric.loader import NeighborSampler

# Create neighbor sampler
train_loader = NeighborSampler(
    data.edge_index,
    node_idx=data.train_mask,
    sizes=[25, 15],  # Sample 25 neighbors in first hop, 15 in second
    batch_size=1024,
    shuffle=True,
    num_workers=4
)

class GATWithSampling(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=8, concat=True))
        self.convs.append(GATConv(hidden_channels * 8, out_channels, heads=1))
    
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target node embeddings
            x = self.convs[i]((x, x_target), edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.6, training=self.training)
        return x.log_softmax(dim=-1)

def train_with_sampling():
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        
        optimizer.zero_grad()
        out = model(data.x[n_id].to(device), adjs)
        loss = F.nll_loss(out, data.y[n_id[:batch_size]].to(device))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

---

## Head Importance Analysis

```python
class HeadImportanceAnalyzer:
    """Analyze importance of different attention heads"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.head_importance = None
    
    def compute_head_importance(self, layer_idx=0):
        """
        Compute importance of each head using:
        - Average attention weight
        - Entropy of attention distribution
        - Correlation with predictions
        """
        
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            _ = self.model(self.data.x, self.data.edge_index)
            attn_weights = self.model.gat1.att  # (num_edges, num_heads, 1)
        
        num_heads = attn_weights.shape[1]
        metrics = {}
        
        for head_idx in range(num_heads):
            head_attn = attn_weights[:, head_idx, 0].cpu().numpy()
            
            # Metric 1: Average attention (higher = more important)
            mean_attn = head_attn.mean()
            
            # Metric 2: Entropy (lower = more focused)
            entropy = -np.sum(head_attn * np.log(head_attn + 1e-10)) / len(head_attn)
            
            # Metric 3: Sparsity (higher = sparser)
            sparsity = np.sum(head_attn < 0.1) / len(head_attn)
            
            metrics[head_idx] = {
                'mean_attention': mean_attn,
                'entropy': entropy,
                'sparsity': sparsity,
                'importance_score': mean_attn * (1 - entropy) * (1 - sparsity)
            }
        
        return metrics
    
    def prune_heads(self, importance_metrics, keep_ratio=0.75):
        """Remove least important heads"""
        
        # Sort by importance
        sorted_heads = sorted(
            importance_metrics.items(),
            key=lambda x: x[1]['importance_score'],
            reverse=True
        )
        
        num_heads = len(importance_metrics)
        num_keep = max(1, int(num_heads * keep_ratio))
        
        heads_to_keep = [h[0] for h in sorted_heads[:num_keep]]
        heads_to_prune = [h[0] for h in sorted_heads[num_keep:]]
        
        print(f"Keeping {num_keep}/{num_heads} heads")
        print(f"Pruning heads: {heads_to_prune}")
        
        return heads_to_keep, heads_to_prune

# Usage
analyzer = HeadImportanceAnalyzer(model, data)
importance = analyzer.compute_head_importance()

for head_idx, metrics in importance.items():
    print(f"Head {head_idx}: {metrics['importance_score']:.4f}")

heads_to_keep, heads_to_prune = analyzer.prune_heads(importance, keep_ratio=0.75)
```

---

## Hyperparameter Tuning

```python
from torch_geometric.nn import GATConv
import optuna

class GATHyperparameterTuner:
    """Hyperparameter tuning using Optuna"""
    
    def __init__(self, data, n_trials=100):
        self.data = data
        self.n_trials = n_trials
        self.study = None
    
    def objective(self, trial):
        # Suggest hyperparameters
        hidden_channels = trial.suggest_categorical('hidden_channels', [8, 16, 32, 64])
        num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
        dropout = trial.suggest_float('dropout', 0.0, 0.8)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Build model
        model = GATNet(self.data.num_features, hidden_channels, 
                      self.data.num_classes, heads=num_heads)
        model = model.to(device)
        data = self.data.to(device)
        
        # Modify dropout
        for module in model.modules():
            if isinstance(module, GATConv):
                module.dropout = dropout
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train
        best_acc = 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            # Validate
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    val_acc = (out.argmax(dim=1)[data.val_mask] == data.y[data.val_mask]).float().mean()
                    best_acc = max(best_acc, val_acc.item())
        
        return best_acc
    
    def tune(self):
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        print(f"Best trial: {self.study.best_trial.number}")
        print(f"Best accuracy: {self.study.best_value:.4f}")
        print(f"Best hyperparameters: {self.study.best_params}")
        
        return self.study.best_params

# Usage
tuner = GATHyperparameterTuner(data, n_trials=50)
best_params = tuner.tune()
```

---

## Performance Monitoring

```python
class GATPerformanceMonitor:
    """Monitor training performance and detect issues"""
    
    def __init__(self, model, patience=20):
        self.model = model
        self.patience = patience
        self.best_val_acc = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def update(self, train_loss, val_loss, train_acc, val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.patience
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(len(self.history['train_loss']))
        
        # Loss
        ax1.plot(epochs, self.history['train_loss'], label='Train')
        ax1.plot(epochs, self.history['val_loss'], label='Val')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(epochs, self.history['train_acc'], label='Train')
        ax2.plot(epochs, self.history['val_acc'], label='Val')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)
        
        # Loss convergence (log scale)
        ax3.semilogy(epochs, self.history['train_loss'], label='Train')
        ax3.semilogy(epochs, self.history['val_loss'], label='Val')
        ax3.set_ylabel('Loss (log scale)')
        ax3.set_xlabel('Epoch')
        ax3.legend()
        ax3.grid(True)
        
        # Best accuracy
        ax4.text(0.5, 0.5, f'Best Val Accuracy: {self.best_val_acc:.4f}',
                ha='center', va='center', fontsize=16, transform=ax4.transAxes)
        ax4.axis('off')
        
        plt.tight_layout()
        return fig

# Usage in training loop
monitor = GATPerformanceMonitor(model)

for epoch in range(300):
    train_loss = train()
    val_loss, val_acc = test()
    train_acc = test_train_acc()
    
    should_stop = monitor.update(train_loss, val_loss, train_acc, val_acc)
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    if should_stop:
        print(f'Early stopping at epoch {epoch}')
        break

monitor.plot_training_history()
plt.show()
```

---

## Export and Deployment

### ONNX Export

```python
import torch.onnx

def export_gat_to_onnx(model, data, output_path='gat_model.onnx'):
    """Export trained GAT to ONNX format"""
    
    model.eval()
    
    # Create dummy inputs
    dummy_x = data.x.unsqueeze(0)  # Batch size 1
    dummy_edge_index = data.edge_index
    
    # Export
    torch.onnx.export(
        model,
        (dummy_x, dummy_edge_index),
        output_path,
        input_names=['x', 'edge_index'],
        output_names=['output'],
        dynamic_axes={
            'x': {0: 'batch_size', 1: 'num_nodes'},
            'output': {0: 'batch_size'}
        },
        opset_version=13,
        verbose=False
    )
    
    print(f"Model exported to {output_path}")
```

### TorchScript Export

```python
def export_gat_to_torchscript(model, output_path='gat_model.pt'):
    """Export trained GAT as TorchScript"""
    
    model.eval()
    
    # Trace the model
    traced_model = torch.jit.trace(model, (data.x, data.edge_index))
    
    # Save
    traced_model.save(output_path)
    
    print(f"Model exported to {output_path}")

# Load TorchScript model
loaded_model = torch.jit.load('gat_model.pt')
output = loaded_model(data.x, data.edge_index)
```

---

## Troubleshooting Guide

### Issue 1: Out of Memory (OOM)

**Solutions:**
1. Reduce hidden channels
2. Use mini-batch sampling
3. Enable mixed precision training
4. Reduce number of attention heads

```python
# Reduce model size
model = GATNet(dataset.num_features, hidden_channels=32, 
              out_channels=dataset.num_classes, heads=4)
```

### Issue 2: Poor Convergence

**Solutions:**
1. Adjust learning rate
2. Add more regularization
3. Use layer normalization
4. Check data preprocessing

```python
# Add LayerNorm
class GATNetWithNorm(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=8, concat=True)
        self.norm1 = torch.nn.LayerNorm(hidden_channels * 8)
        self.gat2 = GATConv(hidden_channels * 8, out_channels, heads=1)
    
    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.norm1(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### Issue 3: Overfitting

**Solutions:**
1. Increase dropout
2. Add weight decay
3. Use early stopping
4. Reduce model capacity

```python
# Increase regularization
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005,
    weight_decay=1e-3  # Increased from 5e-4
)
```

---

## Advanced Techniques

### Knowledge Distillation

```python
class KnowledgeDistillationTrainer:
    """Train small GAT model from large teacher model"""
    
    def __init__(self, teacher_model, student_model, temperature=3.0):
        self.teacher = teacher_model.eval()
        self.student = student_model.train()
        self.temperature = temperature
        self.optimizer = torch.optim.Adam(student_model.parameters())
    
    def train_step(self, x, edge_index, labels, mask):
        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(x, edge_index)
            teacher_logits = teacher_out / self.temperature
        
        # Student predictions
        student_out = self.student(x, edge_index)
        student_logits = student_out / self.temperature
        
        # KL divergence loss
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        )
        
        # Task loss
        task_loss = F.nll_loss(student_out[mask], labels[mask])
        
        # Combined loss
        total_loss = 0.7 * kl_loss + 0.3 * task_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
```

### Contrastive Learning

```python
class GATContrastive(torch.nn.Module):
    """GAT with contrastive learning objective"""
    
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=8, concat=True)
        self.projection = torch.nn.Linear(hidden_channels * 8, 128)
    
    def forward(self, x, edge_index):
        h = F.elu(self.gat(x, edge_index))
        z = F.normalize(self.projection(h), dim=1)
        return z
    
    def contrastive_loss(self, z_i, z_j, temperature=0.07):
        """NT-Xent (normalized temperature-scaled cross entropy)"""
        batch_size = z_i.shape[0]
        
        z = torch.cat([z_i, z_j], dim=0)
        
        # Similarity matrix
        similarity = torch.mm(z, z.t()) / temperature
        
        # Create labels
        labels = torch.arange(batch_size, dtype=torch.long, device=z.device)
        labels = torch.cat([labels, labels], dim=0)
        
        # Mask diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity.masked_fill_(mask, float('-inf'))
        
        loss = F.cross_entropy(similarity, labels)
        return loss
```

---

## Conclusion

This implementation guide covers practical aspects of using Graph Attention Networks, from basic training to advanced techniques. Key takeaways:

1. **Start Simple:** Use PyG's optimized implementations
2. **Monitor Training:** Use early stopping and validation
3. **Optimize for Scale:** Use sampling for large graphs
4. **Interpret Results:** Visualize attention weights
5. **Tune Carefully:** Use automated hyperparameter search

For more details and citations, refer to the main GAT documentation.
