# Knowledge Graph Embedding: Complete Implementation Guide

## Complete Working Example: Training RotatE on FB15k-237

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KGDataset(Dataset):
    """Custom dataset for knowledge graph triples"""
    
    def __init__(self, triples, entity2id, relation2id, num_entities, num_relations):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_entities = num_entities
        self.num_relations = num_relations
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        return torch.tensor(h), torch.tensor(r), torch.tensor(t)


class RotatEModel(nn.Module):
    """RotatE: Knowledge Graph Embedding by Relational Rotation"""
    
    def __init__(self, num_entities, num_relations, embedding_dim=256, 
                 margin=12.0, eps=2.0):
        super(RotatEModel, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.eps = eps
        
        # Entity embeddings (mapped to [0, 1] for complex space)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embeddings (angles in [0, 2π])
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize with proper bounds
        self.entity_embeddings.weight.data = nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-1.0,
            b=1.0
        )
        
        self.relation_embeddings.weight.data = nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-1.0,
            b=1.0
        )
    
    def distance(self, head, relation, tail):
        """
        Compute RotatE distance
        
        The core equation is:
        distance = ||head * exp(i*relation) - tail||
        
        In component form:
        distance = sqrt(sum_i (h_real*cos(r) - h_imag*sin(r) - t_real)^2 
                             + (h_real*sin(r) + h_imag*cos(r) - t_imag)^2)
        """
        # head and relation are in [-1, 1], map to actual representation
        pi = 3.14159265358979323846
        
        # Compute rotation angles from relation embeddings
        relation = torch.abs(relation)  # Ensure positive
        
        # Split entity embeddings into real and imaginary components
        head_real = head[..., :self.embedding_dim // 2]
        head_imag = head[..., self.embedding_dim // 2:]
        
        tail_real = tail[..., :self.embedding_dim // 2]
        tail_imag = tail[..., self.embedding_dim // 2:]
        
        # Rotation angles from relations
        relation_angles = pi * relation
        
        # Compute cos and sin of angles
        cos_relation = torch.cos(relation_angles)
        sin_relation = torch.sin(relation_angles)
        
        # Apply rotation: (h_real + i*h_imag) * exp(i*theta)
        # = h_real*cos(theta) - h_imag*sin(theta) + i*(h_real*sin(theta) + h_imag*cos(theta))
        
        rotated_real = (head_real * cos_relation) - (head_imag * sin_relation)
        rotated_imag = (head_real * sin_relation) + (head_imag * cos_relation)
        
        # Compute L2 distance
        real_diff = rotated_real - tail_real
        imag_diff = rotated_imag - tail_imag
        
        distance = torch.sqrt(torch.sum(real_diff ** 2 + imag_diff ** 2, dim=-1))
        
        return distance
    
    def forward(self, head, relation, tail, mode='train'):
        """
        Forward pass for RotatE
        
        Args:
            head: Head entity indices
            relation: Relation indices
            tail: Tail entity indices
            mode: 'train' or 'eval'
        
        Returns:
            distance: Negative distance score (for ranking loss)
        """
        # Get embeddings
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        
        # Normalize entity embeddings to unit sphere
        head_emb = torch.nn.functional.normalize(head_emb, p=2, dim=-1)
        tail_emb = torch.nn.functional.normalize(tail_emb, p=2, dim=-1)
        
        # Compute distance
        distance = self.distance(head_emb, relation_emb, tail_emb)
        
        return distance
    
    def regularization_loss(self, weight=0.0001):
        """L2 regularization on entity embeddings"""
        return weight * (
            torch.sum(self.entity_embeddings.weight ** 2) +
            torch.sum(self.relation_embeddings.weight ** 2)
        )


class NegativeSampler:
    """Generate negative samples for training"""
    
    def __init__(self, num_entities, corrupt_head_prob=0.5):
        self.num_entities = num_entities
        self.corrupt_head_prob = corrupt_head_prob
    
    def sample(self, batch_size):
        """
        Sample which position to corrupt (head or tail)
        Returns array where True means corrupt head, False means corrupt tail
        """
        corrupt_head = np.random.binomial(1, self.corrupt_head_prob, batch_size)
        return corrupt_head.astype(bool)
    
    def get_negative_sample(self, batch_h, batch_r, batch_t, corrupt_head):
        """
        Generate negative samples
        
        Args:
            batch_h, batch_r, batch_t: Positive triple batch
            corrupt_head: Boolean array indicating which position to corrupt
        
        Returns:
            Corrupted batch
        """
        neg_batch_h = batch_h.clone()
        neg_batch_t = batch_t.clone()
        
        # Sample random entities
        random_entities = torch.randint(0, self.num_entities, batch_h.shape)
        
        # Apply corruption
        neg_batch_h[corrupt_head] = random_entities[corrupt_head]
        neg_batch_t[~corrupt_head] = random_entities[~corrupt_head]
        
        return neg_batch_h, batch_r, neg_batch_t


class RotatETrainer:
    """Training loop for RotatE model"""
    
    def __init__(self, model, num_entities, device='cuda', margin=12.0):
        self.model = model
        self.num_entities = num_entities
        self.device = device
        self.margin = margin
        self.negative_sampler = NegativeSampler(num_entities)
        
        logger.info(f"Device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    def train_step(self, batch_h, batch_r, batch_t, optimizer):
        """
        Single training step
        
        Uses margin-based ranking loss:
        L = max(0, margin + distance(pos) - distance(neg))
        """
        batch_h = batch_h.to(self.device)
        batch_r = batch_r.to(self.device)
        batch_t = batch_t.to(self.device)
        
        # Get positive scores
        pos_distance = self.model(batch_h, batch_r, batch_t)
        
        # Generate negatives
        corrupt_head = self.negative_sampler.sample(batch_h.shape[0])
        corrupt_head = torch.tensor(corrupt_head, dtype=torch.bool, device=self.device)
        
        neg_h = batch_h.clone()
        neg_t = batch_t.clone()
        
        random_entities = torch.randint(
            0, self.num_entities, 
            batch_h.shape, 
            device=self.device
        )
        
        neg_h[corrupt_head] = random_entities[corrupt_head]
        neg_t[~corrupt_head] = random_entities[~corrupt_head]
        
        # Get negative scores
        neg_distance = self.model(neg_h, batch_r, neg_t)
        
        # Margin-based ranking loss
        loss = torch.relu(self.margin + pos_distance - neg_distance).mean()
        
        # Add regularization
        reg_loss = self.model.regularization_loss()
        total_loss = loss + reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Normalize embeddings
        self.model.entity_embeddings.weight.data = torch.nn.functional.normalize(
            self.model.entity_embeddings.weight.data, p=2, dim=1
        )
        
        return total_loss.item(), loss.item(), reg_loss.item()
    
    def train(self, train_loader, num_epochs=100, lr=0.001, 
              valid_loader=None, test_triples=None):
        """
        Complete training loop
        
        Args:
            train_loader: DataLoader for training
            num_epochs: Number of epochs
            lr: Learning rate
            valid_loader: Optional validation loader
            test_triples: Optional test triples for evaluation
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        
        best_mrr = 0.0
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_reg = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, (batch_h, batch_r, batch_t) in enumerate(pbar):
                loss, ranking_loss, reg_loss = self.train_step(
                    batch_h, batch_r, batch_t, optimizer
                )
                
                total_loss += loss
                total_reg += reg_loss
                
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")
            
            # Validation
            if valid_loader is not None and (epoch + 1) % 10 == 0:
                metrics = self.evaluate(valid_loader, self.num_entities)
                logger.info(f"Validation: MRR={metrics['mrr']:.4f}, "
                          f"H@1={metrics['hits@1']:.3f}, "
                          f"H@10={metrics['hits@10']:.3f}")
                
                if metrics['mrr'] > best_mrr:
                    best_mrr = metrics['mrr']
                    torch.save(self.model.state_dict(), 'best_rotate_model.pt')
                    logger.info("Best model saved!")
        
        return best_mrr
    
    def evaluate(self, test_loader, num_entities, filtered=False):
        """
        Evaluate link prediction performance
        
        Args:
            test_loader: DataLoader for test data
            num_entities: Total number of entities
            filtered: Whether to use filtered evaluation
        
        Returns:
            Metrics dict with MRR, Hits@K
        """
        self.model.eval()
        
        mrr_sum = 0.0
        hits1_count = 0
        hits3_count = 0
        hits10_count = 0
        
        with torch.no_grad():
            for batch_h, batch_r, batch_t in test_loader:
                batch_h = batch_h.to(self.device)
                batch_r = batch_r.to(self.device)
                batch_t = batch_t.to(self.device)
                
                # Evaluate tail ranking
                h_emb = self.model.entity_embeddings(batch_h)
                r_emb = self.model.relation_embeddings(batch_r)
                t_emb = self.model.entity_embeddings.weight
                
                h_emb = torch.nn.functional.normalize(h_emb, p=2, dim=-1)
                
                # Compute distance to all entities
                head_real = h_emb[..., :self.model.embedding_dim // 2]
                head_imag = h_emb[..., self.model.embedding_dim // 2:]
                
                tail_real = t_emb[..., :self.model.embedding_dim // 2]
                tail_imag = t_emb[..., self.model.embedding_dim // 2:]
                
                pi = 3.14159265358979323846
                r_emb = torch.abs(r_emb)
                relation_angles = pi * r_emb
                
                cos_relation = torch.cos(relation_angles)
                sin_relation = torch.sin(relation_angles)
                
                rotated_real = (head_real.unsqueeze(1) * cos_relation.unsqueeze(1)) - \
                              (head_imag.unsqueeze(1) * sin_relation.unsqueeze(1))
                rotated_imag = (head_real.unsqueeze(1) * sin_relation.unsqueeze(1)) + \
                              (head_imag.unsqueeze(1) * cos_relation.unsqueeze(1))
                
                real_diff = rotated_real - tail_real.unsqueeze(0)
                imag_diff = rotated_imag - tail_imag.unsqueeze(0)
                
                distances = torch.sqrt(torch.sum(real_diff ** 2 + imag_diff ** 2, dim=-1))
                
                # Rank true tail
                for i in range(batch_h.shape[0]):
                    true_tail_idx = batch_t[i].item()
                    rank = (distances[i] < distances[i, true_tail_idx]).sum().item() + 1
                    
                    mrr_sum += 1.0 / rank
                    if rank <= 1:
                        hits1_count += 1
                    if rank <= 3:
                        hits3_count += 1
                    if rank <= 10:
                        hits10_count += 1
        
        num_samples = len(test_loader.dataset)
        
        return {
            'mrr': mrr_sum / num_samples,
            'hits@1': hits1_count / num_samples,
            'hits@3': hits3_count / num_samples,
            'hits@10': hits10_count / num_samples
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Complete training example"""
    
    # Hyperparameters
    embedding_dim = 256
    margin = 12.0
    learning_rate = 0.001
    batch_size = 1024
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load knowledge graph (example with FB15k-237)
    logger.info("Loading knowledge graph...")
    # This would load actual data:
    # train_triples = load_triples('fb15k-237/train.txt')
    # valid_triples = load_triples('fb15k-237/valid.txt')
    # test_triples = load_triples('fb15k-237/test.txt')
    
    # For demo, create dummy data
    num_entities = 14541
    num_relations = 237
    num_train = 272115
    
    train_triples = np.random.randint(
        0, [num_entities, num_relations, num_entities],
        size=(num_train, 3)
    )
    
    entity2id = {i: i for i in range(num_entities)}
    relation2id = {i: i for i in range(num_relations)}
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = KGDataset(
        train_triples, entity2id, relation2id, 
        num_entities, num_relations
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = RotatEModel(
        num_entities, num_relations,
        embedding_dim=embedding_dim,
        margin=margin
    ).to(device)
    
    # Create trainer
    trainer = RotatETrainer(model, num_entities, device=device, margin=margin)
    
    # Train
    logger.info("Starting training...")
    best_mrr = trainer.train(
        train_loader,
        num_epochs=num_epochs,
        lr=learning_rate
    )
    
    logger.info(f"Training complete! Best MRR: {best_mrr:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'rotate_model_final.pt')
    logger.info("Model saved to rotate_model_final.pt")


# ============================================================================
# INFERENCE EXAMPLES
# ============================================================================

class RotatEInference:
    """Inference utilities for trained RotatE model"""
    
    def __init__(self, model, entity2id, relation2id, device='cuda'):
        self.model = model
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.device = device
        self.model.eval()
    
    def predict_tail(self, head_name, relation_name, k=10):
        """
        Predict top-k tail entities for given head and relation
        
        Example:
            top_tails = model.predict_tail('Steve Jobs', 'born_in', k=5)
        """
        head_id = self.entity2id[head_name]
        relation_id = self.relation2id[relation_name]
        
        head_tensor = torch.tensor([head_id]).to(self.device)
        relation_tensor = torch.tensor([relation_id]).to(self.device)
        
        with torch.no_grad():
            # Get embeddings
            h_emb = self.model.entity_embeddings(head_tensor)
            r_emb = self.model.relation_embeddings(relation_tensor)
            all_t_emb = self.model.entity_embeddings.weight
            
            # Compute distances
            distances = self.model.distance(
                h_emb.expand_as(all_t_emb),
                r_emb.expand_as(all_t_emb),
                all_t_emb
            )
            
            # Get top-k
            top_distances, top_indices = torch.topk(
                distances, k, largest=False
            )
        
        # Convert back to names
        id2entity = {v: k for k, v in self.entity2id.items()}
        results = [
            (id2entity[idx.item()], -dist.item())
            for idx, dist in zip(top_indices, top_distances)
        ]
        
        return results
    
    def predict_head(self, tail_name, relation_name, k=10):
        """Predict top-k head entities"""
        # Similar to predict_tail
        pass
    
    def link_probability(self, head_name, relation_name, tail_name):
        """
        Estimate probability that triple is true
        
        Uses sigmoid to convert distance to probability:
        P(true) = 1 / (1 + exp(distance))
        """
        head_id = self.entity2id[head_name]
        relation_id = self.relation2id[relation_name]
        tail_id = self.entity2id[tail_name]
        
        head_tensor = torch.tensor([head_id]).to(self.device)
        relation_tensor = torch.tensor([relation_id]).to(self.device)
        tail_tensor = torch.tensor([tail_id]).to(self.device)
        
        with torch.no_grad():
            distance = self.model(head_tensor, relation_tensor, tail_tensor)
            
            # Convert distance to probability
            probability = torch.sigmoid(-distance / 12.0).item()
        
        return probability


if __name__ == '__main__':
    main()
```

---

## Performance Optimization Techniques

### 1. Batch Processing Optimization

```python
def optimize_batch_inference(model, batch_size=10000):
    """
    Optimize inference for large-scale ranking
    
    Key techniques:
    1. GPU batching
    2. Mixed precision
    3. Gradient checkpointing
    """
    model.eval()
    
    with torch.cuda.amp.autocast():  # Mixed precision
        with torch.no_grad():
            # Process in chunks to avoid OOM
            for batch_start in range(0, num_entities, batch_size):
                batch_end = min(batch_start + batch_size, num_entities)
                batch_indices = torch.arange(batch_start, batch_end)
                
                # Your ranking logic here
                pass
```

### 2. Early Stopping Implementation

```python
class EarlyStopping:
    """Stop training when validation metric plateaus"""
    
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def step(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric > self.best_score + self.min_delta:
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False
```

### 3. Learning Rate Scheduling

```python
# Warmup + decay schedule
def create_lr_scheduler(optimizer, warmup_steps=1000, total_steps=100000):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.1, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## Comparative Evaluation Framework

```python
def compare_kge_models(models_config, test_triples, num_entities):
    """
    Compare multiple KGE models on same dataset
    
    Args:
        models_config: Dict of model_name -> (model_class, hyperparams)
        test_triples: Test data
        num_entities: Number of entities
    
    Returns:
        Comparison dataframe
    """
    results = []
    
    for model_name, (ModelClass, params) in models_config.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Load pre-trained model
        model = ModelClass(**params)
        model.load_state_dict(torch.load(f'checkpoints/{model_name}.pt'))
        
        # Evaluate
        metrics = evaluate_model(model, test_triples, num_entities)
        
        results.append({
            'Model': model_name,
            'MRR': metrics['mrr'],
            'Hits@1': metrics['hits@1'],
            'Hits@3': metrics['hits@3'],
            'Hits@10': metrics['hits@10'],
            'Parameters': sum(p.numel() for p in model.parameters())
        })
    
    import pandas as pd
    df = pd.DataFrame(results)
    return df.sort_values('MRR', ascending=False)
```

---

## Hyperparameter Tuning Guide

```python
def tune_hyperparameters_optuna():
    """Optuna-based hyperparameter optimization"""
    import optuna
    
    def objective(trial):
        # Suggest hyperparameters
        embedding_dim = trial.suggest_int('embedding_dim', 64, 512, step=64)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        margin = trial.suggest_float('margin', 1.0, 30.0)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
        
        # Train model
        model = RotatEModel(
            num_entities, num_relations,
            embedding_dim=embedding_dim,
            margin=margin
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        trainer = RotatETrainer(model, num_entities)
        best_mrr = trainer.train(
            train_loader,
            num_epochs=50,
            lr=learning_rate,
            valid_loader=valid_loader
        )
        
        return best_mrr
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    return study.best_params
```

---

End of Implementation Guide
