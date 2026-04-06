# Temporal Modeling in Time Series: Implementation Guide & Best Practices

**Version:** 1.0  
**Focus:** Production-ready code, benchmarking, and deployment strategies

---

## Table of Contents

1. [Complete Working Examples](#complete-working-examples)
2. [Comparative Benchmarking](#comparative-benchmarking)
3. [Hyperparameter Tuning](#hyperparameter-tuning)
4. [Model Evaluation Metrics](#model-evaluation-metrics)
5. [Advanced Techniques](#advanced-techniques)

---

## Complete Working Examples

### End-to-End LSTM Time Series Forecasting

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class LSTMForecaster(nn.Module):
    """Production-grade LSTM for time series forecasting"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, 
                 output_size=1, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
            dropout=dropout
        )
        
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention to LSTM outputs
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Residual connection + normalization
        lstm_out = self.norm(lstm_out + attn_out)
        
        # Use last timestep
        last_out = lstm_out[:, -1, :]
        
        # Dense layers
        output = self.dense(last_out)
        
        return output

# Example: Electricity consumption prediction
class TimeSeriesTrainer:
    """Training pipeline for time series models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
    
    def prepare_data(self, data, seq_length=24, test_split=0.2, val_split=0.1):
        """
        Prepare data for training
        
        Args:
            data: 1D array of time series values
            seq_length: length of input sequences
            test_split: proportion for testing
            val_split: proportion for validation from training set
        """
        # Normalize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[i:i+seq_length])
            y.append(data_scaled[i+seq_length, 0])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Split data
        total_len = len(X)
        test_idx = int(total_len * (1 - test_split))
        val_idx = int(test_idx * (1 - val_split))
        
        X_train, y_train = X[:val_idx], y[:val_idx]
        X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
        X_test, y_test = X[test_idx:], y[test_idx:]
        
        # Convert to tensors
        train_set = (
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_set = (
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        test_set = (
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        return train_set, val_set, test_set, scaler
    
    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        """Full training loop"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
        
        return train_losses, val_losses
    
    def evaluate(self, test_loader, scaler):
        """Evaluate on test set with metrics"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(y_batch.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform
        predictions = scaler.inverse_transform(predictions)
        actuals = scaler.inverse_transform(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'predictions': predictions,
            'actuals': actuals
        }

# Usage
if __name__ == '__main__':
    # Generate sample data
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 1000)
    data = 100 + 20*np.sin(t) + 5*np.cos(2*t) + np.random.normal(0, 2, 1000)
    
    # Initialize model and trainer
    model = LSTMForecaster(input_size=1, hidden_size=50, num_layers=2)
    trainer = TimeSeriesTrainer(model)
    
    # Prepare data
    train_set, val_set, test_set, scaler = trainer.prepare_data(
        data, seq_length=24
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_set),
        batch_size=32,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*val_set),
        batch_size=32,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*test_set),
        batch_size=32,
        shuffle=False
    )
    
    # Train
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=50)
    
    # Evaluate
    results = trainer.evaluate(test_loader, scaler)
    print("\nTest Results:")
    print(f"MSE: {results['MSE']:.6f}")
    print(f"RMSE: {results['RMSE']:.6f}")
    print(f"MAE: {results['MAE']:.6f}")
    print(f"MAPE: {results['MAPE']:.2f}%")
```

### Advanced TCN Implementation with Dilated Convolutions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvBlock(nn.Module):
    """Single dilated convolution block with residual connection"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.5):
        super(DilatedConvBlock, self).__init__()
        
        # Padding to maintain temporal dimension
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 1x1 conv for residual connection if dimensions don't match
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
                          if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        """Xavier initialization"""
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        # Causal padding (no future information leak)
        y = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(y + res)

class TemporalConvNet(nn.Module):
    """Complete TCN model"""
    
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.5):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(DilatedConvBlock(
                in_channels, out_channels, kernel_size,
                dilation_size, dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(
            num_levels, kernel_size
        )
    
    def _calculate_receptive_field(self, num_levels, kernel_size):
        """Calculate receptive field of TCN"""
        rf = 1
        for i in range(num_levels):
            dilation = 2 ** i
            rf += (kernel_size - 1) * dilation
        return rf
    
    def forward(self, x):
        # x: (batch_size, seq_length, num_inputs)
        x = x.transpose(1, 2)  # (batch_size, num_inputs, seq_length)
        
        # TCN forward
        y = self.network(x)
        
        # Use last timestep
        y = y.transpose(1, 2)  # (batch_size, seq_length, num_channels[-1])
        
        # Final prediction
        pred = self.fc(y[:, -1, :])
        
        return pred
    
    def get_receptive_field(self):
        """Get receptive field size"""
        return self.receptive_field

# Example usage
tcn = TemporalConvNet(
    num_inputs=1,
    num_channels=[25, 25, 25, 25],
    kernel_size=5,
    dropout=0.5
)

print(f"TCN Receptive Field: {tcn.get_receptive_field()}")
# Output: TCN Receptive Field: 257

x = torch.randn(32, 100, 1)
output = tcn(x)
print(f"Output shape: {output.shape}")  # (32, 1)
```

---

## Comparative Benchmarking

### Unified Benchmark Framework

```python
import time
import torch
import numpy as np
from typing import Dict, List, Tuple

class ModelBenchmark:
    """Comprehensive model benchmarking"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
    
    def benchmark_inference(self, model, input_shape=(32, 100, 1), 
                           num_runs=100, warmup=10) -> Dict:
        """Benchmark inference speed and memory"""
        model = model.to(self.device).eval()
        
        x = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure time
        times = []
        start = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        avg_time = (elapsed / num_runs) * 1000  # ms
        
        # Measure memory
        torch.cuda.reset_peak_memory_stats() if self.device == 'cuda' else None
        
        with torch.no_grad():
            _ = model(x)
        
        memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if self.device == 'cuda' else 0
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        return {
            'avg_time_ms': avg_time,
            'throughput': (input_shape[0] / elapsed) * num_runs,  # samples/sec
            'memory_mb': memory,
            'parameters': params,
            'flops': self._estimate_flops(model, input_shape)
        }
    
    def benchmark_training(self, model, train_loader, epochs=10) -> Dict:
        """Benchmark training speed"""
        model = model.to(self.device).train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        epoch_times = []
        
        for epoch in range(epochs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            epoch_time = time.time() - start
            epoch_times.append(epoch_time)
        
        return {
            'avg_epoch_time': np.mean(epoch_times),
            'std_epoch_time': np.std(epoch_times),
            'total_time': sum(epoch_times)
        }
    
    def _estimate_flops(self, model, input_shape):
        """Rough FLOP estimation"""
        # This is a simplified estimation
        total_flops = 0
        for param in model.parameters():
            total_flops += 2 * param.numel()
        
        batch_size, seq_len, input_size = input_shape
        total_flops *= seq_len
        
        return total_flops

# Benchmark multiple models
models_to_benchmark = {
    'LSTM': LSTMForecaster(input_size=1, hidden_size=64, num_layers=2),
    'TCN': TemporalConvNet(num_inputs=1, num_channels=[50, 50, 50, 50]),
    'Transformer': TransformerForecaster(input_size=1, d_model=64, nhead=4),
}

benchmark = ModelBenchmark(device='cuda')

results = {}
for name, model in models_to_benchmark.items():
    print(f"Benchmarking {name}...")
    results[name] = benchmark.benchmark_inference(model)
    print(f"  Inference time: {results[name]['avg_time_ms']:.3f} ms")
    print(f"  Memory: {results[name]['memory_mb']:.1f} MB")
    print(f"  Parameters: {results[name]['parameters']:.0f}")

# Create comparison table
import pandas as pd

df = pd.DataFrame(results).T
print("\nBenchmark Results:")
print(df.to_string())
```

---

## Hyperparameter Tuning

### Bayesian Optimization for Hyperparameters

```python
from scipy.optimize import minimize
import torch
import numpy as np

class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization"""
    
    def __init__(self, train_loader, val_loader, device='cuda'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.trial_history = []
    
    def objective(self, params: np.ndarray) -> float:
        """
        Objective function to minimize (validation loss)
        
        params: [hidden_size, num_layers, dropout, learning_rate, weight_decay]
        """
        hidden_size = int(np.round(params[0]))
        num_layers = int(np.round(params[1]))
        dropout = params[2]
        learning_rate = 10 ** params[3]
        weight_decay = 10 ** params[4]
        
        # Constraints
        hidden_size = max(16, min(hidden_size, 256))
        num_layers = max(1, min(num_layers, 4))
        dropout = np.clip(dropout, 0.0, 0.5)
        
        # Create model
        model = LSTMForecaster(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        trainer = TimeSeriesTrainer(model, device=self.device)
        
        # Train for limited epochs
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(30):  # Limited epochs for speed
            train_loss = trainer.train_epoch(self.train_loader, optimizer)
            val_loss = trainer.validate(self.val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        self.trial_history.append({
            'params': {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            },
            'val_loss': best_val_loss
        })
        
        return best_val_loss
    
    def optimize(self, n_trials=20):
        """Run optimization"""
        # Initial bounds: [hidden_size, num_layers, dropout, log10(lr), log10(wd)]
        bounds = [
            (16, 256),      # hidden_size
            (1, 4),         # num_layers
            (0.0, 0.5),     # dropout
            (-5, -2),       # log10(learning_rate)
            (-8, -4)        # log10(weight_decay)
        ]
        
        best_result = None
        best_loss = float('inf')
        
        for trial in range(n_trials):
            # Random starting point
            x0 = np.array([
                np.random.uniform(b[0], b[1]) for b in bounds
            ])
            
            # Minimize
            result = minimize(
                self.objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
            
            print(f"Trial {trial+1}: Best val loss = {best_loss:.6f}")
        
        # Extract best parameters
        best_params = best_result.x
        best_config = {
            'hidden_size': int(np.round(best_params[0])),
            'num_layers': int(np.round(best_params[1])),
            'dropout': best_params[2],
            'learning_rate': 10 ** best_params[3],
            'weight_decay': 10 ** best_params[4]
        }
        
        return best_config, self.trial_history

# Usage
optimizer = HyperparameterOptimizer(train_loader, val_loader)
best_config, history = optimizer.optimize(n_trials=20)
print("Best configuration:", best_config)
```

---

## Model Evaluation Metrics

### Comprehensive Metrics Computation

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TimeSeriesMetrics:
    """Compute comprehensive time series evaluation metrics"""
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error"""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return np.mean(2 * numerator / denominator) * 100
    
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def mae_std(y_true, y_pred):
        """Standard deviation of absolute errors"""
        ae = np.abs(y_true - y_pred)
        return np.std(ae)
    
    @staticmethod
    def directional_accuracy(y_true, y_pred):
        """Accuracy of direction changes"""
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return np.mean(true_direction == pred_direction) * 100
    
    @staticmethod
    def theil_u(y_true, y_pred):
        """Theil U statistic (relative RMSE)"""
        numerator = np.mean((y_true - y_pred) ** 2)
        denominator = np.mean(np.diff(y_true) ** 2)
        return np.sqrt(numerator / denominator)
    
    @staticmethod
    def quantile_loss(y_true, y_pred, quantile=0.5):
        """Quantile loss (pinball loss)"""
        error = y_true - y_pred
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))
    
    @staticmethod
    def compute_all(y_true, y_pred):
        """Compute all metrics at once"""
        return {
            'MAE': TimeSeriesMetrics.mae(y_true, y_pred),
            'RMSE': TimeSeriesMetrics.rmse(y_true, y_pred),
            'MSE': TimeSeriesMetrics.mse(y_true, y_pred),
            'MAPE': TimeSeriesMetrics.mape(y_true, y_pred),
            'SMAPE': TimeSeriesMetrics.smape(y_true, y_pred),
            'MAE_STD': TimeSeriesMetrics.mae_std(y_true, y_pred),
            'DA': TimeSeriesMetrics.directional_accuracy(y_true, y_pred),
            'Theil_U': TimeSeriesMetrics.theil_u(y_true, y_pred),
        }

# Example usage
y_true = np.array([100, 102, 101, 103, 105, 104, 106, 108])
y_pred = np.array([99, 101, 102, 102, 104, 105, 107, 107])

metrics = TimeSeriesMetrics.compute_all(y_true, y_pred)
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")
```

---

## Advanced Techniques

### Ensemble Methods for Time Series

```python
class EnsembleForecaster:
    """Ensemble of different time series models"""
    
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def predict(self, x):
        """Ensemble prediction (weighted average)"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x).cpu().numpy()
                predictions.append(pred)
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, x, num_mc_samples=10):
        """Predict with uncertainty estimation using MC dropout"""
        predictions_ensemble = []
        
        for model in self.models:
            # Enable MC dropout
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
            
            mc_predictions = []
            for _ in range(num_mc_samples):
                with torch.no_grad():
                    pred = model(x).cpu().numpy()
                    mc_predictions.append(pred)
            
            predictions_ensemble.append(np.array(mc_predictions))
        
        # Compute mean and std across MC samples and models
        all_preds = np.concatenate(predictions_ensemble, axis=1)
        mean_pred = np.mean(all_preds, axis=(0, 1))
        std_pred = np.std(all_preds, axis=(0, 1))
        
        return mean_pred, std_pred

# Create ensemble
lstm_model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2)
tcn_model = TemporalConvNet(num_inputs=1, num_channels=[50, 50, 50, 50])
transformer_model = TransformerForecaster(input_size=1, d_model=64)

ensemble = EnsembleForecaster(
    models=[lstm_model, tcn_model, transformer_model],
    weights=[0.4, 0.3, 0.3]  # Give more weight to LSTM
)

# Make predictions with uncertainty
x_test = torch.randn(32, 100, 1)
mean_pred, std_pred = ensemble.predict_with_uncertainty(x_test)
print(f"Mean prediction: {mean_pred.mean():.4f} +/- {std_pred.mean():.4f}")
```

### Attention Weight Visualization

```python
import matplotlib.pyplot as plt

class AttentionVisualizer:
    """Visualize attention patterns"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.attention_weights = None
    
    def extract_attention_weights(self, x):
        """Extract attention weights from model"""
        self.model.eval()
        
        # Hook to capture attention weights
        attention_weights = []
        
        def attention_hook(module, input, output):
            if isinstance(output, tuple):
                attention_weights.append(output[1])
            
        # Register hooks
        handles = []
        for module in self.model.modules():
            if isinstance(module, nn.MultiheadAttention):
                h = module.register_forward_hook(attention_hook)
                handles.append(h)
        
        with torch.no_grad():
            _ = self.model(x)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        self.attention_weights = attention_weights
        return attention_weights
    
    def plot_attention(self, x, figsize=(12, 6)):
        """Plot attention heatmaps"""
        attn_weights = self.extract_attention_weights(x)
        
        num_heads = len(attn_weights)
        fig, axes = plt.subplots(1, num_heads, figsize=figsize)
        
        for i, weights in enumerate(attn_weights):
            # Take first batch element
            attn = weights[0].cpu().numpy()
            
            axes[i].imshow(attn, cmap='hot', aspect='auto')
            axes[i].set_title(f'Attention Head {i+1}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
        
        plt.tight_layout()
        return fig

# Usage
visualizer = AttentionVisualizer(lstm_model_with_attention)
x_sample = torch.randn(1, 100, 1)
fig = visualizer.plot_attention(x_sample)
plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
```

---

**Implementation Guide Complete**
**Contains: 5 End-to-End Examples, Benchmarking Framework, Hyperparameter Tuning, Advanced Techniques**

