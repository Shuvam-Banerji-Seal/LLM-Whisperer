# Time Series Forecasting Quick Reference & Code Examples

**Version:** 1.0  
**Focus:** Practical Implementation & Quick Lookup

---

## Quick Method Selection

```
Legend:
- Computational Cost: O(n) Linear, O(n²) Quadratic, O(n log n) Log-linear
- Interpretability: ★★★ High, ★★ Medium, ★ Low
- Max Sequence: Length of supported sequences
- Uncertainty: Native uncertainty support
```

| Method | Comp. Cost | Interp. | Max Seq | Uncertainty | Best For |
|--------|-----------|---------|---------|-------------|----------|
| ARIMA | O(n) | ★★★ | 10k | Yes | Trend + AR patterns |
| SARIMA | O(n) | ★★★ | 10k | Yes | Strong seasonality |
| Exp. Smooth | O(n) | ★★★ | 10k | Limited | Simple trends |
| Holt-Winters | O(n) | ★★★ | 10k | Limited | Trend + seasonality |
| LSTM | O(n) | ★ | 100k | Difficult | Large datasets |
| Transformer | O(n²) | ★ | 50k | Possible | Long sequences |
| Attention-LSTM | O(n²) | ★★ | 50k | Possible | Multi-horizon |
| DLinear | O(n) | ★★ | 100k | No | Fast baseline |

---

## Installation Checklists

### Minimal Setup (Classical Only)
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib
```

### Full Setup (Classical + DL)
```bash
# Classical
pip install statsmodels prophet pmdarima

# Deep Learning
pip install torch pytorch-lightning

# Optimization
pip install optuna

# Visualization & Utils
pip install matplotlib seaborn plotly wandb
```

### Development Environment
```bash
# Jupyter
pip install jupyter notebook jupyterlab ipywidgets

# Testing
pip install pytest pytest-cov

# Documentation
pip install sphinx sphinx-rtd-theme
```

---

## Code Snippets by Use Case

### 1. Quick Baseline Comparison

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compare_models(ts_data, train_ratio=0.8, forecast_horizon=12):
    """Compare classical forecasting models"""
    
    # Split data
    train_size = int(len(ts_data) * train_ratio)
    train, test = ts_data[:train_size], ts_data[train_size:]
    
    results = {}
    
    # 1. Naive forecast
    naive_pred = np.full(len(test), train.iloc[-1])
    results['Naive'] = {
        'rmse': np.sqrt(mean_squared_error(test, naive_pred)),
        'mae': mean_absolute_error(test, naive_pred)
    }
    
    # 2. ARIMA(1,1,1)
    try:
        arima_model = ARIMA(train, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(steps=len(test))
        results['ARIMA(1,1,1)'] = {
            'rmse': np.sqrt(mean_squared_error(test, arima_pred)),
            'mae': mean_absolute_error(test, arima_pred)
        }
    except:
        pass
    
    # 3. Exponential Smoothing
    try:
        es_model = ExponentialSmoothing(train, trend='add')
        es_fit = es_model.fit()
        es_pred = es_fit.forecast(steps=len(test))
        results['Exp. Smooth'] = {
            'rmse': np.sqrt(mean_squared_error(test, es_pred)),
            'mae': mean_absolute_error(test, es_pred)
        }
    except:
        pass
    
    # Print comparison
    print("\nModel Comparison:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:20} | RMSE: {metrics['rmse']:8.2f} | "
              f"MAE: {metrics['mae']:8.2f}")
    
    return results

# Usage
ts = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')['value']
results = compare_models(ts)
```

### 2. Auto ARIMA Selection

```python
from pmdarima import auto_arima

def find_best_arima(ts_data, seasonal_period=None):
    """Auto-select ARIMA order"""
    
    seasonal = seasonal_period is not None
    
    model = auto_arima(
        ts_data,
        seasonal=seasonal,
        m=seasonal_period,
        stepwise=True,        # Efficient grid search
        trace=True,           # Print progress
        error_action='warn',
        information_criterion='aic'
    )
    
    print(f"\nBest ARIMA order: {model.order}")
    if seasonal:
        print(f"Best seasonal order: {model.seasonal_order}")
    
    return model

# Usage
ts = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')['value']
best_model = find_best_arima(ts, seasonal_period=12)
forecast = best_model.predict(n_periods=24)
```

### 3. Prophet for Quick Forecasting

```python
from prophet import Prophet
import pandas as pd

def quick_prophet_forecast(df, forecast_periods=30):
    """
    Quick Prophet setup
    df must have columns: 'ds' (datetime), 'y' (value)
    """
    
    # Initialize and fit
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        interval_width=0.95
    )
    
    model.fit(df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_periods)
    
    # Forecast
    forecast_df = model.predict(future)
    
    # Plot
    fig = model.plot(forecast_df)
    model.plot_components(forecast_df)
    
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Usage
df = pd.read_csv('data.csv')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
forecast_df = quick_prophet_forecast(df, forecast_periods=30)
print(forecast_df.tail())
```

### 4. LSTM Time Series

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, 
                 output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

def create_sequences(data, seq_length):
    """Create sliding window sequences"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm_model(data, seq_length=24, epochs=50, batch_size=32):
    """Train LSTM on time series"""
    
    # Normalize
    data_mean, data_std = data.mean(), data.std()
    data_norm = (data - data_mean) / data_std
    
    # Create sequences
    X, y = create_sequences(data_norm, seq_length)
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    
    # DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = SimpleLSTM(input_size=1, hidden_size=32, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    
    return model, (data_mean, data_std)

def lstm_forecast(model, initial_sequence, n_steps, data_norm_params):
    """Make multi-step forecast"""
    data_mean, data_std = data_norm_params
    
    model.eval()
    current_seq = torch.tensor(initial_sequence, dtype=torch.float32)
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Reshape for model
            x = current_seq.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
            
            # Predict
            next_pred = model(x).item()
            predictions.append(next_pred)
            
            # Update sequence (remove first, add prediction)
            current_seq = torch.cat([current_seq[1:], 
                                    torch.tensor([next_pred])])
    
    # Denormalize
    predictions = np.array(predictions) * data_std + data_mean
    return predictions

# Usage
data = np.random.randn(1000) + np.arange(1000) * 0.01  # Example data
model, norm_params = train_lstm_model(data, seq_length=24, epochs=50)

# Forecast
initial_seq = data[-24:]  # Last 24 observations
forecast = lstm_forecast(model, initial_seq, n_steps=12, 
                        data_norm_params=norm_params)
print("Forecast:", forecast)
```

### 5. Quantile Forecasting

```python
import torch
import torch.nn as nn

class QuantileRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, quantiles):
        super().__init__()
        self.quantiles = quantiles
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(quantiles))
        )
    
    def forward(self, x):
        return self.fc(x)

def quantile_loss(predictions, targets, quantiles):
    """Pinball loss for quantile regression"""
    losses = []
    for i, q in enumerate(quantiles):
        diff = targets - predictions[:, i]
        loss = torch.max(q * diff, (q - 1) * diff)
        losses.append(loss.mean())
    return sum(losses) / len(quantiles)

def train_quantile_model(X_train, y_train, quantiles=[0.1, 0.5, 0.9],
                        epochs=100):
    """Train quantile regression"""
    
    model = QuantileRegressor(input_size=X_train.shape[1], 
                             hidden_size=64, quantiles=quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = quantile_loss(predictions, y_train, quantiles)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model

def predict_with_intervals(model, X_test, quantiles=[0.1, 0.5, 0.9]):
    """Predict with prediction intervals"""
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(X_test)
    
    results = {}
    for i, q in enumerate(quantiles):
        results[f'q_{int(q*100)}'] = predictions[:, i].numpy()
    
    return results

# Usage
X_train = np.random.randn(100, 10)
y_train = np.random.randn(100)

model = train_quantile_model(X_train, y_train)

X_test = np.random.randn(20, 10)
intervals = predict_with_intervals(model, X_test)

print("95% Prediction Intervals:")
for i in range(5):
    q10 = intervals['q_10'][i]
    q50 = intervals['q_50'][i]
    q90 = intervals['q_90'][i]
    print(f"[{q10:.4f}, {q90:.4f}] (point: {q50:.4f})")
```

### 6. Multi-Horizon Forecasting

```python
import torch
import torch.nn as nn

class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Separate head for each horizon
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(forecast_horizon)
        ])
    
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use last hidden state for each horizon
        predictions = []
        for head in self.heads:
            pred = head(lstm_out[:, -1, :])
            predictions.append(pred)
        
        return torch.cat(predictions, dim=1)  # (batch, horizon)

def train_multi_horizon(X_train, y_train, forecast_horizon=12, 
                       epochs=50):
    """Train multi-horizon LSTM"""
    
    model = MultiHorizonLSTM(input_size=1, hidden_size=32, 
                            forecast_horizon=forecast_horizon)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

# Usage
seq_length = 24
forecast_horizon = 12
X_train = np.random.randn(100, seq_length, 1)
y_train = np.random.randn(100, forecast_horizon)

model = train_multi_horizon(X_train, y_train, forecast_horizon)

# Forecast
X_test = np.random.randn(1, seq_length, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    forecast = model(X_test).numpy()

print("Multi-horizon forecast:")
for h, val in enumerate(forecast[0]):
    print(f"Step {h+1}: {val:.4f}")
```

### 7. Evaluation Metrics

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred, y_train=None):
    """Comprehensive evaluation"""
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (safe division)
    mape = np.mean(np.abs((y_true - y_pred) / 
                   np.maximum(np.abs(y_true), 1))) * 100
    
    # MASE (scaled by seasonal naive)
    if y_train is not None:
        seasonal_d = np.mean(np.abs(np.diff(y_train)))
        mase = mae / seasonal_d if seasonal_d > 0 else np.inf
    else:
        mase = np.nan
    
    # Directional accuracy
    true_dirs = np.sign(np.diff(y_true))
    pred_dirs = np.sign(np.diff(y_pred))
    direction_acc = np.mean(true_dirs == pred_dirs) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase,
        'Direction_Accuracy': direction_acc
    }

def crps_score(y_true, y_pred_samples):
    """
    CRPS for probabilistic forecasts
    y_pred_samples: (n_samples, n_forecasts)
    """
    # Empirical CDF
    n_samples = y_pred_samples.shape[0]
    y_pred_sorted = np.sort(y_pred_samples, axis=0)
    
    crps = np.zeros(y_pred_samples.shape[1])
    
    for i in range(y_pred_samples.shape[1]):
        # Heaviside step function
        H = (y_pred_sorted[:, i] <= y_true[i]).astype(float)
        
        # CRPS
        crps[i] = np.mean(H) - 0.5 * np.mean(np.abs(
            y_pred_sorted[:-1, i] - y_pred_sorted[1:, i]
        ))
    
    return crps.mean()

# Usage
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
y_train = np.array([0.9, 1.0, 1.1, 2.0, 2.1])

metrics = compute_metrics(y_true, y_pred, y_train)
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric:20}: {value:8.4f}")
```

### 8. Cross-Validation for Time Series

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def time_series_cross_validate(data, model_fn, n_splits=5, 
                              forecast_horizon=12):
    """
    Time series cross-validation (walk-forward)
    
    model_fn: Function that takes (train_data, forecast_horizon)
              and returns predictions
    """
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(data):
        train_data = data[train_idx]
        test_data = data[test_idx]
        
        # Fit model and forecast
        predictions = model_fn(train_data, forecast_horizon)
        
        # Evaluate on test set
        actual = test_data[:len(predictions)]
        mae = np.mean(np.abs(actual - predictions))
        rmse = np.sqrt(np.mean((actual - predictions) ** 2))
        
        scores.append({'MAE': mae, 'RMSE': rmse})
    
    # Average scores
    avg_mae = np.mean([s['MAE'] for s in scores])
    avg_rmse = np.mean([s['RMSE'] for s in scores])
    std_mae = np.std([s['MAE'] for s in scores])
    std_rmse = np.std([s['RMSE'] for s in scores])
    
    print("Time Series CV Results:")
    print(f"MAE:  {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    
    return scores

# Usage
def simple_arima_forecast(train_data, horizon):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train_data, order=(1, 1, 1))
    fitted = model.fit()
    return fitted.forecast(steps=horizon).values

data = np.random.randn(200)
scores = time_series_cross_validate(data, simple_arima_forecast, 
                                   n_splits=5)
```

---

## Performance Optimization Tips

### 1. Reduce Training Time

```python
# Batch processing
batch_size = 32  # Larger batches = faster but less stable

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X = X.to(device)

# Gradient accumulation for large batch effects
accumulation_steps = 4
for i, (X_batch, y_batch) in enumerate(loader):
    output = model(X_batch)
    loss = criterion(output, y_batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Memory Efficiency

```python
# Reduce sequence length for initial experiments
max_seq_length = 100  # Start small, increase if needed

# Use mixed precision (PyTorch)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for X, y in loader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(X)
        loss = criterion(output, y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Delete intermediate results
del X_batch, y_batch  # Explicit memory release
```

### 3. Parallelization

```python
# PyTorch DataParallel (single machine, multiple GPUs)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# For CPU parallelization
import multiprocessing
torch.set_num_threads(multiprocessing.cpu_count())
```

---

## Common Pitfalls & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Poor test performance | Data leakage | Use walk-forward CV, not random split |
| Exploding gradients | High learning rate | Reduce LR, use gradient clipping |
| Model convergence | Too few layers | Increase depth gradually |
| Overfitting | Training too long | Early stopping, dropout, L1/L2 |
| NaNs in predictions | Normalization issues | Normalize before processing |
| Slow convergence | Poor initialization | Use proper weight initialization |
| Seasonality ignored | Wrong model capacity | Check residuals for patterns |

---

## Debugging Checklist

```python
# 1. Check data
print(f"Data shape: {data.shape}")
print(f"Missing values: {data.isnull().sum()}")
print(f"Mean: {data.mean():.4f}, Std: {data.std():.4f}")

# 2. Visualize splits
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(train_data, label='Train')
plt.plot(range(len(train_data), len(train_data)+len(test_data)), 
         test_data, label='Test')
plt.legend()
plt.show()

# 3. Check predictions
print(f"Min pred: {pred.min():.4f}, Max: {pred.max():.4f}")
print(f"Pred same as input? {np.allclose(pred, X[:len(pred)])}")

# 4. Validate metrics
assert rmse >= mae  # RMSE should be >= MAE
assert mape >= 0    # MAPE should be non-negative

# 5. Test reproducibility
np.random.seed(42)
torch.manual_seed(42)
# Should get same results on re-run
```

---

## Additional Resources

- **Papers**: arXiv.org (search "time series forecasting")
- **Datasets**: Kaggle, UCI ML Repository, M4/M5 Competition
- **Libraries**: PyTorch, TensorFlow, statsmodels, prophet
- **Tutorials**: Towards Data Science, Fast.ai, Papers with Code

---

**Last Updated:** April 2026  
**Maintained By:** Time Series Research Group  
**License:** MIT
