# Time Series Anomaly Detection: Implementation Guide

**Version:** 1.0  
**Last Updated:** April 2026

---

## Quick Start: 5 Methods in 5 Minutes

### 1. Z-Score (Statistical, Real-time)
```python
import numpy as np
from scipy import stats

def detect_zscore(ts, threshold=3.0):
    z_scores = np.abs(stats.zscore(ts))
    return z_scores > threshold, z_scores
```

### 2. IQR (Statistical, Robust)
```python
def detect_iqr(ts, multiplier=1.5):
    q1, q3 = np.percentile(ts, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier*iqr, q3 + multiplier*iqr
    return (ts < lower) | (ts > upper), ts
```

### 3. Isolation Forest (Tree-based, Scalable)
```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05)
predictions = model.fit_predict(X)  # -1 = anomaly, 1 = normal
anomalies = predictions == -1
```

### 4. LOF (Density-based, Local)
```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
anomalies = lof.fit_predict(X) == -1
```

### 5. LSTM Autoencoder (Deep Learning, Complex Patterns)
```python
# See main guide for full implementation
# Key steps: Build → Fit on normal data → Detect by reconstruction error
```

---

## Implementation Checklist

### Phase 1: Data Preparation
- [ ] Load and explore time series
- [ ] Handle missing values
- [ ] Normalize/standardize features
- [ ] Split train/test (70/30)
- [ ] Document data characteristics

### Phase 2: Baseline Detection
- [ ] Implement statistical methods (Z-score, IQR)
- [ ] Test on known anomalies
- [ ] Establish baseline performance
- [ ] Evaluate false positive rate

### Phase 3: Advanced Methods
- [ ] Try Isolation Forest
- [ ] Try LOF if multivariate
- [ ] Compare performance
- [ ] Select best 2-3 methods

### Phase 4: Deep Learning (Optional)
- [ ] Prepare data windows
- [ ] Build LSTM-AE or VAE
- [ ] Train on normal data
- [ ] Evaluate reconstruction error

### Phase 5: Optimization
- [ ] Hyperparameter tuning
- [ ] Threshold optimization
- [ ] Ensemble methods
- [ ] Final validation

### Phase 6: Production
- [ ] Model serialization
- [ ] Real-time pipeline
- [ ] Monitoring setup
- [ ] Performance tracking

---

## Common Implementation Patterns

### Pattern 1: Fixed Threshold Detection
```python
class FixedThresholdDetector:
    def __init__(self, threshold):
        self.threshold = threshold
    
    def predict(self, scores):
        return scores > self.threshold
```

### Pattern 2: Percentile-Based Detection
```python
class PercentileDetector:
    def fit(self, scores, percentile=95):
        self.threshold = np.percentile(scores, percentile)
    
    def predict(self, scores):
        return scores > self.threshold
```

### Pattern 3: Adaptive Threshold
```python
class AdaptiveThresholdDetector:
    def __init__(self, window_size=50, alpha=0.3):
        self.window_size = window_size
        self.alpha = alpha
    
    def predict(self, scores):
        thresholds = []
        for i in range(len(scores)):
            window = scores[max(0, i-self.window_size):i]
            threshold = np.mean(window) + 2*np.std(window)
            thresholds.append(threshold)
        return scores > np.array(thresholds)
```

### Pattern 4: Ensemble Detection
```python
class EnsembleAnomalyDetector:
    def __init__(self, methods, weights=None):
        self.methods = methods
        self.weights = weights or [1/len(methods)] * len(methods)
    
    def predict(self, X):
        scores = []
        for method, weight in zip(self.methods, self.weights):
            scores.append(weight * method.predict_proba(X)[:, 1])
        
        ensemble_score = np.mean(scores, axis=0)
        return ensemble_score > 0.5
```

---

## Debugging Guide

### Issue: Too Many False Positives
**Symptoms:** High FP rate, flagging normal variations as anomalies

**Solutions:**
1. Increase threshold
2. Use adaptive thresholding
3. Add context (seasonal patterns)
4. Use ensemble with voting
5. Add preprocessing (smoothing, detrending)

```python
# Solution: Increase threshold
detector = IsolationForest(contamination=0.02)  # Reduce to 2%

# Solution: Smooth before detection
from scipy.ndimage import gaussian_filter1d
ts_smooth = gaussian_filter1d(ts, sigma=5)
anomalies = detector.predict(ts_smooth) == -1
```

### Issue: Missed Anomalies (High False Negatives)
**Symptoms:** Known anomalies not detected

**Solutions:**
1. Decrease threshold
2. Use different method
3. Extend window size
4. Add domain-specific features
5. Use semi-supervised learning

```python
# Solution: Lower contamination estimate
detector = IsolationForest(contamination=0.1)  # 10% instead of 5%

# Solution: Try different method
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=10)
```

### Issue: Slow Inference
**Symptoms:** Real-time requirements not met

**Solutions:**
1. Use simpler method (Z-score, EWMA)
2. Reduce feature dimensions
3. Use batch processing
4. Implement caching
5. Deploy on GPU

```python
# Solution: Use faster method
def fast_zscore(ts):
    mean, std = np.mean(ts[-100:]), np.std(ts[-100:])
    return np.abs((ts[-1] - mean) / (std + 1e-10))

# Solution: Batch processing
scores = np.array([detector.predict(X[i:i+100]) for i in range(0, len(X), 100)])
```

### Issue: High Memory Usage
**Symptoms:** Out-of-memory errors with large datasets

**Solutions:**
1. Process in mini-batches
2. Use online learning
3. Downsample data
4. Use sparse representations

```python
# Solution: Mini-batch processing
def predict_batch(X, model, batch_size=1000):
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        predictions.append(model.predict(batch))
    return np.concatenate(predictions)

# Solution: Online learning with streaming
from river import anomaly
ad = anomaly.IsolationForest()
for x in stream:
    score = ad.score_one(x)
    ad.learn_one(x)
```

---

## Testing & Validation

### Unit Tests
```python
import pytest

class TestAnomalyDetector:
    def setup_method(self):
        self.detector = MyAnomalyDetector()
        self.normal_data = np.random.randn(1000)
        self.anomalies = np.concatenate([
            self.normal_data[:900],
            np.random.randn(100) * 5
        ])
    
    def test_normal_data_detection(self):
        """Should detect < 5% of normal data as anomalies"""
        predictions = self.detector.predict(self.normal_data)
        false_positive_rate = predictions.sum() / len(self.normal_data)
        assert false_positive_rate < 0.05
    
    def test_anomaly_detection(self):
        """Should detect > 80% of injected anomalies"""
        predictions = self.detector.predict(self.anomalies)
        anomaly_indices = np.where(predictions)[0]
        detected_anomalies = np.sum(anomaly_indices >= 900)
        detection_rate = detected_anomalies / 100
        assert detection_rate > 0.8
    
    def test_edge_cases(self):
        """Should handle edge cases gracefully"""
        # Empty array
        assert len(self.detector.predict(np.array([]))) == 0
        
        # Single point
        result = self.detector.predict(np.array([1.0]))
        assert isinstance(result, np.ndarray)
        
        # All identical values
        result = self.detector.predict(np.ones(100))
        assert isinstance(result, np.ndarray)
```

### Integration Tests
```python
def test_end_to_end_pipeline():
    """Test complete anomaly detection pipeline"""
    # Load data
    data = load_time_series()
    
    # Split
    train, test = train_test_split(data)
    
    # Train
    detector = AnomalyDetector()
    detector.fit(train)
    
    # Predict
    predictions = detector.predict(test)
    scores = detector.predict_proba(test)
    
    # Validate
    assert len(predictions) == len(test)
    assert np.all((scores >= 0) & (scores <= 1))
    assert predictions.dtype == bool
```

---

## Performance Comparison

### Method Complexity Analysis

```python
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

def benchmark_methods(X, n_runs=5):
    results = {}
    
    # Z-Score
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = np.abs(stats.zscore(X))
        times.append(time.time() - start)
    results['zscore'] = np.mean(times)
    
    # Isolation Forest
    times = []
    for _ in range(n_runs):
        model = IsolationForest()
        start = time.time()
        model.fit_predict(X)
        times.append(time.time() - start)
    results['isolation_forest'] = np.mean(times)
    
    # LOF
    times = []
    for _ in range(n_runs):
        model = LocalOutlierFactor()
        start = time.time()
        model.fit_predict(X)
        times.append(time.time() - start)
    results['lof'] = np.mean(times)
    
    return results

# Example
X = np.random.randn(10000, 50)
times = benchmark_methods(X)
for method, t in times.items():
    print(f"{method}: {t*1000:.2f}ms")
```

### Accuracy Comparison
```python
def compare_accuracy(X, y_true, methods):
    results = {}
    
    for name, detector in methods.items():
        y_pred = detector.fit_predict(X)
        
        # Convert -1/1 to 0/1
        if -1 in y_pred:
            y_pred = (y_pred == -1).astype(int)
        
        from sklearn.metrics import f1_score, roc_auc_score
        results[name] = {
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred)
        }
    
    return results
```

---

## Configuration Management

### Configuration File (config.yaml)
```yaml
detection:
  method: isolation_forest  # or zscore, lof, ewma, lstm
  threshold_method: percentile  # or fixed, adaptive
  threshold_value: 0.95
  
isolation_forest:
  contamination: 0.05
  n_estimators: 100
  max_samples: 256
  
ewma:
  alpha: 0.3
  window_size: 50
  
lstm:
  timesteps: 60
  encoding_dim: 32
  epochs: 50
  batch_size: 32

preprocessing:
  normalize: true
  remove_seasonality: false
  detrend: false
  smooth: false
  
monitoring:
  log_predictions: true
  track_performance: true
  detect_drift: true
  drift_threshold: 0.05
```

### Loading Configuration
```python
import yaml

class ConfigLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_detector(self):
        method = self.config['detection']['method']
        
        if method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            return IsolationForest(**self.config['isolation_forest'])
        elif method == 'zscore':
            return ZScoreDetector()
        elif method == 'ewma':
            return EWMADetector(**self.config['ewma'])
        
        return None
```

---

## Logging & Metrics

### Structured Logging
```python
import logging
import json

class AnomalyDetectionLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self, timestamp, value, is_anomaly, score, metadata=None):
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'value': float(value),
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(score),
            'metadata': metadata or {}
        }
        
        self.logger.info(json.dumps(log_entry))

# Usage
logger = AnomalyDetectionLogger('anomalies.log')
logger.log_prediction(
    timestamp=datetime.now(),
    value=sensor_reading,
    is_anomaly=True,
    score=0.95,
    metadata={'sensor_id': 'sensor_01', 'location': 'room1'}
)
```

### Metrics Collection
```python
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record(self, metric_name, value, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def get_summary(self, metric_name, window_size=100):
        values = [m['value'] for m in self.metrics[metric_name][-window_size:]]
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def export_metrics(self, output_file):
        with open(output_file, 'w') as f:
            for metric_name, values in self.metrics.items():
                f.write(f"\n{metric_name}:\n")
                for entry in values:
                    f.write(f"  {entry['timestamp']}: {entry['value']}\n")
```

---

## Advanced Techniques

### 1. Multi-Scale Analysis
```python
def multiscale_anomaly_detection(ts, scales=[1, 5, 20]):
    """Detect anomalies at multiple time scales"""
    anomaly_scores = np.zeros(len(ts))
    
    for scale in scales:
        # Aggregate to current scale
        aggregated = ts.reshape(-1, scale).mean(axis=1)
        
        # Detect anomalies
        z_scores = np.abs(stats.zscore(aggregated))
        
        # Upsample back to original scale
        upsampled = np.repeat(z_scores, scale)[:len(ts)]
        anomaly_scores += upsampled
    
    return anomaly_scores / len(scales)
```

### 2. Ensemble Voting
```python
class EnsembleVoting:
    def __init__(self, detectors):
        self.detectors = detectors
    
    def predict(self, X, min_votes=None):
        """Combine predictions from multiple detectors"""
        if min_votes is None:
            min_votes = len(self.detectors) // 2 + 1
        
        votes = np.zeros(len(X))
        
        for detector in self.detectors:
            predictions = detector.predict(X)
            votes += predictions.astype(int)
        
        return votes >= min_votes
    
    def get_confidence(self, X):
        """Get confidence scores from voting"""
        votes = np.zeros(len(X))
        
        for detector in self.detectors:
            predictions = detector.predict(X)
            votes += predictions.astype(int)
        
        confidence = votes / len(self.detectors)
        return confidence
```

### 3. Online Learning
```python
class OnlineAnomalyDetector:
    def __init__(self, initial_data):
        self.mean = np.mean(initial_data)
        self.std = np.std(initial_data)
        self.n_samples = len(initial_data)
    
    def update(self, x):
        """Update statistics with new point (Welford's algorithm)"""
        self.n_samples += 1
        
        delta = x - self.mean
        self.mean += delta / self.n_samples
        
        delta2 = x - self.mean
        M2 = (self.n_samples - 1) * self.std**2
        M2 += delta * delta2
        self.std = np.sqrt(M2 / (self.n_samples - 1))
    
    def predict(self, x, threshold=3):
        """Predict anomaly for new point"""
        z_score = abs((x - self.mean) / (self.std + 1e-10))
        return z_score > threshold
```

---

## Example: Complete Production System

```python
from datetime import datetime, timedelta
import sqlite3

class ProductionAnomalyDetectionSystem:
    def __init__(self, db_path='anomalies.db'):
        self.db_path = db_path
        self.detector = None
        self.config = None
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for logging"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS anomalies
                     (id INTEGER PRIMARY KEY,
                      timestamp TEXT,
                      value REAL,
                      score REAL,
                      is_anomaly BOOLEAN,
                      sensor_id TEXT)''')
        conn.commit()
        conn.close()
    
    def fit(self, historical_data):
        """Train detector on historical data"""
        from sklearn.ensemble import IsolationForest
        
        self.detector = IsolationForest(contamination=0.05)
        self.detector.fit(historical_data)
    
    def process_reading(self, timestamp, value, sensor_id):
        """Process single sensor reading"""
        
        # Predict
        is_anomaly = self.detector.predict([[value]])[0] == -1
        score = -self.detector.score_samples([[value]])[0]
        
        # Store in database
        self._store_reading(timestamp, value, score, is_anomaly, sensor_id)
        
        # Alert if anomaly
        if is_anomaly:
            self._trigger_alert(timestamp, value, sensor_id, score)
        
        return {
            'is_anomaly': is_anomaly,
            'score': score,
            'timestamp': timestamp
        }
    
    def _store_reading(self, timestamp, value, score, is_anomaly, sensor_id):
        """Store reading in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO anomalies 
                     (timestamp, value, score, is_anomaly, sensor_id)
                     VALUES (?, ?, ?, ?, ?)''',
                 (timestamp.isoformat(), value, score, is_anomaly, sensor_id))
        conn.commit()
        conn.close()
    
    def _trigger_alert(self, timestamp, value, sensor_id, score):
        """Send alert for detected anomaly"""
        alert_msg = f"Anomaly detected at {timestamp} for {sensor_id}: {value:.2f} (score: {score:.3f})"
        print(f"ALERT: {alert_msg}")
        # Could integrate with email, Slack, PagerDuty, etc.
    
    def get_anomaly_summary(self, hours=24):
        """Get summary of anomalies in last N hours"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        c.execute('''SELECT COUNT(*), AVG(score), MAX(score), sensor_id
                     FROM anomalies
                     WHERE timestamp > ? AND is_anomaly = 1
                     GROUP BY sensor_id''', (cutoff_time,))
        
        results = c.fetchall()
        conn.close()
        
        return [{
            'sensor_id': r[3],
            'count': r[0],
            'avg_score': r[1],
            'max_score': r[2]
        } for r in results]

# Usage
system = ProductionAnomalyDetectionSystem()
system.fit(historical_training_data)

# Process stream
for timestamp, value, sensor_id in sensor_stream:
    result = system.process_reading(timestamp, value, sensor_id)

# Get summary
summary = system.get_anomaly_summary(hours=24)
for item in summary:
    print(f"{item['sensor_id']}: {item['count']} anomalies")
```

---

**Last Updated:** April 2026  
**Status:** Production-Ready
