# Experiments

Experiment tracking, ablation studies, and comparative analysis framework.

## Overview

This module provides infrastructure for managing ML experiments:
- Experiment metadata and tracking
- Ablation study designs
- Comparative analysis and reporting
- Results aggregation and visualization
- Reproducibility support

## Structure

```
experiments/
├── README.md (this file)
├── tracking/          # Experiment metadata and schemas
├── ablations/         # Controlled experiment designs
└── reports/           # Result analysis and reports
```

## Directory Purposes

### `tracking/` - Experiment Management

Centralized experiment metadata:

```
tracking/
├── schema.py         # Experiment data schema
├── logger.py         # Experiment logging
├── storage.py        # Storage backends (JSON, DB)
└── experiments.json  # Experiment log
```

**Tracked Metadata**:
- Experiment name and description
- Model configuration
- Dataset configuration
- Training hyperparameters
- Hardware used
- Results (metrics)
- Timestamps
- Code version (git commit)

### `ablations/` - Ablation Studies

Systematic comparison of design choices:

```
ablations/
├── lora_rank_ablation.py     # Compare different LoRA ranks
├── learning_rate_ablation.py # Learning rate sweep
├── dataset_size_ablation.py  # Data size impact
└── architecture_ablation.py  # Architecture choices
```

**Common Ablations**:
- LoRA rank: [8, 16, 32, 64]
- Learning rate: [1e-4, 5e-4, 1e-3]
- Batch size: [8, 16, 32]
- Number of epochs: [1, 2, 3, 5]
- Quantization type: [none, int8, int4]

### `reports/` - Analysis and Reporting

Generate insights from experiments:

```
reports/
├── summary_report.py  # Overall experiment summary
├── comparison.py      # Side-by-side comparison
├── visualization.py   # Plots and charts
└── statistical.py     # Statistical significance
```

## Quick Start

### 1. Log Experiment

```python
from experiments.tracking import ExperimentLogger

logger = ExperimentLogger()
logger.start(
    name="mistral-alpaca-lora",
    model="mistralai/Mistral-7B-v0.1",
    dataset="alpaca",
    lora_rank=64,
    learning_rate=5e-4
)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    logger.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch
    })

logger.end(final_metrics={"alpaca_eval": 82.5})
```

### 2. Run Ablation Study

```python
from experiments.ablations import LoraRankAblation

ablation = LoraRankAblation(
    base_model="mistralai/Mistral-7B-v0.1",
    dataset="alpaca",
    ranks=[8, 16, 32, 64],
    num_trials=2
)

results = ablation.run()
ablation.print_summary()
```

### 3. Generate Report

```python
from experiments.reports import ExperimentReporter

reporter = ExperimentReporter()
reporter.load_experiments("experiments/experiments.json")
reporter.compare(
    exp1="mistral-alpaca-lora-rank64",
    exp2="mistral-alpaca-lora-rank32"
)
reporter.save_report("report.html")
```

## Experiment Schema

```python
# Experiment metadata
{
    "id": "mistral-alpaca-lora-001",
    "timestamp": "2024-04-07T10:30:00Z",
    "name": "Mistral LoRA on Alpaca",
    "description": "Fine-tune Mistral-7B on Alpaca dataset",
    
    # Configuration
    "model": {
        "name": "mistralai/Mistral-7B-v0.1",
        "size_billions": 7.24
    },
    "dataset": {
        "name": "alpaca",
        "split": {"train": 8000, "val": 1000}
    },
    "training": {
        "learning_rate": 5e-4,
        "batch_size": 16,
        "num_epochs": 3,
        "lora_rank": 64,
        "lora_alpha": 16
    },
    
    # Hardware
    "hardware": {
        "device": "gpu",
        "gpu_type": "A100",
        "num_gpus": 1,
        "memory_gb": 80
    },
    
    # Results
    "results": {
        "train_loss": 0.45,
        "val_loss": 0.52,
        "alpaca_eval": 82.5,
        "training_time_hours": 2.5
    },
    
    # Reproducibility
    "code": {
        "git_commit": "abc123def456",
        "git_branch": "main",
        "script": "pipelines/training/orchestrator.py"
    },
    
    "status": "completed"
}
```

## Ablation Study Examples

### Example 1: LoRA Rank Ablation

```python
from experiments.ablations import Ablation

class LoraRankAblation(Ablation):
    def __init__(self, ranks, *args, **kwargs):
        self.ranks = ranks
        super().__init__(*args, **kwargs)
    
    def run(self):
        results = {}
        for rank in self.ranks:
            exp_name = f"lora-rank-{rank}"
            
            # Train with this rank
            model = train_model(
                lora_rank=rank,
                **self.base_config
            )
            
            # Evaluate
            metrics = evaluate(model)
            results[exp_name] = metrics
        
        return results
```

### Example 2: Learning Rate Sweep

```python
from experiments.ablations import Ablation
import numpy as np

class LearningRateAblation(Ablation):
    def __init__(self, learning_rates, *args, **kwargs):
        self.learning_rates = learning_rates
        super().__init__(*args, **kwargs)
    
    def run(self):
        results = {}
        for lr in self.learning_rates:
            exp_name = f"lr-{lr}"
            
            model = train_model(
                learning_rate=lr,
                **self.base_config
            )
            metrics = evaluate(model)
            results[exp_name] = metrics
        
        return results
```

## Visualization

### Metric Comparison

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# LoRA rank comparison
ranks = [8, 16, 32, 64]
scores = [78.2, 80.1, 81.5, 82.5]
axes[0].plot(ranks, scores, 'o-')
axes[0].set_xlabel('LoRA Rank')
axes[0].set_ylabel('AlpacaEval Score')
axes[0].set_title('Effect of LoRA Rank')

# Learning rate comparison
lrs = [1e-4, 5e-4, 1e-3, 2e-3]
scores = [75.3, 82.5, 81.2, 78.1]
axes[1].plot(lrs, scores, 'o-')
axes[1].set_xscale('log')
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('AlpacaEval Score')
axes[1].set_title('Effect of Learning Rate')

plt.tight_layout()
plt.savefig('experiments/reports/ablation_results.png')
```

## Statistical Analysis

### Significance Testing

```python
from scipy import stats

# Compare two experiments
exp1_scores = [82.1, 82.3, 82.5]
exp2_scores = [80.1, 80.3, 80.5]

# T-test
t_stat, p_value = stats.ttest_ind(exp1_scores, exp2_scores)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant: {p_value < 0.05}")
```

## Common Ablations

| Ablation | Variable | Values | Expected Finding |
|----------|----------|--------|------------------|
| LoRA Rank | rank | [8, 16, 32, 64] | Higher rank → better (diminishing returns) |
| Learning Rate | lr | [1e-4, 5e-4, 1e-3] | Sweet spot around 5e-4 |
| Epochs | epochs | [1, 2, 3, 5] | 2-3 epochs optimal, diminishing after |
| Batch Size | batch_size | [8, 16, 32] | 16 usually best balance |
| Quantization | method | [none, int8, int4] | int8 ≈ full quality, int4 ~1-2% loss |

## Tools Integration

### Weights & Biases

```python
import wandb

wandb.init(project="llm-experiments")
wandb.log({
    "train_loss": loss,
    "eval_score": score
})
```

### MLflow

```python
import mlflow

mlflow.start_run()
mlflow.log_param("lora_rank", 64)
mlflow.log_metric("alpaca_eval", 82.5)
mlflow.end_run()
```

### Aim

```python
from aim import Run

run = Run()
run["config"] = {"lora_rank": 64}
run.track(82.5, name="alpaca_eval")
```

## Reproducibility

### Save Experiment Config

```yaml
# experiments/ablations/lora_rank_ablation/config.yaml
base_model: mistralai/Mistral-7B-v0.1
dataset: alpaca
epochs: 3
batch_size: 16
learning_rate: 5e-4

variable_param:
  name: lora_rank
  values: [8, 16, 32, 64]
```

### Reproduce Experiment

```bash
python experiments/ablations/lora_rank_ablation.py \
  --config experiments/ablations/lora_rank_ablation/config.yaml
```

## References

- See `../pipelines/` for pipeline execution
- See `../evaluation/` for evaluation frameworks
- See `../fine_tuning/` for training configurations

## Contributing

When running experiments:
1. Log all hyperparameters
2. Record hardware specifications
3. Save model checkpoints
4. Include error logs
5. Document results
6. Make reproducible

## License

See LICENSE file in repository root.
