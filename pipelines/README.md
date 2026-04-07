# Pipelines

Automated workflows for data processing, training, evaluation, and deployment.

## Overview

This module provides end-to-end pipeline implementations:
- Data preparation and validation
- Training orchestration
- Model evaluation and benchmarking
- Continuous deployment
- Pipeline monitoring and reporting

## Structure

```
pipelines/
├── README.md (this file)
├── data/             # Data processing and validation pipelines
├── training/         # Model training orchestration
├── evaluation/       # Evaluation and benchmarking
└── deployment/       # CI/CD and deployment automation
```

## Directory Purposes

### `data/` - Data Processing

Pipelines for data ingestion, cleaning, and preparation:

```
data/
├── ingestion.py      # Load raw data from sources
├── cleaning.py       # Data validation and cleaning
├── preprocessing.py  # Format conversion and normalization
├── splitting.py      # Train/val/test splitting
└── validation.py     # Quality checks
```

**Pipeline Flow**:
```
Raw Data → Ingestion → Cleaning → Preprocessing → Splitting → Validation → Ready for Training
```

### `training/` - Training Orchestration

Manage training workflows:

```
training/
├── full_finetune.yaml     # Full fine-tuning config
├── lora_finetune.yaml     # LoRA training config
├── dpo_training.yaml      # DPO training config
├── orchestrator.py        # Pipeline orchestration
├── callbacks.py           # Training callbacks
└── checkpointing.py       # Checkpoint management
```

**Features**:
- Multi-experiment tracking
- Hyperparameter sweeps
- Distributed training
- Early stopping
- Checkpoint recovery

### `evaluation/` - Evaluation Pipeline

Automated benchmarking and quality checks:

```
evaluation/
├── benchmark.py      # Run evaluation benchmarks
├── metrics.py        # Compute evaluation metrics
├── reporting.py      # Generate evaluation reports
├── regression.py     # Catch quality regressions
└── dashboards.py     # Visualization
```

**Metrics Computed**:
- Instruction-following quality (AlpacaEval)
- Knowledge (MMLU)
- Common sense (HellaSwag)
- Math (GSM8K)
- Latency and throughput

### `deployment/` - Deployment Pipeline

Automated model deployment and versioning:

```
deployment/
├── packaging.py      # Package model for deployment
├── versioning.py     # Model versioning
├── publishing.py     # Publish to model hub
├── rollback.py       # Rollback capabilities
└── monitoring.py     # Post-deployment monitoring
```

## Quick Start

### 1. Run Data Pipeline

```bash
cd pipelines/data
python -m pipelines.data.ingestion --source s3://bucket/raw
python -m pipelines.data.cleaning --input raw_data/
python -m pipelines.data.preprocessing --input cleaned/
python -m pipelines.data.splitting --input processed/
```

### 2. Run Training Pipeline

```bash
cd pipelines/training
python orchestrator.py \
  --config lora_finetune.yaml \
  --dataset data/processed/alpaca \
  --output_dir ./checkpoints
```

### 3. Run Evaluation Pipeline

```bash
cd pipelines/evaluation
python benchmark.py \
  --model ./checkpoints/lora \
  --base_model mistralai/Mistral-7B-v0.1 \
  --output evaluation_results/
```

### 4. Deploy Model

```bash
cd pipelines/deployment
python publishing.py \
  --model ./checkpoints/lora-merged \
  --version 1.0.0 \
  --push_to huggingface
```

## Pipeline Configuration

Example configuration files:

```yaml
# pipelines/training/lora_finetune.yaml
name: "Mistral-7B LoRA on Alpaca"
description: |
  Fine-tune Mistral-7B on Alpaca dataset using LoRA
  
model:
  name: "mistralai/Mistral-7B-v0.1"
  quantization: false
  lora_rank: 64
  lora_alpha: 16
  lora_dropout: 0.05
  
dataset:
  name: "alpaca"
  path: "data/processed/alpaca"
  train_size: 0.8
  val_size: 0.1
  test_size: 0.1
  
training:
  learning_rate: 5e-4
  batch_size: 16
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01
  max_grad_norm: 1.0
  
optimization:
  mixed_precision: true
  gradient_checkpointing: true
  distributed: false
  
evaluation:
  eval_steps: 500
  eval_strategy: "steps"
  save_strategy: "steps"
  save_steps: 500
  
monitoring:
  wandb_project: "llm-finetuning"
  log_level: "info"
```

## Pipeline Execution

### Local Execution

```bash
# Run single pipeline
python pipelines/training/orchestrator.py --config training.yaml

# With monitoring
python pipelines/training/orchestrator.py \
  --config training.yaml \
  --monitor wandb
```

### Distributed Execution

```bash
# Using Ray
ray start --head
python pipelines/training/orchestrator.py \
  --config training.yaml \
  --backend ray
```

### Kubernetes Execution

```bash
# Using Kubeflow
kubeflow pipeline submit \
  --experiment training \
  pipelines/training/kubeflow_pipeline.yaml
```

## Monitoring and Reporting

### Real-time Monitoring

- **Weights & Biases**: Training curves, system metrics
- **TensorBoard**: Loss, accuracy, learning rate
- **Prometheus**: Custom metrics and performance

### Post-Pipeline Reporting

```bash
# Generate evaluation report
python pipelines/evaluation/reporting.py \
  --results evaluation_results/ \
  --output report.html

# Compare with baseline
python pipelines/evaluation/regression.py \
  --current evaluation_results/ \
  --baseline baseline_results/
```

## Error Handling and Recovery

### Checkpointing

```python
# Automatic checkpoint saving
if step % 500 == 0:
    model.save_checkpoint(f"./checkpoints/step_{step}")
    
# Resume from checkpoint
if resume_from:
    model = load_from_checkpoint(resume_from)
```

### Retry Logic

```python
# Automatic retries
@retry(max_attempts=3, backoff=2)
def run_pipeline(config):
    # Pipeline logic
    pass
```

### Validation Gates

```python
# Quality gates before deployment
if eval_metrics['alpaca_eval'] < baseline - 0.05:
    raise QualityRegressionError(
        f"Quality dropped below threshold: {eval_metrics['alpaca_eval']}"
    )
```

## Performance Metrics

### Training
- Training loss
- Validation loss
- Learning rate
- Throughput (tokens/sec)
- GPU utilization

### Evaluation
- MMLU accuracy
- AlpacaEval win rate
- Inference latency
- BLEU/ROUGE scores
- Human evaluation scores

### Deployment
- Model size
- Inference latency (p50, p99)
- Throughput (requests/sec)
- Memory footprint
- Cost per inference

## Optimization Tips

### Data Pipeline
- Use parallel data loading
- Cache preprocessed data
- Stream large datasets
- Use data augmentation

### Training Pipeline
- Enable mixed precision training
- Use gradient checkpointing
- Increase batch size with gradient accumulation
- Use flash attention for speedup

### Evaluation Pipeline
- Parallelize evaluation across GPU
- Cache model outputs
- Sample for quick evaluation
- Use batched inference

## Integration with Tools

### Experiment Tracking
- **Weights & Biases**: `wandb.init()`
- **MLflow**: `mlflow.start_run()`
- **Aim**: `Aim()`

### Workflow Orchestration
- **Apache Airflow**: DAG-based scheduling
- **Kubeflow**: K8s-native ML pipelines
- **Ray**: Distributed computing
- **Prefect**: Data flow pipelines

### Model Registry
- **HuggingFace Hub**: `push_to_hub()`
- **MLflow Registry**: Model versioning
- **DVC**: Data version control

## References

- See `../fine_tuning/` for training recipes
- See `../evaluation/` for evaluation frameworks
- See `../datasets/` for data organization
- See `../infra/` for deployment infrastructure

## Contributing

When adding pipelines:
1. Create clear stage separation
2. Add comprehensive logging
3. Include error handling
4. Document configuration options
5. Provide example usage
6. Include monitoring integration

## License

See LICENSE file in repository root.
