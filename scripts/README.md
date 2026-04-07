# Scripts

Utility scripts for setup, automation, maintenance, and release management.

## Overview

This directory contains useful scripts for common tasks:
- Environment setup and configuration
- Repository management
- Testing and validation
- Deployment automation
- Data processing
- Model management

## Script Categories

### Setup & Configuration

**Purpose**: Initialize and configure the development environment

```bash
# Bootstrap new development environment
bash scripts/bootstrap.sh

# Install dependencies
pip install -r scripts/requirements.txt

# Configure pre-commit hooks
bash scripts/setup_precommit.sh
```

### Testing & Validation

**Purpose**: Run tests and validate code quality

```bash
# Run all tests
bash scripts/run_tests.sh

# Run specific test suite
bash scripts/run_tests.sh --suite unit

# Code quality checks
bash scripts/lint.sh

# Format code
bash scripts/format.sh
```

### Data Processing

**Purpose**: Process and prepare datasets

```bash
# Download datasets
python scripts/download_datasets.py --dataset alpaca

# Process raw data
python scripts/process_data.py --input raw/ --output processed/

# Validate dataset
python scripts/validate_dataset.py datasets/processed/my_dataset/
```

### Model Management

**Purpose**: Manage model downloads, conversions, and exports

```bash
# Download model
python scripts/download_model.py --model mistralai/Mistral-7B-v0.1

# Convert to quantized format
python scripts/quantize_model.py --model model.safetensors --output-dir ./quantized/

# Export to ONNX
python scripts/export_onnx.py --model model --output model.onnx
```

### Training & Fine-tuning

**Purpose**: Launch training jobs

```bash
# Train LoRA adapter
bash scripts/train_lora.sh --model mistral --dataset alpaca

# Train with distributed settings
bash scripts/train_distributed.sh --num-gpus 4 --config training.yaml

# Monitor training
bash scripts/monitor_training.sh --job-id training_001
```

### Evaluation & Benchmarking

**Purpose**: Run evaluation and benchmarking

```bash
# Run evaluation suite
bash scripts/evaluate.sh --model ./checkpoints/lora --dataset eval_set

# Benchmark inference speed
python scripts/benchmark_inference.py --model model --batch-size 32

# Compare models
python scripts/compare_models.py --model1 model1 --model2 model2
```

### Deployment

**Purpose**: Deploy models and services

```bash
# Build Docker image
bash scripts/build_docker.sh --tag llm-inference:latest

# Deploy to Kubernetes
bash scripts/deploy_kubernetes.sh --image llm-inference:latest --namespace prod

# Health check
bash scripts/health_check.sh --url http://localhost:8000
```

### Maintenance

**Purpose**: Repository and artifact maintenance

```bash
# Clean up artifacts
bash scripts/cleanup.sh --remove-checkpoints --remove-cache

# Backup important files
bash scripts/backup.sh --output backups/

# Database maintenance
python scripts/maintenance.py --task cleanup-old-experiments
```

## Common Scripts

### Setup Development Environment

```bash
#!/bin/bash
# scripts/bootstrap.sh

set -e
echo "Bootstrapping LLM-Whisperer development environment..."

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Create directories
mkdir -p models/{base,adapters,merged,exported}
mkdir -p datasets/{raw,interim,processed}
mkdir -p checkpoints logs artifacts

echo "✓ Development environment ready!"
echo "To activate: source .venv/bin/activate"
```

### Run Tests

```bash
#!/bin/bash
# scripts/run_tests.sh

SUITE=${1:-all}

if [ "$SUITE" = "all" ] || [ "$SUITE" = "unit" ]; then
    echo "Running unit tests..."
    pytest tests/unit/ -v
fi

if [ "$SUITE" = "all" ] || [ "$SUITE" = "integration" ]; then
    echo "Running integration tests..."
    pytest tests/integration/ -v
fi

if [ "$SUITE" = "all" ] || [ "$SUITE" = "e2e" ]; then
    echo "Running e2e tests..."
    pytest tests/e2e/ -v
fi
```

### Train LoRA Adapter

```bash
#!/bin/bash
# scripts/train_lora.sh

MODEL=${1:-mistral}
DATASET=${2:-alpaca}

cd fine_tuning/lora

python train.py \
  --model_name "mistralai/Mistral-7B-v0.1" \
  --dataset $DATASET \
  --output_dir ./checkpoints/$MODEL-$DATASET-lora \
  --lora_rank 64 \
  --learning_rate 5e-4 \
  --num_epochs 3 \
  --batch_size 16

echo "✓ Training complete: ./checkpoints/$MODEL-$DATASET-lora"
```

### Evaluate Model

```bash
#!/bin/bash
# scripts/evaluate.sh

MODEL=${1:-./checkpoints/lora}
BASE_MODEL=${2:-mistralai/Mistral-7B-v0.1}

cd pipelines/evaluation

python benchmark.py \
  --model $MODEL \
  --base_model $BASE_MODEL \
  --output evaluation_results/ \
  --tasks alpaca_eval mmlu hellaswag

echo "✓ Evaluation complete: evaluation_results/"
```

### Deploy to Kubernetes

```bash
#!/bin/bash
# scripts/deploy_kubernetes.sh

IMAGE=${1:-llm-inference:latest}
NAMESPACE=${2:-default}

echo "Deploying $IMAGE to namespace $NAMESPACE..."

kubectl apply -f infra/kubernetes/base/ --namespace=$NAMESPACE
kubectl set image deployment/llm-inference \
  llm-inference=$IMAGE \
  --namespace=$NAMESPACE

echo "Waiting for deployment..."
kubectl rollout status deployment/llm-inference --namespace=$NAMESPACE

echo "✓ Deployment complete!"
```

## Script Best Practices

### Error Handling

```bash
#!/bin/bash
set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Or with explicit error handling
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi
```

### Logging

```bash
# Log important messages
echo "[INFO] Starting training..."
echo "[WARN] This might take a while"
echo "[ERROR] Something went wrong" >&2

# Log to file
echo "[$(date)] Starting" >> logs/script.log
```

### Colors

```bash
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}✓ Success${NC}"
echo -e "${RED}✗ Failed${NC}"
echo -e "${YELLOW}! Warning${NC}"
```

## Directory Structure

```
scripts/
├── README.md (this file)
├── bootstrap.sh           # Initialize environment
├── setup_precommit.sh     # Configure git hooks
├── run_tests.sh           # Run test suite
├── lint.sh                # Code quality checks
├── format.sh              # Code formatting
├── train_lora.sh          # Train LoRA
├── evaluate.sh            # Run evaluation
├── benchmark_inference.py # Inference benchmarks
├── download_model.py      # Download models
├── quantize_model.py      # Quantize models
├── export_onnx.py         # Export to ONNX
├── deploy_kubernetes.sh   # Deploy to K8s
├── build_docker.sh        # Build Docker image
├── health_check.sh        # Health check
├── cleanup.sh             # Clean artifacts
├── backup.sh              # Backup files
└── maintenance.py         # Maintenance tasks
```

## Running Scripts

### From Repository Root

```bash
# Make executable
chmod +x scripts/*.sh

# Run script
bash scripts/bootstrap.sh
# or
./scripts/bootstrap.sh
```

### From Script Directory

```bash
cd scripts
bash bootstrap.sh
```

### With Arguments

```bash
bash scripts/train_lora.sh mistral alpaca
```

## Creating New Scripts

### Template for Shell Scripts

```bash
#!/bin/bash

# Script: name
# Purpose: What this does
# Usage: bash script.sh [options]

set -e
set -u

echo "[INFO] Starting..."

# Main logic here

echo "[INFO] Complete!"
```

### Template for Python Scripts

```python
#!/usr/bin/env python
"""Script purpose and usage."""

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    
    logger.info(f"Processing {args.input}")
    # Main logic here
    logger.info("Complete!")

if __name__ == "__main__":
    main()
```

## Contributing

When adding new scripts:
1. Add script to appropriate category
2. Update README with description
3. Include usage examples
4. Add error handling
5. Include logging
6. Test thoroughly
7. Make scripts reusable

## References

- See `../README.md` for overall structure
- See `../docs/guides/` for detailed guides
- See `../pipelines/` for pipeline execution

## License

See LICENSE file in repository root.
