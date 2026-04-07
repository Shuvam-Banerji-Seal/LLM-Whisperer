"""Tools module README."""

# Tools Module

Command-line tools and utilities.

## Features

- **CLI Tools**: Command-line interfaces
- **Training Tool**: Training management
- **Evaluation Tool**: Model evaluation
- **Deployment Tool**: Deployment automation

## Quick Start

```bash
# Training
python -m tools.cli train --model gpt2 --data data.csv --epochs 3

# Evaluation
python -m tools.cli evaluate --model checkpoint.bin --benchmarks mmlu alpacaeval

# Deployment
python -m tools.cli deploy --model checkpoint.bin --version 1.0.0 --target huggingface
```

## Tools

### Training CLI
- Model selection
- Data specification
- Hyperparameter configuration
- Training management

### Evaluation CLI
- Benchmark selection
- Result reporting
- Comparison tools

### Deployment CLI
- Model packaging
- Version management
- Target selection
- Monitoring setup

## License

MIT
