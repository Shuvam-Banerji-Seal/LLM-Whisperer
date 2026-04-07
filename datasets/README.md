# Datasets

Dataset management, versioning, and organization for training, evaluation, and benchmarking.

## Overview

This module provides structure for managing datasets across the machine learning pipeline:
- Raw data from external sources
- Processed datasets ready for training
- Synthetic and augmented data
- Evaluation and benchmark sets
- Prompt collections

## Structure

```
datasets/
├── README.md (this file)
├── raw/              # Raw unprocessed data
├── interim/          # Processed but not final datasets
├── processed/        # Training-ready datasets
├── synthetic/        # Generated or augmented data
├── prompt_sets/      # Prompt corpora and templates
└── eval_sets/        # Evaluation and benchmark datasets
```

## Quick Start

### 1. Loading a Dataset
```python
from datasets import load_dataset

dataset = load_dataset("path/to/datasets/processed/alpaca")
```

### 2. Creating a Dataset
```python
import json
from pathlib import Path

output_dir = Path("datasets/processed/my_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "data.jsonl", "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")
```

### 3. Dataset Versioning
```yaml
# datasets/processed/my_dataset/METADATA.yaml
name: my_dataset
version: 1.0.0
format: jsonl
count: 10000
split:
  train: 0.8
  val: 0.1
  test: 0.1
description: |
  My custom instruction-following dataset
```

## Directory Purposes

### `raw/` - Raw Source Data
- **Purpose**: Unprocessed data directly from sources
- **Naming**: Use source name or date-based naming
- **Size**: Can be large, consider external storage
- **Retention**: Keep for reproducibility
- **Examples**:
  - `raw/wikipedia_dump_2026_01.tar.gz`
  - `raw/github_code_2026_q1.parquet`
  - `raw/arxiv_papers_2024_2025.jsonl`

### `interim/` - Intermediate Processing
- **Purpose**: Partially processed, not yet final
- **Use**: Checkpoint during multi-stage processing
- **Naming**: Include processing stage
- **Retention**: Can be deleted after final processing
- **Examples**:
  - `interim/alpaca_cleaned.jsonl` (removed duplicates)
  - `interim/code_deduplicated.parquet` (LSH deduplication)
  - `interim/wikipedia_chunked.jsonl` (segmented)

### `processed/` - Training-Ready Data
- **Purpose**: Final, production-ready datasets
- **Format**: JSONL, Parquet, or HuggingFace format
- **Quality**: Validated and tested
- **Naming**: Clear, self-documenting names
- **Examples**:
  - `processed/alpaca_7k/` (clean, deduplicated)
  - `processed/code_python_10k/` (code-only subset)
  - `processed/math_problems_50k/` (math domain)

### `synthetic/` - Generated Data
- **Purpose**: Augmented or synthetically generated examples
- **Types**:
  - Instruction generation (GPT-3 style prompts)
  - Data augmentation (paraphrasing, back-translation)
  - Few-shot examples
  - Adversarial examples
- **Naming**: Include generation method
- **Examples**:
  - `synthetic/alpaca_gpt4_generated/` (synthetic instruction-following)
  - `synthetic/code_perturbed/` (code with variable renaming)
  - `synthetic/qa_augmented/` (back-translated QA pairs)

### `prompt_sets/` - Prompt Collections
- **Purpose**: Curated prompts and templates
- **Types**:
  - System prompts
  - Few-shot examples
  - Evaluation prompts
  - Adversarial prompts
- **Organization**: By task or domain
- **Examples**:
  - `prompt_sets/classification/` (classification tasks)
  - `prompt_sets/generation/` (generation tasks)
  - `prompt_sets/qa/` (question answering)

### `eval_sets/` - Evaluation Datasets
- **Purpose**: Golden datasets for testing and benchmarking
- **Quality**: High-quality, manually validated
- **Size**: Usually small (100-1000 examples)
- **Naming**: Include benchmark name or task
- **Examples**:
  - `eval_sets/mmlu/` (MMLU benchmark subset)
  - `eval_sets/arc_challenge/` (ARC Challenge)
  - `eval_sets/human_eval/` (HumanEval for code)

## Dataset Formats

### JSONL (Recommended for LLM)
```jsonl
{"instruction": "...", "input": "...", "output": "..."}
{"instruction": "...", "input": "...", "output": "..."}
```

### Parquet (For large datasets)
```python
import pandas as pd
df = pd.DataFrame(data)
df.to_parquet("dataset.parquet")
```

### HuggingFace Datasets
```python
from datasets import load_dataset
dataset = load_dataset("path/to/dataset")
```

## Metadata Convention

Every dataset directory should include `METADATA.yaml`:

```yaml
# datasets/processed/my_dataset/METADATA.yaml

name: my_dataset
version: 1.0.0

description: |
  Short description of dataset.
  Can span multiple lines.

source:
  url: https://example.com/dataset
  license: CC-BY-4.0
  paper: https://arxiv.org/abs/2024.xxxxx

format: jsonl
total_examples: 10000

fields:
  - name: instruction
    type: string
    description: Task instruction
  - name: input
    type: string
    description: Input example (optional)
  - name: output
    type: string
    description: Expected output

split:
  train:
    count: 8000
    file: train.jsonl
  val:
    count: 1000
    file: val.jsonl
  test:
    count: 1000
    file: test.jsonl

processing_steps:
  - step: 1
    name: Deduplication
    method: Exact string matching
    removed_count: 500
  - step: 2
    name: Language filtering
    method: langdetect
    kept_languages: [en]

citations:
  - "Author et al. (2024)"

keywords:
  - instruction-following
  - generalist
  - 7k-examples
```

## Best Practices

### Data Quality
1. **Deduplication**: Remove exact and near-duplicates
2. **Filtering**: Remove low-quality examples
3. **Validation**: Ensure format consistency
4. **Documentation**: Include metadata for reproducibility

### Organization
1. **Clear Naming**: Use descriptive names without special characters
2. **Versioning**: Include version in directory name if multiple versions
3. **Splitting**: Separate train/val/test clearly
4. **Size Awareness**: Keep processed datasets under 1GB (use external storage for larger)

### Processing Pipeline
1. **Raw** → **Interim** → **Processed**
2. Document each transformation
3. Keep checkpoints for reproducibility
4. Validate at each stage

### Storage
```
< 500 MB  → Keep in git (if added to LFS)
500MB-5GB → Store in external storage, reference with manifest
> 5GB     → External storage only, include download script
```

## Creating a New Dataset

### Step 1: Create Structure
```bash
mkdir -p datasets/processed/my_dataset/{train,val,test}
cd datasets/processed/my_dataset
touch METADATA.yaml README.md
```

### Step 2: Add Data
```bash
# Add your data files
cp /path/to/train.jsonl train/
cp /path/to/val.jsonl val/
cp /path/to/test.jsonl test/
```

### Step 3: Document
```yaml
# METADATA.yaml
name: my_dataset
version: 1.0.0
description: My custom dataset
format: jsonl
total_examples: 10000
```

### Step 4: Validate
```bash
python validate_dataset.py datasets/processed/my_dataset
```

## Loading in Training

```python
from datasets import load_dataset

# From local directory
dataset = load_dataset(
    "json",
    data_files="datasets/processed/my_dataset/train/*.jsonl"
)

# With train/val/test split
dataset = load_dataset(
    "json",
    data_files={
        "train": "datasets/processed/my_dataset/train.jsonl",
        "validation": "datasets/processed/my_dataset/val.jsonl",
        "test": "datasets/processed/my_dataset/test.jsonl"
    }
)
```

## Common Datasets

### Instruction-Following
- `alpaca` - 52K examples
- `alpaca_eval` - 805 test examples
- `vicuna` - 73K examples
- `open_platypus` - 24K examples

### Code
- `codeparrot` - 12M code snippets
- `the_stack` - 3.1M code files
- `humaneval` - 164 programming tasks

### Chat/Conversation
- `ultrachat` - 1.4M conversations
- `sharegpt` - 90K conversations
- `oasst1` - 66K conversations

### Knowledge
- `wikipedia` - 6.5M articles
- `arxiv` - 2.2M papers
- `bookcorpus` - 74M sentences

## Tools for Dataset Operations

```bash
# Install dataset tools
pip install datasets huggingface-hub

# Validate JSONL
python -m json.tool datasets/processed/my_dataset/train.jsonl > /dev/null

# Count examples
wc -l datasets/processed/my_dataset/train.jsonl

# Sample examples
head -5 datasets/processed/my_dataset/train.jsonl | python -m json.tool
```

## References

- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Data Processing Best Practices](../docs/guides/)
- See `../fine_tuning/` for dataset usage in training
- See `../evaluation/` for evaluation dataset organization

## Contributing

When adding a new dataset:
1. Create appropriate directory structure
2. Include METADATA.yaml with complete information
3. Add README.md with usage examples
4. Validate format and content
5. Include license and attribution
6. Document any preprocessing steps

## License

Each dataset should include its own LICENSE file with proper attribution.
