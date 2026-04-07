"""
Data Pipeline README

Complete guide for data processing pipeline used in LLM-Whisperer.
"""

# Data Pipeline

The data pipeline module provides an end-to-end workflow for preparing datasets:

```
Raw Data → Ingestion → Cleaning → Preprocessing → Splitting → Validation → Ready for Training
```

## Quick Start

### 1. Basic Usage

```bash
python scripts/run_pipeline.py --config configs/default_config.yaml
```

### 2. With Custom Configuration

```bash
python scripts/run_pipeline.py --config configs/instruction_tuning_config.yaml
```

### 3. Using Different Data Source

```bash
# From CSV
python scripts/run_pipeline.py --config configs/default_config.yaml

# From HuggingFace
python -c "
from pipelines.data.src.ingestion import DataIngestion, IngestionConfig
config = IngestionConfig(
    source_type='huggingface',
    source_path='alpaca_data',
    max_samples=1000
)
ingestion = DataIngestion()
data = ingestion.load(config)
"
```

## Architecture

### Core Modules

#### 1. **Ingestion** (`src/ingestion.py`)

Load raw data from multiple sources:

- **CSV Files**: `CSVDataSource`
- **JSON Files**: `JSONDataSource`
- **JSONL Files**: `JSONLDataSource`
- **Parquet Files**: `ParquetDataSource`
- **HuggingFace Datasets**: `HuggingFaceDataSource`

```python
from pipelines.data.src.ingestion import DataIngestion, IngestionConfig

config = IngestionConfig(
    source_type="csv",
    source_path="data/raw/dataset.csv",
    max_samples=5000
)

ingestion = DataIngestion()
data = ingestion.load(config)
```

#### 2. **Cleaning** (`src/cleaning.py`)

Validate and clean text data:

- Remove duplicates
- Handle missing values
- Clean text (remove URLs, HTML, special chars)
- Filter by length
- Language detection

```python
from pipelines.data.src.cleaning import DataCleaning, CleaningConfig

config = CleaningConfig(
    remove_duplicates=True,
    remove_urls=True,
    remove_html=True,
    min_length=10,
    max_length=2000
)

cleaner = DataCleaning(config)
data = cleaner.clean(data, text_columns=["text"])
```

#### 3. **Preprocessing** (`src/preprocessing.py`)

Prepare data for model input:

- Tokenization using transformers
- Support for multiple formats:
  - **Standard**: Plain text
  - **Instruction Tuning**: Instruction + Response
  - **Conversation**: Multi-turn dialogue
- Configurable sequence length and padding

```python
from pipelines.data.src.preprocessing import DataPreprocessing, PreprocessingConfig

config = PreprocessingConfig(
    tokenizer_name="mistralai/Mistral-7B-v0.1",
    max_seq_length=2048,
    format_type="instruction_tuning"
)

preprocessor = DataPreprocessing(config)
data = preprocessor.preprocess(
    data,
    text_columns=["instruction", "output"],
    label_column="label"
)
```

#### 4. **Splitting** (`src/splitting.py`)

Create train/val/test splits:

- Standard train/val/test splitting
- Stratified splits (preserve class distribution)
- Cross-validation support
- Reproducible with seed control

```python
from pipelines.data.src.splitting import DataSplitting, SplittingConfig

config = SplittingConfig(
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    stratify_column="label",
    random_state=42
)

splitter = DataSplitting(config)
splits = splitter.split(data)

# Access splits
train = splits['train']
val = splits['val']
test = splits['test']

# Save splits to disk
splitter.save_splits("data/processed")
```

#### 5. **Validation** (`src/validation.py`)

Quality assurance and statistics:

- Check required columns
- Verify minimum samples
- Monitor duplicate rates
- Validate data types
- Compute dataset statistics

```python
from pipelines.data.src.validation import DataValidation, ValidationConfig

config = ValidationConfig(
    required_columns=["text", "label"],
    min_samples=100,
    max_duplicates_percentage=5.0,
    max_missing_percentage=5.0
)

validator = DataValidation(config)
is_valid = validator.validate(data)

# Get detailed report
report = validator.generate_report()
validator.save_report("validation_report.json")
```

## Configuration Options

### Ingestion Configuration

```yaml
ingestion:
  source_type: csv  # csv, json, jsonl, parquet, huggingface
  source_path: data/raw/dataset.csv
  max_samples: null  # Limit number of samples (null = all)
  sample_fraction: null  # Sample fraction (0.0-1.0)
  encoding: utf-8
  cache_dir: null  # For HuggingFace datasets
```

### Cleaning Configuration

```yaml
cleaning:
  remove_duplicates: true
  duplicate_subset: null  # Columns to check for duplicates
  handle_missing: drop  # drop or fill
  lowercase: false
  remove_urls: false
  remove_html: false
  remove_special_chars: false
  remove_extra_whitespace: true
  min_length: 10  # Minimum text length
  max_length: null  # Maximum text length
  detect_language: false
  target_languages: ["en"]  # For language filtering
```

### Preprocessing Configuration

```yaml
preprocessing:
  tokenizer_name: gpt2
  max_seq_length: 512
  truncation_strategy: longest_first
  padding_strategy: max_length
  format_type: standard  # standard, instruction_tuning, conversation
  instruction_template: "### Instruction:\n{}\n\n"
  response_template: "### Response:\n{}"
```

### Splitting Configuration

```yaml
splitting:
  train_size: 0.8
  val_size: 0.1
  test_size: 0.1
  random_state: 42
  stratify_column: null  # Column to stratify on
  shuffle: true
  cross_validation_folds: null  # For k-fold CV
```

### Validation Configuration

```yaml
validation:
  required_columns: []
  min_samples: 1
  max_duplicates_percentage: 10.0
  max_missing_percentage: 10.0
  check_data_types: true
  expected_dtypes: {}
```

## Examples

### Example 1: Basic CSV Processing

```python
import yaml
from pipelines.data.src.ingestion import DataIngestion, IngestionConfig
from pipelines.data.src.cleaning import DataCleaning, CleaningConfig

# Load config
with open("configs/default_config.yaml") as f:
    config = yaml.safe_load(f)

# Ingest data
ingestion = DataIngestion()
data = ingestion.load(IngestionConfig(**config['ingestion']))

# Clean data
cleaner = DataCleaning(CleaningConfig(**config['cleaning']))
data = cleaner.clean(data, text_columns=['text'])

print(f"Processed: {len(data)} samples")
```

### Example 2: Instruction Tuning Pipeline

```python
from pipelines.data.scripts.run_pipeline import run_pipeline, load_config

config = load_config("configs/instruction_tuning_config.yaml")
run_pipeline(config)
```

### Example 3: Custom Processing

```python
from pipelines.data.src.ingestion import DataIngestion, IngestionConfig
from pipelines.data.src.splitting import DataSplitting, SplittingConfig

# Load data from HuggingFace
config = IngestionConfig(
    source_type="huggingface",
    source_path="alpaca_data",
    max_samples=5000
)
ingestion = DataIngestion()
data = ingestion.load(config)

# Create stratified splits
splitting_config = SplittingConfig(
    train_size=0.8,
    val_size=0.2,
    test_size=0.0,
    stratify_column="label"
)
splitter = DataSplitting(splitting_config)
splits = splitter.split(data)

# Save
splitter.save_splits("output/alpaca")
```

## Pipeline Statistics

The pipeline tracks comprehensive statistics at each stage:

### Ingestion Statistics
- Number of rows loaded
- Number of columns
- Data types
- Memory usage

### Cleaning Statistics
- Initial/final row count
- Removed rows percentage
- Duplicate statistics
- Missing value statistics

### Splitting Statistics
- Split sizes for train/val/test
- Per-split statistics

### Validation Report
- Data shape and types
- Required columns check
- Duplicate detection
- Missing value analysis
- Quality metrics

## Best Practices

1. **Start with small data**: Test pipeline with `max_samples` first
2. **Validate early**: Add validation checks for your specific use case
3. **Monitor statistics**: Review cleaning and splitting statistics
4. **Use stratification**: When dealing with imbalanced datasets
5. **Cache preprocessed data**: Reuse tokenized outputs to save time
6. **Track configuration**: Save config files with results for reproducibility

## Common Issues

### Issue: High missing value percentage
**Solution**: Adjust `max_missing_percentage` in validation config or improve data source

### Issue: Data not being cleaned
**Solution**: Enable specific cleaning options in `CleaningConfig` (e.g., `remove_urls=True`)

### Issue: Text too short after cleaning
**Solution**: Reduce `min_length` in cleaning config

### Issue: Memory issues with large datasets
**Solution**: Use `max_samples` or `sample_fraction` in ingestion config

## Integration with Training

After processing with data pipeline, use splits for training:

```python
from pipelines.data.src.splitting import DataSplitting

splitter = DataSplitting(...)
splitter.load_splits("data/processed")

# In training code
train_data = splitter.splits['train']
val_data = splitter.splits['val']
```

## Contributing

When adding new features:
1. Add new `*Config` dataclass
2. Implement corresponding class
3. Add example in `configs/`
4. Update this README

## License

See LICENSE file in repository root.
