# Evaluation Pipeline README

Complete guide for evaluation pipeline used in LLM-Whisperer.

## Quick Start

### 1. Run All Benchmarks

```bash
python scripts/evaluate.py --model ./checkpoints/lora
```

### 2. Run Specific Benchmarks

```bash
python scripts/evaluate.py --model ./checkpoints/lora --benchmarks mmlu alpacaeval gsm8k
```

### 3. Generate Report

```bash
python scripts/evaluate.py --model ./checkpoints/lora --save-html
```

## Architecture

### Core Modules

#### 1. **Benchmark** (`src/benchmark.py`)

Run comprehensive benchmarks:
- **MMLU**: Multi-task language understanding (57-subject knowledge)
- **AlpacaEval**: Instruction-following quality
- **GSM8K**: Math reasoning
- **HellaSwag**: Common sense reasoning

```python
from pipelines.evaluation.src.benchmark import BenchmarkOrchestrator, BenchmarkConfig

config = BenchmarkConfig(
    model_path="./checkpoints/lora",
    benchmarks=["mmlu", "alpacaeval", "gsm8k", "hellaswag"],
    batch_size=32
)

orchestrator = BenchmarkOrchestrator(config)
results = orchestrator.run_all()
```

#### 2. **Metrics** (`src/metrics.py`)

Compute and track metrics:
- Task benchmark scores
- Latency and throughput
- Safety metrics
- Regression detection

```python
from pipelines.evaluation.src.metrics import MetricsComputer, MetricsConfig

config = MetricsConfig(task_benchmarks=True, latency_analysis=True)
computer = MetricsComputer(config)
metrics = computer.compute_metrics(benchmark_results)
```

#### 3. **Reporting** (`src/reporting.py`)

Generate evaluation reports:
- JSON reports with detailed metrics
- HTML reports for visualization
- Comparison reports across runs

```python
from pipelines.evaluation.src.reporting import ReportGenerator

generator = ReportGenerator(output_dir="./eval_results")
report = generator.generate_report(benchmarks, metrics, "model-name")
generator.save_report(report)
generator.generate_html_report(report)
```

#### 4. **Regression** (integrated in metrics.py)

Detect performance regressions:
- Compare current metrics to baseline
- Alert on threshold violations
- Track improvements

```python
from pipelines.evaluation.src.metrics import RegressionDetector

detector = RegressionDetector(baseline_metrics)
regression_report = detector.detect_regressions(current_metrics, threshold=0.05)
```

## Benchmark Details

### MMLU (Multi-task Language Understanding)
- **Coverage**: 57 domains from science to law
- **Samples**: 14,042 questions
- **Format**: 4-choice multiple choice
- **Score**: 0-100%

### AlpacaEval (Instruction Following)
- **Coverage**: Instruction-following quality
- **Samples**: 805 instructions
- **Format**: Model output vs baseline comparison
- **Score**: Win rate %

### GSM8K (Math Reasoning)
- **Coverage**: Grade school math word problems
- **Samples**: 1,319 problems
- **Format**: Free-form generation
- **Score**: Exact match %

### HellaSwag (Common Sense)
- **Coverage**: Video understanding and common sense
- **Samples**: 10,042 questions
- **Format**: 4-choice multiple choice
- **Score**: 0-100%

## Configuration

Example evaluation configuration:

```yaml
benchmark:
  model_path: ./checkpoints/lora
  benchmarks:
    - mmlu
    - alpacaeval
    - gsm8k
    - hellaswag
  batch_size: 32
  num_shots: 0
  max_samples: null

metrics:
  task_benchmarks: true
  llm_as_judge: false
  safety_checks: false
  latency_analysis: true
  regression_tests: true

regression:
  baseline_path: ./baselines/gpt2.json
  threshold: 0.05
  alert_on_regression: true
```

## Output Formats

### JSON Report
```json
{
  "metadata": {
    "model": "mistral-7b-lora",
    "timestamp": "2024-01-15T10:30:00"
  },
  "benchmarks": {
    "mmlu": {"accuracy": 0.45, "score": 45.0},
    "alpacaeval": {"win_rate": 0.52, "score": 52.0}
  },
  "metrics": {
    "average_score": 48.5,
    "latency": {"p50": 150, "p95": 250}
  }
}
```

### HTML Report
Visual report with charts and tables showing:
- Benchmark scores
- Metric comparisons
- Performance over time
- Regression alerts

## Best Practices

1. **Baseline First**: Establish baseline metrics before optimization
2. **Regular Evaluation**: Evaluate after each training run
3. **Monitor Regressions**: Track metrics over time
4. **Full Benchmark Suite**: Run all benchmarks for comprehensive evaluation
5. **Save Reports**: Archive reports for comparison

## Integration with Training

Evaluate after training:

```bash
# Train model
python pipelines/training/scripts/train.py --config training_config.yaml

# Evaluate model
python pipelines/evaluation/scripts/evaluate.py --model ./training_outputs/lora

# Check for regressions
python pipelines/evaluation/scripts/compare_runs.py --baseline baselines/baseline.json
```

## Common Issues

### Issue: Evaluation too slow
**Solution**:
- Reduce `max_samples` for quick evaluation
- Use GPU for inference
- Reduce `batch_size` if memory issues

### Issue: Benchmark data not found
**Solution**:
- Install datasets: `pip install datasets`
- Check internet connection for downloads
- Use `cache_dir` to specify download location

### Issue: Regression detection too sensitive
**Solution**:
- Increase `threshold` value
- Compare against more recent baseline
- Filter out noisy metrics

## License

See LICENSE file in repository root.
