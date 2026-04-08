# RAG Evaluation Module

Comprehensive evaluation metrics and tools for RAG systems.

## Overview

Evaluate RAG systems across three dimensions:
- **Retrieval Quality**: NDCG, Recall, Precision, MRR
- **Generation Quality**: ROUGE, BLEU, BERTScore
- **Faithfulness**: Hallucination detection, grounding scores

## Key Classes

### RAGEvaluator
Main evaluator supporting multiple evaluation modes.

### MetricCalculator
Computes standard IR and NLG metrics.

### BenchmarkRunner
Runs comprehensive benchmarks on RAG systems.

## Usage

```python
from rag.eval import RAGEvaluator, EvalConfig, EvalMetric

config = EvalConfig(metrics=[
    EvalMetric.RETRIEVAL_NDCG,
    EvalMetric.GENERATION_ROUGE,
    EvalMetric.FAITHFULNESS,
])

evaluator = RAGEvaluator(config)
metrics = evaluator.evaluate_retrieval(retrieved, relevant, query)
```

## See Also

- RAG main README: `../README.md`
- Configuration: `config.py`
- Core implementations: `core.py`
