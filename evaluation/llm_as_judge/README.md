# LLM-as-Judge Evaluation

Evaluate LLM outputs using judge LLMs with standardized rubrics.

## Overview

LLM-as-Judge uses a reference LLM to evaluate the quality of model outputs across multiple dimensions. This approach provides:

- **Flexible Evaluation**: Evaluate any generative task without test cases
- **Standardized Criteria**: Use predefined rubrics or create custom ones
- **Multiple Judges**: Get consensus scores from multiple judge models
- **Calibration**: Measure and correct for judge biases
- **Inter-rater Agreement**: Assess consistency across judges

## Quick Start

### Basic Evaluation

```python
from evaluation.llm_as_judge import Judge, StandardRubrics, JudgeEvaluationRunner

# Create a judge (implement your judge class)
judge = MyJudge(model_id="gpt-4")

# Use a standard rubric
rubric = StandardRubrics.answer_relevance()

# Batch evaluate
runner = JudgeEvaluationRunner(judge, rubric)
items = [
    {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    # ... more items
]

results = runner.evaluate_batch(items, criteria=["Relevance"])
runner.print_summary(results)
runner.save_results("evaluation/results/judge_results.json")
```

### Custom Rubrics

```python
from evaluation.llm_as_judge import Rubric, RubricCriterion

# Create custom rubric
rubric = Rubric("Customer Support Quality", "Evaluate support response quality")

rubric.add_criterion(RubricCriterion(
    name="Problem Resolution",
    description="Does the response solve the customer's problem?",
    anchors={
        1: "Response doesn't address the problem at all",
        2: "Response partially addresses the problem",
        3: "Response addresses the problem but with gaps",
        4: "Response mostly solves the problem",
        5: "Response completely solves the customer's problem"
    }
))

rubric.add_criterion(RubricCriterion(
    name="Tone & Empathy",
    description="Is the tone professional and empathetic?",
    anchors={
        1: "Rude or dismissive tone",
        2: "Neutral but lacks empathy",
        3: "Adequately professional",
        4: "Professional and somewhat empathetic",
        5: "Excellent tone showing genuine care"
    }
))
```

### Multi-Judge Consensus

```python
from evaluation.llm_as_judge import JudgmentAggregator

# Get judgments from multiple judges
judge1 = ClaudeJudge()
judge2 = LlamaJudge()
judge3 = PrometheusJudge()

scores1 = [judge1.judge(...) for item in items]
scores2 = [judge2.judge(...) for item in items]
scores3 = [judge3.judge(...) for item in items]

# Compute agreement
kappa = JudgmentAggregator.fleiss_kappa({
    "judge1": scores1,
    "judge2": scores2,
    "judge3": scores3
})
print(f"Inter-judge agreement (Fleiss' kappa): {kappa:.3f}")
# kappa > 0.81: excellent, > 0.61: substantial, > 0.41: moderate
```

## Standard Rubrics

### Answer Relevance
Evaluates how well the response addresses the question.

```python
rubric = StandardRubrics.answer_relevance()
```

**Scores:**
- 5: Perfectly relevant and focused
- 4: Mostly relevant with minor irrelevance
- 3: Addresses query but with some irrelevant parts
- 2: Tangentially related but misses main points
- 1: Completely off-topic

### Faithfulness
Evaluates whether response is grounded in provided context.

```python
rubric = StandardRubrics.faithfulness()
```

**Scores:**
- 5: Completely faithful, no hallucinations
- 4: Accurate with very few unsupported claims
- 3: Mostly accurate with minor errors
- 2: Several inaccuracies or unsupported claims
- 1: Multiple factual errors and hallucinations

### Coherence
Evaluates logical flow, clarity, and organization.

```python
rubric = StandardRubrics.coherence()
```

**Scores:**
- 5: Perfectly clear, coherent, and well-structured
- 4: Clear and well-organized with minor issues
- 3: Generally clear with some organizational issues
- 2: Difficult to follow with poor organization
- 1: Incoherent and confusing

### Correctness
Evaluates technical and factual accuracy.

```python
rubric = StandardRubrics.correctness()
```

**Scores:**
- 5: Completely correct and accurate
- 4: Correct with minor errors
- 3: Mostly correct with some errors
- 2: Multiple significant errors
- 1: Fundamentally incorrect or dangerous

### Completeness
Evaluates coverage and comprehensiveness.

```python
rubric = StandardRubrics.completeness()
```

**Scores:**
- 5: Complete and thorough coverage
- 4: Comprehensive with minor gaps
- 3: Covers main points but lacks some details
- 2: Significant gaps in coverage
- 1: Extremely incomplete, missing most aspects

### Helpfulness
Evaluates overall utility and actionability.

```python
rubric = StandardRubrics.helpfulness()
```

**Scores:**
- 5: Extremely helpful, actionable, and insightful
- 4: Very helpful and mostly actionable
- 3: Moderately helpful
- 2: Minimally helpful
- 1: Not helpful at all

## Judge Models

### Supported Judges

| Judge | Cost | Latency | Pros | Cons |
|-------|------|---------|------|------|
| GPT-4 | $0.03/1K | Fast | State-of-art, stable | Expensive |
| Claude 3.5 | $0.003/1K | Fast | Good quality, cheaper | External API |
| Prometheus 2 | Free | Slow | Open-source, no cost | Slightly lower quality |
| Llama 3.1 | Free | Medium | Open-source, fast | Requires hosting |

### Implementing a Judge

```python
from evaluation.llm_as_judge import Judge, JudgmentResult

class MyJudge(Judge):
    def judge(self, query: str, response: str, rubric, criterion: str) -> JudgmentResult:
        # Call your LLM API
        judgment_text = self._call_llm(
            query=query,
            response=response,
            rubric=rubric,
            criterion=criterion
        )
        
        # Extract score
        score = self._extract_score(judgment_text)
        
        return JudgmentResult(
            content_id=f"{query[:20]}_{criterion}",
            query=query,
            response=response,
            criterion=criterion,
            score=score,
            rationale=judgment_text
        )
    
    def _call_llm(self, query, response, rubric, criterion):
        # Your API call logic
        pass
```

## Aggregation & Calibration

### Inter-Rater Agreement

```python
from evaluation.llm_as_judge import JudgmentAggregator

# Cohen's kappa for 2 judges
kappa = JudgmentAggregator.cohens_kappa(
    judgments1=[4, 5, 3, 4, 5],
    judgments2=[4, 4, 3, 5, 5]
)

# Fleiss' kappa for multiple judges
kappa_multi = JudgmentAggregator.fleiss_kappa({
    "judge1": [4, 5, 3, 4, 5],
    "judge2": [4, 4, 3, 5, 5],
    "judge3": [5, 5, 3, 4, 5]
})
```

### Judge Calibration

```python
from evaluation.llm_as_judge import JudgeCalibration

# Compute judge bias
bias = JudgeCalibration.compute_bias(
    judge_scores=[4, 5, 3, 4, 5],
    reference_scores=[5, 5, 3, 5, 5]
)  # Negative means judge under-scores

# Compute variance (score spread)
variance = JudgeCalibration.compute_variance(
    judge_scores=[4, 5, 3, 4, 5],
    reference_scores=[5, 5, 3, 5, 5]
)

# Correlation with reference
correlation = JudgeCalibration.pearson_correlation(
    scores1=[4, 5, 3, 4, 5],
    scores2=[5, 5, 3, 5, 5]
)
```

## Configuration

### rubrics.yaml

```yaml
rubrics:
  answer_relevance:
    weight: 1.0
    required: true
  
  faithfulness:
    weight: 1.0
    required: true
  
  coherence:
    weight: 0.8
    required: false

judges:
  gpt4:
    model: gpt-4
    cost_per_1k: 0.03
    latency_ms: 500
  
  claude:
    model: claude-3.5-sonnet
    cost_per_1k: 0.003
    latency_ms: 800
  
  prometheus:
    model: prometheus-2
    cost_per_1k: 0.0
    latency_ms: 2000
```

## References

- DeepEval: https://github.com/confident-ai/deepeval
- Prometheus: https://github.com/prometheus-eval/prometheus-eval
- G-Eval: https://arxiv.org/abs/2303.16634
- LLM-Rubric: https://github.com/microsoft/LLM-Rubric
