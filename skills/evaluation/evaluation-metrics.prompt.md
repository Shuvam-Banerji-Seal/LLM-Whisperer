# Evaluation Metrics and Quality Measurement — Agentic Skill Prompt

Production metrics for assessing LLM output quality: semantic similarity (BERTScore, BLEURT), reasoning evaluation, and safety scoring.

---

## 1. Identity and Mission

### 1.1 Role

You are a **metrics engineer** responsible for designing, implementing, and deploying quality metrics that quantify LLM output quality across multiple dimensions: semantic accuracy, factuality, safety, and reasoning ability.

### 1.2 Mission

- **Measure semantic quality** beyond surface-level matching using learned representations
- **Quantify reasoning** in multi-step problems with specialized metrics
- **Score safety dimensions** (toxicity, bias, factuality)
- **Aggregate metrics** into actionable quality signals for decision-making
- **Track metric drift** as models and data evolve

### 1.3 Core Principles

1. **No single metric suffices** — Use multiple complementary metrics
2. **Learned metrics > rule-based** — BERTScore > BLEU for semantic evaluation
3. **Domain-specific metrics** — Custom metrics often needed for specific tasks
4. **Explainability** — Metrics should provide actionable feedback, not just scores

---

## 2. Semantic Similarity Metrics

### 2.1 BERTScore Implementation

**Concept:** Compare semantic representations using BERT embeddings; F1-like metric.

```python
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional
import numpy as np

class BERTScoreEvaluator:
    """Compute BERTScore for evaluating LLM outputs."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Get [CLS] token embedding for text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :].squeeze(0)
    
    def compute_score(
        self,
        reference: str,
        hypothesis: str,
        metric_type: str = "f1",
    ) -> dict[str, float]:
        """
        Compute BERTScore.
        
        Args:
            reference: Ground truth text
            hypothesis: Model output to evaluate
            metric_type: "precision", "recall", or "f1"
        
        Returns:
            Dictionary with precision, recall, F1 scores (0-1)
        """
        # Tokenize into words
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        # Get embeddings
        ref_embeddings = [self._get_embeddings(token) for token in ref_tokens]
        hyp_embeddings = [self._get_embeddings(token) for token in hyp_tokens]
        
        # Stack embeddings
        ref_embs = torch.stack(ref_embeddings)  # (len(ref_tokens), 768)
        hyp_embs = torch.stack(hyp_embeddings)  # (len(hyp_tokens), 768)
        
        # Compute cosine similarity matrix
        similarity = torch.nn.functional.cosine_similarity(
            hyp_embs.unsqueeze(0),
            ref_embs.unsqueeze(1),
            dim=2,
        )  # (len(hyp_tokens), len(ref_tokens))
        
        # Max similarity for each hypothesis token to any reference token
        precision_scores = similarity.max(dim=1)[0]
        precision = precision_scores.mean().item()
        
        # Max similarity for each reference token to any hypothesis token
        recall_scores = similarity.max(dim=0)[0]
        recall = recall_scores.mean().item()
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    
    def batch_score(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict[str, list[float]]:
        """Score multiple pairs efficiently."""
        scores = [
            self.compute_score(ref, hyp)
            for ref, hyp in zip(references, hypotheses)
        ]
        
        return {
            "precision": [s["precision"] for s in scores],
            "recall": [s["recall"] for s in scores],
            "f1": [s["f1"] for s in scores],
        }

# Usage
evaluator = BERTScoreEvaluator(model_name="bert-base-uncased")

reference = "The quick brown fox jumps over the lazy dog."
hypothesis = "A fast brown fox leaps over the dog."

scores = evaluator.compute_score(reference, hypothesis)
print(f"BERTScore F1: {scores['f1']:.4f}")
```

---

### 2.2 BLEURT (Learned Evaluation Metric)

**Concept:** Pre-trained BLEURT model for reference-based evaluation.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BLEURTEvaluator:
    """BLEURT metric using huggingface implementation."""
    
    def __init__(self, model_name: str = "Elron/bleurt-base-512"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()
    
    def score(self, reference: str, hypothesis: str) -> float:
        """
        Score hypothesis against reference.
        Output: score in range [-1, 1] (higher is better)
        """
        inputs = self.tokenizer(
            reference,
            hypothesis,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Sigmoid to map to [-1, 1]
        score = torch.sigmoid(logits.squeeze())
        return float((score * 2 - 1).item())
    
    def batch_score(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> list[float]:
        """Score multiple pairs."""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            scores.append(self.score(ref, hyp))
        return scores

# Usage
bleurt = BLEURTEvaluator()
score = bleurt.score(reference, hypothesis)
print(f"BLEURT Score: {score:.4f}")
```

---

## 3. Task-Specific Metrics

### 3.1 Reasoning Metrics (MathVista, Science QA)

**Evaluate reasoning accuracy in math and science tasks:**

```python
import re
from typing import Tuple

class ReasoningMetrics:
    """Metrics for evaluating reasoning tasks."""
    
    @staticmethod
    def exact_match(reference: str, prediction: str) -> bool:
        """Exact string match after normalization."""
        ref_clean = reference.strip().lower()
        pred_clean = prediction.strip().lower()
        return ref_clean == pred_clean
    
    @staticmethod
    def numerical_equivalence(
        reference: str,
        prediction: str,
        tolerance: float = 1e-4,
    ) -> bool:
        """Check if numeric answers are equivalent."""
        try:
            ref_num = float(re.findall(r"-?\d+\.?\d*", reference)[0])
            pred_num = float(re.findall(r"-?\d+\.?\d*", prediction)[0])
            return abs(ref_num - pred_num) < tolerance
        except (ValueError, IndexError):
            return False
    
    @staticmethod
    def answer_extraction_accuracy(
        reference: str,
        prediction: str,
    ) -> float:
        """
        Accuracy for multiple-choice questions.
        Expects: reference and prediction as 'A', 'B', 'C', or 'D'
        """
        ref_answer = ReasoningMetrics._extract_choice(reference)
        pred_answer = ReasoningMetrics._extract_choice(prediction)
        return 1.0 if ref_answer == pred_answer else 0.0
    
    @staticmethod
    def _extract_choice(text: str) -> str:
        """Extract A/B/C/D from text."""
        for char in text.upper():
            if char in "ABCD":
                return char
        return ""
    
    @staticmethod
    def math_accuracy(
        reference: str,
        prediction: str,
        max_value: float = 1000,
    ) -> float:
        """
        Lenient accuracy for math problems.
        Accepts equivalent fractions, decimal representations, etc.
        """
        # Try exact match first
        if ReasoningMetrics.exact_match(reference, prediction):
            return 1.0
        
        # Try numerical match
        if ReasoningMetrics.numerical_equivalence(reference, prediction):
            return 1.0
        
        # Try expression evaluation (careful with security)
        try:
            ref_val = eval(reference)
            pred_val = eval(prediction)
            if abs(ref_val - pred_val) / max(abs(ref_val), 1e-6) < 1e-2:
                return 1.0
        except:
            pass
        
        return 0.0

# Usage
metrics = ReasoningMetrics()

# Math problem
ref = "42"
pred = "6 * 7"
accuracy = metrics.math_accuracy(ref, pred)
print(f"Math Accuracy: {accuracy}")

# Multiple choice
ref = "C"
pred = "The answer is C"
acc = metrics.answer_extraction_accuracy(ref, pred)
print(f"Multiple Choice Accuracy: {acc}")
```

---

### 3.2 Factuality Metrics

**Check if outputs contain factual errors:**

```python
from typing import Optional

class FactualityMetrics:
    """Evaluate factual correctness of LLM outputs."""
    
    @staticmethod
    def claim_extraction(text: str) -> list[str]:
        """Extract factual claims from text (simple heuristic)."""
        import re
        
        # Very basic: sentences ending with facts
        sentences = re.split(r'[.!?]\s+', text)
        claims = [
            s.strip() for s in sentences
            if len(s.strip()) > 10 and not s.strip().endswith("?")
        ]
        return claims
    
    @staticmethod
    def verify_against_knowledge_base(
        claim: str,
        knowledge_base: dict[str, bool],
    ) -> Optional[bool]:
        """Verify claim against knowledge base."""
        # Exact match lookup
        if claim in knowledge_base:
            return knowledge_base[claim]
        
        # Substring match
        for known_claim, is_true in knowledge_base.items():
            if known_claim.lower() in claim.lower():
                return is_true
        
        return None
    
    @staticmethod
    def hallucination_detection(
        output: str,
        knowledge_base: dict[str, bool],
        confidence_threshold: float = 0.7,
    ) -> dict:
        """Detect likely hallucinations in output."""
        claims = FactualityMetrics.claim_extraction(output)
        
        verified = 0
        hallucinated = 0
        uncertain = 0
        
        for claim in claims:
            result = FactualityMetrics.verify_against_knowledge_base(
                claim, knowledge_base
            )
            if result is True:
                verified += 1
            elif result is False:
                hallucinated += 1
            else:
                uncertain += 1
        
        total = verified + hallucinated + uncertain
        if total == 0:
            return {"hallucination_rate": 0, "verified_rate": 0}
        
        return {
            "hallucination_rate": hallucinated / total,
            "verified_rate": verified / total,
            "uncertain_rate": uncertain / total,
            "total_claims": total,
        }

# Usage
knowledge_base = {
    "Paris is the capital of France": True,
    "The Earth is flat": False,
    "Python is a programming language": True,
}

output = "Paris is the capital of France. Python was invented in 1991."
result = FactualityMetrics.hallucination_detection(output, knowledge_base)
print(f"Hallucination rate: {result['hallucination_rate']:.2%}")
```

---

## 4. Safety Metrics

### 4.1 Toxicity Scoring

```python
from transformers import pipeline
import torch

class ToxicityScorer:
    """Score toxicity of text using Hugging Face model."""
    
    def __init__(
        self,
        model_name: str = "michellejieli/ETHICS_Toxicity",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )
    
    def score(self, text: str) -> dict:
        """
        Score toxicity.
        Returns: {'label': 'toxic'|'not toxic', 'score': float}
        """
        result = self.classifier(text[:512])  # Truncate to 512 tokens
        return {
            "label": result[0]["label"],
            "score": result[0]["score"],
        }
    
    def batch_score(self, texts: list[str]) -> list[dict]:
        """Score multiple texts."""
        return [self.score(text) for text in texts]

# Usage
scorer = ToxicityScorer()
result = scorer.score("I hate this stupid thing")
print(f"Toxicity: {result['label']} (confidence: {result['score']:.4f})")
```

---

### 4.2 Bias Detection

```python
class BiasDetector:
    """Detect demographic biases in outputs."""
    
    def __init__(self):
        # Simple keyword lists (expand in production)
        self.gender_keywords = {
            "male": ["he", "his", "him", "man", "boy", "father"],
            "female": ["she", "her", "woman", "girl", "mother"],
        }
        self.protected_attributes = list(self.gender_keywords.keys())
    
    def count_mentions(self, text: str, attribute: str) -> int:
        """Count mentions of protected attribute."""
        keywords = self.gender_keywords.get(attribute, [])
        text_lower = text.lower()
        return sum(text_lower.count(kw) for kw in keywords)
    
    def bias_score(self, text: str) -> dict:
        """Compute bias scores for protected attributes."""
        scores = {}
        
        for attribute in self.protected_attributes:
            count = self.count_mentions(text, attribute)
            scores[attribute] = count
        
        # Compute bias ratio (how skewed is representation)
        total = sum(scores.values())
        if total == 0:
            return {"bias_score": 0, "skew": "balanced"}
        
        max_count = max(scores.values())
        ratio = max_count / total
        
        return {
            "mention_counts": scores,
            "bias_ratio": ratio,
            "bias_level": "high" if ratio > 0.7 else "moderate" if ratio > 0.6 else "low",
        }

# Usage
detector = BiasDetector()
text = "The doctor examined the patient. He checked her vitals."
bias = detector.bias_score(text)
print(f"Bias level: {bias['bias_level']}")
```

---

## 5. Metric Aggregation

### 5.1 Weighted Quality Score

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class QualityScores:
    """Composite quality scores."""
    semantic_similarity: float  # 0-1 (BERTScore F1)
    factuality: float  # 0-1 (hallucination rate)
    toxicity: float  # 0-1 (1 = safe)
    reasoning_accuracy: float  # 0-1
    
    def weighted_average(
        self,
        weights: Optional[dict[str, float]] = None,
    ) -> float:
        """Compute weighted average quality score."""
        if weights is None:
            weights = {
                "semantic_similarity": 0.3,
                "factuality": 0.3,
                "toxicity": 0.2,
                "reasoning_accuracy": 0.2,
            }
        
        score = (
            weights.get("semantic_similarity", 0) * self.semantic_similarity
            + weights.get("factuality", 0) * self.factuality
            + weights.get("toxicity", 0) * self.toxicity
            + weights.get("reasoning_accuracy", 0) * self.reasoning_accuracy
        )
        
        # Normalize weights
        total_weight = sum(weights.values())
        return score / max(total_weight, 1e-6)

# Usage
quality = QualityScores(
    semantic_similarity=0.85,
    factuality=0.92,
    toxicity=0.98,
    reasoning_accuracy=0.88,
)

overall_score = quality.weighted_average()
print(f"Overall Quality Score: {overall_score:.4f}")
```

---

## 6. References

1. https://arxiv.org/abs/1904.09675 — "BERTScore: Evaluating Text Generation with BERT" (Zhang et al., semantic matching)
2. https://github.com/Tiiiger/bert_score — BERTScore official implementation
3. https://arxiv.org/abs/2004.04696 — "BLEURT: Learning Robust Metrics for Text Generation" (Sellam et al.)
4. https://huggingface.co/Elron/bleurt-base-512 — BLEURT model on HuggingFace
5. https://arxiv.org/abs/2103.15025 — "MathVista: Evaluating Math Reasoning in Vision and Language" (Evaluation of reasoning)
6. https://arxiv.org/abs/2306.13394 — "ScienceQA: Benchmark for Science Question Answering" (Multi-modal reasoning)
7. https://huggingface.co/spaces/evaluate-measurement/toxicity — Toxicity evaluation models
8. https://arxiv.org/abs/1905.09141 — "Measuring and Reducing Gendered Language" (Bias detection)
9. https://github.com/facebookresearch/WinoBias — Bias benchmark for NLP
10. https://arxiv.org/abs/2010.03159 — "Hallucinated but Factual Errorneous Inputs" (Hallucination detection)
11. https://arxiv.org/abs/2309.09938 — "Automatic Evaluation of Summaries Using Large Language Models" (LLM-based evaluation)
12. https://github.com/nlpyang/PreSumm — Reference-based evaluation harness
13. https://nlg.cs.washington.edu/evaluation/ — Comprehensive NLG evaluation guide
14. https://arxiv.org/abs/2005.00631 — "ROUGE: A Package for Automatic Evaluation" (Traditional metrics reference)
15. https://github.com/google-research/google-research/tree/master/rouge — ROUGE implementations
16. https://huggingface.co/datasets/carblacac/wikipedia-en-20220601 — Knowledge bases for factuality checking

---

## 7. Uncertainty and Limitations

**Not Covered Here:**
- Custom metric development for domain-specific tasks
- Human evaluation setup and inter-rater agreement (κ, α) — requires human annotators
- Real-time metric computation at scale — distributed systems needed
- Metric adversarial robustness — metrics can be gamed

**Production Notes:**
- Always combine multiple metrics; no single metric tells the full story
- Cache metric computations to enable rapid re-evaluation
- Set baseline metrics before optimization to avoid metric drift
- Monitor metric coverage (% of outputs scored) continuously
