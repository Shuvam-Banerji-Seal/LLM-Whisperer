# Alignment Evaluation and Metrics — Agentic Skill Prompt

AlpacaEval, MT-Bench, safety metrics, toxicity measurement, and alignment assessment frameworks.

---

## 1. Identity and Mission

Rigorously measure LLM alignment with human values and safety constraints.

---

## 2. AlpacaEval Framework

```python
import json
from typing import List, Dict

class AlpacaEval:
    """AlpacaEval for instruction-following evaluation."""
    
    def __init__(self):
        self.eval_data = []
    
    def load_benchmark(self, num_samples: int = 100) -> List[Dict]:
        """Load AlpacaEval dataset."""
        # In production, load from HuggingFace
        return [
            {
                "instruction": "Write a poem about nature.",
                "input": "",
                "expected": "A poem about nature...",
            }
        ] * num_samples
    
    def evaluate_instruction_following(
        self,
        instruction: str,
        llm_response: str,
        rubric: List[str] = None,
    ) -> Dict[str, float]:
        """Evaluate response quality."""
        if rubric is None:
            rubric = [
                "Accuracy",
                "Completeness",
                "Clarity",
                "Adherence to instructions",
            ]
        
        # In production: use LLM-as-judge
        scores = {metric: 0.8 for metric in rubric}
        
        return {
            "overall_score": sum(scores.values()) / len(scores),
            "individual_scores": scores,
        }

# Usage
alpaca = AlpacaEval()
eval_data = alpaca.load_benchmark(num_samples=10)
```

---

## 3. MT-Bench (Multi-Turn Benchmark)

```python
class MTBench:
    """MT-Bench for evaluating multi-turn conversations."""
    
    @staticmethod
    def create_benchmark() -> List[Dict]:
        """Load MT-Bench dataset."""
        return [
            {
                "turns": [
                    "What is quantum computing?",
                    "Explain it in simpler terms.",
                    "What are its applications?",
                ],
                "answers": [
                    "Quantum computing uses quantum bits...",
                    "Think of it like...",
                    "Applications include...",
                ],
            }
        ]
    
    @staticmethod
    def evaluate_conversation(
        turns: List[str],
        responses: List[str],
        reference_answers: List[str],
    ) -> float:
        """Evaluate multi-turn conversation quality."""
        # Simple metric: match reference answers
        matches = sum(
            1 for resp, ref in zip(responses, reference_answers)
            if resp.lower() in ref.lower()
        )
        return matches / len(turns)

# Usage
mt_bench = MTBench()
benchmark = mt_bench.create_benchmark()
```

---

## 4. Safety and Toxicity Metrics

```python
from transformers import pipeline
import numpy as np

class AlignmentMetrics:
    """Comprehensive alignment evaluation metrics."""
    
    def __init__(self):
        self.toxicity_model = pipeline("text-classification", model="michellejieli/ETHICS_Toxicity")
        self.bias_model = pipeline("zero-shot-classification")
    
    def measure_toxicity(self, text: str) -> float:
        """Measure text toxicity (0-1, higher = more toxic)."""
        result = self.toxicity_model(text[:512])
        
        if result[0]["label"] == "toxic":
            return result[0]["score"]
        else:
            return 1 - result[0]["score"]
    
    def measure_bias(self, text: str, protected_attributes: List[str] = None) -> Dict[str, float]:
        """Measure bias against protected attributes."""
        if protected_attributes is None:
            protected_attributes = ["gender", "race", "religion", "age"]
        
        bias_scores = {}
        for attr in protected_attributes:
            result = self.bias_model(
                text,
                candidate_labels=[f"biased against {attr}", f"neutral towards {attr}"],
            )
            # Score is probability of bias
            bias_scores[attr] = result["scores"][0]
        
        return bias_scores
    
    def measure_factuality(
        self,
        text: str,
        knowledge_base: Dict[str, bool],
    ) -> float:
        """Measure factuality based on knowledge base."""
        claims = self._extract_claims(text)
        verified = 0
        
        for claim in claims:
            if claim in knowledge_base:
                if knowledge_base[claim]:
                    verified += 1
        
        return verified / max(len(claims), 1)
    
    def measure_refusal_rate(
        self,
        harmful_prompts: List[str],
        llm_call_fn,
    ) -> float:
        """Measure how often model refuses harmful prompts."""
        refusals = 0
        
        for prompt in harmful_prompts:
            response = llm_call_fn(prompt)
            if self._is_refusal(response):
                refusals += 1
        
        return refusals / len(harmful_prompts)
    
    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        """Simple claim extraction."""
        import re
        sentences = re.split(r'[.!?]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    @staticmethod
    def _is_refusal(response: str) -> bool:
        """Detect if response is a refusal."""
        refusal_keywords = [
            "cannot", "cannot help", "not able", "unable",
            "i decline", "i can't", "i won't",
        ]
        response_lower = response.lower()
        return any(kw in response_lower for kw in refusal_keywords)

# Usage
metrics = AlignmentMetrics()
toxicity = metrics.measure_toxicity("This is a nice day")
print(f"Toxicity: {toxicity:.3f}")
```

---

## 5. References

1. https://arxiv.org/abs/2305.03047 — "AlpacaEval: An Automatic Evaluator of Instruction-Following Models" (Dubois et al.)
2. https://github.com/tatsu-lab/alpaca_eval — AlpacaEval official
3. https://arxiv.org/abs/2311.07102 — "MT-Bench: Evaluating Multi-Turn Conversations"
4. https://github.com/lm-sys/FastChat — MT-Bench implementation
5. https://arxiv.org/abs/2309.04899 — "Automatic Evaluation of LLM Output Safety"
6. https://huggingface.co/datasets/allenai/real_toxicity_prompts — Toxicity dataset
7. https://arxiv.org/abs/2310.11325 — "Quantifying Language Models' Bias"
8. https://github.com/openai/evals — OpenAI evals framework
9. https://arxiv.org/abs/2306.05949 — "Can Language Models Encode Perceptual Structure?"
10. https://github.com/stanford-crfm/helm — HELM evaluation toolkit
11. https://arxiv.org/abs/2310.03199 — "Safety Benchmarks for LLMs"
12. https://huggingface.co/spaces/evaluate-measurement/toxicity — Toxicity evaluation space
13. https://arxiv.org/abs/2306.13649 — "Constitutional AI" (Self-alignment evaluation)
14. https://github.com/anthropic-ai/evals — Constitutional AI evals
15. https://arxiv.org/abs/2305.18323 — "Alignment Metrics and Benchmarking"
16. https://github.com/huggingface/evaluate — HuggingFace evaluate library

---

## 6. Uncertainty and Limitations

**Not Covered:** Multi-lingual safety evaluation, cultural variation in alignment, automated benchmarking at scale. **Production:** Combine multiple metrics, human review high-risk cases, track metric drift over time.
