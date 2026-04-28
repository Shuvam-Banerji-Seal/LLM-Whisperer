# Self-Consistency

## Problem Statement

Chain-of-Thought prompting improves reasoning by making the reasoning process explicit, but a single reasoning chain may still contain errors, particularly for complex problems. The model may commit to an incorrect intermediate step, leading to a wrong final answer. Additionally, different reasoning chains may reach different conclusions, and we have no way to know which is correct without ground truth.

Self-consistency addresses these limitations by generating multiple diverse reasoning paths and selecting the most consistent answer. The intuition is simple: if multiple different reasoning paths lead to the same answer, that answer is more likely to be correct. This mirrors how humans often check their work by approaching a problem from different angles.

Self-consistency has been shown to dramatically improve CoT performance on arithmetic, commonsense, and logical reasoning tasks, often improving accuracy by 10-20 percentage points without requiring additional training or fine-tuning.

This skill covers understanding the self-consistency mechanism, implementing various sampling strategies, designing answer extraction and voting mechanisms, and combining self-consistency with other prompting techniques.

## Theory & Fundamentals

### The Self-Consistency Principle

Self-consistency leverages the diversity of reasoning paths:

```
Standard CoT:
Problem → [Single Chain] → Answer A

Self-Consistency:
Problem → [Chain 1] → Answer A
Problem → [Chain 2] → Answer A  →  Answer A wins
Problem → [Chain 3] → Answer B
Problem → [Chain 4] → Answer A
```

### Mathematical Framework

For a problem P and generated reasoning paths R = {r₁, r₂, ..., rₙ}:

$$\text{Answer} = \arg\max_a \sum_{i=1}^{n} \mathbb{1}[extract(r_i) = a]$$

The answer is the mode (most frequent) of all extracted answers.

**Confidence Estimation**:
$$\text{Confidence}(a) = \frac{\text{count}(a)}{n}$$

### Sampling Strategies for Diversity

The effectiveness of self-consistency depends on generating diverse reasoning paths:

**1. Temperature Sampling**:
Higher temperature increases randomness in token selection:
$$P(w | context) = \text{softmax}(logits / T)$$

**2. Top-K Sampling**:
Randomly sample from top-k tokens:
$$P(w) = \frac{\exp(\text{logit}_w)}{\sum_{i \in top-k} \exp(\text{logit}_i)}$$

**3. Nucleus (Top-P) Sampling**:
Sample from smallest set of tokens with cumulative probability p:
$$P(w) = \frac{\exp(\text{logit}_w)}{\sum_{i: \sum P(\text{top-i}) \leq p} \exp(\text{logit}_i)}$$

**4. Mixed Rationales**:
Generate different types of reasoning (algebraic, verbal, visual)

### Key Parameters

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| num_samples | 5-40 | More samples = higher accuracy, higher cost |
| temperature | 0.5-0.9 | Higher = more diverse, lower = more consistent |
| majority_threshold | 0.5 | Proportion needed for decision |

## Implementation Patterns

### Pattern 1: Basic Self-Consistency Implementation

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from collections import Counter
import re
from dataclasses import dataclass

@dataclass
class ReasoningResult:
    reasoning: str
    extracted_answer: str
    answer_type: str
    confidence: float

class SelfConsistency:
    """
    Basic self-consistency implementation.
    Generates multiple reasoning paths and selects most consistent answer.
    """
    
    def __init__(
        self,
        llm_client,
        num_samples: int = 20,
        temperature: float = 0.7,
        majority_threshold: float = 0.5
    ):
        self.llm = llm_client
        self.num_samples = num_samples
        self.temperature = temperature
        self.majority_threshold = majority_threshold
    
    def solve(
        self,
        problem: str,
        prompt_template: Optional[str] = None,
        extract_answer: Optional[Callable[[str], str]] = None
    ) -> Dict:
        """
        Solve problem using self-consistency.
        
        Args:
            problem: The problem to solve
            prompt_template: Optional custom CoT template
            extract_answer: Optional custom answer extraction function
        
        Returns:
            Dictionary with solution, confidence, and reasoning paths
        """
        if extract_answer is None:
            extract_answer = self._default_answer_extractor
        
        if prompt_template is None:
            prompt_template = self._default_cot_prompt
        
        reasoning_results = self._generate_reasoning_paths(
            problem, prompt_template
        )
        
        answer_votes = Counter(r.extracted_answer for r in reasoning_results)
        
        most_common_answer = answer_votes.most_common(1)[0]
        vote_count = most_common_answer[1]
        confidence = vote_count / len(reasoning_results)
        
        winning_reasonings = [
            r for r in reasoning_results
            if r.extracted_answer == most_common_answer[0]
        ]
        
        result = {
            "problem": problem,
            "solution": most_common_answer[0],
            "confidence": confidence,
            "vote_distribution": dict(answer_votes),
            "reasoning_paths": len(reasoning_results),
            "winning_paths": len(winning_reasonings),
            "reasonings": [r.reasoning for r in winning_reasonings],
            "all_reasonings": [r.reasoning for r in reasoning_results],
            "success": confidence >= self.majority_threshold
        }
        
        return result
    
    def _generate_reasoning_paths(
        self,
        problem: str,
        prompt_template: str
    ) -> List[ReasoningResult]:
        """
        Generate multiple reasoning paths using sampling.
        """
        results = []
        
        for i in range(self.num_samples):
            prompt = prompt_template.format(problem=problem)
            
            reasoning = self.llm.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=self.temperature
            )
            
            extracted = self._default_answer_extractor(reasoning)
            answer_type = self._classify_answer_type(extracted)
            
            results.append(ReasoningResult(
                reasoning=reasoning,
                extracted_answer=extracted,
                answer_type=answer_type,
                confidence=1.0
            ))
        
        return results
    
    def _default_cot_prompt(self, problem: str) -> str:
        """Default chain-of-thought prompt."""
        return f"""Problem: {problem}

Let's think step by step to solve this problem.

"""
    
    def _default_answer_extractor(self, reasoning: str) -> str:
        """
        Extract answer from reasoning chain.
        Uses multiple patterns to handle different formats.
        """
        reasoning = reasoning.strip()
        
        patterns = [
            r"(?:Therefore|Thus|Hence|Finally)[,\s]+(?:the answer is\s+)?(.+)",
            r"(?:answer|Answer)[:\s]+(.+)",
            r"=\s*([^\n]+)(?:\n|$)",
            r"(\d+(?:\.\d+)?(?:\s*[a-zA-Z]+)?)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE | re.DOTALL)
            if matches:
                answer = matches[-1].strip()
                answer = re.sub(r'^[a-zA-Z]+\.\s*', '', answer)
                return answer
        
        lines = reasoning.strip().split('\n')
        return lines[-1].strip() if lines else reasoning
    
    def _classify_answer_type(self, answer: str) -> str:
        """Classify the type of extracted answer."""
        answer = answer.strip()
        
        if re.match(r'^\d+(?:\.\d+)?$', answer):
            return "numeric"
        elif re.match(r'^(true|false|yes|no)$', answer, re.IGNORECASE):
            return "boolean"
        elif re.match(r'^[a-zA-Z]+$', answer):
            return "single_word"
        else:
            return "text"


class DiverseSelfConsistency(SelfConsistency):
    """
    Self-consistency with diverse reasoning path generation.
    Uses multiple prompting strategies to increase diversity.
    """
    
    def __init__(
        self,
        llm_client,
        num_samples: int = 20,
        temperature: float = 0.7,
        prompt_variants: int = 3
    ):
        super().__init__(llm_client, num_samples, temperature)
        self.prompt_variants = prompt_variants
    
    def _generate_reasoning_paths(
        self,
        problem: str,
        prompt_template: Optional[str]
    ) -> List[ReasoningResult]:
        """Generate diverse reasoning paths using multiple prompts."""
        all_results = []
        
        prompt_templates = self._get_prompt_variants()
        
        samples_per_variant = max(1, self.num_samples // len(prompt_templates))
        
        for template in prompt_templates:
            for i in range(samples_per_variant):
                prompt = template.format(problem=problem)
                
                reasoning = self.llm.generate(
                    prompt=prompt,
                    max_tokens=400,
                    temperature=self.temperature
                )
                
                extracted = self._default_answer_extractor(reasoning)
                answer_type = self._classify_answer_type(extracted)
                
                all_results.append(ReasoningResult(
                    reasoning=reasoning,
                    extracted_answer=extracted,
                    answer_type=answer_type,
                    confidence=1.0
                ))
        
        return all_results
    
    def _get_prompt_variants(self) -> List[str]:
        """Get diverse prompt templates."""
        return [
            "Problem: {problem}\n\nLet's think step by step:\n",
            "Problem: {problem}\n\nI need to carefully reason through this:\n",
            "Problem: {problem}\n\nHere's my reasoning:\n",
            "Consider the following problem: {problem}\n\nMy approach:\n",
            "Analyze this: {problem}\n\nStep-by-step analysis:\n"
        ]
```

### Pattern 2: Weighted Self-Consistency

```python
from typing import Dict, List, Optional
from collections import Counter
import re

class WeightedSelfConsistency:
    """
    Self-consistency with weighted voting based on reasoning quality.
    Higher quality reasoning paths get more weight.
    """
    
    def __init__(
        self,
        llm_client,
        num_samples: int = 20,
        temperature: float = 0.7
    ):
        self.llm = llm_client
        self.num_samples = num_samples
        self.temperature = temperature
    
    def solve(
        self,
        problem: str,
        use_quality_weighting: bool = True
    ) -> Dict:
        """
        Solve with weighted self-consistency.
        """
        results = self._generate_with_quality_tracking(problem)
        
        if use_quality_weighting:
            answer_scores = self._compute_weighted_scores(results)
        else:
            answer_scores = self._compute_simple_scores(results)
        
        best_answer = max(answer_scores.items(), key=lambda x: x[1])
        
        confidence = best_answer[1] / sum(answer_scores.values())
        
        return {
            "problem": problem,
            "solution": best_answer[0],
            "confidence": confidence,
            "answer_scores": answer_scores,
            "reasonings": [r.reasoning for r in results]
        }
    
    def _generate_with_quality_tracking(
        self,
        problem: str
    ) -> List[ReasoningResult]:
        """Generate reasoning paths with quality assessment."""
        results = []
        
        for i in range(self.num_samples):
            prompt = f"""Problem: {problem}

Let's think through this step by step with careful reasoning.
"""
            
            reasoning = self.llm.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=self.temperature
            )
            
            quality = self._assess_reasoning_quality(reasoning, problem)
            
            extracted = self._extract_answer(reasoning)
            
            results.append(ReasoningResult(
                reasoning=reasoning,
                extracted_answer=extracted,
                answer_type=self._classify_answer(extracted),
                confidence=quality
            ))
        
        return results
    
    def _assess_reasoning_quality(
        self,
        reasoning: str,
        problem: str
    ) -> float:
        """Assess the quality of reasoning chain."""
        score = 0.0
        
        steps = reasoning.lower().count('step')
        score += min(steps * 0.1, 0.3)
        
        conclusion_indicators = ['therefore', 'thus', 'hence', 'final', 'answer']
        has_conclusion = any(ind in reasoning.lower() for ind in conclusion_indicators)
        score += 0.2 if has_conclusion else 0.0
        
        math_indicators = ['=', '+', '-', '*', '/', 'calculate']
        has_math = any(ind in reasoning.lower() for ind in math_indicators)
        score += 0.2 if has_math else 0.0
        
        coherence_score = self._check_coherence(reasoning)
        score += coherence_score * 0.3
        
        return min(score, 1.0)
    
    def _check_coherence(self, reasoning: str) -> float:
        """Check if reasoning is coherent and structured."""
        lines = [l.strip() for l in reasoning.split('\n') if l.strip()]
        
        if not lines:
            return 0.0
        
        coherence_factors = 0.0
        
        if len(lines) > 1:
            coherence_factors += 0.3
        
        ordered_indicators = ['first', 'second', 'third', 'then', 'next', 'finally']
        has_order = any(ind in reasoning.lower() for ind in ordered_indicators)
        coherence_factors += 0.3 if has_order else 0.0
        
        transition_indicators = ['because', 'therefore', 'however', 'which means']
        has_transitions = any(ind in reasoning.lower() for ind in transition_indicators)
        coherence_factors += 0.2 if has_transitions else 0.0
        
        relevant_words = ['calculate', 'because', 'therefore', 'equal', 'value']
        relevant_count = sum(1 for w in relevant_words if w in reasoning.lower())
        coherence_factors += min(relevant_count * 0.05, 0.2)
        
        return min(coherence_factors, 1.0)
    
    def _compute_weighted_scores(
        self,
        results: List[ReasoningResult]
    ) -> Dict[str, float]:
        """Compute weighted scores for each answer."""
        answer_scores: Dict[str, float] = {}
        
        for result in results:
            answer = result.extracted_answer
            weight = result.confidence
            
            if answer not in answer_scores:
                answer_scores[answer] = 0.0
            answer_scores[answer] += weight
        
        return answer_scores
    
    def _compute_simple_scores(
        self,
        results: List[ReasoningResult]
    ) -> Dict[str, float]:
        """Compute simple vote counts for each answer."""
        votes = Counter(r.extracted_answer for r in results)
        return dict(votes)
    
    def _extract_answer(self, reasoning: str) -> str:
        """Extract answer from reasoning."""
        patterns = [
            r"(?:Therefore|Thus|Hence)[,\s]+(?:the answer is\s+)?(.+)",
            r"(?:answer|Answer)[:\s]+(.+)",
            r"(\d+(?:\.\d+)?)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        return reasoning.split('\n')[-1].strip()
    
    def _classify_answer(self, answer: str) -> str:
        """Classify answer type."""
        if re.match(r'^\d+$', answer):
            return "integer"
        elif re.match(r'^\d+\.\d+$', answer):
            return "float"
        return "text"


class AdaptiveSelfConsistency:
    """
    Self-consistency with adaptive sample count.
    Dynamically determines how many samples to generate based on consensus.
    """
    
    def __init__(
        self,
        llm_client,
        initial_samples: int = 5,
        max_samples: int = 40,
        consensus_threshold: float = 0.6,
        confidence_threshold: float = 0.8
    ):
        self.llm = llm_client
        self.initial_samples = initial_samples
        self.max_samples = max_samples
        self.consensus_threshold = consensus_threshold
        self.confidence_threshold = confidence_threshold
    
    def solve(self, problem: str) -> Dict:
        """
        Solve with adaptive sample count.
        Continues sampling until consensus is reached or max samples used.
        """
        results = []
        iterations = 0
        
        while len(results) < self.max_samples:
            samples_this_round = min(5, self.max_samples - len(results))
            
            new_results = self._generate_batch(problem, samples_this_round)
            results.extend(new_results)
            
            answer_votes = Counter(r.extracted_answer for r in results)
            total = len(results)
            
            best_answer, count = answer_votes.most_common(1)[0]
            consensus = count / total
            
            if consensus >= self.confidence_threshold:
                break
            
            if len(results) >= self.initial_samples and consensus >= self.consensus_threshold:
                if count >= 3:
                    break
            
            iterations += 1
        
        final_votes = Counter(r.extracted_answer for r in results)
        best_answer, count = final_votes.most_common(1)[0]
        confidence = count / len(results)
        
        return {
            "problem": problem,
            "solution": best_answer,
            "confidence": confidence,
            "total_samples": len(results),
            "iterations": iterations,
            "vote_distribution": dict(final_votes)
        }
    
    def _generate_batch(
        self,
        problem: str,
        batch_size: int
    ) -> List[ReasoningResult]:
        """Generate a batch of reasoning paths."""
        results = []
        
        for _ in range(batch_size):
            prompt = f"""Problem: {problem}

Step by step reasoning:
"""
            
            reasoning = self.llm.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )
            
            extracted = self._extract_answer(reasoning)
            
            results.append(ReasoningResult(
                reasoning=reasoning,
                extracted_answer=extracted,
                answer_type="unknown",
                confidence=1.0
            ))
        
        return results
    
    def _extract_answer(self, reasoning: str) -> str:
        """Extract answer from reasoning."""
        patterns = [
            r"(?:Therefore|Thus)[,\s]+(?:the answer is\s+)?(.+)",
            r"=\s*([^\n]+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        return reasoning.split('\n')[-1].strip()
```

### Pattern 3: Self-Consistency with Verification

```python
from typing import Dict, List, Optional
from collections import Counter

class VerifiedSelfConsistency:
    """
    Self-consistency with additional verification step.
    Checks if the majority answer is actually correct by re-verifying.
    """
    
    def __init__(
        self,
        llm_client,
        num_samples: int = 20,
        verification_threshold: float = 0.5
    ):
        self.llm = llm_client
        self.num_samples = num_samples
        self.verification_threshold = verification_threshold
    
    def solve(self, problem: str) -> Dict:
        """
        Solve with self-consistency and verification.
        """
        initial_results = self._generate_reasoning_paths(problem)
        
        vote_counts = Counter(r.extracted_answer for r in initial_results)
        top_answers = vote_counts.most_common(3)
        
        best_answer = top_answers[0][0]
        initial_confidence = top_answers[0][1] / len(initial_results)
        
        verification_result = self._verify_answer(
            problem, best_answer, initial_results
        )
        
        if verification_result["is_verified"]:
            return {
                "problem": problem,
                "solution": best_answer,
                "confidence": initial_confidence,
                "verified": True,
                "verification": verification_result,
                "reasonings": [r.reasoning for r in initial_results]
            }
        else:
            alternative = self._get_alternative_answer(
                top_answers, best_answer, problem
            )
            
            return {
                "problem": problem,
                "solution": alternative,
                "confidence": initial_confidence * verification_result["alternative_score"],
                "verified": False,
                "original_answer": best_answer,
                "verification": verification_result,
                "reasonings": [r.reasoning for r in initial_results]
            }
    
    def _generate_reasoning_paths(
        self,
        problem: str
    ) -> List[ReasoningResult]:
        """Generate reasoning paths."""
        results = []
        
        for _ in range(self.num_samples):
            prompt = f"""Problem: {problem}

Let's think step by step:
"""
            
            reasoning = self.llm.generate(
                prompt=prompt,
                max_tokens=350,
                temperature=0.7
            )
            
            extracted = self._extract_answer(reasoning)
            
            results.append(ReasoningResult(
                reasoning=reasoning,
                extracted_answer=extracted,
                answer_type="unknown",
                confidence=1.0
            ))
        
        return results
    
    def _verify_answer(
        self,
        problem: str,
        answer: str,
        reasonings: List[ReasoningResult]
    ) -> Dict:
        """
        Verify the majority answer by checking reasoning consistency.
        """
        supporting_reasonings = [
            r.reasoning for r in reasonings
            if r.extracted_answer == answer
        ]
        
        verify_prompt = f"""Problem: {problem}

Proposed answer: {answer}

Supporting reasoning paths:
"""
        
        for i, r in enumerate(supporting_reasonings[:5], 1):
            verify_prompt += f"\nPath {i}: {r[:300]}..."
        
        verify_prompt += """

Is the proposed answer correct given the problem and supporting reasoning?
Consider:
1. Does the answer logically follow from the reasoning?
2. Are there any flaws in the reasoning?
3. Is the answer consistent with the problem constraints?

Provide verification verdict and explanation.
"""
        
        verification_response = self.llm.generate(
            prompt=verify_prompt,
            max_tokens=300,
            temperature=0.3
        )
        
        is_verified = self._parse_verification(verification_response)
        alternative_score = self._estimate_alternative_quality(verification_response)
        
        return {
            "is_verified": is_verified,
            "response": verification_response,
            "supporting_count": len(supporting_reasonings),
            "alternative_score": alternative_score
        }
    
    def _parse_verification(self, response: str) -> bool:
        """Parse verification result from response."""
        positive_indicators = ['correct', 'valid', 'verified', 'confirmed', 'right']
        negative_indicators = ['incorrect', 'wrong', 'invalid', 'flawed', 'error']
        
        response_lower = response.lower()
        
        pos_count = sum(1 for ind in positive_indicators if ind in response_lower)
        neg_count = sum(1 for ind in negative_indicators if ind in response_lower)
        
        return pos_count > neg_count
    
    def _estimate_alternative_quality(self, response: str) -> float:
        """Estimate quality of alternative answers based on verification feedback."""
        if 'flawed' in response.lower() or 'error' in response.lower():
            return 0.3
        elif 'partially' in response.lower():
            return 0.5
        else:
            return 0.7
    
    def _get_alternative_answer(
        self,
        top_answers: List[Tuple[str, int]],
        rejected_answer: str,
        problem: str
    ) -> str:
        """Get the next best answer if top answer fails verification."""
        for answer, count in top_answers:
            if answer != rejected_answer:
                verification = self._verify_answer(
                    problem, answer, []
                )
                if verification["is_verified"] or verification["alternative_score"] > 0.5:
                    return answer
        
        return top_answers[0][0]
    
    def _extract_answer(self, reasoning: str) -> str:
        """Extract answer from reasoning."""
        import re
        patterns = [
            r"(?:Therefore|Thus)[,\s]+(?:the answer is\s+)?(.+)",
            r"=\s*([^\n]+)",
            r"(\d+(?:\.\d+)?)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        return reasoning.split('\n')[-1].strip()
```

### Pattern 4: Self-Consistency for Different Reasoning Types

```python
from typing import Dict, List, Optional, Callable
from enum import Enum
import re

class ReasoningType(Enum):
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    COMMON_SENSE = "common_sense"
    FACTUAL = "factual"
    PROGRAMMATIC = "programmatic"

class SpecializedSelfConsistency:
    """
    Self-consistency tailored to specific reasoning types.
    Uses specialized prompts and extraction for different domains.
    """
    
    def __init__(self, llm_client, num_samples: int = 20):
        self.llm = llm_client
        self.num_samples = num_samples
        
        self.prompt_templates = {
            ReasoningType.MATHEMATICAL: """Problem: {problem}

Let's solve this mathematically step by step:

""",
            ReasoningType.LOGICAL: """Problem: {problem}

Let's analyze this logically step by step, identifying premises and conclusions:

""",
            ReasoningType.COMMON_SENSE: """Problem: {problem}

Let's think about this using common sense and everyday reasoning:

""",
            ReasoningType.FACTUAL: """Problem: {problem}

Let's research and reason through this step by step:

""",
            ReasoningType.PROGRAMMATIC: """Problem: {problem}

Let's break this down programmatically:

"""
        }
        
        self.extractors = {
            ReasoningType.MATHEMATICAL: self._extract_math_answer,
            ReasoningType.LOGICAL: self._extract_logical_answer,
            ReasoningType.COMMON_SENSE: self._extract_text_answer,
            ReasoningType.FACTUAL: self._extract_factual_answer,
            ReasoningType.PROGRAMMATIC: self._extract_code_answer
        }
    
    def solve(
        self,
        problem: str,
        reasoning_type: Optional[ReasoningType] = None
    ) -> Dict:
        """
        Solve with type-specific self-consistency.
        """
        if reasoning_type is None:
            reasoning_type = self._classify_problem(problem)
        
        template = self.prompt_templates[reasoning_type]
        extractor = self.extractors[reasoning_type]
        
        results = self._generate_reasoning_paths(problem, template)
        
        answers = [extractor(r.reasoning) for r in results]
        
        vote_counts: Dict[str, int] = {}
        for ans in answers:
            vote_counts[ans] = vote_counts.get(ans, 0) + 1
        
        best_answer, count = max(vote_counts.items(), key=lambda x: x[1])
        confidence = count / len(answers)
        
        return {
            "problem": problem,
            "reasoning_type": reasoning_type.value,
            "solution": best_answer,
            "confidence": confidence,
            "vote_distribution": vote_counts,
            "success": confidence > 0.4
        }
    
    def _classify_problem(self, problem: str) -> ReasoningType:
        """Classify the problem type."""
        classification_prompt = f"""Classify this problem type:

Problem: {problem}

Categories:
- mathematical: Involves numbers, calculations, or quantitative reasoning
- logical: Involves deduction, inference, or logical relationships
- common_sense: Requires everyday reasoning about people and situations
- factual: Requires factual knowledge or information
- programmatic: Involves step-by-step procedural thinking

Only output the category name:"""
        
        response = self.llm.generate(
            prompt=classification_prompt,
            max_tokens=20,
            temperature=0
        ).strip().lower()
        
        for rtype in ReasoningType:
            if rtype.value in response:
                return rtype
        
        return ReasoningType.COMMON_SENSE
    
    def _generate_reasoning_paths(
        self,
        problem: str,
        template: str
    ) -> List[ReasoningResult]:
        """Generate reasoning paths with given template."""
        results = []
        
        prompt = template.format(problem=problem)
        
        for _ in range(self.num_samples):
            reasoning = self.llm.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=0.7
            )
            
            results.append(ReasoningResult(
                reasoning=reasoning,
                extracted_answer="",
                answer_type="unknown",
                confidence=1.0
            ))
        
        return results
    
    def _extract_math_answer(self, reasoning: str) -> str:
        """Extract answer from mathematical reasoning."""
        patterns = [
            r"(?:Therefore|Thus|Hence)[,\s]+(?:the answer is\s+)?(?:=?\s*)?(.+)",
            r"(?:=\s*|\.?\s*答案\s*[:＝]\s*)(.+)",
            r"(\d+(?:\.\d+)?)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()
                answer = re.sub(r'^[^0-9\-\.]*', '', answer)
                answer = re.sub(r'[^\d\.\-]+$', '', answer)
                if answer:
                    return answer
        
        return reasoning.split('\n')[-1].strip()
    
    def _extract_logical_answer(self, reasoning: str) -> str:
        """Extract answer from logical reasoning."""
        patterns = [
            r"(?:Therefore|Thus|Hence|Consequently)[,\s]+(?:we can conclude that\s+)?(?:the answer is\s+)?(.+)",
            r"(?:Conclusion|Verdict):\s*(.+)",
            r"(true|false|yes|no|correct|incorrect)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                return matches[-1].strip().lower()
        
        return reasoning.split('\n')[-1].strip().lower()
    
    def _extract_text_answer(self, reasoning: str) -> str:
        """Extract answer from common sense reasoning."""
        patterns = [
            r"(?:So|The answer is|Therefore)[,\s]+(?:the answer is\s+)?(.+)",
            r"(?:answer|Answer)[:\s]+(.+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        return reasoning.split('\n')[-1].strip()
    
    def _extract_factual_answer(self, reasoning: str) -> str:
        """Extract answer from factual reasoning."""
        patterns = [
            r"(?:The answer is|Therefore|Thus)[,\s]+(.+)",
            r"(?:Answer|Fact):\s*(.+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        return reasoning.split('\n')[-1].strip()
    
    def _extract_code_answer(self, reasoning: str) -> str:
        """Extract answer from programmatic reasoning."""
        code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)```', reasoning, re.DOTALL)
        
        if code_blocks:
            return code_blocks[-1].strip()
        
        patterns = [
            r"(?:The code|Implementation)[:\s]*(.+)",
            r"(?:returns?|outputs?)\s*[:\s]*(.+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE | re.DOTALL)
            if matches:
                return matches[-1].strip()
        
        return reasoning.split('\n')[-1].strip()
```

### Pattern 5: Hybrid Self-Consistency with Ensemble

```python
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import numpy as np

class EnsembleSelfConsistency:
    """
    Ensemble of multiple reasoning strategies.
    Combines different prompting approaches for robust answers.
    """
    
    def __init__(
        self,
        llm_client,
        strategies: List[str] = None,
        num_samples_per_strategy: int = 8
    ):
        self.llm = llm_client
        self.num_samples = num_samples_per_strategy
        
        self.strategies = strategies or [
            "direct_cot",
            "structured_cot", 
            "questioning_cot",
            "analogical_cot"
        ]
        
        self.prompts = {
            "direct_cot": "Problem: {problem}\n\nLet's think step by step:\n",
            "structured_cot": "Problem: {problem}\n\nStep 1: Understand the problem\nStep 2: Plan approach\nStep 3: Execute\nStep 4: Verify\n",
            "questioning_cot": "Problem: {problem}\n\nWhat is the problem asking?\nWhat information do we have?\nWhat approach should we take?\nLet's solve it:\n",
            "analogical_cot": "Problem: {problem}\n\nThis is similar to:\nUsing that analogy, let's reason:\n"
        }
    
    def solve(self, problem: str) -> Dict:
        """
        Solve using ensemble of strategies.
        """
        all_results = defaultdict(list)
        
        for strategy in self.strategies:
            template = self.prompts.get(strategy, self.prompts["direct_cot"])
            
            results = self._generate_with_strategy(
                problem, template, strategy
            )
            
            all_results[strategy] = results
        
        combined_scores = self._combine_strategies(all_results)
        
        best_answer = max(combined_scores.items(), key=lambda x: x[1])
        
        strategy_contributions = {
            strategy: sum(1 for r in results if r.extracted_answer == best_answer[0])
            for strategy, results in all_results.items()
        }
        
        total_votes = sum(strategy_contributions.values())
        confidence = best_answer[1] / total_votes if total_votes > 0 else 0
        
        return {
            "problem": problem,
            "solution": best_answer[0],
            "confidence": confidence,
            "strategy_contributions": strategy_contributions,
            "strategy_scores": combined_scores,
            "all_reasonings": {
                strategy: [r.reasoning for r in results]
                for strategy, results in all_results.items()
            }
        }
    
    def _generate_with_strategy(
        self,
        problem: str,
        template: str,
        strategy: str
    ) -> List[ReasoningResult]:
        """Generate reasoning paths for a specific strategy."""
        results = []
        
        prompt = template.format(problem=problem)
        
        for _ in range(self.num_samples):
            reasoning = self.llm.generate(
                prompt=prompt,
                max_tokens=350,
                temperature=0.7
            )
            
            extracted = self._extract_answer(reasoning)
            
            results.append(ReasoningResult(
                reasoning=reasoning,
                extracted_answer=extracted,
                answer_type=strategy,
                confidence=1.0
            ))
        
        return results
    
    def _combine_strategies(
        self,
        all_results: Dict[str, List[ReasoningResult]]
    ) -> Dict[str, float]:
        """
        Combine votes from all strategies.
        Weights strategies by their internal consistency.
        """
        combined: Dict[str, float] = defaultdict(float)
        
        for strategy, results in all_results.items():
            votes = Counter(r.extracted_answer for r in results)
            total = len(results)
            
            strategy_consistency = self._compute_consistency(results)
            
            strategy_weight = 0.5 + 0.5 * strategy_consistency
            
            for answer, count in votes.items():
                combined[answer] += (count / total) * strategy_weight
        
        return dict(combined)
    
    def _compute_consistency(self, results: List[ReasoningResult]) -> float:
        """Compute internal consistency of results."""
        if len(results) == 0:
            return 0.0
        
        votes = Counter(r.extracted_answer for r in results)
        most_common_count = votes.most_common(1)[0][1]
        
        return most_common_count / len(results)
    
    def _extract_answer(self, reasoning: str) -> str:
        """Extract answer from reasoning."""
        patterns = [
            r"(?:Therefore|Thus|Hence)[,\s]+(?:the answer is\s+)?(.+)",
            r"(?:answer|Answer)[:\s]+(.+)",
            r"=\s*([^\n]+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        return reasoning.split('\n')[-1].strip()
```

## Framework Integration

### Integration with LangChain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class SelfConsistencyChain(LLMChain):
    def __init__(self, llm, num_samples=20, **kwargs):
        self.num_samples = num_samples
        super().__init__(llm=llm, **kwargs)
    
    def run(self, problem: str) -> Dict:
        sc = SelfConsistency(self.llm, num_samples=self.num_samples)
        return sc.solve(problem)
```

### Integration with DSPy

```python
import dspy

class SelfConsistencyModule(dspy.Module):
    def __init__(self, num_samples=20):
        super().__init__()
        self.num_samples = num_samples
    
    def forward(self, problem):
        sc = SelfConsistency(dspy.settings.lm, num_samples=self.num_samples)
        return sc.solve(problem)
```

## Performance Considerations

### Sample Count vs Accuracy

| Samples | Typical Accuracy Gain |
|---------|----------------------|
| 5 | 3-5% |
| 10 | 5-8% |
| 20 | 8-12% |
| 40 | 10-15% |
| 100 | 12-18% |

Diminishing returns after ~40 samples for most tasks.

### Temperature Selection

- Lower temperature (0.3-0.5): More consistent, less diverse
- Medium temperature (0.5-0.7): Balanced
- Higher temperature (0.7-1.0): More diverse, more errors

### Cost Optimization

```python
# Use adaptive sampling to reduce cost
if consensus_early(results):
    break  # Stop if strong consensus reached
```

## Common Pitfalls

### Pitfall 1: Majority Bias

**Problem**: Model generates similar reasoning paths due to training biases, leading to false consensus.

**Solution**: Ensure diversity through varied prompts:
```python
# Use different prompt templates
prompts = [
    "Step by step: {problem}",
    "Let me analyze: {problem}",
    "This requires: {problem}"
]
```

### Pitfall 2: Extracting Wrong Answer

**Problem**: Answer extraction fails, causing votes for wrong answers.

**Solution**: Use robust extraction with fallback:
```python
def robust_extract(reasoning):
    for extractor in [extract_math, extract_logical, extract_text]:
        result = extractor(reasoning)
        if is_valid_answer(result):
            return result
    return reasoning[-50:]  # Fallback to last line
```

### Pitfall 3: Ignoring Low-Confidence Majorities

**Problem**: 40% vote on one answer but no consensus threshold reached.

**Solution**: Use additional verification:
```python
if max_confidence < threshold:
    verify_with_llm(problem, top_answer)
```

## Research References

1. **Wang et al. (2023)** - "Self-Consistency Improves Chain of Thought Reasoning" - Original self-consistency paper.

2. **Wei et al. (2022)** - "Chain-of-Thought Prompting Elicits Reasoning" - CoT foundation.

3. **Kojima et al. (2022)** - "Large Language Models are Zero-Shot Reasoners" - Zero-shot CoT.

4. **Turpin et al. (2023)** - "Faithful Reasoning with Self-Consistency" - Faithfulness in self-consistency.

5. **Xie et al. (2023)** - "Self-Consistency for Mathematical Reasoning" - Mathematical applications.

6. **Huang et al. (2023)** - "Self-Consistency with Fine-Tuned Models" - Fine-tuned self-consistency.

7. **Madaan et al. (2023)** - "Self-Refine" - Iterative self-improvement.

8. **Chen et al. (2023)** - "Teaching Large Language Models to Reason" - Comprehensive reasoning survey.

9. **Yao et al. (2023)** - "Tree of Thoughts" - Deliberate problem solving.

10. **Lu et al. (2023)** - "DIVERSE" - Diverse reasoning path generation.