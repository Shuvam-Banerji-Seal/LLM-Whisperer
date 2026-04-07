"""
Advanced Prompting & Reasoning Techniques
==========================================

Demonstrates:
- Chain-of-Thought (CoT) prompting
- Tree-of-Thought (ToT) exploration
- Self-Consistency voting
- Few-shot learning strategies
- RAG-enhanced prompting
- Prompt optimization techniques
"""

import json
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# 1. Chain-of-Thought (CoT) Prompting
# ============================================================================


class CoTStrategy(Enum):
    """Different CoT strategies"""

    BASIC = "basic"  # Simple step-by-step
    DETAILED = "detailed"  # Detailed reasoning
    STRUCTURED = "structured"  # Structured format


@dataclass
class CoTExample:
    """Example for few-shot CoT learning"""

    question: str
    reasoning_steps: List[str]
    final_answer: str


class ChainOfThoughtPrompter:
    """Generate CoT prompts"""

    @staticmethod
    def basic_cot_prompt(
        question: str, examples: Optional[List[CoTExample]] = None
    ) -> str:
        """Generate basic CoT prompt"""
        prompt = ""

        # Add few-shot examples
        if examples:
            prompt += "Examples:\n\n"
            for i, example in enumerate(examples, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Question: {example.question}\n"
                prompt += "Let me think step by step:\n"
                for step in example.reasoning_steps:
                    prompt += f"- {step}\n"
                prompt += f"Answer: {example.final_answer}\n\n"

        # Add instruction
        prompt += f"Question: {question}\n"
        prompt += "Let me think step by step:\n"

        return prompt

    @staticmethod
    def detailed_cot_prompt(question: str) -> str:
        """Generate detailed CoT prompt"""
        return f"""
Question: {question}

Please provide a detailed step-by-step analysis:

1. Identify the problem
2. Break down the problem into sub-problems
3. Solve each sub-problem
4. Combine the solutions
5. Verify the final answer

Reasoning:
"""

    @staticmethod
    def structured_cot_prompt(question: str) -> str:
        """Generate structured CoT prompt"""
        return f"""
Question: {question}

Provide your answer in the following structured format:

PROBLEM ANALYSIS:
- What is being asked?
- What information do we have?
- What information is missing?

REASONING STEPS:
1. First consideration:
2. Second consideration:
3. Final synthesis:

VERIFICATION:
- Does this answer make sense?
- Can we verify it another way?

FINAL ANSWER:
"""

    @staticmethod
    def self_refine_prompt(question: str, initial_answer: str) -> str:
        """Generate prompt for self-refinement"""
        return f"""
Original Question: {question}

Initial Answer: {initial_answer}

Now, please review your answer and improve it:

1. Is the answer correct?
2. Are there any errors in reasoning?
3. Can the explanation be clearer?
4. Is the answer complete?

Improved Answer:
"""


# ============================================================================
# 2. Tree-of-Thought (ToT) Prompting
# ============================================================================


@dataclass
class ThoughtNode:
    """A node in the thought tree"""

    content: str
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = None
    score: float = 0.0  # Score from LLM evaluation

    def __post_init__(self):
        if self.children is None:
            self.children = []


class TreeOfThoughtPrompter:
    """Generate ToT prompts and evaluate thoughts"""

    @staticmethod
    def generate_thoughts_prompt(
        question: str, context: str = "", num_thoughts: int = 3
    ) -> str:
        """Generate prompt for exploring multiple thought paths"""
        return f"""
Question: {question}
{f"Context: {context}" if context else ""}

Generate {num_thoughts} different approaches to solve this problem:

Approach 1:
- Key insight:
- Steps:
- Potential challenges:

Approach 2:
- Key insight:
- Steps:
- Potential challenges:

Approach 3:
- Key insight:
- Steps:
- Potential challenges:
"""

    @staticmethod
    def evaluate_thought_prompt(question: str, thought: str) -> str:
        """Generate prompt to evaluate a thought"""
        return f"""
Question: {question}

Thought/Approach: {thought}

Evaluate this approach on a scale of 1-10:
- Correctness: (1-10)
- Clarity: (1-10)
- Completeness: (1-10)
- Efficiency: (1-10)

Overall Score: [score]
Justification: [brief explanation]

Next Steps: [what should be done next]
"""

    @staticmethod
    def expand_thought_prompt(question: str, thought: str) -> str:
        """Generate prompt to expand on a thought"""
        return f"""
Question: {question}

Current Thought: {thought}

Please expand on this thought by:
1. Adding more details to each step
2. Providing specific examples
3. Addressing potential concerns
4. Connecting to related concepts

Expanded Thought:
"""


# ============================================================================
# 3. Self-Consistency Strategy
# ============================================================================


class SelfConsistencyPrompter:
    """Generate prompts for self-consistency approach"""

    @staticmethod
    def generate_diverse_prompt(question: str, prompt_variant: int) -> str:
        """Generate different prompt variations for same question"""
        variants = [
            # Variant 1: Direct approach
            f"Question: {question}\nAnswer:",
            # Variant 2: Detailed reasoning
            f"Question: {question}\nPlease think through this carefully and explain your reasoning step by step.\nAnswer:",
            # Variant 3: Counterargument
            f"Question: {question}\nConsider multiple perspectives before answering.\nAnswer:",
            # Variant 4: Analogy
            f"Question: {question}\nTry to think of similar problems and how they were solved.\nAnswer:",
            # Variant 5: Decomposition
            f"Question: {question}\nBreak this into smaller parts and solve each part.\nAnswer:",
        ]

        return variants[prompt_variant % len(variants)]

    @staticmethod
    def aggregate_answers(answers: List[str]) -> Dict:
        """Aggregate multiple answers using voting"""
        # Simple majority voting
        answer_counts = {}
        for answer in answers:
            # Normalize answer
            normalized = answer.strip().lower()
            answer_counts[normalized] = answer_counts.get(normalized, 0) + 1

        # Find most common answer
        most_common = max(answer_counts, key=answer_counts.get)
        confidence = answer_counts[most_common] / len(answers)

        return {
            "answer": most_common,
            "confidence": confidence,
            "vote_distribution": answer_counts,
            "total_attempts": len(answers),
        }


# ============================================================================
# 4. Few-Shot Learning Strategies
# ============================================================================


@dataclass
class FewShotExample:
    """Few-shot learning example"""

    input_text: str
    output_text: str
    metadata: Optional[Dict] = None


class FewShotSelector:
    """Select relevant few-shot examples"""

    @staticmethod
    def random_selection(
        examples: List[FewShotExample], num_examples: int = 3
    ) -> List[FewShotExample]:
        """Random selection of examples"""
        return random.sample(examples, min(num_examples, len(examples)))

    @staticmethod
    def similarity_based_selection(
        query: str, examples: List[FewShotExample], num_examples: int = 3
    ) -> List[FewShotExample]:
        """Select examples most similar to query (using keyword overlap)"""
        query_words = set(query.lower().split())

        scored_examples = []
        for example in examples:
            example_words = set(example.input_text.lower().split())
            similarity = len(query_words & example_words) / len(
                query_words | example_words
            )
            scored_examples.append((example, similarity))

        scored_examples.sort(key=lambda x: x[1], reverse=True)
        return [example for example, _ in scored_examples[:num_examples]]

    @staticmethod
    def diverse_selection(
        examples: List[FewShotExample], num_examples: int = 3
    ) -> List[FewShotExample]:
        """Select diverse examples to cover different scenarios"""
        if len(examples) <= num_examples:
            return examples

        selected = [examples[0]]
        remaining = examples[1:]

        for _ in range(num_examples - 1):
            # Select example most different from already selected
            max_diversity = -1
            best_example = None

            for example in remaining:
                min_similarity = float("inf")
                for selected_ex in selected:
                    sim = len(
                        set(example.input_text.lower().split())
                        & set(selected_ex.input_text.lower().split())
                    )
                    min_similarity = min(min_similarity, sim)

                if min_similarity > max_diversity:
                    max_diversity = min_similarity
                    best_example = example

            if best_example:
                selected.append(best_example)
                remaining.remove(best_example)

        return selected


class FewShotPromptBuilder:
    """Build few-shot prompts"""

    @staticmethod
    def build_prompt(
        task_description: str, examples: List[FewShotExample], query: str
    ) -> str:
        """Build complete few-shot prompt"""
        prompt = f"Task: {task_description}\n\n"

        prompt += "Examples:\n"
        for i, example in enumerate(examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Input: {example.input_text}\n"
            prompt += f"Output: {example.output_text}\n"

        prompt += f"\nNow solve this:\n"
        prompt += f"Input: {query}\n"
        prompt += f"Output:\n"

        return prompt


# ============================================================================
# 5. RAG-Enhanced Prompting
# ============================================================================


@dataclass
class RetrievedContext:
    """Retrieved context for RAG"""

    text: str
    source: str
    relevance_score: float


class RAGPromptBuilder:
    """Build prompts enhanced with retrieved context"""

    @staticmethod
    def build_rag_prompt(
        question: str,
        retrieved_contexts: List[RetrievedContext],
        system_prompt: str = "",
    ) -> str:
        """Build prompt with retrieved context"""
        prompt = ""

        if system_prompt:
            prompt += f"{system_prompt}\n\n"

        # Add retrieved context
        if retrieved_contexts:
            prompt += "Relevant Information:\n"
            for i, context in enumerate(retrieved_contexts, 1):
                prompt += f"\n[Source {i}: {context.source}]\n"
                prompt += f"{context.text}\n"

            prompt += "\n" + "=" * 50 + "\n\n"

        # Add question
        prompt += f"Question: {question}\n\n"
        prompt += "Based on the information above, provide a detailed answer:\n"

        return prompt

    @staticmethod
    def build_fact_verification_prompt(claim: str, supporting_facts: List[str]) -> str:
        """Build prompt for fact verification"""
        prompt = f"Claim to verify: {claim}\n\n"

        prompt += "Supporting evidence:\n"
        for i, fact in enumerate(supporting_facts, 1):
            prompt += f"{i}. {fact}\n"

        prompt += (
            "\nBased on the evidence, is the claim true, false, or inconclusive?\n"
        )
        prompt += "Explain your reasoning:\n"

        return prompt


# ============================================================================
# 6. Prompt Optimization
# ============================================================================


class PromptOptimizer:
    """Techniques for optimizing prompts"""

    @staticmethod
    def persona_prompt(base_prompt: str, persona: str) -> str:
        """Add persona to prompt"""
        return f"You are {persona}.\n\n{base_prompt}"

    @staticmethod
    def temperature_directive(prompt: str, temperature: float = 0.7) -> str:
        """Add temperature directive"""
        if temperature < 0.5:
            directive = "Be precise and factual."
        elif temperature > 0.9:
            directive = "Be creative and explore diverse possibilities."
        else:
            directive = "Balance accuracy with some creative thinking."

        return f"{directive}\n\n{prompt}"

    @staticmethod
    def constraint_prompt(base_prompt: str, constraints: List[str]) -> str:
        """Add constraints to prompt"""
        prompt = base_prompt + "\n\nConstraints:\n"
        for i, constraint in enumerate(constraints, 1):
            prompt += f"{i}. {constraint}\n"

        return prompt

    @staticmethod
    def output_format_prompt(base_prompt: str, format_spec: str) -> str:
        """Specify output format"""
        return f"{base_prompt}\n\nOutput format:\n{format_spec}\n"


# ============================================================================
# 7. Complete Reasoning System
# ============================================================================


class AdvancedReasoningSystem:
    """Complete system for advanced reasoning"""

    def __init__(self):
        self.cot = ChainOfThoughtPrompter()
        self.tot = TreeOfThoughtPrompter()
        self.sc = SelfConsistencyPrompter()
        self.few_shot = FewShotPromptBuilder()
        self.rag = RAGPromptBuilder()
        self.optimizer = PromptOptimizer()

    def generate_cot_prompt(
        self, question: str, strategy: CoTStrategy = CoTStrategy.BASIC
    ) -> str:
        """Generate CoT prompt"""
        if strategy == CoTStrategy.BASIC:
            return self.cot.basic_cot_prompt(question)
        elif strategy == CoTStrategy.DETAILED:
            return self.cot.detailed_cot_prompt(question)
        else:
            return self.cot.structured_cot_prompt(question)

    def generate_tot_prompt(self, question: str) -> str:
        """Generate ToT prompt"""
        return self.tot.generate_thoughts_prompt(question)

    def generate_self_consistent_prompts(
        self, question: str, num_variants: int = 5
    ) -> List[str]:
        """Generate multiple prompt variants"""
        return [
            self.sc.generate_diverse_prompt(question, i) for i in range(num_variants)
        ]

    def build_complete_prompt(
        self,
        question: str,
        use_cot: bool = True,
        use_rag: bool = False,
        few_shot_examples: Optional[List[FewShotExample]] = None,
        retrieved_contexts: Optional[List[RetrievedContext]] = None,
    ) -> str:
        """Build complete optimized prompt"""

        # Start with base
        prompt = ""

        # Add RAG context if provided
        if use_rag and retrieved_contexts:
            prompt = self.rag.build_rag_prompt(question, retrieved_contexts)

        # Add few-shot examples if provided
        if few_shot_examples:
            examples_text = "\n\nLearning from examples:\n"
            for i, ex in enumerate(few_shot_examples, 1):
                examples_text += f"Example {i}: {ex.input_text} → {ex.output_text}\n"
            prompt += examples_text

        # Add CoT if requested
        if use_cot:
            prompt += f"\nQuestion: {question}\n"
            prompt += "Please reason step by step:\n"
        else:
            prompt += f"\nQuestion: {question}\nAnswer:\n"

        # Add formatting
        prompt = self.optimizer.output_format_prompt(
            prompt, "- Be concise\n- Be accurate\n- Provide sources"
        )

        return prompt


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    system = AdvancedReasoningSystem()

    question = "What is the capital of France and what is its historical significance?"

    # Basic CoT
    print("=" * 60)
    print("BASIC CoT PROMPT:")
    print("=" * 60)
    print(system.generate_cot_prompt(question, CoTStrategy.BASIC))

    # Tree of Thought
    print("\n" + "=" * 60)
    print("TREE OF THOUGHT PROMPT:")
    print("=" * 60)
    print(system.generate_tot_prompt(question))

    # Self-Consistency variants
    print("\n" + "=" * 60)
    print("SELF-CONSISTENCY VARIANTS:")
    print("=" * 60)
    variants = system.generate_self_consistent_prompts(question, 3)
    for i, variant in enumerate(variants, 1):
        print(f"\nVariant {i}:")
        print(variant)
