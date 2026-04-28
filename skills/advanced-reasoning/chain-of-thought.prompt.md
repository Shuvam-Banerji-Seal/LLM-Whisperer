# Chain-of-Thought Prompting

## Problem Statement

Large language models achieve remarkable capabilities, but their reasoning process remains opaque. When presented with multi-step problems, standard prompting (providing the input and expecting the output) often fails because the model must implicitly perform multiple reasoning steps without explicit guidance. The model may shortcut to an answer, miss crucial intermediate steps, or get lost in complex reasoning chains.

Chain-of-Thought (CoT) prompting addresses this by making the reasoning process explicit. Rather than asking "What is 15% of 80?", we ask "What is 15% of 80? First, calculate 10% which is 8. Then calculate 5% which is 4. Finally, add them together: 8 + 4 = 12." This explicit reasoning chain significantly improves accuracy, especially for complex reasoning tasks.

The power of CoT comes from its ability to leverage the model's pre-trained reasoning capabilities by providing natural language "scratch space." By externalizing the reasoning process, we allow the model to decompose complex problems, catch errors in its own reasoning, and build upon intermediate results.

This skill covers understanding chain-of-thought prompting techniques, implementing various CoT variants, combining CoT with other prompting methods, and applying CoT to complex reasoning tasks.

## Theory & Fundamentals

### The Reasoning Gap

Standard prompting assumes the model can perform reasoning in a "single hop":

```
Input → Model → Output
```

But complex reasoning requires multiple steps:

```
Input → Step 1 → Step 2 → Step 3 → ... → Output
         ↓
    (Implicit reasoning chain)
```

### Chain-of-Thought Properties

**Key Properties of Effective CoT**:

1. **Sequential**: Reasoning steps follow a logical order
2. **Intermediate**: Each step produces a meaningful intermediate result
3. **Grounded**: Each step connects to previous steps
4. **Verifiable**: Steps can be checked for correctness
5. **Decomposable**: Complex problems break into simpler sub-problems

**Mathematical Framework**:

For a problem P requiring n reasoning steps, CoT produces:
$$\text{Output} = f_n(s_{n-1}, c_n) \text{ where } s_{n-1} = f_{n-1}(s_{n-2}, c_{n-1})$$

where $s_i$ is the state after step $i$ and $c_i$ is the reasoning content of step $i$.

### CoT Categories

**Zero-shot CoT** (Kojima et al., 2022):
```
Q: [question]
A: Let's think step by step.
```

**Few-shot CoT** (Wei et al., 2022):
```
Q: [example problem with step-by-step solution]
Q: [example problem with step-by-step solution]
Q: [target problem]
A: [reasoning chain]
```

**Self-Consistency CoT** (Wang et al., 2023):
- Generate multiple reasoning chains
- Select most consistent answer

### When CoT Works Best

**Effective For**:
- Mathematical reasoning
- Logical deduction
- Multi-step calculations
- Causal reasoning
- Planning tasks
- Debugging and error correction

**Less Effective For**:
- Simple factual retrieval
- Single-token predictions
- Tasks requiring world knowledge only
- Highly ambiguous questions

## Implementation Patterns

### Pattern 1: Zero-Shot Chain-of-Thought

```python
from typing import Dict, List, Optional, Tuple
import re

class ZeroShotCoT:
    """
    Zero-shot Chain-of-Thought prompting.
    Simply append "Let's think step by step" to trigger reasoning.
    """
    
    REASONING_PHRASES = [
        "Let's think step by step:",
        "Let me reason through this:",
        "First, let me analyze this:",
        "Step by step, the solution is:",
        "Here's my reasoning:",
        "To solve this, I need to:",
        "Let me break this down:",
        "Starting with the given information:"
    ]
    
    def __init__(
        self,
        llm_client,
        max_reasoning_tokens: int = 500,
        temperature: float = 0.3
    ):
        self.llm = llm_client
        self.max_reasoning_tokens = max_reasoning_tokens
        self.temperature = temperature
    
    def generate(
        self,
        question: str,
        reasoning_phrase: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate answer using zero-shot CoT.
        
        Args:
            question: The question to answer
            reasoning_phrase: Custom reasoning trigger phrase
        
        Returns:
            reasoning: The generated reasoning chain
            answer: The final answer extracted
        """
        reasoning_phrase = reasoning_phrase or self.REASONING_PHRASES[0]
        
        prompt = f"{question}\n\n{reasoning_phrase}"
        
        reasoning = self.llm.generate(
            prompt=prompt,
            max_tokens=self.max_reasoning_tokens,
            temperature=self.temperature
        )
        
        answer = self._extract_answer(reasoning)
        
        return reasoning, answer
    
    def _extract_answer(self, reasoning: str) -> str:
        """
        Extract final answer from reasoning chain.
        Uses pattern matching to find the answer.
        """
        patterns = [
            r"(?:Therefore|Thus|So|Hence|Finally|Answer)[,\s]+(?:the answer is\s+)?(.+)",
            r"(?:=|\:)(.+?)(?:\.|$)",
            r"(\d+(?:\.\d+)?(?:\s*[a-zA-Z]+)?)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE | re.DOTALL)
            if matches:
                return matches[-1].strip()
        
        return reasoning.split('\n')[-1].strip()
    
    def batch_generate(
        self,
        questions: List[str],
        reasoning_phrase: Optional[str] = None
    ) -> List[Dict]:
        """
        Batch process questions with zero-shot CoT.
        """
        results = []
        
        for question in questions:
            try:
                reasoning, answer = self.generate(question, reasoning_phrase)
                results.append({
                    "question": question,
                    "reasoning": reasoning,
                    "answer": answer,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "question": question,
                    "reasoning": None,
                    "answer": None,
                    "success": False,
                    "error": str(e)
                })
        
        return results


class ZeroShotCoTWithVerification(ZeroShotCoT):
    """
    Zero-shot CoT with built-in verification.
    Re-checks the answer using separate verification prompt.
    """
    
    def __init__(
        self,
        llm_client,
        max_reasoning_tokens: int = 500,
        verify_threshold: float = 0.8
    ):
        super().__init__(llm_client, max_reasoning_tokens)
        self.verify_threshold = verify_threshold
    
    def generate_with_verification(
        self,
        question: str
    ) -> Dict:
        """
        Generate answer with verification of the reasoning.
        """
        reasoning, answer = self.generate(question)
        
        verification_result = self._verify_reasoning(question, reasoning, answer)
        
        if verification_result["confidence"] < self.verify_threshold:
            return {
                "question": question,
                "reasoning": reasoning,
                "answer": answer,
                "verified": False,
                "verification": verification_result,
                "alternative_answer": None
            }
        
        return {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
            "verified": True,
            "verification": verification_result
        }
    
    def _verify_reasoning(
        self,
        question: str,
        reasoning: str,
        answer: str
    ) -> Dict:
        """
        Verify the correctness of the reasoning chain.
        """
        verify_prompt = f"""Question: {question}

Reasoning:
{reasoning}

Answer: {answer}

Is the reasoning correct and does the answer follow logically? Rate confidence from 0 to 1 and explain any errors.
"""
        
        verification = self.llm.generate(
            prompt=verify_prompt,
            max_tokens=200,
            temperature=0.1
        )
        
        confidence_match = re.search(r"(\d+\.?\d*)", verification)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        return {
            "verification_text": verification,
            "confidence": min(max(confidence, 0.0), 1.0)
        }
```

### Pattern 2: Few-Shot Chain-of-Thought

```python
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class CoTExample:
    question: str
    reasoning: str
    answer: str

class FewShotCoT:
    """
    Few-shot Chain-of-Thought prompting.
    Uses hand-crafted examples to guide reasoning.
    """
    
    def __init__(
        self,
        llm_client,
        examples: List[CoTExample],
        max_reasoning_tokens: int = 500,
        temperature: float = 0.3
    ):
        self.llm = llm_client
        self.examples = examples
        self.max_reasoning_tokens = max_reasoning_tokens
        self.temperature = temperature
    
    @classmethod
    def from_domain(
        cls,
        llm_client,
        domain: str,
        num_examples: int = 4
    ) -> "FewShotCoT":
        """
        Create FewShotCoT with domain-specific examples.
        """
        examples_by_domain = {
            "math": [
                CoTExample(
                    question="If a store sells 3 apples for $1.50, how much do 12 apples cost?",
                    reasoning="Step 1: Find the price per apple. $1.50 / 3 = $0.50 per apple\nStep 2: Multiply by 12 apples. 12 × $0.50 = $6.00",
                    answer="$6.00"
                ),
                CoTExample(
                    question="A train travels 60 miles per hour. How far will it travel in 2.5 hours?",
                    reasoning="Step 1: Identify the formula. Distance = Speed × Time\nStep 2: Plug in values. Distance = 60 mph × 2.5 hours\nStep 3: Calculate. 60 × 2.5 = 150 miles",
                    answer="150 miles"
                ),
                CoTExample(
                    question="If X > 5 and X < 10, and X is an integer, what are possible values of X?",
                    reasoning="Step 1: X > 5 means X must be 6 or greater\nStep 2: X < 10 means X must be 9 or less\nStep 3: X is an integer, so possible values are 6, 7, 8, 9",
                    answer="6, 7, 8, 9"
                )
            ],
            "logic": [
                CoTExample(
                    question="All A are B. Some B are C. Can we conclude that some A are C?",
                    reasoning="Step 1: We know all A are B (A ⊆ B)\nStep 2: We know some B are C (B ∩ C ≠ ∅)\nStep 3: From A ⊆ B and B ∩ C ≠ ∅, we cannot guarantee A ∩ C ≠ ∅\nStep 4: The intersection of A and C could be empty\nConclusion: No, we cannot conclude that some A are C",
                    answer="Cannot be concluded"
                ),
                CoTExample(
                    question="If it rains, the ground is wet. The ground is wet. Can we conclude it rained?",
                    reasoning="Step 1: P → W (If it rains, ground is wet)\nStep 2: W (Ground is wet)\nStep 3: Modus ponens requires P → Q and Q, not Q → P\nStep 4: We have W → P? No, we have P → W\nStep 5: From W alone, we cannot conclude P (could have watered the grass)",
                    answer="No, this is the fallacy of affirming the consequent"
                )
            ],
            "coding": [
                CoTExample(
                    question="Write a function to check if a string is a palindrome.",
                    reasoning="Step 1: A palindrome reads the same forwards and backwards\nStep 2: We can compare characters from both ends\nStep 3: Use two pointers, one at start, one at end\nStep 4: Move pointers toward center, comparing at each step\nStep 5: If any comparison fails, return False\nStep 6: If pointers cross without mismatch, return True",
                    answer="```python\ndef is_palindrome(s):\n    left, right = 0, len(s) - 1\n    while left < right:\n        if s[left] != s[right]:\n            return False\n        left += 1\n        right -= 1\n    return True\n```"
                )
            ]
        }
        
        return cls(
            llm_client=llm_client,
            examples=examples_by_domain.get(domain, examples_by_domain["math"]),
            max_reasoning_tokens=500
        )
    
    def generate(self, question: str) -> Tuple[str, str]:
        """
        Generate answer using few-shot CoT.
        """
        prompt = self._build_prompt(question)
        
        response = self.llm.generate(
            prompt=prompt,
            max_tokens=self.max_reasoning_tokens,
            temperature=self.temperature
        )
        
        reasoning, answer = self._parse_response(response)
        
        return reasoning, answer
    
    def _build_prompt(self, question: str) -> str:
        """
        Build prompt with few-shot examples.
        """
        parts = []
        
        for example in self.examples:
            parts.append(f"Q: {example.question}")
            parts.append(f"A: {example.reasoning}\nTherefore, the answer is {example.answer}")
            parts.append("")
        
        parts.append(f"Q: {question}")
        parts.append("A:")
        
        return "\n".join(parts)
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse reasoning and answer from response.
        """
        if "Therefore, the answer is" in response:
            parts = response.split("Therefore, the answer is")
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            lines = response.strip().split('\n')
            if lines:
                reasoning = '\n'.join(lines[:-1])
                answer = lines[-1].strip()
            else:
                reasoning = response
                answer = ""
        
        return reasoning, answer


class DynamicFewShotCoT:
    """
    Dynamically selects the most relevant few-shot examples
    based on semantic similarity to the target question.
    """
    
    def __init__(
        self,
        llm_client,
        example_bank: List[CoTExample],
        embedding_model,
        max_examples: int = 4
    ):
        self.llm = llm_client
        self.example_bank = example_bank
        self.embedding_model = embedding_model
        self.max_examples = max_examples
    
    def generate(self, question: str) -> Tuple[str, str]:
        """
        Generate with dynamically selected examples.
        """
        selected = self._select_examples(question)
        
        prompt = self._build_prompt(question, selected)
        
        response = self.llm.generate(prompt=prompt, max_tokens=500)
        
        return self._parse_response(response)
    
    def _select_examples(self, question: str) -> List[CoTExample]:
        """
        Select most relevant examples using embeddings.
        """
        question_embedding = self.embedding_model.encode(question)
        
        similarities = []
        for example in self.example_bank:
            example_embedding = self.embedding_model.encode(
                example.question + " " + example.reasoning
            )
            similarity = self._cosine_similarity(question_embedding, example_embedding)
            similarities.append((example, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [ex for ex, _ in similarities[:self.max_examples]]
    
    def _cosine_similarity(self, a, b):
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _build_prompt(self, question: str, examples: List[CoTExample]) -> str:
        """Build prompt with selected examples."""
        parts = []
        
        for example in examples:
            parts.append(f"Q: {example.question}")
            parts.append(f"A: {example.reasoning}\nTherefore, the answer is {example.answer}\n")
        
        parts.append(f"Q: {question}")
        parts.append("A:")
        
        return "\n".join(parts)
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse response."""
        if "Therefore, the answer is" in response:
            parts = response.split("Therefore, the answer is")
            return parts[0].strip(), parts[1].strip()
        return response, ""
```

### Pattern 3: Program-Aided Chain-of-Thought

```python
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ProgramStep:
    line_number: int
    code: str
    result: Any
    explanation: str

class ProgramAidedCoT:
    """
    Chain-of-Thought enhanced with program execution.
    The model generates reasoning steps that include executable code.
    """
    
    def __init__(
        self,
        llm_client,
        execution_backend=None,
        max_steps: int = 20
    ):
        self.llm = llm_client
        self.execution_backend = execution_backend or PythonExecutor()
        self.max_steps = max_steps
    
    def solve(
        self,
        question: str,
        domain_knowledge: Optional[str] = None
    ) -> Dict:
        """
        Solve problem using program-aided reasoning.
        
        Returns:
            Dictionary with reasoning steps, code, results, and final answer
        """
        solution_steps = []
        context = {}
        current_step = 0
        
        full_question = question
        if domain_knowledge:
            full_question = f"{domain_knowledge}\n\n{question}"
        
        while current_step < self.max_steps:
            prompt = self._build_step_prompt(
                full_question,
                solution_steps,
                context
            )
            
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.3
            )
            
            step = self._parse_step(response)
            
            if step is None:
                break
            
            if step.code:
                execution_result = self.execution_backend.execute(step.code)
                
                if execution_result["success"]:
                    step.result = execution_result["output"]
                    context[step.line_number] = step.result
                else:
                    step.result = f"Error: {execution_result['error']}"
            
            solution_steps.append(step)
            
            if self._is_final_step(step):
                break
            
            current_step += 1
        
        answer = self._extract_final_answer(solution_steps, context)
        
        return {
            "question": question,
            "steps": solution_steps,
            "answer": answer,
            "success": answer is not None
        }
    
    def _build_step_prompt(
        self,
        question: str,
        previous_steps: List[ProgramStep],
        context: Dict
    ) -> str:
        """Build prompt for generating next step."""
        parts = [
            f"Problem: {question}",
            ""
        ]
        
        if previous_steps:
            parts.append("Work so far:")
            for step in previous_steps:
                if step.code:
                    parts.append(f"  Step {step.line_number}: {step.code}")
                    parts.append(f"  Result: {step.result}")
                    parts.append(f"  Explanation: {step.explanation}")
                else:
                    parts.append(f"  {step.explanation}")
            parts.append("")
        
        if context:
            parts.append(f"Known values: {context}")
            parts.append("")
        
        parts.append("What is the next step in the reasoning? Provide code if calculation is needed.")
        
        return "\n".join(parts)
    
    def _parse_step(self, response: str) -> Optional[ProgramStep]:
        """Parse LLM response into a step."""
        lines = response.strip().split('\n')
        
        code_lines = []
        explanation_lines = []
        in_code_block = False
        line_number = len(explanation_lines)
        
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                code_lines.append(line)
            else:
                explanation_lines.append(line)
        
        code = "\n".join(code_lines) if code_lines else None
        explanation = " ".join(explanation_lines)
        
        if not explanation and not code:
            return None
        
        return ProgramStep(
            line_number=len(previous_steps) + 1 if 'previous_steps' in dir() else 1,
            code=code,
            result=None,
            explanation=explanation
        )
    
    def _is_final_step(self, step: ProgramStep) -> bool:
        """Check if this is the final step."""
        final_keywords = ["final answer", "therefore", "conclusion", "answer is"]
        return any(keyword in step.explanation.lower() for keyword in final_keywords)
    
    def _extract_final_answer(
        self,
        steps: List[ProgramStep],
        context: Dict
    ) -> Optional[str]:
        """Extract final answer from steps."""
        for step in reversed(steps):
            answer_match = re.search(
                r"(?:the answer is|answer:|final answer)[:\s]+(.+)",
                step.explanation,
                re.IGNORECASE
            )
            if answer_match:
                return answer_match.group(1).strip()
        
        if context:
            return str(list(context.values())[-1])
        
        return None


class PythonExecutor:
    """Simple Python code executor for program-aided reasoning."""
    
    def execute(self, code: str) -> Dict:
        """Execute Python code and return result."""
        try:
            local_vars = {}
            exec(code, {}, local_vars)
            
            output = None
            for key, value in local_vars.items():
                if not key.startswith('_'):
                    output = value
                    break
            
            return {
                "success": True,
                "output": output,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }
    
    def evaluate_expression(self, expression: str, context: Dict) -> Any:
        """Evaluate a single expression with context."""
        try:
            return eval(expression, {}, context)
        except Exception as e:
            return f"Error: {e}"
```

### Pattern 4: Tree-of-Thought Planning with CoT

```python
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import heapq
from collections import deque

@dataclass
class ThoughtNode:
    """A node in the thought tree."""
    thought: str
    reasoning: str
    value: float
    depth: int
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    state_snapshot: Optional[Dict] = None

class TreeOfThoughtWithCoT:
    """
    Combines Tree-of-Thought exploration with Chain-of-Thought reasoning.
    Explores multiple reasoning paths and prunes low-value branches.
    """
    
    def __init__(
        self,
        llm_client,
        value_function: Callable[[str], float],
        beam_width: int = 2,
        max_depth: int = 5
    ):
        self.llm = llm_client
        self.value_function = value_function
        self.beam_width = beam_width
        self.max_depth = max_depth
    
    def solve(
        self,
        problem: str,
        prompt_template: Optional[str] = None
    ) -> Tuple[str, List[ThoughtNode]]:
        """
        Solve problem using Tree-of-Thought with CoT.
        
        Returns:
            best_solution: The highest-value solution found
            solution_path: The reasoning path leading to it
        """
        root = ThoughtNode(
            thought="Starting point",
            reasoning=f"Problem: {problem}",
            value=0.0,
            depth=0
        )
        
        candidates = [(0.0, root)]
        
        for depth in range(self.max_depth):
            next_candidates = []
            
            for score, node in candidates:
                if depth > 0:
                    new_thoughts = self._generate_thoughts(
                        problem, node, prompt_template
                    )
                else:
                    new_thoughts = self._generate_initial_thoughts(
                        problem, prompt_template
                    )
                
                for thought, reasoning in new_thoughts:
                    value = self.value_function(reasoning)
                    
                    child = ThoughtNode(
                        thought=thought,
                        reasoning=reasoning,
                        value=value,
                        depth=depth + 1,
                        parent=node
                    )
                    
                    node.children.append(child)
                    next_candidates.append((value, child))
            
            candidates = heapq.nlargest(
                self.beam_width,
                next_candidates
            )
        
        best_leaf = max(candidates, key=lambda x: x[1].value)[1]
        
        solution_path = self._extract_path(best_leaf)
        best_solution = best_leaf.reasoning
        
        return best_solution, solution_path
    
    def _generate_initial_thoughts(
        self,
        problem: str,
        template: Optional[str]
    ) -> List[Tuple[str, str]]:
        """Generate initial reasoning directions."""
        prompt = f"""Problem: {problem}

Generate 3 different approaches to solve this problem. For each approach,
provide a brief description and initial reasoning steps.

Format:
Approach 1: [description]
Initial reasoning: [steps]

Approach 2: [description]
Initial reasoning: [steps]

Approach 3: [description]
Initial reasoning: [steps]
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=500)
        
        approaches = self._parse_approaches(response)
        
        return approaches
    
    def _generate_thoughts(
        self,
        problem: str,
        current_node: ThoughtNode,
        template: Optional[str]
    ) -> List[Tuple[str, str]]:
        """Generate next thoughts by extending current reasoning."""
        prompt = f"""Problem: {problem}

Current reasoning:
{current_node.reasoning}

Extend this reasoning by one more step. What comes next in solving this problem?

Provide:
Next thought: [description]
Extended reasoning: [detailed step]
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=300)
        
        thought, reasoning = self._parse_extension(response)
        
        return [(thought, reasoning)]
    
    def _parse_approaches(self, response: str) -> List[Tuple[str, str]]:
        """Parse multiple approaches from response."""
        approaches = []
        
        pattern = r"(?:Approach \d+:|Thought:|Next thought:)\s*(.+?)\n(?:Initial reasoning:|Extended reasoning:)\s*(.+?)(?=(?:Approach \d+:)|(?:$))"
        
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            approach = match[0].strip()
            reasoning = match[1].strip()
            approaches.append((approach, reasoning))
        
        return approaches
    
    def _parse_extension(self, response: str) -> Tuple[str, str]:
        """Parse single thought extension."""
        thought_match = re.search(
            r"(?:Next thought:|Thought:)\s*(.+)",
            response
        )
        reasoning_match = re.search(
            r"(?:Extended reasoning:|Reasoning:)\s*(.+)",
            response
        )
        
        thought = thought_match.group(1).strip() if thought_match else response[:50]
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response
        
        return thought, reasoning
    
    def _extract_path(self, node: ThoughtNode) -> List[ThoughtNode]:
        """Extract reasoning path from root to node."""
        path = []
        current = node
        
        while current.parent is not None:
            path.append(current)
            current = current.parent
        
        path.append(current)
        path.reverse()
        
        return path


class SystematicReasoningCoT:
    """
    Systematic reasoning using CoT with explicit problem decomposition.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def solve(
        self,
        problem: str,
        problem_type: str = "auto"
    ) -> Dict:
        """
        Apply systematic CoT problem solving.
        
        Steps:
        1. Classify problem type
        2. Decompose into sub-problems
        3. Solve each sub-problem with CoT
        4. Combine solutions
        """
        if problem_type == "auto":
            problem_type = self._classify_problem(problem)
        
        if problem_type == "math":
            return self._solve_math_problem(problem)
        elif problem_type == "logical":
            return self._solve_logical_problem(problem)
        elif problem_type == "factual":
            return self._solve_factual_problem(problem)
        else:
            return self._solve_general_problem(problem)
    
    def _classify_problem(self, problem: str) -> str:
        """Classify the problem type."""
        classification_prompt = f"""Classify this problem type:
Problem: {problem}

Categories:
- math: Requires numerical calculation or mathematical reasoning
- logical: Requires logical deduction or reasoning
- factual: Requires factual knowledge or information retrieval
- general: Doesn't fit specific categories

Answer with only the category name:"""
        
        response = self.llm.generate(
            prompt=classification_prompt,
            max_tokens=20,
            temperature=0.0
        )
        
        return response.strip().lower()
    
    def _solve_math_problem(self, problem: str) -> Dict:
        """Solve mathematical problem with systematic CoT."""
        steps = [
            ("identify", "What are the given quantities and what is being asked?"),
            ("formulate", "What mathematical operations or equations are needed?"),
            ("calculate", "Execute the calculations step by step"),
            ("verify", "Check if the answer makes sense")
        ]
        
        reasoning_chain = []
        
        for step_name, question in steps:
            prompt = f"{problem}\n\nPrevious reasoning:\n" + "\n".join(reasoning_chain)
            prompt += f"\n\n{step_name.capitalize()}: {question}"
            
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            reasoning_chain.append(f"{step_name.capitalize()}: {response}")
        
        final_answer = self._extract_math_answer(reasoning_chain[-1])
        
        return {
            "problem": problem,
            "type": "math",
            "reasoning_steps": reasoning_chain,
            "final_answer": final_answer
        }
    
    def _solve_logical_problem(self, problem: str) -> Dict:
        """Solve logical problem with systematic CoT."""
        steps = [
            ("premises", "What are the given facts or premises?"),
            ("inferences", "What logical inferences can we draw?"),
            ("conclusion", "What conclusion follows from the premises?")
        ]
        
        reasoning_chain = []
        
        for step_name, question in steps:
            prompt = f"{problem}\n\nPrevious reasoning:\n" + "\n".join(reasoning_chain)
            prompt += f"\n\n{step_name.capitalize()}: {question}"
            
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            reasoning_chain.append(f"{step_name.capitalize()}: {response}")
        
        return {
            "problem": problem,
            "type": "logical",
            "reasoning_steps": reasoning_chain,
            "final_answer": reasoning_chain[-1]
        }
    
    def _solve_factual_problem(self, problem: str) -> Dict:
        """Solve factual problem with CoT."""
        steps = [
            ("gather", "What information do we need to answer this?"),
            ("retrieve", "Based on available knowledge, what is the answer?"),
            ("cite", "What evidence supports this answer?")
        ]
        
        reasoning_chain = []
        
        for step_name, question in steps:
            prompt = f"{problem}\n\nPrevious reasoning:\n" + "\n".join(reasoning_chain)
            prompt += f"\n\n{step_name.capitalize()}: {question}"
            
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            reasoning_chain.append(f"{step_name.capitalize()}: {response}")
        
        return {
            "problem": problem,
            "type": "factual",
            "reasoning_steps": reasoning_chain,
            "final_answer": reasoning_chain[-1]
        }
    
    def _solve_general_problem(self, problem: str) -> Dict:
        """Solve general problem with CoT."""
        prompt = f"""Solve this problem step by step:

Problem: {problem}

Provide a clear, step-by-step solution with reasoning at each step.
"""
        
        reasoning = self.llm.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )
        
        return {
            "problem": problem,
            "type": "general",
            "reasoning_steps": [reasoning],
            "final_answer": reasoning
        }
    
    def _extract_math_answer(self, text: str) -> str:
        """Extract mathematical answer from text."""
        patterns = [
            r"(?:=|\:)\s*(\d+(?:\.\d+)?)",
            r"(?:answer is|ANSWER)[:\s]+(.+)",
            r"(\d+(?:\.\d+)?(?:\s*[a-zA-Z]+)?)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].strip()
        
        return text.split('\n')[-1].strip()
```

## Framework Integration

### Integration with LangChain

```python
from langchain.prompts import PromptTemplate

class CoTPromptTemplate(PromptTemplate):
    def __init__(self):
        template = """Question: {question}

Let's think step by step:
{reasoning}

Therefore, the answer is {answer}"""
        
        super().__init__(template=template)
```

### Integration with DSPy

```python
import dspy

class CoTSignature(dspy.Signature):
    input_field = dspy.InputField()
    reasoning = dspy.OutputField()
    answer = dspy.OutputField()

class CoTModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CoTSignature)
    
    def forward(self, input_field):
        return self.predict(input_field)
```

## Common Pitfalls

### Pitfall 1: Incomplete Reasoning Chains

**Problem**: Model skips steps, making reasoning incomplete.

**Solution**: Use explicit step markers:
```python
PROMPT_TEMPLATE = """
Step 1: [explicit step]
Step 2: [explicit step]  
Step 3: [explicit step]
Answer: [final answer]
"""
```

### Pitfall 2: Arithmetic Errors in CoT

**Problem**: Model makes mistakes in intermediate calculations.

**Solution**: Use program-aided CoT:
```python
# Generate code for calculations instead of inline arithmetic
Step: Calculate 15% of 80
Code: result = 0.15 * 80
Result: 12
```

### Pitfall 3: Overconfident Wrong Reasoning

**Problem**: Model presents incorrect reasoning as fact.

**Solution**: Add verification step:
```python
# After generating reasoning, verify
verify_prompt = f"""Is this reasoning correct?
{reasoning}
"""
```

## Research References

1. **Wei et al. (2022)** - "Chain-of-Thought Prompting Elicits Reasoning" - Original CoT paper.

2. **Kojima et al. (2022)** - "Large Language Models are Zero-Shot Reasoners" - Zero-shot CoT.

3. **Wang et al. (2023)** - "Self-Consistency Improves CoT Reasoning" - Self-consistency method.

4. **Nye et al. (2021)** - "Show Your Work" - Scratchpad reasoning for computation.

5. **Zhou et al. (2023)** - "Least-to-Most Prompting" - Decomposition-based CoT.

6. **Yao et al. (2023)** - "Tree of Thoughts" - Deliberate problem solving with CoT.

7. **Press et al. (2022)** - "Measuring Chain-of-Thought Reasoning" - Analysis of CoT.

8. **Tafjord et al. (2022)** - "FActScore" - Fine-grained factuality in CoT.

9. **Turpin et al. (2023)** - "CoT Catchy" - Faithfulness in chain-of-thought.

10. **Qiao et al. (2023)** - "Reasoning with Language Model Prompting" - Survey of reasoning approaches.