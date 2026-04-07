# Comprehensive LLM Benchmarks and Evaluation Frameworks Guide 2026

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Version:** 2.0  
**Scope:** 30+ benchmarks, 10+ evaluation frameworks, code examples, leaderboards

---

## Table of Contents

1. [Standard LLM Benchmarks](#standard-llm-benchmarks)
2. [Code Generation Benchmarks](#code-generation-benchmarks)
3. [Mathematical Reasoning Benchmarks](#mathematical-reasoning-benchmarks)
4. [Reasoning and Logic Benchmarks](#reasoning-and-logic-benchmarks)
5. [Specialized Domain Benchmarks](#specialized-domain-benchmarks)
6. [Evaluation Frameworks and Tools](#evaluation-frameworks-and-tools)
7. [Evaluation Metrics and Implementations](#evaluation-metrics-and-implementations)
8. [LLM-as-Judge Evaluation](#llm-as-judge-evaluation)
9. [Benchmark Limitations and Gaming](#benchmark-limitations-and-gaming)
10. [Leaderboards and Rankings](#leaderboards-and-rankings)
11. [Benchmark Creation Guidelines](#benchmark-creation-guidelines)
12. [Tools and Repositories](#tools-and-repositories)

---

## Standard LLM Benchmarks

### 1. MMLU (Massive Multitask Language Understanding)

**Repository:** https://github.com/hendrycks/test  
**Paper:** https://arxiv.org/abs/2009.03300  
**Dataset:** https://huggingface.co/datasets/cais/mmlu

**Description:**
- 15,908 multiple-choice questions across 57 subjects
- Covers domains: STEM, humanities, social sciences, professional knowledge
- Zero-shot and few-shot evaluation settings
- Expected execution time: 4-8 hours on single GPU

**Key Subjects (57 total):**
```
Abstract Algebra, Anatomy, Astronomy, Auxiliary Nurse, Business Ethics,
Clinical Knowledge, College Biology, College Chemistry, College Computer Science,
College Mathematics, Computer Security, Conceptual Physics, Econometrics,
Elementary Mathematics, Engineering Ethics, Formal Logic, Genetics, Global Facts,
High School Biology, High School Chemistry, High School Computer Science,
High School European History, High School Geography, High School Government Politics,
High School MacroEconomics, High School MicroEconomics, High School Physics,
High School Psychology, High School Statistics, High School US History,
Human Aging, Human Sexuality, International Law, Jurisprudence, Kindergarten,
Logical Fallacies, Machine Learning, Moral Disputes, Moral Scenarios,
Nutrition, Philosophy, Prehistory, Professional Accounting, Professional Law,
Professional Medicine, Professional Psychology, Public Relations, Quantum Mechanics,
Secondary Mathematics, Security Studies, Sociology, US Foreign Policy,
Virology, World Religions
```

**Typical Performance (2026):**
- GPT-4o: 88.7%
- Claude Opus 4.5: 88.2%
- Llama 3.1 405B: 85.2%
- Gemini 2.5: 87.8%

**Implementation:**
```python
import json
from pathlib import Path
from tqdm import tqdm

class MMLUBenchmark:
    """MMLU evaluation implementation."""
    
    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "auxiliary_nurse",
        "business_ethics", "clinical_knowledge", "college_biology",
        # ... all 57 subjects
    ]
    
    def __init__(self, data_dir: Path = Path("data/mmlu")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_questions(self, subject: str, split: str = "test") -> list[dict]:
        """Load questions for a subject."""
        file_path = self.data_dir / split / f"{subject}_{split}.csv"
        questions = []
        
        if not file_path.exists():
            return []
        
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 5:
                    question, a, b, c, d = parts[:5]
                    questions.append({
                        "question": question,
                        "choices": [a, b, c, d],
                        "answer": d,  # Last column is the answer
                    })
        
        return questions
    
    def evaluate_subject(
        self, 
        subject: str, 
        llm_call_fn,
        num_shots: int = 0,
    ) -> dict:
        """Evaluate model on a subject."""
        questions = self.load_questions(subject)
        if not questions:
            return {"subject": subject, "accuracy": 0, "n": 0}
        
        correct = 0
        for q in tqdm(questions, desc=subject, leave=False):
            prompt = self._build_prompt(subject, q, num_shots=num_shots)
            response = llm_call_fn(prompt)
            predicted = self._extract_answer(response)
            
            if predicted == q["answer"]:
                correct += 1
        
        return {
            "subject": subject,
            "accuracy": correct / len(questions),
            "n": len(questions),
            "correct": correct,
        }
    
    def run_evaluation(
        self, 
        llm_call_fn,
        num_shots: int = 0,
        subjects: list[str] = None,
    ) -> dict:
        """Run full MMLU evaluation."""
        eval_subjects = subjects or self.SUBJECTS
        results = []
        
        for subject in tqdm(eval_subjects, desc="MMLU"):
            result = self.evaluate_subject(subject, llm_call_fn, num_shots)
            results.append(result)
        
        total_correct = sum(r.get("correct", 0) for r in results)
        total_questions = sum(r.get("n", 0) for r in results)
        
        return {
            "benchmark": "MMLU",
            "num_shots": num_shots,
            "overall_accuracy": total_correct / total_questions if total_questions > 0 else 0,
            "total_questions": total_questions,
            "results_by_subject": results,
        }

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract A/B/C/D from response."""
        for line in response.split("\n"):
            if line.strip():
                first_char = line.strip()[0].upper()
                if first_char in "ABCD":
                    return first_char
        return ""
```

---

### 2. HELLASWAG (Common Sense Inference)

**Repository:** https://github.com/rowanz/hellaswag  
**Paper:** https://arxiv.org/abs/2101.00297  
**Dataset:** https://huggingface.co/datasets/Rowan/hellaswag

**Description:**
- 10,042 multiple-choice questions
- Tests commonsense understanding of everyday situations
- Four completion options per scenario
- Low computational cost (lightweight benchmark)

**Typical Performance (2026):**
- GPT-4o: 96.3%
- Claude Opus 4.5: 95.8%
- Llama 3.1 405B: 92.4%

---

### 3. ARC (AI2 Reasoning Challenge)

**Repository:** https://github.com/allenai/AI2-REASONING-CHALLENGE-V2  
**Paper:** https://arxiv.org/abs/1911.02727  
**Dataset:** https://huggingface.co/datasets/allenai/ai2_arc

**Description:**
- 7,787 science exam questions (14-18 years old)
- ARC-Easy: 5,197 questions
- ARC-Challenge: 2,590 questions
- Tests reasoning and domain knowledge

**Evaluation Setup:**
```python
class ARCBenchmark:
    """ARC evaluation implementation."""
    
    def __init__(self):
        self.easy_dataset = None
        self.challenge_dataset = None
    
    def load_data(self):
        """Load ARC datasets."""
        from datasets import load_dataset
        
        self.easy_dataset = load_dataset(
            "allenai/ai2_arc", 
            "ARC-Easy",
            split="test"
        )
        self.challenge_dataset = load_dataset(
            "allenai/ai2_arc",
            "ARC-Challenge", 
            split="test"
        )
    
    def evaluate(self, llm_call_fn) -> dict:
        """Evaluate on ARC benchmark."""
        easy_results = self._evaluate_subset(
            self.easy_dataset, 
            llm_call_fn
        )
        challenge_results = self._evaluate_subset(
            self.challenge_dataset,
            llm_call_fn
        )
        
        return {
            "benchmark": "ARC",
            "easy": easy_results,
            "challenge": challenge_results,
            "combined_accuracy": (
                easy_results["correct"] + challenge_results["correct"]
            ) / (easy_results["total"] + challenge_results["total"]),
        }
```

**Typical Performance (2026):**
- Easy: 88-92% for frontier models
- Challenge: 82-86% for frontier models

---

### 4. TRUTHFULQA (Factuality)

**Repository:** https://github.com/sylinrl/TruthfulQA  
**Paper:** https://arxiv.org/abs/2109.07958  
**Dataset:** https://huggingface.co/datasets/truthfulqa/truthfulqa

**Description:**
- 817 questions designed to elicit common misconceptions
- Evaluates if model outputs truthful or hallucinated responses
- Two metrics: MC1 (multiple choice with single answer) and MC2 (multiple choice with multiple correct answers)

**Key Insight:** 
Models must balance factuality with informativeness. High accuracy requires:
- Knowledge of common falsehoods
- Avoiding hallucinations
- Distinguishing opinion from fact

---

### 5. GPQA (Graduate-Level Google-Proof Q&A)

**Repository:** https://huggingface.co/datasets/Idavidrein/gpqa  
**Paper:** https://arxiv.org/abs/2311.12022  
**Dataset:** https://huggingface.co/datasets/Idavidrein/gpqa

**Description:**
- 446 multiple-choice graduate-level questions
- Expert-written questions (physics, biology, chemistry)
- Designed to be difficult even with Google search
- Very challenging for current models

**Typical Performance (2026):**
- GPT-4o: 65-70%
- Claude Opus 4.5: 63-68%
- Most models: <50%

---

### 6. SimpleQA (Factual QA)

**Repository:** https://huggingface.co/datasets/openai/simple_qa  
**Paper:** https://openai.com/research/  
**Dataset:** Released by OpenAI Dec 2024

**Description:**
- 4,000 short-factual QA pairs
- Evaluates core knowledge and factuality
- Open-ended (not multiple choice)
- SimpleQA vs SimpleQA Verified (human-verified subset)

**Evaluation Metric:**
```python
def evaluate_simple_qa(predictions: list[str], references: list[str]) -> dict:
    """Evaluate SimpleQA predictions."""
    
    exact_matches = sum(
        p.lower().strip() == r.lower().strip()
        for p, r in zip(predictions, references)
    )
    
    # Fuzzy matching for alternate phrasings
    fuzzy_matches = 0
    for p, r in zip(predictions, references):
        if fuzz.ratio(p.lower(), r.lower()) > 85:
            fuzzy_matches += 1
    
    return {
        "exact_match_accuracy": exact_matches / len(predictions),
        "fuzzy_match_accuracy": fuzzy_matches / len(predictions),
        "total": len(predictions),
    }
```

---

## Code Generation Benchmarks

### 7. HumanEval (Function Generation)

**Repository:** https://github.com/openai/human-eval  
**Paper:** https://arxiv.org/abs/2107.03374  
**Dataset:** https://huggingface.co/datasets/openai/HumanEval

**Description:**
- 164 hand-crafted Python programming problems
- Tests basic coding ability (functions, algorithms)
- Metric: Pass@k (k=1, 10, 100)
- Nearly saturated benchmark (frontier models: 88-92%)

**Evaluation:**
```python
def evaluate_humaneval(
    code_solutions: list[str],
    test_cases: list[dict],
) -> dict:
    """Evaluate HumanEval solutions."""
    
    results = []
    for solution, tests in zip(code_solutions, test_cases):
        try:
            # Execute solution with test cases
            passed = 0
            for test in tests:
                try:
                    exec(solution + "\n" + test, {})
                    passed += 1
                except:
                    pass
            
            results.append({
                "solution_id": tests[0]["id"],
                "passed": passed,
                "total": len(tests),
                "success": passed == len(tests),
            })
        except SyntaxError:
            results.append({
                "solution_id": tests[0]["id"],
                "success": False,
                "error": "SyntaxError",
            })
    
    pass_at_1 = sum(r["success"] for r in results) / len(results)
    
    return {
        "benchmark": "HumanEval",
        "pass@1": pass_at_1,
        "total_problems": len(results),
    }
```

**Typical Performance (2026):**
- Pass@1: 88-92% for GPT-4o, Claude Opus 4.5
- Pass@10: 95-98%
- Pass@100: 98-99%

---

### 8. SWE-Bench (Software Engineering Benchmark)

**Repository:** https://github.com/swe-bench/SWE-bench  
**Leaderboard:** https://www.swebench.com/  
**Paper:** https://arxiv.org/abs/2310.06770  
**Dataset:** https://huggingface.co/datasets/swe-bench/SWE-bench

**Variants:**
- **SWE-bench Lite:** 300 instances (lite subset)
- **SWE-bench**: 2,294 instances (full)
- **SWE-bench Verified:** 500 human-verified instances
- **SWE-bench Pro:** 500 instances with human-in-the-loop

**Description:**
- Real GitHub issues from popular Python repositories
- Task: Write code to fix bugs or implement features
- Tests: Real test suites must pass
- Complexity: Repository-level context understanding required

**Typical Performance (2026):**
- GPT-4o: 35-40% (resolve_rate)
- Claude Opus 4.5: 32-37%
- Best agentic systems: 45-50%

**Evaluation Metrics:**
```python
class SWEBenchEvaluator:
    """SWE-Bench evaluation framework."""
    
    def evaluate_solution(
        self,
        repo_path: str,
        problem_statement: str,
        solution_diff: str,
        test_command: str,
    ) -> dict:
        """
        Evaluate if solution resolves the issue.
        
        Returns:
            - resolve: True if all tests pass
            - test_pass_rate: % of tests passing
            - error_message: if failed
        """
        
        import subprocess
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy repo
            tmp_repo = shutil.copytree(repo_path, f"{tmp_dir}/repo")
            
            # Apply solution patch
            try:
                subprocess.run(
                    ["git", "apply"],
                    input=solution_diff.encode(),
                    cwd=tmp_repo,
                    check=True,
                )
            except subprocess.CalledProcessError:
                return {"resolve": False, "error": "patch_failed"}
            
            # Run tests
            try:
                result = subprocess.run(
                    test_command,
                    shell=True,
                    cwd=tmp_repo,
                    capture_output=True,
                    timeout=300,
                )
                
                return {
                    "resolve": result.returncode == 0,
                    "stdout": result.stdout.decode(),
                    "stderr": result.stderr.decode(),
                    "return_code": result.returncode,
                }
            except subprocess.TimeoutExpired:
                return {"resolve": False, "error": "timeout"}
```

---

### 9. HumanEval+ and MBPP

**HumanEval+:**
- https://github.com/evalplus/evalplus
- Enhanced HumanEval with more test cases
- Reduces unintended pass-throughs

**MBPP (Mostly Basic Programming Problems):**
- https://huggingface.co/datasets/google-research/mbpp
- 974 problems
- Easier than HumanEval
- Good for assessing basic coding

---

### 10. LiveCodeBench (Evolving Code Benchmark)

**Repository:** https://github.com/LiveCodeBench/LiveCodeBench  
**Leaderboard:** https://livecodebench.github.io/  

**Description:**
- Continuously updated benchmark
- New problems added monthly
- Tests: LeetCode, AtCoder problems
- Prevents benchmark gaming via data contamination

---

## Mathematical Reasoning Benchmarks

### 11. GSM8K (Grade School Math)

**Repository:** https://github.com/openai/grade-school-math  
**Paper:** https://arxiv.org/abs/2110.14168  
**Dataset:** https://huggingface.co/datasets/openai/gsm8k

**Description:**
- 8,500 grade-school math problems (elementary to middle school)
- Chain-of-thought reasoning often improves performance
- Testing: arithmetic, percentages, fractions, geometry

**Evaluation Implementation:**
```python
class GSM8KBenchmark:
    """GSM8K evaluation with answer extraction."""
    
    def evaluate(self, solution: str, reference: str) -> bool:
        """
        Check if solution arrives at correct answer.
        Extracts numerical answer from solution.
        """
        
        import re
        
        # Extract answer from solution (text after "####")
        solution_match = re.search(r"####\s*([\d,.-]+)", solution)
        if not solution_match:
            return False
        
        solution_num = solution_match.group(1)
        solution_num = solution_num.replace(",", "")
        
        # Extract expected answer
        ref_match = re.search(r"####\s*([\d,.-]+)", reference)
        if not ref_match:
            return False
        
        ref_num = ref_match.group(1).replace(",", "")
        
        try:
            return float(solution_num) == float(ref_num)
        except ValueError:
            return solution_num == ref_num
```

**Typical Performance (2026):**
- GPT-4o: 92-95%
- Claude Opus 4.5: 91-94%
- Llama 3.1 405B: 85-88%

---

### 12. MATH (Competition Mathematics)

**Repository:** https://github.com/hendrycks/math  
**Paper:** https://arxiv.org/abs/2103.03874  
**Dataset:** https://huggingface.co/datasets/hendrycks/math

**Description:**
- 12,500 competition math problems
- Source: AMC, AIME, MATHCOUNTS
- Difficulty: High school to competition level
- Tests deep mathematical reasoning

**Typical Performance (2026):**
- GPT-4o: 68-72%
- Claude Opus 4.5: 66-70%
- Frontier models: <60%

---

### 13. AIME and AMC Benchmarks

**Description:**
- AIME: American Invitational Mathematics Examination (harder)
- AMC: American Mathematics Competition (moderate)
- Both are subsets within MATH and other benchmarks

---

## Reasoning and Logic Benchmarks

### 14. Big-Bench (Big Benchmark)

**Repository:** https://github.com/google/BIG-bench  
**Paper:** https://arxiv.org/abs/2206.04615  
**Size:** 204 diverse tasks

**Key Tasks:**
- Question answering, common sense reasoning
- Code generation, algorithm understanding
- Translation, summarization, QA
- Mathematical reasoning, logic puzzles

**Performance Across Tasks:**
```python
class BigBenchEvaluator:
    """Big-Bench evaluation orchestration."""
    
    TASKS = [
        "arithmetic", "bbq", "conceptual_combinations",
        "elementary_math", "english_language_understanding",
        "fact_checking", "gender_bias", "logic_grid_puzzle",
        "metaphor_understanding", "misconceptions",
        "question_answering", "strategy_qa",
        # ... 192 more tasks
    ]
    
    def run_task_suite(self, model_id: str) -> dict:
        """Evaluate model on subset of tasks."""
        results = {}
        
        for task in self.TASKS[:20]:  # Run first 20 for speed
            try:
                result = self._run_task(task, model_id)
                results[task] = result
            except Exception as e:
                results[task] = {"error": str(e)}
        
        return results
```

---

### 15. HellaSwag Alternative: RACE

**Description:**
- Reading comprehension from high school exams
- 100K+ questions
- Tests language understanding, not just knowledge

---

### 16. BoolQ (Boolean Questions)

**Paper:** https://arxiv.org/abs/1905.10044  

**Description:**
- Simple yes/no questions
- Tests reading comprehension
- Easier than MMLU, good for baseline evaluation

---

## Specialized Domain Benchmarks

### 17. Medical Benchmarks

**MedQA:**
- https://github.com/jindi-ai/medical-exam-data
- Medical licensing exams (US, China)
- Multiple choice format
- Tests medical knowledge

**MedMCQA:**
- 193K multiple-choice medical questions
- Indian medical licensing exam
- Diverse question types

**ClimateBench:**
- Climate science questions
- Specialization: Climate/environmental science

---

### 18. Legal Benchmarks

**LegalBench:**
- https://github.com/hazyresearch/legalbench
- 162 legal reasoning tasks
- Tests contract understanding, legal analysis
- Covers: contract review, issue spotting, case prediction

**FinanceQA:**
- Financial document understanding
- SEC filings, earnings calls, financial reports

---

### 19. PubMedQA (Biomedical)

**Dataset:** https://huggingface.co/datasets/pubmed_qa  
**Description:**
- Biomedical research article QA
- Yes/No/Maybe answers
- 1,000 research articles with annotations

---

### 20. Toxic Comments and Bias Benchmarks

**Toxicity:**
- TOXIC_COMMENTS: Wikipedia comment toxicity
- RealToxicityPrompts: Generation of toxic content
- PerspectiveAPI: Multi-dimensional toxicity

**Bias:**
- WinoBias: Gender bias in coreference resolution
- StereoSet: Stereotype evaluation
- BOLD: Bias in open-ended language generation

---

### 21. Instruction Following Benchmarks

**IFEval (Instruction Following Eval):**
- https://huggingface.co/datasets/google/IFEval
- 541 instructions with diverse constraints
- Tests: keyword inclusion, length restrictions, format requirements

**Implementation:**
```python
class IFEvalBenchmark:
    """Instruction Following Evaluation."""
    
    def check_keyword_requirement(self, response: str, keywords: list[str]) -> bool:
        """Check if response contains required keywords."""
        response_lower = response.lower()
        return all(kw.lower() in response_lower for kw in keywords)
    
    def check_word_limit(self, response: str, max_words: int) -> bool:
        """Check if response respects word limit."""
        word_count = len(response.split())
        return word_count <= max_words
    
    def check_format_requirement(self, response: str, format_type: str) -> bool:
        """Check if response matches format requirement."""
        
        if format_type == "json":
            try:
                json.loads(response)
                return True
            except:
                return False
        
        elif format_type == "markdown":
            return any(c in response for c in ["#", "**", "-", "*"])
        
        elif format_type == "list":
            return response.strip().startswith(("-", "*", "1."))
        
        return True
```

---

## Evaluation Frameworks and Tools

### Framework 1: lm-evaluation-harness

**Repository:** https://github.com/EleutherAI/lm-evaluation-harness  
**Status:** Most widely-used open-source evaluation framework  
**Supported Benchmarks:** 100+ benchmarks integrated

**Installation & Usage:**
```bash
# Install
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

# Run MMLU evaluation
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --num_fewshot 5 \
  --output_path results/mmlu_llama2

# Run multiple benchmarks
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,hellaswag,arc,truthfulqa \
  --num_fewshot 0,5 \
  --output_path results/comprehensive_eval
```

**Key Features:**
- Unified interface for 100+ benchmarks
- Support for few-shot learning
- Caching for efficient re-evaluation
- Batch processing
- Custom metric definitions

---

### Framework 2: OpenCompass (Chinese LLM Evaluation)

**Repository:** https://github.com/open-compass/opencompass  
**Documentation:** https://opencompass.org.cn/

**Features:**
- 50+ benchmarks (MMLU, CMMLU, C-Eval, etc.)
- Multilingual support (Chinese, English, others)
- Distributed evaluation
- Leaderboard integration

**Usage:**
```bash
# Install
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .

# Run evaluation
python run.py \
  --models hf_llama2_7b \
  --datasets mmlu cmmlu \
  --work_dir ./results
```

---

### Framework 3: DeepEval

**Repository:** https://github.com/confident-ai/deepeval  
**Documentation:** https://docs.deepeval.com/

**Features:**
- LLM-as-judge evaluation
- Custom metric development
- Evaluation dashboard
- Integration with evaluation best practices

**Implementation:**
```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancy

# Define metric
metric = AnswerRelevancy()

# Run evaluation
test_cases = [
    {
        "input": "What is Python?",
        "actual_output": "Python is a programming language.",
        "expected_output": "Python is a high-level programming language."
    }
]

results = evaluate(test_cases, [metric])
```

---

### Framework 4: PromptFoo

**Repository:** https://github.com/promptfoo/promptfoo  
**Purpose:** Prompt evaluation and comparison  

**Features:**
- Compare different prompts systematically
- A/B testing for LLM outputs
- Custom evaluation criteria
- CSV/JSON data support

---

### Framework 5: HELM (Holistic Evaluation of Language Models)

**Repository:** https://github.com/stanford-crfm/helm  
**Paper:** https://arxiv.org/abs/2211.09110  

**Coverage:**
- 16 scenarios, 100+ tasks
- Multiple metrics (accuracy, robustness, fairness)
- Comprehensive leaderboard
- Open source with HuggingFace integration

---

### Framework 6: LightEval

**Repository:** https://github.com/huggingface/lighteval  
**Purpose:** Lightweight evaluation framework  

**Features:**
- Easy custom metric definition
- Support for HuggingFace models
- Integration with Hugging Face Hub
- Fast evaluation with caching

---

## Evaluation Metrics and Implementations

### Metric 1: Exact Match (EM)

**Use Case:** Factoid QA, multiple choice  
**Formula:** Proportion of predictions exactly matching reference

```python
def exact_match(prediction: str, reference: str) -> float:
    """Compute exact match score."""
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

# Batch evaluation
def batch_exact_match(predictions: list[str], references: list[str]) -> float:
    return sum(
        exact_match(p, r) for p, r in zip(predictions, references)
    ) / len(predictions)
```

---

### Metric 2: BLEU (Bilingual Evaluation Understudy)

**Use Case:** Translation, summarization (deprecated but still used)  
**Note:** Not recommended for modern LLM evaluation

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(prediction: str, reference: str, n: int = 4) -> float:
    """Compute BLEU score."""
    
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    
    # Weights for n-grams
    weights = [1.0 / n] * n
    
    try:
        bleu = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=weights,
            smoothing_function=SmoothingFunction().method1
        )
        return bleu
    except:
        return 0.0

# Usage
bleu_score = compute_bleu(
    prediction="the cat is on the mat",
    reference="the cat is on the mat"
)
```

---

### Metric 3: ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Use Case:** Summarization evaluation  
**Variants:** ROUGE-1, ROUGE-2, ROUGE-L

```python
from rouge import Rouge

def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE scores."""
    
    rouge = Rouge()
    
    scores = rouge.get_scores(prediction, reference)[0]
    
    return {
        "rouge1": scores["rouge1"]["f"],
        "rouge2": scores["rouge2"]["f"],
        "rougeL": scores["rougeL"]["f"],
    }

# Usage
scores = compute_rouge(
    prediction="The quick brown fox",
    reference="A fast brown fox"
)
print(f"ROUGE-1 F1: {scores['rouge1']:.4f}")
```

---

### Metric 4: BERTScore (Semantic Similarity)

**Use Case:** Text generation quality assessment  
**Repository:** https://github.com/Tiiiger/bert_score

```python
from bert_score import score
import torch

def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore for batch."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    P, R, F1 = score(
        predictions,
        references,
        lang="en",
        device=device,
        batch_size=64,
        model_type="bert-base-uncased"
    )
    
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }

# Usage
scores = compute_bertscore(
    ["The quick brown fox"],
    ["A fast brown fox"]
)
```

---

### Metric 5: METEOR (Metric for Evaluation of Translation)

**Use Case:** Translation, paraphrase evaluation

```python
from nltk.translate import meteor_score

def compute_meteor(prediction: str, reference: str) -> float:
    """Compute METEOR score."""
    
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    
    score = meteor_score.single_meteor_score(ref_tokens, pred_tokens)
    return score
```

---

### Metric 6: BLEURT (Learned Evaluation Metric)

**Use Case:** General-purpose text generation quality  
**Repository:** https://github.com/google-research/bleurt

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BLEURTEvaluator:
    """BLEURT metric implementation."""
    
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
        Output: score in range [-1, 1]
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
        
        # Map to [-1, 1]
        score = torch.sigmoid(logits.squeeze())
        return float((score * 2 - 1).item())
    
    def batch_score(
        self, 
        references: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        """Score multiple pairs."""
        return [self.score(r, h) for r, h in zip(references, hypotheses)]
```

---

### Metric 7: F1 Score

**Use Case:** Information extraction, QA with multiple valid answers

```python
def compute_f1(prediction: str, reference: str) -> float:
    """
    Compute F1 score for token overlap.
    Common in SQuAD-style evaluation.
    """
    
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    common = pred_tokens & ref_tokens
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1
```

---

### Metric 8: Token Overlap Metrics

**Use Cases:** Factoid QA, extraction tasks

```python
def token_overlap(prediction: str, reference: str) -> dict:
    """Compute various token overlap metrics."""
    
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    common = pred_tokens & ref_tokens
    union = pred_tokens | ref_tokens
    
    return {
        "jaccard": len(common) / len(union) if union else 0,
        "precision": len(common) / len(pred_tokens) if pred_tokens else 0,
        "recall": len(common) / len(ref_tokens) if ref_tokens else 0,
        "exact_match": prediction.lower() == reference.lower(),
    }
```

---

## LLM-as-Judge Evaluation

### Overview and Best Practices

**Advantages:**
- Scalable without human annotators
- Can evaluate open-ended tasks
- Flexible criteria definition
- Fast iteration

**Limitations:**
- Can have bias toward verbose answers
- May not match human judgment for nuanced tasks
- Reliability varies (McDonald's omega often <0.75)
- Expensive (requires API calls)

**2026 Research Finding:**
LLM-as-judge reliability estimated at 60-80% agreement with human annotators, depending on:
- Judge model capability (GPT-4o >Claude >Llama)
- Rubric clarity
- Temperature settings
- Single-shot vs. multi-shot evaluation

---

### G-Eval Framework

**Paper:** https://arxiv.org/abs/2303.16634  
**Implementation:** Chain-of-thought (CoT) for structured LLM evaluation

```python
class GEvalFramework:
    """G-Eval implementation with chain-of-thought."""
    
    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model
    
    def evaluate(
        self,
        task_description: str,
        input_text: str,
        output_text: str,
        rubric: str,
        max_steps: int = 5,
    ) -> dict:
        """
        Evaluate using G-Eval approach.
        
        Steps:
        1. Generate evaluation steps (CoT)
        2. Execute evaluation based on steps
        3. Generate final score
        """
        
        # Step 1: Generate evaluation logic
        logic_prompt = f"""
        You are evaluating the following:
        Task: {task_description}
        Input: {input_text}
        Output: {output_text}
        
        Rubric: {rubric}
        
        Generate {max_steps} evaluation steps to assess the output.
        Format as numbered list.
        """
        
        steps = self._call_judge(logic_prompt)
        
        # Step 2: Execute evaluation steps
        eval_prompt = f"""
        Based on these evaluation steps:
        {steps}
        
        Provide a detailed evaluation.
        """
        
        evaluation = self._call_judge(eval_prompt)
        
        # Step 3: Generate final score
        score_prompt = f"""
        Based on the evaluation above, provide:
        1. Final score (1-5)
        2. Justification
        3. Key strengths
        4. Key weaknesses
        """
        
        final_result = self._call_judge(score_prompt)
        
        return {
            "steps": steps,
            "evaluation": evaluation,
            "final_result": final_result,
        }
    
    def _call_judge(self, prompt: str) -> str:
        """Call judge model with prompt."""
        # Implementation depends on API (OpenAI, Anthropic, etc.)
        pass
```

---

### Pairwise Comparison (Bradley-Terry)

```python
from scipy.stats import chi2

class PairwiseComparisonEvaluator:
    """Pairwise comparison with Bradley-Terry model."""
    
    def compare_outputs(
        self,
        input_text: str,
        output_a: str,
        output_b: str,
        rubric: str,
    ) -> dict:
        """Compare two outputs directly."""
        
        prompt = f"""
        Given the task and rubric:
        {rubric}
        
        Input: {input_text}
        
        Output A: {output_a}
        Output B: {output_b}
        
        Which output is better? Explain your reasoning.
        Then provide:
        1. Winner (A or B)
        2. Confidence (1-5)
        3. Reasoning
        """
        
        # Call judge...
        result = self._call_judge(prompt)
        
        return self._parse_comparison(result)
    
    def aggregate_comparisons(self, comparisons: list[dict]) -> dict:
        """
        Aggregate pairwise comparisons using Bradley-Terry model.
        Estimates strength of each item.
        """
        
        # Count wins for each model
        wins = {}
        for comp in comparisons:
            winner = comp["winner"]
            wins[winner] = wins.get(winner, 0) + 1
        
        # Compute strengths
        total = len(comparisons)
        strengths = {
            model: count / total
            for model, count in wins.items()
        }
        
        return strengths
```

---

### Rubric-Based Evaluation

**Structure:** Define explicit criteria with score levels

```python
def create_rubric(task: str) -> dict:
    """Create evaluation rubric."""
    
    return {
        "task": task,
        "criteria": [
            {
                "name": "Relevance",
                "description": "How relevant is the response to the query?",
                "levels": {
                    "1": "Not relevant, off-topic",
                    "2": "Mostly irrelevant, minor relevance",
                    "3": "Somewhat relevant",
                    "4": "Highly relevant, minor gaps",
                    "5": "Perfectly relevant, addresses all aspects"
                },
                "weight": 0.3,
            },
            {
                "name": "Accuracy",
                "description": "Are the facts stated correctly?",
                "levels": {
                    "1": "Multiple factual errors",
                    "2": "Several significant errors",
                    "3": "Some minor errors",
                    "4": "Mostly accurate with one minor error",
                    "5": "Completely accurate"
                },
                "weight": 0.4,
            },
            {
                "name": "Completeness",
                "description": "Does the response answer all parts of the query?",
                "levels": {
                    "1": "Addresses <25% of query",
                    "2": "Addresses 25-50%",
                    "3": "Addresses 50-75%",
                    "4": "Addresses 75-95%",
                    "5": "Fully addresses query"
                },
                "weight": 0.2,
            },
            {
                "name": "Clarity",
                "description": "Is the response clear and well-organized?",
                "levels": {
                    "1": "Confusing, hard to follow",
                    "2": "Somewhat unclear",
                    "3": "Reasonably clear",
                    "4": "Clear with minor issues",
                    "5": "Extremely clear and well-organized"
                },
                "weight": 0.1,
            },
        ]
    }

def apply_rubric(
    output: str,
    rubric: dict,
    judge_model: str = "gpt-4o",
) -> dict:
    """Apply rubric to evaluate output."""
    
    scores = {}
    
    for criterion in rubric["criteria"]:
        # Evaluate each criterion with judge
        score = _evaluate_criterion(
            output,
            criterion,
            judge_model
        )
        scores[criterion["name"]] = {
            "score": score,
            "weight": criterion["weight"],
        }
    
    # Compute weighted average
    total_weight = sum(s["weight"] for s in scores.values())
    weighted_score = sum(
        s["score"] * s["weight"]
        for s in scores.values()
    ) / total_weight
    
    return {
        "criterion_scores": scores,
        "overall_score": weighted_score,
        "rubric": rubric["task"],
    }
```

---

## Benchmark Limitations and Gaming

### 1. Benchmark Data Contamination

**Problem:** Training data leakage into benchmark data  
**Impact:** Inflated performance scores not reflective of true capability

**Detection Methods:**

```python
class ContaminationDetector:
    """Detect benchmark data contamination."""
    
    def estimate_contamination(
        self,
        model_outputs: list[str],
        benchmark_data: list[str],
        n_gram_size: int = 8,
    ) -> dict:
        """
        Estimate contamination via n-gram overlap.
        """
        
        from nltk import ngrams
        from collections import Counter
        
        model_ngrams = Counter()
        benchmark_ngrams = Counter()
        
        # Extract n-grams
        for output in model_outputs:
            tokens = output.lower().split()
            model_ngrams.update(
                tuple(tokens[i:i+n_gram_size])
                for i in range(len(tokens) - n_gram_size)
            )
        
        for item in benchmark_data:
            tokens = item.lower().split()
            benchmark_ngrams.update(
                tuple(tokens[i:i+n_gram_size])
                for i in range(len(tokens) - n_gram_size)
            )
        
        # Compute overlap
        overlap = len(set(model_ngrams) & set(benchmark_ngrams))
        benchmark_total = len(benchmark_ngrams)
        
        contamination_rate = overlap / benchmark_total if benchmark_total > 0 else 0
        
        return {
            "contamination_rate": contamination_rate,
            "overlap_count": overlap,
            "benchmark_ngrams": benchmark_total,
            "severity": (
                "HIGH" if contamination_rate > 0.1 else
                "MEDIUM" if contamination_rate > 0.05 else
                "LOW"
            ),
        }
```

**Mitigation Strategies:**
1. **Dynamic benchmarks** (e.g., LiveCodeBench): Updated monthly
2. **Version dating**: Track when data was created vs. model training cutoff
3. **Out-of-distribution tests**: Evaluate on new data
4. **Paraphrased versions**: Create alternative phrasings

---

### 2. Benchmark Saturation

**Problem:** Models approaching ceiling performance (>90%)  
**Examples:**
- MMLU: Frontier models at 88%+
- HumanEval: Frontier models at 90%+
- HellaSwag: 96%+
- BoolQ: >95%

**Impact:** Unable to differentiate between strong models

**Solutions:**
```python
class BenchmarkSaturationAnalyzer:
    """Analyze benchmark saturation."""
    
    def analyze_saturation(
        self,
        model_scores: dict[str, float],
    ) -> dict:
        """
        Check if benchmark is saturated.
        Saturated: top models >85%, median >70%
        """
        
        scores = list(model_scores.values())
        scores.sort(reverse=True)
        
        top_5_avg = sum(scores[:5]) / 5 if len(scores) >= 5 else 0
        median = scores[len(scores) // 2]
        spread = scores[0] - scores[-1]
        
        saturation = {
            "top_5_average": top_5_avg,
            "median": median,
            "spread": spread,
            "is_saturated": top_5_avg > 0.85 and spread < 0.15,
            "recommendation": (
                "Use alternative benchmark" if top_5_avg > 0.85 else
                "Still has signal for 1-2 more years"
            ),
        }
        
        return saturation
```

**Saturated Benchmarks (2026):**
- MMLU (frontier models converging at 87-89%)
- HumanEval (frontier models at 88-92%)
- HellaSwag (96%+)
- BoolQ (95%+)

**Emerging Alternatives:**
- GPQA (still <70% for most models)
- SWE-Bench (35-45% resolution rate)
- SimpleQA (room for improvement)
- LiveCodeBench (evolving benchmark)

---

### 3. Model Gaming and Overfitting

**Risk:** Models optimized specifically for benchmark format

```python
class GamingDetectionFramework:
    """Detect overfitting/gaming indicators."""
    
    def detect_gaming_indicators(
        self,
        benchmark_performance: dict,
        ood_performance: dict,  # Out-of-distribution
        related_benchmarks: dict,
    ) -> dict:
        """
        Flag potential gaming by checking:
        1. Large gap between benchmark and OOD performance
        2. Disproportionate strength on specific benchmark
        """
        
        indicators = {}
        
        # Check benchmark vs OOD gap
        for benchmark, score in benchmark_performance.items():
            ood_score = ood_performance.get(benchmark, 0.5)
            gap = score - ood_score
            
            if gap > 0.2:  # >20% gap suggests gaming
                indicators[f"{benchmark}_ood_gap"] = {
                    "gap": gap,
                    "severity": "HIGH" if gap > 0.3 else "MEDIUM",
                }
        
        # Check for disproportionate strength
        benchmark_scores = list(benchmark_performance.values())
        if max(benchmark_scores) - min(benchmark_scores) > 0.25:
            indicators["inconsistent_performance"] = {
                "spread": max(benchmark_scores) - min(benchmark_scores),
                "note": "Large variance may indicate gaming on specific benchmark"
            }
        
        return indicators
```

---

### 4. Benchmark Suite Design

**Best Practice:** Use complementary benchmarks testing different aspects

```python
def design_benchmark_suite(
    model_type: str = "general",
    budget_hours: int = 24,
) -> dict:
    """
    Recommend benchmark suite based on model type and time budget.
    """
    
    suites = {
        "quick_validation": {
            "benchmarks": ["hellaswag", "arc_easy", "boolq"],
            "total_time_hours": 2,
            "coverage": ["common_sense", "reasoning", "reading_comp"],
        },
        "comprehensive": {
            "benchmarks": [
                "mmlu",
                "gpqa",
                "humaneval",
                "gsm8k",
                "truthfulqa",
                "hellaswag",
                "arc",
            ],
            "total_time_hours": 20,
            "coverage": [
                "knowledge", "code", "math", "reasoning",
                "factuality", "common_sense",
            ],
        },
        "code_focused": {
            "benchmarks": [
                "humaneval",
                "swe_bench",
                "apps",
                "mbpp",
                "livecodebench",
            ],
            "total_time_hours": 24,
            "coverage": ["code_generation", "code_understanding"],
        },
        "safety_focused": {
            "benchmarks": [
                "realtoxicity",
                "bias_benchmarks",
                "jailbreak_tests",
                "truthfulqa",
            ],
            "total_time_hours": 8,
            "coverage": ["toxicity", "bias", "safety", "factuality"],
        },
    }
    
    return suites.get(model_type, suites["comprehensive"])
```

---

## Leaderboards and Rankings

### Major LLM Leaderboards (2026)

#### 1. Open LLM Leaderboard (HuggingFace)
**URL:** https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

**Benchmarks:**
- MMLU
- HellaSwag
- ARC
- TruthfulQA
- GPQA
- GSM8K
- DROP

**Status:** Most popular open-source LLM leaderboard

---

#### 2. LMSYS Arena (Elo Rating)
**URL:** https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard

**Methodology:**
- Crowdsourced pairwise comparisons
- Bradley-Terry model for Elo rating
- Real user queries

**Advantages:**
- Reflects actual user preferences
- Avoids benchmark gaming
- Real-world distribution

---

#### 3. SWE-Bench Leaderboard
**URL:** https://www.swebench.com/

**Metrics:**
- Resolve rate (% of issues fixed)
- Test pass rate
- Speed/cost metrics

**Leaders (2026):**
- Claude Opus 4.5: ~36%
- GPT-4o: ~40%
- Best agentic systems: ~50%

---

#### 4. SimpleQA Leaderboard (OpenAI)
**URL:** https://openai.com/research/ (SimpleQA)

**Evaluation:**
- Exact match on factual questions
- No multiple choice
- No context provided

---

#### 5. Chatbot Arena Leaderboard (LMSYS)

**Advantages over benchmark leaderboards:**
- Tests real-world performance
- Multiple interaction styles
- User satisfaction focused

---

### Interpreting Leaderboards

```python
class LeaderboardInterpreter:
    """Interpret and compare leaderboard results."""
    
    @staticmethod
    def compute_statistical_significance(
        model_a_scores: list[float],
        model_b_scores: list[float],
        confidence_level: float = 0.95,
    ) -> dict:
        """
        Compute if difference is statistically significant.
        """
        
        from scipy import stats
        
        # Paired t-test (if same benchmark)
        t_stat, p_value = stats.ttest_ind(model_a_scores, model_b_scores)
        
        is_significant = p_value < (1 - confidence_level)
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "confidence_level": confidence_level,
            "interpretation": (
                "Significant difference" if is_significant else
                "Difference could be due to chance"
            ),
        }
    
    @staticmethod
    def compute_ranking_confidence(
        scores: dict[str, float],
        std_errors: dict[str, float],
    ) -> dict:
        """
        Compute ranking confidence intervals.
        """
        
        rankings = {}
        
        for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            std_error = std_errors.get(model, 0.02)
            ci_lower = score - 1.96 * std_error
            ci_upper = score + 1.96 * std_error
            
            rankings[model] = {
                "score": score,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_width": ci_upper - ci_lower,
            }
        
        return rankings
```

---

## Benchmark Creation Guidelines

### 1. Motivation and Design

**Questions to Answer:**
1. What capability does this benchmark measure?
2. Is it measured elsewhere? (Avoid duplication)
3. How does it complement existing benchmarks?
4. What is the difficulty level?

```python
def benchmark_motivation_checklist() -> list[str]:
    """Ensure benchmark is well-motivated."""
    
    return [
        "Measures distinct capability not covered by existing benchmarks",
        "Addresses real-world use case or known model weakness",
        "Will not saturate in 1-2 years",
        "Covers diverse examples to avoid gaming",
        "Feasible to evaluate at scale",
        "Clear evaluation methodology",
    ]
```

---

### 2. Dataset Construction

**Best Practices:**

```python
class BenchmarkDatasetBuilder:
    """Construct high-quality benchmark dataset."""
    
    @staticmethod
    def collect_data(
        source: str,  # Human-written, existing dataset, etc.
        num_examples: int = 1000,
        quality_threshold: float = 0.9,
    ) -> list[dict]:
        """
        Collect and validate dataset examples.
        """
        
        examples = []
        
        if source == "human_written":
            # Collect from expert annotators
            examples = BenchmarkDatasetBuilder._collect_from_experts(
                num_examples
            )
        
        elif source == "crowdsourced":
            # Collect from crowdsourcing platform
            examples = BenchmarkDatasetBuilder._collect_crowdsourced(
                num_examples
            )
        
        # Quality control
        validated = [
            ex for ex in examples
            if BenchmarkDatasetBuilder._validate_example(ex) >= quality_threshold
        ]
        
        return validated
    
    @staticmethod
    def _validate_example(example: dict) -> float:
        """
        Validate single example.
        Returns quality score 0-1.
        """
        
        checks = [
            len(example.get("question", "")) > 10,  # Min length
            len(example.get("answer", "")) > 0,  # Has answer
            "ambiguous" not in example.get("notes", "").lower(),
            example.get("difficulty") in ["easy", "medium", "hard"],
        ]
        
        return sum(checks) / len(checks)
```

---

### 3. Test-Set Construction

**Key Principle:** Test set should be held-out until final evaluation

```python
def split_benchmark_dataset(
    dataset: list[dict],
    train_ratio: float = 0.0,  # Usually 0% for benchmarks
    val_ratio: float = 0.1,
    test_ratio: float = 0.9,
) -> tuple[list, list, list]:
    """
    Split dataset for benchmark construction.
    Note: Most benchmarks use test-only, no training set.
    """
    
    import random
    random.shuffle(dataset)
    
    n = len(dataset)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size + val_size]
    test_set = dataset[train_size + val_size:]
    
    return train_set, val_set, test_set
```

---

### 4. Annotation Guidelines

**Example for QA Benchmark:**

```python
def create_annotation_guidelines() -> str:
    """Create annotation guidelines document."""
    
    return """
    # Benchmark Annotation Guidelines
    
    ## Question Quality
    - Question must be clear and unambiguous
    - Should not reveal answer through phrasing
    - Avoid yes/no questions (unless relevant)
    
    ## Answer Quality
    - Answer must be correct and complete
    - Should be concise but comprehensive
    - Alternative acceptable answers marked with [ALT]
    
    ## Difficulty Assessment
    - Easy: Answerable by high school student
    - Medium: Requires college-level knowledge
    - Hard: Requires expert knowledge or reasoning
    
    ## Examples of Bad Questions
    - "What is the answer?" (too vague)
    - "The capital of France is ___" (too easy, reveals answer)
    - "What did X do that was important?" (subjective)
    
    ## Review Process
    1. Each question reviewed by 2+ independent annotators
    2. Disagreements resolved through discussion
    3. Inter-annotator agreement tracked (target: 95%+)
    """
```

---

### 5. Inter-Annotator Agreement

**Metrics:** Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha

```python
from krippendorff import krippendorff_alpha

def compute_inter_annotator_agreement(
    annotations: dict[str, list[str]],  # {example_id: [annotator1, annotator2, ...]}
) -> dict:
    """
    Compute inter-annotator agreement.
    """
    
    # Format for Krippendorff's alpha
    data = []
    for example_id, annotators in annotations.items():
        data.append([
            1 if label == "correct" else 0
            for label in annotators
        ])
    
    agreement = krippendorff_alpha(data)
    
    return {
        "krippendorff_alpha": agreement,
        "interpretation": (
            "Excellent" if agreement > 0.8 else
            "Good" if agreement > 0.67 else
            "Moderate" if agreement > 0.5 else
            "Poor"
        ),
    }
```

---

### 6. Baseline Performance

**Establish baseline before public release:**

```python
class BaselineEstimator:
    """Estimate baseline performance."""
    
    @staticmethod
    def estimate_baselines(
        benchmark: list[dict],
        baseline_models: list[str] = ["random", "tiny", "small", "large"],
    ) -> dict:
        """
        Estimate baseline performance for different model sizes.
        Helps interpret leaderboard results.
        """
        
        baselines = {}
        
        # Random baseline
        if "random" in baseline_models:
            baselines["random"] = 1 / len(benchmark[0].get("options", []))
        
        # Model-based baselines
        for model_size in ["small", "large"]:
            score = evaluate_model(f"gpt2-{model_size}", benchmark)
            baselines[model_size] = score
        
        return baselines
```

---

## Tools and Repositories

### Evaluation Tools Summary

| Tool | Purpose | Link |
|------|---------|------|
| **lm-evaluation-harness** | 100+ benchmarks, unified interface | https://github.com/EleutherAI/lm-evaluation-harness |
| **OpenCompass** | Chinese/multilingual evaluation | https://github.com/open-compass/opencompass |
| **DeepEval** | LLM-as-judge evaluation | https://github.com/confident-ai/deepeval |
| **HELM** | Holistic evaluation framework | https://github.com/stanford-crfm/helm |
| **LightEval** | Lightweight evaluation framework | https://github.com/huggingface/lighteval |
| **PromptFoo** | Prompt comparison and testing | https://github.com/promptfoo/promptfoo |
| **Galileo** | Commercial evaluation platform | https://galileo.ai |
| **Langfuse** | Observability + evaluation | https://langfuse.com |
| **Openlayer** | Governance and observability | https://www.openlayer.com |
| **HuggingFace Leaderboard** | Community leaderboard | https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard |

---

### Benchmark Repositories

| Benchmark | Repository | Paper |
|-----------|-----------|-------|
| **MMLU** | https://github.com/hendrycks/test | https://arxiv.org/abs/2009.03300 |
| **HumanEval** | https://github.com/openai/human-eval | https://arxiv.org/abs/2107.03374 |
| **GSM8K** | https://github.com/openai/grade-school-math | https://arxiv.org/abs/2110.14168 |
| **MATH** | https://github.com/hendrycks/math | https://arxiv.org/abs/2103.03874 |
| **SWE-Bench** | https://github.com/swe-bench/SWE-bench | https://arxiv.org/abs/2310.06770 |
| **LegalBench** | https://github.com/hazyresearch/legalbench | https://arxiv.org/abs/2308.11462 |
| **IFEval** | https://huggingface.co/datasets/google/IFEval | https://arxiv.org/abs/2311.07911 |
| **SimpleQA** | https://huggingface.co/datasets/openai/simple_qa | OpenAI Blog |
| **HellaSwag** | https://github.com/rowanz/hellaswag | https://arxiv.org/abs/2101.00297 |
| **ARC** | https://github.com/allenai/AI2-REASONING-CHALLENGE-V2 | https://arxiv.org/abs/1911.02727 |

---

## References and Further Reading

### Core Benchmark Papers

1. **MMLU** - https://arxiv.org/abs/2009.03300 (Hendrycks et al., 2020)
2. **HELM** - https://arxiv.org/abs/2211.09110 (Liang et al., 2022)
3. **Big-Bench** - https://arxiv.org/abs/2206.04615 (Srivastava et al., 2022)
4. **HellaSwag** - https://arxiv.org/abs/2101.00297 (Zellers et al., 2019)
5. **HumanEval** - https://arxiv.org/abs/2107.03374 (Chen et al., 2021)
6. **SWE-Bench** - https://arxiv.org/abs/2310.06770 (Jimenez et al., 2023)
7. **GSM8K** - https://arxiv.org/abs/2110.14168 (Cobbe et al., 2021)
8. **MATH** - https://arxiv.org/abs/2103.03874 (Hendrycks et al., 2021)
9. **TruthfulQA** - https://arxiv.org/abs/2109.07958 (Lin et al., 2021)
10. **GPQA** - https://arxiv.org/abs/2311.12022 (Rein et al., 2023)

### Evaluation Methodology

11. **BERTScore** - https://arxiv.org/abs/1904.09675 (Zhang et al., 2019)
12. **BLEURT** - https://arxiv.org/abs/2004.04696 (Sellam et al., 2020)
13. **G-Eval** - https://arxiv.org/abs/2303.16634 (Liu et al., 2023)
14. **LLM-as-Judge Survey** - https://arxiv.org/abs/2412.12509 (Schroeder & Wood-Doughty, 2024)
15. **Learning to Judge** - https://arxiv.org/abs/2602.08672 (Siro et al., 2026)

### Contamination and Gaming

16. **Benchmark Contamination Survey** - https://arxiv.org/abs/2406.04244 (Xu et al., 2024)
17. **LastingBench** - https://arxiv.org/abs/2506.21614 (Fang et al., 2025)
18. **Fragility of Contamination Detection** - https://arxiv.org/abs/2510.02386 (2025)

### Safety and Bias Evaluation

19. **Jailbreak Attacks and Defenses** - https://arxiv.org/abs/2402.05668 (Inan et al., 2024)
20. **Bias in LLMs** - https://direct.mit.edu/coli/article/50/3/1097/121961 (Blodgett et al., 2024)

### Domain-Specific Benchmarks

21. **LegalBench** - https://arxiv.org/abs/2308.11462 (Guha et al., 2023)
22. **MedQA** - https://arxiv.org/abs/2104.06417 (Jin et al., 2021)
23. **FinanceQA** - https://huggingface.co/datasets

### Leaderboard Strategies

24. **LMSYS Arena** - https://arxiv.org/abs/2306.05685 (Zheng et al., 2023)
25. **Open LLM Leaderboard** - HuggingFace Spaces

---

## Conclusion

LLM evaluation in 2026 is a mature discipline with:

**Strengths:**
- 30+ standardized benchmarks covering diverse capabilities
- Multiple evaluation frameworks enabling reproducible assessment
- LLM-as-judge enabling scalable evaluation without humans
- Sophisticated leaderboards aggregating results

**Challenges:**
- Benchmark saturation for frontier models
- Data contamination inflating scores
- Gaming and overfitting to specific benchmarks
- Mismatch between benchmark performance and real-world usefulness

**Best Practices:**
1. Use multiple complementary benchmarks
2. Report confidence intervals, not point estimates
3. Evaluate on out-of-distribution data
4. Monitor for data leakage
5. Combine automated metrics with human evaluation
6. Track metrics over time for drift detection

---

**Document Version:** 2.0  
**Last Updated:** April 2026  
**Maintained by:** Shuvam Banerji Seal

