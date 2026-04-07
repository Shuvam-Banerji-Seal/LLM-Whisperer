# LLM Benchmarking and Evaluation Implementation Guide 2026

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Scope:** Practical code examples, setup guides, evaluation orchestration

---

## Quick Start: Running Your First Benchmark

### 1. Install lm-evaluation-harness

```bash
# Clone repository
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness

# Install in development mode
pip install -e .

# Verify installation
lm_eval --version
```

### 2. Run MMLU Benchmark on Llama 2

```bash
# Single GPU - MMLU with 5-shot prompting
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size 32 \
  --output_path ./results/mmlu_llama2

# View results
cat ./results/mmlu_llama2/results.json | python -m json.tool
```

### 3. Run Multiple Benchmarks

```bash
# Comprehensive evaluation suite
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-13b-hf \
  --tasks mmlu,hellaswag,arc,truthfulqa_mc,gsm8k \
  --num_fewshot 0,5 \
  --output_path ./results/llama2_comprehensive \
  --limit 1000  # Limit for faster iteration \
  --write_out
```

---

## Complete Evaluation Pipeline

### End-to-End Benchmark Runner

```python
import json
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation."""
    model_name: str
    benchmarks: List[str]
    num_fewshot: int = 0
    batch_size: int = 32
    output_dir: str = "results"
    num_workers: int = 4
    save_outputs: bool = True
    timeout_seconds: int = 3600

class ComprehensiveBenchmarkRunner:
    """Run multiple benchmarks and aggregate results."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def run_all_benchmarks(self) -> Dict:
        """Run all configured benchmarks."""
        
        logger.info(f"Starting evaluation of {self.config.model_name}")
        logger.info(f"Benchmarks: {', '.join(self.config.benchmarks)}")
        
        start_time = time.time()
        
        for benchmark in self.config.benchmarks:
            logger.info(f"Running {benchmark}...")
            
            try:
                result = self._run_benchmark(benchmark)
                self.results[benchmark] = result
                
                # Save intermediate results
                self._save_results()
                
            except Exception as e:
                logger.error(f"Failed to run {benchmark}: {e}")
                self.results[benchmark] = {"error": str(e)}
        
        end_time = time.time()
        
        # Compute summary
        summary = self._compute_summary()
        summary["total_time_seconds"] = end_time - start_time
        
        logger.info(f"Evaluation complete. Total time: {summary['total_time_seconds']:.1f}s")
        
        return summary
    
    def _run_benchmark(self, benchmark: str) -> Dict:
        """Run single benchmark using lm-evaluation-harness."""
        
        import subprocess
        
        output_file = self.output_dir / f"{benchmark}_output.json"
        
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={self.config.model_name}",
            "--tasks", benchmark,
            "--num_fewshot", str(self.config.num_fewshot),
            "--batch_size", str(self.config.batch_size),
            "--output_path", str(output_file.parent / benchmark),
        ]
        
        if self.config.save_outputs:
            cmd.append("--write_out")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Evaluation failed: {result.stderr}")
        
        # Parse results
        with open(output_file.parent / f"{benchmark}_results.json") as f:
            return json.load(f)
    
    def _save_results(self) -> None:
        """Save current results to file."""
        
        output_file = self.output_dir / f"results_{datetime.now().isoformat()}.json"
        
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics across benchmarks."""
        
        summary = {
            "model": self.config.model_name,
            "timestamp": datetime.now().isoformat(),
            "benchmarks_completed": len(self.results),
            "benchmark_results": {},
        }
        
        for benchmark, result in self.results.items():
            if "error" in result:
                summary["benchmark_results"][benchmark] = {"error": result["error"]}
            else:
                # Extract key metrics
                metrics = {}
                if "results" in result:
                    for key, value in result["results"].items():
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                
                summary["benchmark_results"][benchmark] = metrics
        
        return summary

# Usage
if __name__ == "__main__":
    config = EvaluationConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        benchmarks=["mmlu", "hellaswag", "arc", "gsm8k"],
        num_fewshot=5,
        batch_size=32,
    )
    
    runner = ComprehensiveBenchmarkRunner(config)
    summary = runner.run_all_benchmarks()
    
    print(json.dumps(summary, indent=2))
```

---

## Benchmark-Specific Implementations

### MMLU Evaluation with Custom Prompting

```python
from typing import Callable, Optional
import numpy as np
from tqdm import tqdm

class CustomMMLUEvaluator:
    """MMLU with custom prompt templates."""
    
    # Templates for different shot configurations
    TEMPLATES = {
        "zero_shot": "Answer the following question:\n{question}\n\nChoices:\n{choices}\n\nAnswer:",
        
        "few_shot_cot": "Answer the following questions. Use step-by-step reasoning.\n\n{examples}\n\nQuestion: {question}\n\nChoices:\n{choices}\n\nLet me think step by step:\n",
        
        "few_shot_standard": "{examples}\n\nQuestion: {question}\n\nChoices:\n{choices}\n\nAnswer:",
    }
    
    def __init__(
        self,
        data_path: str,
        llm_fn: Callable,
        template: str = "zero_shot",
    ):
        self.data_path = data_path
        self.llm_fn = llm_fn
        self.template = self.TEMPLATES[template]
        self.results = []
    
    def evaluate(
        self,
        subjects: Optional[List[str]] = None,
        num_shots: int = 0,
    ) -> Dict:
        """Evaluate MMLU with optional few-shot examples."""
        
        all_subjects = subjects or self._get_all_subjects()
        total_correct = 0
        total_questions = 0
        subject_results = {}
        
        for subject in tqdm(all_subjects, desc="MMLU"):
            correct, total = self._evaluate_subject(subject, num_shots)
            
            subject_results[subject] = {
                "accuracy": correct / total if total > 0 else 0,
                "correct": correct,
                "total": total,
            }
            
            total_correct += correct
            total_questions += total
        
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        return {
            "benchmark": "MMLU",
            "overall_accuracy": overall_accuracy,
            "total_questions": total_questions,
            "correct": total_correct,
            "subjects": subject_results,
            "num_shots": num_shots,
        }
    
    def _evaluate_subject(self, subject: str, num_shots: int) -> tuple:
        """Evaluate single subject."""
        
        questions = self._load_questions(subject)
        correct = 0
        
        # Get few-shot examples if needed
        few_shot_examples = ""
        if num_shots > 0:
            few_shot_examples = self._get_few_shot_examples(subject, num_shots)
        
        for q in tqdm(questions, leave=False, desc=subject):
            # Build prompt
            choices_str = "\n".join(
                f"{chr(65+i)}) {choice}"
                for i, choice in enumerate(q["choices"])
            )
            
            prompt = self.template.format(
                question=q["question"],
                choices=choices_str,
                examples=few_shot_examples,
            )
            
            # Get LLM response
            response = self.llm_fn(prompt)
            predicted = self._extract_answer(response)
            
            if predicted == q["correct_choice"]:
                correct += 1
        
        return correct, len(questions)
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract A/B/C/D from response."""
        
        # Look for explicit answer pattern
        import re
        
        # Pattern: "The answer is A" or just "A"
        match = re.search(r"(?:answer is |^)\s*([A-D])", response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
        
        # Just look for first A-D
        for char in response.upper():
            if char in "ABCD":
                return char
        
        return None
    
    def _load_questions(self, subject: str) -> List[Dict]:
        """Load questions for subject."""
        
        import csv
        from pathlib import Path
        
        file_path = Path(self.data_path) / "test" / f"{subject}_test.csv"
        
        if not file_path.exists():
            return []
        
        questions = []
        with open(file_path) as f:
            for row in csv.reader(f):
                if len(row) == 5:
                    question, a, b, c, d = row
                    questions.append({
                        "question": question,
                        "choices": [a, b, c, d],
                        "correct_choice": d,  # Last column
                    })
        
        return questions
    
    def _get_few_shot_examples(self, subject: str, num_shots: int) -> str:
        """Get few-shot examples from dev set."""
        
        # Load from dev set
        import csv
        from pathlib import Path
        
        file_path = Path(self.data_path) / "dev" / f"{subject}_dev.csv"
        examples = []
        
        with open(file_path) as f:
            for i, row in enumerate(csv.reader(f)):
                if i >= num_shots:
                    break
                
                if len(row) == 5:
                    question, a, b, c, d = row
                    choices_str = "\n".join(f"{chr(65+j)}) {choice}" for j, choice in enumerate([a, b, c, d]))
                    
                    examples.append(f"Question: {question}\n{choices_str}\nAnswer: D")
        
        return "\n\n".join(examples)
```

### HumanEval with Custom Test Framework

```python
import subprocess
import tempfile
from pathlib import Path

class HumanEvalFramework:
    """Evaluate code generation with HumanEval test framework."""
    
    def __init__(self, dataset_path: str = "data/humaneval.jsonl"):
        self.dataset_path = dataset_path
        self.results = {}
    
    def evaluate(
        self,
        solutions: Dict[str, str],  # {problem_id: solution_code}
        timeout_seconds: int = 10,
    ) -> Dict:
        """
        Evaluate solutions against HumanEval test cases.
        
        Args:
            solutions: Dict mapping problem IDs to solution code
            timeout_seconds: Timeout for each test execution
        
        Returns:
            Dict with pass@k metrics
        """
        
        import json
        
        with open(self.dataset_path) as f:
            test_cases = {json.loads(line)["task_id"]: json.loads(line) for line in f}
        
        results = {}
        passed = 0
        
        for problem_id, solution in solutions.items():
            test_case = test_cases.get(problem_id)
            if not test_case:
                continue
            
            passed_tests = self._test_solution(
                solution,
                test_case["test"],
                test_case.get("entry_point", "solution"),
                timeout_seconds,
            )
            
            results[problem_id] = {
                "passed": passed_tests,
                "status": "PASSED" if passed_tests else "FAILED",
            }
            
            if passed_tests:
                passed += 1
        
        # Compute pass@k metrics
        total = len(results)
        pass_at_1 = passed / total if total > 0 else 0
        
        return {
            "benchmark": "HumanEval",
            "pass@1": pass_at_1,
            "passed": passed,
            "total": total,
            "results": results,
        }
    
    def _test_solution(
        self,
        code: str,
        test_case: str,
        entry_point: str,
        timeout_seconds: int,
    ) -> bool:
        """Test single solution."""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write code and tests
            code_file = Path(tmp_dir) / "solution.py"
            test_file = Path(tmp_dir) / "test.py"
            
            code_file.write_text(code)
            test_file.write_text(test_case)
            
            # Run tests
            try:
                result = subprocess.run(
                    ["python", str(test_file)],
                    cwd=tmp_dir,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
                
                return result.returncode == 0
            
            except subprocess.TimeoutExpired:
                return False
            except Exception:
                return False

# Usage
def evaluate_code_models():
    """Evaluate multiple code generation models on HumanEval."""
    
    evaluator = HumanEvalFramework()
    
    # Simulated solutions from different models
    solutions = {
        "HumanEval/0": "def solution():\n    return 1",  # Placeholder
        "HumanEval/1": "def solution():\n    return [1, 2, 3]",
    }
    
    results = evaluator.evaluate(solutions)
    print(f"Pass@1: {results['pass@1']:.2%}")
```

---

## LLM-as-Judge Implementation

### Full LLM Judge Pipeline

```python
import json
from typing import Optional
from dataclasses import dataclass
import asyncio

@dataclass
class JudgeResult:
    """Result from LLM judge."""
    score: float
    reasoning: str
    confidence: float

class LLMJudgeEvaluator:
    """Production LLM-as-judge evaluator."""
    
    def __init__(
        self,
        judge_model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,  # Deterministic for judging
    ):
        self.judge_model = judge_model
        self.api_key = api_key
        self.temperature = temperature
        
        # Initialize client (example: OpenAI)
        if judge_model.startswith("gpt"):
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        elif judge_model.startswith("claude"):
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def evaluate_batch(
        self,
        items: list[dict],  # [{input: ..., output: ..., reference: ...}, ...]
        rubric: str,
        num_workers: int = 5,
    ) -> list[JudgeResult]:
        """Evaluate batch of items in parallel."""
        
        # Create evaluation tasks
        tasks = [
            self.evaluate_single(item, rubric)
            for item in items
        ]
        
        # Run in parallel
        results = asyncio.run(self._run_parallel(tasks, num_workers))
        return results
    
    async def _run_parallel(self, tasks, num_workers):
        """Run tasks in parallel with worker limit."""
        
        semaphore = asyncio.Semaphore(num_workers)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[bounded_task(task) for task in tasks])
    
    async def evaluate_single(
        self,
        item: dict,
        rubric: str,
    ) -> JudgeResult:
        """Evaluate single item using LLM judge."""
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(item, rubric)
        
        # Call judge model
        response = await self._call_judge(prompt)
        
        # Parse response
        result = self._parse_judge_response(response)
        
        return result
    
    def _build_evaluation_prompt(self, item: dict, rubric: str) -> str:
        """Build evaluation prompt."""
        
        return f"""You are an expert evaluator. Evaluate the following output according to the rubric.

RUBRIC:
{rubric}

INPUT:
{item.get('input', '')}

OUTPUT TO EVALUATE:
{item.get('output', '')}

REFERENCE (if available):
{item.get('reference', 'N/A')}

Provide:
1. Score (1-5)
2. Reasoning (2-3 sentences)
3. Confidence (1-5)

Format as JSON: {{"score": ..., "reasoning": "...", "confidence": ...}}
"""
    
    async def _call_judge(self, prompt: str) -> str:
        """Call judge model."""
        
        if hasattr(self.client, "chat"):  # OpenAI
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        
        else:  # Anthropic or other
            response = self.client.messages.create(
                model=self.judge_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.content[0].text
    
    def _parse_judge_response(self, response: str) -> JudgeResult:
        """Parse judge response."""
        
        try:
            # Extract JSON
            import json
            import re
            
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return JudgeResult(
                    score=float(data.get("score", 3)),
                    reasoning=data.get("reasoning", ""),
                    confidence=float(data.get("confidence", 3)),
                )
        except:
            pass
        
        # Fallback: extract score from text
        import re
        match = re.search(r"\b([1-5])\b", response)
        score = float(match.group(1)) if match else 3.0
        
        return JudgeResult(
            score=score,
            reasoning=response[:200],
            confidence=3.0,
        )

# Usage
async def evaluate_outputs():
    """Example: Evaluate summarization outputs."""
    
    judge = LLMJudgeEvaluator(judge_model="gpt-4o")
    
    items = [
        {
            "input": "Article about climate change...",
            "output": "Climate change is a global crisis.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity.",
        },
    ]
    
    rubric = """
    - Relevance (1-5): How well does summary address main points?
    - Accuracy (1-5): Are facts stated correctly?
    - Conciseness (1-5): Is summary appropriately brief?
    """
    
    results = judge.evaluate_batch(items, rubric)
    
    for result in results:
        print(f"Score: {result.score}/5")
        print(f"Reasoning: {result.reasoning}")
        print(f"Confidence: {result.confidence}/5")
```

---

## Performance Analysis and Reporting

### Evaluation Report Generator

```python
import json
from datetime import datetime
from pathlib import Path

class EvaluationReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, results: Dict):
        self.results = results
    
    def generate_report(self, output_path: str = "eval_report.md") -> str:
        """Generate markdown report."""
        
        report = []
        report.append("# LLM Evaluation Report")
        report.append(f"Generated: {datetime.now().isoformat()}\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.extend(self._executive_summary())
        
        # Detailed Results
        report.append("## Detailed Results\n")
        report.extend(self._detailed_results())
        
        # Comparative Analysis
        report.append("## Comparative Analysis\n")
        report.extend(self._comparative_analysis())
        
        # Recommendations
        report.append("## Recommendations\n")
        report.extend(self._recommendations())
        
        # Save report
        report_text = "\n".join(report)
        Path(output_path).write_text(report_text)
        
        return report_text
    
    def _executive_summary(self) -> list:
        """Generate executive summary."""
        
        summary = []
        
        # Average scores
        if "benchmark_results" in self.results:
            benchmarks = self.results["benchmark_results"]
            
            summary.append("### Performance Summary\n")
            summary.append("| Benchmark | Score | Status |")
            summary.append("|-----------|-------|--------|")
            
            for benchmark, result in benchmarks.items():
                if "error" in result:
                    status = "FAILED"
                    score = "N/A"
                else:
                    # Extract main metric
                    if "accuracy" in result:
                        score = f"{result['accuracy']:.2%}"
                    elif "pass@1" in result:
                        score = f"{result['pass@1']:.2%}"
                    else:
                        score = "N/A"
                    
                    status = "OK"
                
                summary.append(f"| {benchmark} | {score} | {status} |")
        
        return summary
    
    def _detailed_results(self) -> list:
        """Generate detailed results section."""
        
        results = []
        
        for benchmark, data in self.results.get("benchmark_results", {}).items():
            results.append(f"### {benchmark.upper()}\n")
            
            if "error" in data:
                results.append(f"Error: {data['error']}\n")
            else:
                results.append("```json")
                results.append(json.dumps(data, indent=2))
                results.append("```\n")
        
        return results
    
    def _comparative_analysis(self) -> list:
        """Generate comparative analysis."""
        
        analysis = []
        
        # Identify strongest/weakest benchmarks
        benchmarks = self.results.get("benchmark_results", {})
        
        scores = {}
        for benchmark, data in benchmarks.items():
            if "error" not in data:
                if "accuracy" in data:
                    scores[benchmark] = data["accuracy"]
                elif "pass@1" in data:
                    scores[benchmark] = data["pass@1"]
        
        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            worst = min(scores.items(), key=lambda x: x[1])
            
            analysis.append(f"**Strongest benchmark:** {best[0]} ({best[1]:.2%})\n")
            analysis.append(f"**Weakest benchmark:** {worst[0]} ({worst[1]:.2%})\n")
            analysis.append(f"**Performance spread:** {best[1] - worst[1]:.2%}\n")
        
        return analysis
    
    def _recommendations(self) -> list:
        """Generate recommendations."""
        
        recommendations = []
        
        benchmarks = self.results.get("benchmark_results", {})
        
        # Identify weak areas
        weak_benchmarks = []
        for benchmark, data in benchmarks.items():
            if "error" not in data:
                if "accuracy" in data and data["accuracy"] < 0.6:
                    weak_benchmarks.append(benchmark)
                elif "pass@1" in data and data["pass@1"] < 0.5:
                    weak_benchmarks.append(benchmark)
        
        if weak_benchmarks:
            recommendations.append(f"Focus on improving: {', '.join(weak_benchmarks)}\n")
        
        return recommendations

# Usage
if __name__ == "__main__":
    results = {
        "model": "meta-llama/Llama-2-7b-hf",
        "benchmark_results": {
            "mmlu": {"accuracy": 0.46},
            "hellaswag": {"accuracy": 0.78},
            "arc": {"accuracy": 0.52},
        }
    }
    
    generator = EvaluationReportGenerator(results)
    report = generator.generate_report("eval_report.md")
    print(report)
```

---

## Debugging and Troubleshooting

### Common Issues and Solutions

```python
class EvaluationTroubleshooter:
    """Debug common evaluation issues."""
    
    @staticmethod
    def check_data_integrity(dataset_path: str) -> Dict:
        """Validate benchmark dataset."""
        
        issues = []
        
        # Check file exists
        from pathlib import Path
        if not Path(dataset_path).exists():
            issues.append(f"Dataset not found: {dataset_path}")
            return {"valid": False, "issues": issues}
        
        # Check format
        import json
        try:
            with open(dataset_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON: {e}")
            return {"valid": False, "issues": issues}
        
        # Check structure
        if not isinstance(data, list):
            issues.append("Dataset should be a list of examples")
        
        # Sample validation
        if data:
            first = data[0]
            required_fields = ["input", "output", "reference"]
            missing = [f for f in required_fields if f not in first]
            if missing:
                issues.append(f"Missing fields: {missing}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "num_examples": len(data) if isinstance(data, list) else 0,
        }
    
    @staticmethod
    def diagnose_llm_failures(
        failed_examples: list,
        error_type: str = "timeout",
    ) -> Dict:
        """Diagnose LLM call failures."""
        
        diagnosis = {}
        
        if error_type == "timeout":
            diagnosis["likely_causes"] = [
                "Model is too large for available GPU",
                "Network connectivity issues",
                "API rate limiting",
            ]
            diagnosis["solutions"] = [
                "Use smaller model variant",
                "Increase timeout duration",
                "Add retry logic with exponential backoff",
            ]
        
        elif error_type == "oom":  # Out of memory
            diagnosis["likely_causes"] = [
                "Batch size too large",
                "Model doesn't fit in GPU memory",
                "Context length too long",
            ]
            diagnosis["solutions"] = [
                "Reduce batch size",
                "Use quantization (4-bit or 8-bit)",
                "Reduce max sequence length",
            ]
        
        return diagnosis
```

---

## Best Practices Checklist

```python
class EvaluationBestPractices:
    """Validate evaluation follows best practices."""
    
    @staticmethod
    def validate_evaluation_setup(config: Dict) -> Dict:
        """Check if evaluation setup follows best practices."""
        
        checks = {
            "reproducibility": [],
            "robustness": [],
            "efficiency": [],
        }
        
        # Reproducibility
        if "random_seed" in config:
            checks["reproducibility"].append("✓ Random seed set")
        else:
            checks["reproducibility"].append("✗ No random seed")
        
        # Robustness
        if "num_fewshot" in config and config["num_fewshot"] > 0:
            checks["robustness"].append("✓ Few-shot evaluation included")
        
        if "num_fewshot_range" in config:
            checks["robustness"].append("✓ Multiple few-shot values tested")
        
        # Efficiency
        if "batch_size" in config and config["batch_size"] > 1:
            checks["efficiency"].append("✓ Batching enabled")
        
        if "num_workers" in config and config["num_workers"] > 1:
            checks["efficiency"].append("✓ Parallelization enabled")
        
        return checks
```

---

## Performance Optimization

### GPU Memory Optimization

```python
class GPUOptimization:
    """Optimize GPU memory usage during evaluation."""
    
    @staticmethod
    def estimate_memory_requirement(
        model_size_billions: float,
        batch_size: int,
        sequence_length: int,
    ) -> Dict:
        """Estimate GPU memory needed."""
        
        # Rough estimates (varies by architecture)
        model_memory = model_size_billions * 2  # GB (rough estimate)
        
        # Activation memory per sample
        activation_memory_per_sample = (sequence_length * 12 * 2) / 1024 / 1024 / 1024
        
        total_activation = activation_memory_per_sample * batch_size
        
        total_memory = model_memory + total_activation
        
        return {
            "model_memory_gb": model_memory,
            "activation_memory_gb": total_activation,
            "total_memory_gb": total_memory,
            "recommended_gpu": (
                "RTX 4090 (24GB)" if total_memory < 20 else
                "A100 (40GB) or larger"
            ),
        }
    
    @staticmethod
    def suggest_optimization(total_memory_gb: float) -> list:
        """Suggest optimizations if memory is tight."""
        
        suggestions = []
        
        if total_memory_gb > 20:
            suggestions.append("Enable 8-bit quantization (bitsandbytes)")
            suggestions.append("Reduce batch size")
            suggestions.append("Reduce sequence length")
        
        if total_memory_gb > 30:
            suggestions.append("Use 4-bit quantization")
            suggestions.append("Enable gradient checkpointing")
        
        return suggestions
```

---

## Conclusion

This implementation guide provides:
- Production-ready code for benchmark evaluation
- LLM-as-judge pipeline with parallelization
- Report generation and analysis
- Troubleshooting and optimization utilities

For more examples and updates, visit:
- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/open-compass/opencompass
- https://github.com/confident-ai/deepeval

