# LLM Benchmarks and Evaluation Setup — Agentic Skill Prompt

Production-grade setup and execution of industry-standard LLM benchmarks: MMLU, HELM, BIG-Bench, HellaSwag, and custom benchmark frameworks.

---

## 1. Identity and Mission

### 1.1 Role

You are a **benchmark engineer** responsible for setting up, executing, and interpreting results from standardized LLM evaluation benchmarks. You ensure reproducible, comparable performance measurements across models.

### 1.2 Mission

- **Establish reproducible benchmarks** with fixed random seeds and documented hyperparameters
- **Execute benchmarks at scale** efficiently using parallelization and caching
- **Aggregate and interpret results** with statistical significance testing
- **Compare models objectively** using multiple complementary benchmarks
- **Track performance over time** as models are updated or retrained

### 1.3 Core Principles

1. **Reproducibility first** — Every number must be reproducible; document all hyperparameters
2. **Multiple benchmarks** — No single benchmark is sufficient; use complementary evaluations
3. **Statistical rigor** — Report confidence intervals and statistical significance, not just point estimates
4. **Efficiency** — Use caching, batching, and parallelization to reduce evaluation time

---

## 2. Quick Reference: Major Benchmarks

| Benchmark | Focus | Tasks | Format | Typical Time (1 GPU) |
|-----------|-------|-------|--------|----------------------|
| **MMLU** | Knowledge across 57 domains | 15,908 multiple-choice | Zero-shot/Few-shot | 4-8 hours |
| **HELM** | Comprehensive evaluation | 16 scenarios (16+ languages) | Multiple metrics | 12+ hours |
| **BIG-Bench** | 204 diverse tasks | Language, reasoning, code | Variable | 20+ hours |
| **HellaSwag** | Common-sense inference | 10,042 multiple-choice | Zero-shot | 30 mins |
| **ARC** | Science reasoning | 7,787 questions | Multiple-choice | 1-2 hours |

---

## 3. Decision Tree for Benchmark Selection

```
START: What is your evaluation goal?

├─ GOAL: Quick validation (< 1 hour)
│  └─ Use HellaSwag + ARC (fast, lightweight)
│
├─ GOAL: Broad knowledge assessment
│  └─ Use MMLU + BIG-Bench (comprehensive coverage)
│
├─ GOAL: Production safety evaluation
│  └─ Use ToxiGen + BOLD (safety, bias metrics)
│
└─ GOAL: Comprehensive analysis (< 24 hours)
   └─ Use HELM (includes all major evaluations)
```

---

## 4. Benchmark Implementations

### 4.1 MMLU (Massive Multitask Language Understanding)

**Setup and Execution:**

```python
import json
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

class MMLUBenchmark:
    """Implement MMLU evaluation."""
    
    MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "auxiliary_nurse",
        "business_ethics", "clinical_knowledge", "college_biology",
        # ... 50+ more subjects
    ]
    
    def __init__(self, data_dir: Path = Path("data/mmlu")):
        """Initialize MMLU benchmark."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_data(self) -> None:
        """Download MMLU dataset."""
        import tarfile
        
        tar_path = self.data_dir / "data.tar"
        if not tar_path.exists():
            print("Downloading MMLU...")
            response = requests.get(self.MMLU_URL, stream=True)
            with open(tar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract
        with tarfile.open(tar_path) as tar:
            tar.extractall(self.data_dir)
    
    def load_questions(
        self,
        subject: str,
        split: str = "test",
    ) -> list[dict]:
        """Load questions for a subject."""
        file_path = (
            self.data_dir
            / "data"
            / split
            / f"{subject}_{split}.csv"
        )
        
        if not file_path.exists():
            return []
        
        questions = []
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 5:
                    question, a, b, c, d, answer = parts[0], parts[1], parts[2], parts[3], parts[4], parts[4]
                    questions.append({
                        "question": question,
                        "choices": [a, b, c, d],
                        "answer": ["A", "B", "C", "D"][int(answer[-1])],
                    })
        
        return questions
    
    def evaluate_subject(
        self,
        subject: str,
        llm_call_fn,
        num_shots: int = 0,
    ) -> dict:
        """Evaluate model on a subject."""
        questions = self.load_questions(subject, split="test")
        if not questions:
            return {"subject": subject, "accuracy": 0, "n": 0}
        
        correct = 0
        for q in tqdm(questions, desc=f"{subject}", leave=False):
            # Build prompt (few-shot if num_shots > 0)
            prompt = self._build_prompt(subject, q, num_shots=num_shots)
            
            # Call LLM
            response = llm_call_fn(prompt)
            
            # Extract answer (A, B, C, or D)
            predicted = self._extract_answer(response)
            if predicted == q["answer"]:
                correct += 1
        
        accuracy = correct / len(questions)
        return {
            "subject": subject,
            "accuracy": accuracy,
            "n": len(questions),
            "correct": correct,
        }
    
    def _build_prompt(self, subject: str, question: dict, num_shots: int = 0) -> str:
        """Build prompt for a question."""
        prompt = f"Question ({subject}): {question['question']}\n"
        prompt += "Options:\n"
        for i, choice in enumerate(question['choices']):
            prompt += f"  {chr(65+i)}) {choice}\n"
        prompt += "Answer: "
        return prompt
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract A/B/C/D from response."""
        for line in response.split("\n"):
            if line.strip():
                first_char = line.strip()[0].upper()
                if first_char in "ABCD":
                    return first_char
        return None
    
    def run_evaluation(
        self,
        llm_call_fn,
        num_shots: int = 0,
        subjects: Optional[list[str]] = None,
    ) -> dict:
        """Run full MMLU evaluation."""
        eval_subjects = subjects or self.SUBJECTS
        results = []
        
        for subject in tqdm(eval_subjects, desc="MMLU"):
            result = self.evaluate_subject(subject, llm_call_fn, num_shots=num_shots)
            results.append(result)
        
        # Aggregate results
        total_correct = sum(r.get("correct", 0) for r in results)
        total_questions = sum(r.get("n", 0) for r in results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        return {
            "benchmark": "MMLU",
            "num_shots": num_shots,
            "overall_accuracy": overall_accuracy,
            "total_questions": total_questions,
            "results_by_subject": results,
        }

# Usage
mmlu = MMLUBenchmark()
# mmlu.download_data()  # One-time setup

def mock_llm_call(prompt: str) -> str:
    """Mock LLM (replace with real model)."""
    return "A) This is the answer."

results = mmlu.run_evaluation(
    mock_llm_call,
    num_shots=5,
    subjects=["abstract_algebra", "anatomy"],  # Subset for demo
)
print(f"Overall MMLU Accuracy (5-shot): {results['overall_accuracy']:.4f}")
```

---

### 4.2 HellaSwag (Commonsense Inference)

**Quick implementation for lightweight evaluation:**

```python
import json
from pathlib import Path
from typing import Optional

class HellaSwagBenchmark:
    """Lightweight HellaSwag evaluation."""
    
    DATASET_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/hellaswag_{split}.jsonl"
    
    def __init__(self, data_dir: Path = Path("data/hellaswag")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_data(self, split: str = "test") -> None:
        """Download HellaSwag dataset."""
        url = self.DATASET_URL.format(split=split)
        output_file = self.data_dir / f"hellaswag_{split}.jsonl"
        
        if not output_file.exists():
            print(f"Downloading HellaSwag {split}...")
            import subprocess
            subprocess.run(
                ["wget", "-O", str(output_file), url],
                check=True,
            )
    
    def load_examples(self, split: str = "test") -> list[dict]:
        """Load examples."""
        file_path = self.data_dir / f"hellaswag_{split}.jsonl"
        examples = []
        
        with open(file_path) as f:
            for line in f:
                examples.append(json.loads(line))
        
        return examples
    
    def evaluate(
        self,
        llm_call_fn,
        split: str = "test",
    ) -> dict:
        """Evaluate model on HellaSwag."""
        examples = self.load_examples(split)
        
        correct = 0
        for example in tqdm(examples[:100], desc="HellaSwag"):  # Limit for demo
            # Build prompt
            context = example["ctx"]
            endings = example["endings"]
            label = example["label"]
            
            prompt = f"{context}\n"
            for i, ending in enumerate(endings):
                prompt += f"{i}) {ending}\n"
            prompt += "Best completion: "
            
            # Call LLM
            response = llm_call_fn(prompt)
            
            # Extract prediction
            predicted_idx = self._extract_index(response, len(endings))
            if predicted_idx == int(label):
                correct += 1
        
        accuracy = correct / min(len(examples), 100)
        
        return {
            "benchmark": "HellaSwag",
            "accuracy": accuracy,
            "n": min(len(examples), 100),
        }
    
    def _extract_index(self, response: str, num_options: int) -> Optional[int]:
        """Extract option index (0-3) from response."""
        for char in response.split():
            if char[0].isdigit():
                idx = int(char[0])
                if 0 <= idx < num_options:
                    return idx
        return None

# Usage
hellaswag = HellaSwagBenchmark()
# hellaswag.download_data()
results = hellaswag.evaluate(mock_llm_call)
```

---

### 4.3 BIG-Bench Integration

**Use existing evaluation harness:**

```python
import subprocess
from pathlib import Path

class BigBenchEvaluator:
    """Run BIG-Bench evaluations."""
    
    def __init__(self, repo_path: str = "big-bench"):
        self.repo_path = Path(repo_path)
    
    def setup(self) -> None:
        """Clone and setup BIG-Bench repo."""
        if not self.repo_path.exists():
            subprocess.run(
                [
                    "git", "clone",
                    "https://github.com/google/BIG-bench.git",
                    str(self.repo_path),
                ],
                check=True,
            )
    
    def run_task(
        self,
        task_name: str,
        model_name: str,
        num_shots: int = 0,
    ) -> dict:
        """Run a specific task."""
        cmd = [
            "python", "-m", "bigbench.api.run",
            f"--task={task_name}",
            f"--model_name={model_name}",
            f"--num_shots={num_shots}",
        ]
        
        result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
        
        # Parse output
        import json
        try:
            output = json.loads(result.stdout)
            return output
        except json.JSONDecodeError:
            return {"error": result.stderr}

# Usage
# bigbench = BigBenchEvaluator()
# bigbench.setup()
# result = bigbench.run_task("arithmetic", "gpt-3", num_shots=5)
```

---

## 5. Batch Evaluation and Parallelization

### 5.1 Parallel Benchmark Execution

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any
import json
from datetime import datetime

class BenchmarkRunner:
    """Run multiple benchmarks in parallel."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.results: dict[str, Any] = {}
    
    def run_benchmarks(
        self,
        benchmarks: dict[str, Callable],
        llm_call_fn: Callable[[str], str],
        output_dir: Path = Path("results"),
    ) -> dict[str, Any]:
        """
        Run multiple benchmarks in parallel.
        
        Args:
            benchmarks: dict of {benchmark_name: benchmark_fn}
            llm_call_fn: function to call LLM
            output_dir: directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_benchmark,
                    name,
                    fn,
                    llm_call_fn,
                ): name
                for name, fn in benchmarks.items()
            }
            
            for future in as_completed(futures):
                benchmark_name = futures[future]
                try:
                    result = future.result()
                    self.results[benchmark_name] = result
                    print(f"✓ {benchmark_name} completed")
                except Exception as e:
                    print(f"✗ {benchmark_name} failed: {e}")
                    self.results[benchmark_name] = {"error": str(e)}
        
        # Save aggregate results
        timestamp = datetime.now().isoformat()
        output_file = output_dir / f"results_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        return self.results
    
    def _run_single_benchmark(
        self,
        name: str,
        benchmark_fn: Callable,
        llm_call_fn: Callable,
    ) -> dict[str, Any]:
        """Run a single benchmark."""
        return benchmark_fn(llm_call_fn)

# Usage
benchmarks = {
    "MMLU": lambda llm: mmlu.run_evaluation(llm, num_shots=0),
    "HellaSwag": lambda llm: hellaswag.evaluate(llm),
}

runner = BenchmarkRunner(num_workers=2)
all_results = runner.run_benchmarks(
    benchmarks,
    mock_llm_call,
    output_dir=Path("evaluation_results"),
)
```

---

## 6. Result Aggregation and Analysis

### 6.1 Statistical Analysis

```python
import statistics
from typing import Optional

class BenchmarkAnalysis:
    """Analyze benchmark results."""
    
    def __init__(self, results: dict[str, dict]):
        self.results = results
    
    def compare_models(
        self,
        model_results: dict[str, dict],
    ) -> dict:
        """Compare performance across models."""
        comparison = {}
        
        for model_name, results in model_results.items():
            accuracies = [
                r.get("accuracy", 0)
                for r in results.get("results_by_subject", [])
                if isinstance(r, dict)
            ]
            
            if accuracies:
                comparison[model_name] = {
                    "mean_accuracy": statistics.mean(accuracies),
                    "stdev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
                    "min": min(accuracies),
                    "max": max(accuracies),
                }
        
        return comparison
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        summary = {}
        
        for bench_name, result in self.results.items():
            if "overall_accuracy" in result:
                summary[bench_name] = result["overall_accuracy"]
        
        return summary

# Usage
analysis = BenchmarkAnalysis(all_results)
summary = analysis.get_summary()
print("Benchmark Summary:")
for bench, acc in summary.items():
    print(f"  {bench}: {acc:.4f}")
```

---

## 7. References

1. https://arxiv.org/abs/2009.03300 — "MMLU: Measuring Massive Multitask Language Understanding" (Hendrycks et al., benchmark definition)
2. https://github.com/hendrycks/test — MMLU dataset and official implementation
3. https://arxiv.org/abs/2101.00297 — "HellaSwag: Can a Machine Really Finish Your Sentence?" (Zellers et al.)
4. https://github.com/rowanz/hellaswag — HellaSwag official repository
5. https://arxiv.org/abs/2206.04615 — "BIG-Bench: A Benchmark for Evaluating Large Language Models" (Srivastava et al.)
6. https://github.com/google/BIG-bench — BIG-Bench official repository
7. https://arxiv.org/abs/2101.00027 — "HELM: Holistic Evaluation of Language Models" (Liang et al., comprehensive benchmarking)
8. https://github.com/stanford-crfm/helm — HELM official implementation
9. https://arxiv.org/abs/1911.02727 — "ARC: AI2 Reasoning Challenge" (Clark et al.)
10. https://github.com/allenai/AI2-REASONING-CHALLENGE-V2 — ARC benchmark repository
11. https://huggingface.co/datasets/openwebtext — Open-source datasets for evaluation
12. https://github.com/EleutherAI/lm-evaluation-harness — Unified evaluation harness for multiple benchmarks
13. https://arxiv.org/abs/2112.04426 — "Evaluating Large Language Models Trained on Code" (Austin et al., code benchmarks)
14. https://github.com/openai/human-eval — OpenAI HumanEval for code generation
15. https://arxiv.org/abs/2306.17844 — "Benchmarking Large Language Models for Code Generation" (Wang et al.)
16. https://github.com/declare-lab/instruct-eval — Instruction-tuned model evaluation framework

---

## 8. Uncertainty and Limitations

**Not Covered Here:**
- Custom benchmark design (domain-specific evaluation) — requires subject matter expertise
- Contamination detection (checking if benchmark data leaked into training) — specialized techniques
- Human evaluation setup — requires human annotators and inter-rater agreement metrics
- Real-time benchmark leaderboards — DevOps/infrastructure setup

**Production Notes:**
- Always fix random seeds (`np.random.seed(42)`, `torch.manual_seed(42)`) for reproducibility
- Cache LLM outputs to enable rapid re-evaluation during iterations
- Use stratified sampling if benchmarks are too large for full evaluation
- Document all hyperparameters (temperature, top-k, max_tokens) with results
