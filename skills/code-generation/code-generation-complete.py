"""
Code Generation with LLMs - Complete Implementation
====================================================

Demonstrates:
- Loading code generation models
- Various prompting strategies for code
- Integration with IDEs
- Code evaluation and testing
- Fine-tuning for domain-specific code
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import subprocess
import tempfile
import os


# ============================================================================
# Code Generation Models & Configuration
# ============================================================================


class CodeModelType(Enum):
    """Available code generation models"""

    CODEX = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    CLAUDE = "claude-3-opus"
    DEEPSEEK = "deepseek-coder-33b"
    WIZARD = "WizardCoder-Python-34B"
    LLAMA = "Llama-2-7b-chat"


@dataclass
class CodeGenerationConfig:
    """Configuration for code generation"""

    model: CodeModelType
    temperature: float = 0.2  # Lower for code accuracy
    max_tokens: int = 1024
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    include_explanations: bool = True
    include_tests: bool = True
    language: str = "python"


# ============================================================================
# Code Generation Prompts
# ============================================================================


class CodePromptTemplates:
    """Templates for code generation prompts"""

    @staticmethod
    def function_generation_prompt(
        description: str,
        function_signature: Optional[str] = None,
        examples: Optional[List[str]] = None,
    ) -> str:
        """Generate prompt for function implementation"""

        prompt = f"""Write a {CodePromptTemplates.get_language()} function based on this description:

Description: {description}
"""

        if function_signature:
            prompt += f"\nFunction signature:\n{function_signature}\n"

        if examples:
            prompt += "\nExamples of usage:\n"
            for example in examples:
                prompt += f"  {example}\n"

        prompt += "\nImplementation:\n"
        return prompt

    @staticmethod
    def class_generation_prompt(
        description: str, class_name: str, methods: Optional[List[str]] = None
    ) -> str:
        """Generate prompt for class implementation"""

        prompt = f"""Create a Python class with this specification:

Class Name: {class_name}
Description: {description}
"""

        if methods:
            prompt += "\nRequired Methods:\n"
            for method in methods:
                prompt += f"  - {method}\n"

        prompt += "\nImplementation:\n"
        return prompt

    @staticmethod
    def bug_fix_prompt(
        buggy_code: str, error_message: str, description: str = ""
    ) -> str:
        """Generate prompt for bug fixing"""

        prompt = f"""Fix the bug in this code:

Buggy Code:
```
{buggy_code}
```

Error:
{error_message}
"""

        if description:
            prompt += f"\nContext: {description}\n"

        prompt += "\nFixed Code:\n"
        return prompt

    @staticmethod
    def code_review_prompt(code: str, focus_areas: Optional[List[str]] = None) -> str:
        """Generate prompt for code review"""

        prompt = f"""Review this code and provide constructive feedback:

Code:
```
{code}
```
"""

        if focus_areas:
            prompt += "\nFocus areas:\n"
            for area in focus_areas:
                prompt += f"  - {area}\n"

        prompt += "\nReview:\n"
        return prompt

    @staticmethod
    def test_generation_prompt(code: str, test_framework: str = "pytest") -> str:
        """Generate prompt for test case creation"""

        prompt = f"""Write {test_framework} test cases for this code:

Code:
```
{code}
```

Test cases:\n"""
        return prompt

    @staticmethod
    def documentation_prompt(code: str, doc_style: str = "docstring") -> str:
        """Generate prompt for code documentation"""

        prompt = f"""Add {doc_style} documentation to this code:

Code:
```
{code}
```

Documented Code:\n"""
        return prompt

    @staticmethod
    def refactoring_prompt(code: str, goals: Optional[List[str]] = None) -> str:
        """Generate prompt for code refactoring"""

        prompt = f"""Refactor this code to improve it:

Original Code:
```
{code}
```
"""

        if goals:
            prompt += "\nRefactoring Goals:\n"
            for goal in goals:
                prompt += f"  - {goal}\n"

        prompt += "\nRefactored Code:\n"
        return prompt

    @staticmethod
    def get_language() -> str:
        return "Python"


# ============================================================================
# Code Evaluation & Testing
# ============================================================================


@dataclass
class CodeEvaluationResult:
    """Result of code evaluation"""

    is_valid: bool
    is_executable: bool
    test_results: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    performance_score: float  # 0-1


class CodeEvaluator:
    """Evaluate generated code"""

    @staticmethod
    def check_syntax(code: str, language: str = "python") -> Tuple[bool, Optional[str]]:
        """Check if code has valid syntax"""
        try:
            if language == "python":
                compile(code, "<string>", "exec")
                return True, None
            # Add other language checks as needed
            return True, None
        except SyntaxError as e:
            return False, str(e)

    @staticmethod
    def execute_code(code: str, timeout: int = 5) -> Tuple[bool, str, str]:
        """Execute generated code safely"""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                temp_file = f.name

            # Execute with timeout
            result = subprocess.run(
                ["python", temp_file], capture_output=True, text=True, timeout=timeout
            )

            os.unlink(temp_file)

            if result.returncode == 0:
                return True, result.stdout, result.stderr
            else:
                return False, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", "Execution timeout exceeded"
        except Exception as e:
            return False, "", str(e)

    @staticmethod
    def evaluate(code: str) -> CodeEvaluationResult:
        """Comprehensive code evaluation"""
        errors = []
        warnings = []
        test_results = {}

        # Check syntax
        valid, error = CodeEvaluator.check_syntax(code)
        if not valid:
            errors.append(f"Syntax error: {error}")

        # Try execution
        executable, stdout, stderr = CodeEvaluator.execute_code(code)
        if not executable and not errors:
            errors.append(f"Execution error: {stderr}")

        # Calculate score
        score = 1.0
        if errors:
            score -= 0.5
        if warnings:
            score -= 0.2

        return CodeEvaluationResult(
            is_valid=valid,
            is_executable=executable,
            test_results=test_results,
            errors=errors,
            warnings=warnings,
            performance_score=max(0, score),
        )


# ============================================================================
# IDE Integration
# ============================================================================


class IDEIntegration:
    """Integration with IDEs"""

    @staticmethod
    def vscode_extension_template() -> Dict:
        """Template for VSCode extension"""
        return {
            "name": "llm-code-copilot",
            "displayName": "LLM Code Copilot",
            "version": "0.1.0",
            "publisher": "llm-research",
            "engines": {"vscode": "^1.70.0"},
            "activationEvents": ["onLanguage:python", "onLanguage:javascript"],
            "contributes": {
                "commands": [
                    {
                        "command": "llm-copilot.generateFunction",
                        "title": "Generate Function",
                    },
                    {"command": "llm-copilot.fixCode", "title": "Fix Code"},
                    {"command": "llm-copilot.generateTests", "title": "Generate Tests"},
                ],
                "keybindings": [
                    {"command": "llm-copilot.generateFunction", "key": "ctrl+alt+g"}
                ],
            },
        }

    @staticmethod
    def jetbrains_plugin_template() -> Dict:
        """Template for JetBrains IDE plugin"""
        return {
            "name": "LLM Code Assistant",
            "id": "com.llm.codeassistant",
            "version": "0.1.0",
            "vendor": {"name": "LLM Research", "email": "info@llmresearch.com"},
            "description": "Code generation and assistance with LLMs",
            "change-notes": "Initial release",
            "idea-version": "IC-2023.1",
        }


# ============================================================================
# Code Generation Orchestrator
# ============================================================================


@dataclass
class GeneratedCode:
    """Result of code generation"""

    code: str
    explanation: str
    test_code: Optional[str]
    evaluation: CodeEvaluationResult
    model_used: str
    prompt_used: str


class CodeGenerationOrchestrator:
    """Orchestrate code generation workflow"""

    def __init__(self, config: CodeGenerationConfig):
        self.config = config
        self.templates = CodePromptTemplates()
        self.evaluator = CodeEvaluator()

    def generate_function(
        self,
        description: str,
        function_signature: Optional[str] = None,
        examples: Optional[List[str]] = None,
        evaluate: bool = True,
    ) -> GeneratedCode:
        """Generate a function"""

        prompt = self.templates.function_generation_prompt(
            description, function_signature, examples
        )

        # This would call the actual LLM
        # For demonstration, return a structured response
        code = self._call_llm(prompt)

        # Generate tests if requested
        test_code = None
        if self.config.include_tests:
            test_prompt = self.templates.test_generation_prompt(code)
            test_code = self._call_llm(test_prompt)

        # Evaluate
        evaluation = self.evaluator.evaluate(code) if evaluate else None

        return GeneratedCode(
            code=code,
            explanation=self._extract_explanation(code),
            test_code=test_code,
            evaluation=evaluation,
            model_used=self.config.model.value,
            prompt_used=prompt,
        )

    def fix_code(self, buggy_code: str, error_message: str) -> GeneratedCode:
        """Fix buggy code"""

        prompt = self.templates.bug_fix_prompt(buggy_code, error_message)
        fixed_code = self._call_llm(prompt)

        return GeneratedCode(
            code=fixed_code,
            explanation="Bug fix applied",
            test_code=None,
            evaluation=self.evaluator.evaluate(fixed_code),
            model_used=self.config.model.value,
            prompt_used=prompt,
        )

    def review_code(self, code: str, focus_areas: Optional[List[str]] = None) -> Dict:
        """Review code and get feedback"""

        prompt = self.templates.code_review_prompt(code, focus_areas)
        review = self._call_llm(prompt)

        return {
            "code": code,
            "review": review,
            "evaluation": self.evaluator.evaluate(code),
        }

    def generate_tests(self, code: str, framework: str = "pytest") -> str:
        """Generate test cases"""

        prompt = self.templates.test_generation_prompt(code, framework)
        return self._call_llm(prompt)

    def add_documentation(self, code: str, style: str = "docstring") -> str:
        """Add documentation to code"""

        prompt = self.templates.documentation_prompt(code, style)
        return self._call_llm(prompt)

    def refactor_code(
        self, code: str, goals: Optional[List[str]] = None
    ) -> GeneratedCode:
        """Refactor code"""

        prompt = self.templates.refactoring_prompt(code, goals)
        refactored = self._call_llm(prompt)

        return GeneratedCode(
            code=refactored,
            explanation="Code refactored",
            test_code=None,
            evaluation=self.evaluator.evaluate(refactored),
            model_used=self.config.model.value,
            prompt_used=prompt,
        )

    @staticmethod
    def _call_llm(prompt: str) -> str:
        """Call the LLM (placeholder)"""
        # This would integrate with actual LLM API
        return "# Generated code would appear here\n"

    @staticmethod
    def _extract_explanation(code: str) -> str:
        """Extract explanation from generated code"""
        # Extract docstring or comments
        lines = code.split("\n")
        explanation = ""
        for line in lines:
            if line.strip().startswith("#"):
                explanation += line.strip()[1:].strip() + " "
        return explanation.strip()


# ============================================================================
# Example Usage
# ============================================================================


def example_code_generation():
    """Example code generation workflow"""

    config = CodeGenerationConfig(
        model=CodeModelType.GPT4, temperature=0.2, include_tests=True
    )

    orchestrator = CodeGenerationOrchestrator(config)

    # Example 1: Generate a function
    print("=" * 60)
    print("GENERATING FUNCTION")
    print("=" * 60)

    description = (
        "Implement a function that finds the longest common subsequence of two strings"
    )
    result = orchestrator.generate_function(
        description=description,
        function_signature="def longest_common_subsequence(s1: str, s2: str) -> str:",
        examples=[
            'longest_common_subsequence("abc", "ac") -> "ac"',
            'longest_common_subsequence("xyz", "abc") -> ""',
        ],
    )

    print(f"Generated Code:\n{result.code}")
    if result.evaluation:
        print(f"Valid: {result.evaluation.is_valid}")
        print(f"Executable: {result.evaluation.is_executable}")
        print(f"Score: {result.evaluation.performance_score}")

    # Example 2: Generate tests
    print("\n" + "=" * 60)
    print("GENERATING TESTS")
    print("=" * 60)

    test_code = orchestrator.generate_tests(result.code)
    print(f"Generated Tests:\n{test_code}")

    # Example 3: Add documentation
    print("\n" + "=" * 60)
    print("ADDING DOCUMENTATION")
    print("=" * 60)

    documented = orchestrator.add_documentation(result.code)
    print(f"Documented Code:\n{documented}")


if __name__ == "__main__":
    example_code_generation()
