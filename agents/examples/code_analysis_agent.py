"""
AGNO Code Analysis Agent Example

A code analysis agent that understands, analyzes, and refactors code.
Demonstrates code execution, analysis, and recommendation patterns.

Author: Shuvam Banerji Seal
Source: https://docs.agno.com/agents/tools
Source: https://docs.agno.com/cookbook
"""

import logging
from typing import Optional, List, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CodeAnalysisAgent:
    """
    An agent that analyzes and improves code.

    This agent demonstrates:
    - Code understanding and analysis
    - Identifying issues and improvements
    - Suggesting refactoring
    - Generating tests
    - Documentation generation

    Example:
        >>> agent = CodeAnalysisAgent()
        >>> analysis = agent.analyze(code_snippet)
        >>> improvements = agent.suggest_improvements()
        >>> tests = agent.generate_tests()

    Production AGNO Code:
    ```python
    from agno.agent import Agent
    from agno.models.anthropic import Claude
    from agno.tools.coding import CodingTools

    code_analyst = Agent(
        name="Code Analyzer",
        model=Claude(id="claude-3-5-sonnet-20241022"),
        tools=[CodingTools()],
        instructions="Analyze code and provide improvement suggestions.",
    )

    code_analyst.print_response(f"Analyze this code:\\n{code}", stream=True)
    ```
    """

    def __init__(
        self, name: str = "CodeAnalysisAgent", model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize the code analysis agent.

        Args:
            name: Agent identifier
            model: Model to use
        """
        self.name = name
        self.model = model
        self.analyzed_code: Dict[str, str] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.improvement_suggestions: Dict[str, List[str]] = {}
        self.test_cases: Dict[str, List[str]] = {}

        logger.info(f"Initialized {name}")

    def analyze_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Analyze source code.

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            Analysis results

        AGNO Analysis Pattern:
        Agent examines code for:
        - Syntax and structure
        - Performance issues
        - Security concerns
        - Best practice violations
        - Complexity metrics
        """
        code_id = f"code_{len(self.analyzed_code) + 1}"
        self.analyzed_code[code_id] = code

        logger.info(f"Analyzing code ({language}): {code_id}")

        analysis = {
            "code_id": code_id,
            "language": language,
            "lines_of_code": len(code.split("\n")),
            "complexity": self._assess_complexity(code),
            "issues": self._identify_issues(code),
            "patterns": self._identify_patterns(code),
            "security_score": 0.85,
            "maintainability_score": 0.75,
        }

        self.analysis_results[code_id] = analysis

        return analysis

    def _assess_complexity(self, code: str) -> Dict[str, Any]:
        """Assess code complexity."""
        lines = code.split("\n")

        # Simple complexity heuristics
        has_loops = any("for " in line or "while " in line for line in lines)
        has_nested = code.count("    ") > code.count("  ")  # Nesting indicator
        has_recursion = "def " in code and any(
            line.strip().startswith("self.") or line.strip().startswith("recursive_")
            for line in lines
        )

        complexity_score = 1.0
        if has_loops:
            complexity_score += 1.5
        if has_nested:
            complexity_score += 2.0
        if has_recursion:
            complexity_score += 2.5

        return {
            "cyclomatic_complexity": min(complexity_score, 10),
            "has_loops": has_loops,
            "has_nested_logic": has_nested,
            "has_recursion": has_recursion,
            "estimated_maintenance_cost": "medium" if complexity_score < 5 else "high",
        }

    def _identify_issues(self, code: str) -> List[Dict[str, Any]]:
        """Identify code issues."""
        issues = []

        # Check for common issues
        if "TODO" in code or "FIXME" in code:
            issues.append(
                {
                    "severity": "medium",
                    "type": "incomplete_code",
                    "message": "Code contains TODO or FIXME comments",
                }
            )

        if "pass" in code and len(code) < 200:
            issues.append(
                {
                    "severity": "low",
                    "type": "stub_implementation",
                    "message": "Code appears to be a stub or incomplete implementation",
                }
            )

        if code.count("except") > 0 and "except Exception" in code:
            issues.append(
                {
                    "severity": "high",
                    "type": "broad_exception_handling",
                    "message": "Catching broad Exception type - consider being more specific",
                }
            )

        if "global " in code:
            issues.append(
                {
                    "severity": "medium",
                    "type": "global_variables",
                    "message": "Code uses global variables - consider refactoring",
                }
            )

        return issues

    def _identify_patterns(self, code: str) -> List[str]:
        """Identify design patterns and code patterns."""
        patterns = []

        if "class " in code and "__init__" in code:
            patterns.append("Object-Oriented")

        if "def " in code and "->" in code:
            patterns.append("Type Hints")

        if "async " in code or "await " in code:
            patterns.append("Async/Await")

        if "with " in code:
            patterns.append("Context Managers")

        if "list(" in code or "dict(" in code or "@" in code:
            patterns.append("Functional/Decorators")

        if "unittest" in code or "pytest" in code or "test_" in code:
            patterns.append("Testing")

        return patterns

    def suggest_improvements(self, code_id: str) -> List[Dict[str, Any]]:
        """
        Suggest code improvements.

        Args:
            code_id: ID of code to improve

        Returns:
            List of improvement suggestions

        AGNO Refactoring Pattern:
        Agent provides actionable suggestions for:
        - Performance optimization
        - Readability improvements
        - Pattern implementation
        - Test coverage
        """
        if code_id not in self.analysis_results:
            logger.warning(f"Code {code_id} not analyzed")
            return []

        analysis = self.analysis_results[code_id]
        suggestions = []

        # Based on analysis results
        if analysis["complexity"]["cyclomatic_complexity"] > 5:
            suggestions.append(
                {
                    "priority": "high",
                    "category": "refactoring",
                    "suggestion": "Extract complex logic into separate functions",
                    "benefit": "Improved readability and testability",
                }
            )

        if analysis["issues"]:
            suggestions.append(
                {
                    "priority": "high",
                    "category": "bug_fix",
                    "suggestion": "Address identified issues",
                    "benefit": "Better code quality and reliability",
                }
            )

        if "Type Hints" not in analysis["patterns"]:
            suggestions.append(
                {
                    "priority": "medium",
                    "category": "best_practice",
                    "suggestion": "Add type hints to function signatures",
                    "benefit": "Better IDE support and error catching",
                }
            )

        if analysis["maintainability_score"] < 0.8:
            suggestions.append(
                {
                    "priority": "medium",
                    "category": "documentation",
                    "suggestion": "Add docstrings and comments",
                    "benefit": "Improved code documentation",
                }
            )

        if "Testing" not in analysis["patterns"]:
            suggestions.append(
                {
                    "priority": "high",
                    "category": "testing",
                    "suggestion": "Add unit tests",
                    "benefit": "Better code coverage and confidence",
                }
            )

        self.improvement_suggestions[code_id] = [s["suggestion"] for s in suggestions]

        return suggestions

    def generate_refactored_code(self, code_id: str) -> str:
        """
        Generate refactored version of code.

        Args:
            code_id: ID of code to refactor

        Returns:
            Refactored code
        """
        if code_id not in self.analyzed_code:
            return ""

        original_code = self.analyzed_code[code_id]

        # Simulate refactored code
        refactored = f"""# Refactored version of {code_id}
# Improvements applied:
# - Added type hints
# - Improved variable naming
# - Extracted helper functions
# - Added docstrings

{original_code}

# Additional improvements:
# - Consider breaking into smaller functions
# - Add comprehensive error handling
# - Implement logging for debugging
"""

        return refactored

    def generate_tests(self, code_id: str) -> List[str]:
        """
        Generate test cases for code.

        Args:
            code_id: ID of code to test

        Returns:
            List of test cases
        """
        if code_id not in self.analyzed_code:
            return []

        test_cases = [
            "Test with valid input parameters",
            "Test with edge case values (None, empty, zero)",
            "Test error handling for invalid inputs",
            "Test performance with large datasets",
            "Test integration with dependencies",
        ]

        self.test_cases[code_id] = test_cases

        return test_cases

    def get_analysis_summary(self, code_id: str) -> Dict[str, Any]:
        """Get a summary of code analysis."""
        if code_id not in self.analysis_results:
            return {}

        analysis = self.analysis_results[code_id]

        return {
            "code_id": code_id,
            "language": analysis["language"],
            "lines_of_code": analysis["lines_of_code"],
            "complexity_score": round(
                analysis["complexity"]["cyclomatic_complexity"], 2
            ),
            "issues_found": len(analysis["issues"]),
            "patterns_detected": analysis["patterns"],
            "security_score": analysis["security_score"],
            "maintainability_score": analysis["maintainability_score"],
            "suggestions": len(self.improvement_suggestions.get(code_id, [])),
        }


def main():
    """
    Example usage of the Code Analysis Agent.

    Reference Documentation:
    - https://docs.agno.com/agents/tools
    - https://docs.agno.com/cookbook
    """
    print("\n=== AGNO Code Analysis Agent Example ===\n")

    # Create agent
    agent = CodeAnalysisAgent(name="CodeAnalyzer")

    # Example code to analyze
    sample_code = '''def calculate_fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


class DataProcessor:
    """Process and analyze data."""
    
    def __init__(self):
        self.data = []
    
    def load_data(self, source):
        # TODO: Implement data loading
        pass
    
    def process(self):
        try:
            for item in self.data:
                # Process item
                pass
        except Exception:
            # Broad exception - not ideal
            pass
'''

    # Analyze code
    print("1. Analyzing code...")
    analysis = agent.analyze_code(sample_code, language="python")

    print(f"Analysis Summary:")
    import json

    summary = agent.get_analysis_summary(analysis["code_id"])
    print(json.dumps(summary, indent=2))

    # Get improvement suggestions
    print("\n2. Improvement Suggestions:")
    suggestions = agent.suggest_improvements(analysis["code_id"])

    for i, sugg in enumerate(suggestions, 1):
        print(f"\n   {i}. [{sugg['priority'].upper()}] {sugg['suggestion']}")
        print(f"      Benefit: {sugg['benefit']}")

    # Generate tests
    print("\n3. Suggested Test Cases:")
    tests = agent.generate_tests(analysis["code_id"])

    for i, test in enumerate(tests, 1):
        print(f"   {i}. {test}")

    # Show refactored version
    print("\n4. Refactored Code Available:")
    refactored = agent.generate_refactored_code(analysis["code_id"])
    print(f"   {len(refactored)} characters of refactored code generated")

    # Show identified issues
    print("\n5. Issues Found:")
    if analysis["issues"]:
        for issue in analysis["issues"]:
            print(
                f"   [{issue['severity'].upper()}] {issue['type']}: {issue['message']}"
            )
    else:
        print("   No major issues found")

    print("\n6. Design Patterns Detected:")
    for pattern in analysis["patterns"]:
        print(f"   - {pattern}")


if __name__ == "__main__":
    main()
