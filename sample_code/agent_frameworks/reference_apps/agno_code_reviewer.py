"""
AGNO Code Reviewer Agent - Reference Application

A specialized agent for code analysis and review that demonstrates:
- Code parsing and analysis
- Security vulnerability detection
- Performance issue identification
- Code quality metrics
- Refactoring suggestions
- Automated code review workflow

References:
- Building Production-Ready AI Agents: https://medium.com/data-science-collective/building-production-ready-ai-agents-with-agno-a-comprehensive-engineering-guide-22db32413fdd
- Code Review Best Practices: https://example.com/code-review-practices

Author: Shuvam Banerji Seal
"""

# Requirements:
# pip install agno>=1.0.0
# pip install openai>=1.0.0
# pip install ast
# pip install re

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


# ============================================================================
# Issue and Metric Models
# ============================================================================


class IssueSeverity(Enum):
    """Severity levels for code issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(Enum):
    """Categories of code issues."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    BEST_PRACTICES = "best_practices"
    TESTING = "testing"


@dataclass
class CodeIssue:
    """Represents a code issue found during review."""

    id: str
    category: IssueCategory
    severity: IssueSeverity
    line_number: int
    message: str
    code_snippet: str
    suggestion: str
    priority: int = 0


@dataclass
class CodeQualityMetrics:
    """Code quality metrics."""

    lines_of_code: int
    complexity_score: float  # 0.0 - 10.0
    maintainability_index: float  # 0.0 - 100.0
    test_coverage_ratio: float  # 0.0 - 1.0
    documentation_ratio: float  # 0.0 - 1.0
    cyclomatic_complexity: int


@dataclass
class CodeReviewReport:
    """Complete code review report."""

    file_path: str
    issues: List[CodeIssue] = field(default_factory=list)
    metrics: Optional[CodeQualityMetrics] = None
    summary: str = ""
    overall_score: float = 0.0
    recommendation: str = ""


# ============================================================================
# Code Analysis Tools
# ============================================================================


class CodeAnalyzer:
    """Analyzes Python code for issues and quality metrics."""

    SECURITY_PATTERNS = {
        r"eval\s*\(": ("eval() usage", IssueSeverity.CRITICAL),
        r"exec\s*\(": ("exec() usage", IssueSeverity.CRITICAL),
        r"__import__\s*\(": ("__import__() usage", IssueSeverity.HIGH),
        r"subprocess\.call\s*\(": (
            "subprocess.call() without shell=False",
            IssueSeverity.HIGH,
        ),
        r"open\s*\([^)]*'w'": (
            "File opened without proper error handling",
            IssueSeverity.MEDIUM,
        ),
    }

    PERFORMANCE_PATTERNS = {
        r"for\s+\w+\s+in\s+range\(len\(": (
            "Use enumerate instead of range(len())",
            IssueSeverity.LOW,
        ),
        r"\+=\s*str\(": ("String concatenation in loop", IssueSeverity.MEDIUM),
        r"if\s+\w+\s+in\s+\[\s*.*\s*\]": (
            "Use set instead of list for membership",
            IssueSeverity.LOW,
        ),
    }

    BEST_PRACTICES = {
        r"except\s*:": ("Bare except clause", IssueSeverity.HIGH),
        r"except\s+Exception\s*:": (
            "Too broad exception handling",
            IssueSeverity.MEDIUM,
        ),
        r"import\s+\*": ("Wildcard imports", IssueSeverity.MEDIUM),
        r"TODO|FIXME|HACK": ("Code contains TODO/FIXME/HACK", IssueSeverity.INFO),
    }

    @staticmethod
    def analyze_code(code: str, file_path: str) -> CodeReviewReport:
        """
        Analyze Python code and generate review report.

        Args:
            code: The code to analyze
            file_path: Path to the code file

        Returns:
            CodeReviewReport with findings
        """
        report = CodeReviewReport(file_path=file_path)
        lines = code.split("\n")
        issues = []
        issue_id = 0

        # Analyze each line
        for line_num, line in enumerate(lines, 1):
            # Check security patterns
            for pattern, (message, severity) in CodeAnalyzer.SECURITY_PATTERNS.items():
                if re.search(pattern, line):
                    issue_id += 1
                    issue = CodeIssue(
                        id=f"SEC-{issue_id}",
                        category=IssueCategory.SECURITY,
                        severity=severity,
                        line_number=line_num,
                        message=message,
                        code_snippet=line.strip(),
                        suggestion=f"Review {message} and ensure it's safe",
                        priority=10 if severity == IssueSeverity.CRITICAL else 5,
                    )
                    issues.append(issue)

            # Check performance patterns
            for pattern, (
                message,
                severity,
            ) in CodeAnalyzer.PERFORMANCE_PATTERNS.items():
                if re.search(pattern, line):
                    issue_id += 1
                    issue = CodeIssue(
                        id=f"PERF-{issue_id}",
                        category=IssueCategory.PERFORMANCE,
                        severity=severity,
                        line_number=line_num,
                        message=message,
                        code_snippet=line.strip(),
                        suggestion=f"Consider: {message}",
                        priority=3,
                    )
                    issues.append(issue)

            # Check best practices
            for pattern, (message, severity) in CodeAnalyzer.BEST_PRACTICES.items():
                if re.search(pattern, line):
                    issue_id += 1
                    issue = CodeIssue(
                        id=f"BP-{issue_id}",
                        category=IssueCategory.BEST_PRACTICES,
                        severity=severity,
                        line_number=line_num,
                        message=message,
                        code_snippet=line.strip(),
                        suggestion=message,
                        priority=1,
                    )
                    issues.append(issue)

        # Check code length and comment ratio
        non_empty_lines = [l for l in lines if l.strip()]
        comment_lines = len([l for l in lines if l.strip().startswith("#")])
        doc_lines = len([l for l in lines if '"""' in l or "'''" in l])

        # Calculate metrics
        metrics = CodeQualityMetrics(
            lines_of_code=len(non_empty_lines),
            complexity_score=min(10.0, len(issues) / 5),
            maintainability_index=max(0, 100 - (len(issues) * 5)),
            test_coverage_ratio=0.0,  # Would be determined by test suite
            documentation_ratio=min(
                1.0, (comment_lines + doc_lines) / max(1, len(non_empty_lines))
            ),
            cyclomatic_complexity=len(issues) // 3,
        )

        report.issues = issues
        report.metrics = metrics

        # Calculate overall score (0-100)
        critical_count = len(
            [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        )
        high_count = len([i for i in issues if i.severity == IssueSeverity.HIGH])
        medium_count = len([i for i in issues if i.severity == IssueSeverity.MEDIUM])

        penalty = (critical_count * 20) + (high_count * 10) + (medium_count * 5)
        report.overall_score = max(0, 100 - penalty)

        # Generate summary and recommendation
        report.summary = CodeAnalyzer._generate_summary(issues, metrics)
        report.recommendation = CodeAnalyzer._generate_recommendation(
            report.overall_score
        )

        return report

    @staticmethod
    def _generate_summary(issues: List[CodeIssue], metrics: CodeQualityMetrics) -> str:
        """Generate summary text."""
        critical = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        high = len([i for i in issues if i.severity == IssueSeverity.HIGH])

        summary = f"Found {len(issues)} issues: "
        if critical > 0:
            summary += f"{critical} critical, "
        if high > 0:
            summary += f"{high} high severity, "
        summary += f"LOC: {metrics.lines_of_code}, "
        summary += f"Maintainability: {metrics.maintainability_index:.1f}/100"

        return summary

    @staticmethod
    def _generate_recommendation(score: float) -> str:
        """Generate recommendation based on score."""
        if score >= 90:
            return "✅ Code is ready for review/merge"
        elif score >= 70:
            return "⚠️ Address high-severity issues before merging"
        elif score >= 50:
            return "🔴 Significant improvements needed"
        else:
            return "❌ Code requires major refactoring"


# ============================================================================
# Code Reviewer Agent
# ============================================================================


class AGNOCodeReviewerAgent:
    """
    AI code reviewer agent that performs automated code analysis and review.

    Demonstrates:
    - Pattern-based code analysis
    - Security vulnerability detection
    - Performance optimization suggestions
    - Automated report generation
    """

    def __init__(self):
        """Initialize the code reviewer agent."""
        self.reviewed_files: List[CodeReviewReport] = []

    def review_code(self, code: str, file_path: str) -> CodeReviewReport:
        """
        Review code and generate detailed report.

        Args:
            code: The code to review
            file_path: Path to the code file

        Returns:
            CodeReviewReport
        """
        print(f"\n📝 Reviewing: {file_path}")
        print("=" * 70)

        report = CodeAnalyzer.analyze_code(code, file_path)
        self.reviewed_files.append(report)

        return report

    def display_report(self, report: CodeReviewReport) -> None:
        """Display a formatted review report."""
        print(f"\n📋 Code Review Report: {report.file_path}")
        print("=" * 70)

        print(f"\n🎯 Overall Score: {report.overall_score:.1f}/100")
        print(f"📊 {report.summary}")
        print(f"💡 {report.recommendation}")

        if report.metrics:
            print(f"\n📈 Metrics:")
            print(f"  Lines of Code: {report.metrics.lines_of_code}")
            print(f"  Complexity Score: {report.metrics.complexity_score:.2f}/10.0")
            print(
                f"  Maintainability Index: {report.metrics.maintainability_index:.1f}/100"
            )
            print(f"  Documentation Ratio: {report.metrics.documentation_ratio:.1%}")

        if report.issues:
            print(f"\n🔍 Issues Found: {len(report.issues)}")

            # Group by severity
            by_severity = {}
            for issue in report.issues:
                if issue.severity not in by_severity:
                    by_severity[issue.severity] = []
                by_severity[issue.severity].append(issue)

            for severity in [
                IssueSeverity.CRITICAL,
                IssueSeverity.HIGH,
                IssueSeverity.MEDIUM,
                IssueSeverity.LOW,
                IssueSeverity.INFO,
            ]:
                if severity in by_severity:
                    print(
                        f"\n  🔴 {severity.value.upper()} ({len(by_severity[severity])}):"
                    )
                    for issue in by_severity[severity][:3]:  # Show first 3
                        print(f"    Line {issue.line_number}: {issue.message}")
                        print(f"      Code: {issue.code_snippet}")
                        print(f"      Suggestion: {issue.suggestion}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AGNO Code Reviewer Agent - Reference Application")
    print("=" * 70)

    agent = AGNOCodeReviewerAgent()

    # Example 1: Problematic code
    print("\n\n🔍 Example 1: Code with Multiple Issues")
    print("=" * 70)

    problematic_code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        # TODO: optimize this loop
        result += str(data[i])  # String concatenation in loop
    
    try:
        eval("code")  # Security issue: eval()
    except:  # Bare except clause
        pass
    
    if key in ["a", "b", "c"]:  # Use set instead
        return None
    
    return result
"""

    report1 = agent.review_code(problematic_code, "problematic_module.py")
    agent.display_report(report1)

    # Example 2: Better code
    print("\n\n🔍 Example 2: Improved Code")
    print("=" * 70)

    better_code = '''
def process_data(data: list) -> str:
    """
    Process data and return formatted string.
    
    Args:
        data: Input data list
        
    Returns:
        Formatted string
    """
    # Use list comprehension for efficiency
    result = "".join(str(item) for item in data)
    
    allowed_keys = {"a", "b", "c"}  # Use set for O(1) lookup
    
    try:
        if any(key in allowed_keys for key in data):
            return result
    except (ValueError, TypeError) as e:
        # Specific exception handling
        print(f"Error processing data: {e}")
        return ""
    
    return result
'''

    report2 = agent.review_code(better_code, "improved_module.py")
    agent.display_report(report2)

    # Summary
    print("\n\n" + "=" * 70)
    print("Code Review Summary")
    print("=" * 70)
    print(f"Total files reviewed: {len(agent.reviewed_files)}")
    for report in agent.reviewed_files:
        print(f"\n{report.file_path}:")
        print(f"  Score: {report.overall_score:.1f}/100")
        print(f"  Issues: {len(report.issues)}")

    print("\n" + "=" * 70)
    print("Production Enhancements")
    print("=" * 70)
    print("""
✅ Security pattern detection
✅ Performance issue identification
✅ Code quality metrics
✅ Automated report generation
✅ Issue prioritization

Production Features to Add:
1. AST-based analysis for deeper insights
2. Cyclomatic complexity calculation
3. Dead code detection
4. Import optimization
5. Type hint checking
6. Documentation completeness
7. Test coverage integration
8. Git integration for diff reviews
9. IDE plugin integration
10. Custom rule configuration

See:
- https://medium.com/data-science-collective/building-production-ready-ai-agents-with-agno-a-comprehensive-engineering-guide-22db32413fdd
    """)
