# Contributing to LLM-Whisperer

Thank you for your interest in contributing to LLM-Whisperer! This guide will help you understand how to contribute effectively to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [What Can You Contribute?](#what-can-you-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contribution Types](#contribution-types)
- [Folder Contract Template](#folder-contract-template)
- [Documentation Standards](#documentation-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and professional in all interactions
- Provide constructive feedback
- Assume good intentions
- Help others learn and grow
- Report violations to the repository maintainers

---

## What Can You Contribute?

### 1. **New Skills & Patterns**
- Battle-tested LLM engineering techniques
- Prompting strategies that work well
- Fine-tuning methodologies
- RAG optimization patterns
- Agentic orchestration approaches
- Inference optimization tricks
- Production deployment patterns

**Location**: `skills/[category]/`

### 2. **Evaluation Improvements**
- Custom benchmark implementations
- New judge rubrics
- Safety detection enhancements
- Latency profiling tools
- Regression test datasets
- Metric computation improvements

**Location**: `evaluation/[category]/`

### 3. **Fine-Tuning Recipes**
- Novel LoRA configurations
- Multi-task training strategies
- Curriculum learning approaches
- Preference learning techniques
- Domain-specific tuning recipes

**Location**: `fine_tuning/[method]/`

### 4. **Infrastructure & DevOps**
- Kubernetes manifests
- Terraform modules
- Docker optimizations
- Monitoring dashboards
- CI/CD pipeline improvements
- Deployment automation

**Location**: `infra/` and `pipelines/`

### 5. **Documentation & Examples**
- Tutorials and guides
- Code examples and notebooks
- Architecture decision records
- Troubleshooting guides
- Best practices documentation
- Research and analysis reports

**Location**: `docs/` and `notebooks/`

### 6. **Bug Fixes & Improvements**
- Code quality improvements
- Performance optimizations
- Test coverage increases
- Documentation fixes
- Breaking change resolutions

**Anywhere in the repo**

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/LLM-Whisperer.git
cd LLM-Whisperer

# Add upstream remote
git remote add upstream https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer.git
```

### 2. Create a Branch

```bash
# Update your local repository
git fetch upstream
git checkout upstream/main

# Create a new branch for your contribution
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install black flake8 pytest jupyter
```

---

## Development Workflow

### Making Changes

1. **Follow the folder contract** - See [Folder Contract Template](#folder-contract-template) below
2. **Write clear code** - Use type hints, docstrings, meaningful variable names
3. **Add tests** - Include unit and integration tests for your code
4. **Update documentation** - Add or update README and docstrings
5. **Test locally** - Run tests before submitting

### Code Style

- **Python**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
  ```bash
  # Format with black
  black --line-length 100 your_file.py
  
  # Check with flake8
  flake8 your_file.py
  ```

- **Markdown**: Clear, well-structured documentation
  - Use consistent heading levels (# ## ### format)
  - Include code examples
  - Link to relevant resources
  - Update table of contents if needed

- **YAML**: Readable configuration files
  - Use 2-space indentation
  - Include comments explaining options
  - Provide example configurations

---

## Contribution Types

### Type 1: New Skill

**Steps:**

1. Create directory structure:
   ```
   skills/[category]/[skill-name]/
   ├── README.md           # Skill description and usage
   ├── examples/           # Runnable examples
   │   ├── basic.py
   │   └── advanced.py
   ├── src/                # Implementation code
   │   ├── __init__.py
   │   └── core.py
   ├── configs/            # YAML configurations
   │   └── default.yaml
   └── tests/              # Unit tests
       └── test_skill.py
   ```

2. Write comprehensive README:
   ```markdown
   # [Skill Name]
   
   **Purpose**: One-sentence description
   
   ## When to Use
   - Use case 1
   - Use case 2
   
   ## Implementation
   
   [Your implementation details]
   
   ## Examples
   
   [Code examples]
   
   ## References
   
   - Link to papers
   - Link to repositories
   - Related skills
   ```

3. Add examples in `examples/`:
   ```python
   """Basic example of the skill."""
   
   from src.core import YourSkill
   
   # Initialize
   skill = YourSkill()
   
   # Use it
   result = skill.run(input_data)
   ```

4. Add tests in `tests/`:
   ```python
   """Tests for your skill."""
   
   import pytest
   from src.core import YourSkill
   
   def test_basic_functionality():
       skill = YourSkill()
       result = skill.run(test_input)
       assert result == expected_output
   ```

### Type 2: New Evaluation Method

**Steps:**

1. Create directory:
   ```
   evaluation/[category]/[method]/
   ├── README.md
   ├── src/
   │   ├── evaluator.py
   │   ├── metrics.py
   │   └── __init__.py
   ├── configs/
   │   └── config.yaml
   ├── examples/
   │   └── basic.py
   └── tests/
       └── test_evaluator.py
   ```

2. Implement evaluator interface:
   ```python
   from evaluation.src.base import Evaluator
   
   class YourEvaluator(Evaluator):
       def __init__(self, config):
           super().__init__("YourEvaluator", config)
       
       def evaluate(self, inputs, outputs):
           """Evaluate model outputs."""
           # Your evaluation logic
           return results
   ```

3. Add comprehensive tests and documentation

### Type 3: Infrastructure Template

**Steps:**

1. Create directory:
   ```
   infra/[component]/
   ├── README.md
   ├── manifests/
   │   └── *.yaml
   ├── terraform/
   │   ├── main.tf
   │   ├── variables.tf
   │   └── outputs.tf
   ├── docker/
   │   └── Dockerfile
   ├── scripts/
   │   └── deploy.sh
   └── examples/
       └── usage.md
   ```

2. Write clear documentation explaining:
   - What the component does
   - Prerequisites
   - How to deploy
   - Configuration options
   - Troubleshooting

---

## Folder Contract Template

Every major module should follow this structure:

```
module/
├── README.md              # What, why, when, how
├── src/                   # Implementation code
│   ├── __init__.py
│   ├── core.py            # Main implementation
│   ├── utils.py           # Helper functions
│   └── types.py           # Type definitions
├── configs/               # YAML configuration presets
│   ├── default.yaml
│   ├── advanced.yaml
│   └── README.md          # Config documentation
├── scripts/               # Launch and utility scripts
│   ├── run.py
│   ├── setup.sh
│   └── README.md          # Script documentation
├── examples/              # Minimal usage examples
│   ├── basic.py
│   ├── advanced.py
│   └── README.md          # Example guide
├── tests/                 # Unit and integration tests
│   ├── test_core.py
│   ├── test_utils.py
│   └── conftest.py        # Pytest fixtures
├── artifacts/             # Output directory (gitignored)
│   └── .gitkeep
└── .gitignore             # Local ignores
```

---

## Documentation Standards

### README.md Template

```markdown
# [Module/Skill Name]

**Brief description** - one sentence

## Overview

What this does, when to use it, key benefits.

## Quick Start

1. Installation/setup step
2. Basic usage example
3. Next steps

## Key Concepts

- Concept 1: Explanation
- Concept 2: Explanation
- Concept 3: Explanation

## Structure

- `src/`: Core implementation
- `configs/`: Configuration templates
- `examples/`: Runnable examples
- `tests/`: Test suite

## Usage Examples

### Basic Example
```python
code example
```

### Advanced Example
```python
code example
```

## Configuration

```yaml
# YAML configuration example
key: value
```

## References

- [Paper/Resource 1](link)
- [Paper/Resource 2](link)
- Related modules: [Link 1](#), [Link 2](#)

## Contributing

Guidelines for improving this module.
```

### Code Documentation

```python
"""Module docstring explaining the module's purpose."""

def function_name(param1: str, param2: int) -> str:
    """
    Short description of what the function does.
    
    Long description if needed, explaining complex behavior,
    edge cases, or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When this error occurs
    
    Examples:
        >>> result = function_name("input", 42)
        >>> result
        'output'
    """
```

---

## Testing Guidelines

### Unit Tests

```python
"""Test basic functionality in isolation."""

import pytest
from module.src.core import YourClass

@pytest.fixture
def setup_data():
    """Setup test data."""
    return {"input": "data"}

def test_basic_functionality(setup_data):
    """Test basic behavior."""
    obj = YourClass()
    result = obj.process(setup_data["input"])
    assert result is not None

def test_error_handling():
    """Test error cases."""
    obj = YourClass()
    with pytest.raises(ValueError):
        obj.process(invalid_input)
```

### Integration Tests

```python
"""Test components working together."""

def test_end_to_end_workflow():
    """Test complete workflow."""
    # Setup
    component1 = Component1()
    component2 = Component2()
    
    # Execute
    result1 = component1.process(data)
    result2 = component2.process(result1)
    
    # Verify
    assert result2 == expected_output
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_core.py::test_function_name
```

---

## Commit Message Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, semicolons, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Test additions or changes
- **chore**: Dependency updates, build config changes

### Examples

```
feat(evaluation): add MMLU benchmark implementation

- Implement MMLU dataset loading
- Add accuracy scoring
- Include per-subject breakdown

Closes #123

feat(skills/rag): add hybrid search pattern
fix(inference): resolve vLLM batching bug
docs(guides): update Kubernetes deployment guide
test(evaluation): add safety detector tests
```

---

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests locally**:
   ```bash
   pytest
   ```

3. **Check code style**:
   ```bash
   black --check .
   flake8 .
   ```

4. **Update documentation**:
   - Update README if needed
   - Update table of contents
   - Add references to new resources

### Submitting Your PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - **Title**: Follow conventional commits format
   - **Description**: 
     - What you changed
     - Why you changed it
     - How to test the changes
     - Screenshots/examples if applicable
   - **Checklist**:
     - [ ] Tests pass locally
     - [ ] Documentation is updated
     - [ ] Code follows style guidelines
     - [ ] No breaking changes (or documented)

3. **Example PR Description**:
   ```markdown
   ## Description
   
   Implement MMLU benchmark for task evaluation.
   
   ## Changes
   
   - Added MMLUDataset class for loading MMLU questions
   - Implemented accuracy scoring with per-subject breakdown
   - Added 5 test cases for validation
   
   ## How to Test
   
   ```bash
   pytest evaluation/task_benchmarks/tests/test_mmlu.py
   ```
   
   ## Related Issues
   
   Closes #123
   ```

### Review Process

Your PR will be reviewed by maintainers who may:

- Ask for changes
- Request additional tests
- Suggest documentation improvements
- Offer feedback on design

**Be responsive**: Please respond to reviews within 48 hours.

### After Approval

Once approved, your PR will be merged to the main branch. You'll be:

- Added to contributors list
- Credited in release notes
- Acknowledged in documentation

---

## Common Questions

**Q: What's the minimum to contribute?**
A: A well-documented, tested feature in the folder contract format.

**Q: Can I contribute just documentation?**
A: Absolutely! Documentation improvements are valuable contributions.

**Q: How long does review take?**
A: Usually 3-7 days, depending on complexity and maintainer availability.

**Q: What if my PR conflicts with main?**
A: Rebase on main and resolve conflicts:
```bash
git fetch upstream
git rebase upstream/main
# Resolve conflicts
git add .
git rebase --continue
git push -f origin your-branch
```

**Q: Can I contribute proprietary code?**
A: Only if you own/control the IP and license it under MIT.

---

## Getting Help

- **Questions?** Create a GitHub Discussion
- **Found a bug?** Open an Issue with reproduction steps
- **Need guidance?** Ask in an Issue before starting work
- **Want feedback?** Open a draft PR early

---

## Contributor Recognition

We recognize all contributions! Contributors are acknowledged:

- In git history
- In `CONTRIBUTORS.md` (coming soon)
- In release notes
- In repository documentation

---

## License

By contributing to LLM-Whisperer, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for contributing to LLM-Whisperer! 🚀

Your work helps the community build better LLM systems.
