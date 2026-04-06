# Dependency Management & Package Security: Virtual Environments, Packaging, and Vulnerability Scanning

**Author**: Shuvam Banerji Seal  
**Category**: Foundational Skills  
**Difficulty**: Beginner to Intermediate  
**Last Updated**: April 2026

## Problem Statement

Modern Python projects require sophisticated dependency management:
- **Isolation**: Virtual environments prevent conflicts
- **Reproducibility**: Lock files ensure consistent installations
- **Security**: Vulnerability scanning protects against exploits
- **Version Management**: Semantic versioning prevents breaking changes
- **Supply Chain Security**: Verify dependency integrity
- **Performance**: Modern tools (uv) dramatically speed up installations

---

## Theoretical Foundations

### 1. Semantic Versioning (SemVer)

```
Version = MAJOR.MINOR.PATCH

Examples: 1.2.3
  MAJOR = Breaking changes
  MINOR = Backward-compatible features
  PATCH = Bug fixes

Constraints:
  1.2.3     = Exactly 1.2.3
  ^1.2.3    = >=1.2.3, <2.0.0 (Compatible with minor/patch)
  ~1.2.3    = >=1.2.3, <1.3.0 (Compatible with patch only)
  >=1.2.3   = 1.2.3 and later
  >=1.2, <2 = Any 1.x version
```

### 2. Dependency Resolution Model

```
Dependency Graph:
Project
├── requests == 2.31.0
│   ├── urllib3 >= 1.21.1, < 3
│   ├── certifi >= 2017.4.17
│   └── charset-normalizer >= 2, < 3
├── numpy >= 1.20
│   └── setuptools
└── pandas >= 1.3
    ├── numpy >= 1.18.5
    ├── python-dateutil >= 2.8.1
    └── pytz

Resolution must find compatible versions:
  urllib3: Need version satisfying ALL constraints
  numpy: May have multiple consumers, must find common version
```

### 3. Security Model

```
Vulnerability Severity:
  CRITICAL: Remote Code Execution (CVSS 9-10)
  HIGH: Authentication bypass, denial of service (CVSS 7-8.9)
  MEDIUM: Information disclosure (CVSS 4-6.9)
  LOW: Minor issues (CVSS 0-3.9)

Detection Formula:
Vulnerable if: installed_version in affected_versions
Action: Upgrade to patched_version or remove dependency
```

---

## Comprehensive Code Examples

### Example 1: Virtual Environment Setup and Management

```python
import venv
import subprocess
import sys
from pathlib import Path
from typing import Optional

class VirtualEnvironmentManager:
    """Manage Python virtual environments."""
    
    def __init__(self, venv_path: str = ".venv"):
        self.venv_path = Path(venv_path)
        self.python_exe = (
            self.venv_path / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else self.venv_path / "bin" / "python"
        )
        self.pip_exe = (
            self.venv_path / "Scripts" / "pip.exe"
            if sys.platform == "win32"
            else self.venv_path / "bin" / "pip"
        )
    
    def create(self, with_pip: bool = True) -> bool:
        """
        Create virtual environment.
        
        Usage:
            manager = VirtualEnvironmentManager(".venv")
            manager.create()
            # Activate: source .venv/bin/activate
        """
        try:
            venv.create(str(self.venv_path), with_pip=with_pip)
            print(f"✓ Created virtual environment: {self.venv_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to create venv: {e}")
            return False
    
    def is_active(self) -> bool:
        """Check if venv is currently active."""
        return hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and 
            sys.base_prefix != sys.prefix
        )
    
    def install_requirements(self, requirements_file: str) -> bool:
        """Install packages from requirements.txt."""
        try:
            subprocess.run(
                [str(self.pip_exe), "install", "-r", requirements_file],
                check=True,
                capture_output=True
            )
            print(f"✓ Installed from {requirements_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Installation failed: {e.stderr.decode()}")
            return False
    
    def list_packages(self) -> dict[str, str]:
        """List installed packages and versions."""
        try:
            result = subprocess.run(
                [str(self.pip_exe), "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            import json
            packages = json.loads(result.stdout)
            return {pkg["name"]: pkg["version"] for pkg in packages}
        except Exception as e:
            print(f"Error listing packages: {e}")
            return {}
    
    def freeze(self, output_file: str = "requirements.txt") -> bool:
        """Generate requirements.txt with pinned versions."""
        try:
            result = subprocess.run(
                [str(self.pip_exe), "freeze"],
                capture_output=True,
                text=True,
                check=True
            )
            
            with open(output_file, "w") as f:
                f.write(result.stdout)
            
            print(f"✓ Generated {output_file}")
            return True
        except Exception as e:
            print(f"Error freezing requirements: {e}")
            return False


# Usage example
if __name__ == "__main__":
    manager = VirtualEnvironmentManager(".venv_example")
    
    # Create environment
    manager.create()
    
    # Install requirements
    # manager.install_requirements("requirements.txt")
    
    # List packages
    packages = manager.list_packages()
    print(f"Installed packages: {len(packages)}")
    for name, version in list(packages.items())[:5]:
        print(f"  {name}: {version}")
    
    # Freeze for reproducibility
    # manager.freeze("requirements-lock.txt")
```

### Example 2: Modern Dependency Management with Poetry

```python
import subprocess
import json
from typing import Optional
from dataclasses import dataclass

@dataclass
class Dependency:
    """Dependency specification."""
    name: str
    version: str
    extras: list[str] = None  # Optional extras


class PoetryManager:
    """
    Manage dependencies with Poetry.
    
    Poetry provides:
    - Declarative pyproject.toml
    - Automatic lock file generation
    - Dependency resolution
    - Virtual environment management
    """
    
    def __init__(self):
        self.config_file = "pyproject.toml"
    
    def init_project(self, name: str, description: str = "") -> bool:
        """Initialize new Poetry project."""
        try:
            cmd = ["poetry", "new", name]
            subprocess.run(cmd, check=True)
            print(f"✓ Initialized project: {name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to initialize: {e}")
            return False
    
    def add_dependency(self, dependency: str, group: str = "main") -> bool:
        """
        Add dependency to project.
        
        Examples:
            add_dependency("requests>=2.28.0")
            add_dependency("pytest", group="dev")
            add_dependency("numpy[optimization]")
        """
        try:
            cmd = ["poetry", "add", dependency]
            if group != "main":
                cmd.extend(["--group", group])
            
            subprocess.run(cmd, check=True)
            print(f"✓ Added: {dependency}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to add dependency: {e}")
            return False
    
    def lock_dependencies(self) -> bool:
        """Create lock file (poetry.lock) with pinned versions."""
        try:
            subprocess.run(["poetry", "lock"], check=True)
            print("✓ Generated poetry.lock")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to lock: {e}")
            return False
    
    def install_from_lock(self) -> bool:
        """Install dependencies from lock file (reproducible)."""
        try:
            subprocess.run(
                ["poetry", "install", "--no-interaction"],
                check=True
            )
            print("✓ Installed from lock file")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Installation failed: {e}")
            return False
    
    def show_dependencies(self) -> dict:
        """Show dependency tree."""
        try:
            result = subprocess.run(
                ["poetry", "show", "--tree"],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            return {}
        except subprocess.CalledProcessError as e:
            print(f"Error showing dependencies: {e}")
            return {}


# Example pyproject.toml content
EXAMPLE_PYPROJECT = """
[tool.poetry]
name = "my-project"
version = "0.1.0"
description = "Example project"
authors = ["Author <author@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
requests = "^2.31.0"
pandas = "^2.0.0"
numpy = ">=1.20,<3"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0"
black = "^23.0"
mypy = "^1.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
```

### Example 3: UV - Modern, Fast Package Manager

```python
import subprocess
from typing import Optional

class UVManager:
    """
    Manage dependencies with uv (Rust-based, 10-100x faster than pip).
    
    uv replaces: pip, pip-tools, poetry, pipx, pyenv, twine
    """
    
    def __init__(self):
        self.config_file = "uv.lock"
    
    def create_venv(self, path: str = ".venv") -> bool:
        """Create virtual environment."""
        try:
            subprocess.run(
                ["uv", "venv", path],
                check=True
            )
            print(f"✓ Created venv: {path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e}")
            return False
    
    def sync_dependencies(self) -> bool:
        """
        Sync dependencies from pyproject.toml/requirements.txt
        Creates or updates uv.lock
        """
        try:
            subprocess.run(
                ["uv", "sync"],
                check=True
            )
            print("✓ Dependencies synced")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Sync failed: {e}")
            return False
    
    def add_dependency(self, package: str) -> bool:
        """Add new dependency."""
        try:
            subprocess.run(
                ["uv", "add", package],
                check=True
            )
            print(f"✓ Added: {package}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e}")
            return False
    
    def pip_install(self, package: str) -> bool:
        """Direct package installation (uv pip wrapper)."""
        try:
            subprocess.run(
                ["uv", "pip", "install", package],
                check=True
            )
            print(f"✓ Installed: {package}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Installation failed: {e}")
            return False
    
    def compile_requirements(self, 
                           input_file: str = "requirements.in",
                           output_file: str = "requirements.txt") -> bool:
        """
        Compile requirements with pinned versions.
        
        Usage:
            Create requirements.in with loose constraints
            uv compiles to requirements.txt with exact versions
        """
        try:
            subprocess.run(
                ["uv", "pip", "compile", input_file, "-o", output_file],
                check=True
            )
            print(f"✓ Compiled {input_file} → {output_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Compilation failed: {e}")
            return False


# Example usage
"""
Setup with uv:

1. Create project structure:
   uv init my-project
   cd my-project

2. Create virtual environment:
   uv venv

3. Install dependencies:
   uv sync

4. Add new package:
   uv add requests

5. Install dev dependencies:
   uv add --group dev pytest

6. Activate venv:
   source .venv/bin/activate

7. Run Python:
   uv run python script.py
"""
```

### Example 4: Security Scanning and Vulnerability Detection

```python
import subprocess
import json
from typing import Optional
from dataclasses import dataclass

@dataclass
class Vulnerability:
    """Vulnerability information."""
    package: str
    installed_version: str
    affected_versions: list[str]
    fixed_version: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    cve_id: str
    description: str


class SecurityScanner:
    """Scan dependencies for security vulnerabilities."""
    
    def __init__(self):
        self.vulnerabilities: list[Vulnerability] = []
    
    def scan_with_pip_audit(self) -> list[Vulnerability]:
        """
        Scan with pip-audit (from PyPA).
        
        Installation: pip install pip-audit
        """
        try:
            result = subprocess.run(
                ["pip-audit", "--desc", "--format", "json"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                vulnerabilities = data.get("vulnerabilities", [])
                
                for vuln in vulnerabilities:
                    self.vulnerabilities.append(
                        Vulnerability(
                            package=vuln["name"],
                            installed_version=vuln["installed_version"],
                            affected_versions=vuln["affected_versions"],
                            fixed_version=vuln.get("fixed_version", "N/A"),
                            severity=vuln.get("vulnerability_type", "UNKNOWN"),
                            cve_id=vuln.get("id", ""),
                            description=vuln.get("description", "")
                        )
                    )
            
            return self.vulnerabilities
        
        except Exception as e:
            print(f"Error scanning: {e}")
            return []
    
    def scan_with_safety(self) -> list[Vulnerability]:
        """
        Scan with safety (checks against known vulnerabilities).
        
        Installation: pip install safety
        """
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                
                for item in data.get("scanned_packages", {}).items():
                    # Parse safety format
                    pass
            
            return self.vulnerabilities
        
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def check_uv_lock(self) -> bool:
        """
        Check uv.lock file for vulnerabilities.
        
        Installation: pip install uv-secure
        """
        try:
            subprocess.run(
                ["uv-secure", "check"],
                check=True
            )
            print("✓ No vulnerabilities found in uv.lock")
            return True
        except subprocess.CalledProcessError:
            print("✗ Vulnerabilities detected!")
            return False
    
    def report(self) -> str:
        """Generate vulnerability report."""
        if not self.vulnerabilities:
            return "✓ No vulnerabilities found"
        
        report = f"\n⚠️  Found {len(self.vulnerabilities)} vulnerabilities:\n\n"
        
        # Group by severity
        by_severity = {}
        for vuln in self.vulnerabilities:
            if vuln.severity not in by_severity:
                by_severity[vuln.severity] = []
            by_severity[vuln.severity].append(vuln)
        
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity in by_severity:
                report += f"\n{severity}:\n"
                for vuln in by_severity[severity]:
                    report += (
                        f"  {vuln.package} {vuln.installed_version}\n"
                        f"    Fix: Upgrade to {vuln.fixed_version}\n"
                        f"    CVE: {vuln.cve_id}\n"
                    )
        
        return report


# Usage example
if __name__ == "__main__":
    scanner = SecurityScanner()
    
    # Scan for vulnerabilities
    vulnerabilities = scanner.scan_with_pip_audit()
    
    # Generate report
    print(scanner.report())
    
    # Check uv.lock specifically
    # scanner.check_uv_lock()
```

### Example 5: Reproducible Environments

```python
from pathlib import Path
from typing import Optional
import subprocess

class EnvironmentConfig:
    """Manage reproducible Python environments."""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.lock_file = self.project_dir / "requirements-lock.txt"
        self.lock_file_hash = self.project_dir / ".requirements-hash"
    
    def create_lock_file(self, 
                        requirements_file: str = "requirements.txt"
                      ) -> bool:
        """
        Create lock file with cryptographic hashes.
        
        Lock file format:
        package==1.2.3 --hash=sha256:abc123...
        
        Ensures:
        1. Exact version pinning
        2. Package integrity verification
        3. Supply chain security
        """
        try:
            subprocess.run(
                ["pip-compile", 
                 "--generate-hashes",
                 requirements_file,
                 "-o", str(self.lock_file)],
                check=True
            )
            print(f"✓ Created lock file: {self.lock_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create lock file: {e}")
            return False
    
    def install_from_lock_file(self) -> bool:
        """Install dependencies from lock file (reproducible)."""
        if not self.lock_file.exists():
            print(f"Error: Lock file {self.lock_file} not found")
            return False
        
        try:
            subprocess.run(
                ["pip", "install", 
                 "--require-hashes",
                 "-r", str(self.lock_file)],
                check=True
            )
            print("✓ Installed from lock file with hash verification")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Installation failed: {e}")
            return False
    
    def generate_requirements_txt(self) -> bool:
        """Generate requirements.txt from poetry/uv."""
        try:
            # If using Poetry
            result = subprocess.run(
                ["poetry", "export", "-f", "requirements.txt"],
                capture_output=True,
                text=True,
                check=True
            )
            
            with open("requirements.txt", "w") as f:
                f.write(result.stdout)
            
            print("✓ Generated requirements.txt")
            return True
        except:
            print("Poetry not available, try uv export")
            return False
    
    def verify_reproducibility(self) -> bool:
        """Verify environment is reproducible."""
        checks = [
            ("Lock file exists", self.lock_file.exists()),
            ("All deps pinned", self._all_pinned()),
            ("No local versions", self._no_local_versions()),
        ]
        
        print("\nReproducibility Checks:")
        all_pass = True
        for check_name, result in checks:
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}")
            if not result:
                all_pass = False
        
        return all_pass
    
    def _all_pinned(self) -> bool:
        """Check all dependencies are pinned to exact versions."""
        if not self.lock_file.exists():
            return False
        
        with open(self.lock_file) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    if "==" not in line:
                        return False
        return True
    
    def _no_local_versions(self) -> bool:
        """Check for local version identifiers (+local)."""
        if not self.lock_file.exists():
            return False
        
        with open(self.lock_file) as f:
            for line in f:
                if "+local" in line:
                    return False
        return True
```

---

## Step-by-Step Implementation Guide

### 1. Setup Virtual Environment

**Step 1.1: Create venv**
```bash
python -m venv .venv
```

**Step 1.2: Activate**
```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**Step 1.3: Install requirements**
```bash
pip install -r requirements.txt
```

### 2. Upgrade to Modern Tools

**Step 2.1: Install Poetry**
```bash
pip install poetry
poetry init
```

**Step 2.2: Add dependencies**
```bash
poetry add requests pandas
poetry add --group dev pytest
```

**Step 2.3: Generate lock file**
```bash
poetry lock
poetry install
```

### 3. Security Scanning

**Step 3.1: Install audit tools**
```bash
pip install pip-audit safety
```

**Step 3.2: Scan dependencies**
```bash
pip-audit
safety check
```

### 4. Reproducible Builds

**Step 4.1: Create lock file with hashes**
```bash
pip-compile --generate-hashes requirements.in
```

**Step 4.2: Install with verification**
```bash
pip install --require-hashes -r requirements.txt
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Version Conflicts
**Problem**: Incompatible dependency versions
```
ERROR: pip's dependency resolver doesn't work this way
```

**Solution**: Use lock files and explicit constraints
```toml
[tool.poetry.dependencies]
requests = "^2.28.0"
urllib3 = ">=1.21.1,<3"
```

### Pitfall 2: Reproducibility Issues
**Problem**: Different installations have different versions
```bash
pip install requests  # Different version each time
```

**Solution**: Always use lock files
```bash
pip install --require-hashes -r requirements-lock.txt
```

### Pitfall 3: Ignoring Security Updates
**Problem**: Using outdated, vulnerable packages
```
Installation outdated, known CVEs exist
```

**Solution**: Regular security audits
```bash
pip-audit --fix
```

---

## Authoritative Sources

1. **pip documentation**: https://pip.pypa.io/
2. **venv documentation**: https://docs.python.org/3/library/venv.html
3. **Poetry**: https://python-poetry.org/
4. **uv package manager**: https://docs.astral.sh/uv/
5. **pip-audit**: https://github.com/pypa/pip-audit
6. **safety**: https://github.com/pyupio/safety
7. **PEP 508 - Dependency specification**: https://peps.python.org/pep-0508/
8. **Semantic Versioning**: https://semver.org/
9. **PyPA - Python Packaging Authority**: https://www.pypa.io/
10. **Software Composition Analysis**: https://www.gartner.com/reviews/market/software-composition-analysis

---

## Summary

Master dependency management through:
- Virtual environments for isolation
- Modern tools (Poetry, uv) for speed and reliability
- Lock files for reproducibility
- Security scanning for vulnerability detection
- Version pinning for stability

These patterns enable safe, reproducible Python systems with minimal supply chain risk.
