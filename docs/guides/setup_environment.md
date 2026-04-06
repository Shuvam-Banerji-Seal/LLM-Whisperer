# Environment Setup Guide

**Author:** Shuvam Banerji Seal

## System Requirements

### Python Version

**Minimum:** Python 3.10
**Recommended:** Python 3.12+

Check your Python version:
```bash
python3 --version
```

Download Python from: https://www.python.org/downloads/

### Operating System

- **macOS:** 10.14+
- **Linux:** Ubuntu 18.04+, Debian 10+, CentOS 7+
- **Windows:** Windows 10+

## Virtual Environment Setup

### Using uv (Recommended - Fast & Modern)

**Installation:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Create virtual environment with specific Python version:**
```bash
uv venv --python 3.12
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

**Verify activation:**
```bash
which python  # macOS/Linux
# or
where python  # Windows
```

### Using venv (Standard)

**Create virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

**Verify activation:**
```bash
python3 -m pip --version
```

### Using Conda

**Create environment:**
```bash
conda create -n llm-agents python=3.12
conda activate llm-agents
```

**Check activation:**
```bash
which python  # should point to conda env
```

## Dependency Installation

### Agno Framework

**Core installation:**
```bash
pip install -U agno
```

**With Anthropic support (Claude):**
```bash
pip install -U 'agno[anthropic]'
```

**With OpenAI support (GPT):**
```bash
pip install -U 'agno[openai]'
```

**With Google support (Gemini):**
```bash
pip install -U 'agno[google]'
```

**Full installation with AgentOS:**
```bash
pip install -U 'agno[os]' anthropic fastapi uvicorn
```

**Verify installation:**
```bash
python -c "import agno; print(agno.__version__)"
```

### LangChain Framework

**Core installation:**
```bash
pip install -U langchain
```

**With Anthropic support:**
```bash
pip install -U langchain langchain-anthropic
```

**With OpenAI support:**
```bash
pip install -U langchain langchain-openai
```

**With LangGraph (state machines):**
```bash
pip install -U langgraph
```

**Full installation:**
```bash
pip install -U langchain langchain-anthropic langsmith langgraph
```

**Verify installation:**
```bash
python -c "import langchain; print(langchain.__version__)"
python -c "import langgraph; print('LangGraph installed')"
```

### Optional Tools & Integrations

**Web Search Integration:**
```bash
pip install -U duckduckgo-search requests-html
```

**Data Processing:**
```bash
pip install -U pandas numpy scipy
```

**Document Processing:**
```bash
pip install -U pypdf python-docx openpyxl
```

**Development Tools:**
```bash
pip install -U jupyter notebook ipython
```

**Testing:**
```bash
pip install -U pytest pytest-asyncio
```

## API Key Configuration

### Anthropic (Claude)

1. **Get API Key:**
   - Visit https://console.anthropic.com/
   - Sign up or log in
   - Navigate to API keys section
   - Create new key

2. **Set Environment Variable:**
   
   **macOS/Linux:**
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-v1-xxxxx
   
   # Make permanent by adding to ~/.bash_profile or ~/.zshrc
   echo 'export ANTHROPIC_API_KEY=sk-ant-v1-xxxxx' >> ~/.zshrc
   source ~/.zshrc
   ```
   
   **Windows:**
   ```powershell
   setx ANTHROPIC_API_KEY "sk-ant-v1-xxxxx"
   # Restart PowerShell after setting
   ```

3. **Verify Setup:**
   ```bash
   echo $ANTHROPIC_API_KEY  # macOS/Linux
   # or
   echo %ANTHROPIC_API_KEY%  # Windows
   ```

### OpenAI (GPT)

1. **Get API Key:**
   - Visit https://platform.openai.com/api-keys
   - Sign up or log in
   - Create new API key

2. **Set Environment Variable:**
   
   **macOS/Linux:**
   ```bash
   export OPENAI_API_KEY=sk-proj-xxxxx
   echo 'export OPENAI_API_KEY=sk-proj-xxxxx' >> ~/.zshrc
   source ~/.zshrc
   ```
   
   **Windows:**
   ```powershell
   setx OPENAI_API_KEY "sk-proj-xxxxx"
   ```

3. **Verify:**
   ```bash
   python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```

### Google (Gemini)

1. **Get API Key:**
   - Visit https://ai.google.dev/
   - Click "Get API key in Google AI Studio"
   - Create API key

2. **Set Environment Variable:**
   ```bash
   export GOOGLE_API_KEY=xxxxx
   ```

### LangSmith (Observability & Debugging)

1. **Get API Key:**
   - Visit https://smith.langchain.com/
   - Sign up
   - Navigate to API keys
   - Create key

2. **Configure:**
   ```bash
   export LANGSMITH_API_KEY=ls-xxxxx
   export LANGSMITH_TRACING=true
   export LANGSMITH_PROJECT="my-project"
   ```

## .env File Setup

**Create `.env` file in project root:**
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-v1-xxxxx
OPENAI_API_KEY=sk-proj-xxxxx
GOOGLE_API_KEY=xxxxx
LANGSMITH_API_KEY=ls-xxxxx
LANGSMITH_TRACING=true
```

**Load in Python:**
```python
from dotenv import load_dotenv
import os

load_dotenv()

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
```

**Or install python-dotenv:**
```bash
pip install -U python-dotenv
```

## Configuration Options

### Agno Configuration

**Create `agno_config.py`:**
```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.db.sqlite import SqliteDb

# Common configurations
DEFAULT_MODEL = Claude(id="claude-sonnet-4-5")
DEFAULT_DB = SqliteDb(db_file="agno.db")

AGENT_CONFIG = {
    "add_datetime_to_context": True,
    "add_history_to_context": True,
    "num_history_runs": 5,
    "markdown": True,
}

# Create agent with config
agent = Agent(
    name="Configured Agent",
    model=DEFAULT_MODEL,
    db=DEFAULT_DB,
    **AGENT_CONFIG,
)
```

### LangChain Configuration

**Create `langchain_config.py`:**
```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Model selection
MODELS = {
    "anthropic": ChatAnthropic(model="claude-sonnet-4"),
    "openai": ChatOpenAI(model="gpt-4o"),
}

# Default configuration
DEFAULT_MODEL = MODELS["anthropic"]

# Agent settings
AGENT_SETTINGS = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
}
```

## Common Setup Issues and Solutions

### Issue: "Python version too old"

**Solution:** Install Python 3.12 or higher
```bash
# Check version
python3 --version

# Install specific version
# macOS with Homebrew
brew install python@3.12

# Linux
sudo apt-get install python3.12

# Windows - download from python.org
```

### Issue: Virtual environment not found after restart

**Solution:** Activate virtual environment at shell start
```bash
# Add to ~/.zshrc, ~/.bashrc, or equivalent
source /path/to/.venv/bin/activate
```

### Issue: "Module not found" despite installation

**Solution:** Verify you're in the correct virtual environment
```bash
which python  # should point to .venv/bin/python
pip list  # should show installed packages
```

### Issue: API key not recognized

**Solution:** Verify environment variables
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Or test in Python
import os
print(os.getenv("ANTHROPIC_API_KEY"))

# Make sure to restart terminal/IDE after setting
```

### Issue: Permission denied on installation

**Solution:** Use `--user` flag or ensure virtual environment
```bash
pip install --user agno  # Install in user directory
# or
source .venv/bin/activate  # Use virtual environment
pip install agno
```

### Issue: "pip: command not found"

**Solution:** Use python -m pip
```bash
python -m pip install -U agno
# or use uv
uv pip install agno
```

### Issue: Dependency conflicts

**Solution:** Create fresh environment
```bash
deactivate  # Exit current environment
rm -rf .venv  # Remove old environment
python3 -m venv .venv  # Create new
source .venv/bin/activate
pip install -U --upgrade-strategy=eager agno langchain
```

### Issue: slow installation with pip

**Solution:** Use uv for faster installation
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install agno langchain
```

### Issue: ModuleNotFoundError after installation

**Solution:** Reinstall in correct environment
```bash
# Verify environment
which python

# Reinstall
pip uninstall -y agno
pip install -U agno
```

## Dependency Management

### Requirements File

**Create `requirements.txt`:**
```txt
agno[os]==0.7.0
langchain==0.1.0
langchain-anthropic==0.1.0
langgraph==0.1.0
python-dotenv==1.0.0
requests==2.31.0
pandas==2.0.0
jupyter==1.0.0
```

**Install from file:**
```bash
pip install -r requirements.txt
```

**Export current environment:**
```bash
pip freeze > requirements.txt
```

### pyproject.toml (Modern Python Projects)

**Create `pyproject.toml`:**
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-agents"
version = "0.1.0"
dependencies = [
    "agno[os]>=0.7.0",
    "langchain>=0.1.0",
    "langchain-anthropic>=0.1.0",
    "langgraph>=0.1.0",
    "python-dotenv>=1.0.0",
]

[tool.setuptools]
packages = ["src"]
```

**Install from pyproject.toml:**
```bash
pip install -e .
```

## Development Setup Checklist

- [ ] Python 3.12+ installed
- [ ] Virtual environment created and activated
- [ ] Core dependencies installed (agno or langchain)
- [ ] API keys set as environment variables
- [ ] `.env` file created (if using python-dotenv)
- [ ] Verification scripts run successfully
- [ ] Optional tools installed as needed
- [ ] IDE configured with correct Python interpreter

## Quick Start Script

**Create `setup.sh` for automated setup:**
```bash
#!/bin/bash

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -U agno langchain langchain-anthropic langgraph python-dotenv

# Create .env template
cat > .env.template << 'EOF'
ANTHROPIC_API_KEY=sk-ant-v1-xxxxx
OPENAI_API_KEY=sk-proj-xxxxx
LANGSMITH_API_KEY=ls-xxxxx
LANGSMITH_TRACING=true
EOF

echo "Setup complete! Don't forget to:"
echo "1. Copy .env.template to .env"
echo "2. Fill in your API keys"
echo "3. Run: source .venv/bin/activate"
```

**Make executable and run:**
```bash
chmod +x setup.sh
./setup.sh
```

## Next Steps

1. Follow the Quick Start guides for Agno or LangChain
2. Create your first agent
3. Set up IDE with proper interpreter
4. Explore tools and integrations
5. Enable LangSmith for debugging

## Support

- **Agno Docs:** https://docs.agno.com/
- **LangChain Docs:** https://python.langchain.com/docs/
- **Python Docs:** https://docs.python.org/3/
- **Virtual Environments:** https://docs.python.org/3/library/venv.html
