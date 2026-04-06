# Troubleshooting Guide

**Author:** Shuvam Banerji Seal

## Common Issues and Solutions

### Import and Installation Errors

#### Issue: "ModuleNotFoundError: No module named 'agno'"

**Causes:**
- Package not installed
- Wrong virtual environment
- Installation was incomplete

**Solutions:**

1. Verify virtual environment is active:
```bash
which python  # Should show path in .venv
```

2. Reinstall the package:
```bash
pip install --upgrade --force-reinstall agno
```

3. Check installation:
```bash
python -c "import agno; print(agno.__version__)"
```

4. Use a fresh environment:
```bash
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U agno
```

#### Issue: "ModuleNotFoundError: No module named 'langchain'"

**Solution:**
```bash
pip install -U langchain langchain-anthropic
```

#### Issue: "ImportError: cannot import name 'Agent' from 'agno.agent'"

**Solution:** Check that you have the correct version:
```bash
pip install -U --upgrade-strategy=eager agno
pip show agno  # Check version number
```

#### Issue: "No module named 'anthropic'"

**Solution:** Install anthropic separately:
```bash
pip install -U anthropic
```

### API Key and Authentication Errors

#### Issue: "API key not found" or "Invalid API key"

**Solutions:**

1. Check environment variable is set:
```bash
# macOS/Linux
echo $ANTHROPIC_API_KEY

# Windows PowerShell
echo $env:ANTHROPIC_API_KEY
```

2. Set it correctly:
```bash
# macOS/Linux
export ANTHROPIC_API_KEY=sk-ant-v1-your-key-here
# Make it permanent:
echo 'export ANTHROPIC_API_KEY=sk-ant-v1-your-key-here' >> ~/.zshrc
source ~/.zshrc

# Windows
setx ANTHROPIC_API_KEY "sk-ant-v1-your-key-here"
# Restart PowerShell
```

3. Use .env file:
```bash
pip install -U python-dotenv
```

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
```

4. Pass key directly (for testing only):
```python
from agno.models.anthropic import Claude

model = Claude(
    id="claude-sonnet-4-5",
    api_key="sk-ant-v1-your-key-here"
)
```

#### Issue: "Invalid API key" with correct key set

**Solutions:**

1. Check key is not expired:
   - Visit your provider's dashboard
   - Regenerate key if needed

2. Ensure no extra whitespace:
```bash
# Wrong
export ANTHROPIC_API_KEY=" sk-ant-v1-xxxxx "

# Right
export ANTHROPIC_API_KEY=sk-ant-v1-xxxxx
```

3. Check API key format:
   - Anthropic: Starts with `sk-ant-v1-`
   - OpenAI: Starts with `sk-`
   - Google: Long alphanumeric string

4. Verify you have the right key for the provider:
```python
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat

# Different providers need different keys
claude_model = Claude(id="claude-sonnet-4-5")  # Uses ANTHROPIC_API_KEY
openai_model = OpenAIChat(id="gpt-4o")  # Uses OPENAI_API_KEY
```

#### Issue: "401 Unauthorized" or "403 Forbidden"

**Causes:**
- API key lacks permissions
- API key from wrong account
- API key has been revoked
- Rate limit exceeded

**Solutions:**

1. Check API key account:
   - Log into provider dashboard
   - Verify key belongs to correct account

2. Check permissions/quotas:
   - Visit billing dashboard
   - Ensure account is in good standing
   - Check usage limits

3. Generate new key:
```bash
# Go to provider dashboard and create new key
# Then update environment variable
export ANTHROPIC_API_KEY=sk-ant-v1-new-key
```

### Tool and Function Errors

#### Issue: "Tool not found" or "Unknown tool"

**Solution:** Ensure tool is properly defined:
```python
from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools

# Correct - pass tool instances in a list
agent = Agent(
    tools=[HackerNewsTools()],
)

# Wrong - don't pass class names
# agent = Agent(tools=[HackerNewsTools])
```

#### Issue: "Tool execution failed"

**Solutions:**

1. Check tool parameters match function signature:
```python
from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools

@tool
def get_price(symbol: str) -> str:
    """Get stock price - must have type hints."""
    return f"Price: $100"

# Must have clear docstring
```

2. Verify external dependencies:
```bash
# For web search
pip install -U duckduckgo-search

# For finance
pip install -U yfinance

# For news
pip install -U feedparser
```

3. Check internet connection:
```python
import requests
try:
    requests.get("https://www.example.com", timeout=5)
    print("Internet connection OK")
except:
    print("No internet connection")
```

### Database and Memory Errors

#### Issue: "sqlite3.OperationalError: database is locked"

**Causes:**
- Multiple processes accessing same database
- Incomplete write operation

**Solutions:**

1. Check for other processes:
```bash
# macOS/Linux
lsof | grep agent.db

# Windows
tasklist | find "python"
```

2. Close all instances and restart:
```bash
# Kill all Python processes
pkill -f python

# Restart your agent
python my_agent.py
```

3. Use different database file:
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

agent = Agent(
    db=SqliteDb(db_file="agent_new.db")
)
```

4. Delete corrupted database:
```bash
rm agent.db
# Agent will create new one automatically
```

#### Issue: "Memory not persisting between runs"

**Causes:**
- No database configured
- Database file in wrong location
- History not enabled

**Solutions:**

1. Add database configuration:
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

agent = Agent(
    db=SqliteDb(db_file="agent.db"),
    add_history_to_context=True,
    num_history_runs=5,  # Include last 5 conversations
)
```

2. Check database file exists:
```bash
ls -la agent.db  # Should exist after first run
```

3. Verify history context is enabled:
```python
agent = Agent(
    add_history_to_context=True,  # Must be True
    add_datetime_to_context=True,
)
```

### Response and Output Errors

#### Issue: Agent response is empty or "None"

**Causes:**
- Tool execution failed silently
- Model didn't generate output
- Response not properly captured

**Solutions:**

1. Use print_response() for debugging:
```python
agent.print_response("Your question", stream=True)
# Better visibility than invoke()
```

2. Check stream output:
```python
stream = agent.run("Your question", stream=True)
for chunk in stream:
    print(chunk)
```

3. Increase max tokens:
```python
from agno.agent import Agent
from agno.models.anthropic import Claude

agent = Agent(
    model=Claude(
        id="claude-sonnet-4-5",
        max_tokens=4096,  # Increase limit
    ),
)
```

#### Issue: Incomplete or truncated responses

**Causes:**
- Max tokens reached
- Model timeout
- Streaming interrupted

**Solutions:**

1. Increase max tokens:
```python
model = Claude(id="claude-sonnet-4-5", max_tokens=8192)
```

2. Use streaming for long responses:
```python
agent.print_response("Long question", stream=True)
# Streaming shows partial results as they arrive
```

3. Check timeout settings:
```python
from agno.agent import Agent

agent = Agent(
    # Increase timeout for slow operations
    timeout=60,  # seconds
)
```

### Performance Issues

#### Issue: Agent is very slow

**Causes:**
- Slow tool execution
- Network latency
- Large context size
- Inefficient prompts

**Solutions:**

1. Profile tool execution:
```python
import time

@tool
def slow_tool(query: str) -> str:
    start = time.time()
    result = expensive_operation()
    print(f"Tool took {time.time() - start:.2f}s")
    return result
```

2. Reduce history:
```python
agent = Agent(
    num_history_runs=2,  # Use fewer previous runs
    add_history_to_context=True,
)
```

3. Cache tool results:
```python
from functools import lru_cache

@tool
@lru_cache(maxsize=128)
def cached_tool(query: str) -> str:
    return expensive_operation()
```

4. Use faster model:
```python
# Use smaller, faster models
model = Claude(id="claude-3.5-haiku")
```

5. Enable response caching during development:
```python
agent = Agent(
    cache_response=True,  # Cache during development
)
```

#### Issue: High memory usage

**Causes:**
- Large knowledge base
- Long conversation history
- Tool results not cleaned up
- Memory leaks in tools

**Solutions:**

1. Limit context size:
```python
agent = Agent(
    num_history_runs=2,  # Reduce history
    add_history_to_context=True,
)
```

2. Use conversation summary:
```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    max_token_limit=1000,  # Limit memory size
)
```

3. Clean up old sessions:
```python
# Periodically delete old database entries
db.delete_old_sessions(days=30)
```

4. Use streaming instead of loading entire response:
```python
# Good - streams output
agent.print_response("query", stream=True)

# Avoid - loads entire response in memory
response = agent.invoke(...)
```

### Deployment Issues

#### Issue: "Address already in use" (Port conflict)

**Causes:**
- Another process using port 8000
- Previous server not shut down

**Solutions:**

1. Use different port:
```bash
fastapi dev agent_os_app.py --port 8001
```

2. Kill process using port:
```bash
# macOS/Linux
lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

3. Check for hanging processes:
```bash
# macOS/Linux
ps aux | grep python
```

#### Issue: "Connection refused" in production

**Solutions:**

1. Verify service is running:
```bash
curl http://localhost:8000/health
```

2. Check firewall:
```bash
# Allow port through firewall
sudo ufw allow 8000
```

3. Verify environment variables are set:
```python
import os
print(os.getenv("ANTHROPIC_API_KEY"))  # Should not be None
```

4. Check logs:
```bash
# If using systemd
journalctl -u agent-service -f

# If running in Docker
docker logs container-name
```

## Debugging Techniques

### Enable Verbose Logging

**Agno:**
```python
from agno.agent import Agent
import logging

logging.basicConfig(level=logging.DEBUG)
agent = Agent(debug=True)
```

**LangChain:**
```python
from langchain.agents import create_agent
import logging

logging.basicConfig(level=logging.DEBUG)
agent = create_agent(...)
```

### Use LangSmith for Tracing

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=ls-xxxxx
```

Then visit: https://smith.langchain.com

### Print Debug Information

```python
from agno.agent import Agent

agent = Agent(...)

# Inspect agent configuration
print(f"Model: {agent.model}")
print(f"Tools: {agent.tools}")
print(f"Database: {agent.db}")

# Test tool execution
for tool in agent.tools:
    print(f"Testing {tool.name}")
    result = tool.run("test query")
    print(f"Result: {result}")
```

### Test Components Separately

```python
# Test model independently
from agno.models.anthropic import Claude

model = Claude(id="claude-sonnet-4-5")
response = model.invoke("Hello")
print(response)

# Test tool independently
from agno.tools.hackernews import HackerNewsTools

tools = HackerNewsTools()
stories = tools.fetch()
print(stories)

# Then test together in agent
agent = Agent(model=model, tools=[tools])
```

## Performance Debugging Checklist

- [ ] Check internet connection
- [ ] Verify API keys are valid
- [ ] Monitor token usage
- [ ] Profile tool execution times
- [ ] Check database size
- [ ] Review memory usage
- [ ] Enable logging and tracing
- [ ] Test with simpler queries
- [ ] Use streaming for long responses
- [ ] Profile with `cProfile` for Python

## Getting Help

1. **Check Documentation:**
   - Agno: https://docs.agno.com/
   - LangChain: https://python.langchain.com/docs/

2. **Search Issues:**
   - Agno GitHub: https://github.com/agno-ai/agno/issues
   - LangChain GitHub: https://github.com/langchain-ai/langchain/issues

3. **Community Support:**
   - Agno Discord: https://discord.gg/agno
   - LangChain Discord: https://discord.gg/langchain
   - Stack Overflow: Tag with `agno` or `langchain`

4. **Detailed Logs:**
   When reporting issues, include:
   - Full error message
   - Code snippet reproducing the issue
   - Python version and OS
   - Package versions: `pip show agno langchain`
   - Environment details

## Summary

Most issues fall into these categories:
- **Installation:** Reinstall in fresh virtual environment
- **Authentication:** Check API keys and environment variables
- **Tools:** Verify function signatures and docstrings
- **Memory:** Add database configuration
- **Performance:** Use streaming, reduce history, profile code
- **Deployment:** Check ports, logs, and environment setup

Start with simple cases, verify each component works, then build complexity gradually!
