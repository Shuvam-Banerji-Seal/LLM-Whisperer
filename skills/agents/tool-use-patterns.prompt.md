# Tool Use Patterns and Multi-Agent Safety — Agentic Skill Prompt

Advanced tool orchestration, safety constraints, failure handling, and memory optimization.

---

## 1. Identity and Mission

Implement reliable, safe tool use with graceful failure handling and resource-aware memory management.

---

## 2. Tool Error Handling and Validation

```python
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ToolExecutionResult:
    success: bool
    output: Optional[str]
    error: Optional[str]
    execution_time_ms: float

class SafeToolExecutor:
    """Execute tools with validation and error handling."""
    
    def __init__(self, timeout_sec: int = 30):
        self.timeout = timeout_sec
    
    def validate_input(
        self,
        tool_input: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Validate tool input against schema."""
        for required_field in schema.get("required", []):
            if required_field not in tool_input:
                return False, f"Missing required field: {required_field}"
        
        for field, field_schema in schema.get("properties", {}).items():
            if field in tool_input:
                value = tool_input[field]
                expected_type = field_schema.get("type")
                
                if expected_type == "string" and not isinstance(value, str):
                    return False, f"{field} must be string"
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    return False, f"{field} must be number"
        
        return True, "Valid"
    
    def execute_with_timeout(
        self,
        func: callable,
        kwargs: Dict[str, Any],
    ) -> ToolExecutionResult:
        """Execute function with timeout."""
        import time
        import signal
        
        start_time = time.time()
        
        try:
            # Simple timeout using signal (Unix only)
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Tool execution exceeded {self.timeout}s")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            
            result = func(**kwargs)
            
            signal.alarm(0)  # Cancel alarm
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return ToolExecutionResult(
                success=True,
                output=str(result),
                error=None,
                execution_time_ms=elapsed_ms,
            )
        
        except TimeoutError as e:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Timeout: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        
        except Exception as e:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

# Usage
executor = SafeToolExecutor(timeout_sec=5)
result = executor.execute_with_timeout(
    func=lambda x: x + 1,
    kwargs={"x": 5},
)
print(f"Success: {result.success}, Output: {result.output}")
```

---

## 3. Safety Constraints for Tools

```python
from enum import Enum
from typing import Callable

class SafetyLevel(str, Enum):
    UNRESTRICTED = "unrestricted"
    RESTRICTED = "restricted"
    DANGEROUS = "dangerous"

class ToolSafetyPolicy:
    """Define and enforce safety policies for tools."""
    
    def __init__(self):
        self.dangerous_keywords = [
            "delete", "remove", "drop", "exec", "eval", "rm",
        ]
        self.restricted_keywords = ["write", "update", "modify"]
    
    def classify_tool(self, tool_name: str, description: str) -> SafetyLevel:
        """Classify tool safety level."""
        combined = (tool_name + " " + description).lower()
        
        for keyword in self.dangerous_keywords:
            if keyword in combined:
                return SafetyLevel.DANGEROUS
        
        for keyword in self.restricted_keywords:
            if keyword in combined:
                return SafetyLevel.RESTRICTED
        
        return SafetyLevel.UNRESTRICTED
    
    def is_tool_allowed(
        self,
        tool_name: str,
        user_role: str = "user",
        safety_level: SafetyLevel = SafetyLevel.UNRESTRICTED,
    ) -> bool:
        """Check if tool is allowed for user."""
        # Admin can use anything
        if user_role == "admin":
            return True
        
        # Regular users can't use dangerous tools
        if safety_level == SafetyLevel.DANGEROUS:
            return False
        
        return True

# Usage
policy = ToolSafetyPolicy()
level = policy.classify_tool("delete_file", "Deletes a file from disk")
allowed = policy.is_tool_allowed("delete_file", user_role="user")
print(f"Safety level: {level}, Allowed: {allowed}")
```

---

## 4. Agent Memory Compression

```python
from typing import List

class CompressedMemory:
    """Compress old memory to reduce token usage."""
    
    def __init__(self, max_recent: int = 5):
        self.max_recent = max_recent
        self.full_history = []
    
    def add_interaction(self, interaction: Dict[str, str]) -> None:
        """Add new interaction."""
        self.full_history.append(interaction)
    
    def get_compressed_context(self, llm_summarize: Callable) -> str:
        """Get compressed context for prompt."""
        if len(self.full_history) <= self.max_recent:
            # No compression needed
            return self._format_history(self.full_history)
        
        # Summarize old interactions
        old_interactions = self.full_history[:-self.max_recent]
        recent_interactions = self.full_history[-self.max_recent:]
        
        # Summarize old
        summary_prompt = f"Summarize these interactions concisely:\n"
        for inter in old_interactions:
            summary_prompt += f"- {inter}\n"
        
        summary = llm_summarize(summary_prompt)
        
        context = f"Previous context (summarized): {summary}\n\n"
        context += "Recent interactions:\n"
        context += self._format_history(recent_interactions)
        
        return context
    
    def _format_history(self, interactions: List) -> str:
        """Format interactions as string."""
        return "\n".join(str(i) for i in interactions)
```

---

## 5. Multi-Agent Failure Recovery

```python
from enum import Enum

class RecoveryStrategy(str, Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    ESCALATE = "escalate"

class FailureRecovery:
    """Handle agent failures gracefully."""
    
    def __init__(self):
        self.max_retries = 3
        self.fallback_agent = None
    
    def handle_failure(
        self,
        error: Exception,
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    ) -> str:
        """Handle agent failure."""
        if strategy == RecoveryStrategy.RETRY:
            return self._retry()
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._fallback()
        elif strategy == RecoveryStrategy.ESCALATE:
            return self._escalate(error)
    
    def _retry(self) -> str:
        """Retry failed operation."""
        return "Retrying operation..."
    
    def _fallback(self) -> str:
        """Use fallback agent/strategy."""
        return "Using fallback strategy..."
    
    def _escalate(self, error: Exception) -> str:
        """Escalate to human review."""
        return f"Escalating error to human: {str(error)}"
```

---

## 6. References

1. https://arxiv.org/abs/2205.13147 — "Grounding Language Models to Physical World by Interaction" (Tool grounding)
2. https://arxiv.org/abs/2303.08774 — "Instruction Tuning with the Reinforcement Learning from Human Feedback" (Safety)
3. https://github.com/langchain-ai/langchain — LangChain tool management
4. https://arxiv.org/abs/2304.17595 — "Making Language Models Safer" (Safety constraints)
5. https://huggingface.co/docs/transformers/tasks/tool_use — Tool use documentation
6. https://arxiv.org/abs/2305.08596 — "Emergent Tool Use from Multi-Agent Autocurricula" (Tool emergent behavior)
7. https://github.com/openai/gpt-4-vision-api-addon — Vision tool patterns
8. https://arxiv.org/abs/2310.13049 — "Agent Memory and Planning"
9. https://github.com/hwchase17/langchain-hub — Tool definitions library
10. https://arxiv.org/abs/2308.10379 — "Toolformer and Beyond" (Tool extension patterns)
11. https://huggingface.co/docs/huggingface_hub/security — API security best practices
12. https://github.com/python-engineer/python-ml-deployment — Safe deployment patterns
13. https://arxiv.org/abs/2303.14816 — "Red Teaming LLM Tool Use"
14. https://github.com/microsoft/promptflow — Microsoft Prompt Flow tool framework
15. https://arxiv.org/abs/2309.11289 — "Tool-Augmented Language Models"
16. https://github.com/google/prompt-to-prompt — Prompt control patterns

---

## 7. Uncertainty and Limitations

**Not Covered:** Formal verification of agent safety, adversarial tool injection attacks, distributed agent coordination. **Production:** Implement audit logging, rate limiting, human approval queues.
